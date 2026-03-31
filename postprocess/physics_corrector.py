import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import List, Tuple, Optional


class PhysicsCorrector:
    """
    物理校正器，支持关键帧优化，集成 Warp / Taichi 模拟器
    """

    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.keyframe_interval = config.model.phy_corr_keyframe_interval
        self.solver_type = config.model.phy_corr_solver
        self.smooth = config.model.phy_corr_smooth

        # 光流模型（复用现有 PhysicsConstraint 的光流能力）
        from models.physics_constraint import PhysicsConstraint
        self.physics = PhysicsConstraint(device, use_raft=True)

        # 初始化高级模拟器（如果可用）
        self.solver = None
        if self.solver_type == 'warp':
            try:
                import warp as wp
                wp.init()
                self.solver = wp
                print("[PhysicsCorrector] Warp initialized")
            except ImportError:
                print("[PhysicsCorrector] Warp not available, falling back to simple")
                self.solver_type = 'simple'
        elif self.solver_type == 'taichi':
            try:
                import taichi as ti
                ti.init(arch=ti.gpu)
                self.solver = ti
                print("[PhysicsCorrector] Taichi initialized")
            except ImportError:
                print(
                    "[PhysicsCorrector] Taichi not available, falling back to simple")
                self.solver_type = 'simple'
        else:
            self.solver_type = 'simple'

    def correct_video(self, video: np.ndarray, fps: int) -> np.ndarray:
        """
        对视频进行物理校正
        video: (T, H, W, 3) uint8
        fps: 帧率
        返回校正后的视频，形状相同
        """
        T, H, W, C = video.shape

        # 生成关键帧索引
        keyframes = list(range(0, T, self.keyframe_interval))
        if keyframes[-1] != T - 1:
            keyframes.append(T - 1)

        if len(keyframes) < 2:
            return video

        # 1. 对每个关键帧进行物理校正
        corrected_keyframes = []
        for idx, kf_idx in enumerate(keyframes):
            frame = video[kf_idx].astype(np.float32) / 255.0
            frame_t = torch.from_numpy(frame).permute(
                2, 0, 1).unsqueeze(0).to(self.device)

            if self.solver_type == 'simple':
                corrected = self._simple_correct(frame_t)
            elif self.solver_type == 'warp':
                corrected = self._warp_correct(
                    frame_t, prev_frame=None, next_frame=None)
            elif self.solver_type == 'taichi':
                corrected = self._taichi_correct(
                    frame_t, prev_frame=None, next_frame=None)
            else:
                corrected = frame_t

            corrected = corrected.squeeze(0).permute(1, 2, 0).cpu().numpy()
            corrected = np.clip(corrected, 0, 1) * 255
            corrected_keyframes.append(corrected.astype(np.uint8))

        # 2. 对中间帧进行插值（线性混合 + 可选光流平滑）
        result = video.copy()
        for i in range(len(keyframes) - 1):
            start_idx = keyframes[i]
            end_idx = keyframes[i + 1]
            start_frame = corrected_keyframes[i]
            end_frame = corrected_keyframes[i + 1]
            num_frames = end_idx - start_idx + 1
            if num_frames == 1:
                result[start_idx] = start_frame
                continue
            for t in range(num_frames):
                alpha = t / (num_frames - 1)
                interp = (1 - alpha) * start_frame + alpha * end_frame
                result[start_idx +
                       t] = np.clip(interp, 0, 255).astype(np.uint8)

        # 可选：光流引导的平滑
        if self.smooth:
            result = self._optical_flow_smooth(result, fps)

        return result

    def _simple_correct(self, frame_t: torch.Tensor) -> torch.Tensor:
        """
        简化物理校正：基于光流的局部平滑
        需要前后帧才能做光流，这里简化为返回原图
        """
        # 实际应用中，可以传入前后帧，这里保持简单
        return frame_t

    def _warp_correct(self, frame_t: torch.Tensor, prev_frame=None, next_frame=None) -> torch.Tensor:
        """
        使用 Warp 进行物理校正（刚体粒子模拟 + 图像扭曲）
        模拟多个粒子的运动，生成位移场，然后扭曲图像。
        """
        if self.solver is None:
            return frame_t

        import warp as wp
        import numpy as np

        # 获取图像尺寸
        H, W = frame_t.shape[2], frame_t.shape[3]
        frame_np = frame_t.squeeze(0).permute(
            1, 2, 0).cpu().numpy()  # (H,W,3), [0,1]

        # 1. 定义粒子网格（例如 32x32 个粒子，覆盖整个图像）
        grid_size = 32  # 可调，越大变形越精细但计算量增加
        step_y = H / grid_size
        step_x = W / grid_size

        # 创建粒子位置 (N, 3) 和速度 (N, 3)
        positions = []
        velocities = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = j * step_x + step_x / 2
                y = i * step_y + step_y / 2
                # 初始位置 (x, y, 0)
                positions.append([x, y, 0.0])
                # 初始速度 (0, 0, 0)
                velocities.append([0.0, 0.0, 0.0])
        positions = np.array(positions, dtype=np.float32)
        velocities = np.array(velocities, dtype=np.float32)

        # 2. 在 Warp 中创建粒子系统
        builder = wp.sim.ModelBuilder()
        for p, v in zip(positions, velocities):
            builder.add_particle(
                wp.vec3(p[0], p[1], p[2]), wp.vec3(v[0], v[1], v[2]), 1.0)
        # 可选：添加弹簧约束，使粒子间有弹性（使变形更平滑）
        # 这里简单起见不加约束，粒子自由运动

        model = builder.finalize("cuda")
        state = model.state()
        integrator = wp.sim.SemiImplicitIntegrator()

        # 3. 模拟少量步数，更新粒子位置
        dt = 1.0 / 60.0
        for _ in range(5):
            integrator.simulate(model, state, state, dt=dt)

        # 4. 获取更新后的粒子位置
        new_positions = state.particle_q.numpy()  # (N, 3)

        # 5. 计算每个像素的位移（通过双线性插值）
        displacement = np.zeros((H, W, 2), dtype=np.float32)
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                old_x = positions[idx, 0]
                old_y = positions[idx, 1]
                new_x = new_positions[idx, 0]
                new_y = new_positions[idx, 1]
                dx = new_x - old_x
                dy = new_y - old_y

                # 粒子影响的区域：以粒子为中心的网格单元
                y0 = int(i * step_y)
                y1 = int(min((i+1) * step_y, H))
                x0 = int(j * step_x)
                x1 = int(min((j+1) * step_x, W))
                for yy in range(y0, y1):
                    for xx in range(x0, x1):
                        displacement[yy, xx, 0] = dx
                        displacement[yy, xx, 1] = dy

        # 6. 使用位移场生成映射坐标
        x_coords = np.arange(W)
        y_coords = np.arange(H)
        xx, yy = np.meshgrid(x_coords, y_coords)
        map_x = (xx + displacement[..., 0]).astype(np.float32)
        map_y = (yy + displacement[..., 1]).astype(np.float32)

        # 7. 扭曲图像
        warped = cv2.remap(frame_np, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)

        # 转回 Tensor
        warped_t = torch.from_numpy(warped).permute(
            2, 0, 1).unsqueeze(0).to(frame_t.device)
        return warped_t

    def _taichi_correct(self, frame_t: torch.Tensor, prev_frame=None, next_frame=None) -> torch.Tensor:
        """
        使用 Taichi 进行物理校正（弹性网格模拟 + 图像扭曲）
        模拟一个 2D 弹簧质点网格，根据重力产生变形，然后扭曲图像。
        """
        if self.solver is None:
            return frame_t

        import taichi as ti

        # 获取图像尺寸
        H, W = frame_t.shape[2], frame_t.shape[3]
        frame_np = frame_t.squeeze(0).permute(
            1, 2, 0).cpu().numpy()  # (H,W,3), [0,1]

        # 定义网格大小（为了性能，使用较粗网格，例如 64x64）
        grid_h = min(H, 64)
        grid_w = min(W, 64)
        step_h = H / grid_h
        step_w = W / grid_w

        # 定义 Ti 场
        x = ti.Vector.field(2, dtype=ti.f32, shape=(grid_h, grid_w))  # 位置
        v = ti.Vector.field(2, dtype=ti.f32, shape=(grid_h, grid_w))  # 速度

        # 初始化网格点位置
        @ti.kernel
        def init_grid():
            for i, j in ti.ndrange(grid_h, grid_w):
                x[i, j] = ti.Vector(
                    [j * step_w + step_w/2, i * step_h + step_h/2])
                v[i, j] = ti.Vector([0.0, 0.0])

        # 模拟弹簧力（胡克定律）
        @ti.kernel
        def simulate(dt: ti.f32):
            # 计算力场
            f = ti.Vector.field(2, dtype=ti.f32, shape=(grid_h, grid_w))
            k = 100.0          # 弹簧常数
            damping = 0.1      # 阻尼
            gravity = ti.Vector([0.0, 9.8])   # 重力

            for i, j in ti.ndrange(grid_h, grid_w):
                # 重力
                f[i, j] = gravity
                # 水平方向弹簧
                if i < grid_h - 1:
                    disp = x[i+1, j] - x[i, j]
                    force = k * (disp - ti.Vector([step_w, 0.0]))
                    f[i, j] += force
                    f[i+1, j] -= force
                if j < grid_w - 1:
                    disp = x[i, j+1] - x[i, j]
                    force = k * (disp - ti.Vector([0.0, step_h]))
                    f[i, j] += force
                    f[i, j+1] -= force
                # 阻尼
                f[i, j] -= damping * v[i, j]

            # 更新速度和位置（欧拉积分）
            for i, j in ti.ndrange(grid_h, grid_w):
                v[i, j] += f[i, j] * dt
                x[i, j] += v[i, j] * dt

        # 初始化网格
        init_grid()

        # 模拟若干步
        dt = 1.0 / 60.0
        for _ in range(10):
            simulate(dt)

        # 获取网格点位移
        x_np = x.to_numpy()
        displacement = np.zeros((H, W, 2), dtype=np.float32)

        # 将网格位移插值到全分辨率
        for i in range(grid_h):
            for j in range(grid_w):
                old_x = j * step_w + step_w/2
                old_y = i * step_h + step_h/2
                new_x = x_np[i, j][0]
                new_y = x_np[i, j][1]
                dx = new_x - old_x
                dy = new_y - old_y

                y0 = int(i * step_h)
                y1 = int(min((i+1) * step_h, H))
                x0 = int(j * step_w)
                x1 = int(min((j+1) * step_w, W))
                for yy in range(y0, y1):
                    for xx in range(x0, x1):
                        displacement[yy, xx, 0] = dx
                        displacement[yy, xx, 1] = dy

        # 生成映射坐标
        x_coords = np.arange(W)
        y_coords = np.arange(H)
        xx, yy = np.meshgrid(x_coords, y_coords)
        map_x = (xx + displacement[..., 0]).astype(np.float32)
        map_y = (yy + displacement[..., 1]).astype(np.float32)

        # 扭曲图像
        warped = cv2.remap(frame_np, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)

        warped_t = torch.from_numpy(warped).permute(
            2, 0, 1).unsqueeze(0).to(frame_t.device)
        return warped_t

    def _optical_flow_smooth(self, video: np.ndarray, fps: int) -> np.ndarray:
        """
        使用光流对插值后的视频进行平滑（增强中间帧质量）
        实际应用时，可以根据前后帧光流对当前帧进行扭曲，这里简化为返回原视频。
        """
        # 可扩展：使用 OpenCV 的光流进行帧间平滑
        # 此处仅作占位，返回原视频
        return video

# 全局函数，供 inferencer 调用


def correct_video(video: np.ndarray, fps: int, config) -> np.ndarray:
    corrector = PhysicsCorrector(config)
    return corrector.correct_video(video, fps)
