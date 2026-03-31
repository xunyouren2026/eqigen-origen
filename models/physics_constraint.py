# postprocess/physics_corrector.py
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import List, Tuple, Optional


class PhysicsCorrector:
    """
    物理校正器，支持关键帧优化，集成 Warp / Taichi 模拟器。

    主要功能：
    1. 对视频的关键帧进行物理校正（光流异常平滑或物理模拟）。
    2. 对非关键帧使用光流引导的扭曲进行平滑，消除闪烁和物理不合理运动。
    3. 支持两种高级物理模拟后端：Warp（刚体粒子）和 Taichi（弹性网格），
       如果未安装则回退到简单的光流平滑。
    """

    def __init__(self, config, device='cuda'):
        """
        初始化校正器。

        Args:
            config: 模型配置对象，包含以下字段：
                - model.phy_corr_keyframe_interval: 关键帧间隔（帧数）
                - model.phy_corr_solver: 求解器类型 ('simple', 'warp', 'taichi')
                - model.phy_corr_smooth: 是否进行光流引导的中间帧平滑
            device: 运行设备（'cuda' 或 'cpu'）
        """
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
        对视频进行物理校正（主入口）。

        流程：
        1. 选择关键帧（每隔 keyframe_interval 帧）。
        2. 对每个关键帧应用物理校正（_correct_frame）。
        3. 对相邻关键帧之间的中间帧进行插值（线性混合 + 可选光流引导平滑）。
        4. 如果启用 smooth，对最终结果进行光流平滑。

        Args:
            video: 输入视频，形状 (T, H, W, 3)，dtype=np.uint8，RGB 格式。
            fps: 视频帧率（用于光流平滑的参数，未直接使用）。

        Returns:
            校正后的视频，形状与输入相同。
        """
        T, H, W, C = video.shape

        # 1. 生成关键帧索引
        keyframes = list(range(0, T, self.keyframe_interval))
        if keyframes[-1] != T - 1:
            keyframes.append(T - 1)

        if len(keyframes) < 2:
            # 视频太短，无需校正
            return video

        # 2. 对每个关键帧进行物理校正
        corrected_keyframes = []
        for idx, kf_idx in enumerate(keyframes):
            frame = video[kf_idx].astype(
                np.float32) / 255.0          # (H,W,3) in [0,1]
            frame_t = torch.from_numpy(frame).permute(
                2, 0, 1).unsqueeze(0).to(self.device)  # (1,3,H,W)

            # 获取前后关键帧（如果有），用于更准确的物理模拟
            prev_frame = video[keyframes[idx-1]] if idx > 0 else None
            next_frame = video[keyframes[idx+1]
                               ] if idx < len(keyframes)-1 else None
            if prev_frame is not None:
                prev_t = torch.from_numpy(prev_frame.astype(
                    np.float32)/255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            else:
                prev_t = None
            if next_frame is not None:
                next_t = torch.from_numpy(next_frame.astype(
                    np.float32)/255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            else:
                next_t = None

            # 根据 solver_type 选择校正方法
            if self.solver_type == 'simple':
                corrected = self._simple_correct(frame_t, prev_t, next_t)
            elif self.solver_type == 'warp':
                corrected = self._warp_correct(frame_t, prev_t, next_t)
            elif self.solver_type == 'taichi':
                corrected = self._taichi_correct(frame_t, prev_t, next_t)
            else:
                corrected = frame_t

            corrected = corrected.squeeze(0).permute(1, 2, 0).cpu().numpy()
            corrected = np.clip(corrected, 0, 1) * 255
            corrected_keyframes.append(corrected.astype(np.uint8))

        # 3. 对中间帧进行插值（线性混合 + 可选光流平滑）
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

            # 线性混合（基线）
            for t in range(num_frames):
                alpha = t / (num_frames - 1)
                interp = (1 - alpha) * start_frame + alpha * end_frame
                result[start_idx +
                       t] = np.clip(interp, 0, 255).astype(np.uint8)

            # 如果启用光流引导平滑，则对中间帧进行进一步校正（使运动更平滑）
            if self.smooth:
                # 提取当前块的所有帧（包括首尾关键帧）
                block_frames = result[start_idx:end_idx+1].copy()
                block_frames = self._optical_flow_smooth(block_frames, fps)
                result[start_idx:end_idx+1] = block_frames

        # 可选：全局光流平滑（已在上面的块内完成，但为保持一致性，也可再执行一次）
        if self.smooth:
            result = self._optical_flow_smooth(result, fps)

        return result

    def _simple_correct(self, frame_t: torch.Tensor, prev_frame: Optional[torch.Tensor], next_frame: Optional[torch.Tensor]) -> torch.Tensor:
        """
        简化物理校正：基于光流的异常区域平滑。

        原理：
        1. 如果存在前后帧，计算光流，识别运动异常的像素（光流幅度过大）。
        2. 对这些异常区域进行局部平滑（例如用相邻帧的加权平均）。
        3. 如果没有前后帧，直接返回原图。

        Args:
            frame_t: 当前帧张量 (1,3,H,W)，值域 [0,1]。
            prev_frame: 前一帧张量 (1,3,H,W) 或 None。
            next_frame: 后一帧张量 (1,3,H,W) 或 None。

        Returns:
            校正后的帧张量。
        """
        if prev_frame is None or next_frame is None:
            # 没有前后帧，无法进行光流比较，直接返回
            return frame_t

        # 使用 physics 模块计算光流（需要输入为 (B, C, T, H, W)）
        # 堆叠三帧：prev, curr, next
        three_frames = torch.stack([prev_frame.squeeze(0), frame_t.squeeze(
            0), next_frame.squeeze(0)], dim=0)  # (3,3,H,W)
        three_frames = three_frames.unsqueeze(0)  # (1,3,3,H,W)
        # 计算光流：返回 (B,2,T-1,H,W) 速度场，这里 T=3，得到 T-1=2 个光流场
        velocity = self.physics.compute_velocity_field(
            three_frames)  # (1,2,2,H,W)
        # 取前向光流（从 frame_t 到 next_frame）和反向光流（从 prev_frame 到 frame_t）
        flow_fwd = velocity[0, :, 0, :, :]   # (2,H,W) 从 curr 到 next
        flow_bwd = velocity[0, :, 1, :, :]   # (2,H,W) 从 prev 到 curr

        # 计算光流幅度
        mag_fwd = torch.norm(flow_fwd, dim=0)   # (H,W)
        mag_bwd = torch.norm(flow_bwd, dim=0)   # (H,W)
        mag = mag_fwd + mag_bwd                 # (H,W)

        # 异常阈值（可调节）
        threshold = 10.0
        abnormal = (mag > threshold).float().unsqueeze(
            0).unsqueeze(0)  # (1,1,H,W)

        # 对当前帧进行高斯模糊
        blurred = F.avg_pool2d(frame_t, kernel_size=5, stride=1, padding=2)
        # 异常区域用模糊替换，正常区域保留原值
        corrected = abnormal * blurred + (1 - abnormal) * frame_t
        return corrected

    def _warp_correct(self, frame_t: torch.Tensor, prev_frame: Optional[torch.Tensor], next_frame: Optional[torch.Tensor]) -> torch.Tensor:
        """
        使用 Warp 进行物理校正（刚体粒子模拟）。

        流程：
        1. 从光流中提取运动场（或随机初始化）。
        2. 使用 Warp 模拟刚体粒子运动，生成位移场。
        3. 根据位移场对当前帧进行扭曲。

        Args:
            frame_t: 当前帧张量 (1,3,H,W)
            prev_frame: 前一帧张量 (1,3,H,W) 或 None
            next_frame: 后一帧张量 (1,3,H,W) 或 None

        Returns:
            校正后的帧张量。
        """
        if self.solver is None:
            return frame_t

        import warp as wp
        import numpy as np

        # 获取图像尺寸
        H, W = frame_t.shape[2], frame_t.shape[3]

        # 如果存在前后帧，用光流作为初始速度场
        if prev_frame is not None and next_frame is not None:
            # 计算光流（从 prev 到 next，简化为从 prev 到 curr 和 curr 到 next）
            three_frames = torch.stack([prev_frame.squeeze(0), frame_t.squeeze(
                0), next_frame.squeeze(0)], dim=0).unsqueeze(0)
            velocity = self.physics.compute_velocity_field(
                three_frames)  # (1,2,2,H,W)
            flow_fwd = velocity[0, :, 0, :, :]   # 从 curr 到 next
            flow_bwd = velocity[0, :, 1, :, :]   # 从 prev 到 curr
            # 将光流转换为速度场（像素/秒），这里简化处理，直接作为位移
            # 实际上需要根据帧间隔时间 dt = 1/fps，但这里我们直接作为位移向量（单位：像素）
            # 用于初始化粒子的速度
            init_vel = (flow_fwd + flow_bwd) / 2.0   # (2,H,W)
            # 将速度场展平为粒子属性
            # 注意：粒子数 = H*W，每个粒子对应一个像素点
            num_particles = H * W
            # 创建粒子位置网格
            x_coords = np.linspace(0, W-1, W)
            y_coords = np.linspace(0, H-1, H)
            xx, yy = np.meshgrid(x_coords, y_coords)
            positions = np.stack([xx.flatten(), yy.flatten(), np.zeros(
                num_particles)], axis=1).astype(np.float32)  # (N,3)

            # 将速度场转换为每个粒子的速度
            velocities = init_vel.cpu().numpy()  # (2,H,W)
            velocities = velocities.transpose(1, 2, 0).reshape(-1, 2)  # (N,2)
            # 补充 z 分量 0
            velocities = np.pad(velocities, ((0, 0), (0, 1)),
                                mode='constant')  # (N,3)

            # 创建粒子系统
            builder = wp.sim.ModelBuilder()
            # 添加粒子（质量 = 1.0）
            for p, v in zip(positions, velocities):
                builder.add_particle(
                    wp.vec3(p[0], p[1], p[2]), wp.vec3(v[0], v[1], v[2]), 1.0)
            # 添加一些约束（例如保持粒子在图像范围内）
            # 简单起见，不加约束，让粒子自由运动
            model = builder.finalize("cuda")

            # 创建状态和积分器
            state = model.state()
            integrator = wp.sim.SemiImplicitIntegrator()

            # 模拟若干步，更新粒子位置
            dt = 1.0 / 60.0
            for _ in range(5):
                integrator.simulate(model, state, state, dt=dt)

            # 获取更新后的粒子位置
            new_positions = state.particle_q.numpy()  # (N,3)
            # 计算位移场（像素偏移）
            displacement = new_positions[:, :2] - positions[:, :2]  # (N,2)
            displacement = displacement.reshape(H, W, 2)  # (H,W,2)

        else:
            # 没有前后帧，随机初始化位移场（例如微小随机扰动）
            displacement = np.random.randn(H, W, 2) * 0.5  # 0.5 像素扰动

        # 使用位移场对当前帧进行扭曲（remap）
        # 将位移场转换为映射坐标
        x_coords = np.arange(W)
        y_coords = np.arange(H)
        xx, yy = np.meshgrid(x_coords, y_coords)
        map_x = (xx + displacement[:, :, 0]).astype(np.float32)
        map_y = (yy + displacement[:, :, 1]).astype(np.float32)

        # 将 frame_t 转为 numpy (H,W,3)
        frame_np = frame_t.squeeze(0).permute(
            1, 2, 0).cpu().numpy()  # (H,W,3), [0,1]
        # 执行 remap
        warped = cv2.remap(
            frame_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        warped_t = torch.from_numpy(warped).permute(
            2, 0, 1).unsqueeze(0).to(frame_t.device)

        return warped_t

    def _taichi_correct(self, frame_t: torch.Tensor, prev_frame: Optional[torch.Tensor], next_frame: Optional[torch.Tensor]) -> torch.Tensor:
        """
        使用 Taichi 进行物理校正（弹性体网格模拟）。

        这里演示一个简单的 2D 弹簧质点系统，模拟一块弹性布料覆盖在图像上，
        根据外部力（如重力）使布料变形，然后根据变形场扭曲图像。

        Args:
            frame_t: 当前帧张量 (1,3,H,W)
            prev_frame: 前一帧（未使用，仅保留接口）
            next_frame: 后一帧（未使用）

        Returns:
            校正后的帧张量。
        """
        if self.solver is None:
            return frame_t

        import taichi as ti

        H, W = frame_t.shape[2], frame_t.shape[3]
        # 定义网格大小（分辨率可降低以提高速度）
        grid_h, grid_w = min(H, 64), min(W, 64)   # 使用 64x64 的网格
        step_h = H / grid_h
        step_w = W / grid_w

        # 定义 Ti 内核：弹簧质点系统
        ti.init(arch=ti.gpu)
        # 网格点坐标（2D）
        x = ti.Vector.field(2, dtype=ti.f32, shape=(grid_h, grid_w))
        v = ti.Vector.field(2, dtype=ti.f32, shape=(grid_h, grid_w))

        # 初始化网格点位置（均匀分布在图像上）
        @ti.kernel
        def init_grid():
            for i, j in ti.ndrange(grid_h, grid_w):
                x[i, j] = ti.Vector([j * step_w, i * step_h])
                v[i, j] = ti.Vector([0.0, 0.0])

        # 模拟弹簧力（简单的胡克定律）
        @ti.kernel
        def simulate(dt: ti.f32):
            # 计算力场
            f = ti.Vector.field(2, dtype=ti.f32, shape=(grid_h, grid_w))
            # 弹簧常数
            k = 100.0
            # 阻尼
            damping = 0.1
            for i, j in ti.ndrange(grid_h, grid_w):
                # 重力
                f[i, j] = ti.Vector([0.0, 9.8])
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

        # 获取位移场并扭曲图像
        def get_displacement_field():
            # 将网格位移插值到全分辨率
            disp_x = np.zeros((H, W), dtype=np.float32)
            disp_y = np.zeros((H, W), dtype=np.float32)
            # 获取网格点位移
            x_np = x.to_numpy()
            for i in range(grid_h):
                for j in range(grid_w):
                    # 网格点原始位置
                    orig = np.array([j * step_w, i * step_h])
                    new = x_np[i, j]
                    # 位移
                    dx = new[0] - orig[0]
                    dy = new[1] - orig[1]
                    # 填充到对应区域（简单双线性插值，此处简化，直接用最近邻）
                    y0 = int(i * step_h)
                    y1 = min(y0 + int(step_h), H-1)
                    x0 = int(j * step_w)
                    x1 = min(x0 + int(step_w), W-1)
                    disp_y[y0:y1, x0:x1] = dy
                    disp_x[y0:y1, x0:x1] = dx
            return disp_x, disp_y

        # 初始化网格
        init_grid()
        # 模拟若干步
        dt = 1.0 / 60.0
        for _ in range(10):
            simulate(dt)

        # 获取位移场
        disp_x, disp_y = get_displacement_field()

        # 生成映射坐标
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        map_x = (xx + disp_x).astype(np.float32)
        map_y = (yy + disp_y).astype(np.float32)

        # 扭曲图像
        frame_np = frame_t.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H,W,3)
        warped = cv2.remap(
            frame_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        warped_t = torch.from_numpy(warped).permute(
            2, 0, 1).unsqueeze(0).to(frame_t.device)

        return warped_t

    def _optical_flow_smooth(self, video: np.ndarray, fps: int) -> np.ndarray:
        """
        使用光流对视频序列进行平滑：对每个帧，根据前后帧光流进行扭曲，使运动更连续。

        原理：
        1. 计算相邻帧之间的前向和后向光流。
        2. 对每个中间帧，用前后光流进行双向扭曲，然后融合，消除抖动。

        Args:
            video: 输入视频段，形状 (T, H, W, 3)，dtype=np.uint8。
            fps: 帧率（用于光流参数，未直接使用）。

        Returns:
            平滑后的视频段，形状相同。
        """
        T, H, W, C = video.shape
        if T < 3:
            return video

        # 将视频转为 float32 [0,1] 并准备张量
        video_f = video.astype(np.float32) / 255.0
        video_t = torch.from_numpy(video_f).permute(
            0, 3, 1, 2).to(self.device)  # (T,3,H,W)

        # 计算所有相邻帧的光流（使用 physics 模块，需要输入 (1,3,T,H,W)）
        # 将视频扩展为 batch=1
        video_t_batch = video_t.unsqueeze(0)  # (1,3,T,H,W)
        velocity = self.physics.compute_velocity_field(
            video_t_batch)  # (1,2,T-1,H,W)
        velocity = velocity.squeeze(0)  # (2,T-1,H,W)

        # 对每个帧（除了首尾）进行双向扭曲
        smoothed = np.zeros_like(video_f)
        for t in range(T):
            if t == 0 or t == T-1:
                smoothed[t] = video_f[t]
                continue

            # 前向光流（从 t-1 到 t）和反向光流（从 t 到 t+1）
            flow_prev = velocity[:, t-1, :, :]  # (2,H,W)
            flow_next = velocity[:, t, :, :]    # (2,H,W)

            # 使用 flow_prev 将前一帧扭曲到当前帧
            # 生成映射坐标
            xx, yy = np.meshgrid(np.arange(W), np.arange(H))
            map_x_prev = (xx + flow_prev[0].cpu().numpy()).astype(np.float32)
            map_y_prev = (yy + flow_prev[1].cpu().numpy()).astype(np.float32)
            # 将前一帧扭曲到当前帧
            prev_warped = cv2.remap(
                video_f[t-1], map_x_prev, map_y_prev, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            # 使用 flow_next 将当前帧扭曲到下一帧，然后反向扭曲回来（双向补偿）
            map_x_next = (xx + flow_next[0].cpu().numpy()).astype(np.float32)
            map_y_next = (yy + flow_next[1].cpu().numpy()).astype(np.float32)
            # 将当前帧扭曲到下一帧
            curr_warped_to_next = cv2.remap(
                video_f[t], map_x_next, map_y_next, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            # 然后再反向扭曲回来（使用反向光流）
            # 反向光流 = -flow_next（近似）
            flow_back = -flow_next
            map_x_back = (xx + flow_back[0].cpu().numpy()).astype(np.float32)
            map_y_back = (yy + flow_back[1].cpu().numpy()).astype(np.float32)
            next_warped_to_curr = cv2.remap(
                curr_warped_to_next, map_x_back, map_y_back, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            # 融合三种估计：原帧、前向扭曲、反向扭曲
            alpha = 0.5
            beta = 0.25
            gamma = 0.25
            smoothed[t] = alpha * video_f[t] + beta * \
                prev_warped + gamma * next_warped_to_curr

        # 转回 uint8
        smoothed = np.clip(smoothed * 255, 0, 255).astype(np.uint8)
        return smoothed

# 全局函数，供 inferencer 调用


def correct_video(video: np.ndarray, fps: int, config) -> np.ndarray:
    """
    对视频进行物理校正的便捷函数。

    Args:
        video: 输入视频 (T, H, W, 3) uint8。
        fps: 视频帧率。
        config: 配置对象，包含物理校正参数。

    Returns:
        校正后的视频。
    """
    corrector = PhysicsCorrector(config)
    return corrector.correct_video(video, fps)
