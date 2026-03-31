# postprocess/tile_generator.py
import numpy as np
import cv2
from typing import List, Tuple, Optional, Union, Dict, Any


class TileGenerator:
    """
    高分辨率视频分块生成器
    核心策略：
    1. 时间分块由外层 (inferencer.generate_long) 负责，将长视频切分为多个时间段。
    2. 本类仅负责空间分块：将单个时间段的高分辨率画面切分为多个 Tile 并行生成，然后融合。

    优化点：
    - 高斯衰减权重融合：消除 Tile 拼接痕迹。
    - 批量推理支持：一次生成多个 Tile，提高 GPU 利用率。
    - 帧数对齐：自动处理生成帧数与预期不符的情况。
    """

    def __init__(self, base_generator, tile_size: int = 512, overlap: int = 32, batch_size: int = 4):
        """
        :param base_generator: 基础生成器实例 (Inferencer)，用于调用 generate_batch
        :param tile_size: 单个 Tile 的边长 (像素)
        :param overlap: Tile 之间的重叠区域大小 (像素)，用于融合
        :param batch_size: 每次并行生成的 Tile 数量 (根据显存调整，4090 建议 4-8)
        """
        self.base_generator = base_generator
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size

    def generate_1080p(self, prompt: str, num_frames: int, fps: int = 24,
                       prev_state=None, return_state: bool = False, **gen_kwargs) -> np.ndarray:
        """生成 1920x1080 视频"""
        return self._generate_tiled(
            prompt, num_frames, fps, target_size=(1920, 1080),
            prev_state=prev_state, return_state=return_state, **gen_kwargs
        )

    def generate_4k(self, prompt: str, num_frames: int, fps: int = 24,
                    prev_state=None, return_state: bool = False, **gen_kwargs) -> np.ndarray:
        """生成 3840x2160 视频"""
        return self._generate_tiled(
            prompt, num_frames, fps, target_size=(3840, 2160),
            prev_state=prev_state, return_state=return_state, **gen_kwargs
        )

    def generate_8k(self, prompt: str, num_frames: int, fps: int = 24,
                    prev_state=None, return_state: bool = False, **gen_kwargs) -> np.ndarray:
        """生成 7680x4320 视频"""
        return self._generate_tiled(
            prompt, num_frames, fps, target_size=(7680, 4320),
            prev_state=prev_state, return_state=return_state, **gen_kwargs
        )

    def _generate_tiled(self, prompt: str, num_frames: int, fps: int, target_size: Tuple[int, int],
                        prev_state=None, return_state: bool = False, **gen_kwargs) -> np.ndarray:
        """
        内部核心方法：执行空间分块生成与融合
        :param prompt: 提示词
        :param num_frames: 当前时间块的帧数 (由外层决定)
        :param fps: 帧率
        :param target_size: (width, height) 目标分辨率
        :return: 融合后的视频数组 (T, H, W, C)
        """
        target_w, target_h = target_size
        tile_w = self.tile_size
        tile_h = self.tile_size
        overlap = self.overlap

        # 1. 计算需要的行、列数
        # 公式：(总长度 - 重叠) // (步长) + 1
        n_cols = (target_w - overlap) // (tile_w - overlap) + 1
        n_rows = (target_h - overlap) // (tile_h - overlap) + 1

        print(
            f"[TileGenerator] 开始空间分块：{n_cols}列 x {n_rows}行，单块大小 {tile_w}x{tile_h}")

        # 2. 收集所有 Tile 的坐标信息
        tile_infos: List[Dict[str, Any]] = []
        for row in range(n_rows):
            for col in range(n_cols):
                x_start = col * (tile_w - overlap)
                x_end = min(x_start + tile_w, target_w)
                y_start = row * (tile_h - overlap)
                y_end = min(y_start + tile_h, target_h)

                cur_w = x_end - x_start
                cur_h = y_end - y_start

                # 为每个 Tile 构造独立的提示词，暗示其位置（可选，有助于模型理解上下文）
                position_desc = f"{prompt} (Part: col-{col+1}, row-{row+1})"

                tile_infos.append({
                    'row': row,
                    'col': col,
                    'x_start': x_start,
                    'y_start': y_start,
                    'cur_w': cur_w,
                    'cur_h': cur_h,
                    'prompt': position_desc,
                })

        # 3. 分批并行生成
        tile_videos: Dict[Tuple[int, int], np.ndarray] = {}
        last_state = None

        total_batches = (len(tile_infos) + self.batch_size -
                         1) // self.batch_size

        for i in range(0, len(tile_infos), self.batch_size):
            batch_idx = i // self.batch_size
            print(f"[TileGenerator] 处理批次 {batch_idx+1}/{total_batches}")

            batch = tile_infos[i:i+self.batch_size]
            batch_prompts = [info['prompt'] for info in batch]

            # 【关键】调用批量生成
            # duration 必须是当前时间块的时长 (num_frames / fps)，而不是整个视频时长
            batch_paths = self.base_generator.generate_batch(
                prompts=batch_prompts,
                duration=num_frames / fps,
                fps=fps,
                prev_state=prev_state,
                return_state=False,
                **gen_kwargs
            )

            # 4. 加载、校验并缩放每个 Tile 的视频
            for idx, info in enumerate(batch):
                video_path = batch_paths[idx]
                cap = cv2.VideoCapture(video_path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()

                if not frames:
                    raise ValueError(f"Failed to load video from {video_path}")

                video_array = np.array(frames)

                # 【鲁棒性】确保帧数严格一致
                if video_array.shape[0] != num_frames:
                    if video_array.shape[0] < num_frames:
                        # 帧数不足：重复最后一帧
                        last_frame = video_array[-1:]
                        repeat = num_frames - video_array.shape[0]
                        video_array = np.concatenate(
                            [video_array, np.repeat(last_frame, repeat, axis=0)], axis=0)
                    else:
                        # 帧数过多：截断
                        video_array = video_array[:num_frames]

                # 缩放到精确的 Tile 尺寸
                video_resized = self._resize_video(
                    video_array, (info['cur_w'], info['cur_h']))
                tile_videos[(info['row'], info['col'])] = video_resized

        # 5. 融合所有 Tile
        print("[TileGenerator] 正在融合 Tile...")
        merged = self._merge_tiles(
            tile_videos, n_rows, n_cols, target_w, target_h, overlap)

        if return_state:
            return merged, last_state
        return merged

    def _resize_video(self, video: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """将视频帧缩放到目标尺寸 (new_w, new_h)"""
        T, H, W, C = video.shape
        new_w, new_h = target_size

        if H == new_h and W == new_w:
            return video

        resized = np.zeros((T, new_h, new_w, C), dtype=video.dtype)
        for t in range(T):
            resized[t] = cv2.resize(
                video[t], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized

    def _merge_tiles(self, tiles: Dict, n_rows: int, n_cols: int,
                     full_w: int, full_h: int, overlap: int) -> np.ndarray:
        """
        使用高斯衰减权重融合 Tile，消除拼接痕迹
        """
        # 获取时间维度 T
        sample_tile = list(tiles.values())[0]
        T = sample_tile.shape[0]

        # 初始化融合画布和权重图
        merged = np.zeros((T, full_h, full_w, 3), dtype=np.float32)
        weight_map = np.zeros((full_h, full_w), dtype=np.float32)

        for row in range(n_rows):
            for col in range(n_cols):
                tile = tiles[(row, col)]
                H, W = tile.shape[1], tile.shape[2]

                x_start = col * (self.tile_size - overlap)
                x_end = x_start + W
                y_start = row * (self.tile_size - overlap)
                y_end = y_start + H

                # 构建高斯权重掩码
                weight = np.ones((H, W), dtype=np.float32)

                # 左边缘衰减
                if col > 0:
                    left_overlap = min(overlap, W)
                    x = np.linspace(0, 1, left_overlap)
                    gauss = np.exp(-x**2 * 5)  # 高斯系数 5 可调整衰减速度
                    weight[:, :left_overlap] *= gauss

                # 右边缘衰减
                if col < n_cols - 1:
                    right_overlap = min(overlap, W)
                    x = np.linspace(1, 0, right_overlap)
                    gauss = np.exp(-(1-x)**2 * 5)
                    weight[:, -right_overlap:] *= gauss

                # 上边缘衰减
                if row > 0:
                    top_overlap = min(overlap, H)
                    x = np.linspace(0, 1, top_overlap)
                    gauss = np.exp(-x**2 * 5)
                    weight[:top_overlap, :] *= gauss.reshape(-1, 1)

                # 下边缘衰减
                if row < n_rows - 1:
                    bottom_overlap = min(overlap, H)
                    x = np.linspace(1, 0, bottom_overlap)
                    gauss = np.exp(-(1-x)**2 * 5)
                    weight[-bottom_overlap:, :] *= gauss.reshape(-1, 1)

                # 累加像素值和权重
                for t in range(T):
                    merged[t, y_start:y_end, x_start:x_end] += tile[t] * \
                        weight[..., np.newaxis]

                weight_map[y_start:y_end, x_start:x_end] += weight

        # 【关键】权重归一化，避免亮度不均
        weight_map = np.maximum(weight_map, 1e-5)
        for t in range(T):
            merged[t] /= weight_map[..., np.newaxis]

        return np.clip(merged, 0, 255).astype(np.uint8)
