# models/long_video_planner.py
import numpy as np
from typing import List, Dict, Any, Optional
from models.lens_controller import LensScriptParser


class SceneBlock:
    """
    表示一个视频块，包含起始帧、结束帧、过渡帧数和对应的镜头列表。
    """

    def __init__(self, start_frame: int, end_frame: int, transition_frames: int, shots: List[Dict[str, Any]]):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.transition_frames = transition_frames
        self.shots = shots


class LongVideoPlanner:
    """
    长视频规划器：根据镜头脚本和总时长，将视频切分为若干 SceneBlock。
    支持按时间轴裁剪镜头脚本，自动处理过渡帧。
    """

    def __init__(self, block_duration_sec: float, fps: int, overlap_sec: float):
        """
        Args:
            block_duration_sec: 每个块的时长（秒）
            fps: 帧率
            overlap_sec: 块间重叠时长（秒）
        """
        self.block_duration_sec = block_duration_sec
        self.fps = fps
        self.overlap_sec = overlap_sec

    def plan(self, lens_script_path: str, total_duration_sec: float) -> List[SceneBlock]:
        """
        将总视频切分为多个 SceneBlock，并裁剪每个块对应的镜头脚本。
        Args:
            lens_script_path: 镜头脚本 JSON 文件路径
            total_duration_sec: 总时长（秒）
        Returns:
            List[SceneBlock]: 块列表
        """
        # 解析完整镜头脚本
        full_shots = LensScriptParser.parse_script(lens_script_path)

        # 构建时间轴（每个镜头的起始和结束时间）
        shot_timeline = self._build_timeline(full_shots)

        # 计算块参数
        block_frames = int(self.block_duration_sec * self.fps)
        overlap_frames = int(self.overlap_sec * self.fps)
        total_frames = int(total_duration_sec * self.fps)

        blocks = []
        start = 0
        while start < total_frames:
            end = min(start + block_frames, total_frames)
            # 获取该时间区间内的镜头脚本
            block_shots = self._extract_shots_by_time(
                shot_timeline, start / self.fps, end / self.fps)
            # 创建块，过渡帧数使用配置的重叠帧数（可根据实际镜头调整）
            blocks.append(SceneBlock(start, end, overlap_frames, block_shots))
            # 下一个块的起始位置（减去重叠部分）
            start = end - overlap_frames
            if start >= end:  # 防止死循环
                break
        return blocks

    def _build_timeline(self, shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为每个镜头计算其在视频中的起始时间和结束时间（秒）。
        假设每个镜头的 duration 字段已给出。
        返回列表，每个元素包含 shot 原数据及其 start_time, end_time。
        """
        timeline = []
        current_time = 0.0
        for shot in shots:
            duration = shot.get('duration', 2.0)
            start = current_time
            end = current_time + duration
            timeline.append({
                'shot': shot,
                'start_time': start,
                'end_time': end
            })
            current_time = end
        return timeline

    def _extract_shots_by_time(self, timeline: List[Dict[str, Any]],
                               block_start: float, block_end: float) -> List[Dict[str, Any]]:
        """
        根据时间区间裁剪镜头脚本，返回该区间内的镜头列表。
        如果镜头跨区间，会进行裁剪，保留完整描述，并调整 duration。
        """
        result = []
        for item in timeline:
            shot = item['shot'].copy()  # 避免修改原数据
            shot_start = item['start_time']
            shot_end = item['end_time']

            # 完全在区间外
            if shot_end <= block_start or shot_start >= block_end:
                continue

            # 计算重叠部分
            overlap_start = max(shot_start, block_start)
            overlap_end = min(shot_end, block_end)
            new_duration = overlap_end - overlap_start

            # 调整 duration 为重叠时长
            shot['duration'] = new_duration
            # 可选：调整焦点位置等（可根据需要扩展）
            result.append(shot)
        return result
