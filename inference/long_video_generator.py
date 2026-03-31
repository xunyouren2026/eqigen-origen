import numpy as np
from typing import List
from models.long_video_planner import LongVideoPlanner, SceneBlock
from models.lens_controller import LensScriptParser


class LongVideoGenerator:
    def __init__(self, inferencer, planner: LongVideoPlanner, num_workers=4):
        self.inferencer = inferencer
        self.planner = planner
        self.num_workers = num_workers
        self.device = inferencer.device

    def generate(self, lens_script_path: str, total_duration_sec: float,
                 prompt: str, **gen_kwargs) -> np.ndarray:
        blocks = self.planner.plan(lens_script_path, total_duration_sec)
        block_videos = []
        prev_state = None
        for idx, block in enumerate(blocks):
            lens_cond = self._encode_shots(block.shots)
            video, new_state = self._generate_block(
                block, lens_cond, prompt, prev_state, **gen_kwargs
            )
            block_videos.append(video)
            prev_state = new_state
        full_video = self._merge_blocks(block_videos, blocks)
        return full_video

    def _generate_block(self, block: SceneBlock, lens_cond, prompt, prev_state, **kwargs):
        duration = (block.end_frame - block.start_frame) / self.planner.fps
        video_path, state = self.inferencer.generate(
            prompt=prompt,
            duration=duration,
            fps=self.planner.fps,
            lens_cond=lens_cond,
            prev_state=prev_state,
            return_state=True,
            **kwargs
        )
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        video = np.array(frames)
        return video, state

    def _merge_blocks(self, videos: List[np.ndarray], blocks: List[SceneBlock]) -> np.ndarray:
        full_video = None
        for i, (video, block) in enumerate(zip(videos, blocks)):
            if i == 0:
                full_video = video
            else:
                overlap = block.transition_frames
                if overlap > 0:
                    alpha = np.linspace(0, 1, overlap).reshape(-1, 1, 1, 1)
                    full_video[-overlap:] = full_video[-overlap:] * \
                        (1 - alpha) + video[:overlap] * alpha
                    full_video = np.concatenate(
                        [full_video, video[overlap:]], axis=0)
                else:
                    full_video = np.concatenate([full_video, video], axis=0)
        return full_video

    def _encode_shots(self, shots):
        from models.lens_controller import Shot
        shot_objs = []
        for s in shots:
            shot_objs.append(Shot(
                duration=s['duration'],
                scene_desc=s.get('scene_desc', ''),
                shot_type=s['shot_type'],
                camera_motion=s['camera_motion'],
                motion_params=s.get('motion_params', {}),
                focus=tuple(s.get('focus', [0.5, 0.5])),
                lighting=s.get('lighting', 'natural'),
                color_tone=s.get('color_tone', 'neutral'),
                rhythm=s.get('rhythm', 'normal'),
                transition=s.get('transition', 'cut')
            ))
        lens_cond = self.inferencer.lens_controller(shot_objs)
        return lens_cond
