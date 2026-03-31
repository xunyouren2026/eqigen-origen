# models/lens_controller.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
import json


@dataclass
class Shot:
    duration: float = 2.0
    scene_desc: str = ""
    shot_type: str = "medium"
    camera_motion: str = "static"
    motion_params: Dict[str, float] = field(default_factory=dict)
    focus: Tuple[float, float] = (0.5, 0.5)
    lighting: str = "natural"
    color_tone: str = "neutral"
    rhythm: str = "normal"
    transition: str = "cut"


class LensController(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = getattr(config, 'dit_context_dim', 768)
        self.shot_embed_dim = getattr(config, 'lens_controller_dim', 128)
        self.num_heads = 8

        self.shot_type_embed = nn.Embedding(6, self.shot_embed_dim)
        self.camera_motion_embed = nn.Embedding(8, self.shot_embed_dim)
        self.lighting_embed = nn.Embedding(4, self.shot_embed_dim)
        self.color_tone_embed = nn.Embedding(5, self.shot_embed_dim)
        self.rhythm_embed = nn.Embedding(3, self.shot_embed_dim)
        self.transition_embed = nn.Embedding(5, self.shot_embed_dim)

        self.motion_param_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, self.shot_embed_dim)
        )
        self.focus_encoder = nn.Linear(2, self.shot_embed_dim)
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, 1024, self.shot_embed_dim))
        self.shot_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.shot_embed_dim, nhead=8, batch_first=True),
            num_layers=4
        )
        self.output_proj = nn.Linear(self.shot_embed_dim, self.hidden_size)

        # ShotStream
        self.boundary_embed = nn.Embedding(2, self.shot_embed_dim)
        self.use_shotstream = getattr(config, 'use_shotstream', False)

    def forward(self, shots: List[Shot], fps: int = 24) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = len(shots)
        features = []
        boundary_mask = torch.zeros(
            1, seq_len, device=self.temporal_pos_embed.device)

        for i, shot in enumerate(shots):
            shot_emb = self.shot_type_embed(self._to_idx(shot.shot_type, self._shot_type_map)) + \
                self.camera_motion_embed(self._to_idx(shot.camera_motion, self._camera_motion_map)) + \
                self.lighting_embed(self._to_idx(shot.lighting, self._lighting_map)) + \
                self.color_tone_embed(self._to_idx(shot.color_tone, self._color_tone_map)) + \
                self.rhythm_embed(self._to_idx(shot.rhythm, self._rhythm_map)) + \
                self.transition_embed(self._to_idx(
                    shot.transition, self._transition_map))

            motion_params = torch.tensor([shot.motion_params.get('speed', 0.0),
                                          shot.motion_params.get('angle', 0.0),
                                          shot.motion_params.get(
                                              'distance', 0.0),
                                          shot.motion_params.get('curve', 0.0)],
                                         dtype=torch.float32, device=self.temporal_pos_embed.device)
            motion_emb = self.motion_param_encoder(
                motion_params.unsqueeze(0)).squeeze(0)
            focus = torch.tensor(
                shot.focus, dtype=torch.float32, device=self.temporal_pos_embed.device)
            focus_emb = self.focus_encoder(focus.unsqueeze(0)).squeeze(0)
            features.append(shot_emb + motion_emb + focus_emb)

            if self.use_shotstream and i > 0:
                boundary_mask[0, i] = 1.0

        features = torch.stack(features, dim=0).unsqueeze(0)

        if seq_len <= self.temporal_pos_embed.shape[1]:
            pos_emb = self.temporal_pos_embed[:, :seq_len, :]
        else:
            repeats = (
                seq_len + self.temporal_pos_embed.shape[1] - 1) // self.temporal_pos_embed.shape[1]
            pos_emb = self.temporal_pos_embed.repeat(1, repeats, 1)[
                :, :seq_len, :]

        if self.use_shotstream:
            boundary_emb = self.boundary_embed(boundary_mask.long())
            pos_emb = pos_emb + boundary_emb

        features = features + pos_emb
        encoded = self.shot_encoder(features)
        cond = self.output_proj(encoded)
        return cond, boundary_mask

    _shot_type_map = {'extreme_long': 0, 'long': 1,
                      'medium': 2, 'close_up': 3, 'extreme_close': 4}
    _camera_motion_map = {'static': 0, 'pan': 1, 'tilt': 2,
                          'zoom': 3, 'dolly': 4, 'track': 5, 'crane': 6, 'handheld': 7}
    _lighting_map = {'natural': 0, 'dramatic': 1, 'soft': 2, 'hard': 3}
    _color_tone_map = {'neutral': 0, 'warm': 1,
                       'cool': 2, 'sepia': 3, 'high_contrast': 4}
    _rhythm_map = {'slow': 0, 'normal': 1, 'fast': 2}
    _transition_map = {'cut': 0, 'fade': 1,
                       'dissolve': 2, 'wipe': 3, 'zoom': 4}

    def _to_idx(self, key, mapping):
        return mapping.get(key, 0)


class LensScriptParser:
    @staticmethod
    def parse_script(json_path: str) -> List[Shot]:
        with open(json_path, 'r') as f:
            data = json.load(f)
        shots = []
        for item in data['shots']:
            shot = Shot(
                duration=item.get('duration', 2.0),
                scene_desc=item.get('scene_desc', ''),
                shot_type=item.get('shot_type', 'medium'),
                camera_motion=item.get('camera_motion', 'static'),
                motion_params=item.get('motion_params', {}),
                focus=tuple(item.get('focus', [0.5, 0.5])),
                lighting=item.get('lighting', 'natural'),
                color_tone=item.get('color_tone', 'neutral'),
                rhythm=item.get('rhythm', 'normal'),
                transition=item.get('transition', 'cut')
            )
            shots.append(shot)
        return shots