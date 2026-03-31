import numpy as np


def temporal_blend(video1: np.ndarray, video2: np.ndarray, overlap_frames: int, method='linear'):
    """
    对两个视频序列进行时域融合
    video1: 前一段视频（最后 overlap_frames 帧与 video2 重叠）
    video2: 后一段视频（前 overlap_frames 帧与 video1 重叠）
    """
    if method == 'linear':
        alpha = np.linspace(0, 1, overlap_frames).reshape(-1, 1, 1, 1)
        blended = video1[-overlap_frames:] * \
            (1 - alpha) + video2[:overlap_frames] * alpha
        return np.concatenate([video1[:-overlap_frames], blended, video2[overlap_frames:]], axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")
