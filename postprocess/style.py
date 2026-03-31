import cv2
import numpy as np
from typing import List


class StyleFilter:
    @staticmethod
    def cartoon(frame: np.ndarray, sigma_s: int = 60, sigma_r: float = 0.5) -> np.ndarray:
        smooth = cv2.stylization(frame, sigma_s=sigma_s, sigma_r=sigma_r)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(smooth, edges)

    @staticmethod
    def oil_painting(frame: np.ndarray, size: int = 7, intensity: int = 1) -> np.ndarray:
        return cv2.xphoto.oilPainting(frame, size, intensity)

    @staticmethod
    def sketch(frame: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv_gray = 255 - gray
        blurred = cv2.GaussianBlur(inv_gray, (21, 21), sigma)
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def sepia(frame: np.ndarray) -> np.ndarray:
        sepia_matrix = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        return cv2.transform(frame, sepia_matrix)

    @staticmethod
    def pixelate(frame: np.ndarray, block_size: int = 8) -> np.ndarray:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // block_size, h //
                           block_size), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def edge_detect(frame: np.ndarray, low_thresh: int = 50, high_thresh: int = 150) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low_thresh, high_thresh)
        edges = cv2.bitwise_not(edges)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def watercolor(frame: np.ndarray, sigma_s: int = 60, sigma_r: float = 0.8) -> np.ndarray:
        return cv2.stylization(frame, sigma_s=sigma_s, sigma_r=sigma_r)

    @staticmethod
    def pencil_sketch(frame: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
        sketch = cv2.divide(gray, blurred, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def apply_style(frames: List[np.ndarray], style_name: str, **kwargs) -> List[np.ndarray]:
    style_map = {
        'cartoon': StyleFilter.cartoon,
        'oil': StyleFilter.oil_painting,
        'sketch': StyleFilter.sketch,
        'sepia': StyleFilter.sepia,
        'pixelate': StyleFilter.pixelate,
        'edge': StyleFilter.edge_detect,
        'watercolor': StyleFilter.watercolor,
        'pencil': StyleFilter.pencil_sketch,
    }
    if style_name not in style_map:
        raise ValueError(f"Unknown style: {style_name}")
    func = style_map[style_name]
    processed = [func(frame, **kwargs) for frame in frames]
    return processed
