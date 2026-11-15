"""瓦片内亮度平坦化:纠正径向暗角等乘性低频不均。

思路:对灰度做大尺度高斯模糊得到局部亮度场,用「目标亮度 / 局部亮度」
作为逐像素乘子(只保留低频分量,不放大高频纹理),再按通道施加。
平坦化在曝光增益之后、融合之前执行。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .io_utils import to_gray
from .pose import PoseGraph


def flatten_tile(img: np.ndarray, sigma_ratio: float = 0.35,
                 strength: float = 0.85, floor: float = 0.6) -> np.ndarray:
    """消除单张瓦片的低频亮度不均。

    sigma_ratio: 模糊核 = 边长 × 该比例,远大于特征尺度,只抓暗角级低频。
    strength:    纠正强度,1.0 为完全拉平,<1 保留轻微自然感。
    floor:       乘子下限,防止极暗区噪声被过度放大。
    """
    h, w = img.shape[:2]
    sigma = max(h, w) * sigma_ratio
    gray = to_gray(img)
    field = ndi.gaussian_filter(gray, sigma=sigma, mode="nearest")
    target = np.median(field)
    # 理想乘子 target/field;按 strength 做软化混合,避免矫枉过正
    full = target / np.maximum(field, 1e-3)
    mult = 1.0 + strength * (full - 1.0)
    mult = np.clip(mult, floor, 1.0 / floor).astype(np.float32)
    out = img * mult[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def flatten_graph(graph: PoseGraph, **kwargs) -> None:
    """对图中所有有效瓦片就地做亮度平坦化。"""
    for t in graph.tiles:
        if t.image is not None:
            t.image = flatten_tile(t.image, **kwargs)
