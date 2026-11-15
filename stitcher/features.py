"""特征检测与描述(纯 numpy 实现的 Harris + 归一化 patch 描述子)。

刻意不依赖 OpenCV/SIFT:显微镜视野纹理重复且单一,Harris 角点配合
高判别力的归一化互相关描述子足以稳定工作,且每一步都可解释、可调参。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .io_utils import to_gray


@dataclass
class Features:
    points: np.ndarray   # (N,2) 角点坐标 (y, x),亚像素精度
    desc: np.ndarray     # (N,D) 单位长度描述子
    radius: int          # 描述子 patch 半径


def _box_filter(a: np.ndarray, k: int) -> np.ndarray:
    """可分离均值滤波,边界用 'nearest' 延拓。"""
    if k <= 1:
        return a
    pad = k // 2
    ap = np.pad(a, pad, mode="edge")
    cs = np.cumsum(ap, axis=1, dtype=np.float64)
    # 水平方向长度 k 的窗口和
    h = cs[:, k - 1:k - 1 + a.shape[1]].copy()
    h[:, 1:] = cs[:, k:k + a.shape[1]] - cs[:, :a.shape[1] - 1]
    cs2 = np.cumsum(h, axis=0, dtype=np.float64)
    v = cs2[k - 1:k - 1 + a.shape[0], :].copy()
    v[1:, :] = (cs2[k:k + a.shape[0], :]
                - cs2[:a.shape[0] - 1, :])
    return (v / (k * k)).astype(np.float32)


def harris_corners(gray: np.ndarray, max_points: int = 900,
                   quality: float = 0.01, min_dist: int = 9,
                   sigma: float = 1.2) -> np.ndarray:
    """Harris 角点检测,返回 (N,2) 的 (y,x) 像素坐标。

    quality: 响应阈值 = quality × 最大响应;min_dist: 非极大值抑制半径。
    """
    g = gray.astype(np.float32)
    gx = ndi.gaussian_filter1d(g, sigma, axis=1, order=1)
    gy = ndi.gaussian_filter1d(g, sigma, axis=0, order=1)

    ksize = max(5, int(round(3 * sigma)) | 1)
    ixx = _box_filter(gx * gx, ksize)
    iyy = _box_filter(gy * gy, ksize)
    ixy = _box_filter(gx * gy, ksize)

    k = 0.06
    resp = (ixx * iyy - ixy * ixy) - k * (ixx + iyy) ** 2
    resp = np.maximum(resp, 0.0)

    # 局部最大值
    mx = ndi.maximum_filter(resp, size=2 * min_dist + 1)
    cand = (resp == mx) & (resp > quality * resp.max())
    ys, xs = np.nonzero(cand)
    if len(ys) == 0:
        return np.empty((0, 2), dtype=np.float32)

    # 按响应排序后做贪心最小间距抑制
    order = np.argsort(resp[ys, xs])[::-1][: max_points * 4]
    kept: list[tuple[int, int]] = []
    taken = np.zeros(gray.shape, dtype=bool)
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if taken[y, x]:
            continue
        kept.append((y, x))
        y0, y1 = max(0, y - min_dist), y + min_dist + 1
        x0, x1 = max(0, x - min_dist), x + min_dist + 1
        taken[y0:y1, x0:x1] = True
        if len(kept) >= max_points:
            break
    return np.asarray(kept, dtype=np.float32)


def _describe(gray: np.ndarray, points: np.ndarray, radius: int = 8) -> np.ndarray:
    """以每个点为中心取 (2r+1)² patch,减均值除标准差后展平并单位化。"""
    k = 2 * radius + 1
    h, w = gray.shape
    n = len(points)
    desc = np.zeros((n, k * k), dtype=np.float32)
    # 边界外 nearest 延拓
    padded = np.pad(gray, radius, mode="edge")
    for i, (y, x) in enumerate(points.astype(int)):
        patch = padded[y:y + k, x:x + k].astype(np.float32)
        patch -= patch.mean()
        std = patch.std()
        if std > 1e-6:
            patch /= std
        d = patch.ravel()
        nrm = np.linalg.norm(d)
        if nrm > 1e-6:
            d /= nrm
        desc[i] = d
    return desc


def extract_features(img: np.ndarray, max_points: int = 900,
                     desc_radius: int = 8) -> Features:
    gray = to_gray(img)
    pts = harris_corners(gray, max_points=max_points, min_dist=desc_radius + 1)
    if len(pts) == 0:
        return Features(pts, np.empty((0, (2 * desc_radius + 1) ** 2),
                                      dtype=np.float32), desc_radius)
    desc = _describe(gray, pts, desc_radius)
    return Features(pts, desc, desc_radius)


def match_descriptors(fa: Features, fb: Features,
                      ratio_thresh: float = 0.8) -> np.ndarray:
    """双向最近邻 + Lowe 比值测试,返回 (M,2) 配对索引。"""
    if len(fa.points) < 2 or len(fb.points) < 2:
        return np.empty((0, 2), dtype=np.int64)
    # 描述子已单位化,内积越大越相似 -> 负内积当距离
    sim = fa.desc @ fb.desc.T
    nn_b = np.argmax(sim, axis=1)
    row = np.arange(len(fa.points))
    s_sorted = np.sort(sim, axis=1)
    best = s_sorted[:, -1]
    second = s_sorted[:, -2]
    # 相似度域的比值测试: (1-best)/(1-second) < ratio
    denom = np.maximum(1.0 - second, 1e-9)
    ratio_ok = ((1.0 - best) / denom) < ratio_thresh
    mutual = np.argmax(sim, axis=0)[nn_b] == row
    ok = ratio_ok & mutual
    pairs = np.stack([row[ok], nn_b[ok]], axis=1)
    return pairs
