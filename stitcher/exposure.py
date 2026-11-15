"""曝光补偿:估计每个视野每通道的乘性增益,使重叠区亮度一致。

模型 g_b * I_b = g_a * I_a,在所有匹配边的重叠像素上做稀疏最小二乘
(参考 Uyttendaele 的增益补偿思路),增益夹在合理区间防止病态解。
"""
from __future__ import annotations

import numpy as np

from .pose import PoseGraph
from .io_utils import to_gray


def _overlap_samples(img_a: np.ndarray, img_b: np.ndarray, t_ab: np.ndarray,
                     max_samples: int = 2500,
                     seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """取两视野真实重叠区内的对应像素对(各通道)。"""
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    dy, dx = int(round(float(t_ab[0]))), int(round(float(t_ab[1])))
    # B 原点相对 A 原点的偏移;重叠矩形在 A 坐标系中
    ay0, ay1 = max(0, dy), min(ha, hb + dy)
    ax0, ax1 = max(0, dx), min(wa, wb + dx)
    if ay1 - ay0 < 8 or ax1 - ax0 < 8:
        return (np.empty((0, 3), np.float32),
                np.empty((0, 3), np.float32))
    va = img_a[ay0:ay1, ax0:ax1].reshape(-1, 3)
    vb = img_b[ay0 - dy:ay1 - dy, ax0 - dx:ax1 - dx].reshape(-1, 3)
    # 跳过过曝/欠曝像素(那里没有可靠亮度信息)
    lum = 0.5 * (to_gray(va.reshape(-1, 1, 3)).ravel()
                 + to_gray(vb.reshape(-1, 1, 3)).ravel())
    ok = (lum > 0.08) & (lum < 0.95)
    va, vb = va[ok], vb[ok]
    if len(va) > max_samples:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(va), max_samples, replace=False)
        va, vb = va[sel], vb[sel]
    return va, vb


def estimate_gains(graph: PoseGraph, clamp: float = 1.35,
                   reg: float = 0.02) -> np.ndarray:
    """返回 (N,3) 增益矩阵;缺失视野增益为 1。

    每通道独立求解:最小化 Σ w (g_a va - g_b vb)² + reg Σ (g_i - 1)²
    """
    n = len(graph.tiles)
    gains = np.ones((n, 3), dtype=np.float32)
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr

    for ch in range(3):
        rows, cols, data, rhs = [], [], [], []
        rid = 0
        for e in graph.edges:
            ta, tb = graph.tiles[e.a], graph.tiles[e.b]
            if ta.image is None or tb.image is None:
                continue
            va, vb = _overlap_samples(ta.image, tb.image, e.t)
            if len(va) == 0:
                continue
            x = va[:, ch].astype(np.float64)
            y = vb[:, ch].astype(np.float64)
            # 每条边压成一个加权聚合行,避免矩阵随像素数膨胀:
            # g_a*Σx² = g_b*Σxy (最小二乘的法方程形式,按边加权)
            sxx = float(x @ x)
            syy = float(y @ y)
            sxy = float(x @ y)
            w = e.weight
            rows += [rid, rid]; cols += [e.a, e.b]
            data += [w * sxx, -w * sxy]; rhs.append(0.0)
            rid += 1
            # 反向再给一条,平衡 x/y 量纲
            rows += [rid, rid]; cols += [e.b, e.a]
            data += [w * syy, -w * sxy]; rhs.append(0.0)
            rid += 1
        # 正则:把增益拉回 1
        for t in graph.tiles:
            if t.missing:
                continue
            rows.append(rid); cols.append(t.idx)
            data.append(reg * 1e5); rhs.append(reg * 1e5)
            rid += 1
        if rid == 0:
            continue
        A = csr_matrix((data, (rows, cols)), shape=(rid, n), dtype=np.float64)
        g = lsqr(A, np.array(rhs), damp=0.0, atol=1e-10, btol=1e-10)[0]
        g = np.clip(g, 1.0 / clamp, clamp)
        for t in graph.tiles:
            if not t.missing:
                gains[t.idx, ch] = g[t.idx]

    # 绝对曝光锚定:增益只有相对意义(整体尺度自由)。取每个瓦片中心
    # 1/4 区域(暗角最弱)的平均亮度,求使补偿后中心亮度对齐全体中位数
    # 的每通道尺度因子,纠正暗角/增益导致的整体偏暗或偏亮。
    for t in graph.tiles:
        if t.missing or t.image is None:
            continue
        h, w = t.image.shape[:2]
        cy0, cy1 = h * 3 // 8, h * 5 // 8
        cx0, cx1 = w * 3 // 8, w * 5 // 8
        t._center_mean = t.image[cy0:cy1, cx0:cx1].mean(axis=(0, 1))
    valid_tiles = [t for t in graph.tiles if not t.missing]
    centers = np.stack([t._center_mean for t in valid_tiles], axis=0)
    target = np.median(centers, axis=0)
    for ch in range(3):
        # 补偿后中心亮度 = gain_i * center_i,取加权(等权)尺度使中位数对齐
        cur = gains[[t.idx for t in valid_tiles], ch] * centers[:, ch]
        scale = target[ch] / (np.median(cur) + 1e-9)
        gains[:, ch] = np.clip(gains[:, ch] * scale,
                               1.0 / clamp, clamp)
    return gains


def apply_gains(graph: PoseGraph, gains: np.ndarray) -> None:
    """就地把增益乘到各视野图像上。"""
    for t in graph.tiles:
        if t.image is not None:
            t.image = np.clip(
                t.image * gains[t.idx][None, None, :], 0.0, 1.0).astype(np.float32)
