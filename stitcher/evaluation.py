"""拼接质量评估:真值位姿误差 + 盲对齐图像误差 + 接缝指标。

真值只在本模块使用;拼接流程本身只依赖视野图像与名义网格。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .io_utils import save_gray, save_rgb, to_gray
from .pose import PoseGraph


# --------------------------------------------------------------------------
# 位姿误差
# --------------------------------------------------------------------------
def pose_rmse(graph: PoseGraph, poses: np.ndarray,
              truth: dict[int, tuple[float, float]]) -> dict:
    """去掉单个全局平移(拼接只恢复相对几何)后,比较估计位姿与真值。"""
    ids = [i for i in truth if not graph.tiles[i].missing]
    est = poses[ids]
    gt = np.array([truth[i] for i in ids], dtype=np.float64)
    shift = (est - gt).mean(axis=0)
    err = est - shift - gt
    dist = np.linalg.norm(err, axis=1)
    return {
        "rmse_px": float(np.sqrt((dist ** 2).mean())),
        "mae_px": float(dist.mean()),
        "max_px": float(dist.max()),
        "per_tile": {int(i): float(d) for i, d in zip(ids, dist)},
    }


# --------------------------------------------------------------------------
# 盲对齐:不知道真值,用低分辨率相位相关找画布相对原图的全局偏移
# --------------------------------------------------------------------------
def _blind_offset(canvas: np.ndarray, source: np.ndarray,
                  max_side: int = 1400) -> tuple[int, int]:
    gc = to_gray(canvas)
    gs = to_gray(source)
    scale = max_side / max(gc.shape)
    if scale < 1.0:
        gc_s = ndi.zoom(gc, scale, order=1)
        gs_s = ndi.zoom(gs, scale, order=1)
    else:
        gc_s, gs_s, scale = gc, gs, 1.0
    hc, wc = gc_s.shape
    hs, ws = gs_s.shape
    H, W = max(hc, hs) + 64, max(wc, ws) + 64
    pc = np.zeros((H, W), np.float32)
    ps = np.zeros((H, W), np.float32)
    pc[:hc, :wc] = gc_s - gc_s.mean()
    ps[:hs, :ws] = gs_s - gs_s.mean()
    f = np.fft.fft2(pc) * np.conj(np.fft.fft2(ps))
    corr = np.fft.ifft2(f / (np.abs(f) + 1e-12)).real
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    sy = y if y <= H // 2 else y - H
    sx = x if x <= W // 2 else x - W
    return int(round(sy / scale)), int(round(sx / scale))


def image_errors(canvas: np.ndarray, source: np.ndarray,
                 out_prefix: str | None = None) -> dict:
    """盲对齐(整数粗对齐 + NCC 亚像素细化)后在有效区内算误差。"""
    dy_i, dx_i = _blind_offset(canvas, source)
    # 亚像素细化:避免整数量化把 ~0.06 的虚假误差算进指标
    from .matching import ncc_refine
    dy, dx = ncc_refine(to_gray(canvas), to_gray(source),
                        np.array([dy_i, dx_i], np.float32),
                        radius=3, patch=400)
    hc, wc = canvas.shape[:2]
    hs, ws = source.shape[:2]
    # 用亚像素平移把 source 采样到 canvas 坐标系
    shifted = np.zeros((hc, wc, 3), np.float32)
    for ch in range(3):
        shifted[..., ch] = ndi.shift(
            source[..., ch], (float(dy), float(dx)), order=1,
            mode="constant", cval=0.0)[:hc, :wc]
    valid = canvas.sum(axis=2) > 1e-4
    a, b = canvas, shifted
    diff = np.abs(a - b)
    se = (a - b) ** 2
    rmse = float(np.sqrt(se[valid].mean()))
    mae = float(diff[valid].mean())
    psnr = float(-10 * np.log10(se[valid].mean() + 1e-12))

    if out_prefix:
        gray_diff = diff.mean(axis=2)
        gray_diff[~valid] = 0.0
        save_gray(out_prefix + "_diff_heatmap.png", gray_diff)
        viz = np.zeros_like(a)
        viz[valid] = np.clip(np.abs(a - b)[valid] * 4.0, 0, 1)
        save_rgb(out_prefix + "_diff_rgb.png", viz)

    return {"rmse": rmse, "mae": mae, "psnr_db": psnr,
            "blind_shift_yx": [float(dy), float(dx)],
            "coverage": float(valid.mean())}


# --------------------------------------------------------------------------
# 接缝指标:在每条边的重叠区中线(权重相等的过渡带)量灰度跳变
# --------------------------------------------------------------------------
def seam_discontinuity(canvas: np.ndarray, graph: PoseGraph,
                       poses: np.ndarray, strip: int = 2) -> dict:
    """对每条匹配边,在两视野重叠中线量两侧窄条带的平均灰度跳变。

    羽化/多频段融合的过渡带位于重叠区中央而非瓦片名义边界,故中线取
    两原点偏移的一半位置。同时用同方向的天然图像梯度做基线,输出
    相对跳变,便于跨图比较。
    """
    gray = to_gray(canvas)
    H, W = gray.shape
    vals, rels = [], []
    for e in graph.edges:
        pa, pb = poses[e.a], poses[e.b]
        mid = ((pa + pb) / 2.0).round().astype(int)
        if e.axis == "h":                       # 水平邻居,接缝是竖线
            x = mid[1]
            y0, y1 = int(max(pa[0], pb[0])), int(min(pa[0], pb[0])
                                                  ) + graph.tiles[0].image.shape[0]
            y0, y1 = max(0, y0), min(H, y1)
            if x - strip - 1 < 0 or x + strip + 1 >= W or y1 - y0 < 16:
                continue
            left = gray[y0:y1, x - strip:x].mean(axis=1)
            right = gray[y0:y1, x:x + strip].mean(axis=1)
            jump = np.abs(left - right)
            base = np.abs(gray[y0:y1, x + strip:x + strip + 2].mean(axis=1)
                          - gray[y0:y1, x - strip - 2:x - strip].mean(axis=1))
        else:                                   # 垂直邻居,接缝是横线
            y = mid[0]
            x0, x1 = int(max(pa[1], pb[1])), int(min(pa[1], pb[1])
                                                  ) + graph.tiles[0].image.shape[1]
            x0, x1 = max(0, x0), min(W, x1)
            if y - strip - 1 < 0 or y + strip + 1 >= H or x1 - x0 < 16:
                continue
            up = gray[y - strip:y, x0:x1].mean(axis=0)
            down = gray[y:y + strip, x0:x1].mean(axis=0)
            jump = np.abs(up - down)
            base = np.abs(gray[y + strip:y + strip + 2, x0:x1].mean(axis=0)
                          - gray[y - strip - 2:y - strip, x0:x1].mean(axis=0))
        if len(jump):
            vals.append(float(jump.mean()))
            rels.append(float(jump.mean() / (base.mean() + 1e-6)))
    if not vals:
        return {"seam_mad": None, "n_edges": 0}
    return {"seam_mad": float(np.mean(vals)),
            "seam_p95": float(np.percentile(vals, 95)),
            "seam_relative_to_native": float(np.mean(rels)),
            "n_edges": len(vals)}
