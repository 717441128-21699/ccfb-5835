"""相邻视野间的平移估计:几何门控特征配对 -> 稳健共识 -> NCC 亚像素细化,
特征不足时在名义位置附近做局部 NCC 搜索兜底(抗重复纹理)。

全模块统一约定:t = origin_B - origin_A(原点偏移)。
若 B 中场景点对应 A 中同一点 p_a,则 p_b = p_a - t,即 t = p_a - p_b。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import Features, match_descriptors
from .io_utils import to_gray


@dataclass
class MatchResult:
    t: np.ndarray            # 估计的 (origin_B - origin_A) = (dy, dx)
    inliers: int
    method: str              # "harris" | "ncc_search" | "failed"
    residual_rms: float
    confidence: float


# --------------------------------------------------------------------------
# 1) 特征配对的稳健平移共识
# --------------------------------------------------------------------------
def _robust_translation(fa: Features, fb: Features, pairs: np.ndarray,
                        tol: float = 3.0) -> tuple[np.ndarray, np.ndarray, float]:
    """从配对点求原点偏移 t,迭代中位数去离群。"""
    pa = fa.points[pairs[:, 0]]
    pb = fb.points[pairs[:, 1]]
    disp = pa - pb
    t = np.median(disp, axis=0)
    inlier = np.ones(len(disp), bool)
    for _ in range(6):
        res = np.linalg.norm(disp - t, axis=1)
        new_in = res < tol
        if np.array_equal(new_in, inlier):
            break
        inlier = new_in
        t = np.median(disp[inlier], axis=0)
    res = np.linalg.norm(disp[inlier] - t, axis=1)
    rms = float(np.sqrt(np.mean(res ** 2))) if inlier.any() else np.inf
    return t, inlier, rms


# --------------------------------------------------------------------------
# 2) 相位相关(备用手段,名义重叠窗内局部搜索)
# --------------------------------------------------------------------------
def _hann2d(h: int, w: int) -> np.ndarray:
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def _parabolic_subpixel(pc: np.ndarray, py: int, px: int,
                        h: int, w: int) -> tuple[float, float]:
    """峰值附近一维抛物面插值,返回亚像素修正 (dy, dx)。"""
    def shift1(v_prev: float, v: float, v_next: float) -> float:
        den = v_prev - 2 * v + v_next
        if abs(den) < 1e-12:
            return 0.0
        return 0.5 * (v_prev - v_next) / den

    dy = shift1(pc[(py - 1) % h, px], pc[py, px], pc[(py + 1) % h, px])
    dx = shift1(pc[py, (px - 1) % w], pc[py, px], pc[py, (px + 1) % w])
    return float(np.clip(dy, -1, 1)), float(np.clip(dx, -1, 1))


def phase_correlation(gray_a: np.ndarray, gray_b: np.ndarray,
                      t_nom: np.ndarray, search: int = 12
                      ) -> tuple[np.ndarray, float]:
    """名义平移 t_nom 附近 ±search 的局部相位相关。

    重叠窗按名义偏移取齐后,正确平移对应零位移峰,真实峰只在中心附近。
    返回 (相对 t_nom 的原点偏移修正量, 峰值能量占比)。
    """
    ha, wa = gray_a.shape
    hb, wb = gray_b.shape
    dy0, dx0 = int(round(float(t_nom[0]))), int(round(float(t_nom[1])))

    ay0, ax0 = max(0, dy0), max(0, dx0)
    by0, bx0 = ay0 - dy0, ax0 - dx0
    h = min(ha - ay0, hb - by0) - 2 * search
    w = min(wa - ax0, wb - bx0) - 2 * search
    if h < 48 or w < 48:
        return np.zeros(2, np.float32), 0.0

    win_a = gray_a[ay0:ay0 + h, ax0:ax0 + w]
    win_b = gray_b[by0:by0 + h, bx0:bx0 + w]
    hann = _hann2d(h, w)
    fa = np.fft.fft2((win_a - win_a.mean()) * hann)
    fb = np.fft.fft2((win_b - win_b.mean()) * hann)
    cross = fa * np.conj(fb)
    mag = np.abs(cross)
    mag[mag == 0] = 1e-12
    pc = np.fft.ifft2(cross / mag).real

    cy, cx = h // 2, w // 2
    band = np.zeros_like(pc)
    band[max(0, cy - search):cy + search + 1,
         max(0, cx - search):cx + search + 1] = 1.0
    band[0:search + 1, 0:search + 1] = 1.0          # 零峰的环绕等价位置
    band[h - search:, w - search:] = 1.0
    band[h - search:, 0:search + 1] = 1.0
    band[0:search + 1, w - search:] = 1.0
    masked = pc * band
    py, px = np.unravel_index(np.argmax(masked), pc.shape)
    peak = float(masked[py, px])
    sy = py if py <= h // 2 else py - h
    sx = px if px <= w // 2 else px - w
    sub = _parabolic_subpixel(pc, py, px, h, w)
    corr = float(peak / (np.abs(masked).sum() + 1e-12))
    return np.array([sy + sub[0], sx + sub[1]], np.float32), corr


# --------------------------------------------------------------------------
# 3) 重叠窗 NCC:亚像素细化与局部搜索
# --------------------------------------------------------------------------
def _ncc_score(ref: np.ndarray, cand: np.ndarray) -> float:
    a = ref - ref.mean()
    b = cand - cand.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-9 else -2.0


def _search_windows(gray_a, gray_b, t_int, radius, patch,
                    min_size: int = 32):
    """在真实重叠区内取参考窗,保证 ±radius 搜索时 B 窗不越界。

    原点偏移 t=(dy,dx),重叠矩形在 A 坐标为:
      ay∈[max(0,dy), min(Ha,Hb+dy)),ax∈[max(0,dx), min(Wa,Wb+dx))
    B 中对应起点 = A 起点 - t。返回 None 表示重叠区太小。
    """
    ha, wa = gray_a.shape
    hb, wb = gray_b.shape
    dy, dx = int(round(float(t_int[0]))), int(round(float(t_int[1])))

    ay_lo, ay_hi = max(0, dy), min(ha, hb + dy)
    ax_lo, ax_hi = max(0, dx), min(wa, wb + dx)
    # 内缩 radius:候选 B 窗在参考窗四周移动 radius 时仍完整落在图内
    ay_lo += radius; ax_lo += radius
    ay_hi -= radius; ax_hi -= radius
    if ay_hi - ay_lo < min_size or ax_hi - ax_lo < min_size:
        return None
    ph = min(patch, ay_hi - ay_lo)
    pw = min(patch, ax_hi - ax_lo)
    ay0, ax0 = ay_lo, ax_lo
    by0, bx0 = ay0 - dy, ax0 - dx
    ref = gray_a[ay0:ay0 + ph, ax0:ax0 + pw].astype(np.float32)
    return ref, by0, bx0, ph, pw


def ncc_refine(gray_a: np.ndarray, gray_b: np.ndarray, t_int: np.ndarray,
               radius: int = 2, patch: int = 256) -> np.ndarray:
    """以整像素原点偏移 t_int 为中心,±radius 内 NCC + 抛物面亚像素。"""
    ref, by0, bx0, ph, pw = _search_windows(
        gray_a, gray_b, t_int, radius, patch)
    if ref is None:
        return t_int.astype(np.float32)

    scores = np.full((2 * radius + 1, 2 * radius + 1), -2.0, np.float32)
    for i, dy in enumerate(range(-radius, radius + 1)):
        for j, dx in enumerate(range(-radius, radius + 1)):
            cand = gray_b[by0 + dy:by0 + dy + ph,
                          bx0 + dx:bx0 + dx + pw].astype(np.float32)
            if cand.shape == ref.shape:
                scores[i, j] = _ncc_score(ref, cand)

    i, j = np.unravel_index(np.argmax(scores), scores.shape)
    if scores[i, j] < 0.2:
        return t_int.astype(np.float32)
    sy = _parabolic_subpixel(scores, i, j, scores.shape[0], scores.shape[1])
    # cand = B[ref - (t_int + δ)],最佳内容位移 s=δ,原点偏移修正 = -δ
    return np.array([t_int[0] - (i - radius) - sy[0],
                     t_int[1] - (j - radius) - sy[1]], np.float32)


def ncc_local_search(gray_a: np.ndarray, gray_b: np.ndarray,
                     t_nom: np.ndarray, radius: int = 48,
                     patch: int = 320) -> tuple[np.ndarray, float]:
    """无可靠特征时的主力兜底:名义偏移附近两级 NCC 搜索。

    第一级在 1/2 降采样图上以步长 2 粗搜(覆盖 ±radius,抗大抖动);
    第二级回到原图在粗解 ±3 内精修并做亚像素插值。
    """
    # ---- 粗搜:降采样,坐标和半径都减半;半径受重叠区尺寸自适应约束 ----
    ga2 = gray_a[::2, ::2]
    gb2 = gray_b[::2, ::2]
    t_nom2 = t_nom / 2.0
    rad2 = max(3, radius // 2)
    # 重叠区短边的一半决定可承受的最大搜索半径
    dy2, dx2 = int(round(float(t_nom2[0]))), int(round(float(t_nom2[1])))
    ov_h = min(ga2.shape[0], gb2.shape[0] + dy2) - max(0, dy2)
    ov_w = min(ga2.shape[1], gb2.shape[1] + dx2) - max(0, dx2)
    rad2 = min(rad2, max(3, (min(ov_h, ov_w) - 36) // 2))
    win = _search_windows(ga2, gb2, np.round(t_nom2).astype(int),
                          rad2, patch // 2, min_size=24)
    if win is None:
        return t_nom.astype(np.float32), 0.0
    ref, by0, bx0, ph, pw = win
    best = (-3.0, 0, 0)
    for i in range(-rad2, rad2 + 1):
        for j in range(-rad2, rad2 + 1):
            cand = gb2[by0 + i:by0 + i + ph, bx0 + j:bx0 + j + pw]
            if cand.shape == ref.shape:
                s = _ncc_score(ref, cand)
                if s > best[0]:
                    best = (s, i, j)
    score2, i2, j2 = best
    if score2 < 0.12:
        return t_nom.astype(np.float32), float(score2)
    # 降采样图的内容位移 (i2,j2) -> 原点偏移,乘 2 回到原图尺度
    t_coarse = np.array([dy2 - i2, dx2 - j2], np.float32) * 2.0

    # ---- 精修:原图 ±3 整数 + 抛物面亚像素 ----
    t_refined = ncc_refine(gray_a, gray_b, np.round(t_coarse),
                           radius=3, patch=patch)
    return t_refined, float(score2)


# --------------------------------------------------------------------------
# 4) 单对视野的完整匹配流程
# --------------------------------------------------------------------------
def match_pair(img_a: np.ndarray, img_b: np.ndarray,
               fa: Features, fb: Features, t_nom: np.ndarray,
               min_inliers: int = 8, ransac_tol: float = 3.0,
               max_nominal_dev: float = 60.0) -> MatchResult:
    """估计 B 相对 A 的原点偏移 (origin_B - origin_A)。

    描述子配对 -> 名义几何门控(重复纹理误配在此被剔除)
    -> 中位数共识 -> NCC 亚像素细化;特征不足则局部 NCC 搜索兜底。
    """
    pairs = match_descriptors(fa, fb)
    t = None
    inliers = 0
    rms = np.inf
    method = "harris"
    ga, gb = to_gray(img_a), to_gray(img_b)

    if len(pairs) >= 4:
        pa = fa.points[pairs[:, 0]]
        pb = fb.points[pairs[:, 1]]
        disp = pa - pb
        geom = np.linalg.norm(disp - t_nom, axis=1) < max_nominal_dev
        if geom.sum() >= 4:
            t_raw, inlier_mask, rms = _robust_translation(
                fa, fb, pairs[geom], ransac_tol)
            if inlier_mask.sum() >= min_inliers:
                t = t_raw
                inliers = int(inlier_mask.sum())

    if t is None:
        t, score = ncc_local_search(ga, gb, t_nom)
        method = "ncc_search"
        inliers = int(max(0.0, score) * 25)
        rms = 1.2
        if score < 0.15:
            return MatchResult(t, 0, "failed", np.inf, 0.0)
    else:
        t = ncc_refine(ga, gb, np.round(t))

    confidence = float(np.clip(
        inliers / 30.0, 0.0, 1.0) * np.exp(-max(rms, 0.0) / 3.0))
    return MatchResult(np.asarray(t, np.float32), inliers, method,
                       rms, confidence)
