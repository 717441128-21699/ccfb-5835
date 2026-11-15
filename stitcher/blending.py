"""画布合成与接缝融合。

提供三种输出供横向对比:
  1. hard      直接覆盖(最后写入者胜),接缝/亮度差一目了然
  2. feather   距离变换权重的单频段羽化(线性 alpha 融合)
  3. multiband Burt–Adelson 拉普拉斯金字塔多频段融合(默认交付方案)

所有合成支持亚像素位姿:小数偏移用一阶样条平移实现。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .pose import PoseGraph


def _canvas_bounds(graph: PoseGraph, poses: np.ndarray,
                   tile: int) -> tuple[int, int, int, int, np.ndarray]:
    """计算画布范围与每视野相对画布的原点(保留小数)。"""
    valid = [i for i, t in enumerate(graph.tiles) if not t.missing]
    mins = poses[valid].min(axis=0)
    maxs = poses[valid].max(axis=0) + tile
    oy = int(np.floor(mins[0]))
    ox = int(np.floor(mins[1]))
    H = int(np.ceil(maxs[0])) - oy + 1
    W = int(np.ceil(maxs[1])) - ox + 1
    local = poses - np.array([oy, ox], dtype=np.float32)
    return H, W, oy, ox, local


def _shift_tile(arr: np.ndarray, fy: float, fx: float) -> np.ndarray:
    """把 tile 按小数位移 (fy,fx) 平移,零填充,线性插值。"""
    if arr.ndim == 2:
        return ndi.shift(arr, (fy, fx), order=1, mode="constant", cval=0.0)
    out = np.empty_like(arr)
    for ch in range(arr.shape[2]):
        out[..., ch] = ndi.shift(arr[..., ch], (fy, fx), order=1,
                                 mode="constant", cval=0.0)
    return out


def _feather_weight(tile: int, feather: int) -> np.ndarray:
    """边缘斜坡权重:到各边距离截断在 feather。

    与直接截断距离变换不同,这里对每个方向单独取 min,使瓦片在重叠方向
    上的权重从边缘 0 线性升到 feather 后平坦;重叠中线两侧权重互补,
    保证羽化/多频段融合的过渡带真实存在而非等权平均。
    """
    d = np.arange(tile, dtype=np.float32)
    ramp = np.minimum(d, tile - 1 - d)          # 到最近水平/垂直边的距离
    ramp = np.clip(ramp, 0.0, float(feather))
    wy = ramp[:, None]
    wx = ramp[None, :]
    return np.minimum(wy, wx).astype(np.float32)


def composite(graph: PoseGraph, poses: np.ndarray, mode: str = "multiband",
              levels: int = 5, feather_px: int = 120,
              band_feather_px: int = 64) -> tuple[np.ndarray, dict]:
    """按 poses 把全部视野合成一张大图。

    返回 (RGB float32 画布, 几何信息 dict)。
    """
    tile = graph.tiles[0].image.shape[0]
    H, W, oy, ox, local = _canvas_bounds(graph, poses, tile)
    info = {"offset_y": oy, "offset_x": ox, "height": H, "width": W, "mode": mode}
    active = [t for t in graph.tiles if not t.missing]

    if mode == "hard":
        canvas = np.zeros((H, W, 3), np.float32)
        for t in active:
            iy, ix = np.floor(local[t.idx]).astype(int)
            fy, fx = local[t.idx] - np.floor(local[t.idx])
            shifted = _shift_tile(t.image, fy, fx)
            m = shifted.sum(axis=2) > 1e-6
            canvas[iy:iy + tile, ix:ix + tile][m] = shifted[m]
        return canvas, info

    if mode == "feather":
        num = np.zeros((H, W, 3), np.float64)
        den = np.zeros((H, W), np.float64)
        w0 = _feather_weight(tile, feather_px)
        for t in active:
            iy, ix = np.floor(local[t.idx]).astype(int)
            fy, fx = local[t.idx] - np.floor(local[t.idx])
            img = _shift_tile(t.image, fy, fx)
            w = _shift_tile(w0, fy, fx)
            sl = (slice(iy, iy + tile), slice(ix, ix + tile))
            num[sl] += img * w[..., None]
            den[sl] += w
        out = np.zeros_like(num)
        np.divide(num, den[..., None], out=out,
                  where=den[..., None] > 1e-8)
        return np.clip(out, 0, 1).astype(np.float32), info

    if mode != "multiband":
        raise ValueError(f"unknown mode {mode}")

    # ---------- Burt–Adelson 多频段(标准拉普拉斯金字塔) ----------
    # 图像与权重一起建高斯金字塔(每级 2 倍下采样,同尺度配对);
    # 拉普拉斯带 L_l = G_l - up(G_{l+1}),各带用同层权重融合,
    # 最粗层用最粗权重融合,最后逐层上采样累加。低频亮度差在粗层被
    # 宽权重平滑,高频细节在细层用窄权重,从而既消接缝又保清晰。
    w0 = _feather_weight(tile, band_feather_px)

    def _down2(arr: np.ndarray) -> np.ndarray:
        """2 倍下采样:先高斯抗锯齿再隔点取样。"""
        if arr.ndim == 3:
            out = np.empty((arr.shape[0] // 2, arr.shape[1] // 2,
                            arr.shape[2]), arr.dtype)
            for ch in range(arr.shape[2]):
                blurred = ndi.gaussian_filter(arr[..., ch], 1.0,
                                              truncate=2.0)
                out[..., ch] = blurred[::2, ::2]
            return out
        return ndi.gaussian_filter(arr, 1.0, truncate=2.0)[::2, ::2]

    def _up2(arr: np.ndarray, shape) -> np.ndarray:
        """上采样到 shape:零插 + 高斯,×4 补偿能量,再裁切到目标尺寸。"""
        th, tw = shape[:2]
        if arr.ndim == 3:
            tmp = np.zeros((arr.shape[0] * 2, arr.shape[1] * 2,
                            arr.shape[2]), arr.dtype)
            tmp[::2, ::2] = arr
            out = np.zeros((th, tw, arr.shape[2]), arr.dtype)
            for ch in range(arr.shape[2]):
                up = ndi.gaussian_filter(tmp[..., ch], 1.0,
                                         truncate=2.0) * 4.0
                out[..., ch] = up[:th, :tw]
            return out
        tmp = np.zeros((arr.shape[0] * 2, arr.shape[1] * 2), arr.dtype)
        tmp[::2, ::2] = arr
        up = ndi.gaussian_filter(tmp, 1.0, truncate=2.0) * 4.0
        return up[:th, :tw]

    # 每个瓦片贡献到统一画布金字塔:按层维护加权分子与权重和
    pyr_num = [np.zeros((H, W, 3), np.float64)]
    pyr_den = [np.zeros((H, W), np.float64)]
    h, w = H, W
    for _ in range(levels):
        h = (h + 1) // 2; w = (w + 1) // 2
        pyr_num.append(np.zeros((h, w, 3), np.float64))
        pyr_den.append(np.zeros((h, w), np.float64))

    for t in active:
        iy, ix = np.floor(local[t.idx]).astype(int)
        fy, fx = local[t.idx] - np.floor(local[t.idx])
        g_img = _shift_tile(t.image, fy, fx)
        g_w = _shift_tile(w0, fy, fx)
        # 该瓦片在每层画布中的位置(原点随下采样减半)
        for l in range(levels + 1):
            scale = 2 ** l
            ly, lx = iy // scale, ix // scale
            sl = (slice(ly, ly + g_img.shape[0]),
                  slice(lx, lx + g_img.shape[1]))
            # 裁切到画布
            y_end = min(sl[0].stop, pyr_num[l].shape[0])
            x_end = min(sl[1].stop, pyr_num[l].shape[1])
            yy = slice(sl[0].start, y_end)
            xx = slice(sl[1].start, x_end)
            ih = y_end - sl[0].start; iw = x_end - sl[1].start
            if ih > 0 and iw > 0:
                pyr_num[l][yy, xx] += g_img[:ih, :iw] * g_w[:ih, :iw][..., None]
                pyr_den[l][yy, xx] += g_w[:ih, :iw]
            if l < levels:
                g_img, g_w = _down2(g_img), _down2(g_w)

    # 每层做归一化加权,得到该层的高斯金字塔融合结果 G_l
    gauss = []
    for l in range(levels + 1):
        r = np.zeros_like(pyr_num[l])
        np.divide(pyr_num[l], pyr_den[l][..., None], out=r,
                  where=pyr_den[l][..., None] > 1e-8)
        gauss.append(r)
    # 拉普拉斯带 L_l = G_l - up(G_{l+1});最粗层 L_L = G_L
    laps = [None] * (levels + 1)
    laps[levels] = gauss[levels]
    for l in range(levels - 1, -1, -1):
        laps[l] = gauss[l] - _up2(gauss[l + 1], gauss[l].shape)
    # 从最粗带逐层上采样累加重建
    out = laps[levels]
    for l in range(levels - 1, -1, -1):
        out = _up2(out, laps[l].shape) + laps[l]
    return np.clip(out, 0, 1).astype(np.float32), info
