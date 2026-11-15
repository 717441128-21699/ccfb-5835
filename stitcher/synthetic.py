"""合成 H&E 风格组织大图(无真实切片时的可复现测试源图)。

只依赖 numpy + PIL 绘图,生成带多尺度纹理、腺管结构、细胞核和间质纤维的
彩色大图,特征丰富、适合检验特征匹配;同一种子结果完全可复现。
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def _value_noise(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    """双线性插值的 value noise,返回 size×size、约 0~1 的二维场。"""
    grid = rng.random((cells + 1, cells + 1), dtype=np.float32)
    src = Image.fromarray((grid * 255).astype(np.uint8))
    dst = src.resize((size, size), Image.BILINEAR)
    return np.asarray(dst, dtype=np.float32) / 255.0


def _fractal_noise(rng: np.random.Generator, size: int, octaves: int = 5) -> np.ndarray:
    """多倍频 value noise 叠加,模拟组织的多尺度密度起伏。"""
    out = np.zeros((size, size), dtype=np.float32)
    amp, total = 1.0, 0.0
    cells = 8
    for _ in range(octaves):
        out += amp * _value_noise(rng, size, cells)
        total += amp
        amp *= 0.55
        cells *= 2
    return out / total


def make_synthetic_tissue(size: int = 4096, seed: int = 20261115) -> np.ndarray:
    """生成 size×size 的 H&E 风格 RGB 图(float32, 0~1)。"""
    rng = np.random.default_rng(seed)
    s = size

    # ---- 背景:伊红(粉红)染色的间质,密度由分形噪声调制 ----
    density = _fractal_noise(rng, s, octaves=5)
    density = np.clip((density - 0.35) / 0.5, 0.0, 1.0)
    base = np.empty((s, s, 3), dtype=np.float32)
    # 淡染区偏白,浓染区偏粉红
    base[..., 0] = 0.96 - 0.10 * density
    base[..., 1] = 0.90 - 0.22 * density
    base[..., 2] = 0.92 - 0.16 * density

    # ---- 腺管/血管:浅染的椭圆空腔,带一圈深色管壁 ----
    layer = Image.new("RGB", (s, s), (0, 0, 0))
    draw = ImageDraw.Draw(layer)
    n_glands = max(24, s * s // 90000)
    for _ in range(n_glands):
        cx, cy = rng.integers(0, s, 2)
        r = rng.integers(s // 120, s // 45)
        rw, rh = r * rng.uniform(0.7, 1.6), r * rng.uniform(0.7, 1.6)
        # 深色管壁
        wall = tuple(int(v * 255) for v in (0.55, 0.28, 0.42))
        draw.ellipse([cx - rw * 1.18, cy - rh * 1.18,
                      cx + rw * 1.18, cy + rh * 1.18], fill=wall)
        # 浅染空腔
        lumen = tuple(int(v * 255) for v in (0.93, 0.86, 0.88))
        draw.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=lumen)
    glands = np.asarray(layer, dtype=np.float32) / 255.0
    mask = glands.sum(axis=2) > 0
    base[mask] = glands[mask]

    # ---- 间质纤维:细长半透明线条,提供跨视野的高频特征 ----
    fiber_layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    fdraw = ImageDraw.Draw(fiber_layer)
    n_fibers = max(60, s * s // 45000)
    for _ in range(n_fibers):
        x0, y0 = rng.integers(0, s, 2)
        ang = rng.uniform(0, np.pi)
        length = rng.integers(s // 60, s // 18)
        x1 = x0 + length * np.cos(ang)
        y1 = y0 + length * np.sin(ang)
        shade = int(rng.integers(90, 150))
        alpha = int(rng.integers(40, 90))
        fdraw.line([x0, y0, x1, y1], fill=(shade, shade // 2, shade, alpha),
                   width=max(1, s // 1400))
    fiber_rgb = np.asarray(Image.alpha_composite(
        Image.fromarray((base * 255).astype(np.uint8)).convert("RGBA"),
        fiber_layer).convert("RGB"), dtype=np.float32) / 255.0
    base = fiber_rgb

    # ---- 细胞核:大量紫蓝色小圆/椭圆,主要特征载体 ----
    n_nuc = max(400, s * s // 3500)
    cx = rng.integers(4, s - 4, n_nuc)
    cy = rng.integers(4, s - 4, n_nuc)
    rr = rng.integers(max(2, s // 1100), max(3, s // 380), n_nuc)
    pix = base
    ys, xs = np.mgrid[0:s, 0:s]
    # 分块画核,避免一次性超大距离矩阵
    block = 200
    for i in range(0, n_nuc, block):
        sl = slice(i, min(i + block, n_nuc))
        bx, by, br = cx[sl], cy[sl], rr[sl]
        # 用 PIL 椭圆近似(向量化距离矩阵对几千个核太慢)
        for x, y, r in zip(bx, by, br):
            ry = max(1, int(r * rng.uniform(0.8, 1.3)))
            col = (
                float(rng.uniform(0.28, 0.45)),
                float(rng.uniform(0.12, 0.25)),
                float(rng.uniform(0.45, 0.66)),
            )
            x0c, x1c = max(0, x - r), min(s, x + r)
            y0c, y1c = max(0, y - ry), min(s, y + ry)
            if x1c <= x0c or y1c <= y0c:
                continue
            yy = ys[y0c:y1c, x0c:x1c] - y
            xx = xs[y0c:y1c, x0c:x1c] - x
            m = (xx / r) ** 2 + (yy / ry) ** 2 <= 1.0
            sub = pix[y0c:y1c, x0c:x1c]
            sub[m] = sub[m] * 0.25 + np.array(col, dtype=np.float32) * 0.75

    # ---- 传感器级细噪声 ----
    base += rng.normal(0.0, 0.012, base.shape).astype(np.float32)
    return np.clip(base, 0.0, 1.0)
