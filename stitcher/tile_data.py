"""从大图切模拟显微镜视野:网格 + 位置抖动 + 亮度不均 + 随机缺失。

输出每个视野的 PNG 与一份 manifest(记录名义网格位姿、真实抖动位姿、
增益、缺失情况),真实位姿只用于最后打分,拼接流程本身不得使用。
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from .io_utils import save_json


def _vignette(shape: tuple[int, int], strength: float,
              rng: np.random.Generator) -> np.ndarray:
    """径向暗角乘子,中心 1、边缘 1-strength,光心带轻微偏移。"""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ox = rng.uniform(-0.08, 0.08) * w
    oy = rng.uniform(-0.08, 0.08) * h
    d2 = ((xx - w / 2 - ox) / (w / 2)) ** 2 + \
         ((yy - h / 2 - oy) / (h / 2)) ** 2
    return (1.0 - strength * np.clip(d2, 0.0, 1.4) / 1.4).astype(np.float32)


def cut_tiles(source: np.ndarray, out_dir: str, cols: int, rows: int,
              tile: int = 512, overlap_frac: float = 0.2,
              jitter_px: float = 24.0, missing_frac: float = 0.04,
              vignette_strength: float = 0.14, gain_sigma: float = 0.09,
              noise_sigma: float = 0.012, seed: int = 7,
              edge_crop: bool = True) -> dict:
    """把 source 切成 cols×rows 个带缺陷的视野。

    参数
    ----
    overlap_frac: 相邻视野名义重叠比例(相对 tile 边长)。
    jitter_px:    每个视野真实位置相对名义位置的高斯抖动标准差(像素)。
    missing_frac: 随机缺失视野比例。
    vignette_strength / gain_sigma / noise_sigma:
                  暗角强度、视野间曝光增益标准差、传感器噪声标准差。
    edge_crop:    True 时只保留完整落在图内的视野(边缘无黑边)。
    """
    rng = np.random.default_rng(seed)
    h, w = source.shape[:2]
    step = int(round(tile * (1.0 - overlap_frac)))

    # 名义步长网格,整体居中
    total_w = step * (cols - 1) + tile
    total_h = step * (rows - 1) + tile
    x0 = (w - total_w) // 2
    y0 = (h - total_h) // 2

    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for r in range(rows):
        for c in range(cols):
            nx = x0 + c * step
            ny = y0 + r * step
            dx = float(rng.normal(0.0, jitter_px))
            dy = float(rng.normal(0.0, jitter_px))
            tx, ty = nx + dx, ny + dy
            ix, iy = int(round(tx)), int(round(ty))

            entry = {
                "row": r, "col": c,
                "nominal_x": nx, "nominal_y": ny,
                "true_x": tx, "true_y": ty,
                "gain": [1.0, 1.0, 1.0],
                "missing": False,
                "file": None,
            }

            complete = (0 <= ix and iy >= 0 and ix + tile <= w
                        and iy + tile <= h)
            drop = rng.random() < missing_frac
            if drop or (edge_crop and not complete):
                entry["missing"] = True
                entries.append(entry)
                continue

            # 边缘不完整视野:只取有效部分,其余按缺失处理
            x1, y1 = min(ix + tile, w), min(iy + tile, h)
            patch = source[max(iy, 0):y1, max(ix, 0):x1].copy()
            if patch.shape[0] < tile or patch.shape[1] < tile:
                entry["missing"] = True
                entries.append(entry)
                continue

            # 曝光:每通道独立乘性增益 + 轻微偏置 + 径向暗角
            gain = rng.lognormal(0.0, gain_sigma, size=3).astype(np.float32)
            bias = rng.uniform(-0.02, 0.02, size=3).astype(np.float32)
            v = _vignette((tile, tile), vignette_strength, rng)[..., None]
            patch = np.clip(patch * gain[None, None, :] * v
                            + bias[None, None, :], 0.0, 1.0)
            patch += rng.normal(0.0, noise_sigma, patch.shape).astype(np.float32)
            patch = np.clip(patch, 0.0, 1.0)

            fname = f"tile_r{r:03d}_c{c:03d}.png"
            Image.fromarray((patch * 255 + 0.5).astype(np.uint8), "RGB"
                            ).save(os.path.join(out_dir, fname))
            entry["gain"] = [float(g) for g in gain]
            entry["file"] = fname
            entries.append(entry)

    manifest = {
        "tile_size": tile,
        "step": step,
        "cols": cols,
        "rows": rows,
        "overlap_frac": overlap_frac,
        "jitter_px": jitter_px,
        "missing_frac": missing_frac,
        "source_shape": [h, w],
        "tiles": entries,
    }
    save_json(os.path.join(out_dir, "manifest.json"), manifest)
    return manifest
