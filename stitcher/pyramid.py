"""把拼接大图切成多分辨率金字塔瓦片(Deep Zoom 思想的自研轻量格式)。

目录结构:
  pyramid/
    manifest.json          层级尺寸、瓦片大小、文件命名规则
    tiles/level_k/r_c.jpg  第 k 层、瓦片行 r、列 c
层级 0 为全分辨率,每升一级长宽各减半(LANCZOS 抗锯齿)。
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from .io_utils import save_json


def _save_jpeg(path: str, arr: np.ndarray, quality: int) -> None:
    Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8),
                    "RGB").save(path, "JPEG", quality=quality)


def build_pyramid(canvas: np.ndarray, out_dir: str, tile_size: int = 256,
                  jpeg_quality: int = 85) -> dict:
    tiles_dir = os.path.join(out_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    h, w = canvas.shape[:2]

    # 顶层缩到与单瓦片相当,保证整图可一屏总览
    n_levels = int(np.ceil(np.log2(max(h, w) / tile_size))) + 1
    levels = []
    pil = Image.fromarray((canvas * 255 + 0.5).astype(np.uint8), "RGB")
    for level in range(n_levels):
        scale = 2 ** level
        lw, lh = (w + scale - 1) // scale, (h + scale - 1) // scale
        if level == 0:
            img = pil
        else:
            img = pil.resize((lw, lh), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.uint8)

        ldir = os.path.join(tiles_dir, f"level_{level}")
        os.makedirs(ldir, exist_ok=True)
        rows = int(np.ceil(lh / tile_size))
        cols = int(np.ceil(lw / tile_size))
        for r in range(rows):
            for c in range(cols):
                y0, x0 = r * tile_size, c * tile_size
                block = arr[y0:y0 + tile_size, x0:x0 + tile_size]
                if block.max() == 0 and block.min() == 0:
                    continue                       # 全黑背景瓦片不落盘
                _save_jpeg(os.path.join(ldir, f"{r}_{c}.jpg"),
                           block.astype(np.float32) / 255.0, jpeg_quality)
        levels.append({"level": level, "width": lw, "height": lh,
                       "cols": cols, "rows": rows})

    manifest = {
        "tile_size": tile_size,
        "format": "jpg",
        "quality": jpeg_quality,
        "full_width": w,
        "full_height": h,
        "levels": levels,
    }
    save_json(os.path.join(out_dir, "manifest.json"), manifest)
    return manifest
