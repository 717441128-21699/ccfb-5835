"""图像读写与通用小工具。"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image


def load_rgb(path: str) -> np.ndarray:
    """读成 float32 RGB,值域 0~1。"""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def save_rgb(path: str, img: np.ndarray) -> None:
    """把 0~1 float 图像存成 8bit PNG/JPEG(按后缀决定格式)。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = np.clip(img, 0.0, 1.0)
    arr8 = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr8, "RGB").save(path)


def save_gray(path: str, img: np.ndarray) -> None:
    """存单通道浮点图(自动拉伸到 0~255),用于误差热力图。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lo, hi = float(np.min(img)), float(np.max(img))
    if hi - lo < 1e-12:
        arr8 = np.zeros(img.shape, dtype=np.uint8)
    else:
        arr8 = ((img - lo) / (hi - lo) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr8, "L").save(path)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def to_gray(img: np.ndarray) -> np.ndarray:
    """RGB float -> 灰度 float。"""
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
