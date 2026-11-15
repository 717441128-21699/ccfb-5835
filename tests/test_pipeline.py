#!/usr/bin/env python3
"""关键算法自测:不依赖完整 demo,秒级验证各模块正确性。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stitcher.features import extract_features, match_descriptors
from stitcher.matching import match_pair, ncc_refine, phase_correlation
from stitcher.synthetic import make_synthetic_tissue
from stitcher.exposure import estimate_gains
from stitcher.pose import build_graph, solve_global_poses, cumulative_poses


def test_feature_repeatability():
    """已知平移下,Harris 特征应能配对并恢复位移(原点偏移约定)。"""
    img = make_synthetic_tissue(1024, seed=3)
    s = np.array([37.3, -21.7])
    # shifted = shift(img, s):内容移动 s,等价于原点偏移 t = -s
    from scipy import ndimage as ndi
    shifted = np.empty_like(img)
    for ch in range(3):
        shifted[..., ch] = ndi.shift(img[..., ch], s, order=1)
    fa = extract_features(img)
    fb = extract_features(shifted)
    assert len(fa.points) > 50 and len(fb.points) > 50
    pairs = match_descriptors(fa, fb)
    assert len(pairs) > 20, f"配对太少: {len(pairs)}"
    res = match_pair(img, shifted, fa, fb, -s.astype(np.float32))
    err = np.linalg.norm(res.t + s)
    print(f"[特征匹配] 期望原点偏移={-s} 估计=({res.t[0]:.2f},{res.t[1]:.2f}) "
          f"误差={err:.2f}px 方法={res.method} 内点={res.inliers}")
    assert err < 1.5, err


def test_phase_correlation():
    img = make_synthetic_tissue(768, seed=5)
    from scipy import ndimage as ndi
    s = np.array([12.0, -8.0])
    shifted = ndi.shift(img.mean(axis=2), s, order=1)
    gray = img.mean(axis=2)
    shift, peak = phase_correlation(gray, shifted, -s.astype(np.float32))
    est = -s + shift
    err = np.linalg.norm(est + s)
    print(f"[相位相关] 修正={shift.round(2)} 峰值占比={peak:.3f} 误差={err:.2f}px")
    assert err < 1.2


def test_ncc_refine():
    img = make_synthetic_tissue(512, seed=9).mean(axis=2)
    from scipy import ndimage as ndi
    s = np.array([23.6, -15.4])
    shifted = ndi.shift(img, s, order=1)
    true_t = -s                                  # 原点偏移
    t0 = np.round(true_t)
    refined = ncc_refine(img, shifted, t0)
    err = np.linalg.norm(refined - true_t)
    print(f"[NCC 细化] 整数起点={t0} 细化后={refined.round(2)} 误差={err:.2f}px")
    assert err < 1.0


def test_global_vs_chain():
    """构造带噪声的网格平移链:全局优化的误差应显著小于链式累积。"""
    rng = np.random.default_rng(0)
    cols, rows, step = 6, 6, 400
    tiles_meta, images = [], []
    base = make_synthetic_tissue(512, seed=1)
    for r in range(rows):
        for c in range(cols):
            tiles_meta.append({
                "row": r, "col": c,
                "nominal_y": r * step, "nominal_x": c * step,
                "true_y": r * step, "true_x": c * step,
                "missing": False, "file": "x"})
            images.append(base)
    manifest = {"step": step, "tiles": tiles_meta}
    graph = build_graph(manifest, images)
    # 手工塞带噪声的边(模拟匹配误差),真值平移恒为 step
    from stitcher.pose import Edge
    idx = {(t.row, t.col): t.idx for t in graph.tiles}
    for r in range(rows):
        for c in range(cols):
            for dr, dc, axis in ((0, 1, "h"), (1, 0, "v")):
                if (r + dr, c + dc) not in idx:
                    continue
                noise = rng.normal(0, 2.0, 2)
                t = np.array([step * dr, step * dc]) + noise
                graph.edges.append(Edge(idx[(r, c)], idx[(r + dr, c + dc)],
                                        t.astype(np.float32), 3.0,
                                        "test", 30, 1.0, 0.9, axis))
    p_global = solve_global_poses(graph)
    p_chain = cumulative_poses(graph)
    truth = np.array([[r * step, c * step]
                      for r in range(rows) for c in range(cols)])
    e_g = np.sqrt(((p_global - truth - (p_global - truth).mean(0)) ** 2)
                  .sum(1).mean())
    e_c = np.sqrt(((p_chain - truth - (p_chain - truth).mean(0)) ** 2)
                  .sum(1).mean())
    print(f"[全局优化] RMSE={e_g:.2f}px  [链式累积] RMSE={e_c:.2f}px")
    assert e_g < e_c


def test_gain_compensation():
    """两张重叠图,第二张整体偏暗 20%,增益应恢复到 ~1.25。"""
    img = make_synthetic_tissue(512, seed=2)
    dark = np.clip(img * 0.8, 0, 1).astype(np.float32)
    manifest = {"step": 400, "tiles": [
        {"row": 0, "col": 0, "nominal_y": 0, "nominal_x": 0,
         "true_y": 0, "true_x": 0, "missing": False, "file": "a"},
        {"row": 0, "col": 1, "nominal_y": 0, "nominal_x": 400,
         "true_y": 0, "true_x": 400, "missing": False, "file": "b"}]}
    graph = build_graph(manifest, [img, dark])
    from stitcher.pose import Edge
    graph.edges.append(Edge(0, 1, np.array([0.0, 400.0]), 5.0,
                            "test", 30, 0.5, 0.9, "h"))
    gains = estimate_gains(graph)
    ratio = gains[1] / gains[0]
    print(f"[曝光补偿] g0={gains[0].round(3)} g1={gains[1].round(3)} "
          f"比值={ratio.round(3)} (期望 ~1.25)")
    assert np.all(np.abs(ratio - 1.25) < 0.12)


if __name__ == "__main__":
    test_feature_repeatability()
    test_phase_correlation()
    test_ncc_refine()
    test_global_vs_chain()
    test_gain_compensation()
    print("\n全部自测通过 ✓")
