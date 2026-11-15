#!/usr/bin/env python3
"""端到端 demo:合成大图 -> 切缺陷视野 -> 匹配 -> 全局优化 ->
曝光补偿 -> 多频段融合 -> 金字塔瓦片 -> 评估出报告。

用法:
  python scripts/run_demo.py --workdir out_demo --cols 10 --rows 10
所有关键参数都在 config 里,调参指南见 PARAMETERS.md。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stitcher.io_utils import load_rgb, save_json, save_rgb  # noqa: E402
from stitcher.synthetic import make_synthetic_tissue  # noqa: E402
from stitcher.tile_data import cut_tiles  # noqa: E402
from stitcher.pose import (build_graph, cumulative_poses,  # noqa: E402
                           run_pairwise_matching, solve_global_poses)
from stitcher.exposure import apply_gains, estimate_gains  # noqa: E402
from stitcher.flatfield import flatten_graph  # noqa: E402
from stitcher.blending import composite  # noqa: E402
from stitcher.pyramid import build_pyramid  # noqa: E402
from stitcher.evaluation import (image_errors, pose_rmse,  # noqa: E402
                                 seam_discontinuity)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="out_demo")
    ap.add_argument("--source-size", type=int, default=4096)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--jitter", type=float, default=18.0)
    ap.add_argument("--missing", type=float, default=0.04)
    ap.add_argument("--gain-sigma", type=float, default=0.09)
    ap.add_argument("--vignette", type=float, default=0.14)
    ap.add_argument("--bands", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-pyramid", action="store_true")
    args = ap.parse_args()

    wd = os.path.abspath(args.workdir)
    tile_dir = os.path.join(wd, "tiles")
    pyr_dir = os.path.join(wd, "pyramid")
    os.makedirs(wd, exist_ok=True)

    # 1) 源图(若已存在则复用,方便迭代调参)
    src_path = os.path.join(wd, "source.png")
    if os.path.exists(src_path):
        log("复用已有 source.png")
        source = load_rgb(src_path)
    else:
        log(f"生成 {args.source_size}px 合成组织图")
        source = make_synthetic_tissue(args.source_size)
        save_rgb(src_path, source)

    # 2) 切视野
    log(f"切 {args.cols}x{args.rows} 视野:抖动 {args.jitter}px、"
        f"缺失 {args.missing:.0%}、暗角 {args.vignette}")
    manifest = cut_tiles(
        source, tile_dir, args.cols, args.rows, tile=args.tile,
        overlap_frac=args.overlap, jitter_px=args.jitter,
        missing_frac=args.missing, vignette_strength=args.vignette,
        gain_sigma=args.gain_sigma, seed=args.seed)
    n_missing = sum(e["missing"] for e in manifest["tiles"])
    log(f"实际缺失/不完整视野 {n_missing} 个")

    images = [None if e["missing"] else load_rgb(os.path.join(tile_dir, e["file"]))
              for e in manifest["tiles"]]

    # 3) 位姿图 + 两两匹配
    graph = build_graph(manifest, images)
    log("相邻视野特征匹配(Harris + NCC,失败回退相位相关)")
    t0 = time.time()
    edges = run_pairwise_matching(graph)
    n_ncc = sum(e.method == "ncc_search" for e in edges)
    n_harris = sum(e.method == "harris" for e in edges)
    log(f"匹配 {len(edges)} 条边(Harris {n_harris}、局部 NCC 兜底 {n_ncc}),"
        f" 耗时 {time.time() - t0:.1f}s")

    # 4) 全局位姿 vs 链式累积
    poses = solve_global_poses(graph)
    poses_chain = cumulative_poses(graph)

    truth = {i: (e["true_y"], e["true_x"])
             for i, e in enumerate(manifest["tiles"]) if not e["missing"]}
    err_global = pose_rmse(graph, poses, truth)
    err_chain = pose_rmse(graph, poses_chain, truth)
    log(f"位姿 RMSE  全局优化={err_global['rmse_px']:.3f}px  "
        f"链式累积={err_chain['rmse_px']:.3f}px")

    # 5) 曝光补偿 + 瓦片内暗角平坦化
    log("估计并应用曝光增益,平坦化暗角")
    gains = estimate_gains(graph)
    apply_gains(graph, gains)
    flatten_graph(graph)

    # 6) 三种融合方式
    results = {}
    for mode in ("hard", "feather", "multiband"):
        log(f"合成 ({mode})")
        canvas, info = composite(graph, poses, mode=mode, levels=args.bands)
        out = os.path.join(wd, f"mosaic_{mode}.png")
        save_rgb(out, canvas)
        shifted = poses - np.array([info["offset_y"], info["offset_x"]],
                                   dtype=np.float32)
        ie = image_errors(canvas, source,
                          out_prefix=os.path.join(wd, f"eval_{mode}"))
        se = seam_discontinuity(canvas, graph, shifted)
        results[mode] = {"image": os.path.basename(out),
                         "image_error": ie, "seam": se}
        log(f"  {mode:9s} RMSE={ie['rmse']:.4f}  PSNR={ie['psnr_db']:.1f}dB"
            f"  接缝 MAD={se.get('seam_mad')}")

    # 7) 金字塔 + 查看器数据
    if not args.skip_pyramid:
        log("切金字塔瓦片")
        canvas, info = composite(graph, poses, mode="multiband",
                                 levels=args.bands)
        pm = build_pyramid(canvas, pyr_dir)
        viewer_src = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), "viewer",
            "index.html")
        if os.path.exists(viewer_src):
            import shutil
            shutil.copyfile(viewer_src, os.path.join(pyr_dir, "index.html"))
        log(f"金字塔 {len(pm['levels'])} 层,顶层 "
            f"{pm['levels'][-1]['width']}x{pm['levels'][-1]['height']};"
            f" 查看器: python scripts/serve.py {pyr_dir}")

    report = {
        "config": vars(args),
        "n_tiles": len(manifest["tiles"]),
        "n_missing": n_missing,
        "n_edges": len(edges),
        "n_ncc_fallback": n_ncc,
        "pose_error_global": err_global,
        "pose_error_chain": err_chain,
        "mosaics": results,
        "gain_stats": {
            "min": float(gains.min()), "max": float(gains.max()),
            "std": float(gains.std())},
    }
    save_json(os.path.join(wd, "report.json"), report)
    log(f"报告写入 {os.path.join(wd, 'report.json')}")


if __name__ == "__main__":
    main()
