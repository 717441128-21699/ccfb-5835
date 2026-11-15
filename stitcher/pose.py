"""全局位姿优化:把所有相邻视野的相对平移放进一个稀疏最小二乘,
一次求解全部视野原点坐标,消除逐对拼接的累积误差(漂移)。

同时给出 BFS 链式累积的位姿作为对照,量化全局优化的收益。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from .features import extract_features
from .matching import MatchResult, match_pair


@dataclass
class Tile:
    idx: int
    row: int
    col: int
    nominal: np.ndarray          # (y, x) 名义原点
    image: np.ndarray | None = None
    features: object = None
    missing: bool = False


@dataclass
class Edge:
    a: int
    b: int
    t: np.ndarray                # origin_b - origin_a
    weight: float
    method: str
    inliers: int
    residual_rms: float
    confidence: float
    axis: str                    # "h"(b 在 a 右侧) / "v"(b 在 a 下方)


@dataclass
class PoseGraph:
    tiles: list[Tile]
    edges: list[Edge] = field(default_factory=list)
    step: tuple[int, int] = (0, 0)

    def tile_at(self, row: int, col: int) -> Tile | None:
        for t in self.tiles:
            if t.row == row and t.col == col:
                return t
        return None


def build_graph(manifest: dict, images: list[np.ndarray]) -> PoseGraph:
    """按 manifest 构建位姿图(不做匹配,只放名义位姿与图像)。"""
    step = manifest["step"]
    tiles = []
    for e, img in zip(manifest["tiles"], images):
        tiles.append(Tile(
            idx=len(tiles), row=e["row"], col=e["col"],
            nominal=np.array([e["nominal_y"], e["nominal_x"]], np.float32),
            image=None if e["missing"] else img,
            missing=e["missing"],
        ))
    return PoseGraph(tiles=tiles, step=(step, step))


def run_pairwise_matching(graph: PoseGraph, min_inliers: int = 10,
                          max_nominal_dev: float = 60.0) -> list[Edge]:
    """对网格 4 邻域逐对做特征匹配(右侧、下侧两个方向即可)。"""
    index = {(t.row, t.col): t for t in graph.tiles}
    step = graph.step[0]
    edges: list[Edge] = []
    for t in graph.tiles:
        if t.missing:
            continue
        if t.features is None:
            t.features = extract_features(t.image)
        for dr, dc, axis, nom in ((0, 1, "h", (0, step)),
                                  (1, 0, "v", (step, 0))):
            u = index.get((t.row + dr, t.col + dc))
            if u is None or u.missing:
                continue
            if u.features is None:
                u.features = extract_features(u.image)
            res: MatchResult = match_pair(
                t.image, u.image, t.features, u.features,
                np.array(nom, np.float32),
                min_inliers=min_inliers,
                max_nominal_dev=max_nominal_dev)
            if res.method == "failed":
                continue
            edges.append(Edge(
                a=t.idx, b=u.idx, t=res.t,
                weight=float(np.sqrt(max(res.inliers, 1))),
                method=res.method, inliers=res.inliers,
                residual_rms=res.residual_rms,
                confidence=res.confidence, axis=axis,
            ))
    graph.edges = edges
    return edges


def _build_system(graph: PoseGraph, edge_scales: np.ndarray,
                  anchor_idx: int, prior_weight: float
                  ) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """按给定每条边的权重尺度构造加权最小二乘系统 (A, by, bx)。"""
    n = len(graph.tiles)
    triples: list[tuple[int, int, float]] = []
    rhs_y, rhs_x = [], []
    row_id = 0

    def add_row(r, entries, ry, rx):
        for c, v in entries:
            triples.append((r, c, v))
        rhs_y.append(ry); rhs_x.append(rx)

    for k, e in enumerate(graph.edges):
        w = e.weight * edge_scales[k]
        add_row(row_id, [(e.a, -w), (e.b, w)],
                w * e.t[0], w * e.t[1])
        row_id += 1

    for t in graph.tiles:
        if t.idx == anchor_idx and not t.missing:
            add_row(row_id, [(t.idx, 50.0)],
                    50.0 * t.nominal[0], 50.0 * t.nominal[1])
        else:
            w = prior_weight if not t.missing else prior_weight * 2.0
            add_row(row_id, [(t.idx, w)],
                    w * t.nominal[0], w * t.nominal[1])
        row_id += 1

    rows = [tr[0] for tr in triples]
    cols = [tr[1] for tr in triples]
    data = [tr[2] for tr in triples]
    A = csr_matrix((data, (rows, cols)), shape=(row_id, n), dtype=np.float64)
    return A, np.array(rhs_y), np.array(rhs_x)


def solve_global_poses(graph: PoseGraph, anchor_idx: int = 0,
                       prior_weight: float = 0.15,
                       huber_delta: float = 2.5,
                       irls_iter: int = 6) -> np.ndarray:
    """Huber 鲁棒的稀疏加权最小二乘,求所有视野原点 (N,2)。

    观测方程 w·(p_b − p_a) = w·t_ab,边权 ∝ √内点数;再叠:
      - 锚点约束(固定全局坐标系)
      - 弱名义先验(缺失/断开的图仍可解,且不压过真实匹配)
    IRLS:用当前残差对每条边套 Huber 权,迭代压低粗差边的影响,
    避免极端抖动视野产生的错误匹配污染整体位姿。
    """
    n_edge = len(graph.edges)
    scales = np.ones(n_edge, dtype=np.float64)
    poses = None
    for _ in range(irls_iter):
        A, by, bx = _build_system(graph, scales, anchor_idx, prior_weight)
        py = lsqr(A, by, iter_lim=2000, atol=1e-9, btol=1e-9)[0]
        px = lsqr(A, bx, iter_lim=2000, atol=1e-9, btol=1e-9)[0]
        poses = np.stack([py, px], axis=1)
        if n_edge == 0:
            break
        res = np.array([poses[e.b] - poses[e.a] - e.t
                        for e in graph.edges])
        r = np.linalg.norm(res, axis=1)
        sigma = max(1e-3, float(np.median(r)))     # 中位数鲁棒尺度
        z = r / (huber_delta * sigma)
        new_scales = 1.0 / np.maximum(1.0, z)      # Huber 权
        if np.max(np.abs(new_scales - scales)) < 1e-3:
            break
        scales = new_scales
    return poses.astype(np.float32)


def cumulative_poses(graph: PoseGraph, start_idx: int = 0) -> np.ndarray:
    """BFS 链式累积位姿:只用沿生成树的相对平移,展示不做全局优化时的漂移。"""
    n = len(graph.tiles)
    poses = np.full((n, 2), np.nan, dtype=np.float32)
    start = graph.tiles[start_idx]
    poses[start_idx] = start.nominal
    adj: dict[int, list[tuple[int, np.ndarray]]] = {i: [] for i in range(n)}
    for e in graph.edges:
        adj[e.a].append((e.b, e.t))
        adj[e.b].append((e.a, -e.t))
    queue = [start_idx]
    seen = {start_idx}
    head = 0
    while head < len(queue):
        cur = queue[head]; head += 1
        for nxt, t in adj[cur]:
            if nxt in seen:
                continue
            seen.add(nxt)
            poses[nxt] = poses[cur] + t
            queue.append(nxt)
    # 未连通的视野回退到名义位姿
    for i in range(n):
        if np.isnan(poses[i]).any():
            poses[i] = graph.tiles[i].nominal
    return poses
