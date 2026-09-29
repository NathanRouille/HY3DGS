"""Umeyama / ICP alignment variants and Chamfer / F-score metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AlignResult:
    scale: float
    R: np.ndarray  # (3, 3) — maps densified/oriented pred → GT frame
    t: np.ndarray  # (3,)
    aligned: np.ndarray  # main export: FULL cloud after transform
    aligned_filtered: Optional[np.ndarray] = None  # mild near-GT crop of full
    aligned_core: Optional[np.ndarray] = None  # densified cloud after transform
    aligned_full: Optional[np.ndarray] = None  # alias of aligned (compat)
    n_filtered: int = 0
    name: str = ""
    flip_name: str = "identity"
    n_denoised: int = 0
    n_raw: int = 0


def umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    estimate_scale: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Similarity: dst ≈ s * (R @ src.T).T + t."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    assert src.shape == dst.shape and src.shape[0] >= 3

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    n = src.shape[0]

    H = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt

    if estimate_scale:
        var_src = (src_c ** 2).sum() / n
        s = float((S * np.diag(D)).sum() / max(var_src, 1e-12))
    else:
        s = 1.0

    t = dst_mean - s * (R @ src_mean)
    return s, R, t


def apply_similarity(
    points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (s * (pts @ R.T)) + t


def _clamp_scale(s: float, lo: float, hi: float) -> float:
    """Unused leftover — kept only for external callers; prefer no clamping."""
    return float(np.clip(s, lo, hi))



def _knn_indices(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    tree = cKDTree(r)
    _, idx = tree.query(q, k=1, workers=-1)
    return np.asarray(idx, dtype=np.int64)


def _knn_dists(query: np.ndarray, ref: np.ndarray, k: int) -> np.ndarray:
    from scipy.spatial import cKDTree

    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    k = min(k, max(1, r.shape[0]))
    tree = cKDTree(r)
    d, _ = tree.query(q, k=k, workers=-1)
    return np.asarray(d, dtype=np.float64)


def _subsample(xyz: np.ndarray, n: int, seed: int) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] <= n:
        return xyz
    rng = np.random.default_rng(seed)
    return xyz[rng.choice(xyz.shape[0], size=n, replace=False)]


def coarse_rms_align(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    estimate_scale: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    rms_src = float(np.sqrt(max((src_c ** 2).sum() / max(src.shape[0], 1), 1e-12)))
    rms_dst = float(np.sqrt(max((dst_c ** 2).sum() / max(dst.shape[0], 1), 1e-12)))
    s = (rms_dst / rms_src) if estimate_scale else 1.0
    R = np.eye(3, dtype=np.float64)
    t = dst_mean - s * src_mean
    return s, R, t


def _cloud_rms(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    c = pts.mean(axis=0)
    return float(np.sqrt(max(((pts - c) ** 2).sum() / max(pts.shape[0], 1), 1e-18)))


def statistical_outlier_removal(
    pts: np.ndarray,
    *,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """Keep points whose mean kNN distance is not an outlier (Open3D-style SOR)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < max(nb_neighbors + 1, 64):
        return pts
    # Exclude self: query k+1 and drop the 0-distance neighbor
    d = _knn_dists(pts, pts, k=nb_neighbors + 1)
    if d.ndim == 1:
        mean_d = d
    else:
        mean_d = d[:, 1:].mean(axis=1)
    mu = float(mean_d.mean())
    sigma = float(mean_d.std())
    keep = mean_d <= (mu + std_ratio * sigma)
    kept = pts[keep]
    if kept.shape[0] < 64:
        return pts
    return kept


def radius_outlier_removal(
    pts: np.ndarray,
    *,
    radius_frac: float = 0.04,
    min_neighbors: int = 16,
) -> np.ndarray:
    """Keep points with ≥ ``min_neighbors`` others inside radius ``radius_frac * RMS``."""
    from scipy.spatial import cKDTree

    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < max(min_neighbors + 1, 64):
        return pts
    r = max(radius_frac * _cloud_rms(pts), 1e-8)
    tree = cKDTree(pts)
    # counts include self
    counts = np.asarray(tree.query_ball_point(pts, r, return_length=True), dtype=np.int64)
    keep = counts >= (min_neighbors + 1)
    kept = pts[keep]
    if kept.shape[0] < 64:
        # Soften once: half min_neighbors
        keep = counts >= (max(4, min_neighbors // 2) + 1)
        kept = pts[keep]
    if kept.shape[0] < 64:
        return pts
    return kept


def keep_largest_dense_cluster(
    pts: np.ndarray,
    *,
    eps_frac: float = 0.05,
    min_samples: int = 8,
) -> np.ndarray:
    """Radius-graph connected components; keep the largest dense component."""
    from scipy.spatial import cKDTree

    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    n = int(pts.shape[0])
    if n < max(min_samples * 2, 64):
        return pts

    eps = max(eps_frac * _cloud_rms(pts), 1e-8)
    tree = cKDTree(pts)
    neighbors = tree.query_ball_point(pts, eps)

    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, nbrs in enumerate(neighbors):
        if len(nbrs) < min_samples:
            continue
        for j in nbrs:
            if j > i:
                union(i, j)

    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    # Ignore tiny components (noise / floaters)
    uniq, counts = np.unique(roots, return_counts=True)
    order = np.argsort(-counts)
    best_root = None
    for u in uniq[order]:
        sz = int(counts[uniq == u][0])
        if sz >= min_samples:
            best_root = int(u)
            break
    if best_root is None:
        return pts
    kept = pts[roots == best_root]
    if kept.shape[0] < 64:
        return pts
    return kept


def density_core_filter(
    pts: np.ndarray,
    *,
    radius_frac: float = 0.04,
    min_neighbors: int = 14,
    cluster_eps_frac: float = 0.055,
    cluster_min_samples: int = 6,
    sor_std_ratio: float = 1.0,
) -> np.ndarray:
    """GT-free denoise: keep existing points only (no new points).

    Radius density prune → largest dense blob → light SOR. If clustering
    removes too much (>50% of radius-pruned points), fall back to radius+SOR.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 64:
        return pts
    after_radius = radius_outlier_removal(
        pts, radius_frac=radius_frac, min_neighbors=min_neighbors
    )
    clustered = keep_largest_dense_cluster(
        after_radius, eps_frac=cluster_eps_frac, min_samples=cluster_min_samples
    )
    if clustered.shape[0] < 0.5 * after_radius.shape[0]:
        core = after_radius
    else:
        core = clustered
    core = statistical_outlier_removal(
        core, nb_neighbors=20, std_ratio=sor_std_ratio
    )
    return core


def apply_linear(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply 3×3 linear map: out = pts @ M.T."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    M = np.asarray(M, dtype=np.float64).reshape(3, 3)
    return pts @ M.T


def _rot_matrix_axis(axis: str, deg: float) -> np.ndarray:
    """Proper rotation matrix for rotation ``deg`` about ``axis``."""
    rad = float(np.deg2rad(deg))
    c, sn = float(np.cos(rad)), float(np.sin(rad))
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -sn], [0.0, sn, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"Unknown axis {axis!r}")


def _flip_matrix(flip_name: str) -> np.ndarray:
    """3×3 reflection / 180° / 90° matrix for discrete orientation candidates."""
    M = np.eye(3, dtype=np.float64)
    if flip_name == "identity":
        return M
    if flip_name == "rot90_x":
        return _rot_matrix_axis("x", 90)
    if flip_name == "rot90_y":
        return _rot_matrix_axis("y", 90)
    if flip_name == "rot90_z":
        return _rot_matrix_axis("z", 90)
    if flip_name == "rot270_z":
        return _rot_matrix_axis("z", 270)
    if flip_name == "rot270_x":
        return _rot_matrix_axis("x", 270)
    if flip_name == "rot270_y":
        return _rot_matrix_axis("y", 270)
    if flip_name == "flip_x":
        M[0, 0] = -1
    elif flip_name == "flip_y":
        M[1, 1] = -1
    elif flip_name == "flip_z":
        M[2, 2] = -1
    elif flip_name == "flip_xy":
        M[0, 0] = M[1, 1] = -1
    elif flip_name == "flip_xz":
        M[0, 0] = M[2, 2] = -1
    elif flip_name == "flip_yz":
        M[1, 1] = M[2, 2] = -1
    elif flip_name == "rot180_x":
        M[1, 1] = M[2, 2] = -1
    elif flip_name == "rot180_y":
        M[0, 0] = M[2, 2] = -1
    elif flip_name == "rot180_z":
        M[0, 0] = M[1, 1] = -1
    else:
        raise ValueError(f"Unknown flip_name {flip_name!r}")
    return M


def apply_axis_flip(pts: np.ndarray, flip_name: str) -> np.ndarray:
    return apply_linear(pts, _flip_matrix(flip_name))


FLIP_CANDIDATES: Tuple[str, ...] = (
    "identity",
    "flip_y",
    "flip_x",
    "flip_z",
    "rot90_z",
    "rot270_z",
    "rot90_x",
    "rot90_y",
    "rot180_y",
    "rot180_x",
    "rot180_z",
    "flip_xy",
    "flip_yz",
)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random SO(3) via QR of Gaussian matrix."""
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    # Fix signs so det(Q)=+1 and R diagonal positive convention
    s = np.sign(np.diag(R))
    s[s == 0] = 1
    Q = Q * s
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float64)


def _pca_basis(pts: np.ndarray) -> np.ndarray:
    """Rows = principal axes (largest variance first)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=True)
    return vt.astype(np.float64)


def _pca_align_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Linear map sending src PCA axes → dst PCA axes (proper rotation)."""
    Vs = _pca_basis(src)
    Vd = _pca_basis(dst)
    # x_dst ≈ Vd.T @ Vd @ x , map: R @ Vs.T ≈ Vd.T  ⇒ R = Vd.T @ Vs
    R = Vd.T @ Vs
    if np.linalg.det(R) < 0:
        Vd = Vd.copy()
        Vd[2] *= -1
        R = Vd.T @ Vs
    return R


def orientation_candidates(
    core: np.ndarray,
    gt: np.ndarray,
    *,
    mode: str,
    seed: int,
    n_random: int = 16,
) -> List[Tuple[str, np.ndarray]]:
    """List of (name, M) pre-alignments applied as pts @ M.T before ICP."""
    out: List[Tuple[str, np.ndarray]] = []
    seen = set()

    def add(name: str, M: np.ndarray) -> None:
        key = tuple(np.round(M.reshape(-1), 5))
        if key in seen:
            return
        seen.add(key)
        out.append((name, M.astype(np.float64)))

    if mode == "identity":
        add("identity", np.eye(3))
        return out

    if mode in ("flip", "flip_y", "rot"):
        names = ("identity", "flip_y") if mode == "flip_y" else FLIP_CANDIDATES
        for n in names:
            add(n, _flip_matrix(n))

    if mode == "rot":
        add("pca", _pca_align_matrix(core, gt))
        # PCA with axis sign flips (octant)
        R0 = _pca_align_matrix(core, gt)
        for i, axis in enumerate("xyz"):
            S = np.eye(3)
            S[i, i] = -1
            if np.linalg.det(S @ R0) < 0:
                continue  # skip improper; use reflection via flip list already
            add(f"pca_flip_{axis}", S @ R0)
        # Also allow improper via reflecting one PCA axis then fixing with flip_y style
        for i, axis in enumerate("xyz"):
            S = np.eye(3)
            S[i, i] = -1
            add(f"pca_ref_{axis}", S @ R0)

        rng = np.random.default_rng(seed)
        for k in range(n_random):
            add(f"rand_{k}", _random_rotation(rng))

    return out


def _icp_loop(
    src0: np.ndarray,
    dst0: np.ndarray,
    *,
    s: float,
    R: np.ndarray,
    t: np.ndarray,
    max_iters: int,
    trim_frac: float,
    estimate_scale: bool,
    seed: int,
    correspondence: str = "gt_to_pred",
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Trimmed ICP. Never clamps scale."""
    rng = np.random.default_rng(seed)
    aligned = apply_similarity(src0, s, R, t)

    for _ in range(max_iters):
        n_use = min(8192, aligned.shape[0], dst0.shape[0])
        if correspondence == "pred_to_gt":
            a_idx = rng.choice(aligned.shape[0], n_use, replace=False)
            d_sub_n = min(max(n_use * 2, n_use), dst0.shape[0])
            d_idx = rng.choice(dst0.shape[0], d_sub_n, replace=False)
            a = aligned[a_idx]
            d = dst0[d_idx]
            nn = _knn_indices(a, d)
            src_pts, dst_pts = a, d[nn]
        elif correspondence == "gt_to_pred":
            g_idx = rng.choice(dst0.shape[0], n_use, replace=False)
            g = dst0[g_idx]
            nn = _knn_indices(g, aligned)
            src_pts, dst_pts = aligned[nn], g
        else:
            raise ValueError(f"Unknown correspondence={correspondence!r}")

        resid = np.linalg.norm(src_pts - dst_pts, axis=1)
        keep_n = max(3, int(trim_frac * len(resid)))
        keep = np.argpartition(resid, keep_n - 1)[:keep_n]
        s_step, R_step, t_step = umeyama(
            src_pts[keep], dst_pts[keep], estimate_scale=estimate_scale
        )
        R_new = R_step @ R
        s_new = float(s_step * s)
        t_new = s_step * (R_step @ t) + t_step

        if not np.isfinite(s_new) or s_new < 1e-8:
            break
        s, R, t = s_new, R_new, t_new
        aligned = apply_similarity(src0, s, R, t)

    return float(s), R, t


def filter_near_gt(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    mode: str = "dist",
    bbox_scale: float = 1.05,
    dist_frac: float = 0.05,
) -> np.ndarray:
    """Keep pred points near the GT object.

    Prefer ``mode="dist"`` (hard shell). ``bbox_or_dist`` is looser (legacy).
    """
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 3)
    if pred.shape[0] == 0:
        return pred

    gmin, gmax = gt.min(axis=0), gt.max(axis=0)
    center = 0.5 * (gmin + gmax)
    half = 0.5 * (gmax - gmin) * bbox_scale
    diag = float(np.linalg.norm(gmax - gmin) + 1e-8)

    if mode == "bbox":
        mask = np.all((pred >= center - half) & (pred <= center + half), axis=1)
    elif mode == "dist":
        nn = _knn_indices(pred, gt)
        d = np.linalg.norm(pred - gt[nn], axis=1)
        mask = d <= (dist_frac * diag)
    elif mode == "bbox_and_dist":
        in_bbox = np.all((pred >= center - half) & (pred <= center + half), axis=1)
        nn = _knn_indices(pred, gt)
        d = np.linalg.norm(pred - gt[nn], axis=1)
        mask = in_bbox & (d <= (dist_frac * diag))
    elif mode == "bbox_or_dist":
        in_bbox = np.all((pred >= center - half) & (pred <= center + half), axis=1)
        nn = _knn_indices(pred, gt)
        d = np.linalg.norm(pred - gt[nn], axis=1)
        mask = in_bbox | (d <= (dist_frac * diag))
    else:
        raise ValueError(f"Unknown filter mode {mode!r}")

    kept = pred[mask]
    if kept.shape[0] < 32:
        nn = _knn_indices(pred, gt)
        d = np.linalg.norm(pred - gt[nn], axis=1)
        k = min(max(32, pred.shape[0] // 10), pred.shape[0])
        keep = np.argpartition(d, min(k, len(d) - 1))[:k]
        kept = pred[keep]
    return kept


def chamfer_l2(
    a: np.ndarray,
    b: np.ndarray,
    *,
    squared: bool = False,
) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")

    def one_way(x, y):
        nn = _knn_indices(x, y)
        d = np.linalg.norm(x - y[nn], axis=1)
        return float((d * d).mean() if squared else d.mean())

    return one_way(a, b) + one_way(b, a)


def one_way_l2(a: np.ndarray, b: np.ndarray, *, squared: bool = False) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    nn = _knn_indices(a, b)
    d = np.linalg.norm(a - b[nn], axis=1)
    return float((d * d).mean() if squared else d.mean())


def fscore(a: np.ndarray, b: np.ndarray, thr: float) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    d_ab = np.linalg.norm(a - b[_knn_indices(a, b)], axis=1)
    d_ba = np.linalg.norm(b - a[_knn_indices(b, a)], axis=1)
    precision = float((d_ab < thr).mean())
    recall = float((d_ba < thr).mean())
    if precision + recall < 1e-12:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _score_alignment(
    aligned: np.ndarray,
    gt: np.ndarray,
    *,
    n_eval: int,
    seed: int,
) -> float:
    """Lower is better: full (untrimmed) Chamfer on subsampled clouds."""
    a = _subsample(aligned, n_eval, seed=seed)
    b = _subsample(gt, n_eval, seed=seed + 1)
    cd = chamfer_l2(a, b, squared=False)
    comp = one_way_l2(b, a, squared=False)
    return float(cd + 0.25 * comp)


def _refine_icp_full_cloud(
    pred: np.ndarray,
    gt: np.ndarray,
    s: float,
    R: np.ndarray,
    t: np.ndarray,
    *,
    n_align: int,
    seed: int,
    trim_frac: float = 0.75,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """GT-anchored ICP refine on the full prediction cloud (subsampled)."""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 3)
    pred_s = _subsample(pred, n_align, seed=seed + 21)
    gt_s = _subsample(gt, n_align, seed=seed + 22)
    return _icp_loop(
        pred_s,
        gt_s,
        s=s,
        R=R,
        t=t,
        max_iters=45,
        trim_frac=trim_frac,
        estimate_scale=True,
        seed=seed + 23,
        correspondence="gt_to_pred",
    )


def _icp_core_to_gt(
    core: np.ndarray,
    gt: np.ndarray,
    *,
    n_align: int,
    seed: int,
    estimate_scale: bool = True,
    fixed_scale: Optional[float] = None,
    trim_frac: float = 0.55,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Coarse + GT→pred trimmed ICP. No scale clamping."""
    core = np.asarray(core, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 3)
    pred_s = _subsample(core, n_align, seed=seed)
    gt_s = _subsample(gt, n_align, seed=seed + 1)

    if estimate_scale:
        s0, R0, t0 = coarse_rms_align(pred_s, gt_s, estimate_scale=True)
    else:
        s0 = float(
            fixed_scale
            if fixed_scale is not None
            else (_cloud_rms(gt_s) / max(_cloud_rms(pred_s), 1e-12))
        )
        R0 = np.eye(3, dtype=np.float64)
        t0 = gt_s.mean(0) - s0 * (R0 @ pred_s.mean(0))

    return _icp_loop(
        pred_s,
        gt_s,
        s=s0,
        R=R0,
        t=t0,
        max_iters=50,
        trim_frac=trim_frac,
        estimate_scale=estimate_scale,
        seed=seed,
        correspondence="gt_to_pred",
    )


def _normalize_both_orient_icp(
    core: np.ndarray,
    gt: np.ndarray,
    *,
    M: np.ndarray,
    n_align: int,
    seed: int,
    trim_frac: float = 0.55,
    polish_scale: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Densified pred oriented by M; normalize both clouds to unit RMS; ICP R/t.

    Independently:
      core' = M @ core
      center both, divide each by its own RMS → unit clouds
      ICP with scale fixed at 1
    Map back to a similarity on the original (pre-M) core frame:
      s = rms_gt / rms_core
      R_tot = R @ M
      t = c_gt - s * R @ c_core

    Optional free-scale polish on densified core (still no clamp).
    """
    core = np.asarray(core, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 3)
    core_o = apply_linear(core, M)

    c_src = core_o.mean(axis=0)
    c_gt = gt.mean(axis=0)
    core_c = core_o - c_src
    gt_c = gt - c_gt
    rms_src = max(_cloud_rms(core_o), 1e-12)
    rms_gt = max(_cloud_rms(gt), 1e-12)
    core_n = core_c / rms_src
    gt_n = gt_c / rms_gt

    # Unit-space ICP: s=1
    s_u, R, t_u = _icp_core_to_gt(
        core_n,
        gt_n,
        n_align=n_align,
        seed=seed,
        estimate_scale=False,
        fixed_scale=1.0,
        trim_frac=trim_frac,
    )
    # t_u should be ~0 if centers matched; keep it for robustness
    # Map unit result back: x_gt ≈ rms_gt * (R @ (x_core_o - c_src)/rms_src + t_u) + c_gt
    #                     = (rms_gt/rms_src) R @ x_core_o + [c_gt + rms_gt*t_u - (rms_gt/rms_src) R @ c_src]
    s = float(rms_gt / rms_src) * float(s_u)
    t = c_gt + rms_gt * t_u - s * (R @ c_src)
    s, R_tot, t = _compose_preorient(s, R, t, M)

    if polish_scale:
        # Free-scale refine on densified core in original frame (no clamp)
        core_s = _subsample(core, n_align, seed=seed + 5)
        gt_s = _subsample(gt, n_align, seed=seed + 6)
        s, R_tot, t = _icp_loop(
            core_s,
            gt_s,
            s=s,
            R=R_tot,
            t=t,
            max_iters=30,
            trim_frac=trim_frac,
            estimate_scale=True,
            seed=seed + 9,
            correspondence="gt_to_pred",
        )

    return float(s), R_tot, t


def _compose_preorient(
    s: float, R: np.ndarray, t: np.ndarray, M: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compose ICP (s,R,t) after linear pre-orient M: x' = M @ x.

    aligned = s * R @ (M @ x) + t = s * (R @ M) @ x + t
    """
    R_tot = R @ M
    return float(s), R_tot, t


# ---------------------------------------------------------------------------
# Named alignment recipes
# ---------------------------------------------------------------------------

ALIGN_METHOD_INFO: Dict[str, str] = {
    "gt_anchored": (
        "Free-scale GT→pred ICP on raw cloud (no densify); strict near-GT filter"
    ),
    "robust_filter": (
        "Densify → normalize both to unit RMS → orient ICP → optional free-scale "
        "polish (no clamp) → apply to FULL cloud"
    ),
    "robust_flip_y": (
        "Densify → normalize-both → {identity, flip_y} → polish → full"
    ),
    "robust_flip_search": (
        "Densify → normalize-both → discrete flips → polish → full"
    ),
    "robust_rot_search": (
        "Densify → normalize-both → flips/90°/PCA/random inits → core ICP → "
        "full-cloud ICP refine; pick best by full CD (primary NOVA recipe)"
    ),
}

DEFAULT_ALIGN_METHODS: Tuple[str, ...] = (
    "gt_anchored",
    "robust_filter",
)

# Post-align export filters (do not drive ICP)
EXPORT_FILTER_STRICT_DIST_FRAC = 0.05
EXPORT_FILTER_LOOSE_DIST_FRAC = 0.11
EXPORT_FILTER_BBOX_SCALE = 1.05
EXPORT_FILTER_LOOSE_BBOX_SCALE = 1.12
EXPORT_FILTER_MODE = "dist"
# NOVA: prefer keeping thin geometry over tight crops
EXPORT_FILTER_NOVA_DIST_FRAC = 0.11
EXPORT_FILTER_NOVA_BBOX_SCALE = 1.12
EXPORT_FILTER_NOVA_MODE = "bbox_or_dist"

# Legacy aliases
EXPORT_FILTER_DIST_FRAC = EXPORT_FILTER_STRICT_DIST_FRAC

# Per-model recipe for pred_aligned/primary/ (thesis default)
PRIMARY_RECIPE: Dict[str, str] = {
    "exp14": "identity",
    "surflo": "gt_anchored",
    "nova3r": "robust_rot_search",
}


def export_filter_kwargs(
    method: str,
    *,
    pred_method: Optional[str] = None,
) -> Dict[str, float | str]:
    if method == "gt_anchored":
        return {
            "mode": EXPORT_FILTER_MODE,
            "bbox_scale": EXPORT_FILTER_BBOX_SCALE,
            "dist_frac": EXPORT_FILTER_STRICT_DIST_FRAC,
        }
    if pred_method == "nova3r":
        return {
            "mode": EXPORT_FILTER_NOVA_MODE,
            "bbox_scale": EXPORT_FILTER_NOVA_BBOX_SCALE,
            "dist_frac": EXPORT_FILTER_NOVA_DIST_FRAC,
        }
    return {
        "mode": EXPORT_FILTER_MODE,
        "bbox_scale": EXPORT_FILTER_LOOSE_BBOX_SCALE,
        "dist_frac": EXPORT_FILTER_LOOSE_DIST_FRAC,
    }


def align_pred_to_gt(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    method: str = "robust_filter",
    pred_method: Optional[str] = None,
    n_align: int = 8192,
    seed: int = 0,
    n_random_rots: int = 16,
    export_dist_frac: Optional[float] = None,
    export_bbox_scale: Optional[float] = None,
    export_filter_mode: Optional[str] = None,
) -> AlignResult:
    """Align pred to GT. Robust path never clamps scale."""
    if method not in ALIGN_METHOD_INFO:
        raise ValueError(
            f"Unknown align method {method!r}. Choose from {list(ALIGN_METHOD_INFO)}"
        )

    fk = export_filter_kwargs(method, pred_method=pred_method)
    if export_dist_frac is None:
        export_dist_frac = float(fk["dist_frac"])  # type: ignore[assignment]
    if export_bbox_scale is None:
        export_bbox_scale = float(fk["bbox_scale"])  # type: ignore[assignment]
    if export_filter_mode is None:
        export_filter_mode = str(fk["mode"])

    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 3)
    n_raw = int(pred.shape[0])

    if method == "gt_anchored":
        s, R, t = _icp_core_to_gt(
            pred,
            gt,
            n_align=n_align,
            seed=seed,
            estimate_scale=True,
            trim_frac=0.7,
        )
        aligned = apply_similarity(pred, s, R, t)
        filt = filter_near_gt(
            aligned,
            gt,
            mode=export_filter_mode,
            bbox_scale=export_bbox_scale,
            dist_frac=export_dist_frac,
        )
        return AlignResult(
            scale=float(s),
            R=R,
            t=t,
            aligned=aligned,
            aligned_filtered=filt,
            aligned_core=None,
            aligned_full=aligned,
            n_filtered=int(filt.shape[0]),
            name=method,
            flip_name="identity",
            n_denoised=n_raw,
            n_raw=n_raw,
        )

    core = density_core_filter(pred)
    n_den = int(core.shape[0])

    if method == "robust_filter":
        orient_mode = "identity"
    elif method == "robust_flip_y":
        orient_mode = "flip_y"
    elif method == "robust_flip_search":
        orient_mode = "flip"
    elif method == "robust_rot_search":
        orient_mode = "rot"
    else:
        raise ValueError(method)

    cands = orientation_candidates(
        core, gt, mode=orient_mode, seed=seed + 7, n_random=n_random_rots
    )

    best: Optional[AlignResult] = None
    best_score = float("inf")

    for oname, M in cands:
        s, R_tot, t = _normalize_both_orient_icp(
            core,
            gt,
            M=M,
            n_align=n_align,
            seed=seed,
            trim_frac=0.55,
            polish_scale=True,
        )
        # Refine pose on full cloud (not just densified core)
        s, R_tot, t = _refine_icp_full_cloud(
            pred,
            gt,
            s,
            R_tot,
            t,
            n_align=n_align,
            seed=seed,
        )
        aligned_full = apply_similarity(pred, s, R_tot, t)
        aligned_core = apply_similarity(core, s, R_tot, t)
        # Pick best init using full-cloud CD (not trimmed core-only score)
        score = _score_alignment(aligned_full, gt, n_eval=n_align, seed=seed + 11)
        if score >= best_score:
            continue
        # Core export = densified cloud after pose (no GT shell trim — avoids amputation)
        filt = filter_near_gt(
            aligned_full,
            gt,
            mode=export_filter_mode,
            bbox_scale=export_bbox_scale,
            dist_frac=export_dist_frac,
        )
        best_score = score
        best = AlignResult(
            scale=float(s),
            R=R_tot,
            t=t,
            aligned=aligned_full,
            aligned_filtered=filt,
            aligned_core=aligned_core,
            aligned_full=aligned_full,
            n_filtered=int(filt.shape[0]),
            name=method,
            flip_name=oname,
            n_denoised=n_den,
            n_raw=n_raw,
        )

    assert best is not None
    return best
