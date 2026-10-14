"""Offline balanced spatial/appearance partition and query-only descriptors."""
import numpy as np
import torch
from scipy.spatial import cKDTree


def validate_partition(indices, queries, n):
    indices, queries = np.asarray(indices), np.asarray(queries)
    if indices.ndim != 2 or indices.size != n or queries.shape != (len(indices),):
        raise ValueError("Patch dimensions do not cover the surface")
    if not np.array_equal(np.sort(indices.reshape(-1)), np.arange(n)):
        raise ValueError("Each surface point must belong to exactly one patch")
    if not (indices == queries[:, None]).any(1).all():
        raise ValueError("Each query must be a member of its owned patch")


def balanced_partition(xyz, rgb, normals, *, size=16, passes=4, neighbors=8,
                       color_weight=.05, normal_weight=.05, component_ids=None,
                       component_weight=0.):
    """PCA median initialization, then quota-preserving neighbor redistribution.

Appearance costs are scaled to the initial median patch variance, so RGB or
normals do not dominate tiny local spatial costs merely due to their units.
Accept only objective-reducing exchanges. This is a local optimum, not a claim
to solve the global capacity-constrained clustering problem exactly.
"""
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    if n % size or n < size or len(np.unique(xyz, axis=0)) != n:
        raise ValueError("Balanced patches require a divisible, distinct surface")
    if not np.isfinite(xyz).all() or min(color_weight, normal_weight, component_weight) < 0:
        raise ValueError("Invalid geometry or feature weights")
    groups = []
    def split(ids):
        count = len(ids) // size
        if count == 1:
            groups.append(ids)
            return
        points = xyz[ids]
        centered = points - points.mean(0)
        _, axes = np.linalg.eigh(centered.T @ centered)
        direction = axes[:, -1]
        # Resolve PCA sign so initialization is deterministic.
        if direction[np.argmax(np.abs(direction))] < 0:
            direction = -direction
        ordered = ids[np.argsort(centered @ direction, kind="stable")]
        cut = (count // 2) * size
        split(ordered[:cut]); split(ordered[cut:])
    split(np.arange(n))
    groups = np.stack(groups)
    initial_variance = ((xyz[groups] - xyz[groups].mean(1)[:, None]) ** 2).sum(-1).mean(1)
    feature_scale2 = max(float(np.median(initial_variance)), 1e-12)
    rgb = np.asarray(rgb, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    if not np.isfinite(rgb).all() or not np.isfinite(normals).all():
        raise ValueError("Nonfinite colour/normal features")
    features = np.concatenate([xyz,
        rgb * np.sqrt(color_weight * feature_scale2),
        normals * np.sqrt(normal_weight * feature_scale2)], axis=1)
    component_ids = None if component_ids is None else np.asarray(component_ids)
    if component_ids is not None and component_weight:
        # Categorical variance: equal IDs share a one-hot channel. Soft, so
        # component counts need not be divisible by the patch size.
        _, inverse = np.unique(component_ids, return_inverse=True)
        if inverse.max() >= 512:
            raise ValueError("Too many components for the optional categorical refinement")
        categorical = np.eye(inverse.max() + 1)[inverse]
        features = np.concatenate([features, categorical * np.sqrt(component_weight * feature_scale2)], 1)
    def energy(g):
        f = features[g]
        return float(((f - f.mean(1)[:, None]) ** 2).sum())
    history = [energy(groups)]
    for _ in range(passes):
        centroids = xyz[groups].mean(1)
        near = cKDTree(centroids).query(centroids, k=min(neighbors + 1, len(groups)))[1]
        if near.ndim == 1:
            break
        pairs = sorted({tuple(sorted((i, int(j)))) for i, row in enumerate(near)
                        for j in row if i != j})
        changed = 0
        for a, b in pairs:
            ids = np.concatenate([groups[a], groups[b]])
            fa, fb = features[groups[a]].mean(0), features[groups[b]].mean(0)
            f = features[ids]
            difference = ((f - fa) ** 2).sum(1) - ((f - fb) ** 2).sum(1)
            order = np.argsort(difference, kind="stable")
            candidate = np.stack([ids[order[:size]], ids[order[size:]]])
            if energy(candidate) < energy(groups[[a, b]]) - 1e-12:
                groups[a], groups[b] = candidate
                changed += 1
        history.append(energy(groups))
        if not changed:
            break
    points = xyz[groups]
    centers = points.mean(1)
    queries = groups[np.arange(len(groups)), ((points - centers[:, None]) ** 2).sum(-1).argmin(1)]
    validate_partition(groups, queries, n)
    radii = np.sqrt(((points - centers[:, None]) ** 2).sum(-1).mean(1))
    spans = np.linalg.norm(points.max(1) - points.min(1), axis=1)
    stats = {"energy": history, "patch_count": len(groups), "size": size,
             "color_weight": color_weight, "normal_weight": normal_weight,
             "feature_scale2": feature_scale2,
             "radius_quantiles": np.quantile(radii, [0, .5, .9, .99, 1]).tolist(),
             "extent_quantiles": np.quantile(spans, [0, .5, .9, .99, 1]).tolist()}
    return groups.astype(np.int64), queries.astype(np.int64), stats


def patch_descriptors(pc, indices, mode="scale_cov"):
    if mode == "none":
        return None
    if mode not in ("scale", "scale_cov"):
        raise ValueError("descriptor mode must be none/scale/scale_cov")
    points = pc.float()[torch.arange(pc.shape[0], device=pc.device)[:, None, None], indices.long()]
    centered = points - points.mean(2, keepdim=True)
    covariance = centered.transpose(-1, -2) @ centered / points.shape[2]
    trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
    scale = .5 * trace.log()
    if mode == "scale":
        return scale[..., None]
    covariance = covariance / trace[..., None, None]
    row = [0, 0, 0, 1, 1, 2]
    col = [0, 1, 2, 1, 2, 2]
    return torch.cat([scale[..., None], covariance[..., row, col]], -1)
