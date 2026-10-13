"""CPU reconstruction diagnostics; distances and cells use normalized units.

Cell and spacing summaries describe coverage/density, not semantic object scores.
Always compare candidates against the same GT and thresholds.
"""
import numpy as np
from scipy.spatial import cKDTree


def reconstruction_metrics(pred, target, rgb=None, target_rgb=None,
                           thresholds=(.002, .005, .01, .02, .05), cell_size=.1):
    p, q = np.asarray(pred), np.asarray(target)
    if p.ndim != 2 or q.ndim != 2 or p.shape[1] != 3 or q.shape[1] != 3 or not len(p) or not len(q):
        raise ValueError("Expected nonempty Nx3 clouds")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Nonfinite point coordinates")
    if cell_size <= 0 or not thresholds or any(t <= 0 for t in thresholds):
        raise ValueError("Thresholds and cell size must be positive")
    dp, ip = cKDTree(q).query(p)
    dq, iq = cKDTree(p).query(q)
    result = {"cd_pred2gt": float(np.mean(dp**2)), "cd_gt2pred": float(np.mean(dq**2)),
              "cd": float(np.mean(dp**2) + np.mean(dq**2))}
    for quantile in (.5, .95, .99):
        result[f"pred_distance_p{int(100*quantile)}"] = float(np.quantile(dp, quantile))
        result[f"gt_distance_p{int(100*quantile)}"] = float(np.quantile(dq, quantile))
    for t in thresholds:
        precision, recall = float(np.mean(dp < t)), float(np.mean(dq < t))
        result[f"precision_norm_{t}"] = precision
        result[f"coverage_norm_{t}"] = recall
        result[f"fscore_norm_{t}"] = 2 * precision * recall / max(precision + recall, 1e-12)
    if rgb is not None and target_rgb is not None:
        residual = np.abs(np.asarray(rgb) - np.asarray(target_rgb)[ip]).mean(axis=1)
        result["rgb_l1"] = float(.5 * (residual.mean() + np.abs(np.asarray(target_rgb) - np.asarray(rgb)[iq]).mean()))
        result["pred_rgb_error_p95"] = float(np.quantile(residual, .95))

    # Deduplicate GT before density comparisons; sharp sampling may repeat points.
    unique = np.unique(q, axis=0)
    if len(p) > 1 and len(unique) > 1:
        gt_spacing = cKDTree(unique).query(unique, k=2)[0][:, 1]
        pred_spacing = cKDTree(p).query(p, k=2)[0][:, 1]
        reference_spacing = float(np.median(gt_spacing))
        result["gt_unique_nn_median"] = reference_spacing
        result["pred_nn_p10"] = float(np.quantile(pred_spacing, .1))
        result["pred_nn_median"] = float(np.median(pred_spacing))
        result["pred_nn_p90"] = float(np.quantile(pred_spacing, .9))
        result["pred_nn_cv"] = float(pred_spacing.std() / max(pred_spacing.mean(), 1e-12))
        result["pred_near_duplicate_fraction"] = float(np.mean(pred_spacing < .2 * reference_spacing))

    # GT-defined cells are shared across candidates. Sparse cells still count
    # for occupied-cell coverage; macro/tail recall excludes cells with <4 samples.
    cells, inverse, counts = np.unique(np.floor(q / cell_size).astype(np.int64),
                                      axis=0, return_inverse=True, return_counts=True)
    supported = counts >= 4
    result["gt_occupied_cells"] = int(len(cells))
    result["gt_macro_cells"] = int(supported.sum())
    for t in thresholds:
        hits = np.bincount(inverse, weights=(dq < t).astype(float), minlength=len(cells))
        recall = hits / counts
        result[f"gt_cell_coverage_norm_{t}"] = float(np.mean(hits > 0))
        if supported.any():
            values = recall[supported]
            result[f"gt_cell_macro_recall_norm_{t}"] = float(values.mean())
            result[f"gt_cell_worst_decile_recall_norm_{t}"] = float(np.sort(values)[:max(1, int(np.ceil(.1*len(values))))].mean())
    return result, dp, dq
