"""Primary-pair diagnostics against full owned targets, independent of FPS."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def owned_patch_metrics(pred, anchors, gt, indices, query_indices):
    p, a, g = [x.detach().float().cpu().numpy() for x in (pred, anchors, gt)]
    indices = indices.cpu().numpy()
    query_indices = query_indices.cpu().numpy()
    queries = g[query_indices]
    cost = ((a[:, None] - queries[None]) ** 2).sum(-1)
    mapping = linear_sum_assignment(cost)[1]
    p = p.reshape(len(a), indices.shape[1], 3)
    q = g[indices[mapping]]
    pc, qc = p - p.mean(1)[:, None], q - q.mean(1)[:, None]
    rp = np.sqrt((pc ** 2).sum(-1).mean(1))
    rq = np.sqrt((qc ** 2).sum(-1).mean(1))
    ratio = rp / np.maximum(rq, 1e-6)
    ordered = np.argsort(rq)
    count = max(1, len(ordered) // 10)
    local_cost = ((p[:, :, None] - q[:, None]) ** 2).sum(-1)
    emd = [c[np.arange(len(c)), linear_sum_assignment(c)[1]].mean() for c in local_cost]
    return {"anchor_rms": float(np.sqrt(cost[np.arange(len(a)), mapping].mean())),
            "full_patch_emd_squared": float(np.mean(emd)),
            "spread_ratio_median": float(np.median(ratio)),
            "dense_decile_spread_ratio_median": float(np.median(ratio[ordered[:count]])),
            "sparse_decile_spread_ratio_median": float(np.median(ratio[ordered[-count:]])),
            "spread_ratio_p10": float(np.quantile(ratio, .1)),
            "spread_ratio_p90": float(np.quantile(ratio, .9)),
            "centroid_rms": float(np.sqrt(((p.mean(1) - q.mean(1)) ** 2).sum(-1).mean())),
            "scope": "primary pair; disjoint prepared targets; hard spatial anchor assignment; squared EMD without training factor two"}
