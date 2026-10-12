"""Matched local CD / exact small-patch quadratic optimal transport.

Both candidates share distinct GT targets, support gating, correspondence and
patch schedule. No register/FPS index identity or forced sparse-cell padding.
"""
from __future__ import annotations

import numpy as np
import torch


def patch_distance(pred, target, mode):
    # Direct subtraction is cheap for 16x16 patches and avoids MM cancellation.
    cost = (pred[:, :, None] - target[:, None, :]).square().sum(-1)
    if mode == "cd":
        return cost.min(2).values.mean() + cost.min(1).values.mean()
    if mode != "ot":
        raise ValueError(f"Unknown local loss: {mode}")
    from scipy.optimize import linear_sum_assignment
    # A unit-mass, equal-cardinality exact assignment: mean cost, no N multiplier.
    assignments = [linear_sum_assignment(c)[1] for c in cost.detach().cpu().numpy()]
    indices = torch.as_tensor(np.stack(assignments), device=pred.device)
    # Factor two matches the two unit-weight directions of symmetric local CD.
    return 2.0 * cost.gather(2, indices[..., None]).mean()


class LocalPatchLoss:
    def __init__(self, mode="none", patch_count=64, radius_scale=0.5):
        if mode not in ("none", "cd", "ot"):
            raise ValueError("local loss must be none, cd or ot")
        if patch_count <= 0 or radius_scale <= 0:
            raise ValueError("Local patch count and radius scale must be positive")
        self.mode, self.patch_count, self.radius_scale = mode, int(patch_count), float(radius_scale)
        self.cache = []  # Bounded, runtime-only. Each entry checks the full inputs.

    def _targets(self, gt, fps, size):
        for item in self.cache:
            if item["size"] == size and item["gt"].device == gt.device and torch.equal(item["gt"], gt) and torch.equal(item["fps"], fps):
                return item
        from scipy.spatial import cKDTree
        # Duplicated sharp samples cannot become independent OT target particles.
        unique = np.unique(gt.detach().cpu().numpy(), axis=0)
        if len(unique) < size or len(fps) < 2:
            raise ValueError("Insufficient distinct points for local targets")
        f = fps.detach().cpu().numpy()
        distances, neighbors = cKDTree(unique).query(f, k=size)
        # Radius is half the eighth-other-FPS distance, limiting distant fill.
        spacing = cKDTree(f).query(f, k=min(9, len(f)))[0][:, -1]
        radius = self.radius_scale * spacing
        valid = (distances[:, -1] <= radius) & (radius > 0)
        item = {"gt": gt.detach().clone(), "fps": fps.detach().clone(), "size": size,
                "target": torch.as_tensor(unique[neighbors], device=gt.device, dtype=gt.dtype),
                "radius": torch.as_tensor(radius, device=gt.device, dtype=gt.dtype),
                "valid": torch.as_tensor(valid, device=gt.device), "cursor": 0}
        if len(self.cache) >= 16:
            self.cache.pop(0)
        self.cache.append(item)
        return item

    def __call__(self, pred, centers, gt, fps):
        b, r, _ = centers.shape
        if pred.shape[1] % r:
            raise ValueError("Local points must be grouped contiguously by anchor")
        k = pred.shape[1] // r
        locals_ = pred.reshape(b, r, k, 3)
        selected_p, selected_q, valid_counts, used_counts = [], [], [], []
        for i in range(b):
            item = self._targets(gt[i], fps[i], k)
            with torch.no_grad():
                distance = torch.cdist(fps[i].detach(), centers[i].detach())
                d, f_to_c = distance.min(1)
                c_to_f = distance.argmin(0)
                rows = torch.arange(len(fps[i]), device=fps.device)
                # Mutual NN gives a partial one-to-one spatial correspondence.
                valid = item["valid"] & (c_to_f[f_to_c] == rows) & (d <= item["radius"])
                available = rows[valid]
                count = min(self.patch_count, len(available))
                valid_counts.append(int(len(available)))
                used_counts.append(count)
                if not count:
                    continue
                # Deterministic cycling: covers patches without consuming model RNG.
                position = (torch.arange(count, device=fps.device) + item["cursor"]) % len(available)
                chosen = available[position]
                item["cursor"] += count
            selected_p.append(locals_[i, f_to_c[chosen]])
            selected_q.append(item["target"][chosen])
        value = (patch_distance(torch.cat(selected_p), torch.cat(selected_q), self.mode)
                 if selected_p else pred.sum() * 0.0)
        return value, {"local/valid_patches": pred.new_tensor(sum(valid_counts) / b),
                       "local/used_patches": pred.new_tensor(sum(used_counts) / b),
                       "local/valid_fraction": pred.new_tensor(sum(valid_counts) / (b * fps.shape[1]))}
