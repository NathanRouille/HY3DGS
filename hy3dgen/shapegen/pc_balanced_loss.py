"""One anchor bijection, complete owned targets, exact local EMD gradients."""
from collections import defaultdict
import time

import torch

from .pc_assignment import ExactAssignment


class BalancedPatchLoss:
    def __init__(self, patch_count=256, backend="auto"):
        self.patch_count = int(patch_count)
        if self.patch_count <= 0:
            raise ValueError("Patch count must be positive")
        self.anchor_solver = ExactAssignment(backend)
        self.point_solver = ExactAssignment(backend)
        self.cursors = defaultdict(int)

    def __call__(self, pred, centers, gt, queries, indices, keys=None):
        b, r, _ = centers.shape
        k = pred.shape[1] // r
        if indices.shape != (b, r, k) or gt.shape[1] != r * k:
            raise ValueError("Balanced local loss requires exactly R disjoint K-point patches")
        started = time.perf_counter()
        with torch.no_grad():
            cost = torch.cdist(centers.detach().float(), queries.detach().float()).square()
            assignment = self.anchor_solver(cost)  # predicted group → GT patch
        anchor = (centers - queries.gather(1, assignment[..., None].expand(-1, -1, 3))).square().sum(-1).mean()
        count = min(self.patch_count, r)
        groups, targets = [], []
        pred_groups = pred.reshape(b, r, k, 3)
        for i in range(b):
            key = str(keys[i]) if keys is not None else "default"
            # Cycle target patches per room, across all camera pairs. Invert the
            # bijection to select their corresponding predicted groups.
            cursor = self.cursors[key]
            target_ids = (torch.arange(count, device=pred.device) + cursor) % r
            self.cursors[key] += count
            inverse = assignment[i].argsort()
            groups.append(pred_groups[i, inverse[target_ids]])
            targets.append(gt[i, indices[i, target_ids].long()])
        p, q = torch.cat(groups), torch.cat(targets)
        point_cost = (p[:, :, None] - q[:, None, :]).square().sum(-1)
        point_assignment = self.point_solver(point_cost.detach())
        local = 2 * point_cost.gather(2, point_assignment[..., None]).mean()
        return anchor, local, {
            "local/valid_patches": pred.new_tensor(r),
            "local/valid_fraction": pred.new_tensor(1.),
            "local/used_patches": pred.new_tensor(count),
            "matched/seconds": pred.new_tensor(time.perf_counter() - started)}
