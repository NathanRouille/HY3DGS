"""Exact square assignment; optional GPU solver must pass a reference check.

The permutation is discrete. Losses gather from the original, live tensors.
No dependency installation or CUDA compilation is performed by this module.
"""
import logging
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

log = logging.getLogger(__name__)


class ExactAssignment:
    def __init__(self, backend="auto"):
        if backend not in ("auto", "scipy", "gpu"):
            raise ValueError("assignment backend must be auto/scipy/gpu")
        self.backend = backend
        self.selected = {}
        self.evidence = {}

    @staticmethod
    def _cpu(cost):
        arrays = cost.detach().cpu().numpy()
        return torch.as_tensor(np.stack([linear_sum_assignment(c)[1] for c in arrays]),
                               device=cost.device, dtype=torch.long)

    @staticmethod
    def _gpu(cost):
        from torch_linear_assignment import batch_linear_assignment
        return batch_linear_assignment(cost.detach().contiguous()).long()

    @staticmethod
    def _valid(assignment, cost):
        n = cost.shape[-1]
        return (assignment.shape == cost.shape[:2]
                and torch.equal(assignment.sort(-1).values,
                                torch.arange(n, device=cost.device).expand(cost.shape[:2])))

    @torch.no_grad()
    def __call__(self, cost):
        if cost.ndim != 3 or cost.shape[1] != cost.shape[2] or not torch.isfinite(cost).all():
            raise ValueError("Exact assignment requires finite square [B,N,N] costs")
        cost = cost.detach().float()
        key = (cost.device.type, cost.shape[0], cost.shape[1])
        if self.backend == "scipy" or cost.device.type != "cuda":
            self.evidence[str(key)] = {"selected": "scipy", "reason": "requested" if self.backend == "scipy" else "CPU input"}
            return self._cpu(cost)
        if key not in self.selected:
            try:
                # Includes D2H/H2D overhead in the reference, but excludes cold
                # Triton compilation from warm timing. Verify optimal costs, not
                # identical indices: ties admit multiple correct permutations.
                torch.cuda.synchronize(cost.device)
                start = time.perf_counter()
                reference = self._cpu(cost)
                torch.cuda.synchronize(cost.device)
                cpu_seconds = time.perf_counter() - start
                candidate = self._gpu(cost)
                if not self._valid(candidate, cost):
                    raise ValueError("GPU assignment is not a complete bijection")
                ref_cost = cost.gather(2, reference[..., None]).sum((1, 2))
                got_cost = cost.gather(2, candidate[..., None]).sum((1, 2))
                if not torch.allclose(got_cost, ref_cost, rtol=1e-5, atol=1e-6):
                    raise ValueError("GPU assignment failed the SciPy optimal-cost check")
                torch.cuda.synchronize(cost.device)
                start = time.perf_counter()
                for _ in range(2):
                    candidate = self._gpu(cost)
                torch.cuda.synchronize(cost.device)
                gpu_seconds = (time.perf_counter() - start) / 2
                use_gpu = self.backend == "gpu" or gpu_seconds < .9 * cpu_seconds
                self.selected[key] = "gpu" if use_gpu else "scipy"
                self.evidence[str(key)] = {"selected": self.selected[key],
                                          "scipy_seconds": cpu_seconds,
                                          "gpu_warm_seconds": gpu_seconds}
                log.warning("Assignment backend validation: %s", self.evidence[str(key)])
                return candidate if use_gpu else reference
            except (ImportError, RuntimeError, ValueError, AttributeError) as exc:
                if self.backend == "gpu":
                    raise RuntimeError("Requested GPU assignment could not be validated") from exc
                self.selected[key] = "scipy"
                self.evidence[str(key)] = {"selected": "scipy", "reason": str(exc)}
                log.warning("Assignment uses SciPy: %s", exc)
        if self.selected[key] == "gpu":
            candidate = self._gpu(cost)
            if not self._valid(candidate, cost):
                raise RuntimeError("GPU assignment returned an invalid permutation")
            return candidate
        return self._cpu(cost)
