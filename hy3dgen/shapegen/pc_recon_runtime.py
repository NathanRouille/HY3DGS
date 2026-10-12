"""Conservative runtime helpers for fixed-view reconstruction experiments."""
import copy
import itertools
from contextlib import nullcontext

import torch
from torch.utils.data import Dataset


class FixedSurfaceDataset(Dataset):
    """Normalize/load fixed surfaces once; retain only reconstruction metadata."""
    def __init__(self, source):
        if source.seed is None or source.view_sample_mode != "first":
            raise ValueError("Surface caching requires a fixed seed and first-view sampling")
        self.mesh_paths = source.mesh_paths
        self.samples = source.samples
        keys = ("surface", "mesh_path", "surface_frame", "view_idx", "view_indices")
        self.items = [{k: v for k, v in source[i].items() if k in keys}
                      for i in range(len(source))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # Tensor clones protect the stored normalized data from in-place callers.
        return {k: v.clone() if torch.is_tensor(v) else v
                for k, v in self.items[index].items()}


class PooledSurfaceDataset(Dataset):
    """Cache every ordered view tuple while preserving room/pair sampling.

    Images are needed for normalization only during preload. Cached VGGT
    payloads remain on CPU; learned context projections still run every step.
    """
    def __init__(self, source, *, keep_context=False):
        if source.seed is None or source.views_per_sample < 2:
            raise ValueError("Pooled caching requires a seed and multi-view samples")
        if keep_context and not source.use_joint_vggt_cache:
            raise ValueError("Conditioned pooled caching requires joint VGGT caches")
        self.source = source
        self.mesh_paths, self.samples = source.mesh_paths, source.samples
        self.items = {}
        keys = {"surface", "mesh_path", "surface_frame", "view_idx", "view_indices", "align_mode"}
        if keep_context:
            keys.add("vggt_cache")
        for path, _ in self.samples:
            pool = source._mesh_available.get(path) or source._pool_for_mesh(path)
            for views in itertools.permutations(pool, source.views_per_sample):
                item = source._load_multi_view_sample(path, list(views))
                if keep_context and "vggt_cache" not in item:
                    raise ValueError(f"Missing joint context: {path} {views}")
                self.items[path, views] = {k: v for k, v in item.items() if k in keys}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        views = tuple(self.source._choose_views(path, index))
        return {k: v.clone() if torch.is_tensor(v) else v
                for k, v in self.items[path, views].items()}


def autocast_context(device, precision):
    return (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if precision == "bf16" else nullcontext())


def flow_training_output(model, z, weak_ctx, *, recon_only=False, **kwargs):
    if recon_only:
        zero = z.new_zeros(())
        return {"flow/flow_loss": zero, "flow/t_mean": zero,
                "flow/velocity_mse": zero, "flow/x_start_mse": zero}
    return model.forward_denoising(z, weak_ctx, **kwargs)


def precision_preflight(model, surface, criterion, precision, *, p95_limit=0.002):
    """Same-checkpoint FP32/BF16 comparison with finite gradient checks.

    A failed BF16 check falls back to FP32. The probe always uses the *baseline*
    criterion, so choosing local CD/OT cannot influence precision selection.
    Restore RNG and clear probe gradients; no optimizer update happens here.
    """
    if precision != "bf16":
        return precision, {"precision": precision}
    if surface.device.type != "cuda" or not torch.cuda.is_bf16_supported():
        return "fp32", {"precision": "fp32", "reason": "CUDA BF16 unavailable"}
    device = surface.device
    baseline = copy.copy(criterion)
    baseline.local_loss = copy.copy(criterion.local_loss)
    baseline.local_loss.mode = "none"
    results = []
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device)
    try:
        for mode in ("fp32", "bf16"):
            torch.set_rng_state(rng)
            torch.cuda.set_rng_state(cuda_rng, device)
            model.zero_grad(set_to_none=True)
            try:
                with autocast_context(device, mode):
                    _, fps, xyz, rgb, centers = model.forward_tokenizer(surface)
                gt_xyz, gt_rgb = model.surface_gt_points(surface, include_sharp_label=True,
                                                          include_rgb=not model.geometry_only)
                loss, probe_extras = baseline(xyz.float(), rgb.float() if rgb is not None else None,
                                   gt_xyz.float(), gt_rgb, centers.float(), fps.float())
                loss.backward()
            except RuntimeError as error:
                if mode == "fp32":
                    raise
                return "fp32", {"precision": "fp32", "bf16_passed": False,
                                "reason": str(error)}
            grad_sq = [p.grad.detach().float().square().sum() for p in model.parameters()
                       if p.grad is not None]
            norm = torch.stack(grad_sq).sum().sqrt()
            results.append((xyz.detach().float(), float(loss.detach()), float(norm),
                            float(probe_extras["loss_cd"])))
            del loss, probe_extras, xyz, rgb, centers, fps
        deviation = torch.linalg.vector_norm(results[0][0] - results[1][0], dim=-1)
        per_room_p95 = torch.quantile(deviation, 0.95, dim=1)
        p95 = float(per_room_p95.max())
        p99 = float(torch.quantile(deviation, 0.99, dim=1).max())
        maximum = float(deviation.max())
        relative = abs(results[1][1] - results[0][1]) / max(abs(results[0][1]), 1e-8)
        cd_relative = abs(results[1][3] - results[0][3]) / max(abs(results[0][3]), 1e-8)
        norm_ratio = results[1][2] / max(results[0][2], 1e-8)
        passed = (p95 <= p95_limit and p99 <= 0.004 and maximum <= 0.01
                  and relative <= 0.03 and cd_relative <= 0.05 and 0.8 <= norm_ratio <= 1.2
                  and all(torch.isfinite(torch.tensor(x)) for r in results for x in r[1:]))
        return ("bf16" if passed else "fp32"), {
            "precision": "bf16" if passed else "fp32", "bf16_passed": bool(passed),
            "xyz_p95_normalized": p95, "loss_relative_error": relative,
            "xyz_p95_per_room": per_room_p95.cpu().tolist(),
            "xyz_p99_normalized": p99, "xyz_max_normalized": maximum,
            "cd_relative_error": cd_relative,
            "gradient_norm_ratio": norm_ratio, "xyz_p95_limit": p95_limit}
    finally:
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        torch.cuda.set_rng_state(cuda_rng, device)
