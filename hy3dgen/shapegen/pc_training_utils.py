"""Training diagnostics for ShapePCAE / ShapePCUnite (wandb helpers)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .pc_losses import PointCloudAELoss, chamfer_distance, rgb_l1_on_nn


def grad_norm_for_loss(
    parameters: List[torch.nn.Parameter],
    loss_scalar: torch.Tensor,
) -> float:
    """L2 norm of gradients of ``loss_scalar`` w.r.t. trainable parameters."""
    if not loss_scalar.requires_grad:
        return 0.0
    params = [p for p in parameters if p.requires_grad]
    if not params:
        return 0.0
    grads = torch.autograd.grad(
        loss_scalar,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    total_sq = sum(
        g.detach().float().norm().pow(2).item()
        for g in grads
        if g is not None
    )
    return float(total_sq ** 0.5)


def global_grad_norm(model: nn.Module) -> float:
    """L2 norm of all parameter gradients (after backward)."""
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total_sq += float(g.norm().pow(2).item())
    return float(total_sq ** 0.5)


def compute_pc_loss_grad_norms(
    model: nn.Module,
    criterion: PointCloudAELoss,
    *,
    extras: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Per-term gradient L2 norms for PC AE losses."""
    params = list(model.parameters())
    norms: Dict[str, float] = {}
    norms["cd"] = grad_norm_for_loss(params, extras["_cd"])
    if criterion.lambda_rgb > 0:
        norms["rgb"] = grad_norm_for_loss(params, criterion.lambda_rgb * extras["_rgb"])
    if criterion.lambda_anc > 0:
        norms["anc"] = grad_norm_for_loss(params, criterion.lambda_anc * extras["_anc"])
    if getattr(criterion, "lambda_anc_cd", 0.0) > 0 and "_anc_cd" in extras:
        norms["anc_cd"] = grad_norm_for_loss(
            params, criterion.lambda_anc_cd * extras["_anc_cd"]
        )
    return norms


def latent_statistics(latents: torch.Tensor) -> Dict[str, float]:
    """Scalar stats for register latents ``[B, R, D]``."""
    z = latents.detach().float()
    return {
        "latent/mean": float(z.mean().item()),
        "latent/std": float(z.std(unbiased=False).item()),
        "latent/max_abs": float(z.abs().max().item()),
    }


def anchor_diagnostics(
    centers: torch.Tensor,
    fps_xyz: torch.Tensor,
    *,
    collapse_eps: float = 1e-3,
) -> Dict[str, float]:
    """Anchor uniformity / collapse metrics."""
    c = centers.detach().float()
    f = fps_xyz.detach().float()
    # NN distance center -> FPS
    dist_cf = torch.cdist(c, f, p=2)
    min_cf, _ = dist_cf.min(dim=2)
    mean_dist = float(min_cf.mean().item())
    max_dist = float(min_cf.max().item())

    # Pairwise center distances for collapse fraction
    if c.shape[1] > 1:
        pd = torch.cdist(c, c, p=2)
        eye = torch.eye(c.shape[1], device=c.device, dtype=torch.bool)
        pd = pd.masked_fill(eye.unsqueeze(0), float("inf"))
        min_pair = pd.min(dim=2).values
        frac_collapsed = float((min_pair < collapse_eps).float().mean().item())
    else:
        frac_collapsed = 0.0

    spread_std = float(c.std(dim=1).mean().item())
    return {
        "anchor/mean_dist_to_fps": mean_dist,
        "anchor/max_dist_to_fps": max_dist,
        "anchor/frac_collapsed": frac_collapsed,
        "anchor/spread_std": spread_std,
    }


def loss_balance_ratios(
    *,
    loss_cd: float,
    loss_rgb: float,
    loss_anc: float,
    lambda_rgb: float,
    lambda_anc: float,
    loss_anc_cd: float = 0.0,
    lambda_anc_cd: float = 0.0,
) -> Dict[str, float]:
    weighted_rgb = lambda_rgb * loss_rgb
    weighted_anc = lambda_anc * loss_anc
    weighted_anc_cd = lambda_anc_cd * loss_anc_cd
    eps = 1e-8
    out = {
        "loss_ratio/cd_over_rgb": loss_cd / (weighted_rgb + eps),
        "loss_ratio/anc_over_cd": weighted_anc / (loss_cd + eps),
    }
    if lambda_anc_cd > 0:
        out["loss_ratio/anc_cd_over_cd"] = weighted_anc_cd / (loss_cd + eps)
        out["loss_ratio/anc_cd_over_anc"] = weighted_anc_cd / (weighted_anc + eps)
    return out


@torch.no_grad()
def evaluate_fixed_meshes(
    model: nn.Module,
    mesh_paths: List[str],
    *,
    device: torch.device,
    criterion: PointCloudAELoss,
    surface_loader,
    include_sharp_label: bool,
    prefix: str,
) -> Dict[str, float]:
    """Run a tiny fixed-mesh eval (train or val subset)."""
    from hy3dgen.shapegen.models.autoencoders.shape_pc_ae import ShapePCAE

    if not mesh_paths:
        return {}

    cds: List[float] = []
    rgbs: List[float] = []
    ancs: List[float] = []

    model.eval()
    for path in mesh_paths:
        surface = surface_loader(path).squeeze(0).unsqueeze(0).to(device)
        xyz, rgb, centers, fps_xyz, _ = model(surface)
        gt_xyz, gt_rgb = ShapePCAE.surface_gt_points(
            surface, include_sharp_label=include_sharp_label
        )
        _, extras = criterion(
            xyz, rgb, gt_xyz, gt_rgb, centers=centers, fps_xyz=fps_xyz
        )
        cds.append(float(extras["loss_cd"]))
        rgbs.append(float(extras["loss_rgb"]))
        ancs.append(float(extras["loss_anc"]))
    model.train()

    n = max(len(cds), 1)
    return {
        f"fixed_eval/{prefix}_cd": sum(cds) / n,
        f"fixed_eval/{prefix}_rgb": sum(rgbs) / n,
        f"fixed_eval/{prefix}_anc": sum(ancs) / n,
    }


def pick_fixed_mesh_paths(
    dataset,
    n: int,
    *,
    seed: int = 0,
) -> List[str]:
    """Pick ``n`` mesh paths from a SurfaceOnlyDataset."""
    paths = list(dataset.mesh_paths)
    if not paths:
        return []
    n = min(n, len(paths))
    if n == len(paths):
        return paths
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(paths), generator=g).tolist()
    return [paths[i] for i in perm[:n]]
