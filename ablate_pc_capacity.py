#!/usr/bin/env python3
"""Point-cloud capacity ablation for 3DGS reconstruction.

Overfits a **single ShapeNet mesh** with a plain MLP: the point cloud is
sorted deterministically, flattened to a 1-D vector, and fed to a deep MLP
that outputs M Gaussian parameters.  The existing GaussianRenderer and
RGBDLoss are used unchanged.

Goal: find the minimum input density N at which a capacity-unlimited network
can perfectly reconstruct colours – proving (or disproving) that texture
information actually exists in the sampled point cloud.

Quick sweep example
-------------------
    # Try N=256 with M=1024 Gaussians
    python ablate_pc_capacity.py \\
        --mesh_path /path/to/ShapeNetCore/03001627/.../model_normalized.obj \\
        --N 256 --M 1024 --max_steps 10000

    # Richer input – N=2048, more Gaussians
    python ablate_pc_capacity.py \\
        --mesh_path /path/to/.../model_normalized.obj \\
        --N 2048 --M 4096 --hidden_dim 2048 --num_layers 12 --max_steps 20000

Sorting convention
------------------
Points are sorted by (Z, Y, X) so every run with the same random seed
produces the exact same 1-D input vector regardless of the upstream sampler's
output order.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hy3dgen.shapegen.gs_renderer import GaussianRenderer, RGBDLoss
from hy3dgen.shapegen.surface_loaders import RGBSharpEdgeSurfaceLoader
from hy3dgen.shapegen.eval_metrics import compute_psnr_fg, compute_ssim_full
from train_gs_ae import GTRGBDRenderer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLP architecture
# ---------------------------------------------------------------------------

class PointCloudMLP(nn.Module):
    """Flatten-and-fit MLP.

    Input : sorted point cloud  (N_pts, 9) → flattened to (N_pts * 9,)
    Output: M Gaussians, each with 14 raw parameters:
              [pos_delta(3) | log_scale(3) | quat_wxyz(4) | opa_logit(1) | rgb(3)]

    Activations are applied externally (same as ShapeGSAE._parse_gaussians).
    """

    def __init__(
        self,
        n_input_points: int,
        n_gaussians: int,
        hidden_dim: int = 1024,
        num_layers: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_input_points = n_input_points
        self.n_gaussians = n_gaussians

        in_dim = n_input_points * 9       # xyz(3)+normals(3)+rgb(3) per point
        out_dim = n_gaussians * 14        # 14 raw params per Gaussian

        layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

        self._init_output_bias()

    def _init_output_bias(self):
        """Bias the output layer so Gaussians start small and semi-transparent."""
        last_linear: nn.Linear = self.net[-1]       # type: ignore[assignment]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)
        # Per-Gaussian: bias layout [pos(3) | log_scale(3) | quat(4) | opa(1) | rgb(3)]
        for g in range(self.n_gaussians):
            off = g * 14
            last_linear.bias.data[off + 6] = 1.0    # quat w → identity rotation
            last_linear.bias.data[off + 3:off + 6] = -3.0  # small Gaussians
            last_linear.bias.data[off + 10] = 0.4   # opacity logit → σ≈0.6

    def forward(self, surface: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Args:
            surface: (N_pts, 9) — already sorted, single scene (no batch dim).

        Returns:
            means, scales, rotations, opacities, colors
            Each has shape (M, {3|3|4|1|3}).
        """
        M = self.n_gaussians
        x = surface.flatten()                            # (N_pts*9,)
        raw = self.net(x).view(M, 14)                    # (M, 14)

        # Same activations as ShapeGSAE._parse_gaussians.
        # Means: use XYZ centroid of the input cloud as an implicit anchor
        # (the output is an absolute position, not a delta, since there's no
        # single anchor concept in a flat MLP).
        means = raw[:, :3]                               # (M, 3) – no clamping
        scales = torch.exp(raw[:, 3:6].clamp(-5.0, 2.0))
        quat_raw = raw[:, 6:10]
        quat_norm = quat_raw.norm(dim=-1, keepdim=True)
        quat_id = torch.zeros_like(quat_raw)
        quat_id[:, 0] = 1.0
        rotations = torch.where(quat_norm > 1e-8, quat_raw / quat_norm, quat_id)
        opacities = torch.sigmoid(raw[:, 10:11])
        colors = torch.sigmoid(raw[:, 11:14])

        return means, scales, rotations, opacities, colors


# ---------------------------------------------------------------------------
# Deterministic point-cloud preparation
# ---------------------------------------------------------------------------

def load_and_sort_surface(
    mesh_path: str,
    n_pts: int,
    sharp_pts: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample surface from mesh and sort points by (Z, Y, X) for MLP stability.

    Returns:
        surface: (n_pts + sharp_pts, 9) float32 on *device*
    """
    rng_state = np.random.get_state(), torch.get_rng_state()
    np.random.seed(seed)
    torch.manual_seed(seed)

    loader = RGBSharpEdgeSurfaceLoader(
        num_uniform_points=n_pts,
        num_sharp_points=sharp_pts,
    )
    surface = loader(mesh_path).squeeze(0)   # (N, 9) float32

    np.random.set_state(rng_state[0])
    torch.set_rng_state(rng_state[1])

    # Sort by Z then Y then X for permutation invariance.
    xyz = surface[:, :3]                             # (N, 3)
    order = torch.argsort(
        xyz[:, 2] * 1e6 + xyz[:, 1] * 1e3 + xyz[:, 0]
    )
    surface = surface[order]

    return surface.to(device)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    # ---- Ground-truth RGBD ----
    # Reuse the same GT renderer infrastructure (v46 layout, 46 cached views).
    # We slice down to the first --num_views (canonical 6 + first staggered).
    gt_renderer = GTRGBDRenderer(
        height=args.render_height,
        width=args.render_width,
        camera_distance=args.camera_distance,
        elevation_deg=args.elevation_deg,
        train_view_indices=list(range(args.num_views)),
    )
    logger.info("Loading GT RGBD for %s …", args.mesh_path)
    gt_rgbs, gt_depths, c2ws, _ = gt_renderer.get_or_render(args.mesh_path)
    logger.info("Loaded %d GT views (%dx%d)", len(gt_rgbs), args.render_height, args.render_width)

    # ---- Input surface (deterministically sorted) ----
    n_total = args.N + args.N_sharp
    logger.info(
        "Sampling surface: N=%d uniform + %d sharp = %d total points",
        args.N, args.N_sharp, n_total,
    )
    surface = load_and_sort_surface(
        args.mesh_path, args.N, args.N_sharp, args.seed, device
    )
    logger.info("Surface shape: %s  (sorted by ZYX)", tuple(surface.shape))

    # ---- MLP ----
    model = PointCloudMLP(
        n_input_points=n_total,
        n_gaussians=args.M,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "MLP: %d input dims → %d Gaussians | hidden=%d × %d layers | %.2fM params",
        n_total * 9, args.M, args.hidden_dim, args.num_layers, n_params / 1e6,
    )

    # ---- Renderer & loss ----
    renderer = GaussianRenderer(
        height=args.render_height,
        width=args.render_width,
        render_depth=True,
    ).to(device)

    criterion = RGBDLoss(
        lambda_ssim=args.lambda_ssim,
        lambda_lpips=0.0,          # skip LPIPS for speed; enable if you want
        lambda_d=args.lambda_d,
        lambda_alpha=args.lambda_alpha,
        lambda_scale=args.lambda_scale,
        lambda_scale_max=getattr(args, "lambda_scale_max", 0.0),
        lambda_opa=args.lambda_opa,
        rgb_loss_type=args.rgb_loss_type,
    )

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.01
    )

    # ---- Training ----
    logger.info("Starting overfit training for %d steps …", args.max_steps)
    t0 = time.time()
    best_psnr = -float("inf")

    for step in range(1, args.max_steps + 1):
        model.train()
        optimizer.zero_grad()

        means, scales, rotations, opacities, colors = model(surface)

        total_loss = torch.zeros((), device=device)
        num_valid = 0
        for gt_rgb, gt_depth, c2w in zip(gt_rgbs, gt_depths, c2ws):
            gt_rgb = gt_rgb.to(device)
            gt_depth = gt_depth.to(device)
            valid_mask = gt_depth > 0
            if float(valid_mask.float().mean()) < 0.02:
                continue

            out = renderer(means, scales, rotations, opacities, colors, c2w.to(device))
            view_loss, _ = criterion(
                out["rgb"], gt_rgb,
                out["depth"], gt_depth,
                pred_alpha=out["alpha"],
                valid_mask=valid_mask,
            )
            total_loss = total_loss + view_loss
            num_valid += 1

        if num_valid == 0:
            logger.warning("step %d: all views skipped (no foreground GT)", step)
            continue

        total_loss = total_loss / num_valid
        if criterion.lambda_scale > 0 or criterion.lambda_opa > 0:
            gaussian_loss, _ = criterion.gaussian_regularizer(
                scales,
                opacities,
            )
            total_loss = total_loss + gaussian_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ---- Logging ----
        if step % args.log_every == 0 or step == args.max_steps:
            model.eval()
            with torch.no_grad():
                psnr_list, ssim_list = [], []
                for gt_rgb, gt_depth, c2w in zip(gt_rgbs, gt_depths, c2ws):
                    gt_rgb = gt_rgb.to(device)
                    gt_depth = gt_depth.to(device)
                    valid_mask = gt_depth > 0
                    if float(valid_mask.float().mean()) < 0.02:
                        continue
                    out = renderer(means, scales, rotations, opacities, colors, c2w.to(device))
                    pred_rgb = out["rgb"].cpu()
                    gt_rgb_cpu = gt_rgb.cpu()
                    psnr_list.append(compute_psnr_fg(pred_rgb, gt_rgb_cpu, valid_mask.cpu()))
                    ssim_list.append(compute_ssim_full(pred_rgb, gt_rgb_cpu))

            mean_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else float("nan")
            mean_ssim = sum(ssim_list) / len(ssim_list) if ssim_list else float("nan")
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0

            logger.info(
                "step=%06d | loss=%.4f | PSNR=%.2f dB | SSIM=%.4f | lr=%.2e | %.1fs",
                step, float(total_loss.item()), mean_psnr, mean_ssim, lr_now, elapsed,
            )

            if mean_psnr > best_psnr:
                best_psnr = mean_psnr
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "psnr": mean_psnr,
                    "ssim": mean_ssim,
                    "args": vars(args),
                }
                torch.save(ckpt, out_dir / "best.pt")

        # ---- Checkpoint ----
        if step % args.save_every == 0 or step == args.max_steps:
            torch.save(
                {"step": step, "model": model.state_dict(), "args": vars(args)},
                out_dir / f"ckpt_{step:06d}.pt",
            )

    logger.info("Done. Best PSNR = %.2f dB  (saved to %s/best.pt)", best_psnr, out_dir)

    # ---- Final render grid ----
    _save_render_grid(
        model, renderer, gt_rgbs, gt_depths, c2ws, surface, device, out_dir
    )


def _save_render_grid(
    model: PointCloudMLP,
    renderer: GaussianRenderer,
    gt_rgbs: list,
    gt_depths: list,
    c2ws: list,
    surface: torch.Tensor,
    device: torch.device,
    out_dir: Path,
) -> None:
    """Save a side-by-side GT | Pred render grid for visual inspection."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping render grid.")
        return

    model.eval()
    with torch.no_grad():
        means, scales, rotations, opacities, colors = model(surface)
        rows: list = []
        for gt_rgb, gt_depth, c2w in zip(gt_rgbs, gt_depths, c2ws):
            out = renderer(means, scales, rotations, opacities, colors, c2w.to(device))
            pred_np = out["rgb"].cpu().float().clamp(0, 1).numpy()
            gt_np = gt_rgb.float().numpy()
            rows.append(np.concatenate([gt_np, pred_np], axis=1))  # side by side

    grid = np.concatenate(rows, axis=0)
    fig, ax = plt.subplots(figsize=(8, 4 * len(rows)))
    ax.imshow(grid)
    ax.axis("off")
    ax.set_title("GT (left) | Pred (right)")
    grid_path = out_dir / "final_renders.png"
    fig.savefig(grid_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Render grid saved → %s", grid_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MLP overfitting ablation: point cloud capacity vs 3DGS quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Scene ----
    p.add_argument("--mesh_path", required=True,
                   help="Path to the single mesh to overfit (OBJ / GLB / PLY).")
    p.add_argument("--output_dir", default="runs/ablation/pc_capacity",
                   help="Directory for checkpoints and render grids.")

    # ---- Key ablation variables ----
    p.add_argument("--N", type=int, default=512,
                   help="Number of UNIFORM surface points fed to the MLP. "
                        "This is the primary ablation axis.")
    p.add_argument("--N_sharp", type=int, default=512,
                   help="Number of SHARP-EDGE points appended to the uniform cloud. "
                        "Set to 0 to use only uniform sampling.")
    p.add_argument("--M", type=int, default=2048,
                   help="Number of output Gaussians.")

    # ---- MLP ----
    p.add_argument("--hidden_dim", type=int, default=1024,
                   help="Width of each hidden layer.")
    p.add_argument("--num_layers", type=int, default=8,
                   help="Total number of linear layers (including output).")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout rate (keep 0 for pure overfitting).")

    # ---- Rendering ----
    p.add_argument("--render_height", type=int, default=256)
    p.add_argument("--render_width", type=int, default=256)
    p.add_argument("--num_views", type=int, default=6,
                   help="Number of orbit views for GT and training supervision.")
    p.add_argument("--camera_distance", type=float, default=3.5)
    p.add_argument("--elevation_deg", type=float, default=20.0)

    # ---- Loss weights ----
    p.add_argument("--rgb_loss_type", choices=("mse", "l1"), default="mse")
    p.add_argument("--lambda_ssim", type=float, default=0.2)
    p.add_argument("--lambda_d", type=float, default=1.0)
    p.add_argument("--lambda_alpha", type=float, default=0.05)
    p.add_argument("--lambda_scale", type=float, default=0.01,
                   help="AnchorSplat volume penalty weight.")
    p.add_argument("--lambda_opa", type=float, default=0.05,
                   help="Binary opacity entropy weight.")

    # ---- Optimiser ----
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="Zero weight decay is typical for overfitting experiments.")
    p.add_argument("--max_steps", type=int, default=10_000)

    # ---- Misc ----
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--save_every", type=int, default=5_000)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
