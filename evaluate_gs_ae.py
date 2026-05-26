#!/usr/bin/env python3
"""Standalone evaluation script for ShapeGSAE checkpoints.

Evaluates one or more checkpoints on a validation set and produces:
  - Per-sample and mean PSNR (foreground only)
  - Per-sample and mean SSIM (foreground only)
  - Mean predicted alpha on foreground
  - Mean predicted depth coverage
  - Visual comparison grids (GT RGB | Pred RGB | GT depth | Pred depth)
  - Input surface point clouds (``.ply``)
  - FPS encoder anchors / ``query_positions`` (``.ply``)
  - Normalized meshes aligned with the dataloader (``.obj``)
  - Predicted 3D Gaussians (standard 3DGS ``.ply`` + ``.splat`` for web viewers)
  - A summary CSV for easy comparison across experiments
  - A summary JSON with all per-sample metrics

Usage — compare two checkpoints
--------------------------------
    python evaluate_gs_ae.py \\
        --checkpoints runs/exp_a/ckpt_010000.pt runs/exp_b/ckpt_010000.pt \\
        --data_dir  data/shapenet/val \\
        --output_dir eval/step10k \\
        --num_samples 50 \\
        --num_views 4 --camera_azimuths "0,90,180,270" \\
        --categories chair

Usage — evaluate a single checkpoint on training data (overfit test)
--------------------------------------------------------------------
    python evaluate_gs_ae.py \\
        --checkpoints runs/debug/ckpt_005000.pt \\
        --data_dir  data/shapenet/train \\
        --output_dir eval/overfit_check \\
        --num_samples 10 --shuffle \\
        --categories chair
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import trimesh

from hy3dgen.shapegen.gs_export import (
    export_gaussian_splat_file,
    export_gaussian_splat_ply,
    export_input_surface_ply,
    export_xyz_pointcloud_ply,
)
from hy3dgen.shapegen.gs_renderer import GaussianRenderer, _ssim
from hy3dgen.shapegen.models.autoencoders.model import ShapeGSAE
from hy3dgen.shapegen.surface_loaders import normalize_mesh
from train_gs_ae import GTRGBDRenderer, MeshDataset, mesh_path_has_usable_gt_cache, resolve_category_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """PSNR (dB) on foreground pixels. All tensors float in [0,1]."""
    pred_fg = pred[mask.expand_as(pred)]
    gt_fg = gt[mask.expand_as(gt)]
    if len(pred_fg) == 0:
        return float("nan")
    mse = F.mse_loss(pred_fg, gt_fg)
    if mse.item() < 1e-10:
        return 100.0
    return float(-10.0 * torch.log10(mse).item())


def compute_ssim_fg(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """SSIM [0,1] on foreground-masked image (higher is better).

    Background pixels in pred are replaced with GT background before computing
    SSIM to isolate foreground quality.
    """
    mask_f = mask.float()
    pred_m = pred * mask_f + gt * (1.0 - mask_f)
    pred_nchw = pred_m.unsqueeze(0).permute(0, 3, 1, 2)
    gt_nchw = gt.unsqueeze(0).permute(0, 3, 1, 2)
    return float(_ssim(pred_nchw, gt_nchw).item())


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _get_turbo():
    try:
        return matplotlib.colormaps["turbo"]
    except Exception:
        return cm.get_cmap("turbo")


def _depth_to_rgb_u8(depth: torch.Tensor) -> np.ndarray:
    d = depth.squeeze(-1).float().numpy()
    m = d > 0
    out = np.zeros((*d.shape, 3), dtype=np.uint8)
    if not np.any(m):
        return out
    vmin, vmax = float(d[m].min()), float(d[m].max())
    norm = (
        np.zeros_like(d)
        if vmax <= vmin
        else np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
    )
    cmap = _get_turbo()
    rgba = cmap(norm)
    return (rgba[..., :3] * 255.0 * m[..., None]).astype(np.uint8)


def _error_to_rgb_u8(
    err: torch.Tensor,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Render a non-negative scalar error map as a magma heatmap (uint8 RGB).

    Low error → dark/black; high error → bright/yellow.
    ``vmax`` can be set to a fixed value to keep colour scales comparable
    across views; if None the per-image maximum is used.
    """
    e = err.squeeze(-1).float().numpy()
    emax = float(e.max()) if vmax is None else float(vmax)
    if emax <= 0:
        return np.zeros((*e.shape, 3), dtype=np.uint8)
    norm = np.clip(e / emax, 0.0, 1.0)
    try:
        cmap = matplotlib.colormaps["magma"]
    except Exception:
        cmap = cm.get_cmap("magma")
    rgba = cmap(norm)
    return (rgba[..., :3] * 255.0).astype(np.uint8)


def _rgb01_to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.detach().clamp(0, 1).float().cpu().numpy() * 255.0).round().astype(np.uint8)


def _hconcat(images: List[Image.Image]) -> Image.Image:
    wsum = sum(im.width for im in images)
    hmax = max(im.height for im in images)
    canvas = Image.new("RGB", (wsum, hmax), (255, 255, 255))
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width
    return canvas


def _vconcat(images: List[Image.Image]) -> Image.Image:
    hsum = sum(im.height for im in images)
    wmax = max(im.width for im in images)
    canvas = Image.new("RGB", (wmax, hsum), (255, 255, 255))
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height
    return canvas


def add_label_bar(
    image: Image.Image,
    label: str,
    bar_height: int = 18,
    font_size: int = 12,
) -> Image.Image:
    """Add a white label bar at the top of an image."""
    from PIL import ImageDraw, ImageFont
    bar = Image.new("RGB", (image.width, bar_height), (220, 220, 220))
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, 2), label, fill=(30, 30, 30), font=font)
    return _vconcat([bar, image])


# ---------------------------------------------------------------------------
# Debug mesh export (same normalization as RGBSharpEdgeSurfaceLoader)
# ---------------------------------------------------------------------------

def export_normalized_mesh_obj(mesh_path: str, path: Path) -> None:
    """Save normalized mesh as OBJ with per-sample MTL + texture for CloudCompare.

    Uses the same geometry normalization as ``load_surface_sharpedge_rgb``.
    Each export gets ``{stem}.obj``, ``{stem}.mtl``, and ``{stem}_texture.<ext>``
    so multi-sample eval runs do not clobber shared ``material_0.png`` files.
    """
    stem = path.stem
    out_dir = path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = GTRGBDRenderer._load_mesh(mesh_path)
    try:
        mesh_full = trimesh.util.concatenate(mesh.dump())
    except Exception:
        mesh_full = trimesh.util.concatenate(mesh)
    mesh_full = normalize_mesh(mesh_full)

    # If no UV texture is present, bake vertex colors so CloudCompare still shows RGB.
    has_uv_texture = (
        isinstance(mesh_full.visual, trimesh.visual.texture.TextureVisuals)
        and getattr(mesh_full.visual, "uv", None) is not None
        and mesh_full.visual.material is not None
        and getattr(mesh_full.visual.material, "image", None) is not None
    )
    if not has_uv_texture:
        from hy3dgen.shapegen.surface_loaders import _get_vertex_colors

        vc = (_get_vertex_colors(mesh_full) * 255.0).round().astype(np.uint8)
        mesh_full.visual = trimesh.visual.ColorVisuals(
            mesh=mesh_full,
            vertex_colors=vc,
        )

    tmp_dir = out_dir / f".__mesh_export_{stem}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    try:
        tmp_obj = tmp_dir / "mesh.obj"
        mesh_full.export(tmp_obj)

        mtl_src: Optional[Path] = None
        tex_src: Optional[Path] = None
        for f in tmp_dir.iterdir():
            if f.suffix.lower() == ".mtl":
                mtl_src = f
            elif f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tga"):
                tex_src = f

        mtl_dst = out_dir / f"{stem}.mtl"
        tex_dst: Optional[Path] = None
        if tex_src is not None:
            tex_dst = out_dir / f"{stem}_texture{tex_src.suffix.lower()}"
            shutil.copy2(tex_src, tex_dst)

        if mtl_src is not None:
            mtl_text = mtl_src.read_text()
            if tex_dst is not None:
                tex_name = tex_dst.name
                mtl_text = re.sub(
                    r"(?m)^(map_Kd\s+)\S+",
                    lambda m, n=tex_name: f"{m.group(1)}{n}",
                    mtl_text,
                )
            mtl_dst.write_text(mtl_text)

        obj_text = tmp_obj.read_text()
        if mtl_dst.exists():
            obj_text = re.sub(
                r"(?m)^mtllib\s+\S+",
                f"mtllib {mtl_dst.name}",
                obj_text,
                count=1,
            )
        path.write_text(obj_text)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint: str,
    args: argparse.Namespace,
    device: torch.device,
) -> ShapeGSAE:
    model = ShapeGSAE(
        num_latents=args.num_latents,
        embed_dim=args.embed_dim,
        width=args.width,
        heads=args.heads,
        num_decoder_layers=args.num_decoder_layers,
        num_encoder_layers=args.num_encoder_layers,
        pc_size=args.pc_size,
        pc_sharpedge_size=args.pc_sharpedge_size,
        point_feats=6,
        downsample_ratio=args.downsample_ratio,
        num_gs_per_anchor=args.num_gs_per_anchor,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sample(
    model: ShapeGSAE,
    renderer: GaussianRenderer,
    sample: Dict,
    device: torch.device,
    lpips_net=None,
) -> Tuple[Dict[str, float], List[Image.Image], Dict[str, torch.Tensor]]:
    """Evaluate one mesh sample.

    Returns (metrics_dict, list_of_row_images, export_tensors).
    """
    surface = sample["surface"].unsqueeze(0).to(device)
    gt_rgbs = sample["rgbs"]
    gt_depths = sample["depths"]
    c2ws = sample["c2ws"]

    latents, query_positions = model.encode(surface)
    means, scales, rotations, opacities, colors = model.decode(latents, query_positions)
    query_positions = query_positions[0]
    means = means[0]
    scales = scales[0]
    rotations = rotations[0]
    opacities = opacities[0]
    colors = colors[0]

    psnr_list, ssim_list, lpips_list, depth_l1_list = [], [], [], []
    row_images: List[Image.Image] = []

    for gt_rgb, gt_depth, c2w in zip(gt_rgbs, gt_depths, c2ws):
        valid_mask = gt_depth > 0  # (H, W, 1)
        valid_ratio = float(valid_mask.float().mean().item())
        if valid_ratio < 0.02:
            continue

        out = renderer(means, scales, rotations, opacities, colors, c2w.to(device))
        pred_rgb = out["rgb"].cpu()
        pred_depth = out["depth"].cpu()

        psnr_list.append(compute_psnr(pred_rgb, gt_rgb, valid_mask))
        ssim_list.append(compute_ssim_fg(pred_rgb, gt_rgb, valid_mask))

        # LPIPS — full-image, unmasked (matches training loss)
        if lpips_net is not None:
            pred_nchw = pred_rgb.unsqueeze(0).permute(0, 3, 1, 2).to(device)
            gt_nchw = gt_rgb.unsqueeze(0).permute(0, 3, 1, 2).to(device)
            lpips_val = float(lpips_net(pred_nchw, gt_nchw).mean().item())
            lpips_list.append(lpips_val)

        # Depth L1 over foreground only (diagnostic metric; full-image L1 is dominated by bg zeros)
        depth_l1 = F.l1_loss(
            pred_depth[valid_mask].float().reshape(-1),
            gt_depth[valid_mask].float().reshape(-1),
        )
        depth_l1_list.append(float(depth_l1.item()))

        # ---- Error maps ----
        rgb_err = (pred_rgb - gt_rgb).abs()          # (H, W, 3) → mean over channels
        rgb_err_scalar = rgb_err.mean(dim=-1, keepdim=True)  # (H, W, 1)
        depth_err = (pred_depth - gt_depth).abs()             # (H, W, 1)

        # Build visual row: GT RGB | Pred RGB | RGB Error | GT Depth | Pred Depth | Depth Error
        row = _hconcat([
            Image.fromarray(_rgb01_to_u8(gt_rgb)),
            Image.fromarray(_rgb01_to_u8(pred_rgb)),
            Image.fromarray(_error_to_rgb_u8(rgb_err_scalar)),
            Image.fromarray(_depth_to_rgb_u8(gt_depth)),
            Image.fromarray(_depth_to_rgb_u8(pred_depth)),
            Image.fromarray(_error_to_rgb_u8(depth_err)),
        ])
        row_images.append(row)

    def _mean(lst):
        return float(sum(lst) / len(lst)) if lst else float("nan")

    metrics = {
        "psnr_fg": _mean(psnr_list),
        "ssim_fg": _mean(ssim_list),
        "lpips_fg": _mean(lpips_list),
        "mean_depth_l1": _mean(depth_l1_list),
        "n_views": len(psnr_list),
    }
    export_tensors = {
        "surface": sample["surface"].detach().cpu(),
        "query_positions": query_positions.detach().cpu(),
        "means": means.detach().cpu(),
        "scales": scales.detach().cpu(),
        "rotations": rotations.detach().cpu(),
        "opacities": opacities.detach().cpu(),
        "colors": colors.detach().cpu(),
    }
    return metrics, row_images, export_tensors


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: str,
    dataset,
    sample_indices: List[int],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    ckpt_tag: str,
) -> List[Dict]:
    """Evaluate one checkpoint on the given sample indices. Returns per-sample results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {checkpoint}")
    logger.info(f"Tag: {ckpt_tag}")
    logger.info(f"{'='*60}")

    model = load_model(checkpoint, args, device)
    renderer = GaussianRenderer(
        height=args.render_height,
        width=args.render_width,
        render_depth=True,
    ).to(device)

    # Lazy-load LPIPS once per checkpoint (reused across all samples).
    lpips_net = None
    try:
        import lpips as _lpips_mod
        lpips_net = _lpips_mod.LPIPS(net='vgg', verbose=False)
        lpips_net.eval()
        for p in lpips_net.parameters():
            p.requires_grad = False
        lpips_net = lpips_net.to(device)
    except ImportError:
        logger.warning("lpips not installed — lpips_fg will be NaN. Install with: pip install lpips")

    viz_dir = output_dir / ckpt_tag / "visuals"
    viz_dir.mkdir(parents=True, exist_ok=True)
    export_dir = output_dir / ckpt_tag / "exports"
    input_ply_dir = export_dir / "input_clouds"
    fps_anchors_dir = export_dir / "fps_anchors"
    normalized_mesh_dir = export_dir / "normalized_meshes"
    gs_ply_dir = export_dir / "gaussians_ply"
    gs_splat_dir = export_dir / "gaussians_splat"
    if not getattr(args, "no_export_3d", False):
        input_ply_dir.mkdir(parents=True, exist_ok=True)
        fps_anchors_dir.mkdir(parents=True, exist_ok=True)
        normalized_mesh_dir.mkdir(parents=True, exist_ok=True)
        gs_ply_dir.mkdir(parents=True, exist_ok=True)
        gs_splat_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for rank, idx in enumerate(sample_indices):
        try:
            sample = dataset[idx]
        except Exception as e:
            logger.warning(f"Sample {idx} failed: {e}")
            continue

        mesh_stem = Path(sample.get("mesh_path", str(idx))).stem
        logger.info(f"  [{rank+1}/{len(sample_indices)}] {mesh_stem}")

        metrics, row_images, export_tensors = evaluate_sample(
            model, renderer, sample, device, lpips_net=lpips_net,
        )

        if not getattr(args, "no_export_3d", False):
            # File naming: {type_prefix}_{rank:03d}_{mesh_stem}.{ext}
            # Makes multi-sample exports easy to sort and identify in CloudCompare.
            stem = f"{rank:03d}_{mesh_stem}"
            export_input_surface_ply(
                export_tensors["surface"],
                input_ply_dir / f"points_{stem}.ply",
            )
            export_xyz_pointcloud_ply(
                export_tensors["query_positions"],
                fps_anchors_dir / f"anchor_{stem}.ply",
            )
            mesh_path = sample.get("mesh_path")
            if mesh_path:
                try:
                    export_normalized_mesh_obj(
                        mesh_path,
                        normalized_mesh_dir / f"mesh_{stem}.obj",
                    )
                except Exception as e:
                    logger.warning("Normalized mesh export failed for %s: %s", mesh_stem, e)
            export_gaussian_splat_ply(
                export_tensors["means"],
                export_tensors["scales"],
                export_tensors["rotations"],
                export_tensors["opacities"],
                export_tensors["colors"],
                gs_ply_dir / f"gaussian_{stem}.ply",
            )
            export_gaussian_splat_file(
                export_tensors["means"],
                export_tensors["scales"],
                export_tensors["rotations"],
                export_tensors["opacities"],
                export_tensors["colors"],
                gs_splat_dir / f"gaussian_{stem}.splat",
            )

        # Save visual grid
        if row_images:
            grid = _vconcat(row_images)
            grid_labeled = add_label_bar(
                grid,
                f"{ckpt_tag} | {mesh_stem} | "
                f"PSNR={metrics['psnr_fg']:.2f}dB  "
                f"SSIM={metrics['ssim_fg']:.4f}  "
                f"LPIPS={metrics['lpips_fg']:.4f}",
            )
            grid_labeled.save(viz_dir / f"{rank:03d}_{mesh_stem}.png")

        result = {
            "checkpoint": checkpoint,
            "ckpt_tag": ckpt_tag,
            "sample_idx": idx,
            "mesh_path": sample.get("mesh_path", ""),
            **metrics,
        }
        results.append(result)
        logger.info(
            f"PSNR={metrics['psnr_fg']:.2f}dB  "
            f"SSIM={metrics['ssim_fg']:.4f}  "
            f"LPIPS={metrics['lpips_fg']:.4f}  "
            f"L1depth={metrics['mean_depth_l1']:.4f}"
        )

    # Summary for this checkpoint
    if results:
        valid = [r for r in results if not math.isnan(r["psnr_fg"])]
        if valid:
            mean_psnr = sum(r["psnr_fg"] for r in valid) / len(valid)
            mean_ssim = sum(r["ssim_fg"] for r in valid) / len(valid)
            mean_lpips = sum(r["lpips_fg"] for r in valid if not math.isnan(r["lpips_fg"])) / max(
                sum(1 for r in valid if not math.isnan(r["lpips_fg"])), 1
            )
            mean_depth_l1 = sum(r["mean_depth_l1"] for r in valid) / len(valid)
            logger.info(
                f"\n  SUMMARY [{ckpt_tag}] n={len(valid)} samples\n"
                f"    mean PSNR    = {mean_psnr:.3f} dB\n"
                f"    mean SSIM    = {mean_ssim:.4f}\n"
                f"    mean LPIPS   = {mean_lpips:.4f}\n"
                f"    mean L1depth = {mean_depth_l1:.4f}"
            )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate and compare ShapeGSAE checkpoints")

    # Required
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="One or more checkpoint paths to evaluate.")
    p.add_argument("--data_dir", required=True,
                   help="Directory with mesh files to evaluate on.")
    p.add_argument("--output_dir", default="eval/results",
                   help="Where to save metrics, CSVs, and visual grids.")

    # Model architecture (must match the checkpoints)
    p.add_argument("--num_latents", type=int, default=2048)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--num_encoder_layers", type=int, default=8)
    p.add_argument("--num_decoder_layers", type=int, default=8)
    p.add_argument("--pc_size", type=int, default=5120)
    p.add_argument("--pc_sharpedge_size", type=int, default=5120)
    p.add_argument("--downsample_ratio", type=int, default=20)
    p.add_argument("--num_gs_per_anchor", type=int, default=1,
                   help="Must match the value used during training.")

    # Rendering
    p.add_argument("--render_height", type=int, default=512)
    p.add_argument("--render_width", type=int, default=512)
    p.add_argument("--num_views", type=int, default=6)
    p.add_argument("--camera_distance", type=float, default=3.5)
    p.add_argument("--elevation_deg", type=float, default=20.0)
    p.add_argument("--camera_azimuths", type=str, default="0,90,180,270")
    p.add_argument(
        "--gt_view_layout",
        type=str,
        default="v46",
        choices=("legacy", "v46"),
        help="Must match the layout used to pre-cache GT.",
    )
    p.add_argument("--mesh_blacklist", type=str, default=None)
    p.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated category names or synset IDs (same as train_gs_ae.py).",
    )
    p.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Cap dataset size after category filter (same as train_gs_ae.py).",
    )

    # Sample selection
    p.add_argument("--num_samples", type=int, default=50,
                   help="Number of samples to evaluate per checkpoint.")
    p.add_argument("--indices", type=str, default=None,
                   help="Comma-separated specific sample indices to evaluate.")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle the dataset before selecting samples.")
    p.add_argument("--only_cached_gt", action="store_true", default=True)
    p.add_argument("--no_only_cached_gt", action="store_false", dest="only_cached_gt")

    # 3D asset export (eval only; training script unchanged)
    p.add_argument(
        "--no_export_3d",
        action="store_true",
        help="Skip writing input surface, FPS anchors, normalized mesh, and 3DGS exports.",
    )

    # Misc
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse camera azimuths
    azimuths = None
    if str(args.gt_view_layout).lower() == "v46":
        azimuths = None
    elif args.camera_azimuths:
        azimuths = [float(a.strip()) for a in args.camera_azimuths.split(",")]
        if len(azimuths) != args.num_views:
            raise ValueError(
                f"camera_azimuths has {len(azimuths)} values but num_views={args.num_views}"
            )

    categories: Optional[Set[str]] = resolve_category_ids(args.categories)
    if categories is not None:
        logger.info("Category filter: %s", sorted(categories))

    # Build dataset
    dataset = MeshDataset(
        data_dir=args.data_dir,
        pc_size=args.pc_size,
        pc_sharpedge_size=args.pc_sharpedge_size,
        render_height=args.render_height,
        render_width=args.render_width,
        num_views=args.num_views,
        camera_distance=args.camera_distance,
        elevation_deg=args.elevation_deg,
        max_items=args.max_items,
        mesh_blacklist=args.mesh_blacklist,
        azimuths_deg=azimuths,
        categories=categories,
        view_layout=args.gt_view_layout,
    )
    logger.info(f"Dataset: {len(dataset)} meshes in {args.data_dir}")

    # Filter to meshes with pre-cached GT if requested
    mesh_paths = list(dataset.mesh_paths)
    tag = dataset.gt_renderer._tag
    az_list = dataset.gt_renderer.azimuths_deg or [0.0]
    if args.only_cached_gt:
        mesh_paths = [
            p
            for p in mesh_paths
            if mesh_path_has_usable_gt_cache(
                p, tag, args.render_height, args.render_width, az_list
            )
        ]
        logger.info(
            "Filtered to %d meshes with usable GT on disk (tag '%s' or 4-view normv2 slice)",
            len(mesh_paths),
            tag,
        )
    dataset.mesh_paths = mesh_paths
    all_indices = list(range(len(dataset)))
    if args.shuffle:
        random.shuffle(all_indices)
    if args.indices:
        sample_indices = [int(x.strip()) for x in args.indices.split(",")]
    else:
        sample_indices = all_indices[: min(args.num_samples, len(all_indices))]
    logger.info(f"Evaluating on {len(sample_indices)} samples")

    # Evaluate each checkpoint
    all_results = []
    for ckpt_path in args.checkpoints:
        # Build a short tag from the checkpoint path for filenames
        ckpt_tag = Path(ckpt_path).parent.name + "__" + Path(ckpt_path).stem
        ckpt_tag = ckpt_tag.replace("/", "_").replace("\\", "_")

        results = evaluate_checkpoint(
            ckpt_path, dataset, sample_indices, args, device, output_dir, ckpt_tag
        )
        all_results.extend(results)

    # Save full results as JSON
    results_json = output_dir / "results.json"
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Full results saved to {results_json}")

    # Save summary CSV (one row per checkpoint)
    summary_rows = []
    for ckpt_path in args.checkpoints:
        ckpt_tag = Path(ckpt_path).parent.name + "__" + Path(ckpt_path).stem
        ckpt_tag = ckpt_tag.replace("/", "_").replace("\\", "_")
        ckpt_results = [r for r in all_results if r["ckpt_tag"] == ckpt_tag]
        valid = [r for r in ckpt_results if not math.isnan(r["psnr_fg"])]
        if not valid:
            continue
        lpips_valid = [r["lpips_fg"] for r in valid if not math.isnan(r.get("lpips_fg", float("nan")))]
        summary_rows.append({
            "checkpoint": ckpt_path,
            "tag": ckpt_tag,
            "n_samples": len(valid),
            "mean_psnr_fg": round(sum(r["psnr_fg"] for r in valid) / len(valid), 4),
            "mean_ssim_fg": round(sum(r["ssim_fg"] for r in valid) / len(valid), 4),
            "mean_lpips_fg": round(sum(lpips_valid) / len(lpips_valid), 4) if lpips_valid else float("nan"),
            "mean_depth_l1": round(sum(r["mean_depth_l1"] for r in valid) / len(valid), 4),
        })

    csv_path = output_dir / "summary.csv"
    if summary_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"Summary CSV saved to {csv_path}")

        # Print final comparison table
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"{'Checkpoint':<50} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'L1depth':>10}")
        print("-" * 80)
        for row in sorted(summary_rows, key=lambda x: -x["mean_psnr_fg"]):
            name = Path(row["checkpoint"]).parent.name + "/" + Path(row["checkpoint"]).stem
            lpips_str = f"{row['mean_lpips_fg']:>8.4f}" if not math.isnan(row["mean_lpips_fg"]) else "     n/a"
            print(
                f"  {name:<48} "
                f"{row['mean_psnr_fg']:>8.3f} "
                f"{row['mean_ssim_fg']:>8.4f} "
                f"{lpips_str} "
                f"{row['mean_depth_l1']:>10.4f}"
            )
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
