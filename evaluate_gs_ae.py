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
  - Per-checkpoint ``results.json`` and ``summary.csv`` under ``<output_dir>/<ckpt_tag>/``
  - Optional combined ``summary_all_checkpoints.csv`` when evaluating multiple checkpoints

Usage — compare two checkpoints
--------------------------------
    python evaluate_gs_ae.py \\
        --checkpoints runs/exp_a/ckpt_010000.pt runs/exp_b/ckpt_010000.pt \\
        --data_dir  data/shapenet/val \\
        --output_dir eval/step10k \\
        --num_samples 50 \\
        --categories chair

Usage — evaluate a single checkpoint on training data (overfit test)
--------------------------------------------------------------------
    python evaluate_gs_ae.py \\
        --checkpoints runs/debug/ckpt_005000.pt \\
        --data_dir  data/shapenet/train \\
        --output_dir eval/overfit_check \\
        --num_samples 10 --shuffle \\
        --categories chair

Per checkpoint, evaluates two view sets from the full v46 GT cache:

  * **canonical** — always the 6 axis-aligned views (indices 0..5), so 6-view
    and 14-view models are compared fairly on the same cameras
  * **holdout** — 8 grid views not in 14-view training (rows 0 & 3, odd
    azimuth columns 1/3/5/7)

Saves separate visuals/metrics (``*_canonical`` / ``*_holdout``).
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
    export_gaussian_splats_gsplat,
    export_input_surface_ply,
    export_xyz_pointcloud_ply,
)
from hy3dgen.shapegen.gs_renderer import (
    GaussianRenderer,
    compute_anchor_drift_metrics,
    expand_anchor_positions,
)
from hy3dgen.shapegen.eval_metrics import compute_psnr, compute_ssim_fg
from hy3dgen.shapegen.models.autoencoders.model import ShapeGSAE
from hy3dgen.shapegen.surface_loaders import normalize_mesh
from train_gs_ae import (
    GTRGBDRenderer,
    MeshDataset,
    mesh_path_has_usable_gt_cache,
    resolve_category_ids,
    snap_train_views_v46,
    train_view_indices_v46,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _get_turbo():
    try:
        return matplotlib.colormaps["turbo"]
    except Exception:
        return cm.get_cmap("turbo")


def _depth_to_rgb_u8(
    depth: torch.Tensor,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Map a (H, W, 1) depth tensor to a turbo-colormapped RGB image.

    When ``vmin``/``vmax`` are provided, the colormap range is fixed; otherwise
    it is derived per-image from the foreground pixels. Using a shared range
    (from the GT depths of the same sample) prevents the small residual depth
    halo from dominating per-image normalisation in the visualisation.
    """
    d = depth.squeeze(-1).float().numpy()
    m = d > 0
    out = np.zeros((*d.shape, 3), dtype=np.uint8)
    if not np.any(m):
        return out
    if vmin is None or vmax is None:
        vmin = float(d[m].min())
        vmax = float(d[m].max())
    if vmax <= vmin:
        norm = np.zeros_like(d)
    else:
        norm = np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
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
# Diagnostic helpers
# ---------------------------------------------------------------------------

def pca_features_to_rgb(features: torch.Tensor) -> torch.Tensor:
    """Reduce per-anchor encoder features to RGB via PCA → top 3 components.

    Args:
        features : (N, D) float tensor of post-transformer features.

    Returns:
        (N, 3) float tensor in [0, 1] suitable as per-point colour.
    """
    feats = features.detach().cpu().float()
    n, d = feats.shape
    if n == 0:
        return feats.new_zeros((0, 3))
    centered = feats - feats.mean(dim=0, keepdim=True)
    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    except RuntimeError:
        # Cuda SVD can fail on degenerate batches; fall back to CPU
        _, _, vh = torch.linalg.svd(centered.cpu(), full_matrices=False)
    top3 = vh[:3]                                # (3, D)
    proj = centered @ top3.T                     # (N, 3)
    # Min-max normalise each component to [0, 1].
    p_min = proj.amin(dim=0, keepdim=True)
    p_max = proj.amax(dim=0, keepdim=True)
    rng = (p_max - p_min).clamp_min(1e-6)
    rgb = (proj - p_min) / rng
    return rgb.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# v46 view index sets (canonical vs 14-view holdout)
# ---------------------------------------------------------------------------

CANONICAL_VIEW_INDICES_V46: List[int] = list(range(6))


def eval_holdout_view_indices_v46_14grid() -> List[int]:
    """8 grid views omitted from 14-view training (rows 0 & 3, odd az columns)."""
    g = lambda r, c: 6 + r * 8 + c
    return [g(r, c) for r in (0, 3) for c in (1, 3, 5, 7)]


def train_view_indices_from_checkpoint(train_args: dict) -> List[int]:
    """Training view indices implied by checkpoint ``args['num_views']``."""
    num_views = int(train_args.get("num_views", 14))
    snapped = snap_train_views_v46(num_views)
    return train_view_indices_v46(snapped)


def _slice_views_by_indices(
    rgbs: List[torch.Tensor],
    depths: List[torch.Tensor],
    c2ws: List[torch.Tensor],
    indices: List[int],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    return (
        [rgbs[i] for i in indices],
        [depths[i] for i in indices],
        [c2ws[i] for i in indices],
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint: str,
    args: argparse.Namespace,
    device: torch.device,
) -> ShapeGSAE:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    sh_degree = int(train_args.get("sh_degree", getattr(args, "sh_degree", 1)))
    deterministic_encoder = bool(
        train_args.get(
            "deterministic_encoder",
            getattr(args, "deterministic_encoder", True),
        )
    )
    max_anchor_delta = train_args.get(
        "max_anchor_delta",
        getattr(args, "max_anchor_delta", None),
    )
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
        deterministic_encoder=deterministic_encoder,
        sh_degree=sh_degree,
        max_anchor_delta=max_anchor_delta,
    ).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def _mean_metric(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else float("nan")


@torch.no_grad()
def _evaluate_views_subset(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coeffs: torch.Tensor,
    renderer: GaussianRenderer,
    device: torch.device,
    gt_rgbs: List[torch.Tensor],
    gt_depths: List[torch.Tensor],
    c2ws: List[torch.Tensor],
    lpips_net=None,
) -> Tuple[Dict[str, float], List[Image.Image]]:
    """Render and score one list of GT views (shared Gaussian prediction)."""
    gt_depth_vals: List[float] = []
    for gd in gt_depths:
        m = gd > 0
        if m.any():
            gt_depth_vals.append(float(gd[m].min().item()))
            gt_depth_vals.append(float(gd[m].max().item()))
    if gt_depth_vals:
        depth_vmin = min(gt_depth_vals)
        depth_vmax = max(gt_depth_vals)
    else:
        depth_vmin = depth_vmax = None
    depth_err_vmax: Optional[float] = None
    if depth_vmin is not None and depth_vmax is not None:
        depth_err_vmax = max((depth_vmax - depth_vmin) / 5.0, 1e-3)

    psnr_list, ssim_list, lpips_list, depth_l1_list = [], [], [], []
    row_images: List[Image.Image] = []

    for gt_rgb, gt_depth, c2w in zip(gt_rgbs, gt_depths, c2ws):
        valid_mask = gt_depth > 0
        valid_ratio = float(valid_mask.float().mean().item())
        if valid_ratio < 0.02:
            continue

        out = renderer(means, scales, rotations, opacities, sh_coeffs, c2w.to(device))
        pred_rgb = out["rgb"].cpu()
        pred_depth = out["depth"].cpu()

        psnr_list.append(compute_psnr(pred_rgb, gt_rgb, valid_mask))
        ssim_list.append(compute_ssim_fg(pred_rgb, gt_rgb, valid_mask))

        if lpips_net is not None:
            pred_nchw = pred_rgb.unsqueeze(0).permute(0, 3, 1, 2).to(device)
            gt_nchw = gt_rgb.unsqueeze(0).permute(0, 3, 1, 2).to(device)
            lpips_list.append(float(lpips_net(pred_nchw, gt_nchw).mean().item()))

        depth_l1 = F.l1_loss(
            pred_depth[valid_mask].float().reshape(-1),
            gt_depth[valid_mask].float().reshape(-1),
        )
        depth_l1_list.append(float(depth_l1.item()))

        rgb_err = (pred_rgb - gt_rgb).abs()
        rgb_err_scalar = rgb_err.mean(dim=-1, keepdim=True)
        depth_err = (pred_depth - gt_depth).abs()

        row = _hconcat([
            Image.fromarray(_rgb01_to_u8(gt_rgb)),
            Image.fromarray(_rgb01_to_u8(pred_rgb)),
            Image.fromarray(_error_to_rgb_u8(rgb_err_scalar)),
            Image.fromarray(_depth_to_rgb_u8(gt_depth, depth_vmin, depth_vmax)),
            Image.fromarray(_depth_to_rgb_u8(pred_depth, depth_vmin, depth_vmax)),
            Image.fromarray(_error_to_rgb_u8(depth_err, vmax=depth_err_vmax)),
        ])
        row_images.append(row)

    metrics = {
        "psnr_fg": _mean_metric(psnr_list),
        "ssim_fg": _mean_metric(ssim_list),
        "lpips_fg": _mean_metric(lpips_list),
        "mean_depth_l1": _mean_metric(depth_l1_list),
        "n_views": len(psnr_list),
    }
    return metrics, row_images


def _prefix_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{k}_{prefix}": v for k, v in metrics.items()}


@torch.no_grad()
def evaluate_sample(
    model: ShapeGSAE,
    renderer: GaussianRenderer,
    sample: Dict,
    device: torch.device,
    gt_renderer: GTRGBDRenderer,
    canonical_view_indices: List[int],
    holdout_view_indices: List[int],
    ckpt_train_view_indices: Optional[List[int]] = None,
    lpips_net=None,
    drift_threshold: float = 0.1,
) -> Tuple[Dict[str, float], Dict[str, List[Image.Image]], Dict[str, torch.Tensor]]:
    """Evaluate canonical (6) + holdout (8) view sets using full v46 GT from disk.

    Returns (flat_metrics_dict, {"canonical": rows, "holdout": rows}, export_tensors).
    """
    mesh_path = sample["mesh_path"]
    rgbs_full, depths_full, c2ws_full, _ = gt_renderer.get_or_render(
        mesh_path, allow_render=False,
    )

    canon_rgbs, canon_depths, canon_c2ws = _slice_views_by_indices(
        rgbs_full, depths_full, c2ws_full, canonical_view_indices,
    )
    hold_rgbs, hold_depths, hold_c2ws = _slice_views_by_indices(
        rgbs_full, depths_full, c2ws_full, holdout_view_indices,
    )

    surface = sample["surface"].unsqueeze(0).to(device)
    latents, query_positions = model.encode(surface)
    means, scales, rotations, opacities, sh_coeffs, features = model.decode(
        latents, query_positions, return_features=True,
    )
    query_positions = query_positions[0]
    means = means[0]
    scales = scales[0]
    rotations = rotations[0]
    opacities = opacities[0]
    sh_coeffs = sh_coeffs[0]
    features = features[0]

    metrics_canon, rows_canon = _evaluate_views_subset(
        means, scales, rotations, opacities, sh_coeffs,
        renderer, device, canon_rgbs, canon_depths, canon_c2ws, lpips_net,
    )
    metrics_holdout, rows_holdout = _evaluate_views_subset(
        means, scales, rotations, opacities, sh_coeffs,
        renderer, device, hold_rgbs, hold_depths, hold_c2ws, lpips_net,
    )

    anchors = expand_anchor_positions(query_positions, model.num_gs_per_anchor)
    drift_metrics = compute_anchor_drift_metrics(
        means, anchors, drift_threshold=drift_threshold,
    )

    flat_metrics = {
        **_prefix_metrics(metrics_canon, "canonical"),
        **_prefix_metrics(metrics_holdout, "holdout"),
        **drift_metrics,
        "canonical_view_indices": canonical_view_indices,
        "holdout_view_indices": holdout_view_indices,
    }
    if ckpt_train_view_indices is not None:
        flat_metrics["ckpt_train_view_indices"] = ckpt_train_view_indices
    export_tensors = {
        "surface": sample["surface"].detach().cpu(),
        "query_positions": query_positions.detach().cpu(),
        "means": means.detach().cpu(),
        "scales": scales.detach().cpu(),
        "rotations": rotations.detach().cpu(),
        "opacities": opacities.detach().cpu(),
        "sh_coeffs": sh_coeffs.detach().cpu(),
        "features": features.detach().cpu(),
    }
    row_images = {"canonical": rows_canon, "holdout": rows_holdout}
    return flat_metrics, row_images, export_tensors


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

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    canonical_view_indices = list(CANONICAL_VIEW_INDICES_V46)
    holdout_view_indices = eval_holdout_view_indices_v46_14grid()
    ckpt_train_view_indices = train_view_indices_from_checkpoint(train_args)
    logger.info(
        "Eval canonical=%s | holdout=%s | checkpoint trained on %d views %s",
        canonical_view_indices,
        holdout_view_indices,
        len(ckpt_train_view_indices),
        ckpt_train_view_indices,
    )

    model = load_model(checkpoint, args, device)
    renderer = GaussianRenderer(
        height=args.render_height,
        width=args.render_width,
        render_depth=True,
        sh_degree=model.sh_degree,
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
    pca_anchors_dir = export_dir / "fps_anchors_pca"
    normalized_mesh_dir = export_dir / "normalized_meshes"
    gs_ply_dir = export_dir / "gaussians_ply"
    gs_splat_dir = export_dir / "gaussians_splat"
    if not getattr(args, "no_export_3d", False):
        input_ply_dir.mkdir(parents=True, exist_ok=True)
        fps_anchors_dir.mkdir(parents=True, exist_ok=True)
        pca_anchors_dir.mkdir(parents=True, exist_ok=True)
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

        metrics, row_images_by_split, export_tensors = evaluate_sample(
            model,
            renderer,
            sample,
            device,
            dataset.gt_renderer,
            canonical_view_indices,
            holdout_view_indices,
            ckpt_train_view_indices=ckpt_train_view_indices,
            lpips_net=lpips_net,
            drift_threshold=args.drift_threshold,
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
            # PCA-coloured anchors: open in CloudCompare to inspect whether the
            # encoder produces semantically clustered features (good) or noise.
            try:
                pca_rgb = pca_features_to_rgb(export_tensors["features"])
                export_xyz_pointcloud_ply(
                    export_tensors["query_positions"],
                    pca_anchors_dir / f"anchor_pca_{stem}.ply",
                    colors=pca_rgb,
                )
            except Exception as e:
                logger.warning("PCA anchor export failed for %s: %s", mesh_stem, e)
            mesh_path = sample.get("mesh_path")
            if mesh_path:
                try:
                    export_normalized_mesh_obj(
                        mesh_path,
                        normalized_mesh_dir / f"mesh_{stem}.obj",
                    )
                except Exception as e:
                    logger.warning("Normalized mesh export failed for %s: %s", mesh_stem, e)
            export_gaussian_splats_gsplat(
                export_tensors["means"],
                export_tensors["scales"],
                export_tensors["rotations"],
                export_tensors["opacities"],
                export_tensors["sh_coeffs"],
                gs_ply_dir / f"gaussian_{stem}.ply",
                format=getattr(args, "gs_export_format", "ply_compressed"),
                sh_degree=model.sh_degree,
            )
            export_gaussian_splat_file(
                export_tensors["means"],
                export_tensors["scales"],
                export_tensors["rotations"],
                export_tensors["opacities"],
                export_tensors["sh_coeffs"],
                gs_splat_dir / f"gaussian_{stem}.splat",
                sh_degree=model.sh_degree,
            )

        for split_name in ("canonical", "holdout"):
            rows = row_images_by_split.get(split_name) or []
            if not rows:
                continue
            m_psnr = metrics.get(f"psnr_fg_{split_name}", float("nan"))
            m_ssim = metrics.get(f"ssim_fg_{split_name}", float("nan"))
            m_lpips = metrics.get(f"lpips_fg_{split_name}", float("nan"))
            grid = _vconcat(rows)
            grid_labeled = add_label_bar(
                grid,
                f"{ckpt_tag} | {mesh_stem} | {split_name} | "
                f"PSNR={m_psnr:.2f}dB  SSIM={m_ssim:.4f}  LPIPS={m_lpips:.4f}",
            )
            grid_labeled.save(viz_dir / f"{rank:03d}_{mesh_stem}_{split_name}.png")

        result = {
            "checkpoint": checkpoint,
            "ckpt_tag": ckpt_tag,
            "sample_idx": idx,
            "mesh_path": sample.get("mesh_path", ""),
            **metrics,
        }
        results.append(result)
        logger.info(
            "  canonical: PSNR=%.2fdB SSIM=%.4f LPIPS=%.4f L1depth=%.4f (n=%s)",
            metrics.get("psnr_fg_canonical", float("nan")),
            metrics.get("ssim_fg_canonical", float("nan")),
            metrics.get("lpips_fg_canonical", float("nan")),
            metrics.get("mean_depth_l1_canonical", float("nan")),
            metrics.get("n_views_canonical", "?"),
        )
        logger.info(
            "  holdout: PSNR=%.2fdB SSIM=%.4f LPIPS=%.4f L1depth=%.4f (n=%s)",
            metrics.get("psnr_fg_holdout", float("nan")),
            metrics.get("ssim_fg_holdout", float("nan")),
            metrics.get("lpips_fg_holdout", float("nan")),
            metrics.get("mean_depth_l1_holdout", float("nan")),
            metrics.get("n_views_holdout", "?"),
        )
        logger.info(
            "  drift: mean_l2=%.4f max_l2=%.4f p95_l2=%.4f frac_gt_%.2f=%.4f",
            metrics.get("mean_drift_l2", float("nan")),
            metrics.get("max_drift_l2", float("nan")),
            metrics.get("p95_drift_l2", float("nan")),
            metrics.get("drift_threshold", 0.1),
            metrics.get("frac_drift_gt_thresh", float("nan")),
        )

    # Per-checkpoint metrics (under output_dir / ckpt_tag /)
    ckpt_out_dir = output_dir / ckpt_tag
    results_json = ckpt_out_dir / "results.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Per-sample results saved to {results_json}")

    summary_row = _summarize_checkpoint_results(checkpoint, ckpt_tag, results)
    if summary_row:
        csv_path = ckpt_out_dir / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
            writer.writeheader()
            writer.writerow(summary_row)
        logger.info(f"Summary CSV saved to {csv_path}")
        logger.info(
            f"\n  SUMMARY [{ckpt_tag}] n={summary_row['n_samples']} samples\n"
            f"    canonical PSNR={summary_row['mean_psnr_fg_canonical']:.3f}  "
            f"SSIM={summary_row['mean_ssim_fg_canonical']:.4f}  "
            f"LPIPS={summary_row['mean_lpips_fg_canonical']:.4f}\n"
            f"    holdout  PSNR={summary_row['mean_psnr_fg_holdout']:.3f}  "
            f"SSIM={summary_row['mean_ssim_fg_holdout']:.4f}  "
            f"LPIPS={summary_row['mean_lpips_fg_holdout']:.4f}\n"
            f"    drift mean_l2={summary_row['mean_drift_l2']:.4f}  "
            f"max_l2={summary_row['mean_max_drift_l2']:.4f}  "
            f"frac_gt_thresh={summary_row['mean_frac_drift_gt_thresh']:.4f}"
        )

    return results


def _summarize_checkpoint_results(
    checkpoint: str,
    ckpt_tag: str,
    results: List[Dict],
) -> Optional[Dict]:
    """Aggregate per-sample rows into one summary dict, or None if no valid samples."""
    valid = [
        r for r in results
        if not math.isnan(r.get("psnr_fg_canonical", float("nan")))
    ]
    if not valid:
        return None

    def _mean_key(key: str) -> float:
        vals = [r[key] for r in valid if not math.isnan(r.get(key, float("nan")))]
        return round(sum(vals) / len(vals), 4) if vals else float("nan")

    row = {
        "checkpoint": checkpoint,
        "tag": ckpt_tag,
        "n_samples": len(valid),
        "mean_psnr_fg_canonical": _mean_key("psnr_fg_canonical"),
        "mean_ssim_fg_canonical": _mean_key("ssim_fg_canonical"),
        "mean_lpips_fg_canonical": _mean_key("lpips_fg_canonical"),
        "mean_depth_l1_canonical": _mean_key("mean_depth_l1_canonical"),
        "mean_psnr_fg_holdout": _mean_key("psnr_fg_holdout"),
        "mean_ssim_fg_holdout": _mean_key("ssim_fg_holdout"),
        "mean_lpips_fg_holdout": _mean_key("lpips_fg_holdout"),
        "mean_depth_l1_holdout": _mean_key("mean_depth_l1_holdout"),
        "mean_drift_l2": _mean_key("mean_drift_l2"),
        "mean_max_drift_l2": _mean_key("max_drift_l2"),
        "mean_p95_drift_l2": _mean_key("p95_drift_l2"),
        "mean_frac_drift_gt_thresh": _mean_key("frac_drift_gt_thresh"),
    }
    if valid:
        row["canonical_view_indices"] = json.dumps(
            valid[0].get("canonical_view_indices", []),
        )
        row["holdout_view_indices"] = json.dumps(valid[0].get("holdout_view_indices", []))
        row["ckpt_train_view_indices"] = json.dumps(
            valid[0].get("ckpt_train_view_indices", []),
        )
    return row


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
    p.add_argument(
        "--sh_degree",
        type=int,
        default=1,
        choices=(0, 1),
        help="SH degree (overridden by checkpoint args when present).",
    )
    p.add_argument(
        "--deterministic_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Encoder FPS determinism (overridden by checkpoint args when present).",
    )
    p.add_argument(
        "--max_anchor_delta",
        type=float,
        default=None,
        help="Override checkpoint max_anchor_delta (AnchorSplat uses 10/128 ≈ 0.078).",
    )
    p.add_argument(
        "--drift_threshold",
        type=float,
        default=0.1,
        help="L2 drift threshold for frac_drift_gt_thresh metric (normalised coords).",
    )

    # Rendering
    p.add_argument("--render_height", type=int, default=512)
    p.add_argument("--render_width", type=int, default=512)
    p.add_argument("--camera_distance", type=float, default=3.5)
    p.add_argument("--elevation_deg", type=float, default=20.0)
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
    p.add_argument(
        "--gs_export_format",
        type=str,
        default="ply_compressed",
        choices=("ply", "ply_compressed", "splat"),
        help="gsplat export format for gaussians_ply/*.ply (ply_compressed for SH1+; "
        "SH0 checkpoints auto-use standard ply for SuperSplat).",
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

    categories: Optional[Set[str]] = resolve_category_ids(args.categories)
    if categories is not None:
        logger.info("Category filter: %s", sorted(categories))

    # Surface loader only; GT RGBD is read from full v46 cache inside evaluate_sample.
    dataset = MeshDataset(
        data_dir=args.data_dir,
        pc_size=args.pc_size,
        pc_sharpedge_size=args.pc_sharpedge_size,
        render_height=args.render_height,
        render_width=args.render_width,
        num_views=14,
        camera_distance=args.camera_distance,
        elevation_deg=args.elevation_deg,
        max_items=args.max_items,
        mesh_blacklist=args.mesh_blacklist,
        categories=categories,
        precache_full_views=True,
        require_cached_gt=args.only_cached_gt,
    )
    logger.info(f"Dataset: {len(dataset)} meshes in {args.data_dir}")

    mesh_paths = list(dataset.mesh_paths)
    tag = dataset.gt_renderer._tag
    if args.only_cached_gt:
        mesh_paths = [
            p for p in mesh_paths
            if mesh_path_has_usable_gt_cache(p, tag)
        ]
        logger.info(
            "Filtered to %d meshes with usable v46 GT on disk (tag '%s')",
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

    # Evaluate each checkpoint (metrics saved under output_dir/<ckpt_tag>/)
    summary_rows = []
    for ckpt_path in args.checkpoints:
        ckpt_tag = Path(ckpt_path).parent.name + "__" + Path(ckpt_path).stem
        ckpt_tag = ckpt_tag.replace("/", "_").replace("\\", "_")

        results = evaluate_checkpoint(
            ckpt_path, dataset, sample_indices, args, device, output_dir, ckpt_tag
        )
        row = _summarize_checkpoint_results(ckpt_path, ckpt_tag, results)
        if row:
            summary_rows.append(row)

    # Optional combined summary when multiple checkpoints are evaluated together
    if len(summary_rows) > 1:
        combined_csv = output_dir / "summary_all_checkpoints.csv"
        with open(combined_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"Combined summary CSV saved to {combined_csv}")

    if summary_rows:
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        hdr = (
            f"{'Checkpoint':<40} "
            f"{'CA_PSNR':>8} {'CA_SSIM':>8} {'HO_PSNR':>8} {'HO_SSIM':>8}"
        )
        print(hdr)
        print("-" * len(hdr))
        for row in sorted(summary_rows, key=lambda x: -x["mean_psnr_fg_canonical"]):
            name = Path(row["checkpoint"]).parent.name + "/" + Path(row["checkpoint"]).stem
            if len(name) > 38:
                name = ".." + name[-36:]
            print(
                f"  {name:<40} "
                f"{row['mean_psnr_fg_canonical']:>8.3f} "
                f"{row['mean_ssim_fg_canonical']:>8.4f} "
                f"{row['mean_psnr_fg_holdout']:>8.3f} "
                f"{row['mean_ssim_fg_holdout']:>8.4f}"
            )
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
