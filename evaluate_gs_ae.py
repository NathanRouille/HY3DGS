#!/usr/bin/env python3
"""Standalone evaluation script for ShapeGSAE checkpoints.

Evaluates one or more checkpoints on a validation set and produces:
  - Per-sample and mean PSNR (foreground-only and full-image)
  - Per-sample and mean SSIM (full-image; no GT background replacement)
  - Mean predicted composited alpha on foreground and background
  - Full-image LPIPS and foreground depth L1
  - Visual comparison grids (GT RGB | Pred RGB | GT depth | Pred depth)
  - Input surface point clouds (``.ply``)
  - FPS encoder anchors / ``query_positions`` (``.ply``)
  - Anchors coloured by predicted DC RGB and by PCA of latents/features
  - Per-anchor latent tensors (``.pt``: z, pre-attn, post-attn)
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

Per checkpoint, evaluates view splits:

  * **train** (G-Objaverse, ``--match_training``) — all loaded training views
  * **spread** (G-Objaverse, ``--no-match_training``) — ~10 well-separated views
  * **canonical** / **holdout** (ShapeNet v46) — first 6 views + grid holdout rows

Saves separate visuals/metrics per split name.
Summary CSV uses ``spread`` or ``canonical/holdout`` format per metric.

G-Objaverse: pass ``--data_dir`` to ``.../furniture_351/val``; ``gt_source`` is
read from manifest / checkpoint. ShapeNet: unchanged v46 GT cache path.
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
    export_latent_tokens_pt,
    export_xyz_pointcloud_ply,
)
from hy3dgen.shapegen.gobjaverse_gt import (
    GOBJAVERSE_GT_TAG,
    GOBJAVERSE_NUM_VIEWS,
    gobjaverse_eval_view_indices,
    normalize_mesh_gobjaverse,
)
from hy3dgen.shapegen.gs_renderer import (
    CANONICAL_VIEW_INDICES_V46,
    GaussianRenderer,
    compute_anchor_drift_metrics,
    expand_anchor_positions,
    holdout_view_indices_v46,
    sh_dc_to_rgb,
)
from hy3dgen.shapegen.eval_metrics import (
    compute_mean_alpha_bg,
    compute_mean_alpha_fg,
    compute_psnr_fg,
    compute_psnr_full,
    compute_ssim_full,
)
from hy3dgen.shapegen.models.autoencoders.model import ShapeGSAE
from hy3dgen.shapegen.pretrained_profiles import resolve_include_sharp_label
from hy3dgen.shapegen.surface_loaders import normalize_mesh
from train_gs_ae import (
    GTRGBDRenderer,
    MeshDataset,
    load_experiment_manifest,
    mesh_path_has_usable_gt_cache,
    resolve_category_ids,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_ARCH_KEYS = (
    "num_latents",
    "embed_dim",
    "width",
    "heads",
    "num_encoder_layers",
    "num_decoder_layers",
    "pc_size",
    "pc_sharpedge_size",
    "downsample_ratio",
    "num_gs_per_anchor",
    "point_feats",
    "qk_norm",
    "qkv_bias",
    "include_pi",
    "max_log_scale",
    "pretrained_profile",
    "include_sharp_label",
    "gt_source",
    "num_views",
    "views_per_step",
    "gobjaverse_num_views",
    "render_height",
    "render_width",
)


def merge_checkpoint_train_args(
    eval_args: argparse.Namespace,
    train_args: dict,
) -> None:
    """Restore architecture and surface layout from a training checkpoint."""
    if not train_args:
        return
    for key in CHECKPOINT_ARCH_KEYS:
        if key in train_args and train_args[key] is not None:
            setattr(eval_args, key, train_args[key])


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

def export_normalized_mesh_obj(
    mesh_path: str,
    path: Path,
    *,
    gobjaverse_meta: Optional[dict] = None,
) -> None:
    """Save normalized mesh as OBJ with per-sample MTL + texture for CloudCompare.

    Uses the same geometry normalization as ``load_surface_sharpedge_rgb``.
    Each export gets ``{stem}.obj``, ``{stem}.mtl``, and ``{stem}_texture.<ext>``
    so multi-sample eval runs do not clobber shared ``material_0.png`` files.
    """
    stem = path.stem
    out_dir = path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(mesh_path, process=False)
    from hy3dgen.shapegen.surface_loaders import (
        merge_parts_vertex_colored,
        normalize_parts_gobjaverse,
        scene_to_parts,
    )

    parts = scene_to_parts(mesh)
    if gobjaverse_meta is not None:
        parts = normalize_parts_gobjaverse(parts, gobjaverse_meta)
        mesh_full = merge_parts_vertex_colored(parts)
    else:
        from hy3dgen.shapegen.surface_loaders import normalize_parts
        parts = normalize_parts(parts)
        mesh_full = merge_parts_vertex_colored(parts)

    from hy3dgen.shapegen.surface_loaders import extract_mesh_texture, _get_vertex_colors

    vertex_uvs, texture_rgb = extract_mesh_texture(mesh_full)
    has_uv_texture = vertex_uvs is not None and texture_rgb is not None
    if has_uv_texture:
        material = mesh_full.visual.material
        # NOTE: trimesh's OBJ exporter calls `material.to_simple()`, which for a
        # PBRMaterial reads `baseColorTexture` (there is no `.image` attribute on
        # PBRMaterial at all -- setting `.image` silently creates an unused
        # attribute and the exporter falls back to whatever texture/color was
        # already on the material, or none). SimpleMaterial uses `.image` instead.
        # Bake the exact texture we sampled the point cloud from so the exported
        # OBJ is guaranteed to match training data pixel-for-pixel.
        if material is not None:
            from PIL import Image
            baked_image = Image.fromarray(
                (np.clip(texture_rgb, 0, 1) * 255.0).round().astype(np.uint8),
            )
            if hasattr(material, "baseColorTexture"):
                material.baseColorTexture = baked_image
            else:
                material.image = baked_image
    else:
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


def _slice_views_by_indices(
    rgbs: List[torch.Tensor],
    depths: List[torch.Tensor],
    c2ws: List[torch.Tensor],
    indices: List[int],
    view_params: Optional[List[Dict]] = None,
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    Optional[List[Dict]],
]:
    vp_out = None
    if view_params is not None:
        vp_out = [view_params[i] for i in indices]
    return (
        [rgbs[i] for i in indices],
        [depths[i] for i in indices],
        [c2ws[i] for i in indices],
        vp_out,
    )


def _eval_view_splits(
    gt_source: str,
    num_loaded_views: int,
    *,
    match_training: bool,
) -> Dict[str, List[int]]:
    """View index groups for evaluation.

    When ``match_training`` and G-Objaverse, score all loaded training views
    (same set as ``MeshDataset`` / ``views_per_step`` sampling pool).
    Otherwise G-Objaverse uses the spread eval subset from
    :func:`gobjaverse_eval_view_indices`; ShapeNet uses v46 canonical (6) +
    holdout split.
    """
    if match_training and gt_source == "gobjaverse":
        return {"train": list(range(num_loaded_views))}

    if gt_source == "gobjaverse":
        spread = gobjaverse_eval_view_indices(num_loaded_views)
        return {"spread": spread}

    canonical = [i for i in CANONICAL_VIEW_INDICES_V46 if i < num_loaded_views]
    holdout = [i for i in holdout_view_indices_v46() if i < num_loaded_views]
    return {"canonical": canonical, "holdout": holdout}


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
    max_log_scale = float(
        train_args.get("max_log_scale", getattr(args, "max_log_scale", 2.0))
    )
    point_feats = int(train_args.get("point_feats", getattr(args, "point_feats", 6)))
    qk_norm = bool(train_args.get("qk_norm", getattr(args, "qk_norm", False)))
    # qkv_bias and include_pi default to True to match old checkpoints that pre-date
    # these explicit args (those were built with PyTorch Linear bias=True default).
    qkv_bias = bool(train_args.get("qkv_bias", getattr(args, "qkv_bias", True)))
    include_pi = bool(train_args.get("include_pi", getattr(args, "include_pi", True)))
    model = ShapeGSAE(
        num_latents=args.num_latents,
        embed_dim=args.embed_dim,
        width=args.width,
        heads=args.heads,
        num_decoder_layers=args.num_decoder_layers,
        num_encoder_layers=args.num_encoder_layers,
        pc_size=args.pc_size,
        pc_sharpedge_size=args.pc_sharpedge_size,
        point_feats=point_feats,
        downsample_ratio=args.downsample_ratio,
        num_gs_per_anchor=args.num_gs_per_anchor,
        deterministic_encoder=deterministic_encoder,
        sh_degree=sh_degree,
        max_anchor_delta=max_anchor_delta,
        max_log_scale=max_log_scale,
        qk_norm=qk_norm,
        qkv_bias=qkv_bias,
        include_pi=include_pi,
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
    view_params: Optional[List[Dict]] = None,
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

    psnr_fg_list, psnr_full_list = [], []
    ssim_full_list, lpips_list, depth_l1_list = [], [], []
    alpha_bg_list, alpha_fg_list = [], []
    row_images: List[Image.Image] = []

    for vi, (gt_rgb, gt_depth, c2w) in enumerate(zip(gt_rgbs, gt_depths, c2ws)):
        valid_mask = gt_depth > 0
        valid_ratio = float(valid_mask.float().mean().item())
        if valid_ratio < 0.02:
            continue

        vp = view_params[vi] if view_params is not None else None
        out = renderer(
            means, scales, rotations, opacities, sh_coeffs, c2w.to(device),
            fx=vp.get("fx") if vp else None,
            fy=vp.get("fy") if vp else None,
            cx=vp.get("cx") if vp else None,
            cy=vp.get("cy") if vp else None,
        )
        pred_rgb = out["rgb"].cpu()
        pred_depth = out["depth"].cpu()
        pred_alpha = out["alpha"].cpu()

        psnr_fg_list.append(compute_psnr_fg(pred_rgb, gt_rgb, valid_mask))
        psnr_full_list.append(compute_psnr_full(pred_rgb, gt_rgb))
        ssim_full_list.append(compute_ssim_full(pred_rgb, gt_rgb))
        alpha_bg_list.append(compute_mean_alpha_bg(pred_alpha, valid_mask))
        alpha_fg_list.append(compute_mean_alpha_fg(pred_alpha, valid_mask))

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
        "psnr_fg": _mean_metric(psnr_fg_list),
        "psnr_full": _mean_metric(psnr_full_list),
        "ssim_full": _mean_metric(ssim_full_list),
        # Backward-compatible alias (now honest full-image SSIM, no GT bg fill).
        "ssim_fg": _mean_metric(ssim_full_list),
        "lpips_fg": _mean_metric(lpips_list),
        "mean_alpha_bg": _mean_metric(alpha_bg_list),
        "mean_alpha_fg": _mean_metric(alpha_fg_list),
        "mean_depth_l1": _mean_metric(depth_l1_list),
        "n_views": len(psnr_fg_list),
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
    view_splits: Dict[str, List[int]],
    lpips_net=None,
    drift_threshold: float = 0.1,
    sample_posterior: bool = False,
) -> Tuple[Dict[str, float], Dict[str, List[Image.Image]], Dict[str, torch.Tensor]]:
    """Evaluate one or more view index groups from a dataloader sample.

    ``view_splits`` maps a split name (e.g. ``train``, ``canonical``, ``holdout``)
    to global view indices into ``sample['rgbs']``.

    Returns (flat_metrics_dict, {split: row_images}, export_tensors).
    """
    rgbs_full = sample["rgbs"]
    depths_full = sample["depths"]
    c2ws_full = sample["c2ws"]
    view_params_full = sample.get("view_params")

    surface = sample["surface"].unsqueeze(0).to(device)
    latents_z, query_positions = model.encode(
        surface, sample_posterior=sample_posterior,
    )
    means, scales, rotations, opacities, sh_coeffs, decode_diag = model.decode(
        latents_z, query_positions, return_features=True,
    )
    query_positions = query_positions[0]
    means = means[0]
    scales = scales[0]
    rotations = rotations[0]
    opacities = opacities[0]
    sh_coeffs = sh_coeffs[0]
    features = decode_diag["features"][0]
    latents_pre_attn = decode_diag["latents_pre_attn"][0]
    latents_z = latents_z[0]

    # Mean predicted DC RGB over the K Gaussians sharing each FPS anchor.
    k = model.num_gs_per_anchor
    dc_rgb_gs = sh_dc_to_rgb(sh_coeffs[:, 0, :])  # (L*K, 3)
    dc_rgb_anchors = dc_rgb_gs.view(-1, k, 3).mean(dim=1)  # (L, 3)

    flat_metrics: Dict[str, float] = {}
    row_images: Dict[str, List[Image.Image]] = {}

    for split_name, view_indices in view_splits.items():
        if not view_indices:
            continue
        split_rgbs, split_depths, split_c2ws, split_vp = _slice_views_by_indices(
            rgbs_full, depths_full, c2ws_full, view_indices, view_params_full,
        )
        split_metrics, split_rows = _evaluate_views_subset(
            means, scales, rotations, opacities, sh_coeffs,
            renderer, device, split_rgbs, split_depths, split_c2ws, lpips_net,
            view_params=split_vp,
        )
        flat_metrics.update(_prefix_metrics(split_metrics, split_name))
        row_images[split_name] = split_rows

    anchors = expand_anchor_positions(query_positions, model.num_gs_per_anchor)
    drift_metrics = compute_anchor_drift_metrics(
        means, anchors, drift_threshold=drift_threshold,
    )
    flat_metrics.update(drift_metrics)

    export_tensors = {
        "surface": sample["surface"].detach().cpu(),
        "query_positions": query_positions.detach().cpu(),
        "means": means.detach().cpu(),
        "scales": scales.detach().cpu(),
        "rotations": rotations.detach().cpu(),
        "opacities": opacities.detach().cpu(),
        "sh_coeffs": sh_coeffs.detach().cpu(),
        "features": features.detach().cpu(),
        "latents_z": latents_z.detach().cpu(),
        "latents_pre_attn": latents_pre_attn.detach().cpu(),
        "dc_rgb_anchors": dc_rgb_anchors.detach().cpu(),
    }
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
    gt_source = getattr(dataset, "gt_source", "shapenet")
    num_loaded = dataset.num_gt_views
    match_training = bool(getattr(args, "match_training", True))
    view_splits = _eval_view_splits(
        gt_source, num_loaded, match_training=match_training,
    )
    split_desc = ", ".join(
        f"{name}={len(idxs)} views" for name, idxs in view_splits.items() if idxs
    )
    logger.info(
        "Eval gt_source=%s | match_training=%s | splits: %s",
        gt_source,
        match_training,
        split_desc or "(none)",
    )
    include_sharp_label = resolve_include_sharp_label(args)

    model = load_model(checkpoint, args, device)
    det_enc = bool(
        train_args.get(
            "deterministic_encoder",
            getattr(args, "deterministic_encoder", True),
        )
    )
    train_seed = int(train_args.get("seed", getattr(args, "seed", 42)))
    logger.info(
        "Model encoder mode: deterministic=%s (train_seed=%d for input surfaces)",
        det_enc,
        train_seed,
    )
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
    dc_anchors_dir = export_dir / "fps_anchors_dc"
    latent_tokens_dir = export_dir / "latent_tokens"
    normalized_mesh_dir = export_dir / "normalized_meshes"
    gs_ply_dir = export_dir / "gaussians_ply"
    gs_splat_dir = export_dir / "gaussians_splat"
    if not getattr(args, "no_export_3d", False):
        input_ply_dir.mkdir(parents=True, exist_ok=True)
        fps_anchors_dir.mkdir(parents=True, exist_ok=True)
        pca_anchors_dir.mkdir(parents=True, exist_ok=True)
        dc_anchors_dir.mkdir(parents=True, exist_ok=True)
        latent_tokens_dir.mkdir(parents=True, exist_ok=True)
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
            view_splits,
            lpips_net=lpips_net,
            drift_threshold=args.drift_threshold,
            sample_posterior=bool(getattr(args, "sample_posterior", False)),
        )

        gobjaverse_meta = None
        if dataset.gt_source == "gobjaverse" and dataset.gobjaverse_gt is not None:
            mesh_path = sample.get("mesh_path")
            if mesh_path:
                try:
                    gobjaverse_meta = dataset.gobjaverse_gt.load_meta(mesh_path)
                except Exception as e:
                    logger.warning("G-Objaverse meta load failed for %s: %s", mesh_stem, e)

        if not getattr(args, "no_export_3d", False):
            # File naming: {type_prefix}_{rank:03d}_{mesh_stem}.{ext}
            # Makes multi-sample exports easy to sort and identify in CloudCompare.
            stem = f"{rank:03d}_{mesh_stem}"
            export_input_surface_ply(
                export_tensors["surface"],
                input_ply_dir / f"points_{stem}.ply",
                include_sharp_label=include_sharp_label,
            )
            export_xyz_pointcloud_ply(
                export_tensors["query_positions"],
                fps_anchors_dir / f"anchor_{stem}.ply",
            )
            # Predicted DC colour at each FPS anchor (mean over K Gaussians).
            try:
                export_xyz_pointcloud_ply(
                    export_tensors["query_positions"],
                    dc_anchors_dir / f"anchor_dc_{stem}.ply",
                    colors=export_tensors["dc_rgb_anchors"],
                )
            except Exception as e:
                logger.warning("DC-RGB anchor export failed for %s: %s", mesh_stem, e)
            # PCA-coloured anchors at three latent stages for CloudCompare.
            for tag, feats in (
                ("z", export_tensors["latents_z"]),
                ("pre_attn", export_tensors["latents_pre_attn"]),
                ("post_attn", export_tensors["features"]),
            ):
                try:
                    pca_rgb = pca_features_to_rgb(feats)
                    export_xyz_pointcloud_ply(
                        export_tensors["query_positions"],
                        pca_anchors_dir / f"anchor_pca_{tag}_{stem}.ply",
                        colors=pca_rgb,
                    )
                    # Keep legacy filename for post-transformer PCA.
                    if tag == "post_attn":
                        export_xyz_pointcloud_ply(
                            export_tensors["query_positions"],
                            pca_anchors_dir / f"anchor_pca_{stem}.ply",
                            colors=pca_rgb,
                        )
                except Exception as e:
                    logger.warning(
                        "PCA anchor export (%s) failed for %s: %s", tag, mesh_stem, e,
                    )
            # Full latent tensors for stats / notebooks.
            try:
                export_latent_tokens_pt(
                    latent_tokens_dir / f"latents_{stem}.pt",
                    query_positions=export_tensors["query_positions"],
                    latents_z=export_tensors["latents_z"],
                    latents_pre_attn=export_tensors["latents_pre_attn"],
                    features_post_attn=export_tensors["features"],
                    dc_rgb=export_tensors["dc_rgb_anchors"],
                )
            except Exception as e:
                logger.warning("Latent token export failed for %s: %s", mesh_stem, e)
            mesh_path = sample.get("mesh_path")
            if mesh_path:
                try:
                    export_normalized_mesh_obj(
                        mesh_path,
                        normalized_mesh_dir / f"mesh_{stem}.obj",
                        gobjaverse_meta=gobjaverse_meta,
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

        for split_name, rows in row_images_by_split.items():
            if not rows:
                continue
            m_psnr_fg = metrics.get(f"psnr_fg_{split_name}", float("nan"))
            m_psnr_full = metrics.get(f"psnr_full_{split_name}", float("nan"))
            m_ssim = metrics.get(f"ssim_full_{split_name}", float("nan"))
            m_lpips = metrics.get(f"lpips_fg_{split_name}", float("nan"))
            m_alpha_bg = metrics.get(f"mean_alpha_bg_{split_name}", float("nan"))
            grid = _vconcat(rows)
            grid_labeled = add_label_bar(
                grid,
                f"{ckpt_tag} | {mesh_stem} | {split_name} | "
                f"PSNR_fg={m_psnr_fg:.2f}  PSNR={m_psnr_full:.2f}  "
                f"SSIM={m_ssim:.4f}  LPIPS={m_lpips:.4f}  α_bg={m_alpha_bg:.4f}",
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
        for split_name in view_splits:
            if not view_splits[split_name]:
                continue
            logger.info(
                "  %-12s PSNR_fg=%.2f PSNR=%.2f SSIM=%.4f LPIPS=%.4f "
                "α_bg=%.4f L1depth=%.4f (n=%s)",
                f"{split_name}:",
                metrics.get(f"psnr_fg_{split_name}", float("nan")),
                metrics.get(f"psnr_full_{split_name}", float("nan")),
                metrics.get(f"ssim_full_{split_name}", float("nan")),
                metrics.get(f"lpips_fg_{split_name}", float("nan")),
                metrics.get(f"mean_alpha_bg_{split_name}", float("nan")),
                metrics.get(f"mean_depth_l1_{split_name}", float("nan")),
                metrics.get(f"n_views_{split_name}", "?"),
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
            f"    psnr:      {summary_row['psnr']}\n"
            f"    ssim:      {summary_row['ssim']}\n"
            f"    lpips:     {summary_row['lpips']}\n"
            f"    l1_depth:  {summary_row['l1_depth']}"
        )

    return results


def _fmt_metric_pair(canonical: float, holdout: float, precision: int) -> str:
    """Format ``canonical/holdout`` for summary CSV (canonical left, holdout right)."""
    if math.isnan(canonical) or math.isnan(holdout):
        return "nan/nan"
    return f"{canonical:.{precision}f}/{holdout:.{precision}f}"


def _summarize_checkpoint_results(
    checkpoint: str,
    ckpt_tag: str,
    results: List[Dict],
) -> Optional[Dict]:
    """Aggregate per-sample rows into one compact summary dict."""
    primary_split = "train" if any("psnr_full_train" in r for r in results) else (
        "spread" if any("psnr_full_spread" in r for r in results) else "canonical"
    )
    secondary_split = None if primary_split in ("train", "spread") else "holdout"
    primary_key = f"psnr_full_{primary_split}"

    valid = [
        r for r in results
        if not math.isnan(r.get(primary_key, float("nan")))
    ]
    if not valid:
        return None

    def _mean_key(key: str) -> float:
        vals = [r[key] for r in valid if not math.isnan(r.get(key, float("nan")))]
        return sum(vals) / len(vals) if vals else float("nan")

    psnr_p = _mean_key(f"psnr_full_{primary_split}")
    ssim_p = _mean_key(f"ssim_full_{primary_split}")
    lpips_p = _mean_key(f"lpips_fg_{primary_split}")
    depth_p = _mean_key(f"mean_depth_l1_{primary_split}")

    if secondary_split is not None:
        psnr_s = _mean_key(f"psnr_full_{secondary_split}")
        ssim_s = _mean_key(f"ssim_full_{secondary_split}")
        lpips_s = _mean_key(f"lpips_fg_{secondary_split}")
        depth_s = _mean_key(f"mean_depth_l1_{secondary_split}")
        psnr_fmt = _fmt_metric_pair(psnr_p, psnr_s, 2)
        ssim_fmt = _fmt_metric_pair(ssim_p, ssim_s, 3)
        lpips_fmt = _fmt_metric_pair(lpips_p, lpips_s, 3)
        depth_fmt = _fmt_metric_pair(depth_p, depth_s, 3)
    else:
        psnr_fmt = f"{psnr_p:.2f}"
        ssim_fmt = f"{ssim_p:.3f}"
        lpips_fmt = f"{lpips_p:.3f}"
        depth_fmt = f"{depth_p:.3f}"

    return {
        "checkpoint": checkpoint,
        "tag": ckpt_tag,
        "n_samples": len(valid),
        "eval_split": primary_split if secondary_split is None else f"{primary_split}/{secondary_split}",
        "psnr": psnr_fmt,
        "ssim": ssim_fmt,
        "lpips": lpips_fmt,
        "l1_depth": depth_fmt,
    }


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
        "--sample_posterior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample from VAE posterior at eval (default: posterior mode / mean).",
    )
    p.add_argument(
        "--max_anchor_delta",
        type=float,
        default=None,
        help="Override checkpoint max_anchor_delta (AnchorSplat uses 10/128 ≈ 0.078).",
    )
    p.add_argument(
        "--max_log_scale",
        type=float,
        default=2.0,
        help="Override checkpoint max_log_scale (hard clamp on Gaussian log-scales).",
    )
    p.add_argument(
        "--qk_norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="QK norm (overridden by checkpoint args when present).",
    )
    p.add_argument(
        "--qkv_bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="QKV bias on attention projections (overridden by checkpoint args when present).",
    )
    p.add_argument(
        "--include_pi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fourier embedder π scaling (overridden by checkpoint args when present).",
    )
    p.add_argument(
        "--point_feats",
        type=int,
        default=6,
        help="Encoder feature channels (overridden by checkpoint args when present).",
    )
    p.add_argument(
        "--pretrained_profile",
        type=str,
        default="none",
        help="Pretrained profile stored in checkpoint (overridden when present).",
    )
    p.add_argument(
        "--include_sharp_label",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="10ch surface layout (overridden by checkpoint args when present).",
    )
    p.add_argument(
        "--drift_threshold",
        type=float,
        default=0.1,
        help="L2 drift threshold for frac_drift_gt_thresh metric (normalised coords).",
    )

    p.add_argument(
        "--gt_source",
        type=str,
        default=None,
        choices=("shapenet", "gobjaverse"),
        help="GT source (default: manifest.json or checkpoint args).",
    )
    p.add_argument(
        "--gobjaverse_render_root",
        type=str,
        default=None,
        help="G-Objaverse render root (default: manifest render_root).",
    )
    p.add_argument(
        "--gobjaverse_num_views",
        type=int,
        default=None,
        help="G-Objaverse views to load (default: checkpoint num_views or all 40).",
    )
    p.add_argument(
        "--num_views",
        type=int,
        default=None,
        help="Views to load (restored from checkpoint when present).",
    )
    p.add_argument(
        "--match_training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="G-Objaverse: evaluate all loaded training views (same pool as "
             "views_per_step). ShapeNet: use canonical+holdout split when disabled.",
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

    surface_seed = args.seed
    include_sharp_label = resolve_include_sharp_label(args)
    train_args0: dict = {}
    if args.checkpoints:
        ckpt0 = torch.load(args.checkpoints[0], map_location="cpu", weights_only=False)
        train_args0 = ckpt0.get("args", {}) if isinstance(ckpt0, dict) else {}
        if train_args0:
            merge_checkpoint_train_args(args, train_args0)
            include_sharp_label = resolve_include_sharp_label(args)
            surface_seed = int(train_args0.get("seed", surface_seed))
            logger.info(
                "Restored arch from checkpoint: point_feats=%s include_sharp_label=%s "
                "pretrained_profile=%s",
                getattr(args, "point_feats", 6),
                include_sharp_label,
                getattr(args, "pretrained_profile", "none"),
            )
            logger.info(
                "Input surfaces from checkpoint: per-mesh seed derived from global seed=%d",
                surface_seed,
            )

    gt_source_arg = getattr(args, "gt_source", None)
    manifest = load_experiment_manifest(args.data_dir)
    effective_gt = (
        gt_source_arg
        or (manifest or {}).get("gt_source")
        or "shapenet"
    ).lower()

    if effective_gt == "gobjaverse":
        train_num_views = getattr(args, "num_views", None)
        if args.gobjaverse_num_views is None and train_num_views is not None:
            args.gobjaverse_num_views = int(train_num_views)
        eval_num_views = (
            args.gobjaverse_num_views
            if args.gobjaverse_num_views is not None
            else GOBJAVERSE_NUM_VIEWS
        )
        precache_full_views = False
        if getattr(args, "match_training", True):
            logger.info(
                "G-Objaverse eval: match_training=True, loading %d views "
                "(training used views_per_step=%s)",
                eval_num_views,
                getattr(args, "views_per_step", None),
            )
    else:
        eval_num_views = args.num_views if args.num_views is not None else 14
        precache_full_views = True
        if getattr(args, "match_training", True):
            args.match_training = False

    dataset = MeshDataset(
        data_dir=args.data_dir,
        pc_size=args.pc_size,
        pc_sharpedge_size=args.pc_sharpedge_size,
        render_height=args.render_height,
        render_width=args.render_width,
        num_views=eval_num_views,
        camera_distance=args.camera_distance,
        elevation_deg=args.elevation_deg,
        max_items=args.max_items,
        mesh_blacklist=args.mesh_blacklist,
        categories=categories,
        precache_full_views=precache_full_views,
        require_cached_gt=args.only_cached_gt,
        seed=surface_seed,
        include_sharp_label=include_sharp_label,
        gt_source=gt_source_arg,
        gobjaverse_render_root=args.gobjaverse_render_root,
        gobjaverse_num_views=args.gobjaverse_num_views,
    )
    logger.info(
        "Dataset: %d meshes in %s (gt_source=%s, loaded_views=%d)",
        len(dataset),
        args.data_dir,
        dataset.gt_source,
        dataset.num_gt_views,
    )

    mesh_paths = list(dataset.mesh_paths)
    if dataset.gt_source == "gobjaverse":
        assert dataset.gobjaverse_gt is not None
        if args.only_cached_gt:
            before = len(mesh_paths)
            mesh_paths = [
                p for p in mesh_paths
                if dataset.gobjaverse_gt.has_gt(p)
            ]
            logger.info(
                "Filtered to %d / %d meshes with G-Objaverse renders (tag '%s')",
                len(mesh_paths),
                before,
                GOBJAVERSE_GT_TAG,
            )
    elif args.only_cached_gt:
        tag = dataset.gt_renderer._tag
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
        print("\n" + "=" * 72)
        print("EVALUATION SUMMARY")
        print("=" * 72)
        hdr = f"{'Checkpoint':<32} {'split':>12} {'psnr':>14} {'ssim':>14} {'lpips':>14} {'l1_depth':>14}"
        print(hdr)
        print("-" * len(hdr))
        for row in summary_rows:
            name = Path(row["checkpoint"]).parent.name + "/" + Path(row["checkpoint"]).stem
            if len(name) > 30:
                name = ".." + name[-28:]
            split_label = row.get("eval_split", "canonical/holdout")
            print(
                f"  {name:<32} "
                f"{split_label:>12} "
                f"{row['psnr']:>14} "
                f"{row['ssim']:>14} "
                f"{row['lpips']:>14} "
                f"{row['l1_depth']:>14}"
            )
        print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
