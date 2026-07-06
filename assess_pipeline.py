#!/usr/bin/env python3
"""One-stop, byte-accurate audit of the G-Objaverse -> 3DGS training pipeline.

Motivation
----------
Colors of the *same* object look different depending on where you look:
CloudCompare, macOS Preview, the training "visuals" grid, the diagnostic
script... Every one of those viewers applies its own shading / color
management, so eyeballing them side by side is unreliable.

This script renders everything through a *single* code path (matplotlib,
raw sRGB bytes, no lighting, no OS color management) so the panels are
directly, numerically comparable:

  Column 1: Raw GT PNG straight off disk (what G-Objaverse rendered)
  Column 2: GT tensor as loaded by the dataset (what the training loss sees)
  Column 3: Model prediction from the checkpoint (actual Gaussian render)
  Column 4: Mesh, software-rasterized with UV/vertex-color lookup (same
            texture source used to build the input point cloud)
  Column 5: Input point cloud (same tensor fed to the encoder), splatted
            into the same camera view

Below each panel: mean RGB and PSNR-vs-raw-GT, so "which one is right" is a
number, not a vibe.

Usage
-----
    python assess_pipeline.py \\
        --checkpoint runs/train/<run>/ckpt_020000.pt \\
        --data_dir /path/to/furniture_351/train \\
        --mesh_stems 69faf4d3 b4fb2d8b 33b463c9 \\
        --view_idx 0 \\
        --out_dir pipeline_audit

Omit --mesh_stems to run over every mesh in --data_dir (writes a summary
CSV ranking objects by pred-vs-GT PSNR so you can see exactly which ones the
model gets right and which it doesn't).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_gs_ae import load_model
from hy3dgen.shapegen.gobjaverse_gt import (
    intrinsics_from_meta,
    read_gobjaverse_view_meta,
    unity_c2w_from_meta,
    unity_c2w_to_opengl,
)
from hy3dgen.shapegen.gs_export import surface_rgb_slice
from hy3dgen.shapegen.gs_renderer import GaussianRenderer
from hy3dgen.shapegen.pretrained_profiles import resolve_include_sharp_label
from hy3dgen.shapegen.surface_loaders import scene_to_geometry
from train_gs_ae import GTRGBDRenderer, MeshDataset, load_experiment_manifest

from diagnose_gobjaverse_data import (
    extract_mesh_texture as extract_mesh_texture_bgr,
    project_vertices_to_image,
    software_rasterize_textured,
    software_rasterize_vertex_color,
)
from hy3dgen.shapegen.surface_loaders import _get_vertex_colors
from hy3dgen.shapegen.gobjaverse_gt import normalize_mesh_gobjaverse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_args_from_ckpt(train_args: dict) -> SimpleNamespace:
    """Reconstruct only what MeshDataset / ShapeGSAE need, straight from the checkpoint."""
    defaults = dict(
        pc_size=5120, pc_sharpedge_size=5120, render_height=512, render_width=512,
        camera_distance=3.5, elevation_deg=20.0, max_items=None, mesh_blacklist=None,
        seed=42, include_sharp_label=True, gobjaverse_render_root=None,
        gobjaverse_num_views=None, num_views=None,
        num_latents=512, embed_dim=64, width=1024, heads=16,
        num_decoder_layers=16, num_encoder_layers=8, downsample_ratio=20,
        num_gs_per_anchor=10, point_feats=7, qk_norm=False,
        deterministic_encoder=False, max_anchor_delta=None, sh_degree=0,
        pretrained_profile="hunyuan_mini",
    )
    merged = dict(defaults)
    merged.update({k: v for k, v in train_args.items() if v is not None})
    return SimpleNamespace(**merged)


def rgb01_to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def splat_pointcloud(
    xyz: np.ndarray, rgb: np.ndarray, c2w_unity: np.ndarray,
    fx: float, fy: float, cx: float, cy: float, H: int, W: int, radius: int = 2,
) -> np.ndarray:
    """Render a colored point cloud as filled dots (no lighting, raw RGB)."""
    uv, z, in_frame = project_vertices_to_image(xyz, c2w_unity, fx, fy, cx, cy, H, W)
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    order = np.argsort(-z)  # paint far points first so near ones win
    for i in order:
        if not in_frame[i]:
            continue
        u, v = int(round(uv[i, 0])), int(round(uv[i, 1]))
        color = tuple(int(c) for c in rgb01_to_u8(rgb[i])[::-1])  # BGR for cv2
        cv2.circle(canvas, (u, v), radius, color, -1)
    return canvas[..., ::-1]  # back to RGB


def label_panel(ax, img: np.ndarray, title: str, subtitle: str = "") -> None:
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    if subtitle:
        ax.text(
            0.5, -0.06, subtitle, transform=ax.transAxes,
            ha="center", va="top", fontsize=8,
        )
    ax.axis("off")


def assess_one(
    dataset: MeshDataset,
    idx: int,
    model,
    renderer: GaussianRenderer,
    device: torch.device,
    view_idx: int,
    out_dir: Path,
    include_sharp_label: bool,
    H: int,
    W: int,
) -> Dict:
    sample = dataset[idx]
    mesh_path = sample["mesh_path"]
    mesh_stem = Path(mesh_path).stem

    # --- 1. Raw GT PNG straight off disk -----------------------------------
    render_dir = dataset.gobjaverse_gt.render_dir_for_mesh(mesh_path)
    view_json = render_dir / f"{view_idx:05d}" / f"{view_idx:05d}.json"
    meta = read_gobjaverse_view_meta(view_json)
    raw_png_path = render_dir / f"{view_idx:05d}" / f"{view_idx:05d}.png"
    raw_bgr = cv2.imread(str(raw_png_path), cv2.IMREAD_UNCHANGED)
    if raw_bgr.shape[-1] == 4:
        alpha = raw_bgr[..., 3:4].astype(np.float32) / 255.0
        raw_rgb = (raw_bgr[..., :3][..., ::-1].astype(np.float32) / 255.0) * alpha + (1 - alpha)
    else:
        raw_rgb = raw_bgr[..., ::-1].astype(np.float32) / 255.0
    if raw_rgb.shape[:2] != (H, W):
        raw_rgb = cv2.resize(raw_rgb, (W, H), interpolation=cv2.INTER_AREA)

    # --- 2. GT tensor as loaded by the dataset (what the loss sees) --------
    gt_tensor_rgb = sample["rgbs"][view_idx].numpy()

    # --- 3. Model prediction -------------------------------------------------
    surface = sample["surface"].unsqueeze(0).to(device)
    with torch.no_grad():
        latents, query_positions = model.encode(surface)
        means, scales, rotations, opacities, sh_coeffs = model.decode(latents, query_positions)
        c2w = sample["c2ws"][view_idx].to(device)
        vp = sample.get("view_params")
        vp = vp[view_idx] if vp is not None else None
        out = renderer(
            means[0], scales[0], rotations[0], opacities[0], sh_coeffs[0], c2w,
            fx=vp.get("fx") if vp else None, fy=vp.get("fy") if vp else None,
            cx=vp.get("cx") if vp else None, cy=vp.get("cy") if vp else None,
        )
    pred_rgb = out["rgb"].cpu().numpy()

    # --- 4. Mesh software-rasterized (same texture source as point cloud) --
    raw_mesh = GTRGBDRenderer._load_mesh(mesh_path)
    mesh_geo = scene_to_geometry(raw_mesh)
    mesh_norm = normalize_mesh_gobjaverse(mesh_geo, meta)
    c2w_unity = unity_c2w_from_meta(meta)
    fx, fy, cx, cy = intrinsics_from_meta(meta, H, W)

    uvs_bgr, tex_bgr = extract_mesh_texture_bgr(mesh_norm)
    has_uv = uvs_bgr is not None
    if has_uv:
        mesh_render_bgr, _ = software_rasterize_textured(
            mesh_norm.vertices, mesh_norm.faces, uvs_bgr, tex_bgr,
            c2w_unity, fx, fy, cx, cy, H, W,
        )
    else:
        face_colors = _get_vertex_colors(mesh_norm)[mesh_norm.faces].mean(axis=1)
        mesh_render_bgr, _ = software_rasterize_vertex_color(
            mesh_norm.vertices, mesh_norm.faces, face_colors,
            c2w_unity, fx, fy, cx, cy, H, W,
        )
    mesh_render_rgb = mesh_render_bgr[..., ::-1].astype(np.float32) / 255.0

    # --- 5. Input point cloud (exact tensor fed to the encoder) -------------
    surf_np = sample["surface"].numpy()
    rgb_slice = surface_rgb_slice(surf_np.shape[-1], include_sharp_label=include_sharp_label)
    pc_xyz = surf_np[:, :3].astype(np.float64)
    pc_rgb = surf_np[:, rgb_slice].astype(np.float32)
    pc_render_rgb = splat_pointcloud(pc_xyz, pc_rgb, c2w_unity, fx, fy, cx, cy, H, W).astype(np.float32) / 255.0

    # --- Metrics --------------------------------------------------------------
    metrics = {
        "mesh_stem": mesh_stem,
        "has_uv_texture": has_uv,
        "raw_png_mean_rgb": raw_rgb.reshape(-1, 3).mean(0).tolist(),
        "dataset_gt_mean_rgb": gt_tensor_rgb.reshape(-1, 3).mean(0).tolist(),
        "pred_mean_rgb": pred_rgb.reshape(-1, 3).mean(0).tolist(),
        "mesh_render_mean_rgb": mesh_render_rgb.reshape(-1, 3).mean(0).tolist(),
        "pointcloud_mean_rgb": pc_rgb.mean(0).tolist(),
        "psnr_dataset_gt_vs_raw_png": psnr(gt_tensor_rgb, raw_rgb),
        "psnr_pred_vs_raw_png": psnr(pred_rgb, raw_rgb),
        "psnr_mesh_render_vs_raw_png": psnr(mesh_render_rgb, raw_rgb),
        "psnr_pointcloud_vs_raw_png": psnr(pc_render_rgb, raw_rgb),
    }

    # --- Panel -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.6))
    label_panel(axes[0], rgb01_to_u8(raw_rgb), "1. Raw GT PNG (disk)",
                f"mean={np.round(metrics['raw_png_mean_rgb'], 3)}")
    label_panel(axes[1], rgb01_to_u8(gt_tensor_rgb), "2. Dataset GT tensor (loss target)",
                f"PSNR vs (1)={metrics['psnr_dataset_gt_vs_raw_png']:.1f}dB")
    label_panel(axes[2], rgb01_to_u8(pred_rgb), "3. Model prediction (checkpoint)",
                f"PSNR vs (1)={metrics['psnr_pred_vs_raw_png']:.1f}dB")
    label_panel(axes[3], rgb01_to_u8(mesh_render_rgb), "4. Mesh UV/vertex render",
                f"PSNR vs (1)={metrics['psnr_mesh_render_vs_raw_png']:.1f}dB  uv={has_uv}")
    label_panel(axes[4], rgb01_to_u8(pc_render_rgb), "5. Input point cloud (splatted)",
                f"PSNR vs (1)={metrics['psnr_pointcloud_vs_raw_png']:.1f}dB")
    fig.suptitle(f"{mesh_stem}  |  view {view_idx:02d}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{mesh_stem}_view{view_idx:02d}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Saved %s", out_path)

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--mesh_stems", type=str, nargs="*", default=None,
                     help="Substrings of mesh filenames to assess; omit for all.")
    ap.add_argument("--view_idx", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="pipeline_audit")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if hasattr(train_args, "__dict__"):
        train_args = vars(train_args)
    ns = build_args_from_ckpt(train_args)
    include_sharp_label = resolve_include_sharp_label(ns)

    manifest = load_experiment_manifest(args.data_dir)
    gt_source = (train_args.get("gt_source") or (manifest or {}).get("gt_source") or "gobjaverse")

    dataset = MeshDataset(
        data_dir=args.data_dir,
        pc_size=ns.pc_size,
        pc_sharpedge_size=ns.pc_sharpedge_size,
        render_height=ns.render_height,
        render_width=ns.render_width,
        num_views=ns.gobjaverse_num_views or train_args.get("num_views") or 40,
        camera_distance=ns.camera_distance,
        elevation_deg=ns.elevation_deg,
        max_items=ns.max_items,
        mesh_blacklist=ns.mesh_blacklist,
        categories=None,
        precache_full_views=False,
        require_cached_gt=True,
        seed=ns.seed,
        include_sharp_label=include_sharp_label,
        gt_source=gt_source,
        gobjaverse_render_root=ns.gobjaverse_render_root,
        gobjaverse_num_views=ns.gobjaverse_num_views,
    )
    logger.info("Dataset has %d meshes", len(dataset))

    model = load_model(args.checkpoint, ns, device)
    renderer = GaussianRenderer(
        height=ns.render_height, width=ns.render_width,
        render_depth=True, sh_degree=model.sh_degree,
    ).to(device)

    if args.mesh_stems:
        indices = []
        for stem in args.mesh_stems:
            matches = [i for i, p in enumerate(dataset.mesh_paths) if stem in p]
            if not matches:
                logger.warning("No mesh matches stem %r", stem)
            indices.extend(matches)
    else:
        indices = list(range(len(dataset)))

    all_metrics: List[Dict] = []
    for idx in indices:
        try:
            m = assess_one(
                dataset, idx, model, renderer, device, args.view_idx, out_dir,
                include_sharp_label, ns.render_height, ns.render_width,
            )
            all_metrics.append(m)
        except Exception as e:
            logger.exception("Failed on index %d (%s): %s", idx, dataset.mesh_paths[idx], e)

    if not all_metrics:
        logger.warning("No objects assessed.")
        return

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "mesh_stem", "has_uv_texture",
            "psnr_dataset_gt_vs_raw_png", "psnr_pred_vs_raw_png",
            "psnr_mesh_render_vs_raw_png", "psnr_pointcloud_vs_raw_png",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in sorted(all_metrics, key=lambda x: x["psnr_pred_vs_raw_png"]):
            w.writerow({k: m[k] for k in fieldnames})
    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("\n%s", "=" * 78)
    logger.info("SUMMARY (%d objects) — sorted by model prediction quality (worst first)", len(all_metrics))
    logger.info("%s", "=" * 78)
    for m in sorted(all_metrics, key=lambda x: x["psnr_pred_vs_raw_png"]):
        logger.info(
            "%-40s  pred=%5.1fdB  data(gt_tensor)=%5.1fdB  mesh_render=%5.1fdB  "
            "pointcloud=%5.1fdB  uv=%s",
            m["mesh_stem"], m["psnr_pred_vs_raw_png"], m["psnr_dataset_gt_vs_raw_png"],
            m["psnr_mesh_render_vs_raw_png"], m["psnr_pointcloud_vs_raw_png"],
            m["has_uv_texture"],
        )
    logger.info("Wrote %s and per-object panels to %s", csv_path, out_dir)


if __name__ == "__main__":
    main()
