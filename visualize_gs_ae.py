#!/usr/bin/env python3
"""Visualize ShapeGSAE vs GT (val/train). Run from repo root.

Uses the same ``MeshDataset`` layout as ``train_gs_ae.py`` (per-model folders
with ``model_normalized.obj``). Optional ``--categories chair`` filters by
ShapeNet synset prefix; ``--only_cached_gt`` accepts either the exact GT cache
for your view config or the 4-view ``normv2`` superset cache (same as training).

Example (chairs, 1 view, 256² — reuses 4-view precache):

    python visualize_gs_ae.py \\
        --checkpoint runs/p1_baseline_1v/ckpt_020000.pt \\
        --data_dir ~/datasets/shapenet_prepared/val \\
        --categories chair \\
        --render_height 256 --render_width 256 \\
        --num_views 1 --camera_azimuths "0" \\
        --num_samples 8
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Set

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm  # noqa: E402
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hy3dgen.shapegen.gs_renderer import GaussianRenderer  # noqa: E402
from hy3dgen.shapegen.models.autoencoders.model import ShapeGSAE  # noqa: E402
from train_gs_ae import (  # noqa: E402
    MeshDataset,
    mesh_path_has_usable_gt_cache,
    resolve_category_ids,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
    norm = np.zeros_like(d) if vmax <= vmin else np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = _get_turbo()
    rgba = cmap(norm)
    rgb = (rgba[..., :3] * 255.0 * m[..., None]).astype(np.uint8)
    return rgb


def _rgb01_to_u8(rgb: torch.Tensor) -> np.ndarray:
    x = rgb.detach().clamp(0, 1).float().cpu().numpy()
    return (x * 255.0).round().astype(np.uint8)


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


def _parse_azimuths(s: Optional[str], num_views: int) -> Optional[List[float]]:
    if not s:
        return None
    vals = [float(a.strip()) for a in s.split(",")]
    if len(vals) != num_views:
        raise ValueError(
            f"--camera_azimuths has {len(vals)} values but num_views={num_views}"
        )
    return vals


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[0]
    p = argparse.ArgumentParser(description="Visualize ShapeGSAE vs GT RGBD")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", default=str(repo / "runs/safe_glbs_val153"))
    p.add_argument("--output_dir", default=str(repo / "runs/gs_ae_viz_val"))

    p.add_argument("--num_latents", type=int, default=2048)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--num_encoder_layers", type=int, default=8)
    p.add_argument("--num_decoder_layers", type=int, default=8)
    p.add_argument("--pc_size", type=int, default=5120)
    p.add_argument("--pc_sharpedge_size", type=int, default=5120)
    p.add_argument("--downsample_ratio", type=int, default=20)

    # Default 128 matches typical val precache: h128w128az0_90_normv1.pt
    p.add_argument("--render_height", type=int, default=512)
    p.add_argument("--render_width", type=int, default=512)
    p.add_argument("--num_views", type=int, default=2)
    p.add_argument("--camera_distance", type=float, default=3.5)
    p.add_argument("--elevation_deg", type=float, default=20.0)
    p.add_argument("--camera_azimuths", type=str, default="0,90")

    p.add_argument(
        "--gt_view_layout",
        type=str,
        default="legacy",
        choices=("legacy", "v46"),
        help="Must match the layout used to pre-cache GT.",
    )

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--only_cached_gt", action="store_true", default=True)
    p.add_argument("--no_only_cached_gt", action="store_false", dest="only_cached_gt")
    p.add_argument("--mesh_blacklist", type=str, default=None)
    p.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated ShapeNet category names or 8-digit synset IDs "
        "(same as train_gs_ae.py --categories), e.g. 'chair' or '03001627'.",
    )
    p.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Cap dataset size after category filter (same semantics as training).",
    )

    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--indices", type=str, default=None)
    p.add_argument("--shuffle", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    azimuths = None
    if str(args.gt_view_layout).lower() != "v46":
        azimuths = _parse_azimuths(args.camera_azimuths, args.num_views)
    categories: Optional[Set[str]] = resolve_category_ids(args.categories)
    if categories is not None:
        logger.info("Category filter: %s", sorted(categories))

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
    tag = dataset.gt_renderer._tag
    az_list = dataset.gt_renderer.azimuths_deg or [0.0]
    mesh_paths = list(dataset.mesh_paths)
    if args.only_cached_gt:
        mesh_paths = [
            p
            for p in mesh_paths
            if mesh_path_has_usable_gt_cache(
                p, tag, args.render_height, args.render_width, az_list
            )
        ]
        logger.info(
            "GT usable from disk (exact tag '%s' or 4-view normv2 slice): "
            "%d mesh(es)",
            tag,
            len(mesh_paths),
        )
        if not mesh_paths:
            raise RuntimeError(
                "No usable GT cache on disk for this view/H×W config. "
                "Run prepare_shapenet.py --precache with matching settings, "
                "or pass --no_only_cached_gt to render on the fly."
            )
    if args.shuffle:
        random.shuffle(mesh_paths)

    dataset.mesh_paths = mesh_paths

    if args.indices:
        ix = [int(x.strip()) for x in args.indices.split(",")]
        selected = [mesh_paths[i] for i in ix]
    else:
        selected = mesh_paths[: min(args.num_samples, len(mesh_paths))]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    renderer = GaussianRenderer(
        height=args.render_height,
        width=args.render_width,
        render_depth=True,
    ).to(device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for si, mesh_path in enumerate(selected):
        try:
            surface = dataset.loader(mesh_path).squeeze(0).to(device)
            rgbs, depths, c2ws, _vp = dataset.gt_renderer.get_or_render(mesh_path)
        except Exception as e:
            logger.warning("Skip %s: %s", mesh_path, e)
            continue

        means, scales, rotations, opacities, colors = model(surface.unsqueeze(0))
        means = means[0]
        scales = scales[0]
        rotations = rotations[0]
        opacities = opacities[0]
        colors = colors[0]

        rows: List[Image.Image] = []
        for vi, c2w in enumerate(c2ws):
            out = renderer(means, scales, rotations, opacities, colors, c2w.to(device))
            gt_rgb, gt_dep = rgbs[vi], depths[vi]
            rows.append(
                _hconcat(
                    [
                        Image.fromarray(_rgb01_to_u8(gt_rgb)),
                        Image.fromarray(_rgb01_to_u8(out["rgb"])),
                        Image.fromarray(_depth_to_rgb_u8(gt_dep.cpu())),
                        Image.fromarray(_depth_to_rgb_u8(out["depth"].cpu())),
                    ]
                )
            )
        stem = Path(mesh_path).stem
        out_path = out_dir / f"sample_{si:03d}_{stem}.png"
        _vconcat(rows).save(out_path)
        logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()