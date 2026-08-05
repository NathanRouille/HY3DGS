#!/usr/bin/env python3
"""Diagnose vis-vs-eval gap for one overfit object.

Resamples from ckpt_final with:
  - vis settings:  sample_steps=20, guidance=1.0
  - eval settings: sample_steps=50, guidance=1.0 and 3.0
  - null context under both step counts

Exports PLYs + a small metrics.json next to the run.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from evaluate_pc_unite import load_unite_model
from hy3dgen.shapegen.gs_export import export_xyz_pointcloud_ply
from hy3dgen.shapegen.models.autoencoders.shape_pc_ae import ShapePCAE
from hy3dgen.shapegen.pc_losses import chamfer_distance
from hy3dgen.shapegen.pc_render_dataset import (
    build_surface_render_dataset,
    collate_surface_render,
)
from train_gs_ae import load_experiment_manifest
from train_pc_unite import _build_weak_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _cd(pred: torch.Tensor, gt: torch.Tensor) -> float:
    d, _, _ = chamfer_distance(pred, gt)
    return float(d)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default="runs/exp4d_cam_cross_velocity/ckpt_final.pt",
    )
    p.add_argument(
        "--data_dir",
        default="/export/home/nathan/datasets/gobjaverse_experiments/furniture_351/train",
    )
    p.add_argument(
        "--gobjaverse_render_root",
        default="/export/home/nathan/datasets",
    )
    p.add_argument(
        "--vggt_cache_root",
        default="runs/vggt_cache/furn4_view0_cam_cross",
    )
    p.add_argument("--align_mode", default="cross", choices=("cross", "fair_gobK"))
    p.add_argument("--obj_idx", type=int, default=9)
    p.add_argument("--max_items", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output_dir",
        default="runs/exp4d_cam_cross_velocity/diag_vis_vs_eval_obj9",
    )
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model, vggt_builder, train_args = load_unite_model(args.ckpt, device)
    model.eval()
    vggt_builder.eval()
    align_mode = args.align_mode or train_args.get("align_mode", "cross")
    include_sharp = bool(train_args.get("include_sharp_label") or False)

    data_path = Path(args.data_dir).resolve()
    manifest = load_experiment_manifest(str(data_path))
    dataset = build_surface_render_dataset(
        str(data_path),
        max_items=args.max_items,
        use_experiment_manifest=True,
        manifest=manifest,
        render_root=args.gobjaverse_render_root,
        view_idx=int(train_args.get("view_idx", 0)),
        vggt_cache_root=args.vggt_cache_root,
        pc_size=int(train_args.get("pc_size", 5120)),
        pc_sharpedge_size=int(train_args.get("pc_sharpedge_size", 5120)),
        gobjaverse_normalization=not train_args.get("no_gobjaverse_normalization", False),
        surface_in_camera_frame=not train_args.get("no_surface_camera_frame", False),
        align_mode=align_mode,
        include_sharp_label=include_sharp,
    )
    if args.obj_idx < 0 or args.obj_idx >= len(dataset):
        raise SystemExit(f"obj_idx={args.obj_idx} out of range (n={len(dataset)})")

    batch = collate_surface_render([dataset[args.obj_idx]])
    mesh = batch["mesh_path"][0]
    stem = Path(mesh).stem
    logger.info("Object %d: %s  align_mode=%s", args.obj_idx, stem, align_mode)

    surface = batch["surface"].to(device)
    gt_xyz, gt_rgb = ShapePCAE.surface_gt_points(
        surface, include_sharp_label=include_sharp
    )
    weak, keep = _build_weak_context(
        batch, vggt_builder, device, align_mode=align_mode
    )
    assert weak is not None and keep is not None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        z_enc, _ = model.encode(surface)
        xyz_r, rgb_r, _ = model.decode(z_enc, representation_phase=False)
        export_xyz_pointcloud_ply(
            gt_xyz[0].cpu(), out_dir / "gt.ply", colors=gt_rgb[0].cpu()
        )
        export_xyz_pointcloud_ply(
            xyz_r[0].cpu(), out_dir / "recon.ply", colors=rgb_r[0].cpu()
        )

        noise = torch.randn_like(z_enc)
        null = model.null_context(
            1, weak.shape[1], dtype=z_enc.dtype, device=device
        )

        specs = [
            ("gen_vis_steps20_cfg1", weak, keep, 20, 1.0),
            ("gen_null_steps20", null, None, 20, 1.0),
            ("gen_eval_steps50_cfg1", weak, keep, 50, 1.0),
            ("gen_eval_steps50_cfg3", weak, keep, 50, 3.0),
            ("gen_null_steps50", null, None, 50, 1.0),
        ]

        metrics = {
            "mesh": mesh,
            "obj_idx": args.obj_idx,
            "align_mode": align_mode,
            "ckpt": str(Path(args.ckpt).resolve()),
            "seed": args.seed,
            "recon_cd": _cd(xyz_r, gt_xyz),
            "weak_tokens": int(weak.shape[1]),
            "keep_true": int(keep.sum().item()),
        }

        for name, ctx, ctx_keep, steps, gs in specs:
            z_s = model.sample_latents(
                ctx,
                batch_size=1,
                num_steps=steps,
                guidance_scale=gs,
                noise=noise,
                context_keep=ctx_keep,
                device=device,
                dtype=z_enc.dtype,
            )
            xyz, rgb, _ = model.decode(z_s, representation_phase=False)
            export_xyz_pointcloud_ply(
                xyz[0].cpu(), out_dir / f"{name}.ply", colors=rgb[0].cpu()
            )
            metrics[f"{name}_cd"] = _cd(xyz, gt_xyz)
            # proximity to recon (chair) vs null sample
            metrics[f"{name}_cd_to_recon"] = _cd(xyz, xyz_r)
            logger.info(
                "%s: cd_vs_gt=%.5f  cd_vs_recon=%.5f",
                name,
                metrics[f"{name}_cd"],
                metrics[f"{name}_cd_to_recon"],
            )

        # Also dump a second noise draw under vis settings (noise sensitivity).
        noise2 = torch.randn_like(z_enc)
        z2 = model.sample_latents(
            weak,
            batch_size=1,
            num_steps=20,
            guidance_scale=1.0,
            noise=noise2,
            context_keep=keep,
            device=device,
            dtype=z_enc.dtype,
        )
        xyz2, rgb2, _ = model.decode(z2, representation_phase=False)
        export_xyz_pointcloud_ply(
            xyz2[0].cpu(),
            out_dir / "gen_vis_steps20_cfg1_noise2.ply",
            colors=rgb2[0].cpu(),
        )
        metrics["gen_vis_steps20_cfg1_noise2_cd"] = _cd(xyz2, gt_xyz)
        metrics["gen_vis_steps20_cfg1_noise2_cd_to_recon"] = _cd(xyz2, xyz_r)

    # Compare to existing training-vis / eval PLYs if present.
    def _load_xyz(path: Path) -> np.ndarray:
        raw = path.read_text(errors="ignore").splitlines()
        i = next(i for i, l in enumerate(raw) if l.strip() == "end_header") + 1
        pts = []
        for line in raw[i:]:
            sp = line.split()
            if len(sp) >= 3:
                try:
                    pts.append([float(sp[0]), float(sp[1]), float(sp[2])])
                except ValueError:
                    pass
        return np.asarray(pts, dtype=np.float64)

    def _mean_l2(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return float("nan")
        return float(np.mean(np.linalg.norm(a - b, axis=1)))

    run_root = Path(args.ckpt).resolve().parent
    vis_gen = run_root / "vis" / "step_0010000" / "gen.ply"
    eval_gen = run_root / "eval" / "obj_0009_69faf4d3cbd8" / "gen_cfg3.ply"
    new_vis = out_dir / "gen_vis_steps20_cfg1.ply"
    new_eval = out_dir / "gen_eval_steps50_cfg3.ply"
    if vis_gen.is_file() and new_vis.is_file():
        metrics["mean_l2_new_vis_vs_train_vis_gen"] = _mean_l2(
            _load_xyz(new_vis), _load_xyz(vis_gen)
        )
    if eval_gen.is_file() and new_eval.is_file():
        metrics["mean_l2_new_eval50cfg3_vs_old_eval_gen_cfg3"] = _mean_l2(
            _load_xyz(new_eval), _load_xyz(eval_gen)
        )
    if vis_gen.is_file() and new_eval.is_file():
        metrics["mean_l2_new_eval50cfg3_vs_train_vis_gen"] = _mean_l2(
            _load_xyz(new_eval), _load_xyz(vis_gen)
        )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Wrote %s", out_dir)
    logger.info("metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
