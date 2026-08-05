#!/usr/bin/env python3
"""Multi-noise conditioning probe for ShapePCUnite.

For selected objects, draw many ODE noises and score:
  gen  = sample with real VGGT context
  null = sample with null context (same noise)

Exports summary CSV/JSON + a few representative PLYs (best/median/worst gen).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List

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
    return float(chamfer_distance(pred, gt)[0])


def _percentile(xs: List[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/exp4d_cam_cross_velocity/ckpt_final.pt")
    p.add_argument(
        "--data_dir",
        default="/export/home/nathan/datasets/gobjaverse_experiments/furniture_351/train",
    )
    p.add_argument("--gobjaverse_render_root", default="/export/home/nathan/datasets")
    p.add_argument(
        "--vggt_cache_root", default="runs/vggt_cache/furn4_view0_cam_cross"
    )
    p.add_argument("--align_mode", default="cross", choices=("cross", "fair_gobK"))
    p.add_argument("--max_items", type=int, default=32)
    p.add_argument(
        "--obj_idxs",
        type=int,
        nargs="+",
        default=[9, 0, 1, 23],
        help="Dataset indices to probe.",
    )
    p.add_argument("--n_noise", type=int, default=64)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed0", type=int, default=0, help="Base seed; noise i uses seed0+i.")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output_dir",
        default="runs/exp4d_cam_cross_velocity/diag_noise_sweep",
    )
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, builder, train_args = load_unite_model(args.ckpt, device)
    model.eval()
    builder.eval()

    include_sharp = bool(train_args.get("include_sharp_label") or False)
    align_mode = args.align_mode
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
        gobjaverse_normalization=not train_args.get(
            "no_gobjaverse_normalization", False
        ),
        surface_in_camera_frame=not train_args.get("no_surface_camera_frame", False),
        align_mode=align_mode,
        include_sharp_label=include_sharp,
    )

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary = {
        "ckpt": str(Path(args.ckpt).resolve()),
        "n_noise": args.n_noise,
        "sample_steps": args.sample_steps,
        "guidance_scale": args.guidance_scale,
        "seed0": args.seed0,
        "align_mode": align_mode,
        "objects": {},
    }

    for obj_idx in args.obj_idxs:
        if obj_idx < 0 or obj_idx >= len(dataset):
            logger.warning("skip obj_idx=%d (n=%d)", obj_idx, len(dataset))
            continue

        batch = collate_surface_render([dataset[obj_idx]])
        mesh = batch["mesh_path"][0]
        stem = Path(mesh).stem
        obj_dir = out_root / f"obj_{obj_idx:04d}_{stem[:12]}"
        obj_dir.mkdir(parents=True, exist_ok=True)

        surface = batch["surface"].to(device)
        gt_xyz, gt_rgb = ShapePCAE.surface_gt_points(
            surface, include_sharp_label=include_sharp
        )
        weak, keep = _build_weak_context(
            batch, builder, device, align_mode=align_mode
        )
        assert weak is not None and keep is not None

        with torch.no_grad():
            z_enc, _ = model.encode(surface)
            xyz_r, rgb_r, _ = model.decode(z_enc, representation_phase=False)
            recon_cd = _cd(xyz_r, gt_xyz)
            export_xyz_pointcloud_ply(
                gt_xyz[0].cpu(), obj_dir / "gt.ply", colors=gt_rgb[0].cpu()
            )
            export_xyz_pointcloud_ply(
                xyz_r[0].cpu(), obj_dir / "recon.ply", colors=rgb_r[0].cpu()
            )

            null = model.null_context(
                1, weak.shape[1], dtype=z_enc.dtype, device=device
            )

            rows = []
            best = med = worst = None  # (gen_cd, seed, xyz_g, rgb_g, xyz_n, rgb_n)

            for i in range(args.n_noise):
                seed = args.seed0 + i
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                noise = torch.randn_like(z_enc)

                z_g = model.sample_latents(
                    weak,
                    batch_size=1,
                    num_steps=args.sample_steps,
                    guidance_scale=args.guidance_scale,
                    noise=noise,
                    context_keep=keep,
                    device=device,
                    dtype=z_enc.dtype,
                )
                z_n = model.sample_latents(
                    null,
                    batch_size=1,
                    num_steps=args.sample_steps,
                    guidance_scale=1.0,
                    noise=noise,
                    device=device,
                    dtype=z_enc.dtype,
                )
                xyz_g, rgb_g, _ = model.decode(z_g, representation_phase=False)
                xyz_n, rgb_n, _ = model.decode(z_n, representation_phase=False)
                gen_cd = _cd(xyz_g, gt_xyz)
                null_cd = _cd(xyz_n, gt_xyz)
                gen_to_recon = _cd(xyz_g, xyz_r)
                row = {
                    "obj_idx": obj_idx,
                    "stem": stem,
                    "seed": seed,
                    "gen_cd": gen_cd,
                    "null_cd": null_cd,
                    "gain": null_cd - gen_cd,
                    "gen_to_recon_cd": gen_to_recon,
                    "recon_cd": recon_cd,
                }
                rows.append(row)
                all_rows.append(row)

                pack = (gen_cd, seed, xyz_g, rgb_g, xyz_n, rgb_n)
                if best is None or gen_cd < best[0]:
                    best = pack
                if worst is None or gen_cd > worst[0]:
                    worst = pack

                if (i + 1) % 8 == 0 or i == 0:
                    logger.info(
                        "obj %d %s  %d/%d  seed=%d gen=%.4f null=%.4f gain=%.4f",
                        obj_idx,
                        stem[:12],
                        i + 1,
                        args.n_noise,
                        seed,
                        gen_cd,
                        null_cd,
                        null_cd - gen_cd,
                    )

            # median by gen_cd
            rows_sorted = sorted(rows, key=lambda r: r["gen_cd"])
            med_seed = rows_sorted[len(rows_sorted) // 2]["seed"]
            # re-run median seed for PLY (cheaper than storing all)
            torch.manual_seed(med_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(med_seed)
            noise = torch.randn_like(z_enc)
            z_g = model.sample_latents(
                weak,
                batch_size=1,
                num_steps=args.sample_steps,
                guidance_scale=args.guidance_scale,
                noise=noise,
                context_keep=keep,
                device=device,
                dtype=z_enc.dtype,
            )
            z_n = model.sample_latents(
                null,
                batch_size=1,
                num_steps=args.sample_steps,
                guidance_scale=1.0,
                noise=noise,
                device=device,
                dtype=z_enc.dtype,
            )
            xyz_g_m, rgb_g_m, _ = model.decode(z_g, representation_phase=False)
            xyz_n_m, rgb_n_m, _ = model.decode(z_n, representation_phase=False)

            def _dump(tag, xyz, rgb):
                export_xyz_pointcloud_ply(
                    xyz[0].cpu(), obj_dir / f"{tag}.ply", colors=rgb[0].cpu()
                )

            _dump(f"gen_best_seed{best[1]}", best[2], best[3])
            _dump(f"null_best_seed{best[1]}", best[4], best[5])
            _dump(f"gen_worst_seed{worst[1]}", worst[2], worst[3])
            _dump(f"null_worst_seed{worst[1]}", worst[4], worst[5])
            _dump(f"gen_median_seed{med_seed}", xyz_g_m, rgb_g_m)
            _dump(f"null_median_seed{med_seed}", xyz_n_m, rgb_n_m)

            gens = [r["gen_cd"] for r in rows]
            nulls = [r["null_cd"] for r in rows]
            gains = [r["gain"] for r in rows]
            frac_gain_pos = float(np.mean([g > 0 for g in gains]))
            frac_gain_gt005 = float(np.mean([g > 0.05 for g in gains]))
            frac_gen_near_recon = float(np.mean([g < 3 * recon_cd + 0.01 for g in gens]))
            frac_gen_lt_005 = float(np.mean([g < 0.05 for g in gens]))
            frac_gen_lt_002 = float(np.mean([g < 0.02 for g in gens]))

            obj_sum = {
                "mesh": mesh,
                "obj_idx": obj_idx,
                "recon_cd": recon_cd,
                "gen_cd_mean": float(np.mean(gens)),
                "gen_cd_std": float(np.std(gens)),
                "gen_cd_min": float(np.min(gens)),
                "gen_cd_p50": _percentile(gens, 50),
                "gen_cd_p90": _percentile(gens, 90),
                "gen_cd_max": float(np.max(gens)),
                "null_cd_mean": float(np.mean(nulls)),
                "null_cd_std": float(np.std(nulls)),
                "null_cd_min": float(np.min(nulls)),
                "null_cd_p50": _percentile(nulls, 50),
                "gain_mean": float(np.mean(gains)),
                "gain_std": float(np.std(gains)),
                "frac_gain_positive": frac_gain_pos,
                "frac_gain_gt_0.05": frac_gain_gt005,
                "frac_gen_cd_lt_0.02": frac_gen_lt_002,
                "frac_gen_cd_lt_0.05": frac_gen_lt_005,
                "frac_gen_near_recon": frac_gen_near_recon,
                "best_seed": int(best[1]),
                "worst_seed": int(worst[1]),
                "median_seed": int(med_seed),
            }
            summary["objects"][str(obj_idx)] = obj_sum

            with open(obj_dir / "per_noise.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            with open(obj_dir / "summary.json", "w") as f:
                json.dump(obj_sum, f, indent=2)

            logger.info(
                "OBJ %d %s summary: gen mean/p50/min/max=%.4f/%.4f/%.4f/%.4f  "
                "null mean=%.4f  gain mean=%.4f  frac_gen<0.02=%.2f  frac_gain>0=%.2f",
                obj_idx,
                stem[:12],
                obj_sum["gen_cd_mean"],
                obj_sum["gen_cd_p50"],
                obj_sum["gen_cd_min"],
                obj_sum["gen_cd_max"],
                obj_sum["null_cd_mean"],
                obj_sum["gain_mean"],
                obj_sum["frac_gen_cd_lt_0.02"],
                obj_sum["frac_gain_positive"],
            )

    with open(out_root / "all_per_noise.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done → %s", out_root)


if __name__ == "__main__":
    main()
