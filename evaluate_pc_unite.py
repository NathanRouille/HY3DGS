#!/usr/bin/env python3
"""Evaluate ShapePCUnite: reconstruction, image-only generation, and whether the
VGGT weak context is actually used.

Conditioning ablation (the decisive test): the same ODE noise is decoded with
  real     - the object's own render
  null     - the learned null embedding (unconditional prior)
  swapped  - a *different* object's render
plus a cross-object reference (another object's reconstruction scored against
this object's GT). If ``gen_cd(real)`` is not clearly below ``gen_cd(null)`` and
``gen_cd(swapped)``, the model is sampling a prior and ignoring the image.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

from hy3dgen.shapegen.gs_export import export_xyz_pointcloud_ply
from hy3dgen.shapegen.models.autoencoders.shape_pc_ae import ShapePCAE
from hy3dgen.shapegen.models.autoencoders.shape_pc_unite import ShapePCUnite
from hy3dgen.shapegen.pc_debug_export import export_recon_debug_plys
from hy3dgen.shapegen.pc_losses import chamfer_distance, rgb_l1_on_nn
from hy3dgen.shapegen.pc_render_dataset import build_surface_render_dataset, collate_surface_render
from hy3dgen.shapegen.pretrained_profiles import resolve_include_sharp_label
from hy3dgen.shapegen.vggt_context import VGGTContextBuilder, load_state_dict_skip_mismatch
from train_gs_ae import load_experiment_manifest, resolve_category_ids
from train_pc_unite import _build_weak_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_unite_model(ckpt_path: str, device: torch.device, overrides: Optional[Dict] = None):
    """Rebuild the model from the checkpoint's own args (avoids silent mismatch).

    ``overrides`` lets you force flags that differ from the checkpoint (e.g.
    ``no_surface_camera_frame`` / ``no_gobjaverse_normalization`` for legacy runs).

    Legacy ckpts may omit ``ge.t_embedder.W`` (it used to be a non-persistent
    buffer). Training creates ``W`` under ``torch.manual_seed(args.seed)`` before
    ``ShapePCUnite()``; we replay that seed before constructing the model so
    eval matches training when ``W`` is absent from the file.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = dict(ckpt.get("args", {}))
    for k, v in (overrides or {}).items():
        if v is not None:
            if args.get(k) != v:
                logger.warning("Overriding ckpt arg %s: %r -> %r", k, args.get(k), v)
            args[k] = v
    num_latents = int(args.get("num_latents", 1024))
    num_registers = args.get("num_registers")
    num_registers = int(num_registers) if num_registers is not None else num_latents
    width = int(args.get("width", 1024))
    # Replay training seed so GaussianFourierEmbedding.W matches (critical for
    # old checkpoints that never serialized W).
    seed = args.get("seed", 0)
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    model = ShapePCUnite(
        num_latents=num_latents,
        num_registers=num_registers,
        embed_dim=int(args.get("embed_dim", 64)),
        width=width,
        heads=int(args.get("heads", 16)),
        num_ge_layers=int(args.get("num_ge_layers", 8)),
        num_decoder_layers=int(args.get("num_decoder_layers", 4)),
        pc_size=int(args.get("pc_size", 5120)),
        pc_sharpedge_size=int(args.get("pc_sharpedge_size", 5120)),
        point_feats=int(args.get("point_feats", 6)),
        downsample_ratio=int(args.get("downsample_ratio", 20)),
        num_points_per_anchor=int(args.get("num_points_per_anchor", 8)),
        deterministic_encoder=bool(args.get("deterministic_encoder", True)),
        max_anchor_delta=float(args.get("max_anchor_delta", 0.1)),
        qk_norm=bool(args.get("qk_norm", True)),
        use_rope=bool(args.get("use_rope", False)),
        flow_steps_per_recon=int(args.get("flow_steps_per_recon", 3)),
        flow_loss_type=str(args.get("flow_loss_type", "velocity")),
        modulation_recon_timestep_max=float(args.get("modulation_recon_timestep_max", 0.01)),
        noising_t_start=float(args.get("noising_t_start", 0.7)),
        weak_context_dropout=float(args.get("weak_context_dropout", 0.1)),
    )
    if args.get("freeze_recon"):
        # Checkpoint carries a frozen tokenizer GE snapshot; create the slot first.
        model.freeze_tokenizer_ge()
    ckpt_has_w = any(
        k.endswith("t_embedder.W") for k in ckpt.get("model", {})
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        logger.warning("model: %d missing keys (e.g. %s)", len(missing), missing[:3])
    if not ckpt_has_w:
        logger.warning(
            "Checkpoint has no ge.t_embedder.W; using W from manual_seed(%s) "
            "before model init (must match the seed used at training start).",
            seed,
        )
    model.to(device).eval()

    builder = VGGTContextBuilder(width=width)
    builder.attach_fourier(model.fourier_embedder)
    if "vggt_builder" in ckpt:
        load_state_dict_skip_mismatch(
            builder, ckpt["vggt_builder"], log_prefix="vggt_builder: "
        )
    builder.to(device).eval()
    return model, builder, args


def _decode_and_score(model, z, gt_xyz, gt_rgb):
    xyz, rgb, centers = model.decode(z, representation_phase=False)
    cd, idx_p2t, idx_t2p = chamfer_distance(xyz, gt_xyz)
    rgb_l1 = rgb_l1_on_nn(rgb, gt_rgb, idx_p2t, idx_tgt_to_pred=idx_t2p)
    return xyz, rgb, centers, float(cd), float(rgb_l1)


@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, vggt_builder, train_args = load_unite_model(
        args.ckpt,
        device,
        overrides={
            "no_gobjaverse_normalization": (
                None if args.gobjaverse_normalization is None
                else not args.gobjaverse_normalization
            ),
            "no_surface_camera_frame": (
                None if args.surface_camera_frame is None
                else not args.surface_camera_frame
            ),
        },
    )

    include_sharp = bool(train_args.get("include_sharp_label") or False)
    categories = resolve_category_ids(args.categories)
    data_path = Path(args.data_dir).resolve()
    manifest = load_experiment_manifest(str(data_path)) if not args.no_experiment_manifest else None
    align_mode = args.align_mode or train_args.get("align_mode", "cross")

    # Primary eval stays single-view (default view 0) for comparability with
    # single-view runs, even when the checkpoint was trained multi-view.
    dataset = build_surface_render_dataset(
        str(data_path),
        max_items=args.max_items,
        categories=categories,
        include_sharp_label=include_sharp,
        use_experiment_manifest=not args.no_experiment_manifest,
        manifest=manifest,
        render_root=args.gobjaverse_render_root,
        view_idx=args.view_idx,
        vggt_cache_root=args.vggt_cache_root,
        pc_size=int(train_args.get("pc_size", 5120)),
        pc_sharpedge_size=int(train_args.get("pc_sharpedge_size", 5120)),
        gobjaverse_normalization=not train_args.get("no_gobjaverse_normalization", False),
        surface_in_camera_frame=not train_args.get("no_surface_camera_frame", False),
        align_mode=align_mode,
    )
    logger.info(
        "Eval dataset: %d objects at view_idx=%d (single-view primary)",
        len(dataset),
        args.view_idx,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    n = min(len(dataset), args.max_eval_items or len(dataset))
    if n < 2:
        logger.warning("Need >= 2 objects for swapped/cross-object baselines")

    # Cache per-object tensors once: the ablation needs another object's context.
    items: List[Dict] = []
    for i in range(n):
        batch = collate_surface_render([dataset[i]])
        surface = batch["surface"].to(device)
        gt_xyz, gt_rgb = ShapePCAE.surface_gt_points(surface, include_sharp_label=include_sharp)
        weak, keep = _build_weak_context(
            batch, vggt_builder, device, align_mode=align_mode
        )
        z, fps_xyz = model.encode(surface)
        patch_centers = patch_keep = None
        if "vggt_cache" in batch and batch["vggt_cache"]:
            payload = batch["vggt_cache"][0]
            if isinstance(payload, dict):
                patch_centers = payload.get("patch_centers")
                patch_keep = payload.get("patch_keep")
        items.append(
            {
                "mesh": batch["mesh_path"][0],
                "gt_xyz": gt_xyz,
                "gt_rgb": gt_rgb,
                "weak": weak,
                "keep": keep,
                "z": z,
                "fps_xyz": fps_xyz,
                "noise": torch.randn_like(z),
                "patch_centers": patch_centers,
                "patch_keep": patch_keep,
            }
        )
        logger.info("prepared %d/%d %s", i + 1, n, Path(batch["mesh_path"][0]).stem)

    has_ctx = all(it["weak"] is not None for it in items)
    guidance_scales = args.guidance_scales or [1.0]
    per_object: List[Dict] = []

    for i, it in enumerate(items):
        gt_xyz, gt_rgb, z, noise = it["gt_xyz"], it["gt_rgb"], it["z"], it["noise"]
        rec: Dict[str, object] = {"mesh": it["mesh"]}
        clouds: Dict[str, tuple] = {}

        xyz_r, rgb_r, centers_r, cd_r, rgb_lr = _decode_and_score(model, z, gt_xyz, gt_rgb)
        rec["recon_cd"], rec["recon_rgb"] = cd_r, rgb_lr
        clouds["recon"] = (xyz_r, rgb_r)

        # Reference: another object's reconstruction scored against this GT.
        if n > 1:
            other = items[(i + 1) % n]
            xyz_o, _, _, cd_o, _ = _decode_and_score(model, other["z"], gt_xyz, gt_rgb)
            rec["cross_object_cd"] = cd_o
            clouds["cross_object"] = (xyz_o, None)

        if has_ctx:
            n_tok = it["weak"].shape[1]
            null_ctx = model.null_context(1, n_tok, dtype=z.dtype, device=device)

            z_null = model.sample_latents(
                null_ctx, batch_size=1, num_steps=args.sample_steps,
                guidance_scale=1.0, noise=noise, device=device, dtype=z.dtype,
            )
            xyz_n, rgb_n, _, cd_n, _ = _decode_and_score(model, z_null, gt_xyz, gt_rgb)
            rec["gen_null_cd"] = cd_n
            clouds["gen_null"] = (xyz_n, rgb_n)

            # Partial-noise oracle: start the ODE from the true latent at t_start
            # instead of pure noise. Isolates denoiser quality from the difficulty
            # of sampling from scratch.
            z_orc = model.sample_latents(
                it["weak"], batch_size=1, num_steps=args.sample_steps,
                guidance_scale=1.0, z_init=z, t_start=args.oracle_t_start,
                context_keep=it["keep"],
                device=device, dtype=z.dtype,
            )
            _, _, _, cd_orc, _ = _decode_and_score(model, z_orc, gt_xyz, gt_rgb)
            rec[f"oracle_t{args.oracle_t_start:g}_cd"] = cd_orc

            for gs in guidance_scales:
                tag = f"cfg{gs:g}"
                z_gen = model.sample_latents(
                    it["weak"], batch_size=1, num_steps=args.sample_steps,
                    guidance_scale=gs, noise=noise,
                    context_keep=it["keep"],
                    device=device, dtype=z.dtype,
                )
                xyz_g, rgb_g, _, cd_g, rgb_lg = _decode_and_score(model, z_gen, gt_xyz, gt_rgb)
                rec[f"gen_cd_{tag}"], rec[f"gen_rgb_{tag}"] = cd_g, rgb_lg
                clouds[f"gen_{tag}"] = (xyz_g, rgb_g)

                if n > 1:
                    other = items[(i + 1) % n]
                    z_sw = model.sample_latents(
                        other["weak"], batch_size=1, num_steps=args.sample_steps,
                        guidance_scale=gs, noise=noise,
                        context_keep=other["keep"],
                        device=device, dtype=z.dtype,
                    )
                    xyz_s, rgb_s, _, cd_s, _ = _decode_and_score(model, z_sw, gt_xyz, gt_rgb)
                    rec[f"gen_swapped_cd_{tag}"] = cd_s
                    clouds[f"gen_swapped_{tag}"] = (xyz_s, rgb_s)

        per_object.append(rec)
        logger.info(
            "obj %d/%d %s recon_cd=%.5f gen_cd=%.5f null=%.5f swapped=%.5f cross=%.5f",
            i + 1, n, Path(it["mesh"]).stem,
            rec["recon_cd"],
            rec.get(f"gen_cd_cfg{guidance_scales[0]:g}", float("nan")),
            rec.get("gen_null_cd", float("nan")),
            rec.get(f"gen_swapped_cd_cfg{guidance_scales[0]:g}", float("nan")),
            rec.get("cross_object_cd", float("nan")),
        )

        if args.export_ply:
            obj_dir = out_dir / f"obj_{i:04d}_{Path(it['mesh']).stem[:12]}"
            obj_dir.mkdir(parents=True, exist_ok=True)
            export_xyz_pointcloud_ply(gt_xyz[0].cpu(), obj_dir / "gt.ply", colors=gt_rgb[0].cpu())
            for name, (xyz_c, rgb_c) in clouds.items():
                export_xyz_pointcloud_ply(
                    xyz_c[0].cpu(),
                    obj_dir / f"{name}.ply",
                    colors=rgb_c[0].cpu() if rgb_c is not None else None,
                )
            if args.export_recon_debug:
                export_recon_debug_plys(
                    obj_dir,
                    gt_xyz=gt_xyz,
                    recon_xyz=xyz_r,
                    fps_xyz=it["fps_xyz"],
                    centers=centers_r,
                    num_points_per_anchor=int(model.num_points_per_anchor),
                    max_anchor_delta=float(model.max_anchor_delta),
                    patch_centers=it.get("patch_centers"),
                    patch_keep=it.get("patch_keep"),
                    intruder_thresh=float(args.intruder_thresh),
                )

    keys = sorted({k for r in per_object for k in r if k != "mesh"})
    summary = {
        k: sum(float(r[k]) for r in per_object if k in r) / max(sum(k in r for r in per_object), 1)
        for k in keys
    }
    summary["num_eval"] = n
    summary["sample_steps"] = args.sample_steps
    summary["guidance_scales"] = guidance_scales

    verdict = {}
    for gs in guidance_scales:
        tag = f"cfg{gs:g}"
        gen = summary.get(f"gen_cd_{tag}")
        if gen is None:
            continue
        null = summary.get("gen_null_cd", float("nan"))
        swap = summary.get(f"gen_swapped_cd_{tag}", float("nan"))
        verdict[tag] = {
            "gen_cd": gen,
            "null_gain": null - gen,
            "swap_gain": swap - gen,
            "vs_cross_object": summary.get("cross_object_cd", float("nan")) - gen,
            "conditioning_used": bool(gen < 0.9 * null and gen < 0.9 * swap),
        }
    summary["verdict"] = verdict

    with open(out_dir / "results.json", "w") as f:
        json.dump({"summary": summary, "per_object": per_object}, f, indent=2)
    logger.info("Summary: %s", json.dumps(summary, indent=2))
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ShapePCUnite")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="runs/pc_unite_eval")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--max_eval_items", type=int, default=None)
    p.add_argument("--categories", type=str, default=None)
    p.add_argument("--no_experiment_manifest", action="store_true")
    p.add_argument("--gobjaverse_render_root", type=str, default=None)
    p.add_argument("--vggt_cache_root", type=str, default=None)
    p.add_argument(
        "--view_idx",
        type=int,
        default=0,
        help="Primary eval view (single-view; default 0 even for multi-view-trained ckpts).",
    )
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument(
        "--guidance_scales",
        type=float,
        nargs="*",
        default=[1.0],
        help="CFG scales to evaluate; 1.0 disables guidance.",
    )
    p.add_argument(
        "--oracle_t_start",
        type=float,
        default=0.5,
        help="Partial-noise oracle: ODE starts from the true latent noised to this t.",
    )
    p.add_argument("--export_ply", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--export_recon_debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also write fps/anchors/recon_error/gt_uncovered/intruders/"
            "locals_by_anchor/delta_mag/patch_centers PLYs (needs --export_ply)."
        ),
    )
    p.add_argument(
        "--intruder_thresh",
        type=float,
        default=0.02,
        help="Recon points farther than this from GT go into intruders.ply.",
    )
    p.add_argument(
        "--gobjaverse_normalization", action=argparse.BooleanOptionalAction, default=None
    )
    p.add_argument(
        "--surface_camera_frame", action=argparse.BooleanOptionalAction, default=None
    )
    p.add_argument(
        "--align_mode",
        type=str,
        default=None,
        choices=("cross", "fair_gobK"),
        help="Override ckpt align_mode (default: use train args from checkpoint).",
    )
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
