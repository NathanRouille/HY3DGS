#!/usr/bin/env python3
"""Debug suite for ShapePCUnite gen/recon gap (no retraining).

Stages (selectable via ``--stages``):

  data_check
      Export camera-frame PLYs for a curated object set (worst/best gen +
      random): GT xyz/rgb/nx-ny-nz, aligned patch centres, conditioning
      view image, FPS, anchors, recon, gen.

  diffusion_curve
      For many ``t_start`` values: (1) oracle ODE from true latent noised to
      ``t``, (2) analytic velocity error on the true linear path at ``t``.
      Localizes early vs late denoising failure.

  encode_vggt
      Encode with vs without VGGT weak context (tokenizer ablation).  Shows
      whether the tokenizer actually depends on image tokens.

  latent_align
      Gen vs tokenizer latent distance, mean-latent CD, sigma-sweep decisive
      test (isotropic vs structured flow error).

  pca_trajectories
      Fix (z, ε); for many t_start run ODE with waypoints; compare linear bridge
      vs ODE paths in 2D PCA (multi object × latent draws × noises). Plots +
      JSON metrics (endpoint distance, path curvature, basin clustering).

  cfg_sweep
      Gen CD at several guidance scales (eval-only; no retrain).

  multiview_eval
      Same objects, several train views — is a val/train failure view-specific?

Usage::

    python debug_pc_unite.py \\
        --ckpt runs/exp8_.../ckpt_0020000.pt \\
        --eval_results runs/exp8_.../eval_ckpt20k/results.json \\
        --data_dir ... --vggt_cache_root ... --gobjaverse_render_root ... \\
        --output_dir runs/debug/exp8_ckpt20k \\
        --stages data_check diffusion_curve encode_vggt latent_align \\
                 pca_trajectories cfg_sweep multiview_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from hy3dgen.shapegen.cam_align import aligned_centers_from_payload
from hy3dgen.shapegen.gs_export import export_input_surface_ply, export_xyz_pointcloud_ply
from hy3dgen.shapegen.models.autoencoders.shape_pc_ae import ShapePCAE
from hy3dgen.shapegen.pc_debug_export import export_recon_debug_plys
from hy3dgen.shapegen.pc_losses import chamfer_distance
from hy3dgen.shapegen.pc_render_dataset import (
    build_surface_render_dataset,
    collate_surface_render,
)
from evaluate_pc_unite import load_unite_model
from train_gs_ae import load_experiment_manifest, resolve_category_ids
from train_pc_unite import _build_weak_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALL_STAGES = (
    "data_check",
    "diffusion_curve",
    "encode_vggt",
    "latent_align",
    "pca_trajectories",
    "cfg_sweep",
    "multiview_eval",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def token_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())


@torch.no_grad()
def decode_cd(model, z: torch.Tensor, gt_xyz: torch.Tensor):
    xyz, rgb, centers = model.decode(z, representation_phase=False)
    cd, _, _ = chamfer_distance(xyz, gt_xyz)
    return float(cd), xyz, rgb, centers


def _mean(vals: Sequence[float]) -> float:
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def _summarize_rows(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    return {k: _mean([r[k] for r in rows]) for k in keys}


def build_dataset(args, train_args):
    include_sharp = bool(train_args.get("include_sharp_label") or False)
    align_mode = args.align_mode or train_args.get("align_mode", "cross")
    data_path = Path(args.data_dir).resolve()
    manifest = (
        load_experiment_manifest(str(data_path))
        if not args.no_experiment_manifest
        else None
    )
    # Expand to multiple views when multiview_eval is requested.
    view_indices = None
    stages = getattr(args, "stages", None) or []
    if "multiview_eval" in stages and getattr(args, "eval_views", None):
        view_indices = sorted(set([int(args.view_idx)] + [int(v) for v in args.eval_views]))
        logger.info("Dataset view_indices for multiview_eval: %s", view_indices)
    dataset = build_surface_render_dataset(
        str(data_path),
        max_items=args.max_items,
        categories=resolve_category_ids(args.categories),
        include_sharp_label=include_sharp,
        use_experiment_manifest=not args.no_experiment_manifest,
        manifest=manifest,
        render_root=args.gobjaverse_render_root,
        view_idx=args.view_idx,
        view_indices=view_indices,
        vggt_cache_root=args.vggt_cache_root,
        pc_size=int(train_args.get("pc_size", 5120)),
        pc_sharpedge_size=int(train_args.get("pc_sharpedge_size", 5120)),
        gobjaverse_normalization=not train_args.get("no_gobjaverse_normalization", False),
        surface_in_camera_frame=not train_args.get("no_surface_camera_frame", False),
        align_mode=align_mode,
    )
    return dataset, include_sharp, align_mode


def load_eval_ranking(eval_results: Optional[str]) -> List[Dict]:
    """Return per_object list sorted by gen_cd descending (worst first)."""
    if not eval_results:
        return []
    path = Path(eval_results)
    if not path.exists():
        logger.warning("eval_results not found: %s", path)
        return []
    data = json.load(open(path))
    objs = list(data.get("per_object", []))
    key = "gen_cd_cfg1"
    if objs and key not in objs[0]:
        # fall back to any gen_cd_* key
        for k in objs[0]:
            if k.startswith("gen_cd"):
                key = k
                break
    objs.sort(key=lambda o: float(o.get(key, 0.0)), reverse=True)
    for o in objs:
        o["_rank_key"] = key
        o["_gen_cd"] = float(o.get(key, float("nan")))
    return objs


def select_mesh_indices(
    dataset,
    ranked: List[Dict],
    *,
    n_worst: int,
    n_best: int,
    n_random: int,
    seed: int,
    view_idx: int,
) -> List[Dict]:
    """Pick curated dataset indices with tags {worst,best,random}.

    ``SurfaceRenderDataset`` indexes ``(mesh, view)`` pairs in ``samples``.
    We prefer the eval view (``view_idx``) when multiple views exist.
    """
    samples: List[Tuple[str, int]] = getattr(dataset, "samples", None) or []
    if not samples and hasattr(dataset, "mesh_paths"):
        samples = [(p, int(view_idx)) for p in dataset.mesh_paths]

    # mesh path / stem / name → preferred sample index at view_idx
    path_to_idx: Dict[str, int] = {}
    stem_to_idx: Dict[str, int] = {}
    for i, (path, vid) in enumerate(samples):
        if int(vid) != int(view_idx):
            continue
        path_to_idx[str(path)] = i
        path_to_idx[str(Path(path).resolve())] = i
        path_to_idx[Path(path).name] = i
        stem_to_idx[Path(path).stem] = i
    # fallback: any view for that mesh
    for i, (path, _vid) in enumerate(samples):
        stem = Path(path).stem
        if stem not in stem_to_idx:
            stem_to_idx[stem] = i
            path_to_idx.setdefault(str(path), i)
            path_to_idx.setdefault(Path(path).name, i)

    def find_idx(mesh: str) -> Optional[int]:
        for c in (mesh, str(Path(mesh).resolve()), Path(mesh).name):
            if c in path_to_idx:
                return path_to_idx[c]
        return stem_to_idx.get(Path(mesh).stem)

    picked: List[Dict] = []
    used = set()

    def add(tag: str, mesh: Optional[str] = None, idx: Optional[int] = None, gen_cd=None):
        if idx is None and mesh is not None:
            idx = find_idx(mesh)
        if idx is None or idx in used or idx < 0 or idx >= len(samples):
            return
        used.add(idx)
        mesh_path = mesh or samples[idx][0]
        picked.append(
            {
                "tag": tag,
                "idx": int(idx),
                "mesh": mesh_path,
                "view_idx": int(samples[idx][1]),
                "gen_cd": gen_cd,
            }
        )

    if ranked:
        for o in ranked[:n_worst]:
            add("worst", mesh=o["mesh"], gen_cd=o.get("_gen_cd"))
        for o in reversed(ranked[-n_best:]):
            add("best", mesh=o["mesh"], gen_cd=o.get("_gen_cd"))

    g = torch.Generator().manual_seed(seed)
    n = len(samples)
    perm = torch.randperm(n, generator=g).tolist()
    for i in perm:
        if sum(p["tag"] == "random" for p in picked) >= n_random:
            break
        add("random", idx=i)

    if not picked:
        for i in range(min(n_worst + n_best + n_random, n)):
            add("sample", idx=i)

    logger.info(
        "Selected %d objects: worst=%d best=%d random=%d sample=%d",
        len(picked),
        sum(p["tag"] == "worst" for p in picked),
        sum(p["tag"] == "best" for p in picked),
        sum(p["tag"] == "random" for p in picked),
        sum(p["tag"] == "sample" for p in picked),
    )
    return picked


def _aligned_patch_centers(payload: Dict, align_mode: str):
    """Same PE centres the model uses (select + mean_z/Hunyuan bbox)."""
    return aligned_centers_from_payload(payload, align_mode)


def _save_view_rgb(batch: Dict, path: Path) -> bool:
    """Write conditioning render ``view_rgb.png`` (CHW float [0,1] → PNG)."""
    if "rgb" not in batch:
        return False
    from PIL import Image
    import numpy as np

    rgb = batch["rgb"][0].detach().float().cpu()
    if rgb.dim() != 3:
        return False
    arr = (rgb.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)
    return True


@torch.no_grad()
def prepare_item(dataset, idx, model, vggt_builder, device, include_sharp, align_mode, geometry_only):
    batch = collate_surface_render([dataset[idx]])
    surface = batch["surface"].to(device)
    gt_xyz, gt_rgb = ShapePCAE.surface_gt_points(
        surface, include_sharp_label=include_sharp, include_rgb=not geometry_only
    )
    weak, keep, cam = _build_weak_context(
        batch,
        vggt_builder,
        device,
        align_mode=align_mode,
        include_camera_in_sequence=not bool(
            getattr(model, "adaln_camera_cond", False)
        ),
    )
    # Always pass weak context into encode; encode_tokenizer ignores it unless
    # tokenizer_use_weak_context is True.
    z, fps_xyz = model.encode(surface, weak_context=weak, weak_context_keep=keep)
    patch_raw = patch_aligned = patch_keep = None
    if "vggt_cache" in batch and batch["vggt_cache"]:
        payload = batch["vggt_cache"][0]
        patch_raw, patch_aligned, patch_keep = _aligned_patch_centers(payload, align_mode)
    normals = surface[0, :, 3:6].detach()
    return {
        "batch": batch,
        "surface": surface,
        "gt_xyz": gt_xyz,
        "gt_rgb": gt_rgb,
        "normals": normals,
        "weak": weak,
        "keep": keep,
        "cam": cam,
        "z": z,
        "fps_xyz": fps_xyz,
        "patch_centers": patch_aligned,  # training frame (aligned)
        "patch_centers_raw": patch_raw,
        "patch_keep": patch_keep,
        "mesh": batch["mesh_path"][0],
        "view_idx": int(batch["view_idx"][0]) if "view_idx" in batch else 0,
        "align_mode": align_mode,
    }


# ---------------------------------------------------------------------------
# Stage: data_check
# ---------------------------------------------------------------------------


@torch.no_grad()
def stage_data_check(args, model, vggt_builder, dataset, train_args, picked, out_dir: Path):
    include_sharp = bool(train_args.get("include_sharp_label") or False)
    geometry_only = bool(train_args.get("geometry_only", False))
    align_mode = args.align_mode or train_args.get("align_mode", "cross")
    device = next(model.parameters()).device

    stage_dir = out_dir / "data_check"
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for j, sel in enumerate(picked):
        it = prepare_item(
            dataset,
            sel["idx"],
            model,
            vggt_builder,
            device,
            include_sharp,
            align_mode,
            geometry_only,
        )
        stem = Path(it["mesh"]).stem[:16]
        tag = sel["tag"]
        obj_dir = stage_dir / f"{j:02d}_{tag}_{stem}"
        obj_dir.mkdir(parents=True, exist_ok=True)

        noise = torch.randn_like(it["z"])
        xyz_r, rgb_r, centers_r = model.decode(it["z"], representation_phase=False)[:3]
        z_gen = model.sample_latents(
            it["weak"],
            batch_size=1,
            num_steps=args.sample_steps,
            guidance_scale=1.0,
            noise=noise,
            context_keep=it["keep"],
            cam_cond=it.get("cam"),
            device=device,
            dtype=it["z"].dtype,
            renorm_output=False,
        )
        xyz_g, rgb_g, centers_g = model.decode(z_gen, representation_phase=False)[:3]
        cd_r, _, _ = chamfer_distance(xyz_r, it["gt_xyz"])
        cd_g, _, _ = chamfer_distance(xyz_g, it["gt_xyz"])

        # Conditioning view (same RGB used for VGGT / weak context)
        has_view = _save_view_rgb(it["batch"], obj_dir / "view_rgb.png")

        # GT with nx/ny/nz for CloudCompare "Draw normals"
        export_input_surface_ply(
            it["surface"][0].detach().cpu(),
            obj_dir / "gt_with_normals.ply",
            include_sharp_label=include_sharp,
        )
        # Colour-only GT (lighter overlay vs recon/gen)
        export_xyz_pointcloud_ply(
            it["gt_xyz"][0].cpu(),
            obj_dir / "gt.ply",
            colors=None if it["gt_rgb"] is None else it["gt_rgb"][0].cpu(),
        )
        # Optional: normals as RGB (quick glance without arrows)
        nrm = it["normals"].float().cpu()
        export_xyz_pointcloud_ply(
            it["gt_xyz"][0].cpu(),
            obj_dir / "gt_normals_rgb.ply",
            colors=(nrm * 0.5 + 0.5).clamp(0, 1),
        )
        export_xyz_pointcloud_ply(xyz_r[0].cpu(), obj_dir / "recon.ply", colors=_first_cpu(rgb_r))
        export_xyz_pointcloud_ply(xyz_g[0].cpu(), obj_dir / "gen.ply", colors=_first_cpu(rgb_g))
        export_xyz_pointcloud_ply(
            it["fps_xyz"][0].cpu(), obj_dir / "fps.ply", rgb=(0, 220, 220)
        )
        export_xyz_pointcloud_ply(
            centers_r[0].cpu(), obj_dir / "anchors_recon.ply", rgb=(220, 0, 220)
        )
        export_xyz_pointcloud_ply(
            centers_g[0].cpu(), obj_dir / "anchors_gen.ply", rgb=(255, 128, 0)
        )

        def _export_centers(pc, path, rgb):
            if pc is None:
                return 0
            pc_t = pc.float()
            if pc_t.dim() == 3:
                pc_t = pc_t[0]
            keep = it["patch_keep"]
            if keep is not None and torch.is_tensor(keep):
                k = keep.bool().reshape(-1)
                if k.numel() == pc_t.shape[0]:
                    pc_t = pc_t[k]
            export_xyz_pointcloud_ply(pc_t.cpu(), path, rgb=rgb)
            return int(pc_t.shape[0])

        # Aligned centres (same frame as GT / PE). Raw kept for diagnosing export bugs.
        n_aligned = _export_centers(
            it["patch_centers"], obj_dir / "patch_centers.ply", (32, 200, 64)
        )
        n_raw = _export_centers(
            it["patch_centers_raw"], obj_dir / "patch_centers_raw.ply", (255, 200, 0)
        )

        export_recon_debug_plys(
            obj_dir,
            gt_xyz=it["gt_xyz"],
            recon_xyz=xyz_r,
            gt_rgb=it["gt_rgb"],
            recon_rgb=rgb_r,
            fps_xyz=it["fps_xyz"],
            centers=centers_r,
            num_points_per_anchor=int(model.num_points_per_anchor),
            max_anchor_delta=model.max_anchor_delta,
            patch_centers=it["patch_centers"],
            patch_keep=it["patch_keep"],
            intruder_thresh=float(args.intruder_thresh),
        )

        # Brief note for CloudCompare
        (obj_dir / "README_VIEW.txt").write_text(
            "Files (same Hunyuan unit-box camera frame unless noted):\n"
            "  view_rgb.png          — conditioning render for this view_idx\n"
            "  gt.ply                — GT xyz + rgb\n"
            "  gt_with_normals.ply   — GT xyz + nx/ny/nz + rgb  ← use for arrows\n"
            "  gt_normals_rgb.ply    — GT coloured by normal direction\n"
            "  patch_centers.ply     — VGGT PE centres AFTER align (matches GT)\n"
            "  patch_centers_raw.ply — cache centres BEFORE align (do not overlay)\n"
            "  recon.ply / gen.ply / fps.ply / anchors_*.ply\n"
            "\n"
            "CloudCompare normals: load gt_with_normals.ply → select cloud →\n"
            "  Edit → Normals → Set display options → enable Draw normals.\n"
            f"\nalign_mode={it.get('align_mode')}\n"
            f"view_idx={it['view_idx']}\n"
            f"n_patch_aligned={n_aligned}  n_patch_raw={n_raw}\n"
        )

        meta = {
            "tag": tag,
            "mesh": it["mesh"],
            "view_idx": it["view_idx"],
            "align_mode": it.get("align_mode"),
            "view_rgb": has_view,
            "gen_cd_from_eval": sel.get("gen_cd"),
            "recon_cd": float(cd_r),
            "gen_cd": float(cd_g),
            "n_weak_tokens": int(it["weak"].shape[1]) if it["weak"] is not None else 0,
            "n_keep": int(it["keep"].float().sum()) if it["keep"] is not None else 0,
            "n_patch_aligned": n_aligned,
            "n_patch_raw": n_raw,
            "dir": str(obj_dir),
        }
        with open(obj_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        manifest.append(meta)
        logger.info(
            "[data_check %d/%d] %s %s recon=%.4f gen=%.4f → %s",
            j + 1,
            len(picked),
            tag,
            stem,
            float(cd_r),
            float(cd_g),
            obj_dir.name,
        )

    with open(stage_dir / "manifest.json", "w") as f:
        json.dump({"objects": manifest}, f, indent=2)
    return {"num_objects": len(manifest), "objects": manifest}


def _first_cpu(rgb):
    if rgb is None:
        return None
    return rgb[0].detach().float().cpu()


# ---------------------------------------------------------------------------
# Stage: diffusion_curve
# ---------------------------------------------------------------------------


@torch.no_grad()
def velocity_error_at_t(
    model, z, weak, keep, t_scalar: float, noise: torch.Tensor, cam=None
):
    """True-path velocity MSE at a fixed t (no ODE)."""
    t = torch.full((z.shape[0],), float(t_scalar), device=z.device, dtype=z.dtype)
    # x_t = t*z + (1-t)*eps   (ICPlan)
    x_t = t.view(-1, 1, 1) * z + (1.0 - t.view(-1, 1, 1)) * noise
    # u* = z - eps  (constant for linear plan)
    u_gt = z - noise
    train_eps = model.transport.train_eps
    cond = model._resolve_cam_cond(
        cam, z.shape[0], device=z.device, dtype=z.dtype, use_null=(cam is None)
    )
    x_pred = model.latent_norm(
        model._run_ge(
            x_t, t, context_embed=weak, context_keep=keep, cond_embed=cond
        )
    )
    denom = (1.0 - t.view(-1, 1, 1)).clamp_min(train_eps)
    v_pred = (x_pred - x_t) / denom
    v_gt = u_gt  # for linear plan d/dt (t z + (1-t) eps) = z - eps
    # Also report x_start error
    x_start_mse = float(((x_pred - z) ** 2).mean())
    vel_mse = float(((v_pred - v_gt) ** 2).mean())
    # Weighted velocity form used in training
    w = float(1.0 / max(1.0 - t_scalar, train_eps) ** 2)
    return {
        "t": float(t_scalar),
        "velocity_mse": vel_mse,
        "x_start_mse": x_start_mse,
        "train_weight_1_over_1mt2": w,
        "weighted_velocity_mse": vel_mse * w,
        "rel_xpred_to_z": rel_err(x_pred, z),
        "token_cos_xpred_z": token_cosine(x_pred, z),
    }


@torch.no_grad()
def stage_diffusion_curve(args, model, vggt_builder, dataset, train_args, picked, out_dir: Path):
    include_sharp = bool(train_args.get("include_sharp_label") or False)
    geometry_only = bool(train_args.get("geometry_only", False))
    align_mode = args.align_mode or train_args.get("align_mode", "cross")
    device = next(model.parameters()).device
    t_grid = [float(t) for t in args.t_grid]

    stage_dir = out_dir / "diffusion_curve"
    stage_dir.mkdir(parents=True, exist_ok=True)

    per_object = []
    for j, sel in enumerate(picked):
        it = prepare_item(
            dataset,
            sel["idx"],
            model,
            vggt_builder,
            device,
            include_sharp,
            align_mode,
            geometry_only,
        )
        if it["weak"] is None:
            logger.warning("No weak context for %s — skip", it["mesh"])
            continue
        # Shared noise for fair oracle / gen comparison across t
        torch.manual_seed(args.seed + int(sel["idx"]))
        noise = torch.randn_like(it["z"])

        cd_recon, _, _, centers_ref = decode_cd(model, it["z"], it["gt_xyz"])
        oracle_rows = []
        vel_rows = []
        for t0 in t_grid:
            # velocity probe (no ODE)
            vel_rows.append(
                velocity_error_at_t(
                    model, it["z"], it["weak"], it["keep"], t0, noise, cam=it.get("cam")
                )
            )
            # oracle ODE from t0
            z_out = model.sample_latents(
                it["weak"],
                batch_size=1,
                num_steps=args.sample_steps,
                guidance_scale=1.0,
                z_init=it["z"],
                t_start=float(t0),
                noise=noise,
                context_keep=it["keep"],
            cam_cond=it.get("cam"),
                device=device,
                dtype=it["z"].dtype,
                renorm_output=False,
            )
            cd_out, _, _, centers_out = decode_cd(model, z_out, it["gt_xyz"])
            oracle_rows.append(
                {
                    "t_start": float(t0),
                    "cd": cd_out,
                    "rel_latent_err": rel_err(z_out, it["z"]),
                    "token_cos": token_cosine(z_out, it["z"]),
                    "anchor_shift": float(
                        (centers_out - centers_ref).norm(dim=-1).mean()
                    ),
                    "x_recon": cd_out / max(cd_recon, 1e-12),
                }
            )

        # pure gen from t=0 for reference
        z_gen = model.sample_latents(
            it["weak"],
            batch_size=1,
            num_steps=args.sample_steps,
            guidance_scale=1.0,
            noise=noise,
            context_keep=it["keep"],
            cam_cond=it.get("cam"),
            device=device,
            dtype=it["z"].dtype,
            renorm_output=False,
        )
        cd_gen, _, _, centers_gen = decode_cd(model, z_gen, it["gt_xyz"])

        rec = {
            "tag": sel["tag"],
            "mesh": it["mesh"],
            "recon_cd": cd_recon,
            "gen_cd": cd_gen,
            "gen_rel_latent_err": rel_err(z_gen, it["z"]),
            "gen_anchor_shift": float((centers_gen - centers_ref).norm(dim=-1).mean()),
            "oracle": oracle_rows,
            "velocity": vel_rows,
        }
        per_object.append(rec)
        logger.info(
            "[diffusion %d/%d] %s recon=%.4f gen=%.4f oracle_t0.5=%.4f vel_mse_t0.1=%.4f vel_mse_t0.9=%.4f",
            j + 1,
            len(picked),
            Path(it["mesh"]).stem[:12],
            cd_recon,
            cd_gen,
            next(r["cd"] for r in oracle_rows if abs(r["t_start"] - 0.5) < 1e-6)
            if any(abs(r["t_start"] - 0.5) < 1e-6 for r in oracle_rows)
            else float("nan"),
            next(r["velocity_mse"] for r in vel_rows if abs(r["t"] - 0.1) < 1e-6)
            if any(abs(r["t"] - 0.1) < 1e-6 for r in vel_rows)
            else float("nan"),
            next(r["velocity_mse"] for r in vel_rows if abs(r["t"] - 0.9) < 1e-6)
            if any(abs(r["t"] - 0.9) < 1e-6 for r in vel_rows)
            else float("nan"),
        )

    # Aggregate curves
    oracle_curve = []
    vel_curve = []
    for ti, t0 in enumerate(t_grid):
        oracle_curve.append(
            {
                "t_start": t0,
                **_summarize_rows([r["oracle"][ti] for r in per_object]),
            }
        )
        vel_curve.append(
            {
                "t": t0,
                **_summarize_rows([r["velocity"][ti] for r in per_object]),
            }
        )

    summary = {
        "num_objects": len(per_object),
        "recon_cd": _mean([r["recon_cd"] for r in per_object]),
        "gen_cd": _mean([r["gen_cd"] for r in per_object]),
        "oracle_curve": oracle_curve,
        "velocity_curve": vel_curve,
    }
    out = {"summary": summary, "per_object": per_object}
    with open(stage_dir / "diffusion_curve.json", "w") as f:
        json.dump(out, f, indent=2)
    _write_diffusion_csv(stage_dir / "oracle_curve.csv", oracle_curve, "t_start")
    _write_diffusion_csv(stage_dir / "velocity_curve.csv", vel_curve, "t")
    _print_diffusion_report(summary)
    return summary


def _write_diffusion_csv(path: Path, rows: List[Dict], t_key: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


def _print_diffusion_report(summary: Dict):
    print("\n" + "=" * 72)
    print("DIFFUSION CURVE")
    print("=" * 72)
    print(
        f"objects={summary['num_objects']}  recon={summary['recon_cd']:.5f}  "
        f"gen={summary['gen_cd']:.5f}"
    )
    print(
        f"\n{'t':>6}{'oracle_cd':>12}{'x_recon':>10}{'rel_err':>10}"
        f"{'vel_mse':>12}{'x0_mse':>10}{'weight':>10}"
    )
    oc = {r["t_start"]: r for r in summary["oracle_curve"]}
    vc = {r["t"]: r for r in summary["velocity_curve"]}
    for t in sorted(oc):
        o, v = oc[t], vc.get(t, {})
        print(
            f"{t:6.2f}{o.get('cd', float('nan')):12.5f}"
            f"{o.get('x_recon', float('nan')):10.2f}"
            f"{o.get('rel_latent_err', float('nan')):10.4f}"
            f"{v.get('velocity_mse', float('nan')):12.5f}"
            f"{v.get('x_start_mse', float('nan')):10.5f}"
            f"{v.get('train_weight_1_over_1mt2', float('nan')):10.1f}"
        )
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Stage: encode_vggt
# ---------------------------------------------------------------------------


@torch.no_grad()
def stage_encode_vggt(args, model, vggt_builder, dataset, train_args, picked, out_dir: Path):
    include_sharp = bool(train_args.get("include_sharp_label") or False)
    geometry_only = bool(train_args.get("geometry_only", False))
    align_mode = args.align_mode or train_args.get("align_mode", "cross")
    device = next(model.parameters()).device
    tok_flag = bool(getattr(model, "tokenizer_use_weak_context", False))

    stage_dir = out_dir / "encode_vggt"
    stage_dir.mkdir(parents=True, exist_ok=True)

    per_object = []
    for j, sel in enumerate(picked):
        it = prepare_item(
            dataset,
            sel["idx"],
            model,
            vggt_builder,
            device,
            include_sharp,
            align_mode,
            geometry_only,
        )
        # With VGGT (training-consistent for exp7)
        z_with, _ = model.encode(
            it["surface"], weak_context=it["weak"], weak_context_keep=it["keep"]
        )
        # Without VGGT
        z_wo, _ = model.encode(it["surface"], weak_context=None, weak_context_keep=None)
        # Also: force-disable flag temporarily to see encode path ignore
        prev = model.tokenizer_use_weak_context
        model.tokenizer_use_weak_context = False
        z_flag_off, _ = model.encode(
            it["surface"], weak_context=it["weak"], weak_context_keep=it["keep"]
        )
        model.tokenizer_use_weak_context = prev

        cd_with, _, _, _ = decode_cd(model, z_with, it["gt_xyz"])
        cd_wo, _, _, _ = decode_cd(model, z_wo, it["gt_xyz"])
        cd_flag_off, _, _, _ = decode_cd(model, z_flag_off, it["gt_xyz"])

        # Mean of several with-VGGT draws (stochastic registers)
        zs = []
        for _ in range(args.num_draws):
            z_i, _ = model.encode(
                it["surface"], weak_context=it["weak"], weak_context_keep=it["keep"]
            )
            zs.append(z_i)
        z_mean = torch.cat(zs, dim=0).mean(dim=0, keepdim=True)
        cd_mean, _, _, _ = decode_cd(model, z_mean, it["gt_xyz"])
        cd_samples = [decode_cd(model, z, it["gt_xyz"])[0] for z in zs]

        rec = {
            "tag": sel["tag"],
            "mesh": it["mesh"],
            "tokenizer_use_weak_context": tok_flag,
            "cd_with_vggt": cd_with,
            "cd_without_vggt": cd_wo,
            "cd_flag_forced_off": cd_flag_off,
            "rel_with_vs_without": rel_err(z_with, z_wo),
            "cos_with_vs_without": token_cosine(z_with, z_wo),
            "cd_samples_mean": sum(cd_samples) / len(cd_samples),
            "cd_of_mean_latent": cd_mean,
            "rel_pairwise": rel_err(zs[0], zs[1]) if len(zs) > 1 else float("nan"),
        }
        per_object.append(rec)
        logger.info(
            "[encode_vggt %d/%d] %s with=%.5f without=%.5f flag_off=%.5f "
            "rel=%.3f cd_mean_z=%.5f",
            j + 1,
            len(picked),
            Path(it["mesh"]).stem[:12],
            cd_with,
            cd_wo,
            cd_flag_off,
            rec["rel_with_vs_without"],
            cd_mean,
        )

    summary = {
        "num_objects": len(per_object),
        "tokenizer_use_weak_context": tok_flag,
        **{k: _mean([r[k] for r in per_object]) for k in per_object[0] if k not in ("tag", "mesh", "tokenizer_use_weak_context")},
    }
    out = {"summary": summary, "per_object": per_object}
    with open(stage_dir / "encode_vggt.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 72)
    print("ENCODE ± VGGT")
    print("=" * 72)
    print(f"tokenizer_use_weak_context (ckpt) = {tok_flag}")
    print(f"objects = {summary['num_objects']}")
    for k in (
        "cd_with_vggt",
        "cd_without_vggt",
        "cd_flag_forced_off",
        "rel_with_vs_without",
        "cos_with_vs_without",
        "cd_samples_mean",
        "cd_of_mean_latent",
        "rel_pairwise",
    ):
        print(f"  {k:28s} {summary[k]:.5f}")
    print(
        "\nInterpretation: if cd_with ≈ cd_without and rel≈0, tokenizer ignores VGGT.\n"
        "If cd_without ≫ cd_with, tokenizer depends on VGGT (OOD without it).\n"
        "cd_of_mean_latent ≫ cd_samples_mean ⇒ stochastic target / mean off-manifold."
    )
    print("=" * 72 + "\n")
    return summary


# ---------------------------------------------------------------------------
# Stage: latent_align
# ---------------------------------------------------------------------------


@torch.no_grad()
def stage_latent_align(args, model, vggt_builder, dataset, train_args, picked, out_dir: Path):
    include_sharp = bool(train_args.get("include_sharp_label") or False)
    geometry_only = bool(train_args.get("geometry_only", False))
    align_mode = args.align_mode or train_args.get("align_mode", "cross")
    device = next(model.parameters()).device
    sigmas = [float(s) for s in args.sigmas]

    stage_dir = out_dir / "latent_align"
    stage_dir.mkdir(parents=True, exist_ok=True)

    per_object = []
    for j, sel in enumerate(picked):
        it = prepare_item(
            dataset,
            sel["idx"],
            model,
            vggt_builder,
            device,
            include_sharp,
            align_mode,
            geometry_only,
        )
        if it["weak"] is None:
            continue
        noise = torch.randn_like(it["z"])
        cd_ref, _, _, centers_ref = decode_cd(model, it["z"], it["gt_xyz"])

        # sigma sweep
        sweep = []
        for s in sigmas:
            z_p = it["z"] + s * torch.randn_like(it["z"])
            cd_p, _, _, _ = decode_cd(model, z_p, it["gt_xyz"])
            sweep.append(
                {"sigma": s, "rel_err": rel_err(z_p, it["z"]), "cd": cd_p}
            )

        z_gen = model.sample_latents(
            it["weak"],
            batch_size=1,
            num_steps=args.sample_steps,
            guidance_scale=1.0,
            noise=noise,
            context_keep=it["keep"],
            cam_cond=it.get("cam"),
            device=device,
            dtype=it["z"].dtype,
            renorm_output=False,
        )
        cd_gen, _, _, centers_gen = decode_cd(model, z_gen, it["gt_xyz"])
        r = rel_err(z_gen, it["z"])
        pred = _interp_cd(sweep, r)

        # mean latent of K encodes
        zs = [
            model.encode(
                it["surface"], weak_context=it["weak"], weak_context_keep=it["keep"]
            )[0]
            for _ in range(args.num_draws)
        ]
        z_mean = torch.cat(zs, dim=0).mean(dim=0, keepdim=True)
        cd_mean, _, _, _ = decode_cd(model, z_mean, it["gt_xyz"])

        rec = {
            "tag": sel["tag"],
            "mesh": it["mesh"],
            "recon_cd": cd_ref,
            "gen_cd": cd_gen,
            "rel_latent_err": r,
            "token_cos": token_cosine(z_gen, it["z"]),
            "anchor_shift": float((centers_gen - centers_ref).norm(dim=-1).mean()),
            "cd_predicted_isotropic": pred,
            "excess_vs_isotropic": cd_gen / max(pred, 1e-12),
            "cd_of_mean_latent": cd_mean,
            "sigma_sweep": sweep,
        }
        per_object.append(rec)
        logger.info(
            "[align %d/%d] %s recon=%.4f gen=%.4f rel=%.3f excess=%.2fx mean_z=%.4f",
            j + 1,
            len(picked),
            Path(it["mesh"]).stem[:12],
            cd_ref,
            cd_gen,
            r,
            rec["excess_vs_isotropic"],
            cd_mean,
        )

    summary = {
        "num_objects": len(per_object),
        "recon_cd": _mean([r["recon_cd"] for r in per_object]),
        "gen_cd": _mean([r["gen_cd"] for r in per_object]),
        "rel_latent_err": _mean([r["rel_latent_err"] for r in per_object]),
        "token_cos": _mean([r["token_cos"] for r in per_object]),
        "anchor_shift": _mean([r["anchor_shift"] for r in per_object]),
        "cd_predicted_isotropic": _mean(
            [r["cd_predicted_isotropic"] for r in per_object]
        ),
        "excess_vs_isotropic": _mean([r["excess_vs_isotropic"] for r in per_object]),
        "cd_of_mean_latent": _mean([r["cd_of_mean_latent"] for r in per_object]),
    }
    out = {"summary": summary, "per_object": per_object}
    with open(stage_dir / "latent_align.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 72)
    print("LATENT ALIGNMENT (gen vs tok)")
    print("=" * 72)
    for k, v in summary.items():
        if k == "num_objects":
            print(f"  {k}: {v}")
        else:
            print(f"  {k:28s} {v:.5f}")
    print(
        "\nexcess_vs_isotropic ≫ 1 ⇒ structured flow error (wrong basin),\n"
        "not just isotropic off-manifold noise."
    )
    print("=" * 72 + "\n")
    return summary


def _interp_cd(sweep, rel_target: float) -> float:
    pts = sorted((r["rel_err"], r["cd"]) for r in sweep)
    if rel_target <= pts[0][0]:
        return pts[0][1]
    if rel_target >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= rel_target <= x1:
            w = (rel_target - x0) / max(x1 - x0, 1e-12)
            return y0 + w * (y1 - y0)
    return pts[-1][1]


# ---------------------------------------------------------------------------
# Stage: pca_trajectories
# ---------------------------------------------------------------------------


def _flatten_latent(z: torch.Tensor) -> "np.ndarray":
    import numpy as np

    return z.detach().float().cpu().reshape(-1).numpy()


def _pca_fit_project(X: "np.ndarray", n_comp: int = 2):
    """Centre + SVD PCA (no sklearn). Returns (proj [N,k], explained_var_ratio)."""
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # economy SVD
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_comp, Vt.shape[0])
    components = Vt[:k]
    proj = Xc @ components.T
    var = (S ** 2) / max(X.shape[0] - 1, 1)
    ratio = var[:k] / max(var.sum(), 1e-12)
    return proj, ratio, mu, components


@torch.no_grad()
def stage_pca_trajectories(args, model, vggt_builder, dataset, train_args, picked, out_dir: Path):
    """Compare linear noising bridges vs ODE trajectories in latent PCA space."""
    import numpy as np

    include_sharp = bool(train_args.get("include_sharp_label") or False)
    geometry_only = bool(train_args.get("geometry_only", False))
    align_mode = args.align_mode or train_args.get("align_mode", "cross")
    device = next(model.parameters()).device
    t_starts = [float(t) for t in args.t_grid]
    n_latents = max(1, int(args.num_latent_draws))
    n_noises = max(1, int(args.num_noise_draws))
    # Subsample ODE waypoints for plots / storage (keep endpoints).
    waypoint_stride = max(1, int(args.waypoint_stride))

    stage_dir = out_dir / "pca_trajectories"
    stage_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = stage_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Limit objects for this expensive stage
    n_obj = min(len(picked), int(args.pca_max_objects) if args.pca_max_objects else len(picked))
    # Prefer mix of tags
    sel_list = picked[:n_obj]

    all_cases = []  # metrics + paths for JSON (downsample waypoints)
    # Collect vectors for a *global* PCA across all cases
    global_vecs = []
    global_meta = []  # (case_id, kind, t)

    for j, sel in enumerate(sel_list):
        it = prepare_item(
            dataset,
            sel["idx"],
            model,
            vggt_builder,
            device,
            include_sharp,
            align_mode,
            geometry_only,
        )
        if it["weak"] is None:
            logger.warning("No weak context for %s — skip PCA", it["mesh"])
            continue
        stem = Path(it["mesh"]).stem[:12]
        obj_dir = stage_dir / f"{j:02d}_{sel['tag']}_{stem}"
        obj_dir.mkdir(parents=True, exist_ok=True)

        for li in range(n_latents):
            # Fresh encode (stochastic registers) unless first draw uses prepared z
            if li == 0:
                z_star = it["z"]
            else:
                z_star, _ = model.encode(
                    it["surface"],
                    weak_context=it["weak"],
                    weak_context_keep=it["keep"],
                )
            for ni in range(n_noises):
                torch.manual_seed(args.seed + 1000 * sel["idx"] + 17 * li + 31 * ni)
                eps = torch.randn_like(z_star)
                case_id = f"{j:02d}_{sel['tag']}_{stem}_L{li}_N{ni}"
                case_vecs = []
                case_kinds = []
                case_ts = []
                case_paths = {"linear": {}, "ode": {}}
                endpoint_rows = []
                endpoint_vecs = {}  # t_start -> flat vector for pairwise basin test

                # Reference points
                for name, tens in (("z_star", z_star), ("eps", eps)):
                    v = _flatten_latent(tens)
                    case_vecs.append(v)
                    case_kinds.append(name)
                    case_ts.append(float("nan"))
                    global_vecs.append(v)
                    global_meta.append((case_id, name, float("nan")))

                # Mean of several encodes (ceiling / mean-manifold probe)
                zs_extra = [z_star]
                for _ in range(max(0, args.num_draws - 1)):
                    zi, _ = model.encode(
                        it["surface"],
                        weak_context=it["weak"],
                        weak_context_keep=it["keep"],
                    )
                    zs_extra.append(zi)
                z_mean = torch.stack([z.squeeze(0) for z in zs_extra], dim=0).mean(
                    dim=0, keepdim=True
                )
                v_mean = _flatten_latent(z_mean)
                case_vecs.append(v_mean)
                case_kinds.append("z_mean")
                case_ts.append(float("nan"))
                global_vecs.append(v_mean)
                global_meta.append((case_id, "z_mean", float("nan")))

                for t0 in t_starts:
                    # Linear bridge point at t0
                    x_t = t0 * z_star + (1.0 - t0) * eps
                    v_lin = _flatten_latent(x_t)
                    case_vecs.append(v_lin)
                    case_kinds.append(f"linear_t{t0:g}")
                    case_ts.append(t0)
                    global_vecs.append(v_lin)
                    global_meta.append((case_id, "linear", t0))

                    # Dense linear bridge for metrics
                    lin_ts = np.linspace(t0, 1.0, max(8, args.sample_steps // 4))
                    lin_path = [
                        _flatten_latent(float(tt) * z_star + (1.0 - float(tt)) * eps)
                        for tt in lin_ts
                    ]
                    z_flat = _flatten_latent(z_star)
                    z_norm = float(np.linalg.norm(z_flat)) + 1e-12
                    case_paths["linear"][f"{t0:g}"] = {
                        "t": lin_ts.tolist(),
                        "rel_to_z": [
                            float(np.linalg.norm(p - z_flat) / z_norm) for p in lin_path
                        ],
                    }

                    # ODE from t0 with full waypoints
                    z_out, traj, t_grid = model.sample_latents(
                        it["weak"],
                        batch_size=1,
                        num_steps=args.sample_steps,
                        guidance_scale=1.0,
                        z_init=z_star,
                        t_start=float(t0),
                        noise=eps,
                        context_keep=it["keep"],
            cam_cond=it.get("cam"),
                        device=device,
                        dtype=z_star.dtype,
                        renorm_output=False,
                        return_trajectory=True,
                    )
                    idxs = list(range(0, traj.shape[0], waypoint_stride))
                    if idxs[-1] != traj.shape[0] - 1:
                        idxs.append(traj.shape[0] - 1)
                    ode_path = []
                    ode_t = []
                    for ii in idxs:
                        vv = _flatten_latent(traj[ii])
                        ode_path.append(vv)
                        ode_t.append(float(t_grid[ii]))
                        case_vecs.append(vv)
                        case_kinds.append(f"ode_t0={t0:g}")
                        case_ts.append(float(t_grid[ii]))
                        global_vecs.append(vv)
                        global_meta.append((case_id, f"ode@{t0:g}", float(t_grid[ii])))

                    endpoint_vecs[t0] = _flatten_latent(z_out)
                    case_paths["ode"][f"{t0:g}"] = {
                        "t": ode_t,
                        "rel_to_z": [
                            float(np.linalg.norm(p - z_flat) / z_norm) for p in ode_path
                        ],
                        "rel_endpoint_to_z": rel_err(z_out, z_star),
                        "token_cos_endpoint": token_cosine(z_out, z_star),
                        "rel_endpoint_to_mean": rel_err(z_out, z_mean),
                        "token_cos_endpoint_mean": token_cosine(z_out, z_mean),
                    }
                    endpoint_rows.append(
                        {
                            "t_start": t0,
                            "rel_to_z": rel_err(z_out, z_star),
                            "cos_to_z": token_cosine(z_out, z_star),
                            "rel_to_mean": rel_err(z_out, z_mean),
                            "cos_to_mean": token_cosine(z_out, z_mean),
                            "rel_to_eps": rel_err(z_out, eps),
                        }
                    )

                # Pairwise endpoint distances (same noise, different t_start)
                basin = {}
                for ia, ta in enumerate(t_starts):
                    for tb in t_starts[ia + 1 :]:
                        va, vb = endpoint_vecs[ta], endpoint_vecs[tb]
                        basin[f"{ta:g}_vs_{tb:g}"] = float(
                            np.linalg.norm(va - vb) / (np.linalg.norm(va) + 1e-12)
                        )

                # Per-case PCA plot
                X = np.stack(case_vecs, axis=0)
                proj, ratio, _, _ = _pca_fit_project(X, n_comp=2)
                _plot_pca_case(
                    plots_dir / f"{case_id}.png",
                    proj,
                    case_kinds,
                    case_ts,
                    t_starts,
                    title=f"{case_id}  PCA var={ratio[0]:.2f}/{ratio[1]:.2f}",
                )

                ends = {r["t_start"]: r for r in endpoint_rows}
                all_cases.append(
                    {
                        "case_id": case_id,
                        "tag": sel["tag"],
                        "mesh": it["mesh"],
                        "view_idx": it["view_idx"],
                        "latent_draw": li,
                        "noise_draw": ni,
                        "endpoints": endpoint_rows,
                        "endpoint_pairwise_rel": basin,
                        "paths": case_paths,
                        "pca_explained_var": ratio.tolist(),
                    }
                )
                logger.info(
                    "[pca %s] L%d N%d  t0=0 rel=%.3f  t0=0.5 rel=%.3f  cos_mean@0=%.3f",
                    case_id,
                    li,
                    ni,
                    ends.get(0.0, {}).get("rel_to_z", float("nan")),
                    ends.get(0.5, {}).get("rel_to_z", float("nan")),
                    ends.get(0.0, {}).get("cos_to_mean", float("nan")),
                )

    # Global PCA overview plot (all cases)
    if global_vecs:
        Xg = np.stack(global_vecs, axis=0)
        proj_g, ratio_g, _, _ = _pca_fit_project(Xg, n_comp=2)
        _plot_pca_global(
            plots_dir / "global_overview.png",
            proj_g,
            global_meta,
            title=f"Global PCA  var={ratio_g[0]:.2f}/{ratio_g[1]:.2f}",
        )

    # Aggregate endpoint metrics
    summary = _aggregate_pca_summary(all_cases)
    out = {"summary": summary, "cases": all_cases}
    with open(stage_dir / "pca_trajectories.json", "w") as f:
        json.dump(out, f, indent=2)
    _write_pca_csv(stage_dir / "endpoints.csv", all_cases)

    print("\n" + "=" * 72)
    print("PCA TRAJECTORIES")
    print("=" * 72)
    print(f"cases={summary.get('num_cases')}  objects≈{n_obj}")
    for k, v in summary.items():
        if k == "num_cases":
            continue
        if isinstance(v, float):
            print(f"  {k:36s} {v:.5f}")
    print(f"plots → {plots_dir}")
    print("=" * 72 + "\n")
    return summary


def _aggregate_pca_summary(cases: List[Dict]) -> Dict:
    if not cases:
        return {"num_cases": 0}
    # Mean endpoint rel_to_z / cos_to_mean per t_start
    by_t: Dict[float, List[Dict]] = {}
    for c in cases:
        for e in c["endpoints"]:
            by_t.setdefault(e["t_start"], []).append(e)
    summary: Dict = {"num_cases": len(cases)}
    for t, rows in sorted(by_t.items()):
        summary[f"t{t:g}_rel_to_z"] = _mean([r["rel_to_z"] for r in rows])
        summary[f"t{t:g}_cos_to_z"] = _mean([r["cos_to_z"] for r in rows])
        summary[f"t{t:g}_rel_to_mean"] = _mean([r["rel_to_mean"] for r in rows])
        summary[f"t{t:g}_cos_to_mean"] = _mean([r["cos_to_mean"] for r in rows])
    # How close is gen (t=0) endpoint to mean vs to a single sample
    if 0.0 in by_t:
        summary["gen_closer_to_mean_than_z"] = _mean(
            [
                1.0 if r["rel_to_mean"] < r["rel_to_z"] else 0.0
                for r in by_t[0.0]
            ]
        )
    return summary


def _write_pca_csv(path: Path, cases: List[Dict]):
    rows = []
    for c in cases:
        for e in c["endpoints"]:
            rows.append(
                {
                    "case_id": c["case_id"],
                    "tag": c["tag"],
                    "latent_draw": c["latent_draw"],
                    "noise_draw": c["noise_draw"],
                    **e,
                }
            )
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")


def _plot_pca_case(path, proj, kinds, ts, t_starts, title: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib missing — skip plot %s", path)
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    # Special points
    for name, marker, color in (
        ("z_star", "*", "black"),
        ("eps", "X", "red"),
        ("z_mean", "D", "purple"),
    ):
        idx = [i for i, k in enumerate(kinds) if k == name]
        if idx:
            ax.scatter(
                proj[idx, 0],
                proj[idx, 1],
                c=color,
                marker=marker,
                s=120,
                zorder=5,
                label=name,
            )

    cmap = plt.cm.viridis
    for t0 in t_starts:
        # linear
        idx_l = [i for i, k in enumerate(kinds) if k == f"linear_t{t0:g}"]
        # Actually linear only stored one point per t0 in case_vecs; ODE has many
        # Re-plot using kinds starting with ode_t0=
        idx_ode = [i for i, k in enumerate(kinds) if k == f"ode_t0={t0:g}"]
        if len(idx_ode) >= 2:
            color = cmap(t0)
            ax.plot(
                proj[idx_ode, 0],
                proj[idx_ode, 1],
                "-",
                color=color,
                lw=1.5,
                alpha=0.85,
                label=f"ODE t0={t0:g}",
            )
            ax.scatter(
                proj[idx_ode[0], 0],
                proj[idx_ode[0], 1],
                c=[color],
                s=30,
                zorder=4,
            )
            ax.scatter(
                proj[idx_ode[-1], 0],
                proj[idx_ode[-1], 1],
                c=[color],
                s=50,
                marker="o",
                edgecolors="k",
                zorder=4,
            )
        # single linear point at t0
        if idx_l:
            ax.scatter(
                proj[idx_l, 0],
                proj[idx_l, 1],
                c=[cmap(t0)],
                marker="s",
                s=40,
                alpha=0.5,
            )

    # Draw linear bridge from eps to z_star if both exist
    i_eps = next((i for i, k in enumerate(kinds) if k == "eps"), None)
    i_z = next((i for i, k in enumerate(kinds) if k == "z_star"), None)
    if i_eps is not None and i_z is not None:
        ax.plot(
            [proj[i_eps, 0], proj[i_z, 0]],
            [proj[i_eps, 1], proj[i_z, 1]],
            "--",
            color="gray",
            lw=1.0,
            label="linear bridge",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_pca_global(path, proj, meta, title: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    # Plot ODE endpoints (t near 1) and starts; colour by kind
    for kind, color, marker, s in (
        ("z_star", "black", "*", 80),
        ("eps", "red", "x", 50),
        ("z_mean", "purple", "D", 50),
        ("linear", "gray", ".", 10),
    ):
        idx = [i for i, m in enumerate(meta) if m[1] == kind or m[1].startswith(kind)]
        if not idx:
            continue
        ax.scatter(proj[idx, 0], proj[idx, 1], c=color, marker=marker, s=s, alpha=0.6, label=kind)

    # ODE points: colour by t_start prefix
    ode_idx = [i for i, m in enumerate(meta) if str(m[1]).startswith("ode@")]
    if ode_idx:
        ts = np.array([meta[i][2] if meta[i][2] == meta[i][2] else 0.5 for i in ode_idx])
        sc = ax.scatter(
            proj[ode_idx, 0],
            proj[ode_idx, 1],
            c=ts,
            cmap="viridis",
            s=8,
            alpha=0.5,
            label="ODE waypoints",
        )
        fig.colorbar(sc, ax=ax, label="t along ODE")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage: cfg_sweep
# ---------------------------------------------------------------------------


@torch.no_grad()
def stage_cfg_sweep(args, model, vggt_builder, dataset, train_args, picked, out_dir: Path):
    include_sharp = bool(train_args.get("include_sharp_label") or False)
    geometry_only = bool(train_args.get("geometry_only", False))
    align_mode = args.align_mode or train_args.get("align_mode", "cross")
    device = next(model.parameters()).device
    scales = [float(s) for s in args.cfg_scales]

    stage_dir = out_dir / "cfg_sweep"
    stage_dir.mkdir(parents=True, exist_ok=True)
    per_object = []

    for j, sel in enumerate(picked):
        it = prepare_item(
            dataset, sel["idx"], model, vggt_builder, device,
            include_sharp, align_mode, geometry_only,
        )
        if it["weak"] is None:
            continue
        torch.manual_seed(args.seed + sel["idx"])
        noise = torch.randn_like(it["z"])
        cd_recon, _, _, _ = decode_cd(model, it["z"], it["gt_xyz"])
        row = {"tag": sel["tag"], "mesh": it["mesh"], "recon_cd": cd_recon}
        for gs in scales:
            z_gen = model.sample_latents(
                it["weak"],
                batch_size=1,
                num_steps=args.sample_steps,
                guidance_scale=gs,
                noise=noise,
                context_keep=it["keep"],
            cam_cond=it.get("cam"),
                device=device,
                dtype=it["z"].dtype,
                renorm_output=False,
            )
            cd_g, _, _, _ = decode_cd(model, z_gen, it["gt_xyz"])
            row[f"gen_cd_cfg{gs:g}"] = cd_g
            row[f"rel_cfg{gs:g}"] = rel_err(z_gen, it["z"])
        per_object.append(row)
        logger.info(
            "[cfg %d/%d] %s " + " ".join(f"cfg{gs:g}={row[f'gen_cd_cfg{gs:g}']:.4f}" for gs in scales),
            j + 1, len(picked), Path(it["mesh"]).stem[:12],
        )

    summary = {"num_objects": len(per_object)}
    for gs in scales:
        summary[f"gen_cd_cfg{gs:g}"] = _mean([r[f"gen_cd_cfg{gs:g}"] for r in per_object])
        summary[f"rel_cfg{gs:g}"] = _mean([r[f"rel_cfg{gs:g}"] for r in per_object])
    with open(stage_dir / "cfg_sweep.json", "w") as f:
        json.dump({"summary": summary, "per_object": per_object}, f, indent=2)

    print("\n" + "=" * 72)
    print("CFG SWEEP")
    print("=" * 72)
    for k, v in summary.items():
        print(f"  {k}: {v}" if k == "num_objects" else f"  {k:28s} {v:.5f}")
    print("=" * 72 + "\n")
    return summary


# ---------------------------------------------------------------------------
# Stage: multiview_eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def stage_multiview_eval(args, model, vggt_builder, dataset, train_args, picked, out_dir: Path):
    """Re-evaluate the same meshes at several views (view-sensitivity test)."""
    include_sharp = bool(train_args.get("include_sharp_label") or False)
    geometry_only = bool(train_args.get("geometry_only", False))
    align_mode = args.align_mode or train_args.get("align_mode", "cross")
    device = next(model.parameters()).device
    views = [int(v) for v in args.eval_views]

    # Build path→list of (idx, view) in dataset.samples
    samples = getattr(dataset, "samples", [])
    by_stem: Dict[str, List[Tuple[int, int]]] = {}
    for i, (path, vid) in enumerate(samples):
        by_stem.setdefault(Path(path).stem, []).append((i, int(vid)))

    stage_dir = out_dir / "multiview_eval"
    stage_dir.mkdir(parents=True, exist_ok=True)
    per_object = []

    for j, sel in enumerate(picked):
        stem = Path(sel["mesh"]).stem
        candidates = by_stem.get(stem, [])
        # Prefer exact views from --eval_views; fall back to whatever is cached
        view_to_idx = {vid: idx for idx, vid in candidates}
        row = {"tag": sel["tag"], "mesh": sel["mesh"], "views": {}}
        for vid in views:
            if vid not in view_to_idx:
                continue
            it = prepare_item(
                dataset, view_to_idx[vid], model, vggt_builder, device,
                include_sharp, align_mode, geometry_only,
            )
            if it["weak"] is None:
                continue
            torch.manual_seed(args.seed + view_to_idx[vid])
            noise = torch.randn_like(it["z"])
            cd_r, _, _, _ = decode_cd(model, it["z"], it["gt_xyz"])
            z_gen = model.sample_latents(
                it["weak"],
                batch_size=1,
                num_steps=args.sample_steps,
                guidance_scale=1.0,
                noise=noise,
                context_keep=it["keep"],
            cam_cond=it.get("cam"),
                device=device,
                dtype=it["z"].dtype,
                renorm_output=False,
            )
            cd_g, _, _, _ = decode_cd(model, z_gen, it["gt_xyz"])
            row["views"][str(vid)] = {
                "recon_cd": cd_r,
                "gen_cd": cd_g,
                "rel_latent_err": rel_err(z_gen, it["z"]),
            }
        if row["views"]:
            gens = [v["gen_cd"] for v in row["views"].values()]
            row["gen_cd_best"] = min(gens)
            row["gen_cd_worst"] = max(gens)
            row["gen_cd_mean"] = sum(gens) / len(gens)
            row["view_sensitive"] = (row["gen_cd_worst"] / max(row["gen_cd_best"], 1e-12)) > 2.0
            per_object.append(row)
            logger.info(
                "[mview %d/%d] %s best=%.4f worst=%.4f mean=%.4f sensitive=%s",
                j + 1, len(picked), stem[:12],
                row["gen_cd_best"], row["gen_cd_worst"], row["gen_cd_mean"],
                row["view_sensitive"],
            )

    summary = {
        "num_objects": len(per_object),
        "gen_cd_best": _mean([r["gen_cd_best"] for r in per_object]),
        "gen_cd_worst": _mean([r["gen_cd_worst"] for r in per_object]),
        "gen_cd_mean": _mean([r["gen_cd_mean"] for r in per_object]),
        "frac_view_sensitive": _mean(
            [1.0 if r["view_sensitive"] else 0.0 for r in per_object]
        ),
    }
    with open(stage_dir / "multiview_eval.json", "w") as f:
        json.dump({"summary": summary, "per_object": per_object}, f, indent=2)

    print("\n" + "=" * 72)
    print("MULTIVIEW EVAL")
    print("=" * 72)
    for k, v in summary.items():
        print(f"  {k}: {v}" if k == "num_objects" else f"  {k:28s} {v:.5f}")
    print(
        "\nfrac_view_sensitive ≫ 0 ⇒ some failures are view-specific "
        "(multi-view conditioning may help)."
    )
    print("=" * 72 + "\n")
    return summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run(args):
    t_wall = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    model, vggt_builder, train_args = load_unite_model(
        args.ckpt,
        device,
        overrides={"sample_renorm_output": False},
    )
    model.eval()
    vggt_builder.eval()

    dataset, include_sharp, align_mode = build_dataset(args, train_args)
    ranked = load_eval_ranking(args.eval_results)
    picked = select_mesh_indices(
        dataset,
        ranked,
        n_worst=args.n_worst,
        n_best=args.n_best,
        n_random=args.n_random,
        seed=args.seed,
        view_idx=args.view_idx,
    )
    with open(out_dir / "selected_objects.json", "w") as f:
        json.dump(picked, f, indent=2)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    stages = args.stages or list(ALL_STAGES)
    results = {"ckpt": args.ckpt, "stages": {}}
    logger.info(
        "Debug start | ckpt=%s | stages=%s | n_selected=%d | tok_vggt=%s",
        args.ckpt,
        stages,
        len(picked),
        getattr(model, "tokenizer_use_weak_context", False),
    )

    for stage in stages:
        logger.info("===== STAGE: %s =====", stage)
        t0 = time.time()
        if stage == "data_check":
            results["stages"][stage] = stage_data_check(
                args, model, vggt_builder, dataset, train_args, picked, out_dir
            )
        elif stage == "diffusion_curve":
            results["stages"][stage] = stage_diffusion_curve(
                args, model, vggt_builder, dataset, train_args, picked, out_dir
            )
        elif stage == "encode_vggt":
            results["stages"][stage] = stage_encode_vggt(
                args, model, vggt_builder, dataset, train_args, picked, out_dir
            )
        elif stage == "latent_align":
            results["stages"][stage] = stage_latent_align(
                args, model, vggt_builder, dataset, train_args, picked, out_dir
            )
        elif stage == "pca_trajectories":
            results["stages"][stage] = stage_pca_trajectories(
                args, model, vggt_builder, dataset, train_args, picked, out_dir
            )
        elif stage == "cfg_sweep":
            results["stages"][stage] = stage_cfg_sweep(
                args, model, vggt_builder, dataset, train_args, picked, out_dir
            )
        elif stage == "multiview_eval":
            results["stages"][stage] = stage_multiview_eval(
                args, model, vggt_builder, dataset, train_args, picked, out_dir
            )
        else:
            raise ValueError(f"Unknown stage {stage!r}; choose from {ALL_STAGES}")
        logger.info("Stage %s done in %.1fs", stage, time.time() - t0)

    results["wall_seconds"] = time.time() - t_wall
    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("All done in %.1fs → %s", results["wall_seconds"], out_dir)
    return results


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--gobjaverse_render_root", type=str, default=None)
    p.add_argument("--vggt_cache_root", type=str, default=None)
    p.add_argument("--eval_results", type=str, default=None, help="Prior eval results.json for worst/best ranking")
    p.add_argument("--categories", type=str, default=None)
    p.add_argument("--no_experiment_manifest", action="store_true")
    p.add_argument("--align_mode", type=str, default=None)
    p.add_argument("--view_idx", type=int, default=0)
    p.add_argument("--max_items", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--stages",
        nargs="+",
        default=list(ALL_STAGES),
        choices=list(ALL_STAGES),
    )
    p.add_argument("--n_worst", type=int, default=5)
    p.add_argument("--n_best", type=int, default=5)
    p.add_argument("--n_random", type=int, default=10)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--num_draws", type=int, default=6)
    p.add_argument(
        "--t_grid",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
    )
    p.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
    )
    p.add_argument("--intruder_thresh", type=float, default=0.02)
    # pca_trajectories
    p.add_argument("--num_latent_draws", type=int, default=2, help="Independent encode draws per object")
    p.add_argument("--num_noise_draws", type=int, default=2, help="Independent ε draws per latent")
    p.add_argument("--pca_max_objects", type=int, default=8, help="Cap objects for PCA stage")
    p.add_argument("--waypoint_stride", type=int, default=5, help="Keep every N-th ODE waypoint")
    # cfg_sweep
    p.add_argument(
        "--cfg_scales",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 3.0],
    )
    # multiview_eval
    p.add_argument(
        "--eval_views",
        type=int,
        nargs="+",
        default=[0, 5, 10, 15, 20],
        help="Views to score in multiview_eval (must be cached)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
