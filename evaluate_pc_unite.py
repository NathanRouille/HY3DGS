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
import hashlib
import json
import logging
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from hy3dgen.shapegen.cam_align import aligned_centers_from_payload
from hy3dgen.shapegen.gs_export import export_xyz_pointcloud_ply
from hy3dgen.shapegen.models.autoencoders.shape_pc_ae import ShapePCAE
from hy3dgen.shapegen.models.autoencoders.shape_pc_unite import ShapePCUnite
from hy3dgen.shapegen.pc_debug_export import (
    collect_vggt_debug_clouds,
    export_recon_debug_plys,
    export_vggt_debug_plys,
)
from hy3dgen.shapegen.pc_losses import chamfer_distance, rgb_l1_on_nn
from hy3dgen.shapegen.pc_render_dataset import (
    build_internscenes_render_dataset,
    build_surface_render_dataset,
    collate_surface_render,
)
from hy3dgen.shapegen.pc_unite_diagnostics import DEFAULT_T_GRID, run_eval_diagnostics
from hy3dgen.shapegen.pretrained_profiles import resolve_include_sharp_label
from hy3dgen.shapegen.surface_loaders import stable_mesh_seed
from hy3dgen.shapegen.surflo_global_camera import (
    SurfloGlobalCameraBranch,
    extract_raw_vggt_camera_tokens,
)
from hy3dgen.shapegen.vggt_context import VGGTContextBuilder, load_state_dict_skip_mismatch
from train_gs_ae import load_experiment_manifest, resolve_category_ids
from train_pc_unite import _build_weak_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _mesh_id(mesh_path: str) -> str:
    """Stable object id (Objaverse glb stem)."""
    return Path(mesh_path).stem


def _derived_seed(base_seed: int, mesh_path: str, tag: str) -> int:
    digest = hashlib.sha256(
        f"{int(base_seed)}:{os.path.realpath(os.path.abspath(mesh_path))}:{tag}".encode()
    ).hexdigest()
    return int(digest[:8], 16)


def _seeded_randn(
    shape: Sequence[int],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) % (2**31 - 1))
    return torch.randn(shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)


def _git_fingerprint(repo: Path) -> Dict[str, object]:
    out: Dict[str, object] = {"repo": str(repo)}
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        out["git_head"] = head
        out["git_dirty"] = bool(dirty)
    except Exception:
        out["git_head"] = None
        out["git_dirty"] = None
    return out


def _mean_unique_mesh(rows: List[Dict], key: str) -> Optional[float]:
    """Mean of ``key`` after averaging duplicate rows that share the same mesh id."""
    by_id: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if key not in r:
            continue
        v = r[key]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        mid = str(r.get("mesh_id") or _mesh_id(str(r.get("mesh", ""))))
        by_id[mid].append(float(v))
    if not by_id:
        return None
    per_mesh = [sum(vs) / len(vs) for vs in by_id.values()]
    return sum(per_mesh) / len(per_mesh)


def _rgb_chw_to_u8(rgb: torch.Tensor) -> "np.ndarray":
    import numpy as np

    rgb = rgb.detach().float().cpu()
    if rgb.dim() != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB, got shape {tuple(rgb.shape)}")
    return (rgb.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)


def _batch_view_ids(batch: Dict, sample_index: int = 0) -> List[int]:
    """Ordered conditioning view ids for one batch item (multi- or single-view)."""
    if "view_indices" in batch:
        vids = batch["view_indices"][sample_index]
        if torch.is_tensor(vids):
            return [int(v) for v in vids.detach().cpu().tolist()]
        return [int(v) for v in vids]
    if "view_idx" in batch:
        v = batch["view_idx"][sample_index]
        return [int(v.item() if torch.is_tensor(v) else v)]
    return [0]


def _save_view_rgbs(
    batch: Dict,
    obj_dir: Path,
    *,
    sample_index: int = 0,
) -> Dict[str, str]:
    """Write all conditioning renders for one object.

    Multi-view batches (``rgb_views`` ``[B,S,3,H,W]``) write
    ``view_rgb_{vid:02d}.png`` for every view plus ``view_rgb.png`` as an alias
    of the reference (first) view. Single-view batches write the same pair from
    ``rgb`` / ``view_idx``.
    """
    from PIL import Image

    written: Dict[str, str] = {}
    obj_dir = Path(obj_dir)
    obj_dir.mkdir(parents=True, exist_ok=True)
    view_ids = _batch_view_ids(batch, sample_index)

    frames: List[torch.Tensor] = []
    if "rgb_views" in batch:
        rgb_views = batch["rgb_views"][sample_index]  # [S,3,H,W]
        if not torch.is_tensor(rgb_views) or rgb_views.dim() != 4:
            return written
        frames = [rgb_views[s] for s in range(int(rgb_views.shape[0]))]
    elif "rgb" in batch:
        rgb = batch["rgb"][sample_index]
        if not torch.is_tensor(rgb) or rgb.dim() != 3:
            return written
        frames = [rgb]
    else:
        return written

    if len(view_ids) < len(frames):
        view_ids = view_ids + list(range(len(view_ids), len(frames)))
    view_ids = view_ids[: len(frames)]

    for vid, frame in zip(view_ids, frames):
        path = obj_dir / f"view_rgb_{int(vid):02d}.png"
        Image.fromarray(_rgb_chw_to_u8(frame)).save(path)
        written[f"view_rgb_{int(vid):02d}"] = str(path)

    # Backward-compatible alias: reference view (first in conditioning order).
    if frames:
        ref_path = obj_dir / "view_rgb.png"
        Image.fromarray(_rgb_chw_to_u8(frames[0])).save(ref_path)
        written["view_rgb"] = str(ref_path)
    return written


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
        point_feats=int(args.get("point_feats", 7)),
        downsample_ratio=int(args.get("downsample_ratio", 20)),
        num_points_per_anchor=int(args.get("num_points_per_anchor", 8)),
        deterministic_encoder=bool(args.get("deterministic_encoder", True)),
        register_noise_mode=str(args.get("register_noise_mode", "random")),
        geometry_only=bool(args.get("geometry_only", False)),
        sample_renorm_output=bool(args.get("sample_renorm_output", False)),
        tokenizer_use_weak_context=bool(args.get("tokenizer_use_weak_context", False)),
        adaln_camera_cond=bool(args.get("adaln_camera_cond", False)),
        use_surflo_global_cam=bool(args.get("surflo_global_cam", False)),
        max_anchor_delta=(
            float(args["max_anchor_delta"])
            if args.get("max_anchor_delta") not in (None, 0, 0.0)
            else None
        ),
        qk_norm=bool(args.get("qk_norm", True)),
        use_rope=bool(args.get("use_rope", False)),
        flow_steps_per_recon=int(args.get("flow_steps_per_recon", 8)),
        flow_loss_type=str(args.get("flow_loss_type", "velocity")),
        modulation_recon_timestep_max=float(args.get("modulation_recon_timestep_max", 0.01)),
        noising_t_start=float(args.get("noising_t_start", 0.9)),
        weak_context_dropout=float(args.get("weak_context_dropout", 0.1)),
        use_lognorm=bool(args.get("use_lognorm", False)),
        lognorm_mu=float(args.get("lognorm_mu", -0.8)),
        lognorm_sigma=float(args.get("lognorm_sigma", 0.8)),
        t0_force_prob=float(args.get("t0_force_prob", 0.0)),
        timestep_shift_alpha=float(args.get("timestep_shift_alpha", 0.0)),
    )
    if bool(args.get("surflo_global_cam", False)):
        branch = SurfloGlobalCameraBranch.from_surflo_checkpoint(
            str(args.get("surflo_ckpt", "/export/home/nathan/Surflo/checkpoints/surflo_v0.pt")),
            width=width,
            surflo_root=str(args.get("surflo_root", "/export/home/nathan/Surflo")),
            freeze_backbone=True,
        )
        model.attach_surflo_global_cam(branch)
    if args.get("freeze_recon") or any(
        k.startswith("tokenizer_ge.") for k in ckpt.get("model", {})
    ):
        # Allocate tokenizer_ge before load so ckpt keys are applied (not dropped).
        if model.tokenizer_ge is None:
            model.freeze_tokenizer_ge(replace=True)
    ckpt_has_w = any(
        k.endswith("t_embedder.W") for k in ckpt.get("model", {})
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if "denoiser_pos_embed" not in ckpt.get("model", {}):
        model.sync_denoiser_pos_embed_from_tokenizer()
        logger.info(
            "ckpt missing denoiser_pos_embed — initialized from register_pos_embed"
        )
    if model.tokenizer_ge is not None:
        model.ensure_tokenizer_ge_eval()
    if missing:
        logger.warning("model: %d missing keys (e.g. %s)", len(missing), missing[:3])
    if not ckpt_has_w:
        logger.warning(
            "Checkpoint has no ge.t_embedder.W; using W from manual_seed(%s) "
            "before model init (must match the seed used at training start).",
            seed,
        )
    model.to(device).eval()

    ds_kind = str(args.get("dataset", "gobjaverse"))
    builder = VGGTContextBuilder(
        width=width, mask_white_bg=(ds_kind != "internscenes")
    )
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
    if rgb is None or gt_rgb is None:
        return xyz, rgb, centers, float(cd), float("nan")
    rgb_l1, _ = rgb_l1_on_nn(rgb, gt_rgb, idx_p2t, idx_tgt_to_pred=idx_t2p)
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
            "sample_renorm_output": args.sample_renorm_output,
        },
    )

    include_sharp = bool(train_args.get("include_sharp_label") or False)
    geometry_only = bool(train_args.get("geometry_only", False))
    categories = resolve_category_ids(args.categories)
    data_path = Path(args.data_dir).resolve()
    manifest = load_experiment_manifest(str(data_path)) if not args.no_experiment_manifest else None
    align_mode = args.align_mode or train_args.get("align_mode", "cross")
    views_per_sample = int(
        getattr(args, "views_per_sample", None)
        or train_args.get("views_per_sample", 1)
        or 1
    )
    use_joint_cache = bool(
        getattr(args, "vggt_joint_cache", False)
        or train_args.get("vggt_joint_cache", False)
    )
    eval_view_indices = None
    if views_per_sample > 1:
        from hy3dgen.shapegen.gobjaverse_gt import parse_view_indices

        # Prefer explicit CLI --view_indices (fixed eval pair). Else use train
        # args; if train used a larger pool, take the first views_per_sample
        # and warn (eval must be deterministic).
        cli_views = getattr(args, "view_indices", None)
        if cli_views:
            eval_view_indices = parse_view_indices(
                view_idx=args.view_idx,
                num_views=None,
                view_indices=cli_views,
            )
        else:
            eval_view_indices = parse_view_indices(
                view_idx=args.view_idx,
                num_views=train_args.get("num_views"),
                view_indices=train_args.get("view_indices"),
            )
        if len(eval_view_indices) < views_per_sample:
            raise ValueError(
                f"Need >= {views_per_sample} eval view indices, got {eval_view_indices}"
            )
        dataset_kind_pre = args.dataset or train_args.get("dataset", "gobjaverse")
        if len(eval_view_indices) > views_per_sample:
            if dataset_kind_pre != "internscenes":
                logger.warning(
                    "Eval view pool %s is larger than views_per_sample=%d; "
                    "using fixed first-%d ordered tuple %s (pass --view_indices "
                    "to choose a different fixed pair)",
                    eval_view_indices,
                    views_per_sample,
                    views_per_sample,
                    eval_view_indices[:views_per_sample],
                )
                eval_view_indices = eval_view_indices[:views_per_sample]

    # Primary eval: single-view (view_idx) unless the ckpt used multi-view c_meanrms.
    surface_seed = None if args.no_surface_seed else int(args.seed)
    dataset_kind = args.dataset or train_args.get("dataset", "gobjaverse")
    if dataset_kind == "internscenes":
        room_ids = None
        if getattr(args, "internscenes_room_ids", None):
            room_ids = [
                x.strip()
                for x in str(args.internscenes_room_ids).split(",")
                if x.strip()
            ]
        elif train_args.get("internscenes_room_ids"):
            room_ids = [
                x.strip()
                for x in str(train_args["internscenes_room_ids"]).split(",")
                if x.strip()
            ]
        dataset = build_internscenes_render_dataset(
            str(data_path),
            split=str(
                getattr(args, "internscenes_split", None)
                or train_args.get("internscenes_split", "train")
            ),
            room_ids=room_ids,
            max_items=args.max_items,
            include_sharp_label=include_sharp,
            view_indices=eval_view_indices,
            views_per_sample=views_per_sample,
            view_sample_mode="first",
            vggt_cache_root=args.vggt_cache_root,
            use_joint_vggt_cache=use_joint_cache,
            pc_size=int(train_args.get("pc_size", 5120)),
            pc_sharpedge_size=int(train_args.get("pc_sharpedge_size", 5120)),
            seed=surface_seed,
            surface_in_camera_frame=not train_args.get(
                "no_surface_camera_frame", False
            ),
            align_mode=align_mode,
            strict_load=bool(args.strict_load),
        )
    else:
        dataset = build_surface_render_dataset(
            str(data_path),
            max_items=args.max_items,
            categories=categories,
            include_sharp_label=include_sharp,
            use_experiment_manifest=not args.no_experiment_manifest,
            manifest=manifest,
            render_root=args.gobjaverse_render_root,
            view_idx=args.view_idx,
            view_indices=eval_view_indices,
            views_per_sample=views_per_sample,
            view_sample_mode="first",
            vggt_cache_root=args.vggt_cache_root,
            use_joint_vggt_cache=use_joint_cache,
            pc_size=int(train_args.get("pc_size", 5120)),
            pc_sharpedge_size=int(train_args.get("pc_sharpedge_size", 5120)),
            seed=surface_seed,
            gobjaverse_normalization=not train_args.get(
                "no_gobjaverse_normalization", False
            ),
            surface_in_camera_frame=not train_args.get(
                "no_surface_camera_frame", False
            ),
            align_mode=align_mode,
            strict_load=bool(args.strict_load),
        )
    logger.info(
        "Eval dataset: %d samples (views_per_sample=%d, align_mode=%s, "
        "view_indices=%s, joint_cache=%s, surface_seed=%s, strict_load=%s)",
        len(dataset),
        views_per_sample,
        align_mode,
        eval_view_indices if eval_view_indices is not None else [args.view_idx],
        use_joint_cache,
        surface_seed,
        bool(args.strict_load),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    n = min(len(dataset), args.max_eval_items or len(dataset))
    if n < 2:
        logger.warning("Need >= 2 objects for swapped/cross-object baselines")

    gen_seeds = [int(s) for s in (args.gen_seeds or [0])]
    primary_gen_seed = gen_seeds[0]

    # Cache per-object tensors once: the ablation needs another object's context.
    items: List[Dict] = []
    failed_loads: List[Dict] = []
    for i in range(n):
        try:
            sample = dataset[i]
        except Exception as e:
            mesh_hint = view_hint = None
            try:
                mesh_hint, view_hint = dataset.samples[i]
            except Exception:
                pass
            failed_loads.append(
                {"index": i, "mesh": mesh_hint, "view": view_hint, "error": str(e)}
            )
            logger.error("Excluding failed eval sample idx=%d: %s", i, e)
            continue
        batch = collate_surface_render([sample])
        mesh_path = batch["mesh_path"][0]
        mid = _mesh_id(mesh_path)
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
        raw_cams = (
            extract_raw_vggt_camera_tokens(batch, device)
            if getattr(model, "use_surflo_global_cam", False)
            else None
        )
        tok_seed = _derived_seed(int(args.seed), mesh_path, "tokenizer")
        torch.manual_seed(tok_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(tok_seed)
        register_noise = _seeded_randn(
            (1, model.num_registers, model.embed_dim),
            seed=_derived_seed(int(args.seed), mesh_path, "register"),
            device=device,
            dtype=surface.dtype,
        )
        z, fps_xyz = model.encode(
            surface,
            weak_context=weak,
            weak_context_keep=keep,
            register_noise=register_noise,
        )
        patch_centers = patch_keep = None
        if "vggt_cache" in batch and batch["vggt_cache"]:
            payload = batch["vggt_cache"][0]
            if isinstance(payload, dict):
                _raw, patch_centers, patch_keep = aligned_centers_from_payload(
                    payload, align_mode
                )
                del _raw
        view_ids = _batch_view_ids(batch, 0)
        noise = _seeded_randn(
            tuple(z.shape),
            seed=_derived_seed(int(args.seed), mesh_path, f"gen|{primary_gen_seed}"),
            device=device,
            dtype=z.dtype,
        )
        items.append(
            {
                "mesh": mesh_path,
                "mesh_id": mid,
                "surface_seed": (
                    None
                    if surface_seed is None
                    else stable_mesh_seed(surface_seed, mesh_path)
                ),
                "tokenizer_seed": tok_seed,
                "gen_seed_tag": primary_gen_seed,
                "view_idx": view_ids[0] if view_ids else 0,
                "view_indices": view_ids,
                "batch": batch,
                "surface": surface,
                "gt_xyz": gt_xyz,
                "gt_rgb": gt_rgb,
                "weak": weak,
                "keep": keep,
                "cam": cam,
                "raw_camera_tokens": raw_cams,
                "z": z,
                "fps_xyz": fps_xyz,
                "noise": noise,
                "register_noise": register_noise,
                "patch_centers": patch_centers,
                "patch_keep": patch_keep,
            }
        )
        logger.info(
            "prepared %d/%d %s views=%s (unique_so_far=%d)",
            len(items),
            n,
            mid,
            view_ids,
            len({it["mesh_id"] for it in items}),
        )

    if failed_loads:
        fail_path = out_dir / "failed_loads.json"
        with open(fail_path, "w") as f:
            json.dump(failed_loads, f, indent=2)
        logger.warning(
            "Excluded %d failed loads (see %s); evaluated %d objects",
            len(failed_loads),
            fail_path,
            len(items),
        )

    has_ctx = bool(items) and all(it["weak"] is not None for it in items)
    guidance_scales = args.guidance_scales or [1.0]
    stash_diag_traj = bool(
        args.diagnostics
        and has_ctx
        and (args.diag_path or args.diag_oracle or args.diag_velocity or args.diag_pca)
    )
    per_object: List[Dict] = []
    n_eval = len(items)

    for i, it in enumerate(items):
        gt_xyz, gt_rgb, z, noise = it["gt_xyz"], it["gt_rgb"], it["z"], it["noise"]
        rec: Dict[str, object] = {
            "mesh": it["mesh"],
            "mesh_id": it["mesh_id"],
            "view_idx": it.get("view_idx", 0),
            "view_indices": list(it.get("view_indices") or [it.get("view_idx", 0)]),
            "surface_seed": it.get("surface_seed"),
            "tokenizer_seed": it.get("tokenizer_seed"),
            "gen_seed_tag": it.get("gen_seed_tag"),
        }
        clouds: Dict[str, tuple] = {}

        xyz_r, rgb_r, centers_r, cd_r, rgb_lr = _decode_and_score(model, z, gt_xyz, gt_rgb)
        rec["recon_cd"], rec["recon_rgb"] = cd_r, rgb_lr
        clouds["recon"] = (xyz_r, rgb_r)

        if n_eval > 1:
            other = items[(i + 1) % n_eval]
            if other["mesh_id"] == it["mesh_id"] and n_eval > 2:
                other = items[(i + 2) % n_eval]
            xyz_o, _, _, cd_o, _ = _decode_and_score(model, other["z"], gt_xyz, gt_rgb)
            rec["cross_object_cd"] = cd_o
            rec["cross_object_mesh_id"] = other["mesh_id"]
            clouds["cross_object"] = (xyz_o, None)

        if has_ctx:
            n_tok = it["weak"].shape[1]
            null_ctx = model.null_context(1, n_tok, dtype=z.dtype, device=device)

            z_null = model.sample_latents(
                null_ctx, batch_size=1, num_steps=args.sample_steps,
                guidance_scale=1.0, noise=noise, device=device, dtype=z.dtype,
                cam_cond=None,
                raw_camera_tokens=None,
            )
            xyz_n, rgb_n, _, cd_n, _ = _decode_and_score(model, z_null, gt_xyz, gt_rgb)
            rec["gen_null_cd"] = cd_n
            clouds["gen_null"] = (xyz_n, rgb_n)

            z_orc = model.sample_latents(
                it["weak"], batch_size=1, num_steps=args.sample_steps,
                guidance_scale=1.0, z_init=z, t_start=args.oracle_t_start,
                noise=noise,
                context_keep=it["keep"],
                cam_cond=it.get("cam"),
                raw_camera_tokens=it.get("raw_camera_tokens"),
                device=device, dtype=z.dtype,
            )
            _, _, _, cd_orc, _ = _decode_and_score(model, z_orc, gt_xyz, gt_rgb)
            rec[f"oracle_t{args.oracle_t_start:g}_cd"] = cd_orc
            it["diag_oracle_endpoints"] = {float(args.oracle_t_start): z_orc}
            it["diag_oracle_noise"] = noise

            for gs in guidance_scales:
                tag = f"cfg{gs:g}"
                take_traj = stash_diag_traj and abs(float(gs) - 1.0) < 1e-9
                if take_traj:
                    z_gen, traj0, t_grid0 = model.sample_latents(
                        it["weak"], batch_size=1, num_steps=args.sample_steps,
                        guidance_scale=gs, noise=noise,
                        context_keep=it["keep"],
                        cam_cond=it.get("cam"),
                        raw_camera_tokens=it.get("raw_camera_tokens"),
                        device=device, dtype=z.dtype,
                        return_trajectory=True,
                    )
                    it["diag_z_gen"] = traj0[-1]
                    it["diag_traj0"] = traj0
                    it["diag_t_grid0"] = t_grid0
                else:
                    z_gen = model.sample_latents(
                        it["weak"], batch_size=1, num_steps=args.sample_steps,
                        guidance_scale=gs, noise=noise,
                        context_keep=it["keep"],
                        cam_cond=it.get("cam"),
                        raw_camera_tokens=it.get("raw_camera_tokens"),
                        device=device, dtype=z.dtype,
                    )
                xyz_g, rgb_g, _, cd_g, rgb_lg = _decode_and_score(model, z_gen, gt_xyz, gt_rgb)
                rec[f"gen_cd_{tag}"], rec[f"gen_rgb_{tag}"] = cd_g, rgb_lg
                clouds[f"gen_{tag}"] = (xyz_g, rgb_g)

                if len(gen_seeds) > 1:
                    seed_cds = [cd_g]
                    seed_rgbs = [rgb_lg]
                    for extra_seed in gen_seeds[1:]:
                        noise_e = _seeded_randn(
                            tuple(z.shape),
                            seed=_derived_seed(
                                int(args.seed), it["mesh"], f"gen|{extra_seed}"
                            ),
                            device=device,
                            dtype=z.dtype,
                        )
                        z_e = model.sample_latents(
                            it["weak"], batch_size=1, num_steps=args.sample_steps,
                            guidance_scale=gs, noise=noise_e,
                            context_keep=it["keep"],
                            cam_cond=it.get("cam"),
                            raw_camera_tokens=it.get("raw_camera_tokens"),
                            device=device, dtype=z.dtype,
                        )
                        _, _, _, cd_e, rgb_e = _decode_and_score(
                            model, z_e, gt_xyz, gt_rgb
                        )
                        seed_cds.append(cd_e)
                        seed_rgbs.append(rgb_e)
                    mean_cd = sum(seed_cds) / len(seed_cds)
                    rec[f"gen_cd_{tag}_mean_seeds"] = mean_cd
                    rec[f"gen_cd_{tag}_seed_std"] = (
                        sum((x - mean_cd) ** 2 for x in seed_cds)
                        / max(len(seed_cds) - 1, 1)
                    ) ** 0.5
                    finite_rgbs = [x for x in seed_rgbs if x == x]
                    if finite_rgbs:
                        rec[f"gen_rgb_{tag}_mean_seeds"] = sum(finite_rgbs) / len(
                            finite_rgbs
                        )

                if n_eval > 1:
                    other = items[(i + 1) % n_eval]
                    if other["mesh_id"] == it["mesh_id"] and n_eval > 2:
                        other = items[(i + 2) % n_eval]
                    z_sw = model.sample_latents(
                        other["weak"], batch_size=1, num_steps=args.sample_steps,
                        guidance_scale=gs, noise=noise,
                        context_keep=other["keep"],
                        cam_cond=other.get("cam"),
                        raw_camera_tokens=other.get("raw_camera_tokens"),
                        device=device, dtype=z.dtype,
                    )
                    xyz_s, rgb_s, _, cd_s, _ = _decode_and_score(model, z_sw, gt_xyz, gt_rgb)
                    rec[f"gen_swapped_cd_{tag}"] = cd_s
                    clouds[f"gen_swapped_{tag}"] = (xyz_s, rgb_s)

        if args.export_ply:
            obj_dir = out_dir / f"obj_{i:04d}_{Path(it['mesh']).stem[:12]}"
            obj_dir.mkdir(parents=True, exist_ok=True)
            view_pngs = _save_view_rgbs(it["batch"], obj_dir, sample_index=0)
            if view_pngs:
                rec["view_rgb"] = view_pngs.get("view_rgb")
                rec["view_rgbs"] = {
                    k: v for k, v in view_pngs.items() if k.startswith("view_rgb_")
                }
            export_xyz_pointcloud_ply(
                gt_xyz[0].cpu(),
                obj_dir / "gt.ply",
                colors=None if gt_rgb is None else gt_rgb[0].cpu(),
            )
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
                    gt_rgb=gt_rgb,
                    recon_rgb=rgb_r if rgb_r is not None else None,
                    fps_xyz=it["fps_xyz"],
                    centers=centers_r,
                    num_points_per_anchor=int(model.num_points_per_anchor),
                    max_anchor_delta=model.max_anchor_delta,
                    patch_centers=it.get("patch_centers"),
                    patch_keep=it.get("patch_keep"),
                    intruder_thresh=float(args.intruder_thresh),
                )
                try:
                    vggt_clouds = collect_vggt_debug_clouds(
                        vggt_builder,
                        it["batch"],
                        align_mode=align_mode,
                        device=device,
                        sample_index=0,
                    )
                    if vggt_clouds:
                        export_vggt_debug_plys(obj_dir, vggt_clouds)
                except Exception as e:
                    logger.warning("VGGT debug export failed for %s: %s", it["mesh"], e)

        per_object.append(rec)
        logger.info(
            "obj %d/%d %s recon_cd=%.5f gen_cd=%.5f null=%.5f swapped=%.5f cross=%.5f",
            i + 1, n_eval, it["mesh_id"],
            rec["recon_cd"],
            rec.get(f"gen_cd_cfg{guidance_scales[0]:g}", float("nan")),
            rec.get("gen_null_cd", float("nan")),
            rec.get(f"gen_swapped_cd_cfg{guidance_scales[0]:g}", float("nan")),
            rec.get("cross_object_cd", float("nan")),
        )

    def _metric_val(v: object) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    keys = sorted(
        {
            k
            for r in per_object
            for k, v in r.items()
            if k not in ("mesh", "mesh_id", "cross_object_mesh_id") and _metric_val(v)
        }
    )
    summary = {
        k: sum(float(r[k]) for r in per_object if k in r and _metric_val(r[k]))
        / max(sum(1 for r in per_object if k in r and _metric_val(r[k])), 1)
        for k in keys
    }
    summary_unique = {}
    for k in keys:
        mu = _mean_unique_mesh(per_object, k)
        if mu is not None:
            summary_unique[k] = mu
    mesh_ids = [str(r.get("mesh_id") or _mesh_id(str(r["mesh"]))) for r in per_object]
    unique_ids = sorted(set(mesh_ids))
    dup_ids = sorted({m for m in mesh_ids if mesh_ids.count(m) > 1})
    summary["num_eval"] = n_eval
    summary["num_eval_requested"] = n
    summary["num_unique_meshes"] = len(unique_ids)
    summary["duplicate_mesh_ids"] = dup_ids
    summary["failed_loads"] = len(failed_loads)
    summary["sample_steps"] = args.sample_steps
    summary["euler_updates"] = max(int(args.sample_steps) - 1, 0)
    summary["guidance_scales"] = guidance_scales
    summary["protocol"] = "primary_deterministic"
    summary["surface_seed"] = surface_seed
    summary["eval_seed"] = int(args.seed)
    summary["gen_seeds"] = gen_seeds
    summary["strict_load"] = bool(args.strict_load)
    summary["unique_mesh_means"] = summary_unique
    for k, v in summary_unique.items():
        summary[f"unique/{k}"] = v

    verdict = {}
    for gs in guidance_scales:
        tag = f"cfg{gs:g}"
        gen = summary_unique.get(f"gen_cd_{tag}", summary.get(f"gen_cd_{tag}"))
        if gen is None:
            continue
        null = summary_unique.get("gen_null_cd", summary.get("gen_null_cd", float("nan")))
        swap = summary_unique.get(
            f"gen_swapped_cd_{tag}", summary.get(f"gen_swapped_cd_{tag}", float("nan"))
        )
        verdict[tag] = {
            "gen_cd": gen,
            "null_gain": null - gen,
            "swap_gain": swap - gen,
            "vs_cross_object": summary_unique.get(
                "cross_object_cd", summary.get("cross_object_cd", float("nan"))
            )
            - gen,
            "conditioning_used": bool(gen < 0.9 * null and gen < 0.9 * swap),
        }
    summary["verdict"] = verdict

    metadata = {
        "ckpt": str(args.ckpt),
        "data_dir": str(data_path),
        "output_dir": str(out_dir),
        "view_indices": eval_view_indices if eval_view_indices is not None else [args.view_idx],
        "views_per_sample": views_per_sample,
        "align_mode": align_mode,
        "vggt_cache_root": args.vggt_cache_root,
        "vggt_joint_cache": use_joint_cache,
        "sample_steps": int(args.sample_steps),
        "euler_updates": max(int(args.sample_steps) - 1, 0),
        "guidance_scales": guidance_scales,
        "oracle_t_start": float(args.oracle_t_start),
        "point_budget_pc_size": int(train_args.get("pc_size", 5120)),
        "point_budget_pc_sharpedge_size": int(train_args.get("pc_sharpedge_size", 5120)),
        "sample_renorm_output": bool(getattr(model, "sample_renorm_output", False)),
        "eval_seed": int(args.seed),
        "surface_seed": surface_seed,
        "gen_seeds": gen_seeds,
        "strict_load": bool(args.strict_load),
        "num_eval_requested": n,
        "num_eval": n_eval,
        "num_unique_meshes": len(unique_ids),
        "unique_mesh_ids": unique_ids,
        "duplicate_mesh_ids": dup_ids,
        "failed_loads": failed_loads,
        "worktree": _git_fingerprint(Path(__file__).resolve().parent),
    }
    with open(out_dir / "eval_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    diag_summary = None
    if args.diagnostics and has_ctx:
        logger.info("Running latent diagnostics under %s/diagnostics", out_dir)
        diag_summary = run_eval_diagnostics(
            model,
            items,
            out_dir=out_dir,
            sample_steps=args.sample_steps,
            device=device,
            t_grid=args.diag_t_grid or list(DEFAULT_T_GRID),
            pca_max_objects=int(args.diag_pca_max_objects),
            seed=int(args.seed or 0),
            run_path=bool(args.diag_path),
            run_oracle=bool(args.diag_oracle),
            run_velocity=bool(args.diag_velocity),
            run_pca=bool(args.diag_pca),
        )
        summary["diagnostics"] = diag_summary.get("flat", {})
        for k, v in (diag_summary.get("flat") or {}).items():
            summary[k] = v

    with open(out_dir / "results.json", "w") as f:
        json.dump(
            {
                "summary": summary,
                "per_object": per_object,
                "diagnostics": diag_summary,
                "metadata": metadata,
            },
            f,
            indent=2,
        )
    logger.info(
        "Summary (unique meshes=%d / rows=%d): %s",
        len(unique_ids),
        n_eval,
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "unique/recon_cd",
                    "unique/gen_cd_cfg1",
                    "unique/gen_cd_cfg2",
                    "unique/gen_cd_cfg3",
                    "num_unique_meshes",
                    "duplicate_mesh_ids",
                    "failed_loads",
                )
                if k in summary
            },
            indent=2,
        ),
    )
    return summary



def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ShapePCUnite")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=("gobjaverse", "internscenes"),
        help="Defaults to value stored in checkpoint args.",
    )
    p.add_argument("--internscenes_split", type=str, default=None)
    p.add_argument("--internscenes_room_ids", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="runs/pc_unite_eval")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gen_seeds",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Generation noise seed tags (derived with --seed + mesh id). "
            "Primary metrics use the first seed; extra seeds add mean/std fields."
        ),
    )
    p.add_argument(
        "--strict_load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If a sample fails to load, exclude it and record the failure "
            "(default). Pass --no-strict_load for the old silent next-sample fallback."
        ),
    )
    p.add_argument(
        "--no_surface_seed",
        action="store_true",
        help=(
            "Leave surface subsample RNG unseeded (legacy non-deterministic GT). "
            "Default is to pass --seed into the surface loader."
        ),
    )
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--max_eval_items", type=int, default=None)
    p.add_argument("--categories", type=str, default=None)
    p.add_argument("--no_experiment_manifest", action="store_true")
    p.add_argument("--gobjaverse_render_root", type=str, default=None)
    p.add_argument("--vggt_cache_root", type=str, default=None)
    p.add_argument(
        "--vggt_joint_cache",
        action="store_true",
        help="Use joint multi-view VGGT cache (must match train / --view_indices).",
    )
    p.add_argument(
        "--view_idx",
        type=int,
        default=0,
        help="Primary eval view (single-view; default 0 even for multi-view-trained ckpts).",
    )
    p.add_argument(
        "--view_indices",
        type=str,
        default=None,
        help=(
            'Fixed multi-view eval pair/tuple, e.g. "4,16". Overrides the '
            "checkpoint's train pool; required for a deterministic pair when "
            "the ckpt trained on a larger joint pool."
        ),
    )
    p.add_argument(
        "--views_per_sample",
        type=int,
        default=None,
        help="Override ckpt views_per_sample (default: from train args; >1 uses fixed ordered views).",
    )
    p.add_argument(
        "--sample_renorm_output",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override the checkpoint's ODE-endpoint normalization. The flow "
            "already targets normalized latents, so the default second "
            "application costs ~4x recon on the oracle; pass "
            "--no-sample_renorm_output to score any existing checkpoint the way "
            "UNITE does."
        ),
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
    p.add_argument("--export_ply", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Export PLYs plus conditioning render(s) per object: "
            "view_rgb_{vid:02d}.png for every view and view_rgb.png (ref alias)."
        ),
    )
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
        choices=("cross", "fair_gobK", "c_meanrms"),
        help="Override ckpt align_mode (default: use train args from checkpoint).",
    )
    p.add_argument(
        "--diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write latent diagnostics under output_dir/diagnostics/ "
            "(path Curve A, oracle Curve B, on-ODE velocity cosine, PCA). "
            "Disable with --no-diagnostics for a fast CD-only eval."
        ),
    )
    p.add_argument(
        "--diag_path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Curve A: ODE from t=0 vs linear interpolant (euclid + cosine, per-obj + global).",
    )
    p.add_argument(
        "--diag_oracle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Curve B: ODE started at each t_grid; endpoint vs target z and mean encode "
            "(euclid + cosine, per-obj + global)."
        ),
    )
    p.add_argument(
        "--diag_velocity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "On-ODE velocity cosine: cos(v_pred, (z−x_t)/(1−t)) at every waypoint; "
            "scalar = mean over the curve (per-obj + global)."
        ),
    )
    p.add_argument(
        "--diag_pca",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PCA trajectory plots (reuses Curve B ODE trajectories for a subset).",
    )
    p.add_argument(
        "--diag_t_grid",
        type=float,
        nargs="*",
        default=None,
        help="t_start grid for Curve B / PCA (default: 0 0.1 0.25 0.5 0.75 0.9 1).",
    )
    p.add_argument(
        "--diag_pca_max_objects",
        type=int,
        default=8,
        help="Cap objects for PCA plots (Curve A/B still use all objects).",
    )
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
