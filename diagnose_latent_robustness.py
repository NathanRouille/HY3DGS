#!/usr/bin/env python3
"""Locate the source of the generation gap in a ShapePCUnite checkpoint.

``gen_cd >> recon_cd`` has three candidate explanations that the standard eval
cannot separate:

A. **Target stochasticity** — the tokenizer latent ``z`` is a random variable
   (fresh ``randn`` register seed + stochastic FPS every call), so flow matching
   regresses its conditional *mean*.  If the decoder is only valid on individual
   samples and not on their mean, no amount of flow training can close the gap.

B. **Decoder brittleness** — the decoder never saw a perturbed latent during
   training (``representation_noising`` off), so a tiny off-manifold error may be
   amplified into a large Chamfer error.

C. **Flow inaccuracy** — the ODE genuinely lands far from the encoded latent.

The script measures each, then applies the decisive test: take the flow's
*observed* latent error and read off what Chamfer error an isotropic
perturbation of that size would produce.  If observed ``gen_cd`` matches that
prediction, the gap is (A)+(B) and the fix is on the autoencoder side.  If
``gen_cd`` is far worse, the flow error is *structured* and the fix is on the
flow/conditioning side.

Usage::

    python diagnose_latent_robustness.py \
        --ckpt runs/<run>/ckpt_0030000.pt \
        --data_dir /export/home/nathan/datasets/gobjaverse_experiments/furniture_351/train \
        --gobjaverse_render_root /export/home/nathan/datasets \
        --vggt_cache_root runs/vggt_cache/furn4_mv40_cam_cross \
        --output_dir runs/<run>/diag_latent
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from hy3dgen.shapegen.models.autoencoders.shape_pc_ae import ShapePCAE
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


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative Frobenius distance ``||a - b|| / ||b||``."""
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def token_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-register cosine similarity between two latents [1, R, D]."""
    return float(
        torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()
    )


@torch.no_grad()
def decode_cd(model, z: torch.Tensor, gt_xyz: torch.Tensor):
    xyz, _rgb, centers = model.decode(z, representation_phase=False)
    cd, _, _ = chamfer_distance(xyz, gt_xyz)
    return float(cd), centers


@torch.no_grad()
def encode_repeats(
    model,
    surface: torch.Tensor,
    *,
    k: int,
    deterministic_fps: bool,
    fixed_register_noise: bool,
) -> List[torch.Tensor]:
    """``k`` tokenizer latents under a controlled noise source.

    ``deterministic_fps`` freezes the point subset + FPS seed; combined with
    ``fixed_register_noise`` the encoder becomes a pure function of the surface,
    which is the reference for measuring each noise source in isolation.
    """
    prev = model.encoder.deterministic
    model.encoder.deterministic = deterministic_fps
    try:
        reg = None
        if fixed_register_noise:
            reg = torch.randn(
                surface.shape[0],
                model.num_registers,
                model.embed_dim,
                device=surface.device,
                dtype=surface.dtype,
            )
        out = []
        for _ in range(k):
            z, _fps = model.encode(surface, register_noise=reg)
            out.append(z)
        return out
    finally:
        model.encoder.deterministic = prev


def spread_stats(zs: Sequence[torch.Tensor]) -> Dict[str, float]:
    """Dispersion of a set of latents about their mean."""
    stack = torch.cat(zs, dim=0)
    mean = stack.mean(dim=0, keepdim=True)
    rels = [rel_err(z, mean) for z in zs]
    coss = [token_cosine(z, mean) for z in zs]
    return {
        "rel_spread_mean": sum(rels) / len(rels),
        "rel_spread_max": max(rels),
        "token_cos_to_mean": sum(coss) / len(coss),
        # Pairwise distance between two independent draws: the scale of error a
        # flow regressing the mean would have to explain away.
        "rel_pairwise": rel_err(zs[0], zs[1]) if len(zs) > 1 else float("nan"),
    }


@torch.no_grad()
def sigma_sweep(
    model,
    z: torch.Tensor,
    gt_xyz: torch.Tensor,
    sigmas: Sequence[float],
) -> List[Dict[str, float]]:
    """Chamfer error of ``decode(latent_norm(z + sigma * eps))``.

    ``sample_latents`` applies ``latent_norm`` to the ODE output, so the
    renormalized branch is the faithful analogue of a generated latent; the raw
    branch is reported to show the effect is not a norm artefact.
    """
    rows = []
    for s in sigmas:
        eps = torch.randn_like(z)
        z_raw = z + s * eps
        z_norm = model.latent_norm(z_raw)
        cd_raw, _ = decode_cd(model, z_raw, gt_xyz)
        cd_norm, _ = decode_cd(model, z_norm, gt_xyz)
        rows.append(
            {
                "sigma": float(s),
                "rel_err_raw": rel_err(z_raw, z),
                "rel_err_norm": rel_err(z_norm, z),
                "cd_raw": cd_raw,
                "cd_norm": cd_norm,
            }
        )
    return rows


def interp_cd_at(rows: Sequence[Dict[str, float]], rel_target: float) -> float:
    """Linear interpolation of Chamfer error at a given relative latent error.

    Uses the *raw* (not renormalized) branch: it is a clean isotropic
    perturbation of the reference latent, whereas the renormalized branch also
    carries the double-``latent_norm`` offset.
    """
    pts = sorted((r["rel_err_raw"], r["cd_raw"]) for r in rows)
    if rel_target <= pts[0][0]:
        return pts[0][1]
    if rel_target >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= rel_target <= x1:
            w = (rel_target - x0) / max(x1 - x0, 1e-12)
            return y0 + w * (y1 - y0)
    return pts[-1][1]


@torch.no_grad()
def diagnose(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, vggt_builder, train_args = load_unite_model(args.ckpt, device)
    include_sharp = bool(train_args.get("include_sharp_label") or False)
    align_mode = args.align_mode or train_args.get("align_mode", "cross")
    trained_det = bool(train_args.get("deterministic_encoder", True))

    data_path = Path(args.data_dir).resolve()
    manifest = (
        load_experiment_manifest(str(data_path))
        if not args.no_experiment_manifest
        else None
    )
    dataset = build_surface_render_dataset(
        str(data_path),
        max_items=args.max_items,
        categories=resolve_category_ids(args.categories),
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
    n = min(len(dataset), args.max_eval_items)
    logger.info(
        "Diagnosing %d objects | trained deterministic_encoder=%s | rep_noising=%s",
        n,
        trained_det,
        train_args.get("representation_noising"),
    )
    if args.seed is not None:
        torch.manual_seed(args.seed)

    sigmas = [float(s) for s in args.sigmas]
    per_object: List[Dict] = []

    for i in range(n):
        batch = collate_surface_render([dataset[i]])
        surface = batch["surface"].to(device)
        gt_xyz, _gt_rgb = ShapePCAE.surface_gt_points(
            surface, include_sharp_label=include_sharp
        )
        weak, keep, cam = _build_weak_context(
            batch, vggt_builder, device, align_mode=align_mode,
            include_camera_in_sequence=not bool(getattr(model, "adaln_camera_cond", False)),
        )

        rec: Dict[str, object] = {"mesh": batch["mesh_path"][0]}

        # --- A. tokenizer stochasticity, decomposed by noise source -----------
        conditions = {
            # register seed only (point subset + FPS frozen)
            "register": dict(deterministic_fps=True, fixed_register_noise=False),
            # point subset + FPS only (register seed frozen)
            "fps": dict(deterministic_fps=False, fixed_register_noise=True),
            # exactly what training saw
            "as_trained": dict(
                deterministic_fps=trained_det, fixed_register_noise=False
            ),
        }
        for name, cfg in conditions.items():
            zs = encode_repeats(model, surface, k=args.num_draws, **cfg)
            cds = [decode_cd(model, z, gt_xyz)[0] for z in zs]
            z_mean = torch.cat(zs, dim=0).mean(dim=0, keepdim=True)
            cd_mean_latent, _ = decode_cd(model, z_mean, gt_xyz)
            cd_mean_renorm, _ = decode_cd(model, model.latent_norm(z_mean), gt_xyz)
            rec[f"tok/{name}"] = {
                **spread_stats(zs),
                "cd_samples_mean": sum(cds) / len(cds),
                "cd_samples_max": max(cds),
                # The quantity a mean-regressing flow would achieve at best.
                "cd_of_mean_latent": cd_mean_latent,
                "cd_of_mean_latent_renorm": cd_mean_renorm,
            }

        # Reference latent for everything below: one as-trained draw, matching
        # what evaluate_pc_unite.py scores as recon.
        z_ref, _fps = model.encode(surface)
        cd_ref, centers_ref = decode_cd(model, z_ref, gt_xyz)
        rec["recon_cd"] = cd_ref

        # Cost of a *second* latent_norm on an already-normalized latent. This
        # is the transform sample_latents applies to the ODE endpoint, so it
        # bounds how much of the generation gap is pure normalization artefact.
        cd_double_ln, _ = decode_cd(model, model.latent_norm(z_ref), gt_xyz)
        rec["double_layernorm"] = {
            "cd": cd_double_ln,
            "rel_latent_err": rel_err(model.latent_norm(z_ref), z_ref),
            "x_recon": cd_double_ln / max(cd_ref, 1e-12),
        }

        # --- B. decoder tolerance to off-manifold latents ---------------------
        rec["sigma_sweep"] = sigma_sweep(model, z_ref, gt_xyz, sigmas)

        # --- C. what the flow actually produces ------------------------------
        if weak is not None:
            noise = torch.randn_like(z_ref)
            flow_rows = {}
            # ``renorm`` = current shipped behaviour, ``raw`` = UNITE behaviour
            # (decode the ODE endpoint directly).
            for renorm in (True, False):
                tag = "renorm" if renorm else "raw"
                flow_rows[f"gen_{tag}"] = model.sample_latents(
                    weak,
                    batch_size=1,
                    num_steps=args.sample_steps,
                    guidance_scale=1.0,
                    noise=noise,
                    context_keep=keep,
                    cam_cond=cam,
                    device=device,
                    dtype=z_ref.dtype,
                    renorm_output=renorm,
                )
                for t0 in args.oracle_t_starts:
                    flow_rows[f"oracle_t{t0:g}_{tag}"] = model.sample_latents(
                        weak,
                        batch_size=1,
                        num_steps=args.sample_steps,
                        guidance_scale=1.0,
                        z_init=z_ref,
                        t_start=float(t0),
                        noise=noise,
                        context_keep=keep,
                        cam_cond=cam,
                        device=device,
                        dtype=z_ref.dtype,
                        renorm_output=renorm,
                    )

            for name, z_out in flow_rows.items():
                cd_out, centers_out = decode_cd(model, z_out, gt_xyz)
                r = rel_err(z_out, z_ref)
                rec[f"flow/{name}"] = {
                    "cd": cd_out,
                    "rel_latent_err": r,
                    "token_cos": token_cosine(z_out, z_ref),
                    # Mean anchor displacement: distinguishes "same anchors,
                    # slightly moved" from "anchors scrambled".
                    "anchor_shift": float(
                        (centers_out - centers_ref).norm(dim=-1).mean()
                    ),
                    # Decisive test: CD that isotropic noise of the same size
                    # would cause. cd_out >> this => structured flow error.
                    "cd_predicted_from_sigma_curve": interp_cd_at(
                        rec["sigma_sweep"], r
                    ),
                }

        per_object.append(rec)
        logger.info(
            "[%d/%d] %s recon=%.5f gen_renorm=%.5f gen_raw=%.5f doubleLN=%.5f "
            "cd_of_mean=%.5f rel_pairwise=%.3f",
            i + 1,
            n,
            Path(rec["mesh"]).stem,
            rec["recon_cd"],
            rec.get("flow/gen_renorm", {}).get("cd", float("nan")),
            rec.get("flow/gen_raw", {}).get("cd", float("nan")),
            rec["double_layernorm"]["cd"],
            rec["tok/as_trained"]["cd_of_mean_latent"],
            rec["tok/as_trained"]["rel_pairwise"],
        )

    summary = _summarize(per_object, sigmas)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "diagnosis.json", "w") as f:
        json.dump({"summary": summary, "per_object": per_object}, f, indent=2)
    _report(summary, sigmas)
    logger.info("Wrote %s", out_dir / "diagnosis.json")
    return summary


def _mean(vals: Sequence[float]) -> float:
    vals = [v for v in vals if v == v]  # drop NaN
    return sum(vals) / len(vals) if vals else float("nan")


def _summarize(per_object: List[Dict], sigmas: Sequence[float]) -> Dict:
    keys_tok = ("register", "fps", "as_trained")
    summary: Dict[str, object] = {
        "num_objects": len(per_object),
        "recon_cd": _mean([r["recon_cd"] for r in per_object]),
    }
    for name in keys_tok:
        rows = [r[f"tok/{name}"] for r in per_object if f"tok/{name}" in r]
        if not rows:
            continue
        summary[f"tok/{name}"] = {
            k: _mean([row[k] for row in rows]) for k in rows[0]
        }
    sweep = {}
    for j, s in enumerate(sigmas):
        rows = [r["sigma_sweep"][j] for r in per_object]
        sweep[f"{s:g}"] = {k: _mean([row[k] for row in rows]) for k in rows[0]}
    summary["sigma_sweep"] = sweep
    dln = [r["double_layernorm"] for r in per_object if "double_layernorm" in r]
    if dln:
        summary["double_layernorm"] = {
            k: _mean([row[k] for row in dln]) for k in dln[0]
        }
    flow_names = sorted(
        {k[5:] for r in per_object for k in r if k.startswith("flow/")}
    )
    for name in flow_names:
        rows = [r[f"flow/{name}"] for r in per_object if f"flow/{name}" in r]
        summary[f"flow/{name}"] = {
            k: _mean([row[k] for row in rows]) for k in rows[0]
        }
    return summary


def _report(summary: Dict, sigmas: Sequence[float]) -> None:
    print("\n" + "=" * 78)
    print("LATENT ROBUSTNESS DIAGNOSIS")
    print("=" * 78)
    print(f"objects: {summary['num_objects']}   recon_cd: {summary['recon_cd']:.5f}")

    print("\n--- A. tokenizer latent stochasticity (is the flow target a point?) ---")
    print(
        f"{'noise source':<14}{'rel spread':>12}{'cos':>8}"
        f"{'CD(samples)':>13}{'CD(mean z)':>12}"
    )
    for name in ("register", "fps", "as_trained"):
        s = summary.get(f"tok/{name}")
        if not s:
            continue
        print(
            f"{name:<14}{s['rel_spread_mean']:>12.4f}{s['token_cos_to_mean']:>8.4f}"
            f"{s['cd_samples_mean']:>13.5f}{s['cd_of_mean_latent']:>12.5f}"
        )

    base = summary["recon_cd"]
    dln = summary.get("double_layernorm")
    if dln:
        print(
            f"\n--- B0. cost of a second latent_norm (what sample_latents does) ---"
            f"\n  CD {dln['cd']:.5f} = {dln['x_recon']:.1f}x recon, "
            f"rel latent err {dln['rel_latent_err']:.4f}"
        )

    print("\n--- B. decoder tolerance to off-manifold latents ---")
    print(
        f"{'sigma':>8}{'rel(raw)':>10}{'CD(raw)':>11}{'x rec':>7}"
        f"{'rel(norm)':>11}{'CD(norm)':>11}{'x rec':>7}"
    )
    for s in sigmas:
        row = summary["sigma_sweep"][f"{s:g}"]
        print(
            f"{s:>8.3f}{row['rel_err_raw']:>10.4f}{row['cd_raw']:>11.5f}"
            f"{row['cd_raw'] / max(base, 1e-12):>7.1f}"
            f"{row['rel_err_norm']:>11.4f}{row['cd_norm']:>11.5f}"
            f"{row['cd_norm'] / max(base, 1e-12):>7.1f}"
        )

    print("\n--- C. flow output vs the sigma curve (decisive test) ---")
    print(
        f"{'variant':<14}{'CD':>10}{'rel err':>10}{'cos':>8}"
        f"{'anchor dx':>11}{'CD pred':>10}{'excess':>9}"
    )
    for key in sorted(k for k in summary if k.startswith("flow/")):
        s = summary[key]
        pred = s["cd_predicted_from_sigma_curve"]
        print(
            f"{key[5:]:<14}{s['cd']:>10.5f}{s['rel_latent_err']:>10.4f}"
            f"{s['token_cos']:>8.4f}{s['anchor_shift']:>11.5f}"
            f"{pred:>10.5f}{s['cd'] / max(pred, 1e-12):>9.2f}x"
        )
    print("=" * 78 + "\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--gobjaverse_render_root", type=str, default=None)
    p.add_argument("--vggt_cache_root", type=str, default=None)
    p.add_argument("--categories", type=str, default=None)
    p.add_argument("--no_experiment_manifest", action="store_true")
    p.add_argument("--align_mode", type=str, default=None)
    p.add_argument("--view_idx", type=int, default=0)
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--max_eval_items", type=int, default=8)
    p.add_argument("--num_draws", type=int, default=6)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument(
        "--oracle_t_starts", type=float, nargs="+", default=[0.5, 0.9, 0.99]
    )
    p.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    diagnose(parse_args())
