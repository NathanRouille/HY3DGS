#!/usr/bin/env python3
"""Align predictions and export PLYs + metrics.

Default methods: ``gt_anchored`` (Surflo) + ``robust_filter`` (NOVA).
Also writes ``pred_aligned/primary/`` with per-model best recipe.

Run in ``hy3dgs`` from HY3DGS repo root::

    python compare_baselines/align_and_score.py \\
      --compare_dir runs/compare_exp14_baselines/val \\
      --identity_exp14

Debug one object::

    python compare_baselines/align_and_score.py \\
      --compare_dir runs/compare_exp14_baselines/val \\
      --identity_exp14 --index 0

Exports per method folder::

    pred_aligned/<method>/{exp14,nova3r,surflo}.ply          # full cloud
    pred_aligned/<method>/{model}_core.ply                   # robust ICP subset
    pred_aligned/<method>/{model}_filtered.ply               # near-GT crop

``pred_aligned/primary/`` — exp14 identity, Surflo gt_anchored, NOVA robust_filter.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from compare_baselines.align_metrics import (
    ALIGN_METHOD_INFO,
    DEFAULT_ALIGN_METHODS,
    PRIMARY_RECIPE,
    AlignResult,
    align_pred_to_gt,
    chamfer_l2,
    export_filter_kwargs,
    filter_near_gt,
    fscore,
    one_way_l2,
)
from compare_baselines.ply_io import read_ply_xyz, subsample_points, write_ply

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("align_and_score")

PRED_METHODS = ("exp14", "nova3r", "surflo")
PRED_RGB = {
    "exp14": (220, 64, 64),
    "nova3r": (80, 200, 80),
    "surflo": (220, 160, 40),
}


def _agg_init(align_names: List[str]) -> Dict:
    out = {}
    for an in align_names:
        out[an] = {
            pm: {
                "cd_l2": [],
                "cd_l2sq": [],
                "accuracy": [],
                "completeness": [],
                "scale": [],
                "n_filtered": [],
            }
            for pm in PRED_METHODS
        }
    return out


def _summarize(vals: List[float]) -> Dict:
    if not vals:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def _identity_align(pred: np.ndarray, gt: np.ndarray) -> AlignResult:
    fk = export_filter_kwargs("gt_anchored")
    filt = filter_near_gt(pred, gt, **fk)
    return AlignResult(
        scale=1.0,
        R=np.eye(3),
        t=np.zeros(3),
        aligned=pred,
        aligned_filtered=filt,
        aligned_core=None,
        aligned_full=pred,
        n_filtered=int(filt.shape[0]),
        name="identity",
        flip_name="identity",
        n_denoised=int(pred.shape[0]),
        n_raw=int(pred.shape[0]),
    )


def _compute_align(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    pm: str,
    method: str,
    identity_exp14: bool,
    n_eval_points: int,
    seed: int,
) -> AlignResult:
    if pm == "exp14" and identity_exp14:
        return _identity_align(pred, gt)
    if method == "identity":
        return _identity_align(pred, gt)
    return align_pred_to_gt(
        pred,
        gt,
        method=method,
        pred_method=pm,
        n_align=n_eval_points,
        seed=seed,
    )


def _write_align_exports(
    out_dir: Path,
    pm: str,
    align: AlignResult,
) -> None:
    """Write PLY/NPZ; remove stale ``*_full.ply`` from older runs."""
    write_ply(out_dir / f"{pm}.ply", align.aligned, rgb=PRED_RGB[pm])
    if align.aligned_core is not None and align.aligned_core.shape[0] > 0:
        write_ply(out_dir / f"{pm}_core.ply", align.aligned_core, rgb=PRED_RGB[pm])
    if align.aligned_filtered is not None and align.aligned_filtered.shape[0] > 0:
        write_ply(
            out_dir / f"{pm}_filtered.ply",
            align.aligned_filtered,
            rgb=PRED_RGB[pm],
        )
    stale = out_dir / f"{pm}_full.ply"
    if stale.is_file():
        stale.unlink()
    np.savez(
        out_dir / f"{pm}_transform.npz",
        scale=align.scale,
        R=align.R,
        t=align.t,
        flip_name=np.asarray(align.flip_name),
        n_denoised=align.n_denoised,
        n_raw=align.n_raw,
    )


def _metrics_for_align(
    align: AlignResult,
    pred: np.ndarray,
    gt_eval: np.ndarray,
    *,
    compare_dir: Path,
    out_dir: Path,
    pm: str,
    score_on_filtered: bool,
    n_eval_points: int,
    seed: int,
    fscore_thr: List[float],
) -> Tuple[Dict, Dict[str, List[float]]]:
    """Return per-object metrics dict and lists to append into aggregates."""
    score_cloud = (
        align.aligned_filtered
        if score_on_filtered
        and align.aligned_filtered is not None
        and align.aligned_filtered.shape[0] >= 32
        else align.aligned
    )
    pred_eval = subsample_points(score_cloud, n_eval_points, seed=seed)
    cd = chamfer_l2(pred_eval, gt_eval, squared=False)
    cd_sq = chamfer_l2(pred_eval, gt_eval, squared=True)
    acc = one_way_l2(pred_eval, gt_eval, squared=False)
    comp = one_way_l2(gt_eval, pred_eval, squared=False)

    m = {
        "n_pred": int(pred.shape[0]),
        "n_raw": int(align.n_raw or pred.shape[0]),
        "n_core": int(
            align.aligned_core.shape[0]
            if align.aligned_core is not None
            else align.n_denoised
        ),
        "n_scored": int(score_cloud.shape[0]),
        "n_filtered": int(align.n_filtered),
        "n_denoised": int(align.n_denoised),
        "flip_name": str(align.flip_name),
        "scale": float(align.scale),
        "cd_l2": float(cd),
        "cd_l2sq": float(cd_sq),
        "accuracy": float(acc),
        "completeness": float(comp),
        "aligned_ply": str((out_dir / f"{pm}.ply").relative_to(compare_dir)),
        "core_ply": str((out_dir / f"{pm}_core.ply").relative_to(compare_dir))
        if (out_dir / f"{pm}_core.ply").is_file()
        else None,
        "filtered_ply": str(
            (out_dir / f"{pm}_filtered.ply").relative_to(compare_dir)
        )
        if (out_dir / f"{pm}_filtered.ply").is_file()
        else None,
    }
    agg_append: Dict[str, List[float]] = {
        "cd_l2": [float(cd)],
        "cd_l2sq": [float(cd_sq)],
        "accuracy": [float(acc)],
        "completeness": [float(comp)],
        "scale": [float(align.scale)],
        "n_filtered": [float(align.n_filtered)],
    }
    for thr in fscore_thr:
        fs = fscore(pred_eval, gt_eval, thr)
        m[f"fscore@{thr}"] = float(fs)
        agg_append[f"fscore@{thr}"] = [float(fs)]
    return m, agg_append


def _select_objects(
    objects: List[dict],
    *,
    max_objects: Optional[int],
    indices: Optional[List[int]],
    uid: Optional[str],
) -> List[dict]:
    selected = list(objects)
    if indices is not None:
        want = set(int(i) for i in indices)
        selected = [o for o in selected if int(o["index"]) in want]
        missing = want - {int(o["index"]) for o in selected}
        if missing:
            raise SystemExit(f"No objects with index in {sorted(missing)}")
    if uid is not None:
        needle = uid.strip().lower()
        selected = [
            o
            for o in selected
            if needle in str(o["uid"]).lower()
            or needle in str(o.get("obj_dir", "")).lower()
        ]
        if not selected:
            raise SystemExit(f"No objects matching --uid {uid!r}")
    if max_objects is not None:
        selected = selected[: max_objects]
    return selected


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compare_dir", required=False, default=None)
    p.add_argument("--n_eval_points", type=int, default=8192)
    p.add_argument("--fscore_thr", type=float, nargs="*", default=[0.05, 0.02, 0.01])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--align_methods",
        type=str,
        nargs="*",
        default=list(DEFAULT_ALIGN_METHODS),
        help=(
            "Alignment recipes to export/score. Default: all. "
            f"Choices: {', '.join(ALIGN_METHOD_INFO)}"
        ),
    )
    p.add_argument(
        "--list_align_methods",
        action="store_true",
        help="Print available alignment methods and exit.",
    )
    p.add_argument(
        "--identity_exp14",
        action="store_true",
        help="For exp14 only: skip alignment (already GT camera frame).",
    )
    p.add_argument(
        "--max_objects",
        type=int,
        default=None,
        help="Only process the first N objects after other filters (debug).",
    )
    p.add_argument(
        "--index",
        type=int,
        nargs="+",
        default=None,
        help="Process only these manifest object indices (e.g. --index 0 or --index 0 3).",
    )
    p.add_argument(
        "--uid",
        type=str,
        default=None,
        help="Process only objects whose uid (or obj_dir) contains this substring.",
    )
    p.add_argument(
        "--score_on_filtered",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score on near-GT filtered clouds (default: on). Use --no-score_on_filtered for core.",
    )
    p.add_argument(
        "--no_export_primary",
        action="store_true",
        help="Skip pred_aligned/primary/ (per-model best recipe folder).",
    )
    args = p.parse_args()

    if args.list_align_methods:
        for k, v in ALIGN_METHOD_INFO.items():
            print(f"  {k:22s}  {v}")
        return

    if not args.compare_dir:
        raise SystemExit("--compare_dir is required (unless --list_align_methods)")

    align_names = list(args.align_methods)
    unknown = [a for a in align_names if a not in ALIGN_METHOD_INFO]
    if unknown:
        raise SystemExit(f"Unknown align methods: {unknown}")

    compare_dir = Path(args.compare_dir).resolve()
    manifest = json.loads((compare_dir / "manifest.json").read_text())
    objects = _select_objects(
        manifest["objects"],
        max_objects=args.max_objects,
        indices=args.index,
        uid=args.uid,
    )
    subset_run = (
        args.max_objects is not None
        or args.index is not None
        or args.uid is not None
    )
    logger.info("Processing %d object(s)%s", len(objects), " [subset/debug]" if subset_run else "")

    export_primary = not args.no_export_primary
    score_align_names = list(align_names)
    if export_primary and "primary" not in score_align_names:
        score_align_names.append("primary")

    aggregates = _agg_init(score_align_names)
    for an in score_align_names:
        for pm in PRED_METHODS:
            for thr in args.fscore_thr:
                aggregates[an][pm][f"fscore@{thr}"] = []

    per_object = []

    for i, entry in enumerate(objects):
        obj_dir = compare_dir / entry["obj_dir"]
        gt_path = compare_dir / entry["gt_ply"]
        gt = read_ply_xyz(gt_path)
        gt_eval = subsample_points(gt, args.n_eval_points, seed=args.seed)
        obj_seed = args.seed + int(entry["index"])

        obj_metrics = {
            "uid": entry["uid"],
            "index": entry["index"],
            "n_gt": int(gt.shape[0]),
            "alignments": {},
        }
        align_cache: Dict[Tuple[str, str], AlignResult] = {}
        preds: Dict[str, np.ndarray] = {}

        for an in align_names:
            out_dir = obj_dir / "pred_aligned" / an
            out_dir.mkdir(parents=True, exist_ok=True)
            write_ply(out_dir / "gt.ply", gt, rgb=(40, 120, 255))
            obj_metrics["alignments"][an] = {}

            for pm in PRED_METHODS:
                raw_path = obj_dir / "pred_raw" / f"{pm}.ply"
                if not raw_path.is_file():
                    logger.warning("Missing %s", raw_path)
                    obj_metrics["alignments"][an][pm] = {"error": "missing_pred"}
                    continue

                pred = read_ply_xyz(raw_path)
                preds[pm] = pred
                try:
                    align = _compute_align(
                        pred,
                        gt,
                        pm=pm,
                        method=an,
                        identity_exp14=args.identity_exp14,
                        n_eval_points=args.n_eval_points,
                        seed=obj_seed,
                    )
                except Exception as e:
                    logger.exception(
                        "Align failed %s / %s / %s: %s", entry["uid"], an, pm, e
                    )
                    obj_metrics["alignments"][an][pm] = {"error": str(e)}
                    continue

                align_cache[(an, pm)] = align
                _write_align_exports(out_dir, pm, align)
                m, agg_append = _metrics_for_align(
                    align,
                    pred,
                    gt_eval,
                    compare_dir=compare_dir,
                    out_dir=out_dir,
                    pm=pm,
                    score_on_filtered=args.score_on_filtered,
                    n_eval_points=args.n_eval_points,
                    seed=args.seed + 17 + int(entry["index"]),
                    fscore_thr=args.fscore_thr,
                )
                for k, vals in agg_append.items():
                    aggregates[an][pm][k].extend(vals)
                obj_metrics["alignments"][an][pm] = m
                logger.info(
                    "[%d/%d] idx=%s %-12s %-20s %-7s  CD=%.4f acc=%.4f comp=%.4f "
                    "scale=%.3f raw=%d core=%d filt=%d orient=%s",
                    i + 1,
                    len(objects),
                    entry["index"],
                    entry["uid"][:12],
                    an,
                    pm,
                    m["cd_l2"],
                    m["accuracy"],
                    m["completeness"],
                    align.scale,
                    align.n_raw or pred.shape[0],
                    m["n_core"],
                    align.n_filtered,
                    align.flip_name,
                )

        if export_primary:
            primary_dir = obj_dir / "pred_aligned" / "primary"
            primary_dir.mkdir(parents=True, exist_ok=True)
            write_ply(primary_dir / "gt.ply", gt, rgb=(40, 120, 255))
            obj_metrics["alignments"]["primary"] = {}
            recipe_note = {
                pm: PRIMARY_RECIPE[pm] for pm in PRED_METHODS
            }
            obj_metrics["alignments"]["primary"]["_recipe"] = recipe_note

            for pm in PRED_METHODS:
                raw_path = obj_dir / "pred_raw" / f"{pm}.ply"
                if not raw_path.is_file():
                    obj_metrics["alignments"]["primary"][pm] = {"error": "missing_pred"}
                    continue
                recipe = PRIMARY_RECIPE[pm]
                pred = preds[pm] if pm in preds else read_ply_xyz(raw_path)
                cache_key = (recipe, pm) if recipe != "identity" else None
                try:
                    if recipe == "identity":
                        align = _identity_align(pred, gt)
                    elif cache_key in align_cache:
                        align = align_cache[cache_key]
                    else:
                        align = _compute_align(
                            pred,
                            gt,
                            pm=pm,
                            method=recipe,
                            identity_exp14=args.identity_exp14,
                            n_eval_points=args.n_eval_points,
                            seed=obj_seed,
                        )
                except Exception as e:
                    logger.exception(
                        "Primary align failed %s / %s: %s", entry["uid"], pm, e
                    )
                    obj_metrics["alignments"]["primary"][pm] = {"error": str(e)}
                    continue

                _write_align_exports(primary_dir, pm, align)
                m, agg_append = _metrics_for_align(
                    align,
                    pred,
                    gt_eval,
                    compare_dir=compare_dir,
                    out_dir=primary_dir,
                    pm=pm,
                    score_on_filtered=args.score_on_filtered,
                    n_eval_points=args.n_eval_points,
                    seed=args.seed + 17 + int(entry["index"]),
                    fscore_thr=args.fscore_thr,
                )
                m["recipe"] = recipe
                for k, vals in agg_append.items():
                    aggregates["primary"][pm][k].extend(vals)
                obj_metrics["alignments"]["primary"][pm] = m
                logger.info(
                    "[%d/%d] idx=%s %-12s %-20s %-7s  CD=%.4f (recipe=%s)",
                    i + 1,
                    len(objects),
                    entry["index"],
                    entry["uid"][:12],
                    "primary",
                    pm,
                    m["cd_l2"],
                    recipe,
                )

        with open(obj_dir / "metrics.json", "w") as f:
            json.dump(obj_metrics, f, indent=2)
        per_object.append(obj_metrics)

    # Build summary
    summary = {
        "n_objects": len(per_object),
        "subset_run": bool(subset_run),
        "n_eval_points": args.n_eval_points,
        "fscore_thresholds": args.fscore_thr,
        "score_on_filtered": bool(args.score_on_filtered),
        "identity_exp14": bool(args.identity_exp14),
        "export_primary": bool(export_primary),
        "primary_recipe": dict(PRIMARY_RECIPE),
        "align_method_info": {
            k: ALIGN_METHOD_INFO[k] for k in align_names if k in ALIGN_METHOD_INFO
        },
        "by_align": {},
    }
    for an in score_align_names:
        summary["by_align"][an] = {}
        for pm in PRED_METHODS:
            summary["by_align"][an][pm] = {
                k: _summarize(v) for k, v in aggregates[an][pm].items()
            }

    out = {"summary": summary, "objects": per_object}

    # Avoid clobbering full-dataset results when debugging a subset
    if subset_run:
        tag_parts = []
        if args.uid:
            tag_parts.append(f"uid_{args.uid[:16]}")
        if args.index is not None:
            tag_parts.append("idx_" + "_".join(str(i) for i in args.index[:5]))
        if args.max_objects is not None and args.index is None and args.uid is None:
            tag_parts.append(f"first{args.max_objects}")
        tag = "_".join(tag_parts) or "subset"
        out_path = compare_dir / f"results_by_align_debug_{tag}.json"
        table_path = compare_dir / f"results_table_debug_{tag}.txt"
    else:
        out_path = compare_dir / "results_by_align.json"
        table_path = compare_dir / "results_table_by_align.txt"

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    primary = "primary" if export_primary and "primary" in summary["by_align"] else (
        "robust_filter" if "robust_filter" in align_names else align_names[0]
    )

    if not subset_run:
        legacy = {
            "summary": {
                "n_objects": summary["n_objects"],
                "n_eval_points": args.n_eval_points,
                "fscore_thresholds": args.fscore_thr,
                "primary_align": primary,
                "primary_recipe": dict(PRIMARY_RECIPE),
                "methods": summary["by_align"].get(primary, {}),
            },
            "note": (
                f"Primary folder={primary} (per-model recipes). "
                "Full results in results_by_align.json"
            ),
        }
        with open(compare_dir / "results.json", "w") as f:
            json.dump(legacy, f, indent=2)

    lines = [
        f"Comparison with multiple alignments ({summary['n_objects']} objects"
        f"{', SUBSET/DEBUG' if subset_run else ''})",
        f"Primary: {primary}",
        f"score_on_filtered={args.score_on_filtered}",
        "",
    ]
    for an in score_align_names:
        if an == "primary":
            lines.append(f"=== {an} ===")
            lines.append(f"  (exp14={PRIMARY_RECIPE['exp14']}, surflo={PRIMARY_RECIPE['surflo']}, nova3r={PRIMARY_RECIPE['nova3r']})")
        elif an in ALIGN_METHOD_INFO:
            lines.append(f"=== {an} ===")
            lines.append(f"  ({ALIGN_METHOD_INFO[an]})")
        else:
            lines.append(f"=== {an} ===")
        lines.append(
            f"  {'method':8s} {'CD_L2':>8s} {'acc':>8s} {'comp':>8s} "
            f"{'F@0.05':>8s} {'scale':>7s}"
        )
        for pm in PRED_METHODS:
            b = summary["by_align"][an][pm]
            if not b.get("cd_l2"):
                lines.append(f"  {pm:8s}  (no results)")
                continue
            fs = b.get("fscore@0.05")
            fs_m = fs["median"] if fs else float("nan")
            lines.append(
                f"  {pm:8s} {b['cd_l2']['median']:8.4f} {b['accuracy']['median']:8.4f} "
                f"{b['completeness']['median']:8.4f} {fs_m:8.3f} "
                f"{b['scale']['median']:7.3f}"
            )
        lines.append("")

    table = "\n".join(lines) + "\n"
    table_path.write_text(table)
    if not subset_run:
        (compare_dir / "results_table.txt").write_text(table)
    print(table)
    logger.info("Wrote %s and %s", out_path, table_path.name)
    logger.info(
        "PLY: …/pred_aligned/{gt_anchored,robust_filter,primary}/ — "
        "primary uses per-model best recipe; stale *_full.ply removed on write"
    )


if __name__ == "__main__":
    main()
