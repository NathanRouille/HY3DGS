#!/usr/bin/env python3
"""Score mv_frame0 PLYs (3 multi-view methods) with the same metrics as gt_norm.

Reads gt.ply + vggt_pred_vggtK.ply under each method folder.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from score_gt_norm_ablation import load_xyz_ply, summarize
from export_camera_frame_debug import alignment_metrics

METHODS = (
    "A_cross_gobK_VGGT_E",
    "C_gt_depth_filter_zrobust",
    "A_cross_gobK_VGGT_E_meanrms",
    "C_gt_depth_filter_zrobust_meanrms",
    "indep_meanrms",
)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/debug_cam_frame")
    rng = np.random.default_rng(0)
    rows: list[dict] = []

    obj_dirs = sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "mv_frame0").is_dir()
    )
    if not obj_dirs:
        print(f"no mv_frame0 folders under {root}", file=sys.stderr)
        return 1

    for obj_dir in obj_dirs:
        mv = obj_dir / "mv_frame0"
        stem = obj_dir.name
        for method in METHODS:
            mdir = mv / method
            gt_p = mdir / "gt.ply"
            fg_p = mdir / "vggt_pred_vggtK.ply"
            if not gt_p.is_file() or not fg_p.is_file():
                print(f"skip missing PLYs: {mdir}", file=sys.stderr)
                continue
            gt = load_xyz_ply(gt_p)
            pe = load_xyz_ply(fg_p)
            m_fg = alignment_metrics(pe, gt, rng=rng)
            with open(mdir / "metrics.txt", "w", encoding="utf-8") as f:
                f.write(f"mesh={stem}\nmethod={method}\n")
                f.write("scored from exported PLYs (vggt FG vs GT)\n\n")
                f.write("[vggt_fg_vs_gt_mesh]\n")
                for k, v in m_fg.items():
                    f.write(f"  {k}={v}\n")
            rows.append(
                {
                    "mesh": stem,
                    "method": method,
                    "n_pe": m_fg["n_pe"],
                    "chamfer_mean": m_fg["chamfer_mean"],
                    "nn_pe2gt_mean": m_fg["nn_pe2gt_mean"],
                    "nn_pe2gt_med": m_fg["nn_pe2gt_med"],
                    "nn_gt2pe_mean": m_fg["nn_gt2pe_mean"],
                    "overlap_pe_frac": m_fg["overlap_pe_frac"],
                    "hausdorff_p95": m_fg["hausdorff_p95"],
                    "bbox_side_ratio": m_fg["bbox_side_ratio"],
                    "bbox_center_dist": m_fg["bbox_center_dist"],
                    "xy_span_ratio": m_fg["xy_span_ratio"],
                    "mean_z_diff": m_fg["mean_z_diff"],
                }
            )

    csv_path = root / "mv_frame0_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)")

    by = {m: [r for r in rows if r["method"] == m] for m in METHODS}
    keys = (
        "nn_pe2gt_mean",
        "nn_pe2gt_med",
        "overlap_pe_frac",
        "bbox_side_ratio",
        "bbox_center_dist",
        "xy_span_ratio",
    )
    print("\n=== Aggregate (vggt FG vs GT) ===")
    print(f"{'method':28s}  " + "  ".join(f"{k:>16s}" for k in keys))
    for method, rs in by.items():
        parts = [f"{method:28s}"]
        for k in keys:
            st = summarize([float(r[k]) for r in rs])
            parts.append(f"{st['mean']:7.4f}/{st['median']:6.4f}")
        print("  ".join(parts) + "   (mean/median)")

    meshes = sorted(set(r["mesh"] for r in rows))
    per = {m: {r["mesh"]: r for r in rs} for m, rs in by.items()}
    print(f"\n=== Wins among {len(meshes)} objects ===")
    for metric, higher in (("nn_pe2gt_mean", False), ("overlap_pe_frac", True)):
        wins = {m: 0 for m in METHODS}
        for mesh in meshes:
            vals = {m: float(per[m][mesh][metric]) for m in METHODS if mesh in per[m]}
            if len(vals) < 2:
                continue
            best = max(vals, key=vals.get) if higher else min(vals, key=vals.get)
            wins[best] += 1
        direction = "higher" if higher else "lower"
        print(
            f"  {metric:22s} ({direction:6s}):  "
            + "  ".join(f"{m}={wins[m]}" for m in METHODS)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
