#!/usr/bin/env python3
"""Score already-exported gt_norm_ablation PLYs with bakeoff metrics.

Reads gt.ply + patch_centers.ply (+ vggt_pred_vggtK.ply if present) and writes
metrics.txt next to them, plus a CSV summary. Does not re-run VGGT.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from export_camera_frame_debug import alignment_metrics

METHODS = (
    "A_cross_train",
    "C_gt_depth_filter_zrobust",
    "D_mesh_own_indep",
)


def load_xyz_ply(path: Path) -> np.ndarray:
    """Load xyz from ASCII or binary_little_endian xyz(+rgb) PLY."""
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"truncated PLY header: {path}")
            header_lines.append(line.decode("ascii", errors="replace").strip())
            if header_lines[-1] == "end_header":
                break
        fmt = "ascii"
        n = 0
        props = []
        for h in header_lines:
            if h.startswith("format "):
                fmt = h.split()[1]
            elif h.startswith("element vertex "):
                n = int(h.split()[-1])
            elif h.startswith("property "):
                props.append(h.split()[1:])
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if fmt == "ascii":
            rest = f.read().decode("ascii", errors="replace").splitlines()
            xyz = np.zeros((n, 3), dtype=np.float32)
            for i, row in enumerate(rest[:n]):
                toks = row.split()
                xyz[i] = (float(toks[0]), float(toks[1]), float(toks[2]))
            return xyz
        if fmt != "binary_little_endian":
            raise ValueError(f"unsupported PLY format {fmt}: {path}")
        # Packed xyz float32 + optional uchar rgb/rgba.
        dt_fields = []
        for p in props:
            if p[0] == "float" and p[1] in ("x", "y", "z"):
                dt_fields.append((p[1], "<f4"))
            elif p[0] == "uchar":
                dt_fields.append((p[1], "u1"))
            else:
                raise ValueError(f"unsupported property {p} in {path}")
        rec = np.frombuffer(f.read(), dtype=np.dtype(dt_fields), count=n)
        return np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float32)


def summarize(vals: list[float]) -> dict[str, float]:
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "median": float("nan")}
    return {"mean": float(np.mean(a)), "median": float(np.median(a))}


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/debug_cam_frame")
    rng = np.random.default_rng(0)
    rows: list[dict] = []

    obj_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "gt_norm_ablation").is_dir())
    if not obj_dirs:
        print(f"no gt_norm_ablation folders under {root}", file=sys.stderr)
        return 1

    for obj_dir in obj_dirs:
        gna = obj_dir / "gt_norm_ablation"
        stem = obj_dir.name
        for method in METHODS:
            mdir = gna / method
            gt_p = mdir / "gt.ply"
            cen_p = mdir / "patch_centers.ply"
            if not gt_p.is_file() or not cen_p.is_file():
                print(f"skip missing PLYs: {mdir}", file=sys.stderr)
                continue
            gt = load_xyz_ply(gt_p)
            cen = load_xyz_ply(cen_p)
            m_cen = alignment_metrics(cen, gt, rng=rng)
            fg_p = mdir / "vggt_pred_vggtK.ply"
            m_fg = (
                alignment_metrics(load_xyz_ply(fg_p), gt, rng=rng)
                if fg_p.is_file()
                else {}
            )
            with open(mdir / "metrics.txt", "w", encoding="utf-8") as f:
                f.write(f"mesh={stem}\nmethod={method}\n")
                f.write("scored from exported PLYs (no VGGT re-run)\n\n")
                f.write("[patch_centers_vs_gt_mesh]\n")
                for k, v in m_cen.items():
                    f.write(f"  {k}={v}\n")
                if m_fg:
                    f.write("\n[vggt_fg_vs_gt_mesh]\n")
                    for k, v in m_fg.items():
                        f.write(f"  {k}={v}\n")
            rows.append(
                {
                    "mesh": stem,
                    "method": method,
                    "n_centres": m_cen["n_pe"],
                    "chamfer_mean": m_cen["chamfer_mean"],
                    "nn_pe2gt_mean": m_cen["nn_pe2gt_mean"],
                    "nn_pe2gt_med": m_cen["nn_pe2gt_med"],
                    "nn_gt2pe_mean": m_cen["nn_gt2pe_mean"],
                    "overlap_pe_frac": m_cen["overlap_pe_frac"],
                    "hausdorff_p95": m_cen["hausdorff_p95"],
                    "bbox_side_ratio": m_cen["bbox_side_ratio"],
                    "bbox_center_dist": m_cen["bbox_center_dist"],
                    "xy_span_ratio": m_cen["xy_span_ratio"],
                    "mean_z_diff": m_cen["mean_z_diff"],
                    "fg_nn_pe2gt_mean": m_fg.get("nn_pe2gt_mean", float("nan")),
                    "fg_overlap_pe_frac": m_fg.get("overlap_pe_frac", float("nan")),
                }
            )

    csv_path = root / "gt_norm_ablation_summary.csv"
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
        "fg_nn_pe2gt_mean",
        "fg_overlap_pe_frac",
    )
    print("\n=== Aggregate (patch centres vs GT, unless fg_*) ===")
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
    for metric, higher in (
        ("nn_pe2gt_mean", False),
        ("overlap_pe_frac", True),
        ("fg_nn_pe2gt_mean", False),
        ("fg_overlap_pe_frac", True),
    ):
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

    print("\n=== Per-object centres pe2gt / overlap@0.05 ===")
    print(f"{'obj':22s}  {'A pe2gt':>8s} {'A ov':>6s}  {'C pe2gt':>8s} {'C ov':>6s}  {'D pe2gt':>8s} {'D ov':>6s}  best")
    for mesh in meshes:
        short = mesh.split("_")[0]
        bits = [f"{short:22s}"]
        scores = {}
        ovs = {}
        for m in METHODS:
            r = per[m].get(mesh)
            if r is None:
                bits.append(f"{'—':>8s} {'—':>6s}")
                continue
            scores[m] = float(r["nn_pe2gt_mean"])
            ovs[m] = float(r["overlap_pe_frac"])
            bits.append(f"{scores[m]:8.4f} {ovs[m]:6.3f}")
        best = min(scores, key=scores.get) if scores else "?"
        print("  ".join(bits) + f"  {best[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
