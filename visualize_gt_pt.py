#!/usr/bin/env python3
"""Visualize cached GT RGBD .pt files (e.g. v46 debug precache).

Usage:
  python visualize_gt_pt.py /path/to/cache.pt --out_dir ./viz_out
  python visualize_gt_pt.py /path/to/gt_cache/ --out_dir ./viz_out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
except ImportError:
    cm = None


def _turbo_colormap():
    try:
        return matplotlib.colormaps["turbo"]
    except Exception:
        return cm.get_cmap("turbo")


def depth_to_rgb_u8(depth_hw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """depth_hw: (H,W) float; valid: (H,W) bool. RGB uint8."""
    h, w = depth_hw.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if not np.any(valid):
        return out
    d = depth_hw[valid]
    vmin, vmax = float(d.min()), float(d.max())
    if vmax <= vmin:
        norm = np.zeros_like(depth_hw, dtype=np.float32)
    else:
        norm = np.clip((depth_hw - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = _turbo_colormap()
    rgba = cmap(norm.astype(np.float64))
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    out[valid] = rgb[valid]
    return out


def save_rgb(path: Path, tensor_chw: torch.Tensor) -> None:
    """tensor: (H,W,3) float [0,1]."""
    x = tensor_chw.detach().clamp(0, 1).cpu().numpy()
    u8 = (x * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(u8).save(path)


def save_depth_viz(path: Path, depth_hw1: torch.Tensor) -> None:
    """depth: (H,W,1); visualize valid pixels (depth > 0)."""
    d = depth_hw1.squeeze(-1).detach().cpu().numpy().astype(np.float32)
    valid = d > 0
    rgb = depth_to_rgb_u8(d, valid)
    Image.fromarray(rgb).save(path)


def save_raw_depth_npz(path: Path, depth_hw1: torch.Tensor) -> None:
    d = depth_hw1.squeeze(-1).detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(path, depth=d)


def view_params_to_jsonable(vp: Any) -> Any:
    if vp is None:
        return None
    if isinstance(vp, list):
        return [view_params_to_jsonable(x) for x in vp]
    if isinstance(vp, dict):
        return {k: float(v) if isinstance(v, (float, int, np.floating)) else v for k, v in vp.items()}
    return vp


def process_one_pt(pt_path: Path, out_root: Path, save_raw_npz: bool) -> None:
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    rgbs: List[torch.Tensor] = data.get("rgbs", [])
    depths: List[torch.Tensor] = data.get("depths", [])
    c2ws = data.get("c2ws", [])
    if data.get('storage_dtype') == 'float16':
        rgbs = [x.float() for x in rgbs]
        depths = [x.float() for x in depths]
    view_params: Optional[List[Dict[str, Any]]] = data.get("view_params")

    stem = pt_path.stem
    od = out_root / stem
    od.mkdir(parents=True, exist_ok=True)

    n = len(rgbs)
    if len(depths) != n:
        raise ValueError(f"rgbs ({n}) and depths ({len(depths)}) length mismatch in {pt_path}")
    if c2ws and len(c2ws) != n:
        print(f"Warning: c2ws length {len(c2ws)} != num views {n}", file=sys.stderr)

    meta: Dict[str, Any] = {
        "source_pt": str(pt_path.resolve()),
        "num_views": n,
        "keys_in_pt": sorted(data.keys()),
        "view_layout": data.get("view_layout"),
    }
    if view_params is not None:
        meta["view_params"] = view_params_to_jsonable(view_params)

    with open(od / "view_params.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Optional: full c2w as list-of-lists (large file)
    if c2ws:
        c2w_list = [c.detach().cpu().numpy().tolist() for c in c2ws]
        with open(od / "c2ws.json", "w") as f:
            json.dump(c2w_list, f)

    for i in range(n):
        save_rgb(od / f"view{i:02d}_rgb.png", rgbs[i])
        save_depth_viz(od / f"view{i:02d}_depth_color.png", depths[i])
        if save_raw_npz:
            save_raw_depth_npz(od / f"view{i:02d}_depth_raw.npz", depths[i])

    print(f"Wrote {n} views → {od.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser(description="Export RGB / depth / view_params from GT .pt cache")
    p.add_argument("input", type=str, help="Path to one .pt file or a directory of .pt files")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory root")
    p.add_argument("--raw_depth_npz", action="store_true", help="Also save raw depth per view as .npz")
    args = p.parse_args()

    if cm is None:
        print("matplotlib is required for depth colormap. pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    inp = Path(args.input).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if inp.is_file():
        if inp.suffix.lower() != ".pt":
            print("Input file should be .pt", file=sys.stderr)
            sys.exit(1)
        process_one_pt(inp, out_root, args.raw_depth_npz)
        return

    if inp.is_dir():
        pts = sorted(inp.glob("*.pt"))
        if not pts:
            print(f"No .pt files in {inp}", file=sys.stderr)
            sys.exit(1)
        for pt in pts:
            process_one_pt(pt, out_root, args.raw_depth_npz)
        return

    print(f"Not found: {inp}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()