#!/usr/bin/env python3
"""Sample fixed-size surface point clouds for InternScenes rooms.

Robust 64k recipe (default):
  - target 32768 uniform + 32768 sharp
  - if sharp pool is smaller than requested, take all available sharp and
    put the remainder into uniform so n_total stays fixed
  - seeds stored as int64 (sha256 can exceed int32)

Full-gen layout (recommended)::

    --mesh_root /mnt/hdd2/ismail/InternScenes/processed/glb_output/sample_scenes
    --out_root  ~/Documents/research/internscenes_gen_pc_v0
    --list_file .../gen_light.txt
    --no_ply

Writes ``{out_root}/rooms/{scene_id}/surface.npz`` (+ meta.json).
Does not modify source InternScenes / Ismail storage.

Pilot layout (no --mesh_root): reads/writes under
``{out_root}/internscenes/{scene_id}/`` (local GLB copy).

Example:
  conda activate hy3dgs
  export PYTHONPATH="$HOME/Documents/research/HY3DGS:$PYTHONPATH"
  python sample_internscenes_pilot.py --mesh_root ... --out_root ... --no_ply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("sample_internscenes_pilot")

_SHARP_SHORT_RE = re.compile(
    r"Not enough sharp samples \((\d+)\) for num_sharp_points=(\d+)"
)
_UNIFORM_SHORT_RE = re.compile(
    r"Not enough surface samples \((\d+)\) for num_points=(\d+)"
)


def scene_seed(global_seed: int, scene_id: str) -> int:
    digest = hashlib.sha256(f"{int(global_seed)}:{scene_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def write_ply(path: Path, xyz: np.ndarray, rgb_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb_u8):
            f.write(
                f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def sample_fixed_total(
    loader,
    mesh_path: Path,
    *,
    n_total: int,
    n_sharp_target: int,
) -> Tuple[object, int, int]:
    """Return (surface_tensor, n_uniform, n_sharp) with fixed n_total.

    If sharp availability < target, use all sharp and fill the rest with uniform.
    """
    if n_sharp_target < 0 or n_sharp_target > n_total:
        raise ValueError(f"n_sharp_target={n_sharp_target} invalid for n_total={n_total}")

    n_s = int(n_sharp_target)
    last_err: Optional[BaseException] = None

    for attempt in range(4):
        n_u = int(n_total - n_s)
        try:
            surf = loader(
                str(mesh_path),
                num_uniform_points=n_u,
                num_sharp_points=n_s,
            )
            return surf, n_u, n_s
        except ValueError as exc:
            last_err = exc
            msg = str(exc)

            m_sharp = _SHARP_SHORT_RE.search(msg)
            if m_sharp is not None:
                avail = int(m_sharp.group(1))
                logger.warning(
                    "%s: sharp pool has %d < requested %d (attempt %d); "
                    "retry with n_sharp=%d, n_uniform=%d",
                    mesh_path.name,
                    avail,
                    n_s,
                    attempt + 1,
                    avail,
                    n_total - avail,
                )
                if avail <= 0:
                    n_s = 0
                elif avail < n_s:
                    n_s = avail
                else:
                    raise
                continue

            m_uni = _UNIFORM_SHORT_RE.search(msg)
            if m_uni is not None:
                avail_u = int(m_uni.group(1))
                raise RuntimeError(
                    f"{mesh_path}: uniform pool only has {avail_u} points; "
                    f"cannot build n_total={n_total} (n_sharp={n_s}). "
                    "Mesh may be degenerate or intermediate pool too small."
                ) from exc

            raise

    assert last_err is not None
    raise last_err


def sample_one(
    *,
    mesh_path: Path,
    out_dir: Path,
    dataset: str,
    scene_id: str,
    n_total: int,
    n_sharp_target: int,
    global_seed: int,
    overwrite: bool,
    write_ply_preview: bool,
) -> None:
    from hy3dgen.shapegen.surface_loaders import RGBSharpEdgeSurfaceLoader

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "surface.npz"
    ply_path = out_dir / "preview.ply"
    meta_path = out_dir / "meta.json"

    # Resume on training artifact only (npz). PLY is optional viz.
    if npz_path.exists() and not overwrite:
        logger.info("SKIP %s (exists)", scene_id)
        return

    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    seed = scene_seed(global_seed, scene_id)
    loader = RGBSharpEdgeSurfaceLoader(
        num_uniform_points=n_total - n_sharp_target,
        num_sharp_points=n_sharp_target,
        seed=seed,
        include_sharp_label=True,
    )

    logger.info(
        "SAMPLE %s  target total=%d sharp=%d ...",
        scene_id,
        n_total,
        n_sharp_target,
    )
    surf, n_u, n_s = sample_fixed_total(
        loader,
        mesh_path,
        n_total=n_total,
        n_sharp_target=n_sharp_target,
    )

    x = surf[0].detach().cpu().numpy().astype(np.float32)
    if x.shape[0] != n_total:
        raise RuntimeError(f"{scene_id}: got {x.shape[0]} points, expected {n_total}")

    xyz, nrm, sharp, rgb = x[:, :3], x[:, 3:6], x[:, 6:7], x[:, 7:10]
    rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)

    np.savez_compressed(
        npz_path,
        xyz=xyz,
        normals=nrm,
        sharp=sharp.astype(np.float32),
        rgb=rgb_u8,
        n_uniform=np.int32(n_u),
        n_sharp=np.int32(n_s),
        n_total=np.int32(n_total),
        seed=np.int64(seed),
    )
    if write_ply_preview:
        write_ply(ply_path, xyz, rgb_u8)

    meta = {
        "dataset": dataset,
        "scene_id": scene_id,
        "mesh": str(mesh_path.resolve()),
        "n_total": int(n_total),
        "n_uniform": int(n_u),
        "n_sharp": int(n_s),
        "n_sharp_target": int(n_sharp_target),
        "sharp_capped": bool(n_s < n_sharp_target),
        "seed": int(seed),
        "layout": "xyz|normals|sharp|rgb",
        "preview_ply": bool(write_ply_preview),
        "note": (
            "Fixed n_total; sharp capped to available pool with uniform fill. "
            "surface_loaders.normalize_parts applies unit-box normalization."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "OK %s -> %s  (uniform=%d sharp=%d%s)",
        scene_id,
        npz_path,
        n_u,
        n_s,
        " [sharp capped]" if n_s < n_sharp_target else "",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--out_root",
        type=Path,
        default=Path.home() / "Documents/research/internscenes_gen_pc_v0",
        help="Output root (rooms/ or pilot internscenes/)",
    )
    p.add_argument(
        "--mesh_root",
        type=Path,
        default=None,
        help=(
            "Read {mesh_root}/{scene_id}/scene_split.glb (Ismail, no copy). "
            "If omitted, uses pilot local {out_root}/internscenes/{id}/scene_split.glb"
        ),
    )
    p.add_argument(
        "--list_file",
        type=Path,
        default=None,
        help="Scene id list (default: <out_root>/lists/pilot_internscenes_20.txt)",
    )
    p.add_argument(
        "--scene_id",
        action="append",
        default=None,
        help="Sample only these ids (repeatable). Default: all ids in list_file.",
    )
    p.add_argument("--n_total", type=int, default=65536)
    p.add_argument("--n_sharp_target", type=int, default=32768)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--no_ply",
        action="store_true",
        help="Do not write preview.ply (recommended for full-dataset runs)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.n_sharp_target > args.n_total:
        logger.error("n_sharp_target > n_total")
        return 2

    out_root: Path = args.out_root.expanduser().resolve()
    mesh_root: Optional[Path] = (
        args.mesh_root.expanduser().resolve() if args.mesh_root is not None else None
    )
    list_file = (
        args.list_file.expanduser().resolve()
        if args.list_file is not None
        else out_root / "lists" / "pilot_internscenes_20.txt"
    )

    if args.scene_id:
        scene_ids = list(args.scene_id)
    else:
        if not list_file.is_file():
            logger.error("Missing list file: %s", list_file)
            return 2
        scene_ids = [
            ln.strip()
            for ln in list_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]

    write_ply_preview = not args.no_ply
    ok, fail, skipped = 0, 0, 0

    for sid in scene_ids:
        if mesh_root is not None:
            mesh = mesh_root / sid / "scene_split.glb"
            out_dir = out_root / "rooms" / sid
        else:
            mesh = out_root / "internscenes" / sid / "scene_split.glb"
            out_dir = out_root / "internscenes" / sid

        npz_path = out_dir / "surface.npz"
        already = npz_path.exists() and not args.overwrite

        try:
            sample_one(
                mesh_path=mesh,
                out_dir=out_dir,
                dataset="internscenes",
                scene_id=sid,
                n_total=args.n_total,
                n_sharp_target=args.n_sharp_target,
                global_seed=args.seed,
                overwrite=args.overwrite,
                write_ply_preview=write_ply_preview,
            )
            if already:
                skipped += 1
            else:
                ok += 1
        except Exception:
            fail += 1
            logger.exception("FAIL %s", sid)

    logger.info("DONE ok=%d skipped=%d fail=%d", ok, skipped, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
