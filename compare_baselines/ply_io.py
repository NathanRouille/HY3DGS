"""Minimal PLY xyz(+rgb) I/O without open3d (hy3dgs-friendly)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def write_ply(
    path: Union[str, Path],
    xyz: np.ndarray,
    colors: Optional[np.ndarray] = None,
    *,
    rgb: Tuple[int, int, int] = (180, 180, 180),
) -> None:
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    n = int(pts.shape[0])

    if colors is not None:
        cols = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        cols = cols[finite] if cols.shape[0] == finite.shape[0] else cols[:n]
        if cols.max() <= 1.0 + 1e-6:
            cols_u8 = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
        else:
            cols_u8 = np.clip(cols, 0, 255).astype(np.uint8)
    else:
        cols_u8 = np.empty((n, 3), dtype=np.uint8)
        cols_u8[:, :] = np.array(rgb, dtype=np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = pts[i]
            r, g, b = cols_u8[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def read_ply_xyz(path: Union[str, Path]) -> np.ndarray:
    """Read xyz from ascii or binary_little_endian PLY (xyz required)."""
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read()

    # Header is always ascii
    header_end = raw.find(b"end_header")
    if header_end < 0:
        raise ValueError(f"No end_header in {path}")
    header = raw[: header_end + len(b"end_header")].decode("ascii", errors="replace")
    body = raw[header_end + len(b"end_header") :]
    if body.startswith(b"\n"):
        body = body[1:]
    elif body.startswith(b"\r\n"):
        body = body[2:]

    fmt = "ascii"
    n_vert = 0
    props = []
    for line in header.splitlines():
        if line.startswith("format "):
            fmt = line.split()[1]
        elif line.startswith("element vertex"):
            n_vert = int(line.split()[-1])
        elif line.startswith("property "):
            parts = line.split()
            props.append((parts[1], parts[2]))  # type, name

    if n_vert == 0:
        return np.zeros((0, 3), dtype=np.float64)

    name_to_idx = {name: i for i, (_, name) in enumerate(props)}
    for need in ("x", "y", "z"):
        if need not in name_to_idx:
            raise ValueError(f"{path} missing property {need}")

    if fmt == "ascii":
        text = body.decode("ascii", errors="replace").strip().splitlines()
        pts = np.zeros((n_vert, 3), dtype=np.float64)
        for i in range(n_vert):
            toks = text[i].split()
            pts[i, 0] = float(toks[name_to_idx["x"]])
            pts[i, 1] = float(toks[name_to_idx["y"]])
            pts[i, 2] = float(toks[name_to_idx["z"]])
        return pts

    if fmt not in ("binary_little_endian", "binary_big_endian"):
        raise ValueError(f"Unsupported PLY format {fmt} in {path}")

    endian = "<" if fmt == "binary_little_endian" else ">"
    type_map = {
        "float": "f4",
        "float32": "f4",
        "double": "f8",
        "float64": "f8",
        "uchar": "u1",
        "uint8": "u1",
        "char": "i1",
        "int": "i4",
        "int32": "i4",
        "uint": "u4",
        "uint32": "u4",
        "short": "i2",
        "ushort": "u2",
    }
    dtype = np.dtype(
        [(name, endian + type_map[t]) for t, name in props]
    )
    arr = np.frombuffer(body, dtype=dtype, count=n_vert)
    pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    return pts


def subsample_points(xyz: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] <= n:
        return xyz
    rng = np.random.default_rng(seed)
    idx = rng.choice(xyz.shape[0], size=n, replace=False)
    return xyz[idx]
