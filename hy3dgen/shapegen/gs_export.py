"""Export input point clouds and 3D Gaussian splats for offline / web viewing."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

# Zero-order SH basis constant (INRIA 3DGS convention).
SH_C0 = 0.28209479177387814

# Standard 3DGS PLY uses 45 higher-order SH coefficients (degree 3).
NUM_F_REST = 45


def _to_numpy_f32(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def export_input_surface_ply(
    surface: Union[torch.Tensor, np.ndarray],
    path: Union[str, Path],
) -> None:
    """Save model input surface as a colored point cloud PLY.

    Args:
        surface: [N, 9] — xyz(0:3) | normals(3:6) | rgb(6:9), values in [0, 1] for rgb.
        path: Output ``.ply`` path.
    """
    pts = _to_numpy_f32(surface)
    if pts.ndim != 2 or pts.shape[1] < 9:
        raise ValueError(f"surface must be [N, 9], got {pts.shape}")

    xyz = pts[:, :3]
    normals = pts[:, 3:6]
    rgb_u8 = (np.clip(pts[:, 6:9], 0.0, 1.0) * 255.0).round().astype(np.uint8)
    n = xyz.shape[0]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack(
                "<3f3f3B",
                float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]),
                float(normals[i, 0]), float(normals[i, 1]), float(normals[i, 2]),
                int(rgb_u8[i, 0]), int(rgb_u8[i, 1]), int(rgb_u8[i, 2]),
            ))


def export_xyz_pointcloud_ply(
    xyz: Union[torch.Tensor, np.ndarray],
    path: Union[str, Path],
    rgb: tuple[int, int, int] = (255, 64, 64),
    colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> None:
    """Save ``[N, 3]`` positions as a colored point cloud PLY (e.g. FPS anchors).

    Args:
        xyz    : ``(N, 3)`` positions.
        path   : output ``.ply`` file path.
        rgb    : uniform colour used when ``colors`` is ``None``.
        colors : optional ``(N, 3)`` per-point colour in ``[0, 1]`` (e.g. PCA→RGB).
    """
    pts = _to_numpy_f32(xyz).reshape(-1, 3)
    n = pts.shape[0]

    if colors is not None:
        rgb_arr = _to_numpy_f32(colors).reshape(-1, 3)
        if rgb_arr.shape[0] != n:
            raise ValueError(
                f"colors has {rgb_arr.shape[0]} entries but xyz has {n}"
            )
        rgb_u8 = np.clip(rgb_arr * 255.0, 0, 255).astype(np.uint8)
    else:
        r, g, b = (int(np.clip(c, 0, 255)) for c in rgb)
        rgb_u8 = np.broadcast_to(np.array([r, g, b], dtype=np.uint8), (n, 3))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack(
                "<3f3B",
                float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2]),
                int(rgb_u8[i, 0]), int(rgb_u8[i, 1]), int(rgb_u8[i, 2]),
            ))


def _rgb_to_f_dc(rgb: np.ndarray) -> np.ndarray:
    """Map linear RGB in [0, 1] to 3DGS SH DC coefficients."""
    return ((rgb - 0.5) / SH_C0).astype(np.float32)


def _opacity_to_logit(opacity: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(opacity.astype(np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p)).astype(np.float32)


def export_gaussian_splat_ply(
    means: Union[torch.Tensor, np.ndarray],
    scales: Union[torch.Tensor, np.ndarray],
    rotations: Union[torch.Tensor, np.ndarray],
    opacities: Union[torch.Tensor, np.ndarray],
    colors: Union[torch.Tensor, np.ndarray],
    path: Union[str, Path],
) -> None:
    """Write 3D Gaussians in standard 3DGS ``.ply`` format (web viewers, INRIA).

    Stored encodings match the original 3DGS trainer:
      - ``scale_*``: log(linear scale)
      - ``opacity``: logit(alpha)
      - ``f_dc_*``: SH DC from RGB
      - ``f_rest_*``: zeros (view-independent color)
      - ``rot_*``: unit quaternion w, x, y, z
    """
    xyz = _to_numpy_f32(means).reshape(-1, 3)
    scale_lin = _to_numpy_f32(scales).reshape(-1, 3)
    rot = _to_numpy_f32(rotations).reshape(-1, 4)
    opa = _to_numpy_f32(opacities).reshape(-1)
    rgb = _to_numpy_f32(colors).reshape(-1, 3)

    n = xyz.shape[0]
    if scale_lin.shape[0] != n or rot.shape[0] != n or opa.shape[0] != n or rgb.shape[0] != n:
        raise ValueError("Gaussian parameter tensors must have the same number of splats")

    scale_log = np.log(np.clip(scale_lin, 1e-8, None))
    opacity_logit = _opacity_to_logit(opa)
    f_dc = _rgb_to_f_dc(rgb)
    f_rest = np.zeros((n, NUM_F_REST), dtype=np.float32)
    normals = np.zeros((n, 3), dtype=np.float32)

    props = [
        "x", "y", "z",
        "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
    ] + [f"f_rest_{i}" for i in range(NUM_F_REST)] + [
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
    ]
    for p in props:
        header_lines.append(f"property float {p}")
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    rows = np.empty(
        n,
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ]
        + [(f"f_rest_{i}", "f4") for i in range(NUM_F_REST)]
        + [
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ],
    )
    rows["x"], rows["y"], rows["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rows["nx"], rows["ny"], rows["nz"] = 0.0, 0.0, 0.0
    rows["f_dc_0"], rows["f_dc_1"], rows["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    for i in range(NUM_F_REST):
        rows[f"f_rest_{i}"] = f_rest[:, i]
    rows["opacity"] = opacity_logit
    rows["scale_0"], rows["scale_1"], rows["scale_2"] = (
        scale_log[:, 0], scale_log[:, 1], scale_log[:, 2],
    )
    rows["rot_0"], rows["rot_1"], rows["rot_2"], rows["rot_3"] = (
        rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3],
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        rows.tofile(f)


def export_gaussian_splat_file(
    means: Union[torch.Tensor, np.ndarray],
    scales: Union[torch.Tensor, np.ndarray],
    rotations: Union[torch.Tensor, np.ndarray],
    opacities: Union[torch.Tensor, np.ndarray],
    colors: Union[torch.Tensor, np.ndarray],
    path: Union[str, Path],
) -> None:
    """Write Gaussians in antimatter15 ``.splat`` format (32 bytes / splat)."""
    xyz = _to_numpy_f32(means).reshape(-1, 3)
    scale_lin = _to_numpy_f32(scales).reshape(-1, 3)
    rot = _to_numpy_f32(rotations).reshape(-1, 4)
    opa = np.clip(_to_numpy_f32(opacities).reshape(-1), 0.0, 1.0)
    rgb = (np.clip(_to_numpy_f32(colors).reshape(-1, 3), 0.0, 1.0) * 255.0).round().astype(np.uint8)

    n = xyz.shape[0]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    buf = bytearray(n * 32)
    for i in range(n):
        off = i * 32
        struct.pack_into("<3f", buf, off, xyz[i, 0], xyz[i, 1], xyz[i, 2])
        struct.pack_into("<3f", buf, off + 12, scale_lin[i, 0], scale_lin[i, 1], scale_lin[i, 2])
        a = int(opa[i] * 255.0)
        struct.pack_into("<4B", buf, off + 24, rgb[i, 0], rgb[i, 1], rgb[i, 2], a)
        # antimatter15/splat: normalized quaternion * 128 + 128 per component (w,x,y,z).
        r = rot[i].astype(np.float64)
        rn = np.linalg.norm(r)
        if rn < 1e-8:
            r = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            r = r / rn
        rot_u8 = ((r * 128.0) + 128.0).clip(0, 255).astype(np.uint8)
        struct.pack_into("<4B", buf, off + 28, int(rot_u8[0]), int(rot_u8[1]), int(rot_u8[2]), int(rot_u8[3]))

    path.write_bytes(buf)
