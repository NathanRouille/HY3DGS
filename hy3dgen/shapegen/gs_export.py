"""Export input point clouds and 3D Gaussian splats for offline / web viewing."""

from __future__ import annotations

import struct
import warnings
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch

# Zero-order SH basis constant (INRIA 3DGS convention).
SH_C0 = 0.28209479177387814

# Standard 3DGS PLY uses 45 higher-order SH coefficient slots (degree 3 layout).
NUM_F_REST = 9


def _flatten_sh_coeffs(
    sh_coeffs: Union[torch.Tensor, np.ndarray],
    sh_degree: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split packed SH coefficients into f_dc (3,) and f_rest (45,) per splat.

    Args:
        sh_coeffs: (N, K, 3) with K >= (sh_degree+1)^2, or legacy (N, 3) DC RGB.
        sh_degree: active SH degree used during training.

    Returns:
        f_dc: (N, 3), f_rest: (N, 45) with unused higher bands zeroed.
    """
    arr = _to_numpy_f32(sh_coeffs)
    n = arr.shape[0]
    if arr.ndim == 2 and arr.shape[1] == 3:
        f_dc = _rgb_to_f_dc(arr)
        return f_dc, np.zeros((n, NUM_F_REST), dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"sh_coeffs must be (N, K, 3) or (N, 3), got {arr.shape}")
    K = (sh_degree + 1) ** 2
    if arr.shape[1] < K:
        raise ValueError(f"sh_coeffs has {arr.shape[1]} bases but sh_degree={sh_degree} needs {K}")
    f_dc = arr[:, 0, :]
    # INRIA / gsplat channel-major: coeff index varies fastest, then RGB.
    rest_flat = arr[:, 1:K, :].transpose(0, 2, 1).reshape(n, -1)
    f_rest = np.zeros((n, NUM_F_REST), dtype=np.float32)
    n_rest = min(rest_flat.shape[1], NUM_F_REST)
    f_rest[:, :n_rest] = rest_flat[:, :n_rest]
    return f_dc.astype(np.float32), f_rest


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


def export_gaussian_splats_gsplat(
    means: Union[torch.Tensor, np.ndarray],
    scales: Union[torch.Tensor, np.ndarray],
    rotations: Union[torch.Tensor, np.ndarray],
    opacities: Union[torch.Tensor, np.ndarray],
    sh_coeffs: Union[torch.Tensor, np.ndarray],
    path: Union[str, Path],
    *,
    format: Literal["ply", "ply_compressed", "splat"] = "ply_compressed",
    sh_degree: Optional[int] = None,
) -> None:
    """Export 3D Gaussians via ``gsplat.exporter.export_splats`` (SuperSplat-compatible).

    Model tensors use linear scales, sigmoid opacities in [0, 1], and ``sh_coeffs``
    shaped ``(N, K, 3)`` with ``K = (sh_degree + 1) ** 2``. gsplat expects log-scales,
    logit opacity, ``sh0`` ``(N, 1, 3)``, and ``shN`` ``(N, K-1, 3)``.

    Use ``format='ply_compressed'`` for https://superspl.at/editor (PlayCanvas compressed PLY).
    """
    from gsplat.exporter import export_splats

    from .gs_renderer import rgb_to_sh_dc

    def _t(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu()
        return torch.from_numpy(np.asarray(x, dtype=np.float32))

    means_t = _t(means).reshape(-1, 3)
    scales_lin = _t(scales).reshape(-1, 3)
    quats = _t(rotations).reshape(-1, 4)
    opa = _t(opacities).reshape(-1)
    sh = _t(sh_coeffs)

    n = means_t.shape[0]
    if scales_lin.shape[0] != n or quats.shape[0] != n or opa.shape[0] != n:
        raise ValueError("Gaussian parameter tensors must have the same number of splats")

    if sh.ndim == 2 and sh.shape[1] == 3:
        sh = rgb_to_sh_dc(sh).unsqueeze(1)
    if sh.ndim != 3 or sh.shape[-1] != 3:
        raise ValueError(f"sh_coeffs must be (N, K, 3) or legacy (N, 3), got {tuple(sh.shape)}")

    if sh_degree is None:
        sh_degree = int(round(sh.shape[1] ** 0.5) - 1)
    k_active = (sh_degree + 1) ** 2
    if sh.shape[1] < k_active:
        raise ValueError(
            f"sh_coeffs has {sh.shape[1]} bases but sh_degree={sh_degree} needs {k_active}"
        )
    sh = sh[:, :k_active, :]
    sh0 = sh[:, :1, :]
    shN = sh[:, 1:, :]

    scales_log = torch.log(scales_lin.clamp_min(1e-8))
    opa_clamped = opa.clamp(1e-6, 1.0 - 1e-6)
    opacity_logit = torch.log(opa_clamped / (1.0 - opa_clamped))

    export_format = format
    # gsplat ply_compressed writes ``element sh`` with f_rest_* columns only; SH0 has
    # none, which breaks SuperSplat ("DataTable must have at least one column").
    if sh_degree == 0 and format == "ply_compressed":
        export_format = "ply"
        warnings.warn(
            "SH0 export: using standard PLY instead of ply_compressed (SuperSplat "
            "requires at least one f_rest column in compressed format).",
            stacklevel=2,
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_splats(
        means_t,
        scales_log,
        quats,
        opacity_logit,
        sh0,
        shN,
        format=export_format,
        save_to=str(path),
    )


def export_gaussian_splat_ply(
    means: Union[torch.Tensor, np.ndarray],
    scales: Union[torch.Tensor, np.ndarray],
    rotations: Union[torch.Tensor, np.ndarray],
    opacities: Union[torch.Tensor, np.ndarray],
    colors: Union[torch.Tensor, np.ndarray],
    path: Union[str, Path],
    sh_degree: int = 1,
) -> None:
    """Write 3D Gaussians in standard 3DGS ``.ply`` format (web viewers, INRIA).

    Stored encodings match the original 3DGS trainer:
      - ``scale_*``: log(linear scale)
      - ``opacity``: logit(alpha)
      - ``f_dc_*`` / ``f_rest_*``: SH coefficients (``colors`` as (N,K,3) or legacy RGB)
      - ``rot_*``: unit quaternion w, x, y, z
    """
    xyz = _to_numpy_f32(means).reshape(-1, 3)
    scale_lin = _to_numpy_f32(scales).reshape(-1, 3)
    rot = _to_numpy_f32(rotations).reshape(-1, 4)
    opa = _to_numpy_f32(opacities).reshape(-1)

    n = xyz.shape[0]
    if scale_lin.shape[0] != n or rot.shape[0] != n or opa.shape[0] != n:
        raise ValueError("Gaussian parameter tensors must have the same number of splats")

    scale_log = np.log(np.clip(scale_lin, 1e-8, None))
    opacity_logit = _opacity_to_logit(opa)
    f_dc, f_rest = _flatten_sh_coeffs(colors, sh_degree=sh_degree)
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
    sh_degree: int = 1,
) -> None:
    """Write Gaussians in antimatter15 ``.splat`` format (32 bytes / splat).

    Uses DC SH band only for RGB (view-independent preview).
    """
    from .gs_renderer import sh_dc_to_rgb

    xyz = _to_numpy_f32(means).reshape(-1, 3)
    scale_lin = _to_numpy_f32(scales).reshape(-1, 3)
    rot = _to_numpy_f32(rotations).reshape(-1, 4)
    opa = np.clip(_to_numpy_f32(opacities).reshape(-1), 0.0, 1.0)
    f_dc, _ = _flatten_sh_coeffs(colors, sh_degree=sh_degree)
    rgb = (
        sh_dc_to_rgb(torch.from_numpy(f_dc)).numpy()
    )
    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)

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
