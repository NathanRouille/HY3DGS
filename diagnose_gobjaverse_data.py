#!/usr/bin/env python3
"""Diagnostic script for G-Objaverse data quality.

For each object, produces a visual report comparing:
  1. Software-rasterized views of the GLB (with vertex colors) vs G-Objaverse GT renders
  2. Extracted surface point cloud colors vs GT albedo images
  3. Mesh vertex projection overlay on GT images (camera alignment check)
  4. Per-view metrics: silhouette IoU, depth correlation, color similarity

Camera / coordinate conventions (derived from RichDreamer depth_warp_example.py):
  - G-Objaverse view JSONs store Unity camera basis as column vectors (x, y, z, origin).
  - trimesh.to_geometry() applies the GLB scene graph transform (typically a Z-up → Y-up
    rotation + scale). However G-Objaverse / Unity effectively uses the *pre-rotation*
    (Z-up) mesh coordinates with just the scale factor applied. We therefore undo
    the rotation via R_x(+90°) after to_geometry(): new_Y = -old_Z, new_Z = old_Y.
  - The projection uses the depth_warp convention (NO Y-negation):
        u = fx * X_cam / Z_cam + cx
        v = fy * Y_cam / Z_cam + cy
    with cx = W/2, cy = H/2.

Usage:
    python diagnose_gobjaverse_data.py --manifest <path> [--max_items N] [--output_dir <dir>]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# ---------------------------------------------------------------------------
# Mesh loading helpers
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str) -> Dict:
    with open(manifest_path) as f:
        return json.load(f)


def load_view_meta(json_path: str) -> Dict:
    with open(json_path) as f:
        return json.load(f)


def load_and_prepare_mesh(
    mesh_path: str, meta: Dict,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Load GLB and normalise using G-Objaverse convention.

    Uses ``normalize_mesh_gobjaverse`` from the training pipeline, which applies
    the R_x(+90°) coordinate fix + center + scale.

    Returns (mesh_combined, mesh_norm).
    """
    from hy3dgen.shapegen.gobjaverse_gt import normalize_mesh_gobjaverse
    from hy3dgen.shapegen.surface_loaders import scene_to_geometry

    raw_mesh = trimesh.load(mesh_path, process=False)
    mesh_combined = scene_to_geometry(raw_mesh)

    mesh_norm = normalize_mesh_gobjaverse(mesh_combined, meta)
    try:
        mesh_norm.visual = mesh_combined.visual
    except Exception:
        pass
    return mesh_combined, mesh_norm


# ---------------------------------------------------------------------------
# Camera helpers  (matching RichDreamer depth_warp_example.py)
# ---------------------------------------------------------------------------

def unity_c2w_from_meta(meta: Dict) -> np.ndarray:
    """Build 4x4 c2w from G-Objaverse view JSON (raw Unity basis columns)."""
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = np.asarray(meta["x"], dtype=np.float64)
    c2w[:3, 1] = np.asarray(meta["y"], dtype=np.float64)
    c2w[:3, 2] = np.asarray(meta["z"], dtype=np.float64)
    c2w[:3, 3] = np.asarray(meta["origin"], dtype=np.float64)
    return c2w


def intrinsics_from_meta(
    meta: Dict, H: int, W: int,
) -> Tuple[float, float, float, float]:
    """Pinhole intrinsics matching the depth_warp convention (cx=W/2, cy=H/2)."""
    x_fov = float(meta["x_fov"])
    y_fov = float(meta.get("y_fov", meta["x_fov"]))
    fx = W / (2.0 * math.tan(x_fov * 0.5))
    fy = H / (2.0 * math.tan(y_fov * 0.5))
    cx = W / 2.0
    cy = H / 2.0
    return fx, fy, cx, cy


# ---------------------------------------------------------------------------
# Projection  (depth_warp convention – NO Y-negation)
# ---------------------------------------------------------------------------

def project_vertices_to_image(
    vertices: np.ndarray,
    c2w_unity: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D vertices to image coords (depth_warp convention, no Y flip).

    Returns pixel coords (N, 2), depths (N,), and in-frame mask (N,).
    """
    w2c = np.linalg.inv(c2w_unity)
    pts_cam = (w2c[:3, :3] @ vertices.T + w2c[:3, 3:4]).T
    z = pts_cam[:, 2]
    valid = z > 0.01
    u = fx * pts_cam[:, 0] / (z + 1e-8) + cx
    v = fy * pts_cam[:, 1] / (z + 1e-8) + cy
    in_frame = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return np.stack([u, v], axis=1), z, in_frame


# ---------------------------------------------------------------------------
# Texture extraction & UV rasteriser
# ---------------------------------------------------------------------------

def _rgb01_to_bgr_u8(rgb: np.ndarray) -> np.ndarray:
    """Convert float RGB in [0,1] to uint8 BGR for OpenCV buffers."""
    c = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return c[[2, 1, 0]]


def extract_mesh_texture(
    mesh: trimesh.Trimesh,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return per-vertex UVs (V,2) and texture image (H,W,3) float32 BGR in [0,1]."""
    visual = mesh.visual
    if not isinstance(visual, trimesh.visual.texture.TextureVisuals):
        return None, None
    if visual.uv is None:
        return None, None

    img = None
    material = visual.material
    if material is not None:
        img = getattr(material, "image", None)
        if img is None:
            img = getattr(material, "baseColorTexture", None)
    if img is None:
        return None, None

    if hasattr(img, "mode"):
        tex_rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    else:
        tex_rgb = np.asarray(img, dtype=np.float32)
        if tex_rgb.max() > 1.0:
            tex_rgb = tex_rgb / 255.0
        if tex_rgb.shape[-1] == 4:
            tex_rgb = tex_rgb[..., :3]
    tex_bgr = tex_rgb[..., ::-1].copy()
    uvs = np.asarray(visual.uv, dtype=np.float64)
    if len(uvs) != len(mesh.vertices):
        return None, None
    return uvs, tex_bgr


def _sample_texture_bgr(tex_bgr: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear-ish nearest sampling; UV in [0,1], glTF-style (v=0 bottom)."""
    h, w = tex_bgr.shape[:2]
    u = float(np.clip(uv[0], 0.0, 1.0))
    v = float(np.clip(uv[1], 0.0, 1.0))
    xi = int(u * (w - 1) + 0.5)
    yi = int((1.0 - v) * (h - 1) + 0.5)
    return tex_bgr[yi, xi]


def _barycentric_bulk(
    px: np.ndarray, py: np.ndarray,
    x0: float, y0: float, x1: float, y1: float, x2: float, y2: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Barycentric weights (w0,w1,w2) for points (px,py) in triangle (0,1,2)."""
    v0x, v0y = x2 - x0, y2 - y0
    v1x, v1y = x1 - x0, y1 - y0
    v2x, v2y = px - x0, py - y0
    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y
    denom = dot00 * dot11 - dot01 * dot01
    inv = 1.0 / (denom + 1e-12)
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    w0 = 1.0 - u - v
    return w0, v, u


def software_rasterize_textured(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_uvs: np.ndarray,
    texture_bgr: np.ndarray,
    c2w_unity: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
    vertex_colors_rgb: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterise with per-pixel UV texture lookup (unlit albedo)."""
    w2c = np.linalg.inv(c2w_unity)
    pts_cam = (w2c[:3, :3] @ vertices.T + w2c[:3, 3:4]).T
    z = pts_cam[:, 2]
    su = fx * pts_cam[:, 0] / (z + 1e-8) + cx
    sv = fy * pts_cam[:, 1] / (z + 1e-8) + cy

    color_buf = np.full((H, W, 3), 255, dtype=np.uint8)
    depth_buf = np.full((H, W), np.inf, dtype=np.float32)

    face_z = z[faces].mean(axis=1)
    order = np.argsort(-face_z)

    for idx in order:
        fi = faces[idx]
        zs = z[fi]
        if (zs < 0.01).any():
            continue

        x0, y0 = su[fi[0]], sv[fi[0]]
        x1, y1 = su[fi[1]], sv[fi[1]]
        x2, y2 = su[fi[2]], sv[fi[2]]

        tri_pts = np.array([[x0, y0], [x1, y1], [x2, y2]], dtype=np.int32)
        if (tri_pts[:, 0].max() < -W or tri_pts[:, 0].min() > 2 * W or
                tri_pts[:, 1].max() < -H or tri_pts[:, 1].min() > 2 * H):
            continue

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [tri_pts], 1)
        py, px = np.where(mask > 0)
        if len(px) == 0:
            continue

        w0, w1, w2 = _barycentric_bulk(
            px.astype(np.float64), py.astype(np.float64),
            x0, y0, x1, y1, x2, y2,
        )
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue
        px = px[inside]
        py = py[inside]
        w0, w1, w2 = w0[inside], w1[inside], w2[inside]

        z_pix = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
        closer = z_pix < depth_buf[py, px]
        if not closer.any():
            continue
        px = px[closer]
        py = py[closer]
        w0, w1, w2 = w0[closer], w1[closer], w2[closer]
        z_pix = z_pix[closer]

        uv0, uv1, uv2 = vertex_uvs[fi[0]], vertex_uvs[fi[1]], vertex_uvs[fi[2]]
        u_pix = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0]
        v_pix = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1]

        th, tw = texture_bgr.shape[:2]
        xi = np.clip((u_pix * (tw - 1) + 0.5).astype(np.int32), 0, tw - 1)
        yi = np.clip(((1.0 - v_pix) * (th - 1) + 0.5).astype(np.int32), 0, th - 1)
        colors = texture_bgr[yi, xi]  # (N, 3) float BGR

        depth_buf[py, px] = z_pix
        color_buf[py, px] = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    depth_out = np.where(depth_buf < np.inf, depth_buf, 0.0).astype(np.float32)
    return color_buf, depth_out


def software_rasterize_vertex_color(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors_rgb: np.ndarray,
    c2w_unity: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback rasteriser: flat per-face vertex colours (RGB→BGR corrected)."""
    w2c = np.linalg.inv(c2w_unity)
    pts_cam = (w2c[:3, :3] @ vertices.T + w2c[:3, 3:4]).T
    z = pts_cam[:, 2]
    u = fx * pts_cam[:, 0] / (z + 1e-8) + cx
    v = fy * pts_cam[:, 1] / (z + 1e-8) + cy

    color_buf = np.full((H, W, 3), 255, dtype=np.uint8)
    depth_buf = np.full((H, W), np.inf, dtype=np.float32)

    face_z = z[faces].mean(axis=1)
    order = np.argsort(-face_z)

    for idx in order:
        fi = faces[idx]
        zs = z[fi]
        if (zs < 0.01).any():
            continue
        tri_u = u[fi].astype(np.int32)
        tri_v = v[fi].astype(np.int32)
        if (tri_u < -W).all() or (tri_u > 2 * W).all():
            continue
        if (tri_v < -H).all() or (tri_v > 2 * H).all():
            continue

        mean_z = zs.mean()
        pts_arr = np.array([[tri_u[0], tri_v[0]],
                             [tri_u[1], tri_v[1]],
                             [tri_u[2], tri_v[2]]], dtype=np.int32)

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_arr], 1)
        fill = mask > 0
        closer = mean_z < depth_buf
        paint = fill & closer
        if not paint.any():
            continue
        color_buf[paint] = _rgb01_to_bgr_u8(face_colors_rgb[idx])
        depth_buf[paint] = mean_z

    depth_out = np.where(depth_buf < np.inf, depth_buf, 0.0).astype(np.float32)
    return color_buf, depth_out


# ---------------------------------------------------------------------------
# Software rasteriser  (legacy alias kept for compatibility)
# ---------------------------------------------------------------------------

def software_rasterize(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
    c2w_unity: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterise mesh triangles using G-Objaverse projection (no Y flip)."""
    return software_rasterize_vertex_color(
        vertices, faces, face_colors, c2w_unity, fx, fy, cx, cy, H, W,
    )


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def read_gt_rgb(path: str, H: int, W: int) -> np.ndarray:
    """Read G-Objaverse GT RGB, composite on white, return float32 BGR array."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    if img.shape[2] == 4:
        bgr = img[..., :3].astype(np.float32) / 255.0
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        bgr = bgr * alpha + (1.0 - alpha)
    else:
        bgr = img[..., :3].astype(np.float32) / 255.0
    if bgr.shape[0] != H or bgr.shape[1] != W:
        bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
    return bgr


def read_gt_depth(path: str, cam_origin: np.ndarray, max_depth: float,
                  H: int, W: int) -> np.ndarray:
    nd = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if nd is None:
        raise FileNotFoundError(f"Cannot read {path}")
    depth = nd[..., 3:4].astype(np.float32).copy()
    cam_dist = float(np.linalg.norm(cam_origin))
    near = cam_dist - math.sqrt(3.0) * 0.5
    depth[depth < near] = 0.0
    depth[depth > max_depth] = 0.0
    if depth.shape[0] != H or depth.shape[1] != W:
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
    if depth.ndim == 2:
        depth = depth[..., np.newaxis]
    return depth


def read_albedo_image(path: str, H: int, W: int) -> Optional[np.ndarray]:
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 4:
        bgr = img[..., :3].astype(np.float32) / 255.0
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        bgr = bgr * alpha + (1.0 - alpha)
    else:
        bgr = img[..., :3].astype(np.float32) / 255.0
    if bgr.shape[0] != H or bgr.shape[1] != W:
        bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
    return bgr


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def get_face_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """Per-face RGB in [0,1] from a trimesh (vertex-color average)."""
    from hy3dgen.shapegen.surface_loaders import _get_vertex_colors
    vertex_colors = _get_vertex_colors(mesh)
    face_vc = vertex_colors[mesh.faces]
    return face_vc.mean(axis=1)


def draw_projected_vertices(
    image: np.ndarray,
    uv: np.ndarray,
    in_frame: np.ndarray,
    colors: Optional[np.ndarray] = None,
    radius: int = 1,
    alpha: float = 0.6,
) -> np.ndarray:
    """Overlay projected vertices. ``colors`` are RGB float in [0,1]."""
    overlay = (image * 255).astype(np.uint8).copy()
    for i in np.where(in_frame)[0]:
        u, v = int(round(uv[i, 0])), int(round(uv[i, 1]))
        if colors is not None:
            rgb = np.asarray(colors[i, :3], dtype=np.float64)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            c = tuple(int(x) for x in _rgb01_to_bgr_u8(rgb))
        else:
            c = (0, 255, 0)
        cv2.circle(overlay, (u, v), radius, c, -1)
    blended = cv2.addWeighted(
        (image * 255).astype(np.uint8), 1.0 - alpha,
        overlay, alpha, 0,
    )
    return blended


def color_diff_heatmap(img_a: np.ndarray, img_b: np.ndarray,
                       mask: np.ndarray, gain: float = 4.0) -> np.ndarray:
    """Visualise per-pixel BGR L1 error (magma heatmap, white background)."""
    m = mask.astype(bool)
    if m.ndim == 3:
        m = m[..., 0]
    diff = np.abs(img_a - img_b).mean(axis=-1)
    if m.any():
        vmax = max(float(diff[m].max()), 1e-6)
    else:
        vmax = 1.0
    norm = np.zeros_like(diff)
    norm[m] = np.clip(diff[m] / vmax * gain, 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    heat[~m] = 255
    return heat


# ---------------------------------------------------------------------------
# Surface point cloud (training pipeline)
# ---------------------------------------------------------------------------

def extract_surface_pointcloud(
    mesh_path: str,
    meta: Dict,
    num_points: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample surface points with colours using the training pipeline."""
    from hy3dgen.shapegen.gobjaverse_gt import normalize_mesh_gobjaverse
    from hy3dgen.shapegen.surface_loaders import sample_pointcloud_with_color, scene_to_geometry

    raw_mesh = trimesh.load(mesh_path, process=False)
    mesh_full = scene_to_geometry(raw_mesh)

    mesh_norm = normalize_mesh_gobjaverse(mesh_full, meta)
    pts, nrm, clr = sample_pointcloud_with_color(mesh_norm, num=num_points)
    return pts.numpy(), nrm.numpy(), clr.numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def silhouette_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / max(union, 1))


def masked_color_l1(img_a: np.ndarray, img_b: np.ndarray,
                    mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if m.ndim == 3:
        m = m[..., 0]
    if not m.any():
        return float("nan")
    return float(np.abs(img_a[m] - img_b[m]).mean())


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def diagnose_object(
    mesh_path: str,
    gobjaverse_id: str,
    render_root: str,
    output_dir: Path,
    H: int = 512, W: int = 512,
    view_indices: Optional[List[int]] = None,
    max_views: int = 6,
) -> Dict:
    """Run full diagnostics on one object."""
    render_dir = Path(render_root) / gobjaverse_id
    obj_name = Path(mesh_path).stem

    meta_path = render_dir / "00000" / "00000.json"
    if not meta_path.exists():
        logger.warning(f"No renders for {obj_name} at {render_dir}")
        return {"mesh_path": mesh_path, "error": "no renders"}

    meta = load_view_meta(str(meta_path))

    if view_indices is None:
        available = sorted(int(d) for d in os.listdir(render_dir) if d.isdigit())
        view_indices = available[:max_views]

    mesh_combined, mesh_norm = load_and_prepare_mesh(mesh_path, meta)

    from hy3dgen.shapegen.surface_loaders import _get_vertex_colors
    vertex_colors = _get_vertex_colors(mesh_norm)
    face_colors = get_face_colors(mesh_norm)

    vertex_uvs, texture_bgr = extract_mesh_texture(mesh_norm)
    has_uv_texture = vertex_uvs is not None and texture_bgr is not None
    if has_uv_texture:
        logger.info(f"  {obj_name}: UV texture {texture_bgr.shape[1]}x{texture_bgr.shape[0]}")
    else:
        logger.info(f"  {obj_name}: no UV texture, using vertex colours")

    pts, _, pt_colors = extract_surface_pointcloud(mesh_path, meta, num_points=20000)

    obj_dir = output_dir / obj_name
    obj_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "mesh_path": mesh_path,
        "gobjaverse_id": gobjaverse_id,
        "n_vertices": len(mesh_norm.vertices),
        "n_faces": len(mesh_norm.faces),
        "views": [],
    }

    per_view_panels = []

    for view_idx in view_indices:
        stem = f"{view_idx:05d}"
        view_dir = render_dir / stem
        json_path = view_dir / f"{stem}.json"
        rgb_path = view_dir / f"{stem}.png"
        nd_path = view_dir / f"{stem}_nd.exr"
        albedo_path = view_dir / f"{stem}_albedo.png"

        if not json_path.exists():
            continue

        view_meta = load_view_meta(str(json_path))
        max_depth = float(view_meta.get("max_depth", 5.0))

        gt_rgb = read_gt_rgb(str(rgb_path), H, W)
        gt_depth = read_gt_depth(str(nd_path), unity_c2w_from_meta(view_meta)[:3, 3],
                                 max_depth, H, W)
        gt_albedo = read_albedo_image(str(albedo_path), H, W)

        c2w_unity = unity_c2w_from_meta(view_meta)
        fx, fy, cx, cy = intrinsics_from_meta(view_meta, H, W)

        # 1. UV-textured rasterisation (unlit albedo) or vertex-colour fallback
        try:
            if has_uv_texture:
                rast_color, rast_depth = software_rasterize_textured(
                    mesh_norm.vertices, mesh_norm.faces, vertex_uvs, texture_bgr,
                    c2w_unity, fx, fy, cx, cy, H, W)
            else:
                rast_color, rast_depth = software_rasterize_vertex_color(
                    mesh_norm.vertices, mesh_norm.faces, face_colors,
                    c2w_unity, fx, fy, cx, cy, H, W)
            mesh_bgr = rast_color.astype(np.float32) / 255.0
            pr_depth = rast_depth
        except Exception as e:
            logger.warning(f"Rasterize failed for {obj_name} view {view_idx}: {e}")
            mesh_bgr = np.ones((H, W, 3), dtype=np.float32)
            pr_depth = np.zeros((H, W), dtype=np.float32)

        # 2. Vertex projection overlay on GT lit render
        uv, z, in_frame = project_vertices_to_image(
            mesh_norm.vertices, c2w_unity, fx, fy, cx, cy, H, W)
        overlay_green = draw_projected_vertices(gt_rgb, uv, in_frame,
                                                 colors=None, radius=1, alpha=0.4)

        # 3. Surface point-cloud overlay
        uv_pts, z_pts, in_pts = project_vertices_to_image(
            pts, c2w_unity, fx, fy, cx, cy, H, W)
        overlay_pts = draw_projected_vertices(gt_rgb, uv_pts, in_pts,
                                                colors=pt_colors, radius=1, alpha=0.7)

        # 4. Metrics
        gt_mask = (gt_depth[..., 0] > 0).astype(np.float32)
        pr_mask = (pr_depth > 0.01).astype(np.float32)
        iou = silhouette_iou(gt_mask, pr_mask)

        # Primary colour metric: mesh unlit render vs GT albedo
        if gt_albedo is not None:
            albedo_l1 = masked_color_l1(gt_albedo, mesh_bgr, gt_mask)
        else:
            albedo_l1 = float("nan")

        lit_rgb_l1 = masked_color_l1(gt_rgb, mesh_bgr, gt_mask)

        gt_fg_d = gt_depth[..., 0][gt_mask > 0]
        pr_fg_d = pr_depth[gt_mask > 0] if pr_depth.ndim == 2 else pr_depth[..., 0][gt_mask > 0]
        if len(gt_fg_d) > 0 and len(pr_fg_d) > 0 and pr_fg_d.std() > 0:
            depth_corr = float(np.corrcoef(gt_fg_d, pr_fg_d)[0, 1])
            depth_l1 = float(np.abs(gt_fg_d - pr_fg_d).mean())
        else:
            depth_corr = float("nan")
            depth_l1 = float("nan")

        vtx_coverage = float(in_frame.sum()) / max(len(in_frame), 1)

        view_report = {
            "view_idx": view_idx,
            "silhouette_iou": round(iou, 4),
            "albedo_l1": round(albedo_l1, 4) if not math.isnan(albedo_l1) else None,
            "lit_rgb_l1": round(lit_rgb_l1, 4) if not math.isnan(lit_rgb_l1) else None,
            "fg_color_l1": round(albedo_l1, 4) if not math.isnan(albedo_l1) else None,
            "depth_correlation": round(depth_corr, 4) if not math.isnan(depth_corr) else None,
            "depth_l1": round(depth_l1, 4) if not math.isnan(depth_l1) else None,
            "vertex_coverage": round(vtx_coverage, 4),
            "has_uv_texture": has_uv_texture,
        }
        report["views"].append(view_report)

        # ---- Build visualisation panel ----
        gt_lit_u8 = (np.clip(gt_rgb, 0, 1) * 255).astype(np.uint8)
        mesh_u8 = (np.clip(mesh_bgr, 0, 1) * 255).astype(np.uint8)

        if gt_albedo is not None:
            albedo_u8 = (np.clip(gt_albedo, 0, 1) * 255).astype(np.uint8)
            albedo_diff = color_diff_heatmap(gt_albedo, mesh_bgr, gt_mask, gain=4.0)
        else:
            albedo_u8 = np.full((H, W, 3), 200, dtype=np.uint8)
            albedo_diff = np.full((H, W, 3), 200, dtype=np.uint8)

        def depth_colormap(d, vmin=None, vmax=None):
            m = d > 0
            if not m.any():
                return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
            if vmin is None:
                vmin = d[m].min()
            if vmax is None:
                vmax = d[m].max()
            norm = np.zeros_like(d)
            norm[m] = (d[m] - vmin) / max(vmax - vmin, 1e-6)
            norm = np.clip(norm, 0, 1)
            cmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            cmap[~m] = 255
            return cmap

        d_gt = gt_depth[..., 0] if gt_depth.ndim == 3 else gt_depth
        d_pr = pr_depth if pr_depth.ndim == 2 else pr_depth[..., 0]
        vmin = d_gt[d_gt > 0].min() if (d_gt > 0).any() else 0
        vmax = d_gt[d_gt > 0].max() if (d_gt > 0).any() else 1
        gt_depth_viz = depth_colormap(d_gt, vmin, vmax)
        pr_depth_viz = depth_colormap(d_pr, vmin, vmax)

        text_img = np.full((H, W, 3), 30, dtype=np.uint8)
        tex_mode = "UV" if has_uv_texture else "VtxColor"
        lines = [
            f"View {view_idx}  ({tex_mode})",
            f"Silhouette IoU: {iou:.3f}",
            f"Albedo L1: {albedo_l1:.4f}" if not math.isnan(albedo_l1) else "Albedo L1: N/A",
            f"Lit RGB L1: {lit_rgb_l1:.4f}" if not math.isnan(lit_rgb_l1) else "Lit RGB L1: N/A",
            f"Depth Corr: {depth_corr:.3f}" if not math.isnan(depth_corr) else "Depth Corr: N/A",
            f"Depth L1: {depth_l1:.4f}" if not math.isnan(depth_l1) else "Depth L1: N/A",
            f"Vtx Coverage: {vtx_coverage:.3f}",
        ]
        for li, line in enumerate(lines):
            cv2.putText(text_img, line, (10, 26 + li * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        row1 = np.concatenate(
            [albedo_u8, mesh_u8, albedo_diff, gt_lit_u8, overlay_green], axis=1)
        row2 = np.concatenate(
            [gt_depth_viz, pr_depth_viz, overlay_pts, np.zeros_like(gt_lit_u8), text_img],
            axis=1,
        )

        label_h = 24
        label_row1 = np.full((label_h, row1.shape[1], 3), 0, dtype=np.uint8)
        labels1 = ["GT Albedo", "Mesh UV Render", "Albedo Diff", "GT Lit Render", "Vtx Proj"]
        for ci, lab in enumerate(labels1):
            cv2.putText(label_row1, lab, (ci * W + 5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        label_row2 = np.full((label_h, row2.shape[1], 3), 0, dtype=np.uint8)
        labels2 = ["GT Depth", "Mesh Depth", "PC Proj", "", "Metrics"]
        for ci, lab in enumerate(labels2):
            cv2.putText(label_row2, lab, (ci * W + 5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        panel = np.concatenate([label_row1, row1, label_row2, row2], axis=0)
        per_view_panels.append(panel)

        cv2.imwrite(str(obj_dir / f"view_{view_idx:02d}.png"), panel)

    if per_view_panels:
        full_grid = np.concatenate(per_view_panels, axis=0)
        cv2.imwrite(str(obj_dir / "all_views.png"), full_grid)

    if report["views"]:
        ious = [v["silhouette_iou"] for v in report["views"]]
        report["mean_silhouette_iou"] = round(float(np.mean(ious)), 4)
        albedo_l1s = [v["albedo_l1"] for v in report["views"] if v.get("albedo_l1") is not None]
        if albedo_l1s:
            report["mean_albedo_l1"] = round(float(np.mean(albedo_l1s)), 4)
            report["mean_fg_color_l1"] = report["mean_albedo_l1"]
        lit_l1s = [v["lit_rgb_l1"] for v in report["views"] if v.get("lit_rgb_l1") is not None]
        if lit_l1s:
            report["mean_lit_rgb_l1"] = round(float(np.mean(lit_l1s)), 4)
        depth_corrs = [v["depth_correlation"] for v in report["views"] if v["depth_correlation"] is not None]
        if depth_corrs:
            report["mean_depth_correlation"] = round(float(np.mean(depth_corrs)), 4)

    with open(obj_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(
        f"  {obj_name}: IoU={report.get('mean_silhouette_iou', 'N/A'):.3f} "
        f"AlbedoL1={report.get('mean_albedo_l1', 'N/A')} "
        f"LitL1={report.get('mean_lit_rgb_l1', 'N/A')} "
        f"DepthCorr={report.get('mean_depth_correlation', 'N/A')}"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="G-Objaverse data diagnostic")
    parser.add_argument("--manifest", type=str,
                        default="/export/home/nathan/datasets/gobjaverse_experiments/furniture_351/manifest.json")
    parser.add_argument("--output_dir", type=str, default="diagnose_gobjaverse_output")
    parser.add_argument("--max_items", type=int, default=5)
    parser.add_argument("--max_views", type=int, default=6,
                        help="Max views per object (default 6, evenly spaced from 40)")
    parser.add_argument("--split", choices=("train", "val", "all"), default="train")
    parser.add_argument("--mesh_indices", type=str, default=None,
                        help="Comma-separated indices into the mesh list (e.g. '0,7,15')")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    render_root = manifest["render_root"]
    H = int(manifest.get("render_height", 512))
    W = int(manifest.get("render_width", 512))
    mesh_map = manifest.get("mesh_to_gobjaverse_id", {})

    if args.split == "train":
        mesh_paths = manifest.get("train_mesh_paths", [])
    elif args.split == "val":
        mesh_paths = manifest.get("val_mesh_paths", [])
    else:
        mesh_paths = manifest.get("train_mesh_paths", []) + manifest.get("val_mesh_paths", [])

    if args.mesh_indices:
        indices = [int(x.strip()) for x in args.mesh_indices.split(",")]
        mesh_paths = [mesh_paths[i] for i in indices if i < len(mesh_paths)]
    elif args.max_items and len(mesh_paths) > args.max_items:
        step = max(1, len(mesh_paths) // args.max_items)
        mesh_paths = mesh_paths[::step][:args.max_items]

    total_views = int(manifest.get("num_views", 40))
    if args.max_views < total_views:
        step = max(1, total_views // args.max_views)
        view_indices = list(range(0, total_views, step))[:args.max_views]
    else:
        view_indices = list(range(total_views))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Diagnosing {len(mesh_paths)} objects, {len(view_indices)} views each")
    logger.info(f"View indices: {view_indices}")
    logger.info(f"Output: {output_dir}")

    all_reports = []
    for i, mesh_path in enumerate(mesh_paths):
        real_path = os.path.realpath(mesh_path)
        gobj_id = mesh_map.get(real_path) or mesh_map.get(mesh_path)
        if not gobj_id:
            logger.warning(f"No gobjaverse ID for {mesh_path}")
            continue

        logger.info(f"[{i+1}/{len(mesh_paths)}] {Path(mesh_path).stem} (ID={gobj_id})")
        try:
            report = diagnose_object(
                mesh_path, gobj_id, render_root, output_dir,
                H=H, W=W, view_indices=view_indices, max_views=args.max_views,
            )
            all_reports.append(report)
        except Exception as e:
            logger.error(f"Failed on {mesh_path}: {e}", exc_info=True)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_reports, f, indent=2)

    if all_reports:
        valid = [r for r in all_reports if "error" not in r]
        if valid:
            mean_iou = np.mean([r["mean_silhouette_iou"] for r in valid
                                if "mean_silhouette_iou" in r])
            logger.info(f"\n=== SUMMARY ({len(valid)} objects) ===")
            logger.info(f"  Mean silhouette IoU: {mean_iou:.3f}")
            albedo_l1s = [r["mean_albedo_l1"] for r in valid
                          if "mean_albedo_l1" in r]
            if albedo_l1s:
                logger.info(f"  Mean albedo L1 (mesh vs GT albedo): {np.mean(albedo_l1s):.4f}")
            lit_l1s = [r["mean_lit_rgb_l1"] for r in valid
                       if "mean_lit_rgb_l1" in r]
            if lit_l1s:
                logger.info(f"  Mean lit RGB L1 (mesh vs GT lit): {np.mean(lit_l1s):.4f}")
            depth_corrs = [r["mean_depth_correlation"] for r in valid
                           if "mean_depth_correlation" in r]
            if depth_corrs:
                logger.info(f"  Mean depth correlation: {np.mean(depth_corrs):.4f}")

    logger.info(f"Done. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
