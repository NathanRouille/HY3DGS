import hashlib
import os
from contextlib import contextmanager
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np

import torch
import trimesh


def stable_mesh_seed(global_seed: int, mesh_path: Union[str, os.PathLike]) -> int:
    """Per-mesh seed derived from the training seed and canonical mesh path."""
    canonical = os.path.realpath(os.path.abspath(str(mesh_path)))
    digest = hashlib.sha256(f"{int(global_seed)}:{canonical}".encode()).hexdigest()
    return int(digest[:8], 16)


@contextmanager
def _numpy_seed_context(seed: Optional[int]) -> Iterator[np.random.Generator]:
    """Seed legacy ``np.random`` (for trimesh) and yield a ``default_rng``."""
    if seed is None:
        yield np.random.default_rng()
        return
    legacy_state = np.random.get_state()
    np.random.seed(int(seed) % (2**32))
    try:
        yield np.random.default_rng(int(seed))
    finally:
        np.random.set_state(legacy_state)


def scene_to_parts(mesh) -> List[trimesh.Trimesh]:
    """Return scene sub-meshes with scene transforms baked in, each keeping its own material/UV."""
    if isinstance(mesh, trimesh.Trimesh):
        return [mesh]
    if isinstance(mesh, trimesh.Scene):
        return [
            p for p in mesh.dump(concatenate=False)
            if isinstance(p, trimesh.Trimesh)
        ]
    if isinstance(mesh, (list, tuple)):
        return [p for p in mesh if isinstance(p, trimesh.Trimesh)]
    return [trimesh.util.concatenate(mesh)]


def normalize_parts_gobjaverse(parts: List[trimesh.Trimesh], meta: dict) -> List[trimesh.Trimesh]:
    """Apply G-Objaverse coord fix + render normalization to every part consistently."""
    from hy3dgen.shapegen.gobjaverse_gt import apply_gobjaverse_coord_fix

    fixed: List[trimesh.Trimesh] = []
    for part in parts:
        p = part.copy()
        visual = p.visual
        p.vertices = apply_gobjaverse_coord_fix(p.vertices)
        try:
            p.visual = visual
        except Exception:
            pass
        fixed.append(p)

    if not fixed:
        return fixed

    vmin = np.min([p.bounds[0] for p in fixed], axis=0)
    vmax = np.max([p.bounds[1] for p in fixed], axis=0)
    centroid = (vmin + vmax) * 0.5
    scale = float(meta["scale"][0])

    out: List[trimesh.Trimesh] = []
    for part in fixed:
        p = part.copy()
        visual = p.visual
        p.vertices = (p.vertices.astype(np.float64) - centroid) * scale
        try:
            p.visual = visual
        except Exception:
            pass
        out.append(p)
    return out


def normalize_parts(parts: List[trimesh.Trimesh], scale: float = 0.9999) -> List[trimesh.Trimesh]:
    """Center and scale a list of parts using their combined bounding box."""
    if not parts:
        return parts
    unified, _ = parts_to_trimesh(parts)
    bbox = unified.bounds
    center = (bbox[1] + bbox[0]) / 2
    scale_ = (bbox[1] - bbox[0]).max()
    if scale_ <= 0:
        return [p.copy() for p in parts]

    out: List[trimesh.Trimesh] = []
    for part in parts:
        p = part.copy()
        p.apply_translation(-center)
        p.apply_scale(1 / scale_ * 2 * scale)
        out.append(p)
    return out


def merge_parts_vertex_colored(parts: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Merge parts into one mesh for export/visualization (vertex colors when multi-material)."""
    if not parts:
        return trimesh.Trimesh()
    if len(parts) == 1:
        uv, tex = extract_mesh_texture(parts[0])
        if uv is not None and tex is not None:
            return parts[0]
        return _bake_geometry_to_vertex_colors(parts[0])
    colored_parts = [_bake_geometry_to_vertex_colors(p) for p in parts]
    return trimesh.util.concatenate(colored_parts)


def parts_to_trimesh(parts: List[trimesh.Trimesh]) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """Concatenate part geometries; return unified mesh and per-face part indices."""
    if not parts:
        empty = trimesh.Trimesh()
        return empty, np.zeros(0, dtype=np.int32)

    verts_list: List[np.ndarray] = []
    faces_list: List[np.ndarray] = []
    face_part_ids: List[np.ndarray] = []
    v_offset = 0
    for pid, part in enumerate(parts):
        verts_list.append(part.vertices)
        faces_list.append(part.faces + v_offset)
        face_part_ids.append(np.full(len(part.faces), pid, dtype=np.int32))
        v_offset += len(part.vertices)

    mesh = trimesh.Trimesh(
        vertices=np.vstack(verts_list),
        faces=np.vstack(faces_list),
        process=False,
    )
    return mesh, np.concatenate(face_part_ids)


def _allocate_part_counts(
    areas: np.ndarray,
    total: int,
    *,
    min_per_part: int = 64,
) -> np.ndarray:
    """Area-proportional point budget with a floor so tiny parts are not starved."""
    areas = np.asarray(areas, dtype=np.float64)
    n_parts = len(areas)
    if n_parts == 0:
        return np.zeros(0, dtype=np.int32)
    if total <= 0:
        return np.zeros(n_parts, dtype=np.int32)

    min_total = min_per_part * n_parts
    if total < min_total:
        min_per_part = max(1, total // n_parts)
        min_total = min_per_part * n_parts

    residual = max(0, total - min_total)
    area_sum = float(areas.sum())
    if area_sum <= 0:
        extra = np.full(n_parts, residual // n_parts, dtype=np.int32)
        extra[: residual % n_parts] += 1
    else:
        extra = np.floor(residual * (areas / area_sum)).astype(np.int32)
        remainder = residual - int(extra.sum())
        if remainder > 0:
            order = np.argsort(-(areas - (extra / max(residual, 1)) * area_sum))
            extra[order[:remainder]] += 1

    counts = extra + min_per_part
    diff = total - int(counts.sum())
    if diff > 0:
        order = np.argsort(-areas)
        counts[order[:diff]] += 1
    elif diff < 0:
        order = np.argsort(areas)
        for idx in order:
            if diff == 0:
                break
            reducible = counts[idx] - min_per_part
            if reducible <= 0:
                continue
            take = min(reducible, -diff)
            counts[idx] -= take
            diff += take
    return counts.astype(np.int32)


def _bake_geometry_to_vertex_colors(geom: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Resolve one sub-mesh's own texture/material to per-vertex RGB.

    Uses the geometry's own (un-merged) UV + texture, so this is a direct,
    exact lookup into the texture that geometry's UVs were actually authored
    against -- no atlas packing, no remapping, nothing to get wrong.
    """
    uv, tex = extract_mesh_texture(geom)
    if uv is not None and tex is not None:
        rgb = _sample_texture_rgb_batch(tex, uv)
    else:
        rgb = _get_vertex_colors(geom)
    rgba = np.concatenate(
        [np.clip(rgb, 0.0, 1.0), np.ones((len(rgb), 1), dtype=np.float32)], axis=1,
    )
    geom = geom.copy()
    geom.visual = trimesh.visual.ColorVisuals(
        mesh=geom, vertex_colors=(rgba * 255.0).round().astype(np.uint8),
    )
    return geom


def scene_to_geometry(mesh) -> trimesh.Trimesh:
    """Collapse a Scene or Trimesh to a single Trimesh (preserves scene transforms).

    Multi-material scenes are merged via per-part vertex-color baking (no texture
    atlas packing). Single-material meshes keep their original UV texture.
    """
    parts = scene_to_parts(mesh)
    if len(parts) == 0:
        return trimesh.Trimesh()
    return merge_parts_vertex_colored(parts)


def normalize_mesh(mesh, scale=0.9999):
    bbox = mesh.bounds
    center = (bbox[1] + bbox[0]) / 2
    scale_ = (bbox[1] - bbox[0]).max()

    mesh.apply_translation(-center)
    mesh.apply_scale(1 / scale_ * 2 * scale)

    return mesh


def sample_pointcloud(mesh, num=200000):
    points, face_idx = mesh.sample(num, return_index=True)
    normals = mesh.face_normals[face_idx]
    points = torch.from_numpy(points.astype(np.float32))
    normals = torch.from_numpy(normals.astype(np.float32))
    return points, normals


def load_surface(mesh, num_points=8192):
    mesh = normalize_mesh(mesh, scale=0.98)
    surface, normal = sample_pointcloud(mesh)

    rng = np.random.default_rng(seed=0)
    ind = rng.choice(surface.shape[0], num_points, replace=False)
    surface = torch.FloatTensor(surface[ind])
    normal = torch.FloatTensor(normal[ind])

    surface = torch.cat([surface, normal], dim=-1).unsqueeze(0)

    return surface, mesh


def sharp_sample_pointcloud(mesh, num=16384):
    V = mesh.vertices
    N = mesh.face_normals
    VN = mesh.vertex_normals
    F = mesh.faces
    VN2 = np.ones(V.shape[0])
    for i in range(3):
        dot = np.stack((VN2[F[:, i]], np.sum(VN[F[:, i]] * N, axis=-1)), axis=-1)
        VN2[F[:, i]] = np.min(dot, axis=-1)

    sharp_mask = VN2 < 0.985
    # collect edge
    edge_a = np.concatenate((F[:, 0], F[:, 1], F[:, 2]))
    edge_b = np.concatenate((F[:, 1], F[:, 2], F[:, 0]))
    sharp_edge = ((sharp_mask[edge_a] * sharp_mask[edge_b]))
    edge_a = edge_a[sharp_edge > 0]
    edge_b = edge_b[sharp_edge > 0]

    sharp_verts_a = V[edge_a]
    sharp_verts_b = V[edge_b]
    sharp_verts_an = VN[edge_a]
    sharp_verts_bn = VN[edge_b]

    weights = np.linalg.norm(sharp_verts_b - sharp_verts_a, axis=-1)
    weights /= np.sum(weights)

    random_number = np.random.rand(num)
    w = np.random.rand(num, 1)
    index = np.searchsorted(weights.cumsum(), random_number)
    samples = w * sharp_verts_a[index] + (1 - w) * sharp_verts_b[index]
    normals = w * sharp_verts_an[index] + (1 - w) * sharp_verts_bn[index]
    return samples, normals


def load_surface_sharpegde(
    mesh,
    num_points=4096,
    num_sharp_points=4096,
    sharpedge_flag=True,
    gobjaverse_meta: Optional[dict] = None,
):
    try:
        mesh_full = scene_to_geometry(mesh)
    except Exception:
        mesh_full = trimesh.util.concatenate(mesh)
    if gobjaverse_meta is not None:
        from hy3dgen.shapegen.gobjaverse_gt import normalize_mesh_gobjaverse
        mesh_full = normalize_mesh_gobjaverse(mesh_full, gobjaverse_meta)
    else:
        mesh_full = normalize_mesh(mesh_full)

    origin_num = mesh_full.faces.shape[0]
    original_vertices = mesh_full.vertices
    original_faces = mesh_full.faces

    # process=False: preserve exact vertex indices/order (no re-merging), since
    # downstream code relies on original_vertices indexing staying valid.
    mesh = trimesh.Trimesh(vertices=original_vertices, faces=original_faces[:origin_num], process=False)
    mesh_fill = trimesh.Trimesh(vertices=original_vertices, faces=original_faces[origin_num:], process=False)
    area = mesh.area
    area_fill = mesh_fill.area
    sample_num = 499712 // 2
    num_fill = int(sample_num * (area_fill / (area + area_fill)))
    num = sample_num - num_fill

    random_surface, random_normal = sample_pointcloud(mesh, num=num)
    if num_fill == 0:
        random_surface_fill, random_normal_fill = np.zeros((0, 3)), np.zeros((0, 3))
    else:
        random_surface_fill, random_normal_fill = sample_pointcloud(mesh_fill, num=num_fill)
    random_sharp_surface, sharp_normal = sharp_sample_pointcloud(mesh, num=sample_num)

    # save_surface
    surface = np.concatenate((random_surface, random_normal), axis=1).astype(np.float16)
    surface_fill = np.concatenate((random_surface_fill, random_normal_fill), axis=1).astype(np.float16)
    sharp_surface = np.concatenate((random_sharp_surface, sharp_normal), axis=1).astype(np.float16)
    surface = np.concatenate((surface, surface_fill), axis=0)
    if sharpedge_flag:
        sharpedge_label = np.zeros((surface.shape[0], 1))
        surface = np.concatenate((surface, sharpedge_label), axis=1)
        sharpedge_label = np.ones((sharp_surface.shape[0], 1))
        sharp_surface = np.concatenate((sharp_surface, sharpedge_label), axis=1)
    rng = np.random.default_rng()
    ind = rng.choice(surface.shape[0], num_points, replace=False)
    surface = torch.FloatTensor(surface[ind])
    ind = rng.choice(sharp_surface.shape[0], num_sharp_points, replace=False)
    sharp_surface = torch.FloatTensor(sharp_surface[ind])

    return torch.cat([surface, sharp_surface], dim=0).unsqueeze(0), mesh_full


class SurfaceLoader:
    def __init__(self, num_points=8192):
        self.num_points = num_points

    def __call__(self, mesh_or_mesh_path, num_points=None):
        if num_points is None:
            num_points = self.num_points

        mesh = mesh_or_mesh_path
        if isinstance(mesh, str):
            mesh = trimesh.load(mesh, force="mesh", merge_primitives=True)
        if isinstance(mesh, trimesh.scene.Scene):
            for idx, obj in enumerate(mesh.geometry.values()):
                if idx == 0:
                    temp_mesh = obj
                else:
                    temp_mesh = temp_mesh + obj
            mesh = temp_mesh
        surface, mesh = load_surface(mesh, num_points=num_points)
        return surface


class SharpEdgeSurfaceLoader:
    def __init__(self, num_uniform_points=8192, num_sharp_points=8192, **kwargs):
        self.num_uniform_points = num_uniform_points
        self.num_sharp_points = num_sharp_points
        self.num_points = num_uniform_points + num_sharp_points

    def __call__(self, mesh_or_mesh_path, num_uniform_points=None, num_sharp_points=None):
        if num_uniform_points is None:
            num_uniform_points = self.num_uniform_points
        if num_sharp_points is None:
            num_sharp_points = self.num_sharp_points

        mesh = mesh_or_mesh_path
        if isinstance(mesh, str):
            mesh = trimesh.load(mesh, force="mesh", merge_primitives=True)
        if isinstance(mesh, trimesh.scene.Scene):
            for idx, obj in enumerate(mesh.geometry.values()):
                if idx == 0:
                    temp_mesh = obj
                else:
                    temp_mesh = temp_mesh + obj
            mesh = temp_mesh
        surface, mesh = load_surface_sharpegde(mesh, num_points=num_uniform_points, num_sharp_points=num_sharp_points)
        return surface


# ---------------------------------------------------------------------------
# RGB-aware helpers
# ---------------------------------------------------------------------------

def extract_mesh_texture(
    mesh: trimesh.Trimesh,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return per-vertex UVs (V,2) and base-color texture (H,W,3) float32 RGB in [0,1].

    Reads ``material.image`` or PBR ``baseColorTexture`` from TextureVisuals.
    """
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
    uvs = np.asarray(visual.uv, dtype=np.float64)
    if len(uvs) != len(mesh.vertices):
        return None, None
    return uvs, tex_rgb.astype(np.float32)


def _sample_texture_rgb_batch(tex_rgb: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """Sample RGB texture at (N,2) UV coordinates; glTF convention (v=0 at bottom)."""
    h, w = tex_rgb.shape[:2]
    u = np.clip(uvs[:, 0], 0.0, 1.0) * (w - 1)
    v = np.clip(1.0 - uvs[:, 1], 0.0, 1.0) * (h - 1)

    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (u - x0).astype(np.float32)[:, None]
    wy = (v - y0).astype(np.float32)[:, None]

    c00 = tex_rgb[y0, x0]
    c10 = tex_rgb[y0, x1]
    c01 = tex_rgb[y1, x0]
    c11 = tex_rgb[y1, x1]
    top = c00 * (1.0 - wx) + c10 * wx
    bot = c01 * (1.0 - wx) + c11 * wx
    return (top * (1.0 - wy) + bot * wy).astype(np.float32)


def _get_vertex_colors(mesh):
    """Return float32 RGB vertex colors in [0, 1] for the given trimesh Trimesh.

    Handles vertex-color meshes, solid PBR materials, UV-textured meshes
    (via ``to_color()``), and meshes with no color (neutral gray 0.5).
    """
    visual = mesh.visual

    if isinstance(visual, trimesh.visual.color.ColorVisuals):
        vc = visual.vertex_colors
        if vc is not None and len(vc) == len(mesh.vertices):
            return (np.asarray(vc)[:, :3] / 255.0).astype(np.float32)

    try:
        color_mesh = visual.to_color()
        vc = color_mesh.vertex_colors  # (V, 4) RGBA uint8
        return (vc[:, :3] / 255.0).astype(np.float32)
    except Exception:
        pass

    if isinstance(visual, trimesh.visual.texture.TextureVisuals):
        material = visual.material
        if material is not None:
            color = getattr(material, "baseColorFactor", None)
            if color is None:
                color = getattr(material, "main_color", None)
            if color is not None:
                c = np.asarray(color, dtype=np.float32).reshape(-1)[:3]
                if c.max() > 1.0:
                    c = c / 255.0
                return np.broadcast_to(c, (len(mesh.vertices), 3)).copy()

    return np.full((len(mesh.vertices), 3), 0.5, dtype=np.float32)


def _colors_at_surface_points(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    face_idx: np.ndarray,
    vertex_uvs: Optional[np.ndarray] = None,
    texture_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Barycentric color at surface sample points: UV texture lookup when available."""
    face_verts = mesh.faces[face_idx]
    tri_verts = mesh.vertices[face_verts]
    bary = trimesh.triangles.points_to_barycentric(tri_verts, points)
    bary = np.clip(bary, 0, 1)
    bary /= bary.sum(axis=1, keepdims=True) + 1e-8

    if vertex_uvs is not None and texture_rgb is not None:
        uv_face = vertex_uvs[face_verts]
        uv_pts = (bary[:, :, None] * uv_face).sum(axis=1)
        return _sample_texture_rgb_batch(texture_rgb, uv_pts)

    vertex_colors = _get_vertex_colors(mesh)
    face_colors = vertex_colors[face_verts]
    return (bary[:, :, None] * face_colors).sum(axis=1).astype(np.float32)


def _colors_at_surface_points_parts(
    parts: List[trimesh.Trimesh],
    points: np.ndarray,
    global_face_idx: np.ndarray,
    face_part_ids: np.ndarray,
) -> np.ndarray:
    """Per-point color using each part's own UV/texture (no atlas packing)."""
    colors = np.empty((len(points), 3), dtype=np.float32)
    part_ids = face_part_ids[global_face_idx]
    face_offsets = np.zeros(len(parts), dtype=np.int64)
    offset = 0
    for i, part in enumerate(parts):
        face_offsets[i] = offset
        offset += len(part.faces)

    for pid, part in enumerate(parts):
        mask = part_ids == pid
        if not mask.any():
            continue
        local_face_idx = global_face_idx[mask] - face_offsets[pid]
        vertex_uvs, texture_rgb = extract_mesh_texture(part)
        colors[mask] = _colors_at_surface_points(
            part, points[mask], local_face_idx, vertex_uvs, texture_rgb,
        )
    return colors


def _sample_parts_with_color(
    parts: List[trimesh.Trimesh],
    counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-quota surface samples aggregated across parts."""
    pts_all, nrm_all, clr_all = [], [], []
    for part, num in zip(parts, counts):
        if num <= 0:
            continue
        pts, nrm, clr = sample_pointcloud_with_color(part, num=int(num))
        pts_all.append(pts.numpy())
        nrm_all.append(nrm.numpy())
        clr_all.append(clr.numpy())

    if not pts_all:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty, empty
    return (
        np.concatenate(pts_all, axis=0),
        np.concatenate(nrm_all, axis=0),
        np.concatenate(clr_all, axis=0),
    )


def _sharp_sample_parts_with_color(
    parts: List[trimesh.Trimesh],
    counts: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sharp-edge samples aggregated across parts."""
    pts_all, nrm_all, clr_all = [], [], []
    for part, num in zip(parts, counts):
        if num <= 0:
            continue
        pts, nrm, clr = sharp_sample_pointcloud_with_color(
            part, num=int(num), rng=rng,
        )
        if len(pts) == 0:
            continue
        pts_all.append(pts)
        nrm_all.append(nrm)
        clr_all.append(clr)

    if not pts_all:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty, empty
    return (
        np.concatenate(pts_all, axis=0),
        np.concatenate(nrm_all, axis=0),
        np.concatenate(clr_all, axis=0),
    )


def sample_pointcloud_with_color(
    mesh,
    num=200000,
    vertex_uvs: Optional[np.ndarray] = None,
    texture_rgb: Optional[np.ndarray] = None,
):
    """Sample surface points with normals and RGB colors (UV texture when available).

    Returns:
        points  : float32 tensor (num, 3)
        normals : float32 tensor (num, 3)
        colors  : float32 tensor (num, 3), RGB in [0, 1]
    """
    if vertex_uvs is None or texture_rgb is None:
        vertex_uvs, texture_rgb = extract_mesh_texture(mesh)

    points, face_idx = mesh.sample(num, return_index=True)
    normals = mesh.face_normals[face_idx]
    colors = _colors_at_surface_points(
        mesh, points, face_idx, vertex_uvs, texture_rgb,
    )

    return (
        torch.from_numpy(points.astype(np.float32)),
        torch.from_numpy(normals.astype(np.float32)),
        torch.from_numpy(colors.astype(np.float32)),
    )


def sharp_sample_pointcloud_with_color(
    mesh,
    num=16384,
    rng: Optional[np.random.Generator] = None,
    vertex_uvs: Optional[np.ndarray] = None,
    texture_rgb: Optional[np.ndarray] = None,
):
    """Sample points along sharp edges with interpolated normals and RGB colors."""
    if rng is None:
        rng = np.random.default_rng()
    if vertex_uvs is None or texture_rgb is None:
        vertex_uvs, texture_rgb = extract_mesh_texture(mesh)

    V = mesh.vertices
    N = mesh.face_normals
    VN = mesh.vertex_normals
    F = mesh.faces
    VN2 = np.ones(V.shape[0])
    for i in range(3):
        dot = np.stack((VN2[F[:, i]], np.sum(VN[F[:, i]] * N, axis=-1)), axis=-1)
        VN2[F[:, i]] = np.min(dot, axis=-1)

    sharp_mask = VN2 < 0.985
    edge_a = np.concatenate((F[:, 0], F[:, 1], F[:, 2]))
    edge_b = np.concatenate((F[:, 1], F[:, 2], F[:, 0]))
    sharp_edge = sharp_mask[edge_a] * sharp_mask[edge_b]
    edge_a = edge_a[sharp_edge > 0]
    edge_b = edge_b[sharp_edge > 0]

    if len(edge_a) == 0:
        # No sharp edges: return empty arrays
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty, empty

    sharp_verts_a = V[edge_a]
    sharp_verts_b = V[edge_b]
    sharp_verts_an = VN[edge_a]
    sharp_verts_bn = VN[edge_b]

    weights = np.linalg.norm(sharp_verts_b - sharp_verts_a, axis=-1)
    weights /= np.sum(weights)

    random_number = rng.random(num)
    w = rng.random((num, 1))
    index = np.searchsorted(weights.cumsum(), random_number)
    index = np.clip(index, 0, len(edge_a) - 1)

    samples = w * sharp_verts_a[index] + (1 - w) * sharp_verts_b[index]
    normals = w * sharp_verts_an[index] + (1 - w) * sharp_verts_bn[index]

    if vertex_uvs is not None and texture_rgb is not None:
        uv_a = vertex_uvs[edge_a]
        uv_b = vertex_uvs[edge_b]
        uv_pts = w * uv_a[index] + (1 - w) * uv_b[index]
        colors = _sample_texture_rgb_batch(texture_rgb, uv_pts)
    else:
        vertex_colors = _get_vertex_colors(mesh)
        colors = w * vertex_colors[edge_a][index] + (1 - w) * vertex_colors[edge_b][index]

    return samples, normals, colors


def _append_sharp_label_block(xyz_nrm_rgb: np.ndarray, label_value: float) -> np.ndarray:
    """Insert sharp_label after normals: xyz | normals | label | rgb."""
    xyz = xyz_nrm_rgb[:, :3]
    nrm = xyz_nrm_rgb[:, 3:6]
    rgb = xyz_nrm_rgb[:, 6:9]
    label = np.full((xyz_nrm_rgb.shape[0], 1), label_value, dtype=xyz_nrm_rgb.dtype)
    return np.concatenate([xyz, nrm, label, rgb], axis=1).astype(np.float16)


def load_surface_sharpedge_rgb(
    mesh,
    num_points=4096,
    num_sharp_points=4096,
    seed: Optional[int] = None,
    include_sharp_label: bool = False,
    gobjaverse_meta: Optional[dict] = None,
):
    """Build a colored surface tensor for ShapeGSAE.

    Default layout (9 channels): xyz(0:3) | normals(3:6) | rgb(6:9).

    With ``include_sharp_label=True`` (10 channels, Hunyuan-pretrained path):
    xyz(0:3) | normals(3:6) | sharp_label(6) | rgb(7:10).
    Uniform block rows use label 0; sharp-edge block rows use label 1.

    When ``seed`` is set, all subsampling (including trimesh face sampling) is
    reproducible for a given mesh geometry.
    """
    with _numpy_seed_context(seed) as rng:
        parts = scene_to_parts(mesh)
        if gobjaverse_meta is not None:
            parts = normalize_parts_gobjaverse(parts, gobjaverse_meta)
        else:
            parts = normalize_parts(parts)

        mesh_full = merge_parts_vertex_colored(parts)
        sample_num = 499712 // 2
        min_per_part = max(32, sample_num // max(len(parts) * 32, 1))
        areas = np.array([p.area for p in parts], dtype=np.float64)
        uniform_counts = _allocate_part_counts(areas, sample_num, min_per_part=min_per_part)
        sharp_counts = _allocate_part_counts(areas, sample_num, min_per_part=min_per_part)

        pts, nrm, clr = _sample_parts_with_color(parts, uniform_counts)
        sharp_pts, sharp_nrm, sharp_clr = _sharp_sample_parts_with_color(parts, sharp_counts, rng)

        surface = np.concatenate([pts, nrm, clr], axis=1).astype(np.float16)

        if len(sharp_pts) == 0:
            sharp_pts, sharp_nrm, sharp_clr = _sample_parts_with_color(parts, sharp_counts)

        sharp_surface = np.concatenate([sharp_pts, sharp_nrm, sharp_clr], axis=1).astype(np.float16)

        if include_sharp_label:
            surface = _append_sharp_label_block(surface, 0.0)
            sharp_surface = _append_sharp_label_block(sharp_surface, 1.0)

        if surface.shape[0] < num_points:
            raise ValueError(
                f"Not enough surface samples ({surface.shape[0]}) for num_points={num_points}"
            )
        if sharp_surface.shape[0] < num_sharp_points:
            raise ValueError(
                f"Not enough sharp samples ({sharp_surface.shape[0]}) for "
                f"num_sharp_points={num_sharp_points}"
            )

        ind = rng.choice(surface.shape[0], num_points, replace=False)
        surface = torch.FloatTensor(surface[ind])
        ind = rng.choice(sharp_surface.shape[0], num_sharp_points, replace=False)
        sharp_surface = torch.FloatTensor(sharp_surface[ind])

        return torch.cat([surface, sharp_surface], dim=0).unsqueeze(0), mesh_full


class RGBSharpEdgeSurfaceLoader:
    """Load a textured mesh and return a colored surface tensor.

    Surface layout (default): (1, num_uniform_points + num_sharp_points, 9)
        channels: xyz(0:3) | normals(3:6) | rgb(6:9)

    With ``include_sharp_label=True``: 10 channels
        xyz(0:3) | normals(3:6) | sharp_label(6) | rgb(7:10)
    """

    def __init__(
        self,
        num_uniform_points=8192,
        num_sharp_points=8192,
        *,
        seed: Optional[int] = None,
        deterministic: bool = False,
        include_sharp_label: bool = False,
        **kwargs,
    ):
        self.num_uniform_points = num_uniform_points
        self.num_sharp_points = num_sharp_points
        self.num_points = num_uniform_points + num_sharp_points
        self.seed = seed
        self.deterministic = deterministic
        self.include_sharp_label = include_sharp_label

    def __call__(
        self,
        mesh_or_mesh_path,
        num_uniform_points=None,
        num_sharp_points=None,
        gobjaverse_meta: Optional[dict] = None,
    ):
        if num_uniform_points is None:
            num_uniform_points = self.num_uniform_points
        if num_sharp_points is None:
            num_sharp_points = self.num_sharp_points

        mesh_path: Optional[str] = None
        mesh = mesh_or_mesh_path
        if isinstance(mesh, str):
            mesh_path = mesh
            mesh = trimesh.load(mesh, process=False)

        subsample_seed: Optional[int] = None
        if self.seed is not None and mesh_path is not None:
            # Fixed per-mesh surface subsample (cached in MeshDataset RAM cache).
            subsample_seed = stable_mesh_seed(self.seed, mesh_path)
        elif self.deterministic and self.seed is not None:
            subsample_seed = int(self.seed)

        surface, _ = load_surface_sharpedge_rgb(
            mesh,
            num_points=num_uniform_points,
            num_sharp_points=num_sharp_points,
            seed=subsample_seed,
            include_sharp_label=self.include_sharp_label,
            gobjaverse_meta=gobjaverse_meta,
        )
        return surface
