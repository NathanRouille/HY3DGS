import hashlib
import os
from contextlib import contextmanager
from typing import Iterator, Optional, Union

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


def load_surface_sharpegde(mesh, num_points=4096, num_sharp_points=4096, sharpedge_flag=True):
    try:
        mesh_full = trimesh.util.concatenate(mesh.dump())
    except Exception:
        mesh_full = trimesh.util.concatenate(mesh)
    mesh_full = normalize_mesh(mesh_full)

    origin_num = mesh_full.faces.shape[0]
    original_vertices = mesh_full.vertices
    original_faces = mesh_full.faces

    mesh = trimesh.Trimesh(vertices=original_vertices, faces=original_faces[:origin_num])
    mesh_fill = trimesh.Trimesh(vertices=original_vertices, faces=original_faces[origin_num:])
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

def _get_vertex_colors(mesh):
    """Return float32 RGB vertex colors in [0, 1] for the given trimesh Trimesh.

    Handles vertex-color meshes, UV-textured meshes, and meshes with no
    color (falls back to neutral gray 0.5).
    """
    try:
        color_mesh = mesh.visual.to_color()
        vc = color_mesh.vertex_colors  # (V, 4) RGBA uint8
        return (vc[:, :3] / 255.0).astype(np.float32)
    except Exception:
        return np.full((len(mesh.vertices), 3), 0.5, dtype=np.float32)


def sample_pointcloud_with_color(mesh, num=200000):
    """Sample surface points with normals and barycentric-interpolated RGB colors.

    Returns:
        points  : float32 tensor (num, 3)
        normals : float32 tensor (num, 3)
        colors  : float32 tensor (num, 3), RGB in [0, 1]
    """
    points, face_idx = mesh.sample(num, return_index=True)
    normals = mesh.face_normals[face_idx]

    vertex_colors = _get_vertex_colors(mesh)          # (V, 3) float32
    face_verts = mesh.faces[face_idx]                 # (num, 3) vertex indices

    # Barycentric interpolation of colors
    tri_verts = mesh.vertices[face_verts]             # (num, 3, 3)
    bary = trimesh.triangles.points_to_barycentric(tri_verts, points)  # (num, 3)
    bary = np.clip(bary, 0, 1)
    bary /= bary.sum(axis=1, keepdims=True) + 1e-8   # re-normalize

    face_colors = vertex_colors[face_verts]           # (num, 3, 3)
    colors = (bary[:, :, None] * face_colors).sum(axis=1)  # (num, 3)

    return (
        torch.from_numpy(points.astype(np.float32)),
        torch.from_numpy(normals.astype(np.float32)),
        torch.from_numpy(colors.astype(np.float32)),
    )


def sharp_sample_pointcloud_with_color(
    mesh,
    num=16384,
    rng: Optional[np.random.Generator] = None,
):
    """Sample points along sharp edges with interpolated normals and RGB colors."""
    if rng is None:
        rng = np.random.default_rng()
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

    vertex_colors = _get_vertex_colors(mesh)
    sharp_colors_a = vertex_colors[edge_a]
    sharp_colors_b = vertex_colors[edge_b]

    weights = np.linalg.norm(sharp_verts_b - sharp_verts_a, axis=-1)
    weights /= np.sum(weights)

    random_number = rng.random(num)
    w = rng.random((num, 1))
    index = np.searchsorted(weights.cumsum(), random_number)
    index = np.clip(index, 0, len(edge_a) - 1)

    samples = w * sharp_verts_a[index] + (1 - w) * sharp_verts_b[index]
    normals = w * sharp_verts_an[index] + (1 - w) * sharp_verts_bn[index]
    colors = w * sharp_colors_a[index] + (1 - w) * sharp_colors_b[index]

    return samples, normals, colors


def load_surface_sharpedge_rgb(
    mesh,
    num_points=4096,
    num_sharp_points=4096,
    seed: Optional[int] = None,
):
    """Build an RGB surface tensor of shape (1, num_points+num_sharp_points, 9).

    Channel layout: xyz(0:3) | normals(3:6) | rgb(6:9).

    When ``seed`` is set, all subsampling (including trimesh face sampling) is
    reproducible for a given mesh geometry.
    """
    with _numpy_seed_context(seed) as rng:
        try:
            mesh_full = trimesh.util.concatenate(mesh.dump())
        except Exception:
            mesh_full = trimesh.util.concatenate(mesh)
        mesh_full = normalize_mesh(mesh_full)

        origin_num = mesh_full.faces.shape[0]
        original_vertices = mesh_full.vertices
        original_faces = mesh_full.faces

        mesh_geo = trimesh.Trimesh(vertices=original_vertices, faces=original_faces[:origin_num])
        mesh_fill = trimesh.Trimesh(vertices=original_vertices, faces=original_faces[origin_num:])

        # Copy visual from full mesh so color sampling works on geometry submesh
        try:
            mesh_geo.visual = mesh_full.visual
        except Exception:
            pass

        area = mesh_geo.area
        area_fill = mesh_fill.area
        sample_num = 499712 // 2
        num_fill = int(sample_num * (area_fill / (area + area_fill)))
        num = sample_num - num_fill

        pts, nrm, clr = sample_pointcloud_with_color(mesh_geo, num=num)
        pts = pts.numpy()
        nrm = nrm.numpy()
        clr = clr.numpy()

        if num_fill == 0:
            pts_fill = np.zeros((0, 3), dtype=np.float32)
            nrm_fill = np.zeros((0, 3), dtype=np.float32)
            clr_fill = np.zeros((0, 3), dtype=np.float32)
        else:
            pts_fill, nrm_fill, clr_fill = sample_pointcloud_with_color(mesh_fill, num=num_fill)
            pts_fill = pts_fill.numpy()
            nrm_fill = nrm_fill.numpy()
            clr_fill = clr_fill.numpy()

        sharp_pts, sharp_nrm, sharp_clr = sharp_sample_pointcloud_with_color(
            mesh_geo, num=sample_num, rng=rng,
        )

        # Build surface block (random + fill) and sharp block
        surface = np.concatenate(
            [np.concatenate([pts, nrm, clr], axis=1),
             np.concatenate([pts_fill, nrm_fill, clr_fill], axis=1)],
            axis=0
        ).astype(np.float16)

        if len(sharp_pts) == 0:
            # Fall back to uniform samples when no sharp edges detected
            sharp_pts, sharp_nrm, sharp_clr = sample_pointcloud_with_color(mesh_geo, num=sample_num)
            sharp_pts = sharp_pts.numpy()
            sharp_nrm = sharp_nrm.numpy()
            sharp_clr = sharp_clr.numpy()

        sharp_surface = np.concatenate([sharp_pts, sharp_nrm, sharp_clr], axis=1).astype(np.float16)

        ind = rng.choice(surface.shape[0], num_points, replace=False)
        surface = torch.FloatTensor(surface[ind])
        ind = rng.choice(sharp_surface.shape[0], num_sharp_points, replace=False)
        sharp_surface = torch.FloatTensor(sharp_surface[ind])

        return torch.cat([surface, sharp_surface], dim=0).unsqueeze(0), mesh_full


class RGBSharpEdgeSurfaceLoader:
    """Load a textured mesh and return a colored surface tensor.

    Surface layout: (1, num_uniform_points + num_sharp_points, 9)
        channels: xyz(0:3) | normals(3:6) | rgb(6:9)
    """

    def __init__(
        self,
        num_uniform_points=8192,
        num_sharp_points=8192,
        *,
        seed: Optional[int] = None,
        deterministic: bool = False,
        **kwargs,
    ):
        self.num_uniform_points = num_uniform_points
        self.num_sharp_points = num_sharp_points
        self.num_points = num_uniform_points + num_sharp_points
        self.seed = seed
        self.deterministic = deterministic

    def __call__(self, mesh_or_mesh_path, num_uniform_points=None, num_sharp_points=None):
        if num_uniform_points is None:
            num_uniform_points = self.num_uniform_points
        if num_sharp_points is None:
            num_sharp_points = self.num_sharp_points

        mesh_path: Optional[str] = None
        mesh = mesh_or_mesh_path
        if isinstance(mesh, str):
            mesh_path = mesh
            mesh = trimesh.load(mesh, process=False)
        if isinstance(mesh, trimesh.scene.Scene):
            mesh = mesh.dump(concatenate=True)

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
        )
        return surface
