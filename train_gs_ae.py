"""Training script for ShapeGSAE: Point Cloud → 3D Gaussian Splatting.

Usage:
    python train_gs_ae.py --data_dir /path/to/meshes --output_dir runs/gs_ae

Mesh directory should contain GLB, OBJ, or PLY files (one object per file).

GT RGBD supervision is rendered offline the first time a mesh is encountered
and cached to disk (as .pt tensors) next to the mesh files.

Dependencies (beyond base requirements.txt):
    pip install gsplat pytorch-msssim pyrender
    pip install trimesh[easy]   # for texture support
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import trimesh
try:
    import wandb
except ImportError:
    wandb = None

from hy3dgen.shapegen.models.autoencoders.model import ShapeGSAE
from hy3dgen.shapegen.surface_loaders import RGBSharpEdgeSurfaceLoader, normalize_mesh
from hy3dgen.shapegen.gs_renderer import (
    GaussianRenderer,
    RGBDLoss,
    build_orbit_cameras,
    orbit_c2w,
    _ssim,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ShapeNet categories
# ---------------------------------------------------------------------------

# Human-readable name → ShapeNet Core v2 synset ID. Folders in the prepared
# dataset are named "<synset_id>_<model_hash>"; the prefix before the first
# underscore is matched against these IDs to filter by category.
CATEGORY_NAME_TO_ID: Dict[str, str] = {
    "airplane": "02691156",
    "bench": "02828884",
    "cabinet": "02933112",
    "car": "02958343",
    "chair": "03001627",
    "lamp": "03636649",
    "sofa": "04256520",
    "table": "04379243",
    "watercraft": "04530566",
}


def resolve_category_ids(categories_str: Optional[str]) -> Optional[Set[str]]:
    """Parse a comma-separated list of category names and/or 8-digit IDs.

    Returns ``None`` when no value is supplied (=> no filtering), otherwise a
    set of ShapeNet synset IDs ready to match against folder name prefixes.
    Raises ``ValueError`` for unknown tokens so typos fail fast instead of
    silently dropping every sample.
    """
    if not categories_str:
        return None
    ids: Set[str] = set()
    for tok in categories_str.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if tok in CATEGORY_NAME_TO_ID:
            ids.add(CATEGORY_NAME_TO_ID[tok])
        elif tok.isdigit() and len(tok) == 8:
            ids.add(tok)
        else:
            raise ValueError(
                f"Unknown category '{tok}'. Expected a known name "
                f"({', '.join(sorted(CATEGORY_NAME_TO_ID))}) or an "
                f"8-digit ShapeNet synset ID."
            )
    return ids


# The 4-azimuth set used by the prepared ShapeNet GT cache. When training with
# a subset of these views (e.g. 1 or 2 views), GTRGBDRenderer reuses the
# 4-view cache and slices it instead of re-rendering.
SUPERSET_AZIMUTHS_DEG: List[float] = [0.0, 90.0, 180.0, 270.0]


def mesh_path_has_usable_gt_cache(
    mesh_path: str,
    gt_tag: str,
    height: int,
    width: int,
    azimuths_deg: List[float],
) -> bool:
    """True if ``GTRGBDRenderer.get_or_render`` can satisfy GT from disk only.

    Matches the resolution order in ``get_or_render``: exact tag file, else
    4-view ``normv2`` superset slice (same H×W, azimuths ⊆ {0,90,180,270}).
    Used by ``visualize_gs_ae`` / ``evaluate_gs_ae`` when filtering with
    ``--only_cached_gt`` so they stay aligned with training cache layout.
    """
    if os.path.exists(f"{mesh_path}.gt_rgbd_{gt_tag}.pt"):
        return True
    if not all(a in SUPERSET_AZIMUTHS_DEG for a in azimuths_deg):
        return False
    super_az_str = '_'.join(str(int(round(a))) for a in SUPERSET_AZIMUTHS_DEG)
    super_tag = f"h{height}w{width}az{super_az_str}_normv2"
    if super_tag == gt_tag:
        return False
    return os.path.exists(f"{mesh_path}.gt_rgbd_{super_tag}.pt")


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """PSNR (dB) on foreground pixels only. All tensors float in [0, 1]."""
    pred_fg = pred[mask.expand_as(pred)]
    gt_fg = gt[mask.expand_as(gt)]
    if len(pred_fg) == 0:
        return 0.0
    mse = F.mse_loss(pred_fg, gt_fg)
    if mse.item() < 1e-10:
        return 100.0
    return float(-10.0 * torch.log10(mse).item())


def compute_ssim_fg(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """SSIM on foreground-masked image (0–1 scale, higher is better).

    pred / gt : (H, W, 3) float [0, 1]
    mask      : (H, W, 1) bool
    """
    # Replace background in pred with GT so SSIM window only measures fg quality
    mask_f = mask.float()
    pred_m = pred * mask_f + gt * (1.0 - mask_f)
    pred_nchw = pred_m.unsqueeze(0).permute(0, 3, 1, 2)
    gt_nchw = gt.unsqueeze(0).permute(0, 3, 1, 2)
    return float(_ssim(pred_nchw, gt_nchw).item())


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    renderer: GaussianRenderer,
    val_dataset,
    device: torch.device,
    num_samples: int = 20,
) -> Dict[str, float]:
    """Evaluate model on a subset of val_dataset.

    Returns mean PSNR (fg) and mean SSIM (fg) across all samples and views.
    """
    model.eval()
    psnr_list, ssim_list, alpha_list = [], [], []

    indices = list(range(len(val_dataset)))[:num_samples]
    for idx in indices:
        try:
            sample = val_dataset[idx]
        except Exception as e:
            logger.warning(f"Val sample {idx} failed: {e}")
            continue

        surface = sample['surface'].unsqueeze(0).to(device)
        gt_rgbs = sample['rgbs']
        gt_depths = sample['depths']
        c2ws = sample['c2ws']

        means, scales, rotations, opacities, colors = model(surface)
        means = means[0]
        scales = scales[0]
        rotations = rotations[0]
        opacities = opacities[0]
        colors = colors[0]

        for gt_rgb, gt_depth, c2w in zip(gt_rgbs, gt_depths, c2ws):
            valid_mask = (gt_depth > 0)                       # (H, W, 1)
            if float(valid_mask.float().mean().item()) < 0.02:
                continue
            out = renderer(means, scales, rotations, opacities, colors, c2w.to(device))
            pred_rgb = out['rgb'].cpu()
            pred_alpha = out['alpha'].cpu()

            psnr_list.append(compute_psnr(pred_rgb, gt_rgb, valid_mask))
            ssim_list.append(compute_ssim_fg(pred_rgb, gt_rgb, valid_mask))
            alpha_list.append(float(pred_alpha[valid_mask].mean().item()))

    model.train()
    if not psnr_list:
        return {'val/psnr': 0.0, 'val/ssim': 0.0, 'val/alpha_fg_mean': 0.0}
    return {
        'val/psnr': float(sum(psnr_list) / len(psnr_list)),
        'val/ssim': float(sum(ssim_list) / len(ssim_list)),
        'val/alpha_fg_mean': float(sum(alpha_list) / len(alpha_list)),
    }


# ---------------------------------------------------------------------------
# GT RGBD rendering (offline, CPU, cached to disk)
# ---------------------------------------------------------------------------

class GTRGBDRenderer:
    """Render ground-truth RGBD from a textured trimesh using pyrender.

    Falls back to a simple vertex-color renderer if pyrender is unavailable.
    Results are cached to disk (``<mesh_path>.gt_rgbd_<tag>.pt``) and reloaded
    on subsequent calls so each mesh is only rendered once.
    """

    def __init__(
        self,
        height: int = 256,
        width: int = 256,
        fov_deg: float = 49.13,
        camera_distance: float = 2.5,
        elevation_deg: float = 20.0,
        num_views: int = 8,
        device: str = 'cpu',
        azimuths_deg: Optional[List[float]] = None,
    ):
        # Normalize azimuths to an explicit list so the cache tag is always
        # in the deterministic "az<a0>_<a1>_..." form. This also lets the
        # superset-cache fallback in get_or_render know exactly which view
        # indices to slice.
        if azimuths_deg is None:
            azimuths_deg = [360.0 * i / num_views for i in range(num_views)]

        self.height = height
        self.width = width
        self.fov_deg = fov_deg
        self.camera_distance = camera_distance
        self.elevation_deg = elevation_deg
        self.num_views = num_views
        self.device = device
        self.azimuths_deg = list(azimuths_deg)

        # Cache tag encodes the rendering configuration so different camera setups
        # never share the same on-disk cache. "0,90" and "0,180" always produce
        # separate cache files (critical for multi-view correctness).
        # ``normv2`` invalidates all caches produced before the orbit_c2w
        # handedness fix — those files have empty depth for half the azimuths.
        az_str = '_'.join(str(int(round(a))) for a in self.azimuths_deg)
        self._tag = f"h{height}w{width}az{az_str}_normv2"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_render(self, mesh_path: str, mesh: Optional[trimesh.Trimesh] = None):
        """Return cached GT RGBD or render it now.

        Resolution order:
          1. Exact-match cache (``<mesh_path>.gt_rgbd_<self._tag>.pt``).
          2. Superset slice: if the requested azimuths are all contained in
             ``SUPERSET_AZIMUTHS_DEG`` (the 4-view set 0/90/180/270) and the
             corresponding 4-view cache exists, slice it. This lets Phase 1
             (1 view) and Phase 2a (2 views) experiments reuse the single
             4-view cache without re-rendering.
          3. Render from scratch and save under the exact tag.

        Returns:
            rgbs   : list of (H, W, 3) float tensors, values in [0, 1]
            depths : list of (H, W, 1) float tensors, values ≥ 0
            c2ws   : list of (4, 4) float tensors (camera-to-world)
        """
        cache_path = f"{mesh_path}.gt_rgbd_{self._tag}.pt"
        cached = self._try_load_cache(cache_path)
        if cached is not None:
            return cached

        sliced = self._try_load_superset_slice(mesh_path)
        if sliced is not None:
            return sliced

        if mesh is None:
            mesh = self._load_mesh(mesh_path)
        # Match point-cloud loader coordinates: render GT in normalized mesh space.
        mesh = normalize_mesh(mesh)

        c2ws = build_orbit_cameras(
            num_views=self.num_views,
            elevation_deg=self.elevation_deg,
            radius=self.camera_distance,
            device='cpu',
            azimuths_deg=self.azimuths_deg,
        )

        rgbs, depths = self._render_views(mesh, c2ws)
        data = {'rgbs': rgbs, 'depths': depths, 'c2ws': c2ws}
        torch.save(data, cache_path)
        return rgbs, depths, c2ws

    @staticmethod
    def _try_load_cache(cache_path: str):
        """Load a cache file or return None if missing/empty.

        An "empty" cache (all-zero depths) is treated as missing so we
        re-render rather than training against blank GT.
        """
        if not os.path.exists(cache_path):
            return None
        data = torch.load(cache_path, map_location='cpu')
        cached_depths = data.get('depths', [])
        mean_cached_valid_depth_ratio = float(
            sum(float((d > 0).float().mean().item()) for d in cached_depths)
            / max(len(cached_depths), 1)
        )
        if mean_cached_valid_depth_ratio > 0.0:
            return data['rgbs'], data['depths'], data['c2ws']
        logger.warning(f"Empty GT cache at {cache_path}; ignoring.")
        return None

    def _try_load_superset_slice(self, mesh_path: str):
        """If our azimuths are a subset of the 4-view cache, load and slice it."""
        if not all(a in SUPERSET_AZIMUTHS_DEG for a in self.azimuths_deg):
            return None

        super_az_str = '_'.join(str(int(round(a))) for a in SUPERSET_AZIMUTHS_DEG)
        super_tag = f"h{self.height}w{self.width}az{super_az_str}_normv2"
        if super_tag == self._tag:
            return None  # exact-match path already handled above

        super_cache_path = f"{mesh_path}.gt_rgbd_{super_tag}.pt"
        cached = self._try_load_cache(super_cache_path)
        if cached is None:
            return None

        rgbs, depths, c2ws = cached
        idx = [SUPERSET_AZIMUTHS_DEG.index(a) for a in self.azimuths_deg]
        return (
            [rgbs[i] for i in idx],
            [depths[i] for i in idx],
            [c2ws[i] for i in idx],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mesh(path: str) -> trimesh.Trimesh:
        scene_or_mesh = trimesh.load(path, process=False)
        if isinstance(scene_or_mesh, trimesh.scene.Scene):
            # `Scene.dump(concatenate=True)` is deprecated; use `to_geometry()`.
            geom = scene_or_mesh.to_geometry()
            if isinstance(geom, trimesh.Trimesh):
                return geom
            if isinstance(geom, dict):
                meshes = [g for g in geom.values() if isinstance(g, trimesh.Trimesh)]
            elif isinstance(geom, (list, tuple)):
                meshes = [g for g in geom if isinstance(g, trimesh.Trimesh)]
            else:
                meshes = []
            if len(meshes) == 0:
                raise ValueError(f"No mesh geometry found in scene: {path}")
            if len(meshes) == 1:
                return meshes[0]
            return trimesh.util.concatenate(meshes)
        return scene_or_mesh

    def _render_views(self, mesh, c2ws):
        """Dispatch to pyrender or vertex-color fallback."""
        try:
            rgbs, depths = self._render_pyrender(mesh, c2ws)
            mean_valid_depth_ratio = float(
                sum(float((d > 0).float().mean().item()) for d in depths) / max(len(depths), 1)
            )
            if mean_valid_depth_ratio > 0.0:
                return rgbs, depths
            logger.warning("pyrender produced empty depth; using vertex-color fallback.")
            return self._render_vertex_color(mesh, c2ws)
        except Exception as e:
            logger.warning(f"pyrender failed ({e}); using vertex-color fallback renderer.")
            return self._render_vertex_color(mesh, c2ws)

    @staticmethod
    def _mesh_for_pyrender(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Drop GLTF/GLB texture images that pyrender cannot upload (e.g. 2-channel).

        pyrender then rasterizes from simple vertex colors (sampled via trimesh).
        """
        m = mesh.copy()
        try:
            c = m.visual.to_color()
            vc = np.asanyarray(c.vertex_colors)
            if vc.shape[0] != len(m.vertices):
                raise ValueError("vertex color count mismatch")
            if vc.shape[1] < 4:
                alpha = np.full((vc.shape[0], 1), 255, dtype=vc.dtype)
                vc = np.concatenate([vc[:, :3], alpha], axis=1)
            m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=vc)
        except Exception:
            gray = np.full((len(m.vertices), 4), 200, dtype=np.uint8)
            gray[:, 3] = 255
            m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=gray)
        return m

    def _render_pyrender(self, mesh, c2ws):
        """GPU-less offscreen RGBD rendering via pyrender + EGL."""
        # Avoid repetitive optional-acceleration info spam in console output.
        logging.getLogger("OpenGL.acceleratesupport").setLevel(logging.ERROR)
        import pyrender  # noqa: import inside method so it's optional

        # Build pyrender mesh from trimesh
        mesh_gl = self._mesh_for_pyrender(mesh)
        pr_mesh = pyrender.Mesh.from_trimesh(mesh_gl, smooth=False)
        # Enable double-sided rendering so back faces are visible.
        # GLB/GLTF meshes commonly use single-sided materials; without this,
        # cameras at azimuth=180° (back view) see an empty scene.
        for primitive in pr_mesh.primitives:
            primitive.material.doubleSided = True
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.5, 0.5, 0.5])
        scene.add(pr_mesh)

        fy = self.height / (2 * math.tan(math.radians(self.fov_deg) / 2))
        camera = pyrender.IntrinsicsCamera(
            fx=fy, fy=fy,
            cx=self.width / 2, cy=self.height / 2,
            znear=0.01, zfar=100.0,
        )

        # EGL/offscreen flags
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
        renderer = pyrender.OffscreenRenderer(self.width, self.height)

        rgbs, depths = [], []
        for c2w in c2ws:
            # pyrender camera node expects OpenGL c2w (Y-up, Z-backward)
            cam_pose = c2w.numpy().astype(np.float64)
            cam_node = scene.add(camera, pose=cam_pose)

            color, depth = renderer.render(scene)
            scene.remove_node(cam_node)
            rgb = torch.from_numpy(color.astype(np.float32) / 255.0)   # (H, W, 3)
            dep = torch.from_numpy(depth.astype(np.float32)).unsqueeze(-1)  # (H, W, 1)
            rgbs.append(rgb)
            depths.append(dep)

        renderer.delete()
        return rgbs, depths

    def _render_vertex_color(self, mesh, c2ws):
        """CPU-only fallback: rasterize per-vertex colors using trimesh ray-casting.

        This is a software renderer and is slow; prefer pyrender in production.
        """
        from trimesh.ray.ray_triangle import RayMeshIntersector

        H, W = self.height, self.width
        fy = H / (2 * math.tan(math.radians(self.fov_deg) / 2))

        # Vertex colors
        try:
            vc = mesh.visual.to_color().vertex_colors[:, :3].astype(np.float32) / 255.0
        except Exception:
            vc = np.full((len(mesh.vertices), 3), 0.7, dtype=np.float32)

        intersector = RayMeshIntersector(mesh)

        # Pixel grid (row = y, col = x)
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        dirs_cam = np.stack(
            [(xs - W / 2) / fy, -(ys - H / 2) / fy, -np.ones_like(xs)],
            axis=-1,
        ).reshape(-1, 3)

        rgbs, depths = [], []
        for c2w in c2ws:
            R = c2w[:3, :3].numpy().astype(np.float64)
            t = c2w[:3, 3].numpy().astype(np.float64)

            dirs_world = (R @ dirs_cam.T).T
            dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True)
            origins = np.broadcast_to(t[None], dirs_world.shape).copy()

            locs, ray_idx, tri_idx = intersector.intersects_location(
                origins, dirs_world, multiple_hits=False
            )

            rgb_img = np.ones((H * W, 3), dtype=np.float32)
            dep_img = np.zeros((H * W, 1), dtype=np.float32)

            if len(locs) > 0:
                # Depth = distance along z axis (camera space)
                cam_pts = ((locs - t) @ R)
                dep_vals = -cam_pts[:, 2]   # camera z is -forward

                # Barycentric color interpolation
                tri_verts = mesh.faces[tri_idx]
                v0 = mesh.vertices[tri_verts[:, 0]]
                v1 = mesh.vertices[tri_verts[:, 1]]
                v2 = mesh.vertices[tri_verts[:, 2]]
                bary = trimesh.triangles.points_to_barycentric(
                    np.stack([v0, v1, v2], axis=1), locs
                )
                bary = np.clip(bary, 0, 1)
                bary /= bary.sum(axis=1, keepdims=True) + 1e-8
                c0 = vc[tri_verts[:, 0]]
                c1 = vc[tri_verts[:, 1]]
                c2 = vc[tri_verts[:, 2]]
                colors = (bary[:, 0:1] * c0 + bary[:, 1:2] * c1 + bary[:, 2:3] * c2)

                rgb_img[ray_idx] = colors
                dep_img[ray_idx, 0] = dep_vals.astype(np.float32)

            rgbs.append(torch.from_numpy(rgb_img.reshape(H, W, 3)))
            depths.append(torch.from_numpy(dep_img.reshape(H, W, 1)))

        return rgbs, depths


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

MESH_EXTENSIONS = {'.glb', '.gltf', '.obj', '.ply', '.stl', '.off'}


class MeshDataset(Dataset):
    """Loads textured mesh files and returns colored surface point clouds
    together with pre-rendered GT RGBD views.

    Each item is a dict with:
        surface  : (pc_size+pc_sharpedge_size, 9) float32 tensor
        rgbs     : list of (H, W, 3) float32 tensors
        depths   : list of (H, W, 1) float32 tensors
        c2ws     : list of (4, 4) float32 tensors
        mesh_path: str
    """

    def __init__(
        self,
        data_dir: str,
        pc_size: int = 5120,
        pc_sharpedge_size: int = 5120,
        render_height: int = 256,
        render_width: int = 256,
        num_views: int = 8,
        camera_distance: float = 2.5,
        elevation_deg: float = 20.0,
        max_items: Optional[int] = None,
        mesh_blacklist: Optional[str] = None,
        azimuths_deg: Optional[List[float]] = None,
        categories: Optional[Iterable[str]] = None,
    ):
        self.loader = RGBSharpEdgeSurfaceLoader(
            num_uniform_points=pc_size,
            num_sharp_points=pc_sharpedge_size,
        )
        self.gt_renderer = GTRGBDRenderer(
            height=render_height,
            width=render_width,
            num_views=num_views,
            camera_distance=camera_distance,
            elevation_deg=elevation_deg,
            azimuths_deg=azimuths_deg,
        )

        # Load blacklist of known-bad meshes (one path per line; tab-separated label ignored)
        blacklist: Set[str] = set()
        if mesh_blacklist and os.path.exists(mesh_blacklist):
            with open(mesh_blacklist) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        path_part = line.split('\t')[0].strip()  # ignore optional tab label
                        if path_part:
                            blacklist.add(os.path.realpath(path_part))
            logger.info(f"Blacklist: {len(blacklist)} mesh(es) loaded from {mesh_blacklist}")

        allowed_cats: Optional[Set[str]] = set(categories) if categories else None

        data_path = Path(data_dir)
        self.mesh_paths: List[str] = []
        skipped_category = 0
        skipped_blacklist = 0
        for folder in sorted(data_path.iterdir()):
            if not folder.is_dir():
                continue
            if allowed_cats is not None:
                # Prepared layout: "<synset_id>_<model_hash>". The prefix before
                # the first underscore is the category synset ID.
                cat_id = folder.name.split('_', 1)[0]
                if cat_id not in allowed_cats:
                    skipped_category += 1
                    continue
            potential_obj = folder / "model_normalized.obj"
            if not potential_obj.exists():
                continue
            if blacklist and os.path.realpath(str(potential_obj)) in blacklist:
                skipped_blacklist += 1
                continue
            self.mesh_paths.append(str(potential_obj))

        if max_items is not None:
            self.mesh_paths = self.mesh_paths[:max_items]

        if allowed_cats is not None:
            logger.info(
                f"Category filter cats={sorted(allowed_cats)}: kept "
                f"{len(self.mesh_paths)} mesh(es), skipped {skipped_category} folder(s)"
            )
        if skipped_blacklist:
            logger.info(f"Blacklist filter: {skipped_blacklist} mesh(es) excluded")
        logger.info(f"Dataset: {len(self.mesh_paths)} meshes in {data_dir}")

    def __len__(self) -> int:
        return len(self.mesh_paths)

    def __getitem__(self, idx: int) -> Dict:
        # Iterate forward (non-recursively) to find a working sample
        for attempt in range(len(self.mesh_paths)):
            path = self.mesh_paths[(idx + attempt) % len(self.mesh_paths)]
            try:
                surface = self.loader(path)                         # (1, N, 9)
                rgbs, depths, c2ws = self.gt_renderer.get_or_render(path)
                return {
                    'surface': surface.squeeze(0),   # (N, 9) — DataLoader adds batch dim
                    'rgbs': rgbs,
                    'depths': depths,
                    'c2ws': c2ws,
                    'mesh_path': path,
                }
            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")
        raise RuntimeError(f"All {len(self.mesh_paths)} meshes failed to load")


def collate_fn(batch):
    """Custom collate: stack surfaces, keep views as list-of-lists."""
    surfaces = torch.stack([item['surface'] for item in batch], dim=0)
    # Transpose: batch-of-view-lists → list-of-batched-tensors
    num_views = len(batch[0]['rgbs'])
    rgbs = [
        torch.stack([item['rgbs'][v] for item in batch], dim=0)
        for v in range(num_views)
    ]
    depths = [
        torch.stack([item['depths'][v] for item in batch], dim=0)
        for v in range(num_views)
    ]
    c2ws = [batch[0]['c2ws'][v] for v in range(num_views)]  # cameras are shared
    return {
        'surface': surfaces,
        'rgbs': rgbs,
        'depths': depths,
        'c2ws': c2ws,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # ---- Parse camera azimuths ----
    azimuths_deg: Optional[List[float]] = None
    if args.camera_azimuths:
        azimuths_deg = [float(a.strip()) for a in args.camera_azimuths.split(',')]
        if len(azimuths_deg) != args.num_views:
            raise ValueError(
                f"--camera_azimuths has {len(azimuths_deg)} values "
                f"but --num_views={args.num_views}. They must match."
            )
        logger.info(f"Camera azimuths: {azimuths_deg}")

    # ---- Resolve category filter ----
    categories = resolve_category_ids(args.categories)
    if categories is not None:
        logger.info(f"Filtering to categories: {sorted(categories)}")

    # ---- Pre-cache-only mode: render GT for all meshes then exit ----
    if args.precache_only:
        logger.info("Pre-cache mode: rendering GT RGBD for all meshes (no training).")
        logger.info("Running single-threaded to avoid EGL conflicts.")
        dataset = MeshDataset(
            data_dir=args.data_dir,
            pc_size=args.pc_size,
            pc_sharpedge_size=args.pc_sharpedge_size,
            render_height=args.render_height,
            render_width=args.render_width,
            num_views=args.num_views,
            camera_distance=args.camera_distance,
            elevation_deg=args.elevation_deg,
            max_items=args.max_items,
            mesh_blacklist=args.mesh_blacklist,
            azimuths_deg=azimuths_deg,
            categories=categories,
        )
        n = len(dataset)
        logger.info(f"Pre-caching GT for {n} meshes with tag '{dataset.gt_renderer._tag}' ...")
        ok, failed = 0, []
        for i in range(n):
            path = dataset.mesh_paths[i]
            try:
                dataset.gt_renderer.get_or_render(path)
                ok += 1
                if (i + 1) % 10 == 0 or (i + 1) == n:
                    logger.info(f"  [{i+1}/{n}] done so far: {ok} ok, {len(failed)} failed")
            except Exception as e:
                failed.append(path)
                logger.warning(f"  [{i+1}/{n}] FAILED {path}: {e}")
        logger.info(f"Pre-cache complete. {ok}/{n} succeeded, {len(failed)} failed.")
        if failed:
            logger.warning("Failed meshes:\n" + "\n".join(f"  {p}" for p in failed))
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    max_grad_norm = 1.0
    if args.smoke_overfit:
        # Minimal learning sanity-check: one view + pure RGB L1 only.
        args.num_views = 1
        args.lambda_ssim = 0.0
        args.lambda_d = 0.0
        args.lambda_alpha = 0.0
        args.lambda_scale = 0.0
        args.lambda_opa = 0.0
        # Keep smoke test numerically stable to diagnose learnability.
        args.lr = min(args.lr, 3e-5)
        args.weight_decay = 0.0
        max_grad_norm = 0.1
        logger.info(
            "Smoke overfit mode enabled: num_views=1, lambda_ssim=0, lambda_d=0, "
            "lambda_alpha=0, lambda_scale=0, lambda_opa=0, "
            f"lr={args.lr:.1e}, weight_decay={args.weight_decay:.1e}, "
            f"max_grad_norm={max_grad_norm:.2f}"
        )

    # ---- Model ----
    model = ShapeGSAE(
        num_latents=args.num_latents,
        embed_dim=args.embed_dim,
        width=args.width,
        heads=args.heads,
        num_decoder_layers=args.num_decoder_layers,
        num_encoder_layers=args.num_encoder_layers,
        pc_size=args.pc_size,
        pc_sharpedge_size=args.pc_sharpedge_size,
        point_feats=6,              # normals(3) + rgb(3)
        downsample_ratio=args.downsample_ratio,
    ).to(device)

    if args.shapevae_ckpt:
        logger.info(f"Warm-starting encoder from {args.shapevae_ckpt}")
        model.load_shapevae_encoder(args.shapevae_ckpt)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"ShapeGSAE: {total_params / 1e6:.1f}M trainable parameters")

    # ---- Renderer & Loss ----
    renderer = GaussianRenderer(
        height=args.render_height,
        width=args.render_width,
        render_depth=True,
    ).to(device)

    criterion = RGBDLoss(
        lambda_ssim=args.lambda_ssim,
        lambda_d=args.lambda_d,
        lambda_alpha=args.lambda_alpha,
        lambda_scale=args.lambda_scale,
        lambda_opa=args.lambda_opa,
        target_log_scale=args.target_log_scale,
        target_opacity=args.target_opacity,
        fg_weight=args.fg_weight,
        min_valid_ratio=args.min_valid_depth_ratio,
    )

    # ---- Data ----
    dataset = MeshDataset(
        data_dir=args.data_dir,
        pc_size=args.pc_size,
        pc_sharpedge_size=args.pc_sharpedge_size,
        render_height=args.render_height,
        render_width=args.render_width,
        num_views=args.num_views,
        camera_distance=args.camera_distance,
        elevation_deg=args.elevation_deg,
        max_items=args.max_items,
        mesh_blacklist=args.mesh_blacklist,
        azimuths_deg=azimuths_deg,
        categories=categories,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )

    # ---- Optional validation dataset ----
    val_dataset = None
    if args.val_dir:
        val_dataset = MeshDataset(
            data_dir=args.val_dir,
            pc_size=args.pc_size,
            pc_sharpedge_size=args.pc_sharpedge_size,
            render_height=args.render_height,
            render_width=args.render_width,
            num_views=args.num_views,
            camera_distance=args.camera_distance,
            elevation_deg=args.elevation_deg,
            azimuths_deg=azimuths_deg,
            categories=categories,
        )
        logger.info(f"Val set: {len(val_dataset)} meshes in {args.val_dir}")

    # ---- Optimiser & LR schedule ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    total_steps = args.max_steps
    warmup_steps = args.warmup_steps if not args.smoke_overfit else 0
    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=args.lr * 0.01
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        logger.info(f"LR schedule: {warmup_steps}-step linear warmup → cosine decay to {args.lr * 0.01:.1e}")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.lr * 0.01
        )

    # ---- Resume ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        global_step = ckpt['step']
        logger.info(f"Resumed from step {global_step}")

    # ---- Optional Weights & Biases ----
    use_wandb = bool(args.use_wandb)
    if use_wandb and wandb is None:
        raise ImportError("wandb is not installed. Install it with: pip install wandb")
    if use_wandb:
        run_name = args.wandb_run_name or output_dir.name
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            dir=str(output_dir),
            resume='allow',
        )
        logger.info(f"wandb run: {wandb.run.url if wandb.run else 'unknown'}")

    # ---- Training loop ----
    model.train()
    epoch = 0
    t0 = time.time()
    t_data_start = time.time()

    while global_step < total_steps:
        epoch += 1
        for batch in loader:
            if global_step >= total_steps:
                break

            t_data_end = time.time()
            t_fwd_start = time.time()

            surface = batch['surface'].to(device, non_blocking=True)   # (B, N, 9)
            gt_rgbs = batch['rgbs']     # list of (B, H, W, 3)
            gt_depths = batch['depths'] # list of (B, H, W, 1)
            c2ws = batch['c2ws']        # list of (4,4)

            # Forward pass
            means, scales, rotations, opacities, colors = model(surface)

            # Accumulate rendering loss over all views
            total_loss = torch.zeros(1, device=device)
            log_components: Dict[str, float] = {}

            # Per-view stats for aggregated logging
            view_valid_ratios: List[float] = []
            view_pred_alpha_means: List[float] = []
            view_pred_alpha_maxes: List[float] = []
            view_pred_depth_pos_ratios: List[float] = []

            B = surface.shape[0]
            num_views = len(c2ws)
            num_valid_views = 0  # views with enough foreground GT to be useful
            for view_idx, (gt_rgb_b, gt_depth_b, c2w) in enumerate(
                zip(gt_rgbs, gt_depths, c2ws)
            ):
                gt_rgb_b = gt_rgb_b.to(device)       # (B, H, W, 3)
                gt_depth_b = gt_depth_b.to(device)   # (B, H, W, 1)
                c2w = c2w.to(device)

                gt_valid_ratio = float((gt_depth_b > 0).float().mean().item())
                view_valid_ratios.append(gt_valid_ratio)

                # Skip views with no foreground GT: including them in the loss
                # creates a gradient toward alpha=0 (predicting white bg = 0 loss).
                if gt_valid_ratio < args.min_valid_depth_ratio:
                    view_pred_alpha_means.append(-1.0)
                    view_pred_alpha_maxes.append(-1.0)
                    view_pred_depth_pos_ratios.append(-1.0)
                    continue

                num_valid_views += 1

                # Render each item in the batch separately (gsplat is per-scene)
                pred_rgbs_list, pred_depths_list, pred_alphas_list = [], [], []
                for b in range(B):
                    out = renderer(
                        means[b], scales[b], rotations[b],
                        opacities[b], colors[b], c2w,
                    )
                    pred_rgbs_list.append(out['rgb'])
                    pred_depths_list.append(out['depth'])
                    pred_alphas_list.append(out['alpha'])

                pred_rgb = torch.stack(pred_rgbs_list, dim=0)     # (B, H, W, 3)
                pred_depth = torch.stack(pred_depths_list, dim=0) # (B, H, W, 1)
                pred_alpha = torch.stack(pred_alphas_list, dim=0) # (B, H, W, 1)

                # Collect per-view rendering diagnostics
                view_pred_alpha_means.append(float(pred_alpha.detach().mean().item()))
                view_pred_alpha_maxes.append(float(pred_alpha.detach().max().item()))
                view_pred_depth_pos_ratios.append(float((pred_depth.detach() > 0).float().mean().item()))

                valid_mask = gt_depth_b > 0

                view_loss, comps = criterion(
                    pred_rgb, gt_rgb_b,
                    pred_depth, gt_depth_b,
                    pred_alpha=pred_alpha,
                    valid_mask=valid_mask,
                    scales=scales.view(-1, 3).log(),  # log(scale) for regulariser
                    opacities=opacities.view(-1, 1),
                )
                # Accumulate; divide by valid views below so scale stays consistent
                total_loss = total_loss + view_loss

                for k, v in comps.items():
                    log_components[k] = log_components.get(k, 0.0) + float(v.detach().item())

            # Normalize by number of valid views (avoids collapsed solution when
            # many views are skipped — loss scale stays constant regardless)
            if num_valid_views > 0:
                total_loss = total_loss / num_valid_views
                log_components = {k: v / num_valid_views for k, v in log_components.items()}
            log_components['valid_views_fraction'] = num_valid_views / max(num_views, 1)

            t_bwd_start = time.time()
            optimizer.zero_grad()
            total_loss.backward()
            grad_norm_preclip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            t_step_end = time.time()

            global_step += 1

            # Logging
            lr = scheduler.get_last_lr()[0]
            if use_wandb:
                wandb_log = {
                    "train/total_loss": float(total_loss.detach().item()),
                    "train/lr": float(lr),
                    "train/grad_norm_preclip": float(
                        grad_norm_preclip.detach().item()
                        if torch.is_tensor(grad_norm_preclip) else grad_norm_preclip
                    ),
                    # GT valid depth ratios (all views)
                    "train/mean_view_valid_depth_ratio": float(sum(view_valid_ratios) / max(len(view_valid_ratios), 1)),
                    "train/min_view_valid_depth_ratio": float(min(view_valid_ratios)) if view_valid_ratios else -1.0,
                    "train/num_valid_views": float(num_valid_views),
                    # Pred metrics (only over rendered=valid views; -1 sentinels excluded)
                    "train/mean_view_pred_alpha_mean": float(
                        sum(v for v in view_pred_alpha_means if v >= 0) / max(sum(1 for v in view_pred_alpha_means if v >= 0), 1)
                    ),
                    "train/min_view_pred_alpha_mean": float(
                        min((v for v in view_pred_alpha_means if v >= 0), default=-1.0)
                    ),
                    "train/mean_view_pred_depth_pos_ratio": float(
                        sum(v for v in view_pred_depth_pos_ratios if v >= 0) / max(sum(1 for v in view_pred_depth_pos_ratios if v >= 0), 1)
                    ),
                    "train/min_view_pred_depth_pos_ratio": float(
                        min((v for v in view_pred_depth_pos_ratios if v >= 0), default=-1.0)
                    ),
                    # First view for backward compat
                    "train/first_view_valid_depth_ratio": float(view_valid_ratios[0]) if view_valid_ratios else -1.0,
                    "train/first_view_pred_alpha_mean": float(view_pred_alpha_means[0]) if view_pred_alpha_means else -1.0,
                    "train/first_view_pred_alpha_max": float(view_pred_alpha_maxes[0]) if view_pred_alpha_maxes else -1.0,
                    "train/first_view_pred_depth_pos_ratio": float(view_pred_depth_pos_ratios[0]) if view_pred_depth_pos_ratios else -1.0,
                    # Step timing
                    "train/time_data_s": float(t_data_end - t_data_start),
                    "train/time_fwdbwd_s": float(t_step_end - t_fwd_start),
                }
                for k, v in log_components.items():
                    wandb_log[f"train/{k}"] = float(v)
                wandb.log(wandb_log, step=global_step)

            if global_step % args.log_every == 0:
                elapsed = time.time() - t0
                parts = ' | '.join(f"{k}={v:.4f}" for k, v in log_components.items())
                logger.info(
                    f"step={global_step:06d} | lr={lr:.2e} | {parts} | "
                    f"data={t_data_end - t_data_start:.2f}s | "
                    f"fwd+bwd={t_step_end - t_fwd_start:.2f}s | "
                    f"{elapsed / global_step:.2f}s/step"
                )

            # Validation
            if val_dataset is not None and (
                global_step % args.val_every == 0 or global_step == total_steps
            ):
                val_metrics = run_validation(
                    model, renderer, val_dataset, device,
                    num_samples=args.num_val_samples,
                )
                val_str = ' | '.join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                logger.info(f"step={global_step:06d} | VALIDATION | {val_str}")
                if use_wandb:
                    wandb.log(val_metrics, step=global_step)

            # Checkpoint
            if global_step % args.save_every == 0 or global_step == total_steps:
                ckpt_path = output_dir / f"ckpt_{global_step:06d}.pt"
                torch.save({
                    'step': global_step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'args': vars(args),
                }, ckpt_path)
                logger.info(f"Saved checkpoint → {ckpt_path}")

            t_data_start = time.time()

    logger.info("Training complete.")
    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train ShapeGSAE')

    # Data
    p.add_argument('--data_dir', required=True)
    p.add_argument('--output_dir', default='runs/gs_ae')
    p.add_argument('--val_dir', type=str, default=None,
                   help='Directory of validation meshes (separate from train). '
                        'GT RGBD must be pre-cached with same camera settings.')
    p.add_argument('--max_items', type=int, default=None,
                   help='Cap dataset size (useful for debugging)')
    p.add_argument('--mesh_blacklist', type=str, default=None,
                   help='Path to a text file listing mesh paths to exclude (one per line).')
    p.add_argument('--categories', type=str, default=None,
                   help='Comma-separated category names or 8-digit ShapeNet synset IDs '
                        'to include (e.g. "chair", "chair,table", or "03001627"). '
                        'Folders are kept iff their name starts with "<synset_id>_". '
                        'Known names: ' + ', '.join(sorted(CATEGORY_NAME_TO_ID)) +
                        '. Default: no filtering.')

    # Model
    p.add_argument('--num_latents', type=int, default=2048)
    p.add_argument('--embed_dim', type=int, default=64)
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--heads', type=int, default=16)
    p.add_argument('--num_encoder_layers', type=int, default=8)
    p.add_argument('--num_decoder_layers', type=int, default=8)
    p.add_argument('--pc_size', type=int, default=5120)
    p.add_argument('--pc_sharpedge_size', type=int, default=5120)
    p.add_argument('--downsample_ratio', type=int, default=20)
    p.add_argument('--shapevae_ckpt', type=str, default=None,
                   help='Optional ShapeVAE .ckpt for warm-starting the encoder')

    # Rendering
    p.add_argument('--render_height', type=int, default=256)
    p.add_argument('--render_width', type=int, default=256)
    p.add_argument('--num_views', type=int, default=8)
    p.add_argument('--camera_distance', type=float, default=2.5)
    p.add_argument('--elevation_deg', type=float, default=20.0)
    p.add_argument('--camera_azimuths', type=str, default=None,
                   help='Comma-separated azimuth angles in degrees, one per view. '
                        'Length must equal --num_views. '
                        'IMPORTANT: for 2-view training use "0,90" (front + right side) '
                        'NOT the default "0,180" — the back view (180°) has no geometry '
                        'for most Objaverse objects. Default: evenly spaced (0,180 for 2 views).')

    # Loss weights
    p.add_argument('--lambda_ssim', type=float, default=0.2,
                   help='SSIM loss weight. Applied only on foreground pixels.')
    p.add_argument('--lambda_d', type=float, default=1.0,
                   help='Depth L1 weight. Key lever for Gaussian placement quality.')
    p.add_argument('--lambda_alpha', type=float, default=0.05,
                   help='Alpha supervision on fg pixels. Prevents opacity collapse. '
                        'Critical: do not set to 0 unless debugging.')
    p.add_argument('--lambda_scale', type=float, default=0.01)
    p.add_argument('--lambda_opa', type=float, default=0.01)
    p.add_argument('--target_log_scale', type=float, default=-3.0,
                   help='Target log-scale for scale regularizer. -3.0 → scale≈0.05 units. '
                        'Increase toward -2.0 for coarser Gaussians.')
    p.add_argument('--target_opacity', type=float, default=0.5,
                   help='Target opacity for opacity regularizer.')
    p.add_argument('--fg_weight', type=float, default=0.75,
                   help='Foreground pixel weight in RGB L1 (0.75 = fg gets 3× bg weight). '
                        'Gradient scale is preserved regardless of fg ratio.')
    p.add_argument('--min_valid_depth_ratio', type=float, default=0.02,
                   help='Skip depth loss for views with fewer than this fraction of valid pixels.')

    # Optimiser
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--warmup_steps', type=int, default=500,
                   help='Linear LR warmup steps before cosine decay. Set 0 to disable.')
    p.add_argument('--max_steps', type=int, default=200_000)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)

    # Misc
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--save_every', type=int, default=5_000)
    p.add_argument('--val_every', type=int, default=2000,
                   help='Run validation every N steps (requires --val_dir).')
    p.add_argument('--num_val_samples', type=int, default=20,
                   help='Max number of val meshes to evaluate per validation pass.')
    p.add_argument('--resume_ckpt', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke_overfit', action='store_true',
                   help='Minimal learning sanity-check: single view + RGB L1 only.')
    p.add_argument('--precache_only', action='store_true',
                   help='Pre-render and cache GT RGBD for all meshes then exit (no training). '
                        'Run this with CUDA_VISIBLE_DEVICES="" and --num_workers 0 before '
                        'multi-view training to avoid EGL conflicts during the training loop.')
    p.add_argument('--use_wandb', action='store_true',
                   help='Enable Weights & Biases logging.')
    p.add_argument('--wandb_project', type=str, default='hy3dgs',
                   help='W&B project name.')
    p.add_argument('--wandb_entity', type=str, default='nathanr-ntu',
                   help='W&B entity/team name.')
    p.add_argument('--wandb_run_name', type=str, default=None,
                   help='Optional W&B run name (defaults to output_dir basename).')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train(args)
