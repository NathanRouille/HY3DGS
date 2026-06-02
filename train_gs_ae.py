"""Training script for ShapeGSAE: Point Cloud → 3D Gaussian Splatting.

Usage:
    python train_gs_ae.py --data_dir /path/to/meshes --output_dir runs/gs_ae

Mesh directory should contain GLB, OBJ, or PLY files (one object per file).

GT RGBD supervision is rendered offline the first time a mesh is encountered
and cached to disk (as .pt tensors) next to the mesh files.

Dependencies (beyond base requirements.txt):
    pip install gsplat pytorch-msssim pyrender lpips
    pip install trimesh[easy]   # for texture support
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

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
from hy3dgen.shapegen.gt_cache_util import (
    canonical_obj_path,
    experiment_manifest_path,
    gt_cache_file_path,
    is_usable_gt_cache_file,
)
from hy3dgen.shapegen.gs_renderer import (
    GT_CACHE_TAG_V46,
    GaussianRenderer,
    RGBDLoss,
    TOTAL_V46_STAGGER,
    VIEW46_TRAIN_ALLOWED,
    build_view46_c2ws,
)
from hy3dgen.shapegen.eval_metrics import compute_psnr, compute_ssim_fg


class GtCacheNotFoundError(FileNotFoundError):
    """Raised when training requires a pre-cached GT file that is missing."""

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


def mesh_path_has_usable_gt_cache(mesh_path: str, gt_tag: str) -> bool:
    """True if a valid GT cache (v46) exists on disk for this mesh."""
    return is_usable_gt_cache_file(gt_cache_file_path(mesh_path, gt_tag))


def load_experiment_manifest(data_dir: str) -> Optional[Dict]:
    """Load ``manifest.json`` from the experiment root (parent of train/ or val/)."""
    data_path = Path(data_dir).resolve()
    if data_path.name in ("train", "val"):
        manifest_path = experiment_manifest_path(data_path.parent)
    else:
        manifest_path = experiment_manifest_path(data_path)
    if not manifest_path.is_file():
        return None
    import json
    with open(manifest_path) as f:
        return json.load(f)


def snap_train_views_v46(requested: int) -> int:
    """Snap requested view count to nearest allowed preset for v46 layout."""
    allowed = list(VIEW46_TRAIN_ALLOWED)
    return min(allowed, key=lambda x: abs(x - requested))


def train_view_indices_v46(num_train: int) -> List[int]:
    """Subset indices into the fixed 46-view ordering (6 canonical + 5×8 grid)."""
    k = snap_train_views_v46(num_train)
    # Grid view at row r, col c → global index 6 + r*8 + c
    G = lambda r, c: 6 + r * 8 + c
    canon = list(range(6))
    if k == 46:
        return list(range(46))
    if k == 38:
        return canon + [G(r, c) for r in [0,1,2,4] for c in range(8)]
    if k == 30:
        return canon + [G(r, c) for r in [0,1,3] for c in range(8)]
    if k == 22:
        return canon + [G(r, c) for r in [0,3] for c in range(8)] # or canon + [G(r, 2*c) for r in [0,1,3,4] for c in range(4)]
    if k == 14:
        return canon + [G(r, 2*c) for r in [0,3] for c in range(4)] # or canon + [G(3, c) for c in range(8)]
    if k == 6:
        return canon
    raise ValueError(f"Unsupported train view count: {k}")


def debug_gt_cache_path(mesh_path: str, debug_root: str) -> str:
    """Separate cache file path under debug_renders/ (does not touch dataset tree)."""
    h = hashlib.sha256(os.path.abspath(mesh_path).encode("utf-8")).hexdigest()[:20]
    p = Path(debug_root) / "gt_cache"
    p.mkdir(parents=True, exist_ok=True)
    return str(p / f"{h}.pt")


# ---------------------------------------------------------------------------
# Evaluation metrics: imported from hy3dgen.shapegen.eval_metrics
# ---------------------------------------------------------------------------


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
    psnr_list, ssim_list = [], []

    indices = list(range(len(val_dataset)))[:num_samples]
    for idx in indices:
        try:
            sample = val_dataset[idx]
        except Exception as e:
            logger.warning(f"Val sample {idx} failed: {e}")
            continue

        surface = sample['surface'].unsqueeze(0).to(device)
        # Defensive slice: even if val_dataset was built with more views, we always
        # evaluate on the 6 canonical views (indices 0..5 in v46) for fair
        # cross-experiment comparison.
        gt_rgbs = sample['rgbs'][:6]
        gt_depths = sample['depths'][:6]
        c2ws = sample['c2ws'][:6]

        means, scales, rotations, opacities, sh_coeffs = model(surface)
        means = means[0]
        scales = scales[0]
        rotations = rotations[0]
        opacities = opacities[0]
        sh_coeffs = sh_coeffs[0]

        for gt_rgb, gt_depth, c2w in zip(gt_rgbs, gt_depths, c2ws):
            valid_mask = (gt_depth > 0)                       # (H, W, 1)
            if float(valid_mask.float().mean().item()) < 0.02:
                continue
            out = renderer(means, scales, rotations, opacities, sh_coeffs, c2w.to(device))
            pred_rgb = out['rgb'].cpu()

            psnr_list.append(compute_psnr(pred_rgb, gt_rgb, valid_mask))
            ssim_list.append(compute_ssim_fg(pred_rgb, gt_rgb, valid_mask))

    model.train()
    if not psnr_list:
        return {'val/psnr_fg': 0.0, 'val/ssim_fg': 0.0}
    return {
        'val/psnr_fg': float(sum(psnr_list) / len(psnr_list)),
        'val/ssim_fg': float(sum(ssim_list) / len(ssim_list)),
    }


# ---------------------------------------------------------------------------
# GT RGBD rendering (offline, CPU, cached to disk)
# ---------------------------------------------------------------------------

class GTRGBDRenderer:
    """Render ground-truth RGBD from a textured trimesh using pyrender.

    Falls back to a simple vertex-color renderer if pyrender is unavailable.
    Results are cached to disk (``<mesh_path>.gt_rgbd_<tag>.pt`` by default, or
    under ``debug_renders_root/gt_cache/`` in debug mode) and reloaded on
    subsequent calls so each mesh is only rendered once.
    """

    def __init__(
        self,
        height: int = 256,
        width: int = 256,
        fov_deg: float = 49.13,
        camera_distance: float = 2.5,
        elevation_deg: float = 20.0,
        device: str = 'cpu',
        debug_renders_root: Optional[str] = None,
        train_view_indices: Optional[List[int]] = None,
    ):
        self.debug_renders_root = debug_renders_root
        self._train_view_indices = (
            None if train_view_indices is None else [int(i) for i in train_view_indices]
        )

        self.height = height
        self.width = width
        self.fov_deg = fov_deg
        self.camera_distance = camera_distance
        self.elevation_deg = elevation_deg
        self.device = device

        self._tag = f"h{height}w{width}_{GT_CACHE_TAG_V46}"
        self._full_num_views = TOTAL_V46_STAGGER
        self.num_views = (
            self._full_num_views if self._train_view_indices is None
            else len(self._train_view_indices)
        )

    def cache_path(self, mesh_path: str) -> str:
        if self.debug_renders_root:
            return debug_gt_cache_path(mesh_path, self.debug_renders_root)
        return gt_cache_file_path(mesh_path, self._tag)

    @staticmethod
    def canonical_mesh_path(mesh_path: str) -> str:
        return canonical_obj_path(mesh_path)

    def _slice_views(
        self,
        rgbs: List[torch.Tensor],
        depths: List[torch.Tensor],
        c2ws: List[torch.Tensor],
        view_params: Optional[List[Dict[str, float]]],
    ):
        idx = self._train_view_indices
        if idx is None:
            return rgbs, depths, c2ws, view_params
        rgbs = [rgbs[i] for i in idx]
        depths = [depths[i] for i in idx]
        c2ws = [c2ws[i] for i in idx]
        if view_params is not None:
            view_params = [view_params[i] for i in idx]
        return rgbs, depths, c2ws, view_params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_gt_tensors(rgbs, depths, c2ws, use_fp16: bool = True):
        if not use_fp16:
            return rgbs, depths, [c.float() for c in c2ws]
        return (
            [x.half() for x in rgbs],
            [x.half() for x in depths],
            [c.float() for c in c2ws],  # 4x4 poses: keep float32
        )

    @staticmethod
    def _unpack_gt_tensors(rgbs, depths, c2ws, storage_dtype=None):
        # storage_dtype from data.get('storage_dtype', 'float32')
        if storage_dtype == 'float16':
            rgbs = [x.float() for x in rgbs]
            depths = [x.float() for x in depths]
        return rgbs, depths, c2ws

    def load_cached(
        self, mesh_path: str
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Optional[List[Dict]]]:
        """Load GT from disk only (canonical ShapeNet cache path). Used during training."""
        return self.get_or_render(mesh_path, mesh=None, allow_render=False)

    def get_or_render(
        self,
        mesh_path: str,
        mesh: Optional[trimesh.Trimesh] = None,
        allow_render: bool = True,
    ):
        """Return cached GT RGBD (and per-view camera params) or render the v46 set.

        Caches are always stored next to the **canonical** OBJ under ShapeNetCore
        (``realpath``), so experiment symlinks and ``prepare_shapenet`` agree.

        Set ``allow_render=False`` to require a pre-existing cache (training).
        """
        cache_path = self.cache_path(mesh_path)
        cached = self._try_load_cache_tuple(cache_path)
        if cached is not None:
            rgbs, depths, c2ws, vp = cached
            return self._slice_views(rgbs, depths, c2ws, vp)

        if not allow_render:
            raise GtCacheNotFoundError(
                f"No usable GT cache at {cache_path} (canonical storage under ShapeNetCore)."
            )

        if mesh is None:
            mesh = self._load_mesh(mesh_path)
        mesh = normalize_mesh(mesh)

        canon = self.canonical_mesh_path(mesh_path)
        c2ws, view_params = build_view46_c2ws(
            canon,
            radius=self.camera_distance,
            fov_deg=self.fov_deg,
            device='cpu',
        )
        rgbs, depths = self._render_views(mesh, c2ws)
        rgbs, depths, c2ws = self._pack_gt_tensors(rgbs, depths, c2ws, use_fp16=True)
        data = {
            'rgbs': rgbs,
            'depths': depths,
            'c2ws': c2ws,
            'view_params': view_params,
            'view_layout': 'v46',
            'storage_dtype': 'float16',
        }
        d = os.path.dirname(cache_path)
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save(data, cache_path)
        rgbs, depths, c2ws = self._unpack_gt_tensors(rgbs, depths, c2ws, storage_dtype='float16')
        return self._slice_views(rgbs, depths, c2ws, view_params)

    @staticmethod
    def _try_load_cache_tuple(cache_path: str):
        """Load cache file → (rgbs, depths, c2ws, view_params|None) or None."""
        if not os.path.exists(cache_path):
            return None
        data = torch.load(cache_path, map_location='cpu')
        rgbs: List[torch.Tensor] = data.get("rgbs", [])
        depths: List[torch.Tensor] = data.get("depths", [])
        c2ws = data.get("c2ws", [])
        rgbs, depths, c2ws = GTRGBDRenderer._unpack_gt_tensors(
            rgbs, depths, c2ws, storage_dtype=data.get("storage_dtype"),
        )
        mean_valid_depth_ratio = float(
            sum(float((d > 0).float().mean().item()) for d in depths)
            / max(len(depths), 1)
        )
        if mean_valid_depth_ratio <= 0.0:
            logger.warning(f"Empty GT cache at {cache_path}; ignoring.")
            return None
        vp = data.get('view_params')
        if vp is not None:
            vp = list(vp)
        return rgbs, depths, c2ws, vp

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
        num_views: int = 14,
        camera_distance: float = 2.5,
        elevation_deg: float = 20.0,
        max_items: Optional[int] = None,
        mesh_blacklist: Optional[str] = None,
        categories: Optional[Iterable[str]] = None,
        precache_full_views: bool = False,
        debug_renders_root: Optional[str] = None,
        require_cached_gt: bool = True,
        use_experiment_manifest: bool = True,
        train_view_indices: Optional[List[int]] = None,
    ):
        self.require_cached_gt = require_cached_gt
        self.loader = RGBSharpEdgeSurfaceLoader(
            num_uniform_points=pc_size,
            num_sharp_points=pc_sharpedge_size,
        )

        if train_view_indices is not None:
            train_idx: Optional[List[int]] = list(train_view_indices)
            logger.info(
                "GT view layout v46: using %d explicit train view indices "
                "(full %d views cached on disk).",
                len(train_idx),
                TOTAL_V46_STAGGER,
            )
        elif precache_full_views:
            train_idx = None
        else:
            snapped = snap_train_views_v46(num_views)
            train_idx = train_view_indices_v46(snapped)
            logger.info(
                f"GT view layout v46: training/preview uses {len(train_idx)} views "
                f"(requested num_views={num_views} → snapped {snapped}); "
                f"full {TOTAL_V46_STAGGER} views are cached on disk."
            )
        self.gt_renderer = GTRGBDRenderer(
            height=render_height,
            width=render_width,
            camera_distance=camera_distance,
            elevation_deg=elevation_deg,
            debug_renders_root=debug_renders_root,
            train_view_indices=train_idx,
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

        data_path = Path(data_dir).resolve()
        self.mesh_paths: List[str] = []
        manifest = load_experiment_manifest(str(data_path)) if use_experiment_manifest else None

        if manifest is not None:
            split_key = "train_mesh_paths" if data_path.name == "train" else (
                "val_mesh_paths" if data_path.name == "val" else None
            )
            if split_key and manifest.get(split_key):
                self.mesh_paths = list(manifest[split_key])
                logger.info(
                    "Loaded %d mesh path(s) from experiment manifest (%s)",
                    len(self.mesh_paths),
                    split_key,
                )
            else:
                logger.warning(
                    "manifest.json found but no paths for split %r; scanning %s",
                    data_path.name,
                    data_path,
                )
                manifest = None

        if manifest is None:
            skipped_category = 0
            skipped_blacklist = 0
            for folder in sorted(data_path.iterdir()):
                if not folder.is_dir():
                    continue
                if allowed_cats is not None:
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
                self.mesh_paths.append(str(potential_obj.resolve()))

            if allowed_cats is not None:
                logger.info(
                    f"Category filter cats={sorted(allowed_cats)}: kept "
                    f"{len(self.mesh_paths)} mesh(es), skipped {skipped_category} folder(s)"
                )
            if skipped_blacklist:
                logger.info(f"Blacklist filter: {skipped_blacklist} mesh(es) excluded")

        if manifest is not None and manifest.get("gt_tag") != self.gt_renderer._tag:
            logger.warning(
                "Experiment manifest gt_tag=%r differs from renderer tag=%r",
                manifest.get("gt_tag"),
                self.gt_renderer._tag,
            )

        if max_items is not None:
            self.mesh_paths = self.mesh_paths[:max_items]

        if require_cached_gt:
            cached_paths = manifest.get("cached_canonical_objs") if manifest else None
            if cached_paths is not None:
                cached_set = set(cached_paths)
                before = len(self.mesh_paths)
                self.mesh_paths = [
                    p for p in self.mesh_paths
                    if canonical_obj_path(p) in cached_set
                ]
                logger.info(
                    "Manifest GT filter: %d / %d meshes have pre-cached GT",
                    len(self.mesh_paths),
                    before,
                )
            else:
                before = len(self.mesh_paths)
                self.mesh_paths = [
                    p for p in self.mesh_paths
                    if is_usable_gt_cache_file(self.gt_renderer.cache_path(p))
                ]
                logger.info(
                    "GT cache filter: %d / %d meshes have usable on-disk GT",
                    len(self.mesh_paths),
                    before,
                )

        logger.info(f"Dataset: {len(self.mesh_paths)} meshes in {data_dir}")
        self._ram_cache: Dict[int, Dict] = {}

    def __len__(self) -> int:
        return len(self.mesh_paths)

    @staticmethod
    def _clone_cached_sample(sample: Dict) -> Dict:
        """Return a copy so training cannot mutate tensors stored in the RAM cache."""
        out: Dict = {
            'surface': sample['surface'].clone(),
            'rgbs': [t.clone() for t in sample['rgbs']],
            'depths': [t.clone() for t in sample['depths']],
            'c2ws': [t.clone() for t in sample['c2ws']],
            'mesh_path': sample['mesh_path'],
        }
        if 'view_params' in sample:
            out['view_params'] = sample['view_params']
        return out

    def __getitem__(self, idx: int) -> Dict:
        if idx in self._ram_cache:
            return self._clone_cached_sample(self._ram_cache[idx])

        # Iterate forward (non-recursively) to find a working sample
        for attempt in range(len(self.mesh_paths)):
            path = self.mesh_paths[(idx + attempt) % len(self.mesh_paths)]
            try:
                surface = self.loader(path)                         # (1, N, 9)
                if self.require_cached_gt:
                    rgbs, depths, c2ws, view_params = self.gt_renderer.load_cached(path)
                else:
                    rgbs, depths, c2ws, view_params = self.gt_renderer.get_or_render(path)
                out: Dict = {
                    'surface': surface.squeeze(0),   # (N, 9) — DataLoader adds batch dim
                    'rgbs': rgbs,
                    'depths': depths,
                    'c2ws': c2ws,
                    'mesh_path': path,
                }
                if view_params is not None:
                    out['view_params'] = view_params
                self._ram_cache[idx] = out
                if len(self._ram_cache) == len(self.mesh_paths):
                    logger.info(
                        "MeshDataset RAM cache full (%d samples)", len(self._ram_cache)
                    )
                elif len(self._ram_cache) == 1:
                    logger.info("MeshDataset RAM cache: loading samples on first access")
                return self._clone_cached_sample(out)
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
    out = {
        'surface': surfaces,
        'rgbs': rgbs,
        'depths': depths,
        'c2ws': c2ws,
    }
    if batch[0].get('view_params') is not None:
        out['view_params'] = [batch[0]['view_params'][v] for v in range(num_views)]
    return out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _accum_train_log(
    acc: Dict[str, torch.Tensor],
    key: str,
    val: torch.Tensor,
) -> None:
    """Accumulate detached loss components on GPU (no .item() sync)."""
    v = val.detach()
    if key in acc:
        acc[key] = acc[key] + v
    else:
        acc[key] = v


def _train_log_to_floats(comp: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.item()) if torch.is_tensor(v) else float(v) for k, v in comp.items()}


def train(args):
    # ---- Resolve category filter ----
    categories = resolve_category_ids(args.categories)
    if categories is not None:
        logger.info(f"Filtering to categories: {sorted(categories)}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    max_grad_norm = 1.0

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
        num_gs_per_anchor=args.num_gs_per_anchor,
        deterministic_encoder=args.deterministic_encoder,
        sh_degree=args.sh_degree,
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
        sh_degree=args.sh_degree,
    ).to(device)

    criterion = RGBDLoss(
        lambda_ssim=args.lambda_ssim,
        lambda_lpips=args.lambda_lpips,
        lambda_d=args.lambda_d,
        lambda_alpha=args.lambda_alpha,
        lambda_scale=args.lambda_scale,
        lambda_opa=args.lambda_opa,
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
        categories=categories,
        precache_full_views=False,
        require_cached_gt=not args.allow_on_the_fly_gt,
        use_experiment_manifest=not args.no_experiment_manifest,
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
    # Validation always uses the 6 canonical views (indices 0..5) for cross-experiment
    # comparability, regardless of how many views training uses.
    val_dataset = None
    if args.val_dir:
        val_dataset = MeshDataset(
            data_dir=args.val_dir,
            pc_size=args.pc_size,
            pc_sharpedge_size=args.pc_sharpedge_size,
            render_height=args.render_height,
            render_width=args.render_width,
            num_views=6,
            camera_distance=args.camera_distance,
            elevation_deg=args.elevation_deg,
            categories=categories,
            precache_full_views=False,
            require_cached_gt=not args.allow_on_the_fly_gt,
            use_experiment_manifest=not args.no_experiment_manifest,
            train_view_indices=list(range(6)),
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
    warmup_steps = args.warmup_steps
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
            will_log = (global_step + 1) % args.log_every == 0

            surface = batch['surface'].to(device, non_blocking=True)   # (B, N, 9)
            gt_rgbs = batch['rgbs']     # list of (B, H, W, 3)
            gt_depths = batch['depths'] # list of (B, H, W, 1)
            c2ws = batch['c2ws']        # list of (4,4)

            # Forward pass
            means, scales, rotations, opacities, sh_coeffs = model(surface)

            # Accumulate rendering loss over all views
            total_loss = torch.zeros((), device=device)
            log_components: Dict[str, torch.Tensor] = {}

            B = surface.shape[0]
            num_views = len(c2ws)
            for view_idx, (gt_rgb_b, gt_depth_b, c2w) in enumerate(
                zip(gt_rgbs, gt_depths, c2ws)
            ):
                gt_rgb_b = gt_rgb_b.to(device)       # (B, H, W, 3)
                gt_depth_b = gt_depth_b.to(device)   # (B, H, W, 1)
                c2w = c2w.to(device)

                # Render each item in the batch separately (gsplat is per-scene)
                pred_rgbs_list, pred_depths_list, pred_alphas_list = [], [], []
                for b in range(B):
                    out = renderer(
                        means[b], scales[b], rotations[b],
                        opacities[b], sh_coeffs[b], c2w,
                    )
                    pred_rgbs_list.append(out['rgb'])
                    pred_depths_list.append(out['depth'])
                    pred_alphas_list.append(out['alpha'])

                pred_rgb = torch.stack(pred_rgbs_list, dim=0)     # (B, H, W, 3)
                pred_depth = torch.stack(pred_depths_list, dim=0) # (B, H, W, 1)
                pred_alpha = torch.stack(pred_alphas_list, dim=0) # (B, H, W, 1)

                valid_mask = gt_depth_b > 0

                view_loss, comps = criterion(
                    pred_rgb, gt_rgb_b,
                    pred_depth, gt_depth_b,
                    pred_alpha=pred_alpha,
                    valid_mask=valid_mask,
                    scales=scales.view(-1, 3).log(),
                    opacities=opacities.view(-1, 1),
                )
                total_loss = total_loss + view_loss

                for k, v in comps.items():
                    _accum_train_log(log_components, k, v)

            # Mean over views: keeps loss magnitude constant across 6/14/22/etc.
            if num_views > 0:
                inv_n = 1.0 / num_views
                total_loss = total_loss * inv_n
                log_components = {k: v * inv_n for k, v in log_components.items()}

            optimizer.zero_grad()
            total_loss.backward()
            grad_norm_preclip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            t_step_end = time.time()

            global_step += 1

            # Logging (defer all .item() / wandb to log_every to avoid GPU sync stalls)
            if global_step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                log_floats = _train_log_to_floats(log_components)

                elapsed = time.time() - t0
                parts = ' | '.join(f"{k}={v:.4f}" for k, v in log_floats.items())
                logger.info(
                    f"step={global_step:06d} | lr={lr:.2e} | {parts} | "
                    f"data={t_data_end - t_data_start:.2f}s | "
                    f"fwd+bwd={t_step_end - t_fwd_start:.2f}s | "
                    f"{elapsed / global_step:.2f}s/step"
                )

                if use_wandb:
                    # Panels follow dict insertion order in wandb. Order chosen so
                    # the most important plots (lr, grad_norm, regularisers) come
                    # first, then individual loss components, with total_loss last.
                    grad_norm_val = float(
                        grad_norm_preclip.detach().item()
                        if torch.is_tensor(grad_norm_preclip) else grad_norm_preclip
                    )
                    wandb_log: Dict[str, float] = {
                        "train/lr": float(lr),
                        "train/grad_norm": grad_norm_val,
                    }
                    # Regularisers first
                    for k in ("scale_reg", "opa_reg"):
                        if k in log_floats:
                            wandb_log[f"train/{k}"] = log_floats[k]
                    # Per-loss components
                    for k in (args.rgb_loss_type, "ssim", "lpips", "depth", "alpha_sup"):
                        if k in log_floats:
                            wandb_log[f"train/{k}"] = log_floats[k]
                    wandb_log["train/total_loss"] = float(total_loss.detach().item())
                    wandb.log(wandb_log, step=global_step)

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
    p.add_argument('--num_gs_per_anchor', type=int, default=1,
                   help='Number of Gaussians predicted per FPS anchor (default 1). '
                        'Total Gaussians = num_latents * num_gs_per_anchor.')
    p.add_argument(
        '--sh_degree',
        type=int,
        default=1,
        choices=(0, 1),
        help='SH degree for view-dependent color (0=flat RGB, 1=SH1).',
    )
    p.add_argument('--shapevae_ckpt', type=str, default=None,
                   help='Optional ShapeVAE .ckpt for warm-starting the encoder')

    # Rendering
    p.add_argument('--render_height', type=int, default=512)
    p.add_argument('--render_width', type=int, default=512)
    p.add_argument(
        '--num_views', type=int, default=14,
        help=(
            f'Number of v46-staggered training views per step. Snapped to one of '
            f'{list(VIEW46_TRAIN_ALLOWED)}. Validation always uses 6 canonical views.'
        ),
    )
    p.add_argument('--camera_distance', type=float, default=3.5)
    p.add_argument('--elevation_deg', type=float, default=20.0)
    p.add_argument(
        '--allow_on_the_fly_gt',
        action='store_true',
        help='Allow rendering GT during training if cache is missing (slow; default is '
             'load-only from ShapeNetCore caches prepared by prepare_shapenet.py).',
    )
    p.add_argument(
        '--no_experiment_manifest',
        action='store_true',
        help='Scan data_dir for symlinks instead of reading experiment manifest.json.',
    )

    # Encoder behaviour
    p.add_argument(
        '--deterministic_encoder',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Sequential point subset + FPS random_start=False so the encoder is a '
            'pure function of the parameters (required for clean overfit loss '
            'curves). Use --no-deterministic_encoder for the original stochastic '
            'augmentation behaviour during generalisation training.'
        ),
    )

    # Loss weights
    p.add_argument('--rgb_loss_type', choices=('mse', 'l1'), default='mse',
                   help='RGB photometric loss: MSE (LGM/GRM/GS-LRM default; preserves high frequencies) '
                        'or L1.')
    p.add_argument('--lambda_ssim', type=float, default=0.2,
                   help='SSIM loss weight. Full-image (no masking).')
    p.add_argument('--lambda_lpips', type=float, default=0.1,
                   help='LPIPS perceptual loss weight. Full-image; ramped in via --lpips_warmup_steps.')
    p.add_argument('--lpips_warmup_steps', type=int, default=5000,
                   help='Linear LPIPS ramp 0→1 over this many steps (0 disables). Mitigates the '
                        '"perceptual mean" texture-washout failure mode.')
    p.add_argument('--lambda_d', type=float, default=1.0,
                   help='Full-image depth L1 weight (GT background depth is 0).')
    p.add_argument('--lambda_alpha', type=float, default=0.05,
                   help='Alpha supervision L1: gt_alpha = valid_mask (full image).')
    p.add_argument('--lambda_scale', type=float, default=0.01,
                   help='AnchorSplat volume penalty: penalises mean(s0*s1*s2) per Gaussian.')
    p.add_argument('--lambda_opa', type=float, default=0.05,
                   help='Binary opacity entropy weight: pushes each Gaussian to fully opaque '
                        '(carves thin features) or fully transparent (carves holes). Replaces '
                        'AnchorSplat\'s linear (1-opacity) penalty which allowed semi-transparent '
                        'Gaussians at 0.5 ("vanishing colors").')

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
