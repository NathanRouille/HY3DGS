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
    anchor_position_deltas,
    build_view46_c2ws,
    expand_anchor_positions,
)
from hy3dgen.shapegen.eval_metrics import (
    compute_psnr_fg,
    compute_psnr_full,
    compute_ssim_full,
)


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
# Periodic canonical eval (6 v46 views)
# ---------------------------------------------------------------------------

CANONICAL_EVAL_NUM_VIEWS = 6


def _nan_eval_metrics(prefix: str) -> Dict[str, float]:
    return {
        f'{prefix}/psnr': float('nan'),
        f'{prefix}/psnr_fg': float('nan'),
        f'{prefix}/ssim': float('nan'),
        f'{prefix}/lpips': float('nan'),
        f'{prefix}/l1_depth': float('nan'),
    }


@torch.no_grad()
def run_canonical_eval(
    model: torch.nn.Module,
    renderer: GaussianRenderer,
    eval_dataset,
    device: torch.device,
    num_samples: int,
    prefix: str,
    lpips_net=None,
) -> Dict[str, float]:
    """Evaluate on a subset of meshes using the 6 canonical v46 views only.

    ``{prefix}/psnr`` is full-image PSNR (``psnr_full``), matching
    ``evaluate_gs_ae.py`` summary.csv canonical scores.
    ``{prefix}/psnr_fg`` is foreground-only PSNR for diagnostics.
    """
    if num_samples <= 0 or eval_dataset is None or len(eval_dataset) == 0:
        return _nan_eval_metrics(prefix)

    model.eval()
    psnr_full_list, psnr_fg_list = [], []
    ssim_list, lpips_list, depth_l1_list = [], [], []

    indices = list(range(min(len(eval_dataset), num_samples)))
    for idx in indices:
        try:
            sample = eval_dataset[idx]
        except Exception as e:
            logger.warning("%s sample %d failed: %s", prefix, idx, e)
            continue

        surface = sample['surface'].unsqueeze(0).to(device)
        latents, query_positions = model.encode(surface)
        means, scales, rotations, opacities, sh_coeffs = model.decode(
            latents, query_positions,
        )
        gt_rgbs = sample['rgbs'][:CANONICAL_EVAL_NUM_VIEWS]
        gt_depths = sample['depths'][:CANONICAL_EVAL_NUM_VIEWS]
        c2ws = sample['c2ws'][:CANONICAL_EVAL_NUM_VIEWS]
        means = means[0]
        scales = scales[0]
        rotations = rotations[0]
        opacities = opacities[0]
        sh_coeffs = sh_coeffs[0]

        for gt_rgb, gt_depth, c2w in zip(gt_rgbs, gt_depths, c2ws):
            valid_mask = gt_depth > 0
            if float(valid_mask.float().mean().item()) < 0.02:
                continue
            out = renderer(means, scales, rotations, opacities, sh_coeffs, c2w.to(device))
            pred_rgb = out['rgb'].cpu()
            pred_depth = out['depth'].cpu()

            psnr_full_list.append(compute_psnr_full(pred_rgb, gt_rgb))
            psnr_fg_list.append(compute_psnr_fg(pred_rgb, gt_rgb, valid_mask))
            ssim_list.append(compute_ssim_full(pred_rgb, gt_rgb))
            depth_l1_list.append(float(F.l1_loss(
                pred_depth[valid_mask].float().reshape(-1),
                gt_depth[valid_mask].float().reshape(-1),
            ).item()))

            if lpips_net is not None:
                pred_nchw = pred_rgb.unsqueeze(0).permute(0, 3, 1, 2).to(device)
                gt_nchw = gt_rgb.unsqueeze(0).permute(0, 3, 1, 2).to(device)
                lpips_list.append(float(lpips_net(pred_nchw, gt_nchw).mean().item()))

    model.train()
    if not psnr_full_list:
        return _nan_eval_metrics(prefix)

    n = len(psnr_full_list)
    metrics = {
        f'{prefix}/psnr': float(sum(psnr_full_list) / n),
        f'{prefix}/psnr_fg': float(sum(psnr_fg_list) / n),
        f'{prefix}/ssim': float(sum(ssim_list) / n),
        f'{prefix}/l1_depth': float(sum(depth_l1_list) / n),
    }
    if lpips_list:
        metrics[f'{prefix}/lpips'] = float(sum(lpips_list) / n)
    else:
        metrics[f'{prefix}/lpips'] = float('nan')
    return metrics


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(
        self,
        metric: str,
        patience: int,
        min_delta: float = 0.0,
        mode: str = 'max',
    ):
        self.metric = metric
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best: Optional[float] = None
        self.best_step = 0
        self.bad_epochs = 0

    def _is_improvement(self, value: float) -> bool:
        if self.best is None or math.isnan(self.best):
            return True
        if self.mode == 'max':
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def update(self, metrics: Dict[str, float], step: int) -> bool:
        """Record metrics; return True if training should stop."""
        if self.metric not in metrics:
            logger.warning(
                "Early stopping metric %s missing from eval metrics; skipping check.",
                self.metric,
            )
            return False

        value = metrics[self.metric]
        if math.isnan(value):
            return False

        if self._is_improvement(value):
            self.best = value
            self.best_step = step
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def _load_lpips_eval_net(device: torch.device):
    try:
        import lpips as lpips_mod
    except ImportError:
        logger.warning(
            "lpips not installed — eval lpips will be NaN. Install with: pip install lpips"
        )
        return None
    net = lpips_mod.LPIPS(net='vgg', verbose=False)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net.to(device)


def _save_checkpoint(
    path: Path,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
) -> None:
    torch.save({
        'step': global_step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'args': vars(args),
    }, path)


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
        seed: Optional[int] = None,
    ):
        self.require_cached_gt = require_cached_gt
        self.seed = seed
        self.loader = RGBSharpEdgeSurfaceLoader(
            num_uniform_points=pc_size,
            num_sharp_points=pc_sharpedge_size,
            seed=seed,
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
    c2ws = [
        torch.stack([item['c2ws'][v] for item in batch], dim=0)  # (B, 4, 4) per view
        for v in range(num_views)
    ]
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


def _accum_train_component(
    acc: Dict[str, torch.Tensor],
    key: str,
    val: torch.Tensor,
) -> None:
    """Accumulate loss components with gradients (for per-term grad-norm logging)."""
    if key in acc:
        acc[key] = acc[key] + val
    else:
        acc[key] = val


def _grad_norm_for_loss(
    parameters: List[torch.nn.Parameter],
    loss_scalar: torch.Tensor,
) -> float:
    """L2 norm of gradients of ``loss_scalar`` w.r.t. trainable parameters."""
    if not loss_scalar.requires_grad:
        return 0.0
    params = [p for p in parameters if p.requires_grad]
    grads = torch.autograd.grad(
        loss_scalar,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    total_sq = sum(
        g.detach().float().norm().pow(2).item()
        for g in grads
        if g is not None
    )
    return float(total_sq ** 0.5)


def _compute_loss_grad_norms(
    model: torch.nn.Module,
    criterion: RGBDLoss,
    components: Dict[str, torch.Tensor],
    delta_loss: Optional[torch.Tensor] = None,
    gaussian_components: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    """Per-term gradient L2 norms (approximate loss balancing diagnostic)."""
    params = list(model.parameters())
    norms: Dict[str, float] = {}
    for name, term in criterion.weighted_terms_for_grad_norm(components).items():
        norms[name] = _grad_norm_for_loss(params, term)
    if gaussian_components is not None:
        for name, term in criterion.weighted_3d_terms_for_grad_norm(gaussian_components).items():
            norms[name] = _grad_norm_for_loss(params, term)
    if delta_loss is not None and criterion.lambda_delta > 0:
        norms["delta_reg"] = _grad_norm_for_loss(params, delta_loss)
    return norms


def _global_grad_norm(model: torch.nn.Module) -> float:
    """L2 norm of all parameter gradients (after backward)."""
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total_sq += float(g.norm().pow(2).item())
    return float(total_sq ** 0.5)


def train(args):
    # ---- Resolve category filter ----
    categories = resolve_category_ids(args.categories)
    if categories is not None:
        logger.info(f"Filtering to categories: {sorted(categories)}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    if args.deterministic_encoder:
        logger.info(
            "Deterministic encoder: sequential point subset + FPS random_start=False "
            "(fixed FPS anchors across steps)."
        )
    else:
        logger.info(
            "Stochastic encoder (--no-deterministic_encoder): fresh point subsample "
            "and FPS random_start=True each forward pass."
        )
    if args.seed is not None:
        logger.info(
            "Input surfaces: per-mesh subsample seeded from global seed=%d, "
            "cached in RAM after first load.",
            args.seed,
        )
    max_grad_norm = float(args.max_grad_norm)
    if max_grad_norm > 0:
        logger.info(f"Gradient clipping enabled: max_grad_norm={max_grad_norm:.4f}")
    else:
        logger.info("Gradient clipping disabled (--max_grad_norm <= 0)")

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
        max_anchor_delta=args.max_anchor_delta,
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
        alpha_bg_weight=args.alpha_bg_weight,
        lambda_scale=args.lambda_scale,
        lambda_opa=args.lambda_opa,
        lambda_delta=args.lambda_delta,
        rgb_loss_type=args.rgb_loss_type,
        lambda_rgb=args.lambda_rgb,
        lambda_edge=args.lambda_edge,
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
        seed=args.seed,
    )
    views_loaded = int(dataset.gt_renderer.num_views)
    views_per_step_cfg = (
        views_loaded if args.views_per_step is None else max(1, int(args.views_per_step))
    )
    logger.info(
        "View sampling: loaded_views=%d (from --num_views=%d), views_per_step=%d",
        views_loaded,
        args.num_views,
        min(views_per_step_cfg, views_loaded),
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

    # ---- Optional eval datasets (canonical views only, for fast periodic eval) ----
    canonical_eval_kwargs = dict(
        pc_size=args.pc_size,
        pc_sharpedge_size=args.pc_sharpedge_size,
        render_height=args.render_height,
        render_width=args.render_width,
        num_views=CANONICAL_EVAL_NUM_VIEWS,
        camera_distance=args.camera_distance,
        elevation_deg=args.elevation_deg,
        categories=categories,
        precache_full_views=False,
        require_cached_gt=not args.allow_on_the_fly_gt,
        use_experiment_manifest=not args.no_experiment_manifest,
        train_view_indices=list(range(CANONICAL_EVAL_NUM_VIEWS)),
        seed=args.seed,
    )

    train_eval_dataset = None
    if args.num_train_eval_samples > 0:
        train_eval_dataset = MeshDataset(
            data_dir=args.data_dir,
            **canonical_eval_kwargs,
        )
        logger.info(
            "Train eval: up to %d meshes from %s (6 canonical views)",
            args.num_train_eval_samples,
            args.data_dir,
        )

    val_dataset = None
    if args.val_dir:
        val_dataset = MeshDataset(
            data_dir=args.val_dir,
            **canonical_eval_kwargs,
        )
        logger.info(
            "Val eval: up to %d meshes from %s (6 canonical views)",
            args.num_val_samples,
            args.val_dir,
        )

    run_periodic_eval = (
        args.num_train_eval_samples > 0 or (val_dataset is not None and args.num_val_samples > 0)
    )
    lpips_eval_net = _load_lpips_eval_net(device) if run_periodic_eval else None

    early_stopper: Optional[EarlyStopping] = None
    if args.early_stopping:
        if val_dataset is None or args.num_val_samples <= 0:
            raise ValueError(
                "--early_stopping requires --val_dir and --num_val_samples > 0"
            )
        early_stopper = EarlyStopping(
            metric=args.early_stopping_metric,
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            mode=args.early_stopping_mode,
        )
        logger.info(
            "Early stopping enabled: metric=%s patience=%d min_delta=%g mode=%s",
            args.early_stopping_metric,
            args.early_stopping_patience,
            args.early_stopping_min_delta,
            args.early_stopping_mode,
        )

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
    training_done = False

    while global_step < total_steps and not training_done:
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
            c2ws = batch['c2ws']        # list of (B, 4, 4)

            # Forward pass (encode+decode so anchor deltas are available for regularisation)
            latents, query_positions = model.encode(surface)
            means, scales, rotations, opacities, sh_coeffs = model.decode(
                latents, query_positions,
            )
            anchors = expand_anchor_positions(query_positions, model.num_gs_per_anchor)
            pos_deltas = anchor_position_deltas(means, anchors)

            # Accumulate rendering loss over all views
            total_loss = torch.zeros((), device=device)
            log_components: Dict[str, torch.Tensor] = {}
            grad_components: Optional[Dict[str, torch.Tensor]] = (
                {} if (will_log and use_wandb and args.log_grad_norms) else None
            )

            B = surface.shape[0]
            num_views_total = len(c2ws)
            if num_views_total == 0:
                logger.warning("Batch has zero GT views; skipping.")
                t_data_start = time.time()
                continue

            views_per_step = int(getattr(args, "views_per_step", 0) or 0)
            if views_per_step <= 0:
                views_per_step = num_views_total
            views_per_step = min(views_per_step, num_views_total)
            if views_per_step < num_views_total:
                sampled_view_indices = sorted(random.sample(range(num_views_total), views_per_step))
            else:
                sampled_view_indices = list(range(num_views_total))

            for b in range(B):
                obj_view_loss = torch.zeros((), device=device)
                obj_grad_components: Optional[Dict[str, torch.Tensor]] = (
                    {} if (will_log and use_wandb and args.log_grad_norms) else None
                )

                for view_idx in sampled_view_indices:
                    gt_rgb_b = gt_rgbs[view_idx][b:b + 1].to(device)       # (1, H, W, 3)
                    gt_depth_b = gt_depths[view_idx][b:b + 1].to(device)   # (1, H, W, 1)
                    c2w_b = c2ws[view_idx][b].to(device)

                    out = renderer(
                        means[b], scales[b], rotations[b],
                        opacities[b], sh_coeffs[b], c2w_b,
                    )
                    pred_rgb = out['rgb'].unsqueeze(0)      # (1, H, W, 3)
                    pred_depth = out['depth'].unsqueeze(0)  # (1, H, W, 1)
                    pred_alpha = out['alpha'].unsqueeze(0)  # (1, H, W, 1)
                    valid_mask = gt_depth_b > 0

                    view_loss, comps = criterion(
                        pred_rgb, gt_rgb_b,
                        pred_depth, gt_depth_b,
                        pred_alpha=pred_alpha,
                        valid_mask=valid_mask,
                    )
                    obj_view_loss = obj_view_loss + view_loss

                    for k, v in comps.items():
                        _accum_train_log(log_components, k, v)
                    if obj_grad_components is not None:
                        for k, v in comps.items():
                            if k not in ("total", "valid_ratio"):
                                _accum_train_component(obj_grad_components, k, v)

                inv_views = 1.0 / views_per_step
                obj_view_loss = obj_view_loss * inv_views
                total_loss = total_loss + obj_view_loss

                if obj_grad_components is not None:
                    obj_grad_components = {k: v * inv_views for k, v in obj_grad_components.items()}
                    for k, v in obj_grad_components.items():
                        _accum_train_component(grad_components, k, v)

            # Mean over batch objects and sampled views.
            inv_batch = 1.0 / B
            inv_views = 1.0 / views_per_step
            inv_norm = inv_batch * inv_views
            total_loss = total_loss * inv_batch
            log_components = {k: v * inv_norm for k, v in log_components.items()}
            if grad_components is not None:
                grad_components = {k: v * inv_batch for k, v in grad_components.items()}

            log_components["n_views_total"] = torch.tensor(float(num_views_total), device=device)
            log_components["n_views_sampled"] = torch.tensor(float(views_per_step), device=device)

            gaussian_components: Optional[Dict[str, torch.Tensor]] = None
            if args.lambda_scale > 0 or args.lambda_opa > 0:
                gaussian_loss, gaussian_components = criterion.gaussian_regularizer(
                    scales,
                    opacities,
                )
                total_loss = total_loss + gaussian_loss
                for k, v in gaussian_components.items():
                    _accum_train_log(log_components, k, v)

            delta_loss_tensor: Optional[torch.Tensor] = None
            if args.lambda_delta > 0:
                delta_loss_tensor, delta_comps = criterion.anchor_delta_regularizer(pos_deltas)
                total_loss = total_loss + delta_loss_tensor
                for k, v in delta_comps.items():
                    if k != "total":
                        _accum_train_log(log_components, k, v)

            grad_norms: Optional[Dict[str, float]] = None
            if grad_components is not None:
                grad_norms = _compute_loss_grad_norms(
                    model, criterion, grad_components, delta_loss_tensor,
                    gaussian_components=gaussian_components,
                )

            optimizer.zero_grad()
            total_loss.backward()
            if max_grad_norm > 0:
                grad_norm_preclip = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm,
                )
            else:
                grad_norm_preclip = _global_grad_norm(model)
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
                    grad_norm_val = float(
                        grad_norm_preclip.detach().item()
                        if torch.is_tensor(grad_norm_preclip) else grad_norm_preclip
                    )
                    mean_drift_l2 = float(pos_deltas.detach().norm(dim=-1).mean().item())
                    wandb_log: Dict[str, float] = {
                        "train/grad_norm": grad_norm_val,
                        "train/mean_drift_l2": mean_drift_l2,
                    }
                    if grad_norms is not None:
                        for term_name, gn in grad_norms.items():
                            wandb_log[f"grad_norm/{term_name}"] = gn
                    for k in ("scale_reg", "opa_reg", "delta_reg"):
                        if k in log_floats:
                            wandb_log[f"train/{k}"] = log_floats[k]
                    for k in (args.rgb_loss_type, "ssim", "lpips", "depth", "alpha_sup"):
                        if k in log_floats:
                            wandb_log[f"train/{k}"] = log_floats[k]
                    wandb_log["train/total_loss"] = float(total_loss.detach().item())
                    wandb.log(wandb_log, step=global_step)

            # Periodic canonical eval (train + val subsets)
            if run_periodic_eval and (
                global_step % args.val_every == 0 or global_step == total_steps
            ):
                eval_metrics: Dict[str, float] = {}
                val_metrics: Dict[str, float] = {}
                if train_eval_dataset is not None and args.num_train_eval_samples > 0:
                    eval_metrics.update(run_canonical_eval(
                        model, renderer, train_eval_dataset, device,
                        num_samples=args.num_train_eval_samples,
                        prefix='train',
                        lpips_net=lpips_eval_net,
                    ))
                if val_dataset is not None and args.num_val_samples > 0:
                    val_metrics = run_canonical_eval(
                        model, renderer, val_dataset, device,
                        num_samples=args.num_val_samples,
                        prefix='val',
                        lpips_net=lpips_eval_net,
                    )
                    eval_metrics.update(val_metrics)

                if eval_metrics:
                    eval_str = ' | '.join(
                        f"{k}={v:.4f}" for k, v in sorted(eval_metrics.items())
                        if not math.isnan(v)
                    )
                    logger.info(f"step={global_step:06d} | EVAL | {eval_str}")
                    if use_wandb:
                        wandb.log(eval_metrics, step=global_step)

                if early_stopper is not None and val_metrics:
                    should_stop = early_stopper.update(val_metrics, global_step)
                    if (
                        args.early_stopping_save_best
                        and early_stopper.bad_epochs == 0
                        and early_stopper.best is not None
                    ):
                        best_path = output_dir / 'ckpt_best.pt'
                        _save_checkpoint(
                            best_path, global_step, model, optimizer, scheduler, args,
                        )
                        logger.info(
                            "New best %s=%.4f at step %d → %s",
                            early_stopper.metric,
                            early_stopper.best,
                            global_step,
                            best_path,
                        )
                    if should_stop:
                        logger.info(
                            "Early stopping at step %d: %s did not improve for %d eval(s) "
                            "(best=%.4f at step %d)",
                            global_step,
                            early_stopper.metric,
                            early_stopper.patience,
                            early_stopper.best,
                            early_stopper.best_step,
                        )
                        training_done = True
                        break

            # Checkpoint
            if global_step % args.save_every == 0 or global_step == total_steps:
                ckpt_path = output_dir / f"ckpt_{global_step:06d}.pt"
                _save_checkpoint(ckpt_path, global_step, model, optimizer, scheduler, args)
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
    p.add_argument(
        '--views_per_step', type=int, default=None,
        help=(
            'How many views to sample from the loaded --num_views at each optimization '
            'step. Default: use all loaded views. Example: --num_views 22 '
            '--views_per_step 6.'
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
            'Sequential point subset + FPS random_start=False so FPS anchors are '
            'fixed across training steps (pure function of parameters). '
            'Use --no-deterministic_encoder for fresh subsampling and FPS anchors '
            'every forward pass.'
        ),
    )

    # Loss weights
    p.add_argument('--rgb_loss_type', choices=('mse', 'l1'), default='mse',
                   help='RGB photometric loss: MSE (LGM/GRM/GS-LRM default; preserves high frequencies) '
                        'or L1.')
    p.add_argument('--lambda_rgb', type=float, default=1.0,
                   help='Global multiplier on the foreground RGB (L1/MSE) term.')
    p.add_argument('--lambda_edge', type=float, default=4.0,
                   help='Extra per-pixel weight on high-|∇GT| foreground pixels in the RGB '
                        'term (0 = uniform foreground L1/MSE).')
    p.add_argument('--lambda_ssim', type=float, default=0.2,
                   help='SSIM loss weight. Full-image (no masking).')
    p.add_argument('--lambda_lpips', type=float, default=0.1,
                   help='LPIPS perceptual loss weight. Full-image (VGG ignores flat white bg).')
    p.add_argument('--lambda_d', type=float, default=1.0,
                   help='Foreground-only depth L1 weight. Set 0 to disable depth supervision.')
    p.add_argument('--lambda_alpha', type=float, default=0.05,
                   help='Alpha supervision weight. Foreground→1 and background→0 (weighted by '
                        '--alpha_bg_weight). Essential when using foreground-masked RGB loss to '
                        'prevent Gaussians drifting into the background.')
    p.add_argument('--alpha_bg_weight', type=float, default=5.0,
                   help='Extra multiplier on the background alpha L1 term (pred_alpha→0 on bg '
                        'pixels). Higher values push harder against background Gaussians.')
    p.add_argument('--lambda_scale', type=float, default=0.01,
                   help='AnchorSplat volume penalty: penalises mean(s0*s1*s2) per Gaussian.')
    p.add_argument('--lambda_opa', type=float, default=0.05,
                   help='Binary opacity entropy weight: pushes each Gaussian to fully opaque '
                        '(carves thin features) or fully transparent (carves holes). Replaces '
                        'AnchorSplat\'s linear (1-opacity) penalty which allowed semi-transparent '
                        'Gaussians at 0.5 ("vanishing colors").')
    p.add_argument('--lambda_delta', type=float, default=0.0,
                   help='Anchor offset L2 penalty: mean(||means - anchor||^2). '
                        'Use with --max_anchor_delta for AnchorSplat-style hard+soft constraints.')
    p.add_argument('--max_anchor_delta', type=float, default=None,
                   help='Hard cap on anchor offsets via tanh(raw)*(max). AnchorSplat uses 10/128 '
                        '≈ 0.078 in normalised space. Default None = unbounded raw offsets.')

    # Optimiser
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help='Global gradient clipping threshold. Set <= 0 to disable clipping.',
    )
    p.add_argument('--warmup_steps', type=int, default=500,
                   help='Linear LR warmup steps before cosine decay. Set 0 to disable.')
    p.add_argument('--max_steps', type=int, default=200_000)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)

    # Misc
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument(
        '--log_grad_norms',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Log per-loss-term gradient L2 norms to wandb (grad_norm/*) for weight tuning.',
    )
    p.add_argument('--save_every', type=int, default=5_000)
    p.add_argument('--val_every', type=int, default=2000,
                   help='Run canonical eval every N steps (train and/or val subsets).')
    p.add_argument('--num_val_samples', type=int, default=20,
                   help='Max val meshes per eval pass (6 canonical views each).')
    p.add_argument('--num_train_eval_samples', type=int, default=0,
                   help='Max train meshes per eval pass (6 canonical views each). '
                        '0 disables train-subset eval.')
    p.add_argument('--early_stopping', action='store_true',
                   help='Stop when --early_stopping_metric stops improving on the val subset.')
    p.add_argument('--early_stopping_patience', type=int, default=10,
                   help='Number of eval checks without improvement before stopping.')
    p.add_argument('--early_stopping_metric', type=str, default='val/psnr',
                   help='Metric key from periodic eval to monitor (e.g. val/psnr '
                        'for full-image PSNR, val/psnr_fg, val/lpips).')
    p.add_argument('--early_stopping_min_delta', type=float, default=0.0,
                   help='Minimum change in the metric to qualify as an improvement.')
    p.add_argument('--early_stopping_mode', type=str, default='max',
                   choices=('max', 'min'),
                   help='max for PSNR/SSIM; min for LPIPS/depth loss.')
    p.add_argument(
        '--early_stopping_save_best',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Save ckpt_best.pt whenever the monitored metric improves.',
    )
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
