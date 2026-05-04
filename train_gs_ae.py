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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import trimesh

from hy3dgen.shapegen.models.autoencoders.model import ShapeGSAE
from hy3dgen.shapegen.surface_loaders import RGBSharpEdgeSurfaceLoader
from hy3dgen.shapegen.gs_renderer import (
    GaussianRenderer,
    RGBDLoss,
    build_orbit_cameras,
    orbit_c2w,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


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
    ):
        self.height = height
        self.width = width
        self.fov_deg = fov_deg
        self.camera_distance = camera_distance
        self.elevation_deg = elevation_deg
        self.num_views = num_views
        self.device = device

        # Cache tag encodes the rendering configuration
        self._tag = f"h{height}w{width}v{num_views}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_render(self, mesh_path: str, mesh: Optional[trimesh.Trimesh] = None):
        """Return cached GT RGBD or render it now.

        Returns:
            rgbs   : list of (H, W, 3) float tensors, values in [0, 1]
            depths : list of (H, W, 1) float tensors, values ≥ 0
            c2ws   : list of (4, 4) float tensors (camera-to-world)
        """
        cache_path = f"{mesh_path}.gt_rgbd_{self._tag}.pt"
        if os.path.exists(cache_path):
            data = torch.load(cache_path, map_location='cpu')
            return data['rgbs'], data['depths'], data['c2ws']

        if mesh is None:
            mesh = self._load_mesh(mesh_path)

        c2ws = build_orbit_cameras(
            num_views=self.num_views,
            elevation_deg=self.elevation_deg,
            radius=self.camera_distance,
            device='cpu',
        )

        rgbs, depths = self._render_views(mesh, c2ws)

        data = {'rgbs': rgbs, 'depths': depths, 'c2ws': c2ws}
        torch.save(data, cache_path)
        return rgbs, depths, c2ws

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mesh(path: str) -> trimesh.Trimesh:
        scene_or_mesh = trimesh.load(path, process=False)
        if isinstance(scene_or_mesh, trimesh.scene.Scene):
            return scene_or_mesh.dump(concatenate=True)
        return scene_or_mesh

    def _render_views(self, mesh, c2ws):
        """Dispatch to pyrender or vertex-color fallback."""
        try:
            return self._render_pyrender(mesh, c2ws)
        except Exception as e:
            logger.warning(f"pyrender failed ({e}); using vertex-color fallback renderer.")
            return self._render_vertex_color(mesh, c2ws)

    def _render_pyrender(self, mesh, c2ws):
        """GPU-less offscreen RGBD rendering via pyrender + EGL."""
        import pyrender  # noqa: import inside method so it's optional

        # Build pyrender mesh from trimesh
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
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
        surface  : (1, pc_size+pc_sharpedge_size, 9) float32 tensor
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
        cache_renders: bool = True,
        max_items: Optional[int] = None,
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
        )
        self.cache_renders = cache_renders

        # Discover mesh files
        data_path = Path(data_dir)
        self.mesh_paths = sorted([
            str(p) for p in data_path.rglob('*')
            if p.suffix.lower() in MESH_EXTENSIONS
        ])
        if max_items is not None:
            self.mesh_paths = self.mesh_paths[:max_items]
        logger.info(f"Dataset: {len(self.mesh_paths)} meshes in {data_dir}")

    def __len__(self) -> int:
        return len(self.mesh_paths)

    def __getitem__(self, idx: int) -> Dict:
        path = self.mesh_paths[idx]
        try:
            surface = self.loader(path)                         # (1, N, 9)
            rgbs, depths, c2ws = self.gt_renderer.get_or_render(path)
        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")
            # Return a dummy item; DataLoader will average this out
            return self.__getitem__((idx + 1) % len(self))

        return {
            'surface': surface.squeeze(0),   # (N, 9) — DataLoader adds batch dim
            'rgbs': rgbs,
            'depths': depths,
            'c2ws': c2ws,
            'mesh_path': path,
        }


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
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")

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

    # ---- Optimiser & LR schedule ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    total_steps = args.max_steps
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

    # ---- Training loop ----
    model.train()
    epoch = 0
    t0 = time.time()

    while global_step < total_steps:
        epoch += 1
        for batch in loader:
            if global_step >= total_steps:
                break

            surface = batch['surface'].to(device, non_blocking=True)   # (B, N, 9)
            gt_rgbs = batch['rgbs']     # list of (B, H, W, 3)
            gt_depths = batch['depths'] # list of (B, H, W, 1)
            c2ws = batch['c2ws']        # list of (4,4)

            # Forward pass
            means, scales, rotations, opacities, colors = model(surface)

            # Accumulate rendering loss over all views
            total_loss = torch.zeros(1, device=device)
            log_components: Dict[str, float] = {}

            B = surface.shape[0]
            for view_idx, (gt_rgb_b, gt_depth_b, c2w) in enumerate(
                zip(gt_rgbs, gt_depths, c2ws)
            ):
                gt_rgb_b = gt_rgb_b.to(device)       # (B, H, W, 3)
                gt_depth_b = gt_depth_b.to(device)   # (B, H, W, 1)
                c2w = c2w.to(device)

                # Render each item in the batch separately (gsplat is per-scene)
                pred_rgbs_list, pred_depths_list = [], []
                for b in range(B):
                    out = renderer(
                        means[b], scales[b], rotations[b],
                        opacities[b], colors[b], c2w,
                    )
                    pred_rgbs_list.append(out['rgb'])
                    pred_depths_list.append(out['depth'])

                pred_rgb = torch.stack(pred_rgbs_list, dim=0)     # (B, H, W, 3)
                pred_depth = torch.stack(pred_depths_list, dim=0) # (B, H, W, 1)

                valid_mask = gt_depth_b > 0

                view_loss, comps = criterion(
                    pred_rgb, gt_rgb_b,
                    pred_depth, gt_depth_b,
                    valid_mask=valid_mask,
                    scales=scales.view(-1, 3).log(),  # log(scale) for regulariser
                    opacities=opacities.view(-1, 1),
                )
                total_loss = total_loss + view_loss / len(c2ws)

                for k, v in comps.items():
                    log_components[k] = log_components.get(k, 0.0) + v.item() / len(c2ws)

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            # Logging
            if global_step % args.log_every == 0:
                elapsed = time.time() - t0
                lr = scheduler.get_last_lr()[0]
                parts = ' | '.join(f"{k}={v:.4f}" for k, v in log_components.items())
                logger.info(
                    f"step={global_step:06d} | lr={lr:.2e} | {parts} | "
                    f"{elapsed / global_step:.2f}s/step"
                )

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

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train ShapeGSAE')

    # Data
    p.add_argument('--data_dir', required=True)
    p.add_argument('--output_dir', default='runs/gs_ae')
    p.add_argument('--max_items', type=int, default=None,
                   help='Cap dataset size (useful for debugging)')

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

    # Loss weights
    p.add_argument('--lambda_ssim', type=float, default=0.2)
    p.add_argument('--lambda_d', type=float, default=0.5)
    p.add_argument('--lambda_scale', type=float, default=0.01)
    p.add_argument('--lambda_opa', type=float, default=0.01)

    # Optimiser
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--max_steps', type=int, default=200_000)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)

    # Misc
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--save_every', type=int, default=5_000)
    p.add_argument('--resume_ckpt', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train(args)
