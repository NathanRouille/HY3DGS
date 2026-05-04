"""Differentiable 3D Gaussian Splatting renderer and training losses.

Wraps gsplat (https://github.com/nerfstudio-project/gsplat) to provide:
  - GaussianRenderer: renders a batch of 3DGS scenes from arbitrary cameras.
  - RGBDLoss      : combined L1 + SSIM (RGB) + depth L1 loss with regularisers.

Camera convention used throughout:
  - World space is OpenGL-style (Y-up, Z-towards viewer).
  - Camera extrinsics are [R | t] in world-to-camera form.
  - Intrinsics are provided as (fx, fy, cx, cy) in pixel units.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SSIM (lightweight, no external dependency)
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Per-pixel SSIM map averaged to a scalar. pred/target: (B, C, H, W)."""
    B, C, H, W = pred.shape
    device, dtype = pred.device, pred.dtype

    k1d = _gaussian_kernel_1d(window_size, sigma, device, dtype)
    k2d = k1d[:, None] * k1d[None, :]          # (ws, ws)
    kernel = k2d.expand(C, 1, window_size, window_size)

    pad = window_size // 2

    def _filt(x):
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu_x = _filt(pred)
    mu_y = _filt(target)
    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_xx = _filt(pred ** 2) - mu_xx
    sigma_yy = _filt(target ** 2) - mu_yy
    sigma_xy = _filt(pred * target) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_xx + mu_yy + C1) * (sigma_xx + sigma_yy + C2)
    return (num / den.clamp_min(1e-8)).mean()


# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------

def make_camera_rays(
    height: int,
    width: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    c2w: torch.Tensor,          # (4, 4) camera-to-world
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ray origins and directions (H*W, 3) in world space."""
    device = c2w.device
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing='ij',
    )
    dirs_cam = torch.stack(
        [(xs - cx) / fx, -(ys - cy) / fy, -torch.ones_like(xs)], dim=-1
    )  # (H, W, 3) in camera space
    dirs_cam = dirs_cam.reshape(-1, 3)

    R = c2w[:3, :3]
    t = c2w[:3, 3]
    dirs_world = (R @ dirs_cam.T).T        # (H*W, 3)
    origins = t.unsqueeze(0).expand_as(dirs_world)
    return origins, dirs_world


def orbit_c2w(
    elevation_deg: float,
    azimuth_deg: float,
    radius: float = 2.5,
    device: str = 'cpu',
) -> torch.Tensor:
    """Camera-to-world matrix for an orbit camera."""
    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)

    # Camera position in world space
    x = radius * math.cos(el) * math.sin(az)
    y = radius * math.sin(el)
    z = radius * math.cos(el) * math.cos(az)
    pos = torch.tensor([x, y, z], dtype=torch.float32, device=device)

    # Look-at construction (OpenGL convention, Y-up)
    forward = -F.normalize(pos, dim=0)
    world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
    if abs(forward[1].item()) > 0.99:
        world_up = torch.tensor([0.0, 0.0, 1.0], device=device)
    right = F.normalize(torch.cross(world_up, forward, dim=0), dim=0)
    up = torch.cross(forward, right, dim=0)

    c2w = torch.eye(4, device=device)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = pos
    return c2w


def default_intrinsics(
    height: int,
    width: int,
    fov_deg: float = 49.13,
) -> Tuple[float, float, float, float]:
    """Simple pinhole intrinsics from a vertical FoV."""
    fy = height / (2 * math.tan(math.radians(fov_deg) / 2))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


# ---------------------------------------------------------------------------
# Gaussian renderer
# ---------------------------------------------------------------------------

class GaussianRenderer(nn.Module):
    """Thin wrapper around gsplat.rasterization for training.

    Usage::

        renderer = GaussianRenderer(height=256, width=256)
        rgb, depth, alpha = renderer(means, scales, rotations, opacities, colors, c2w)
    """

    def __init__(
        self,
        height: int = 256,
        width: int = 256,
        fov_deg: float = 49.13,
        near: float = 0.01,
        far: float = 100.0,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        render_depth: bool = True,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.near = near
        self.far = far
        self.register_buffer(
            'bg', torch.tensor(list(background_color), dtype=torch.float32)
        )
        self.render_depth = render_depth

        fx, fy, cx, cy = default_intrinsics(height, width, fov_deg)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def _build_proj_matrix(self, device, dtype):
        fx, fy = self.fx, self.fy
        cx, cy = self.cx, self.cy
        H, W = self.height, self.width
        n, f = self.near, self.far

        # OpenGL-style projection for gsplat (NDC z in [-1,1])
        proj = torch.zeros(4, 4, device=device, dtype=dtype)
        proj[0, 0] = 2 * fx / W
        proj[1, 1] = 2 * fy / H
        proj[0, 2] = 1 - 2 * cx / W
        proj[1, 2] = 2 * cy / H - 1
        proj[2, 2] = -(f + n) / (f - n)
        proj[2, 3] = -2 * f * n / (f - n)
        proj[3, 2] = -1.0
        return proj

    def forward(
        self,
        means: torch.Tensor,        # (N, 3)
        scales: torch.Tensor,       # (N, 3)
        rotations: torch.Tensor,    # (N, 4) wxyz quaternion
        opacities: torch.Tensor,    # (N, 1)
        colors: torch.Tensor,       # (N, 3) RGB
        c2w: torch.Tensor,          # (4, 4) camera-to-world
    ) -> Dict[str, torch.Tensor]:
        """Render a single scene.

        Returns a dict with keys:
            'rgb'   : (H, W, 3) float in [0, 1]
            'depth' : (H, W, 1) float ≥ 0  (0 where nothing rendered)
            'alpha' : (H, W, 1) float in [0, 1]
        """
        try:
            from gsplat import rasterization
        except ImportError as e:
            raise ImportError(
                "gsplat is required for GaussianRenderer. "
                "Install with: pip install gsplat"
            ) from e

        device, dtype = means.device, means.dtype

        # World-to-camera: invert c2w
        w2c = torch.inverse(c2w.to(dtype))   # (4, 4)
        viewmat = w2c.unsqueeze(0)            # (1, 4, 4)

        proj = self._build_proj_matrix(device, dtype).unsqueeze(0)  # (1, 4, 4)

        # gsplat expects (C, N, ...) where C = number of cameras
        N = means.shape[0]
        means_ = means.unsqueeze(0)        # (1, N, 3)
        scales_ = scales.unsqueeze(0)      # (1, N, 3)
        quats_ = rotations.unsqueeze(0)    # (1, N, 4)
        opacs_ = opacities.squeeze(-1).unsqueeze(0)   # (1, N)
        colors_ = colors.unsqueeze(0)      # (1, N, 3)

        renders, alphas, meta = rasterization(
            means=means_,
            quats=quats_,
            scales=scales_,
            opacities=opacs_,
            colors=colors_,
            viewmats=viewmat,
            Ks=torch.tensor(
                [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
                device=device, dtype=dtype,
            ).unsqueeze(0),
            width=self.width,
            height=self.height,
            near_plane=self.near,
            far_plane=self.far,
            backgrounds=self.bg.unsqueeze(0).to(dtype),
            render_mode='RGB+D' if self.render_depth else 'RGB',
        )

        # renders: (1, H, W, 3 or 4)
        img = renders[0]                   # (H, W, 3 or 4)
        alpha = alphas[0, ..., None]       # (H, W, 1)

        if self.render_depth:
            rgb = img[..., :3]
            depth = img[..., 3:4]
        else:
            rgb = img
            depth = torch.zeros(*img.shape[:2], 1, device=device, dtype=dtype)

        return {'rgb': rgb, 'depth': depth, 'alpha': alpha}


# ---------------------------------------------------------------------------
# Rendering loss
# ---------------------------------------------------------------------------

class RGBDLoss(nn.Module):
    """Multi-view RGBD reconstruction loss for 3DGS.

    Loss = L1(rgb) + lambda_ssim*(1-SSIM(rgb)) + lambda_d*L1(depth)[valid]
         + lambda_scale * mean(log_scale)
         + lambda_opa * mean(-log(opacity + eps))

    Args:
        lambda_ssim  : weight for SSIM term (default 0.2).
        lambda_d     : weight for depth L1 term (default 0.5).
        lambda_scale : regulariser on log-scale (default 0.01).
        lambda_opa   : regulariser pulling opacities away from 0 (default 0.01).
        eps          : numerical epsilon for opacity log (default 1e-6).
    """

    def __init__(
        self,
        lambda_ssim: float = 0.2,
        lambda_d: float = 0.5,
        lambda_scale: float = 0.01,
        lambda_opa: float = 0.01,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.lambda_d = lambda_d
        self.lambda_scale = lambda_scale
        self.lambda_opa = lambda_opa
        self.eps = eps

    def forward(
        self,
        pred_rgb: torch.Tensor,         # (H, W, 3) or (B, H, W, 3)
        gt_rgb: torch.Tensor,
        pred_depth: torch.Tensor,       # (H, W, 1) or (B, H, W, 1)
        gt_depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,  # same shape as depth, bool
        scales: Optional[torch.Tensor] = None,      # (N, 3) or (B, N, 3)
        opacities: Optional[torch.Tensor] = None,   # (N, 1) or (B, N, 1)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined loss and return (total, component_dict)."""

        # Bring to (B, C, H, W) for SSIM if needed
        if pred_rgb.dim() == 3:
            pred_rgb = pred_rgb.unsqueeze(0)
            gt_rgb = gt_rgb.unsqueeze(0)
            pred_depth = pred_depth.unsqueeze(0)
            gt_depth = gt_depth.unsqueeze(0)
            if valid_mask is not None:
                valid_mask = valid_mask.unsqueeze(0)

        # Permute (B, H, W, C) → (B, C, H, W)
        pred_rgb_nchw = pred_rgb.permute(0, 3, 1, 2).contiguous()
        gt_rgb_nchw = gt_rgb.permute(0, 3, 1, 2).contiguous()

        # ---- RGB losses ----
        loss_l1 = F.l1_loss(pred_rgb_nchw, gt_rgb_nchw)
        loss_ssim = 1.0 - _ssim(pred_rgb_nchw, gt_rgb_nchw)

        # ---- Depth loss ----
        if valid_mask is None:
            valid_mask = gt_depth > 0
        depth_l1 = F.l1_loss(pred_depth[valid_mask], gt_depth[valid_mask])
        loss_depth = depth_l1 if valid_mask.any() else pred_depth.new_zeros(1).squeeze()

        # ---- Regularisers ----
        loss_scale = scales.mean() if scales is not None else pred_rgb.new_zeros(1).squeeze()
        loss_opa = (-torch.log(opacities + self.eps)).mean() if opacities is not None \
            else pred_rgb.new_zeros(1).squeeze()

        total = (
            loss_l1
            + self.lambda_ssim * loss_ssim
            + self.lambda_d * loss_depth
            + self.lambda_scale * loss_scale
            + self.lambda_opa * loss_opa
        )

        components = {
            'l1': loss_l1,
            'ssim': loss_ssim,
            'depth': loss_depth,
            'scale_reg': loss_scale,
            'opa_reg': loss_opa,
            'total': total,
        }
        return total, components


# ---------------------------------------------------------------------------
# Convenience: build a fixed orbit camera set
# ---------------------------------------------------------------------------

def build_orbit_cameras(
    num_views: int = 8,
    elevation_deg: float = 20.0,
    radius: float = 2.5,
    device: str = 'cpu',
) -> list:
    """Return a list of (4, 4) camera-to-world matrices evenly spaced in azimuth."""
    azimuths = [360.0 * i / num_views for i in range(num_views)]
    return [orbit_c2w(elevation_deg, az, radius=radius, device=device) for az in azimuths]
