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
from typing import Dict, List, Optional, Tuple

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
        # gsplat expects a world-to-camera matrix in OpenCV-like camera coordinates.
        # Our orbit/pyrender poses are OpenGL-style c2w, so convert GL->CV first.
        gl_to_cv = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, -1.0, 0.0, 0.0],
             [0.0, 0.0, -1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        c2w_cv = c2w.to(dtype) @ gl_to_cv
        viewmat = torch.linalg.inv(c2w_cv).unsqueeze(0)  # (1, 4, 4)

        # Keep gaussian tensors unbatched: (N, ...)
        # gsplat treats camera count via viewmats/Ks, not via gaussian leading dim.
        means_ = means                      # (N, 3)
        scales_ = scales                    # (N, 3)
        quats_ = rotations                  # (N, 4)
        opacs_ = opacities.squeeze(-1)      # (N,)
        colors_ = colors                    # (N, 3)
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
            packed=False,
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

    Loss = fg_weight*L1(rgb)[fg] + (1-fg_weight)*L1(rgb)[bg]
         + lambda_ssim*(1-SSIM(rgb, foreground-masked))
         + lambda_d*L1(depth)[valid]       (skipped if valid_ratio < min_valid_ratio)
         + lambda_alpha*(1-pred_alpha)^2   (on foreground pixels; prevents collapse)
         + lambda_scale * (log_scale - target_log_scale)^2
         + lambda_opa   * (opacity    - target_opacity)^2

    SSIM is computed with background pixels replaced by GT background in the
    prediction, so the network receives no SSIM gradient from background regions.

    Args:
        lambda_ssim       : weight for SSIM term (default 0.2).
        lambda_d          : weight for depth L1 term (default 1.0).
        lambda_alpha      : weight for alpha supervision on fg pixels (default 0.05).
                            Creates a direct gradient signal preventing opacity collapse.
        lambda_scale      : regulariser on log-scale (default 0.01).
        lambda_opa        : regulariser pulling opacities toward target (default 0.01).
        target_log_scale  : target for log-scale regulariser (default -3.0, scale≈0.05).
        target_opacity    : target for opacity regulariser (default 0.5).
        fg_weight         : fraction of RGB L1 allocated to foreground pixels (default 0.75).
                            Ensures the model focuses on object appearance, not background.
        min_valid_ratio   : if foreground pixel fraction falls below this, the depth
                            loss is disabled for that view (degenerate camera / bad mesh).
    """

    def __init__(
        self,
        lambda_ssim: float = 0.2,
        lambda_d: float = 1.0,
        lambda_alpha: float = 0.05,
        lambda_scale: float = 0.01,
        lambda_opa: float = 0.01,
        target_log_scale: float = -3.0,
        target_opacity: float = 0.5,
        fg_weight: float = 0.75,
        min_valid_ratio: float = 0.02,
    ):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.lambda_d = lambda_d
        self.lambda_alpha = lambda_alpha
        self.lambda_scale = lambda_scale
        self.lambda_opa = lambda_opa
        self.target_log_scale = target_log_scale
        self.target_opacity = target_opacity
        self.fg_weight = fg_weight
        self.min_valid_ratio = min_valid_ratio

    def forward(
        self,
        pred_rgb: torch.Tensor,         # (H, W, 3) or (B, H, W, 3)
        gt_rgb: torch.Tensor,
        pred_depth: torch.Tensor,       # (H, W, 1) or (B, H, W, 1)
        gt_depth: torch.Tensor,
        pred_alpha: Optional[torch.Tensor] = None,  # same spatial shape as depth, float [0,1]
        valid_mask: Optional[torch.Tensor] = None,  # same shape as depth, bool
        scales: Optional[torch.Tensor] = None,      # (N, 3) or (B, N, 3) log-scales
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
            if pred_alpha is not None:
                pred_alpha = pred_alpha.unsqueeze(0)

        if valid_mask is None:
            valid_mask = gt_depth > 0

        valid_ratio = float(valid_mask.float().mean().item())

        # ---- Foreground-weighted RGB L1 ----
        # Use a pixel-wise weight map so the gradient scale is preserved
        # regardless of what fraction of pixels are foreground.
        # Weight is normalised so mean(weight) == 1 (same total magnitude as unweighted L1).
        fg_float = valid_mask.float().expand_as(pred_rgb)  # (B, H, W, 3)
        weight = fg_float * self.fg_weight + (1.0 - fg_float) * (1.0 - self.fg_weight)
        n_fg_px = fg_float.sum()
        n_bg_px = (1.0 - fg_float).sum()
        if n_fg_px > 0 and n_bg_px > 0:
            per_px_l1 = (pred_rgb - gt_rgb).abs()
            loss_l1 = (per_px_l1 * weight).sum() / weight.sum()
            l1_fg = per_px_l1[fg_float.bool()].mean().detach()
            l1_bg = per_px_l1[~fg_float.bool()].mean().detach()
        else:
            loss_l1 = F.l1_loss(pred_rgb, gt_rgb)
            l1_fg = loss_l1.detach()
            l1_bg = loss_l1.detach()

        # ---- SSIM (masked to foreground) ----
        # Replace background pixels in the prediction with the GT background value.
        # This zeroes out SSIM gradients from background regions so the network only
        # receives structural similarity feedback on the actual object pixels.
        fg_mask_hw1 = valid_mask.float()                             # (B, H, W, 1)
        pred_rgb_fg = pred_rgb * fg_mask_hw1 + gt_rgb * (1.0 - fg_mask_hw1)
        pred_rgb_nchw = pred_rgb_fg.permute(0, 3, 1, 2).contiguous()
        gt_rgb_nchw = gt_rgb.permute(0, 3, 1, 2).contiguous()
        loss_ssim = 1.0 - _ssim(pred_rgb_nchw, gt_rgb_nchw)

        # ---- Depth loss (skipped for degenerate views) ----
        if valid_ratio >= self.min_valid_ratio:
            depth_l1 = F.l1_loss(pred_depth[valid_mask], gt_depth[valid_mask])
            loss_depth = depth_l1
        else:
            loss_depth = pred_depth.new_zeros(())

        # ---- Alpha supervision on foreground pixels ----
        # Directly penalizes empty renders where the GT mesh is present.
        # This is the key signal that prevents opacity collapse.
        if pred_alpha is not None and valid_mask.any() and self.lambda_alpha > 0:
            fg_alpha_mask = valid_mask  # (B, H, W, 1)
            loss_alpha = ((1.0 - pred_alpha[fg_alpha_mask]) ** 2).mean()
        else:
            loss_alpha = pred_rgb.new_zeros(())

        # ---- Regularisers ----
        loss_scale = ((scales - self.target_log_scale) ** 2).mean() if scales is not None \
            else pred_rgb.new_zeros(())
        loss_opa = ((opacities - self.target_opacity) ** 2).mean() if opacities is not None \
            else pred_rgb.new_zeros(())

        total = (
            loss_l1
            + self.lambda_ssim * loss_ssim
            + self.lambda_d * loss_depth
            + self.lambda_alpha * loss_alpha
            + self.lambda_scale * loss_scale
            + self.lambda_opa * loss_opa
        )

        components = {
            'l1': loss_l1,
            'l1_fg': l1_fg.detach() if torch.is_tensor(l1_fg) else pred_rgb.new_zeros(()),
            'l1_bg': l1_bg.detach() if torch.is_tensor(l1_bg) else pred_rgb.new_zeros(()),
            'ssim': loss_ssim,
            'depth': loss_depth,
            'alpha_sup': loss_alpha,
            'scale_reg': loss_scale,
            'opa_reg': loss_opa,
            'total': total,
            'valid_ratio': pred_rgb.new_tensor(valid_ratio),
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
    azimuths_deg: Optional[List[float]] = None,
) -> list:
    """Return a list of (4, 4) camera-to-world matrices.

    If ``azimuths_deg`` is provided it must have exactly ``num_views`` entries;
    otherwise views are evenly spaced starting at azimuth 0.

    NOTE: The default even-spacing places a camera at azimuth=180° for 2 views,
    which is the exact rear of most Objaverse objects (no visible geometry).
    For 2-view training pass ``azimuths_deg=[0.0, 90.0]`` (front + right side).
    """
    if azimuths_deg is None:
        azimuths_deg = [360.0 * i / num_views for i in range(num_views)]
    if len(azimuths_deg) != num_views:
        raise ValueError(
            f"azimuths_deg has {len(azimuths_deg)} entries but num_views={num_views}"
        )
    return [orbit_c2w(elevation_deg, az, radius=radius, device=device) for az in azimuths_deg]
