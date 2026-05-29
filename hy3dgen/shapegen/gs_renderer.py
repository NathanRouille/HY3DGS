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

import hashlib
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
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

    # Look-at construction (OpenGL convention, Y-up, right-handed).
    # Standard gluLookAt-style basis: side = cross(forward, up).
    # The previous formulation used cross(up, forward), which produced a
    # left-handed (det = -1) c2w. pyrender silently rendered empty depth
    # for half of all azimuths under that pose, which in turn caused
    # opacity collapse during training.
    forward = -F.normalize(pos, dim=0)
    world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
    if abs(forward[1].item()) > 0.99:
        world_up = torch.tensor([0.0, 0.0, 1.0], device=device)
    right = F.normalize(torch.cross(forward, world_up, dim=0), dim=0)
    up = torch.cross(right, forward, dim=0)

    c2w = torch.eye(4, device=device)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = pos
    return c2w


# ---------------------------------------------------------------------------
# 46-view staggered layout (canonical + jittered azimuth/elevation grid)
# ---------------------------------------------------------------------------

TOTAL_V46_STAGGER: int = 46
GT_CACHE_TAG_V46: str = "v46_fp16_norm"

# Allowed training subsample counts (must match presets in train_gs_ae.snap_train_views_v46)
VIEW46_TRAIN_ALLOWED: Tuple[int, ...] = (6, 14, 22, 30, 38, 46)


def mesh_seed_from_path(mesh_path: str, extra: int = 0) -> int:
    """Deterministic RNG seed per mesh path (reproducible cameras / jitter)."""
    h = hashlib.md5(os.path.abspath(mesh_path).encode("utf-8")).hexdigest()
    return (int(h[:8], 16) + extra) & 0x7FFFFFFF


def build_view46_elev_az_pairs(mesh_path: str) -> List[Tuple[float, float]]:
    """46 (elevation, azimuth) pairs in degrees for one mesh.

    * 6 canonical: top/bottom/front/back/left/right.
    * 40 staggered: 5 base elevations × 8 base azimuths (45° steps), with
      per-mesh elevation jitter ±2.5° on each base elevation, and one random
      azimuth offset in [0°, 44°) per elevation row (brick pattern).
    """
    rng = np.random.default_rng(mesh_seed_from_path(mesh_path))
    pairs: List[Tuple[float, float]] = []

    # Canonical (no jitter on these six)
    pairs.append((89.9, 0.0))    # top
    pairs.append((-89.9, 0.0))   # bottom
    pairs.append((0.0, 0.0))     # front
    pairs.append((0.0, 180.0))   # back
    pairs.append((0.0, 270.0))   # left
    pairs.append((0.0, 90.0))    # right

    base_elevs = [-30, 0.0, 20.0, 40.0, 60.0]
    base_azs = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

    for _el_base in base_elevs:
        elev_j = float(rng.uniform(-2.5, 2.5))
        az_row_off = float(rng.uniform(0.0, 44.0))
        el = _el_base + elev_j
        for az_b in base_azs:
            az = (az_b + az_row_off) % 360.0
            pairs.append((el, az))

    assert len(pairs) == TOTAL_V46_STAGGER, len(pairs)
    return pairs


def build_view46_c2ws(
    mesh_path: str,
    radius: float,
    fov_deg: float,
    device: str = "cpu",
) -> Tuple[List[torch.Tensor], List[Dict[str, float]]]:
    """Return c2w list and per-view parameter dicts (for caching / training)."""
    pairs = build_view46_elev_az_pairs(mesh_path)
    c2ws: List[torch.Tensor] = []
    params: List[Dict[str, float]] = []
    for el, az in pairs:
        c2w = orbit_c2w(el, az, radius=radius, device=device)
        c2ws.append(c2w)
        params.append(
            {
                "elevation_deg": float(el),
                "azimuth_deg": float(az),
                "radius": float(radius),
                "fov_deg": float(fov_deg),
            }
        )
    return c2ws, params


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
        alpha = alphas[0].reshape(self.height, self.width, 1)

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
    """Multi-view RGBD reconstruction loss for object FF-3DGS.

    Loss = rgb_loss(pred_rgb, gt_rgb)                  [full image; MSE or L1]
         + lambda_ssim  * (1 - SSIM(pred_rgb, gt_rgb)) [full image]
         + lambda_lpips * LPIPS(pred_rgb, gt_rgb) * lpips_ramp(step) [full image]
         + lambda_d     * L1(pred_depth, gt_depth)[fg]  [FG-masked, see plan §2a]
         + lambda_alpha * L1(pred_alpha, gt_alpha)      [full image, gt_alpha = valid_mask]
         + lambda_scale * mean(s0*s1*s2)                [AnchorSplat volume penalty]
         + lambda_opa   * mean(H(opacity))              [binary entropy → 0 or 1]

    Why this recipe:
        - Full-image RGB/SSIM/LPIPS on white background matches the unanimous
          object FF-3DGS recipe (LGM, GRM, AGG, TriplaneGaussian).
        - FG-masked depth removes the BG-depth-to-zero pressure that otherwise
          forces edge Gaussians to shrink/become transparent (vanishing edges).
        - Binary entropy on opacity peaks at 0.5 and pushes each Gaussian toward
          either 0 (carves holes) or 1 (opaque thin features) — see GSurf 2024
          and NGS Oct 2025 false-transparency analysis.

    Args:
        lambda_ssim         : SSIM weight (default 0.2).
        lambda_lpips        : LPIPS weight (default 0.1; ramped in by lpips_warmup_steps).
        lambda_d            : FG depth L1 weight (default 1.0).
        lambda_alpha        : full-image alpha L1 weight (default 0.05).
        lambda_scale        : AnchorSplat volume penalty mean(s0*s1*s2) weight (default 0.01).
        lambda_opa          : binary opacity entropy weight (default 0.05).
        rgb_loss_type       : 'mse' (default; LGM/GRM/GS-LRM) or 'l1'.
        lpips_warmup_steps  : linear LPIPS ramp from 0→1 over this many steps (default 5000;
                              set 0 to disable). Mitigates "perceptual mean" texture washout.
    """

    def __init__(
        self,
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.1,
        lambda_d: float = 1.0,
        lambda_alpha: float = 0.05,
        lambda_scale: float = 0.01,
        lambda_opa: float = 0.05,
        rgb_loss_type: str = 'mse',
        lpips_warmup_steps: int = 5000,
    ):
        super().__init__()
        if rgb_loss_type not in ('mse', 'l1'):
            raise ValueError(f"rgb_loss_type must be 'mse' or 'l1', got {rgb_loss_type!r}")
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_d = lambda_d
        self.lambda_alpha = lambda_alpha
        self.lambda_scale = lambda_scale
        self.lambda_opa = lambda_opa
        self.rgb_loss_type = rgb_loss_type
        self.lpips_warmup_steps = int(lpips_warmup_steps)
        self._lpips_net = None

    def _compute_lpips(
        self,
        pred_nchw: torch.Tensor,
        gt_nchw: torch.Tensor,
    ) -> torch.Tensor:
        """LPIPS on full-image RGB (NCHW, values in [0, 1])."""
        if self.lambda_lpips <= 0:
            return pred_nchw.new_zeros(())
        if self._lpips_net is None:
            try:
                import lpips
            except ImportError as e:
                raise ImportError(
                    "lpips is required when lambda_lpips > 0. Install with: pip install lpips"
                ) from e
            net = lpips.LPIPS(net='vgg', verbose=False)
            net.eval()
            for p in net.parameters():
                p.requires_grad = False
            self._lpips_net = net
        net = self._lpips_net.to(device=pred_nchw.device, dtype=pred_nchw.dtype)
        return net(pred_nchw, gt_nchw).mean()

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
        step: Optional[int] = None,                 # global step, for LPIPS warmup ramp
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

        # ---- Full-image RGB photometric loss (MSE by default, L1 optional) ----
        if self.rgb_loss_type == 'mse':
            loss_rgb = F.mse_loss(pred_rgb, gt_rgb)
        else:
            loss_rgb = F.l1_loss(pred_rgb, gt_rgb)

        # ---- Full-image SSIM + LPIPS ----
        pred_rgb_nchw = pred_rgb.permute(0, 3, 1, 2).contiguous()
        gt_rgb_nchw = gt_rgb.permute(0, 3, 1, 2).contiguous()
        loss_ssim = 1.0 - _ssim(pred_rgb_nchw, gt_rgb_nchw)
        loss_lpips_raw = self._compute_lpips(pred_rgb_nchw, gt_rgb_nchw)
        if self.lpips_warmup_steps > 0 and step is not None:
            ramp = min(1.0, max(0.0, float(step) / float(self.lpips_warmup_steps)))
        else:
            ramp = 1.0
        loss_lpips = loss_lpips_raw * ramp

        # ---- FG-masked depth L1 (plan §2a: removes BG-depth-to-zero pressure on edge Gaussians) ----
        if valid_mask.any():
            loss_depth = F.l1_loss(pred_depth[valid_mask], gt_depth[valid_mask])
        else:
            loss_depth = pred_depth.new_zeros(())

        # ---- Full-image alpha L1 (gt_alpha = valid_mask as float) ----
        loss_alpha = pred_depth.new_zeros(())
        if pred_alpha is not None and self.lambda_alpha > 0:
            gt_alpha = valid_mask.float().expand_as(pred_alpha)
            loss_alpha = F.l1_loss(pred_alpha, gt_alpha)

        # ---- AnchorSplat volume penalty ----
        # scales are log(physical_scale); sum over dims gives log-volume.
        # Clamp before exp for numerical safety (scale≫e^10 is already degenerate).
        if scales is not None:
            log_vol = scales.view(-1, 3).sum(dim=-1)           # log(s0*s1*s2) per splat
            loss_scale = torch.exp(log_vol.clamp(max=10.0)).mean()
        else:
            loss_scale = pred_rgb.new_zeros(())

        # ---- Binary opacity entropy: pushes opacity to 0 or 1, not 0.5 ----
        # H(p) = -p log p - (1-p) log(1-p); peaks at p=0.5, zero at p∈{0,1}.
        # Replaces AnchorSplat's monotone (1-p) penalty which let Gaussians sit
        # at 0.5 ("vanishing colors" / semi-transparent edges).
        if opacities is not None:
            eps = 1e-6
            p = opacities.view(-1).clamp(eps, 1.0 - eps)
            entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
            loss_opa = entropy.mean()
        else:
            loss_opa = pred_rgb.new_zeros(())

        total = (
            loss_rgb
            + self.lambda_ssim * loss_ssim
            + self.lambda_lpips * loss_lpips
            + self.lambda_d * loss_depth
            + self.lambda_alpha * loss_alpha
            + self.lambda_scale * loss_scale
            + self.lambda_opa * loss_opa
        )

        components = {
            self.rgb_loss_type: loss_rgb,
            'ssim': loss_ssim,
            'lpips': loss_lpips,
            'depth': loss_depth,
            'alpha_sup': loss_alpha,
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
