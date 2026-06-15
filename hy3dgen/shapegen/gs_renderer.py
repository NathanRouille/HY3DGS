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

# INRIA 3DGS SH basis constant (DC coefficient scaling).
SH_C0 = 0.28209479177387814


def num_sh_bases(sh_degree: int) -> int:
    """Number of real SH bases for the given active degree (inclusive)."""
    return (sh_degree + 1) ** 2


def rgb_to_sh_dc(rgb: torch.Tensor) -> torch.Tensor:
    """Map linear RGB in [0, 1] to degree-0 SH coefficients (per channel)."""
    return (rgb - 0.5) / SH_C0


def sh_dc_to_rgb(f_dc: torch.Tensor) -> torch.Tensor:
    """Map SH DC coefficients back to linear RGB in [0, 1]."""
    return (f_dc * SH_C0 + 0.5).clamp(0.0, 1.0)


def pack_sh_coeffs(
    dc_rgb: torch.Tensor,
    sh_rest: Optional[torch.Tensor],
    sh_degree: int,
) -> torch.Tensor:
    """Build gsplat SH coefficient tensor from DC RGB and higher-order coeffs.

    Args:
        dc_rgb  : (..., 3) sigmoid RGB in [0, 1] for the DC band.
        sh_rest : (..., (K-1)*3) raw higher-order coeffs, or None when ``sh_degree==0``.
        sh_degree: active SH degree (0 = view-independent DC only).

    Returns:
        (..., K, 3) with K = (sh_degree + 1)^2.
    """
    K = num_sh_bases(sh_degree)
    f_dc = rgb_to_sh_dc(dc_rgb).unsqueeze(-2)  # (..., 1, 3)
    if sh_degree == 0:
        return f_dc
    if sh_rest is None:
        raise ValueError("sh_rest is required when sh_degree > 0")
    rest = sh_rest.reshape(*sh_rest.shape[:-1], K - 1, 3)
    return torch.cat([f_dc, rest], dim=-2)


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


def _compute_gt_edge_weight_map(
    gt_rgb: torch.Tensor,
    lambda_edge: float,
) -> torch.Tensor:
    """Per-pixel weights in ``[1, 1 + lambda_edge]`` from GT RGB image gradients.

    Args:
        gt_rgb: ``(B, H, W, 3)`` in ``[0, 1]``.
        lambda_edge: extra weight on high-gradient pixels (0 = uniform).

    Returns:
        ``(B, H, W, 1)`` weight map (detached).
    """
    if lambda_edge <= 0:
        return torch.ones(*gt_rgb.shape[:-1], 1, device=gt_rgb.device, dtype=gt_rgb.dtype)
    gt = gt_rgb.detach()
    gt_nchw = gt.permute(0, 3, 1, 2).contiguous()
    gx = (gt_nchw[:, :, :, 1:] - gt_nchw[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    gy = (gt_nchw[:, :, 1:, :] - gt_nchw[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
    B, _, H, W = gt_nchw.shape
    edge = torch.zeros(B, 1, H, W, device=gt.device, dtype=gt.dtype)
    edge[:, :, :, 1:] += gx
    edge[:, :, 1:, :] += gy
    emax = edge.flatten(1).amax(dim=1).view(B, 1, 1, 1).clamp_min(1e-6)
    edge_norm = edge / emax
    return (1.0 + lambda_edge * edge_norm).permute(0, 2, 3, 1)


def _foreground_weighted_rgb_loss(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    valid_mask: torch.Tensor,
    rgb_loss_type: str,
    edge_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Foreground-only RGB loss with optional per-pixel edge weights."""
    if rgb_loss_type == "mse":
        per_pixel = (pred_rgb - gt_rgb).pow(2).sum(dim=-1, keepdim=True)
    else:
        per_pixel = (pred_rgb - gt_rgb).abs().sum(dim=-1, keepdim=True)
    fg = valid_mask.float()
    if fg.dim() == pred_rgb.dim() - 1:
        fg = fg.unsqueeze(-1)
    w = fg if edge_weight is None else fg * edge_weight
    denom = w.sum().clamp_min(1.0)
    return (per_pixel * w).sum() / denom


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

# v46 layout: indices 0–5 canonical; indices 6–45 = 5 elevation rows × 8 azimuth cols.
CANONICAL_VIEW_INDICES_V46: Tuple[int, ...] = tuple(range(6))


def v46_grid_view_index(row: int, col: int) -> int:
    """Global view index for grid row ``row`` (0–4) and column ``col`` (0–7)."""
    return 6 + row * 8 + col


def holdout_view_indices_v46() -> List[int]:
    """Elevation grid rows 1 and 2 (16 views).

    For 22-view training (rows 0 and 3 only), these views are never seen during
    training.  Fixed across experiments so canonical vs holdout comparisons use
    the same cameras.
    """
    return [v46_grid_view_index(r, c) for r in (1, 2) for c in range(8)]


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
        sh_degree: int = 1,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.sh_degree = int(sh_degree)
        if self.sh_degree < 0:
            raise ValueError(f"sh_degree must be >= 0, got {sh_degree}")
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
        colors: torch.Tensor,       # (N, 3) RGB if sh_degree==0, else (N, K, 3) SH coeffs
        c2w: torch.Tensor,          # (4, 4) camera-to-world
    ) -> Dict[str, torch.Tensor]:
        """Render a single scene.

        When ``sh_degree > 0``, ``colors`` must be SH coefficients with shape
        ``(N, (sh_degree+1)^2, 3)``.  When ``sh_degree == 0``, pass linear RGB
        ``(N, 3)`` or a single-band SH tensor ``(N, 1, 3)``.

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
        if self.sh_degree == 0:
            if colors.dim() == 3 and colors.shape[-2] == 1:
                colors_ = sh_dc_to_rgb(colors[..., 0, :])
            else:
                colors_ = colors
            sh_degree_arg = None
        else:
            K = num_sh_bases(self.sh_degree)
            if colors.dim() == 2:
                raise ValueError(
                    f"sh_degree={self.sh_degree} requires colors shape (N, {K}, 3), "
                    f"got (N, {colors.shape[-1]})"
                )
            if colors.shape[-2] < K:
                raise ValueError(
                    f"colors has {colors.shape[-2]} SH bases but sh_degree={self.sh_degree} "
                    f"needs at least {K}"
                )
            colors_ = colors[..., :K, :]
            sh_degree_arg = self.sh_degree
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
            sh_degree=sh_degree_arg,
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
# Anchor position utilities
# ---------------------------------------------------------------------------

def expand_anchor_positions(
    query_positions: torch.Tensor,
    num_gs_per_anchor: int,
) -> torch.Tensor:
    """Repeat each FPS anchor ``K`` times to align with per-anchor Gaussians.

    Args:
        query_positions: (B, L, 3) or (L, 3)
        num_gs_per_anchor: K Gaussians predicted per anchor.

    Returns:
        (B, L*K, 3) or (L*K, 3) with the same batching as the input.
    """
    K = num_gs_per_anchor
    if query_positions.dim() == 2:
        L = query_positions.shape[0]
        return (
            query_positions.unsqueeze(1)
            .expand(L, K, 3)
            .reshape(L * K, 3)
        )
    B, L, _ = query_positions.shape
    return (
        query_positions.unsqueeze(2)
        .expand(B, L, K, 3)
        .reshape(B, L * K, 3)
    )


def anchor_position_deltas(
    means: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    """Per-Gaussian displacement from anchor to predicted mean."""
    return means - anchors


def anchor_delta_loss(pos_deltas: torch.Tensor) -> torch.Tensor:
    """Mean squared L2 anchor offset penalty (AnchorSplat-style soft regulariser).

    AnchorSplat hard-constrains offsets with ``tanh(raw) * (10/128)`` at decode time;
    this loss term provides a differentiable soft alternative when no hard cap is set.
    """
    return (pos_deltas ** 2).sum(dim=-1).mean()


@torch.no_grad()
def compute_anchor_drift_metrics(
    means: torch.Tensor,
    anchors: torch.Tensor,
    drift_threshold: float = 0.1,
) -> Dict[str, float]:
    """Scalar drift statistics for eval (means vs FPS anchors)."""
    delta = anchor_position_deltas(means, anchors)
    drift_l2 = delta.norm(dim=-1)
    return {
        "mean_drift_l2": float(drift_l2.mean().item()),
        "max_drift_l2": float(drift_l2.max().item()),
        "p95_drift_l2": float(torch.quantile(drift_l2, 0.95).item()),
        "mean_delta_l2_sq": float((delta ** 2).sum(dim=-1).mean().item()),
        "frac_drift_gt_thresh": float((drift_l2 > drift_threshold).float().mean().item()),
        "drift_threshold": drift_threshold,
    }


# ---------------------------------------------------------------------------
# Rendering loss
# ---------------------------------------------------------------------------

class RGBDLoss(nn.Module):
    """Multi-view RGBD reconstruction loss for 3DGS.

    Loss = lambda_rgb * rgb_loss(pred_rgb[fg], gt_rgb[fg])  [foreground-only; edge-weighted L1/MSE]
         + lambda_ssim*(1-SSIM(pred_rgb, gt_rgb))     [full-image; flat white bg ≈ 0 gradient]
         + lambda_lpips*LPIPS(pred_rgb, gt_rgb)        [full-image; VGG ignores flat bg]
         + lambda_d * L1(pred_depth[fg], gt_depth[fg]) [foreground-only; avoids depth halos]
         + lambda_alpha * (L1_fg(alpha→1) + alpha_bg_weight*L1_bg(alpha→0))
         + lambda_scale * mean(s0*s1*s2)               [AnchorSplat volume penalty]
         + lambda_opa   * mean(1 - opacity)            [AnchorSplat opacity penalty]
         + lambda_delta * mean(||pos_delta||^2)       [anchor offset L2 penalty]

    The RGB and depth losses are computed on foreground pixels only (valid_mask = gt_depth>0)
    to prevent the white background from dominating photometric gradients and washing out
    dark object colours.  SSIM and LPIPS remain full-image so that holes and voids in the
    object (e.g. chair grid backs) still generate structural gradients; the flat white
    background contributes negligible signal to both.

    Background suppression is handled by the alpha supervision term: foreground pixels
    push pred_alpha→1 with weight 1; background pixels push pred_alpha→0 with weight
    alpha_bg_weight (default 5).  Combined with the opacity regulariser (all Gaussians
    toward opacity=1), the optimiser can only satisfy alpha=0 on background by not placing
    Gaussians there, effectively confining Gaussians to the object.

    Args:
        lambda_ssim       : weight for SSIM term (default 0.2).
        lambda_lpips      : weight for LPIPS (full-image, default 0.1).
        lambda_d          : weight for foreground depth L1 (default 1.0).
        lambda_alpha      : weight for alpha supervision fg+bg (default 0.05).
        alpha_bg_weight   : extra multiplier on background alpha L1 (default 5.0).
        lambda_scale      : weight for AnchorSplat volume penalty mean(s0*s1*s2) (default 0.01).
        lambda_opa        : weight for AnchorSplat opacity penalty mean(1 - opacity) (default 0.01).
        lambda_delta      : weight for mean squared anchor offset ||means - anchor||^2 (default 0).
        rgb_loss_type     : ``'l1'`` or ``'mse'`` for the photometric RGB term.
        lambda_rgb        : global multiplier on the foreground RGB term (default 1.0).
        lambda_edge       : extra per-pixel weight on high-|∇GT| fg pixels (default 4.0;
                            0 = uniform foreground L1/MSE).
        min_valid_ratio   : if foreground pixel fraction falls below this, depth terms
                            are zeroed for that view (degenerate camera / bad mesh).
    """

    def __init__(
        self,
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.1,
        lambda_d: float = 1.0,
        lambda_alpha: float = 0.05,
        lambda_scale: float = 0.01,
        lambda_opa: float = 0.01,
        lambda_delta: float = 0.0,
        rgb_loss_type: str = "mse",
        lambda_rgb: float = 1.0,
        lambda_edge: float = 4.0,
        min_valid_ratio: float = 0.02,
        alpha_bg_weight: float = 5.0,
    ):
        super().__init__()
        if rgb_loss_type not in ("l1", "mse"):
            raise ValueError(f"rgb_loss_type must be 'l1' or 'mse', got {rgb_loss_type!r}")
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_d = lambda_d
        self._lpips_net = None
        self.lambda_alpha = lambda_alpha
        self.lambda_scale = lambda_scale
        self.lambda_opa = lambda_opa
        self.lambda_delta = lambda_delta
        self.rgb_loss_type = rgb_loss_type
        self.lambda_rgb = lambda_rgb
        self.lambda_edge = lambda_edge
        self.min_valid_ratio = min_valid_ratio
        self.alpha_bg_weight = alpha_bg_weight

    def _compute_lpips(
        self,
        pred_nchw: torch.Tensor,
        gt_nchw: torch.Tensor,
    ) -> torch.Tensor:
        """LPIPS on foreground-masked RGB (NCHW, values in [0, 1])."""
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

        valid_ratio = valid_mask.float().mean()
        depth_scale = (valid_ratio >= self.min_valid_ratio).to(pred_depth.dtype)

        # ---- Foreground-only RGB photometric (optional GT-gradient edge weights) ----
        # Masking prevents the white background (≈85% of pixels) from dominating
        # gradients and pulling dark object colours toward white.  Edge weights
        # upweight material boundaries (high |∇GT|) to reduce colour bleeding.
        edge_weight = _compute_gt_edge_weight_map(gt_rgb, self.lambda_edge)
        if valid_mask.any():
            loss_rgb = _foreground_weighted_rgb_loss(
                pred_rgb,
                gt_rgb,
                valid_mask,
                self.rgb_loss_type,
                edge_weight=edge_weight,
            )
        else:
            loss_rgb = pred_rgb.new_zeros(())

        # ---- Full-image SSIM + LPIPS ----
        # Kept full-image so that holes and voids in the object (e.g. chair grid
        # backs) still produce structural gradients; the flat white background
        # contributes negligible signal to both metrics.
        pred_rgb_nchw = pred_rgb.permute(0, 3, 1, 2).contiguous()
        gt_rgb_nchw = gt_rgb.permute(0, 3, 1, 2).contiguous()
        loss_ssim = 1.0 - _ssim(pred_rgb_nchw, gt_rgb_nchw)
        loss_lpips = self._compute_lpips(pred_rgb_nchw, gt_rgb_nchw)

        # ---- Foreground-only depth L1 ----
        # Foreground-only avoids depth halos: edge Gaussians with soft 2D footprints
        # that bleed into adjacent background pixels are no longer penalised for
        # having non-zero depth there.  Background suppression is handled instead
        # by the alpha supervision term below.
        fg_d = valid_mask.expand_as(pred_depth)
        if fg_d.any():
            loss_depth = depth_scale * F.l1_loss(pred_depth[fg_d], gt_depth[fg_d])
        else:
            loss_depth = pred_depth.new_zeros(())

        # ---- Alpha supervision with background weighting ----
        # Foreground pixels: pred_alpha → 1 (weight 1).
        # Background pixels: pred_alpha → 0 (weight alpha_bg_weight).
        # Combined with the opacity regulariser (Gaussians → opacity=1), the
        # optimizer can only satisfy alpha=0 on background by not placing Gaussians
        # there, confining Gaussians to the object surface.
        loss_alpha = pred_depth.new_zeros(())
        if pred_alpha is not None and self.lambda_alpha > 0:
            gt_alpha = valid_mask.float().expand_as(pred_alpha)
            fg_a = valid_mask.expand_as(pred_alpha)
            bg_a = ~fg_a
            fg_alpha_loss = (
                F.l1_loss(pred_alpha[fg_a], gt_alpha[fg_a])
                if fg_a.any() else pred_alpha.new_zeros(())
            )
            bg_alpha_loss = (
                F.l1_loss(pred_alpha[bg_a], gt_alpha[bg_a])
                if bg_a.any() else pred_alpha.new_zeros(())
            )
            loss_alpha = fg_alpha_loss + self.alpha_bg_weight * bg_alpha_loss

        # ---- AnchorSplat 3D regularisers ----
        # Volume penalty: penalise the mean physical volume of each Gaussian.
        # scales are log(physical_scale); sum over dims gives log-volume.
        # Clamp before exp for numerical safety (scale≫e^10 is already degenerate).
        if scales is not None:
            log_vol = scales.view(-1, 3).sum(dim=-1)           # log(s0*s1*s2) per splat
            loss_scale = torch.exp(log_vol.clamp(max=10.0)).mean()
        else:
            loss_scale = pred_rgb.new_zeros(())

        # Opacity penalty: pull each Gaussian toward fully opaque (opacity → 1).
        # opacities are sigmoid outputs in [0, 1].
        if opacities is not None:
            loss_opa = (1.0 - opacities.view(-1)).abs().mean()
        else:
            loss_opa = pred_rgb.new_zeros(())

        loss_rgb_weighted = self.lambda_rgb * loss_rgb
        total = (
            loss_rgb_weighted
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
            'valid_ratio': valid_ratio.detach(),
        }
        return total, components

    def weighted_terms_for_grad_norm(
        self,
        components: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return per-term scalar losses as they enter ``total`` (for grad-norm logging)."""
        rgb_key = self.rgb_loss_type
        terms: Dict[str, torch.Tensor] = {
            rgb_key: self.lambda_rgb * components[rgb_key],
            "ssim": self.lambda_ssim * components["ssim"],
            "lpips": self.lambda_lpips * components["lpips"],
            "depth": self.lambda_d * components["depth"],
            "alpha_sup": self.lambda_alpha * components["alpha_sup"],
            "scale_reg": self.lambda_scale * components["scale_reg"],
            "opa_reg": self.lambda_opa * components["opa_reg"],
        }
        return terms

    def anchor_delta_regularizer(
        self,
        pos_deltas: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Anchor offset penalty applied once per training step (not per view)."""
        loss_delta = anchor_delta_loss(pos_deltas)
        weighted = self.lambda_delta * loss_delta
        return weighted, {
            "delta_reg": loss_delta.detach(),
            "total": weighted.detach(),
        }


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
