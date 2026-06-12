"""Shared image-quality metrics used by both training validation and the
standalone evaluation script. Keeps the two paths from drifting.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .gs_renderer import _ssim


def _to_nchw(img: torch.Tensor) -> torch.Tensor:
    """(H, W, 3) -> (1, 3, H, W)."""
    return img.unsqueeze(0).permute(0, 3, 1, 2)


def compute_psnr_fg(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """PSNR (dB) on foreground pixels only (``mask == True``).

    Ignores background pixels, so halos outside the GT silhouette do not
    affect this score.  Pair with :func:`compute_psnr_full` and
    :func:`compute_mean_alpha_bg` for a complete picture.
    """
    pred_fg = pred[mask.expand_as(pred)]
    gt_fg = gt[mask.expand_as(gt)]
    if len(pred_fg) == 0:
        return float("nan")
    mse = F.mse_loss(pred_fg, gt_fg)
    if mse.item() < 1e-10:
        return 100.0
    return float(-10.0 * torch.log10(mse).item())


def compute_psnr_full(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR (dB) on the full image (foreground + background).

    Penalises background blobs and halos that :func:`compute_psnr_fg` ignores.
    """
    mse = F.mse_loss(pred, gt)
    if mse.item() < 1e-10:
        return 100.0
    return float(-10.0 * torch.log10(mse).item())


def compute_ssim_full(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """SSIM [0, 1] on the full image (higher = better).

    No background replacement — predicted background is scored as-is, so
    blob/halos around the object reduce the score.
    """
    return float(_ssim(_to_nchw(pred), _to_nchw(gt)).item())


def compute_ssim_fg(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Alias for :func:`compute_ssim_full` (kept for backward compatibility).

    The ``mask`` argument is ignored.  Older versions replaced the predicted
    background with GT before scoring, which hid background artefacts; that
    behaviour has been removed.
    """
    del mask  # unused; retained so existing call sites need no signature change
    return compute_ssim_full(pred, gt)


def compute_mean_alpha(
    pred_alpha: torch.Tensor,
    mask: torch.Tensor,
    *,
    foreground: bool,
) -> float:
    """Mean composited alpha on foreground or background pixels.

    Args:
        pred_alpha: (H, W, 1) rendered alpha in [0, 1].
        mask:       (H, W, 1) bool, ``True`` where GT depth > 0.
        foreground: if ``True``, average over ``mask``; else over ``~mask``.

    Background mean should be near 0 when alpha supervision is working.
    Foreground mean should be near 1.
    """
    if foreground:
        sel = mask.expand_as(pred_alpha)
    else:
        sel = (~mask).expand_as(pred_alpha)
    if not sel.any():
        return float("nan")
    return float(pred_alpha[sel].float().mean().item())


def compute_mean_alpha_bg(pred_alpha: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean composited alpha on background pixels (``mask == False``)."""
    return compute_mean_alpha(pred_alpha, mask, foreground=False)


def compute_mean_alpha_fg(pred_alpha: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean composited alpha on foreground pixels (``mask == True``)."""
    return compute_mean_alpha(pred_alpha, mask, foreground=True)


# Backward-compatible alias used by older call sites.
compute_psnr = compute_psnr_fg
