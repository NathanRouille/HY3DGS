"""Shared image-quality metrics used by both training validation and the
standalone evaluation script. Keeps the two paths from drifting.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .gs_renderer import _ssim


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """PSNR (dB) on foreground pixels only.

    Args:
        pred / gt : (H, W, 3) float [0, 1]
        mask      : (H, W, 1) bool

    Returns:
        PSNR in dB. ``NaN`` if the foreground mask is empty.
    """
    pred_fg = pred[mask.expand_as(pred)]
    gt_fg = gt[mask.expand_as(gt)]
    if len(pred_fg) == 0:
        return float("nan")
    mse = F.mse_loss(pred_fg, gt_fg)
    if mse.item() < 1e-10:
        return 100.0
    return float(-10.0 * torch.log10(mse).item())


def compute_ssim_fg(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """SSIM [0, 1] on foreground-masked image (higher = better).

    Background pixels in ``pred`` are replaced with GT background before
    computing SSIM so the metric only reflects foreground reconstruction
    quality (the BG is always white so it would otherwise inflate scores).
    """
    mask_f = mask.float()
    pred_m = pred * mask_f + gt * (1.0 - mask_f)
    pred_nchw = pred_m.unsqueeze(0).permute(0, 3, 1, 2)
    gt_nchw = gt.unsqueeze(0).permute(0, 3, 1, 2)
    return float(_ssim(pred_nchw, gt_nchw).item())
