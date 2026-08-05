"""Flow loss helpers (from UNITE models/unite.py)."""

from __future__ import annotations

from typing import Dict, Optional

import torch


FLOW_LOSS_TYPES = ("velocity", "x_start")


def compute_flow_loss(
    flow_dict: Dict[str, torch.Tensor],
    *,
    train_eps: float = 5e-2,
    latent_norm=None,
    loss_type: str = "velocity",
) -> Dict[str, torch.Tensor]:
    """Flow loss from the denoised latent prediction.

    ``velocity`` is the UNITE form; because the velocity divides by ``1 - t``
    (clamped at ``train_eps``) it up-weights near-clean timesteps by up to
    ``1/train_eps**2`` (400x at the default), which makes the per-step loss very
    spiky. ``x_start`` regresses the clean latent directly and is unweighted.

    Returns both the training loss and the unweighted x-start MSE for logging.
    """
    if loss_type not in FLOW_LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {FLOW_LOSS_TYPES}, got {loss_type}")
    x1 = flow_dict["x1"]
    xt = flow_dict["xt"]
    t = flow_dict["sampled_t"][:, None, None]
    x_pred = flow_dict["model_output"]
    if latent_norm is not None:
        x_pred = latent_norm(x_pred)

    x_start_mse = ((x_pred - x1) ** 2).mean(dim=(1, 2)).mean()
    denom = (1 - t).clamp_min(train_eps)
    v_gt = (x1 - xt) / denom
    v_pred = (x_pred - xt) / denom
    velocity_mse = ((v_gt - v_pred) ** 2).mean(dim=(1, 2)).mean()

    loss = velocity_mse if loss_type == "velocity" else x_start_mse
    return {
        "loss": loss,
        "velocity_mse": velocity_mse.detach(),
        "x_start_mse": x_start_mse.detach(),
    }


def noising_latents(
    transport,
    x: torch.Tensor,
    *,
    noising_t_start: float = 0.7,
    sampling_prob: float = 0.5,
) -> torch.Tensor:
    """Partial corruption of clean latents before decode (UNITE recon robustness)."""
    b = x.shape[0]
    mask = (torch.rand(b, device=x.device) < sampling_prob).view(b, *([1] * (x.dim() - 1)))
    t, x0, x1 = transport.sample(x, sp_timesteps=[noising_t_start, 1.0])
    _, x_t, _ = transport.path_sampler.plan(t, x0, x1)
    return torch.where(mask, x_t, x)
