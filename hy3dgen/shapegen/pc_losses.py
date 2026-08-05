"""Point-cloud reconstruction losses for ShapePCAE."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_dist2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared pairwise distances. a: [B,N,3], b: [B,M,3] -> [B,N,M]."""
    return torch.cdist(a, b, p=2).pow(2)


def chamfer_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    bidirectional: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric Chamfer (mean of squared NN distances).

    Returns:
        loss: scalar Chamfer
        idx_pred_to_tgt: [B, N] NN indices into target for each pred point
        idx_tgt_to_pred: [B, M] NN indices into pred for each target point
    """
    dist = pairwise_dist2(pred, target)  # [B, N, M]
    pred_to_tgt, idx_p2t = dist.min(dim=2)
    tgt_to_pred, idx_t2p = dist.min(dim=1)
    if bidirectional:
        loss = pred_to_tgt.mean() + tgt_to_pred.mean()
    else:
        loss = pred_to_tgt.mean()
    return loss, idx_p2t, idx_t2p


def _sinkhorn_transport_cost(
    C: torch.Tensor,
    *,
    epsilon: float,
    n_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Log-domain Sinkhorn; returns ``(<P, C> per batch, P)``.

    ``C`` is ``[B, N, M]`` squared costs. Uniform marginals ``1/N``, ``1/M``.
    """
    B, N, M = C.shape
    eps = max(float(epsilon), 1e-8)
    log_mu = C.new_full((B, N), -math.log(N))
    log_nu = C.new_full((B, M), -math.log(M))
    log_K = -C / eps

    log_u = torch.zeros(B, N, device=C.device, dtype=C.dtype)
    log_v = torch.zeros(B, M, device=C.device, dtype=C.dtype)
    for _ in range(int(n_iters)):
        log_u = log_mu - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
        log_v = log_nu - torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)

    log_P = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)
    P = torch.exp(log_P)
    # sum_{ij} P_ij = 1; scale by N → mean cost per source point
    cost = (P * C).sum(dim=(1, 2)) * float(N)
    return cost, P


def sinkhorn_matching_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 0.02,
    n_iters: int = 50,
    detach_plan: bool = True,
    debias: bool = True,
) -> torch.Tensor:
    """Entropic OT soft matching loss (optionally Sinkhorn divergence).

    Builds a doubly-stochastic plan ``P`` with uniform marginals. With
    ``debias=True`` (default), uses Sinkhorn divergence

        OT(pred, target) - 0.5 OT(pred, pred) - 0.5 OT(target, target)

    so identical clouds score ~0 (removes entropic bias). Detaching ``P``
    yields soft Hungarian-style regress-to-match targets and avoids noisy
    grads through the iterations — duplicates are still penalized because
    mass cannot pile on one FPS point.
    """
    if pred.shape[0] != target.shape[0] or pred.shape[-1] != 3 or target.shape[-1] != 3:
        raise ValueError(
            f"Expected pred/target [B,N,3] and [B,M,3], got {tuple(pred.shape)} / {tuple(target.shape)}"
        )
    B, N, _ = pred.shape
    M = target.shape[1]
    if N == 0 or M == 0:
        return pred.new_zeros(())

    C_xy = pairwise_dist2(pred, target)
    cost_xy, P_xy = _sinkhorn_transport_cost(C_xy, epsilon=epsilon, n_iters=n_iters)
    if detach_plan:
        # Regress positions under a frozen soft assignment (Hungarian-like).
        cost_xy = (P_xy.detach() * C_xy).sum(dim=(1, 2)) * float(N)

    if not debias:
        return cost_xy.mean()

    # Sinkhorn divergence correction (stop-grad on self-OT plans).
    C_xx = pairwise_dist2(pred, pred)
    C_yy = pairwise_dist2(target, target)
    cost_xx, P_xx = _sinkhorn_transport_cost(C_xx, epsilon=epsilon, n_iters=n_iters)
    cost_yy, P_yy = _sinkhorn_transport_cost(C_yy, epsilon=epsilon, n_iters=n_iters)
    if detach_plan:
        cost_xx = (P_xx.detach() * C_xx).sum(dim=(1, 2)) * float(N)
        cost_yy = (P_yy.detach() * C_yy).sum(dim=(1, 2)) * float(N)
    # Self-OT of the fixed target has no pred grads; keep for the scalar bias.
    div = cost_xy - 0.5 * cost_xx - 0.5 * cost_yy.detach()
    return div.mean()


def gather_nn(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``values`` [B,M,C] with indices [B,N] -> [B,N,C]."""
    B, N = indices.shape
    C = values.shape[-1]
    idx = indices.unsqueeze(-1).expand(B, N, C)
    return torch.gather(values, 1, idx)


def rgb_l1_on_nn(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    idx_pred_to_tgt: torch.Tensor,
    *,
    bidirectional: bool = True,
    idx_tgt_to_pred: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L1 color loss using Chamfer NN matches from xyz."""
    matched = gather_nn(target_rgb, idx_pred_to_tgt)
    loss = F.l1_loss(pred_rgb, matched)
    if bidirectional and idx_tgt_to_pred is not None:
        matched_rev = gather_nn(pred_rgb, idx_tgt_to_pred)
        loss = loss + F.l1_loss(target_rgb, matched_rev)
        loss = loss * 0.5
    return loss


class PointCloudAELoss(nn.Module):
    """Recon CD + RGB + optional centre–FPS coupling (Sinkhorn and/or CD).

    Total::

        L = L_cd + λ_rgb L_rgb
          + λ_anc L_sinkhorn(centers, fps)
          + λ_anc_cd L_cd(centers, fps)

    Sinkhorn provides soft nearly-bijective matching; centre–FPS Chamfer adds
    an independent hard NN geometric penalty so outlier centres far from every
    FPS (and uncovered FPS) still pay a full squared distance cost.
    """

    def __init__(
        self,
        *,
        lambda_rgb: float = 1.0,
        lambda_anc: float = 0.1,
        lambda_anc_cd: float = 0.0,
        bidirectional_rgb: bool = True,
        sinkhorn_eps: float = 0.02,
        sinkhorn_iters: int = 50,
    ):
        super().__init__()
        self.lambda_rgb = float(lambda_rgb)
        self.lambda_anc = float(lambda_anc)
        self.lambda_anc_cd = float(lambda_anc_cd)
        self.bidirectional_rgb = bool(bidirectional_rgb)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)

    def forward(
        self,
        pred_xyz: torch.Tensor,
        pred_rgb: torch.Tensor,
        gt_xyz: torch.Tensor,
        gt_rgb: torch.Tensor,
        centers: Optional[torch.Tensor] = None,
        fps_xyz: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cd, idx_p2t, idx_t2p = chamfer_distance(pred_xyz, gt_xyz)
        rgb = rgb_l1_on_nn(
            pred_rgb,
            gt_rgb,
            idx_p2t,
            bidirectional=self.bidirectional_rgb,
            idx_tgt_to_pred=idx_t2p if self.bidirectional_rgb else None,
        )
        total = cd + self.lambda_rgb * rgb
        zero = pred_xyz.new_zeros(())
        extras: Dict[str, torch.Tensor] = {
            "loss_cd": cd.detach(),
            "loss_rgb": rgb.detach(),
            "_cd": cd,
            "_rgb": rgb,
            "loss_anc": zero.detach(),
            "_anc": zero,
            "loss_anc_cd": zero.detach(),
            "_anc_cd": zero,
        }

        have_anchors = centers is not None and fps_xyz is not None
        if have_anchors and self.lambda_anc > 0:
            anc = sinkhorn_matching_loss(
                centers,
                fps_xyz,
                epsilon=self.sinkhorn_eps,
                n_iters=self.sinkhorn_iters,
            )
            total = total + self.lambda_anc * anc
            extras["loss_anc"] = anc.detach()
            extras["_anc"] = anc

        if have_anchors and self.lambda_anc_cd > 0:
            # Bidirectional squared Chamfer: hard NN pin centres↔FPS.
            # fps is typically already detached by the trainer.
            anc_cd, _, _ = chamfer_distance(centers, fps_xyz, bidirectional=True)
            total = total + self.lambda_anc_cd * anc_cd
            extras["loss_anc_cd"] = anc_cd.detach()
            extras["_anc_cd"] = anc_cd

        extras["loss_total"] = total.detach()
        return total, extras

    def weighted_terms_for_grad_norm(
        self,
        extras: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Weighted scalar loss terms for per-term grad-norm logging."""
        terms: Dict[str, torch.Tensor] = {"cd": extras["loss_cd"]}
        if self.lambda_rgb > 0:
            terms["rgb"] = self.lambda_rgb * extras["_rgb"]
        if self.lambda_anc > 0 and "_anc" in extras:
            terms["anc"] = self.lambda_anc * extras["_anc"]
        if self.lambda_anc_cd > 0 and "_anc_cd" in extras:
            terms["anc_cd"] = self.lambda_anc_cd * extras["_anc_cd"]
        return terms
