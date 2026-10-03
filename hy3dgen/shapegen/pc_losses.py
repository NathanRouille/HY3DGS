"""Point-cloud reconstruction losses for ShapePCAE."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def pairwise_dist2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared pairwise distances. a: [B,N,3], b: [B,M,3] -> [B,N,M]."""
    return torch.cdist(a, b, p=2).pow(2)


def chamfer_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    bidirectional: bool = True,
    w_pred2gt: float = 1.0,
    w_gt2pred: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chamfer (mean of squared NN distances), optionally direction-weighted.

    ``w_pred2gt`` scales precision (pred→GT); ``w_gt2pred`` scales coverage
    (GT→pred). Defaults ``1, 1`` recover classic symmetric Chamfer.

    Returns:
        loss: scalar Chamfer (weighted if bidirectional)
        idx_pred_to_tgt: [B, N] NN indices into target for each pred point
        idx_tgt_to_pred: [B, M] NN indices into pred for each target point
    """
    dist = pairwise_dist2(pred, target)  # [B, N, M]
    pred_to_tgt, idx_p2t = dist.min(dim=2)
    tgt_to_pred, idx_t2p = dist.min(dim=1)
    if bidirectional:
        loss = float(w_pred2gt) * pred_to_tgt.mean() + float(w_gt2pred) * tgt_to_pred.mean()
    else:
        loss = float(w_pred2gt) * pred_to_tgt.mean()
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


def per_point_rgb_l1(
    src_rgb: torch.Tensor,
    ref_rgb: torch.Tensor,
    idx_src_to_ref: torch.Tensor,
) -> torch.Tensor:
    """Per-point mean |src - ref[nn]| over channels → [B, N]."""
    matched = gather_nn(ref_rgb, idx_src_to_ref)
    return (src_rgb - matched).abs().mean(dim=-1)


def _mean_and_topk(per_point_err: torch.Tensor, topk_frac: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean of per-point errors and mean of the worst top-k% (per batch item).

    ``per_point_err``: [B, N].
    """
    mean = per_point_err.mean()
    frac = float(topk_frac)
    if frac <= 0.0:
        return mean, per_point_err.new_zeros(())
    n = per_point_err.shape[1]
    k = max(1, min(n, int(round(frac * n))))
    # torch.topk is differentiable w.r.t. the selected values.
    topk_vals = torch.topk(per_point_err, k, dim=1, largest=True).values
    return mean, topk_vals.mean()


def rgb_l1_on_nn(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    idx_pred_to_tgt: torch.Tensor,
    *,
    bidirectional: bool = True,
    idx_tgt_to_pred: Optional[torch.Tensor] = None,
    topk_frac: float = 0.0,
    topk_beta: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """L1 colour loss using Chamfer NN matches from xyz.

    Returns ``(loss, parts)`` where::

        loss = mean + topk_beta * topk_mean   (if topk_frac > 0 and topk_beta > 0)
        loss = mean                           (otherwise; matches historical scalar)

    ``parts`` always includes ``mean`` / ``topk`` (topk is 0 when disabled).
    Bidirectional: average of pred→GT and GT→pred for both mean and top-k.
    """
    err_p = per_point_rgb_l1(pred_rgb, target_rgb, idx_pred_to_tgt)
    mean_p, topk_p = _mean_and_topk(err_p, topk_frac)

    if bidirectional and idx_tgt_to_pred is not None:
        err_t = per_point_rgb_l1(target_rgb, pred_rgb, idx_tgt_to_pred)
        mean_t, topk_t = _mean_and_topk(err_t, topk_frac)
        mean = 0.5 * (mean_p + mean_t)
        topk = 0.5 * (topk_p + topk_t)
    else:
        mean = mean_p
        topk = topk_p

    beta = float(topk_beta)
    frac = float(topk_frac)
    if frac > 0.0 and beta > 0.0:
        loss = mean + beta * topk
    else:
        loss = mean
        topk = mean.new_zeros(())

    return loss, {
        "mean": mean,
        "topk": topk,
        "beta": mean.new_tensor(beta if frac > 0.0 else 0.0),
    }


class PointCloudAELoss(nn.Module):
    """Recon CD + RGB + optional centre–FPS coupling (Sinkhorn and/or CD).

    Total::

        L = L_cd + λ_rgb L_rgb
          + λ_anc L_sinkhorn(centers, fps)
          + λ_anc_cd L_cd(centers, fps)
          + λ_delta mean(||x - center||^2)

    with geometry::

        L_cd = cd_pred2gt · mean(pred→GT) + cd_gt2pred · mean(GT→pred)

    with colour::

        L_rgb = mean_NN_L1 + β · mean(top-k% of per-point NN L1)

    (both directions of the Chamfer correspondence when bidirectional).
    Top-k pushes high residual colour (frets / paint edges); mean keeps
    bulk panels. Disable top-k with ``rgb_topk_frac=0`` or ``rgb_topk_beta=0``.

    Sinkhorn provides soft nearly-bijective matching; centre–FPS Chamfer adds
    an independent hard NN geometric penalty so outlier centres far from every
    FPS (and uncovered FPS) still pay a full squared distance cost.

    ``λ_delta`` softly keeps locals near their parent anchor (replacement for a
    hard ``max_anchor_delta`` tanh ball).

    ``cd_pred2gt`` / ``cd_gt2pred`` reweight the two Chamfer directions on the
    recon cloud (precision vs coverage). Defaults ``1, 1`` = classic CD.
    """

    def __init__(
        self,
        *,
        lambda_rgb: float = 1.0,
        lambda_anc: float = 0.1,
        lambda_anc_cd: float = 0.0,
        lambda_delta: float = 0.0,
        cd_pred2gt: float = 1.0,
        cd_gt2pred: float = 1.0,
        bidirectional_rgb: bool = True,
        sinkhorn_eps: float = 0.02,
        sinkhorn_iters: int = 50,
        rgb_topk_frac: float = 0.0,
        rgb_topk_beta: float = 1.0,
        geometry_only: bool = False,
    ):
        super().__init__()
        self.geometry_only = bool(geometry_only)
        self.lambda_rgb = 0.0 if self.geometry_only else float(lambda_rgb)
        self.lambda_anc = float(lambda_anc)
        self.lambda_anc_cd = float(lambda_anc_cd)
        self.lambda_delta = float(lambda_delta)
        self.cd_pred2gt = float(cd_pred2gt)
        self.cd_gt2pred = float(cd_gt2pred)
        self.bidirectional_rgb = bool(bidirectional_rgb)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.rgb_topk_frac = float(rgb_topk_frac)
        self.rgb_topk_beta = float(rgb_topk_beta)

    def forward(
        self,
        pred_xyz: torch.Tensor,
        pred_rgb: torch.Tensor,
        gt_xyz: torch.Tensor,
        gt_rgb: torch.Tensor,
        centers: Optional[torch.Tensor] = None,
        fps_xyz: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Unweighted direction means for logging; weighted sum enters the loss.
        dist = pairwise_dist2(pred_xyz, gt_xyz)
        pred_to_tgt, idx_p2t = dist.min(dim=2)
        tgt_to_pred, idx_t2p = dist.min(dim=1)
        cd_p2g = pred_to_tgt.mean()
        cd_g2p = tgt_to_pred.mean()
        cd = self.cd_pred2gt * cd_p2g + self.cd_gt2pred * cd_g2p
        zero = pred_xyz.new_zeros(())
        if self.geometry_only or pred_rgb is None or gt_rgb is None:
            # Skip the colour term outright rather than weighting it to zero: the
            # NN gathers and top-k are pure waste in a geometry-only run.
            rgb = zero
            rgb_parts = {"mean": zero, "topk": zero, "beta": zero}
        else:
            rgb, rgb_parts = rgb_l1_on_nn(
                pred_rgb,
                gt_rgb,
                idx_p2t,
                bidirectional=self.bidirectional_rgb,
                idx_tgt_to_pred=idx_t2p if self.bidirectional_rgb else None,
                topk_frac=self.rgb_topk_frac,
                topk_beta=self.rgb_topk_beta,
            )
        total = cd + self.lambda_rgb * rgb
        extras: Dict[str, torch.Tensor] = {
            "loss_cd": cd.detach(),
            "loss_cd_pred2gt": cd_p2g.detach(),
            "loss_cd_gt2pred": cd_g2p.detach(),
            "loss_rgb": rgb.detach(),
            "loss_rgb_mean": rgb_parts["mean"].detach(),
            "loss_rgb_topk": rgb_parts["topk"].detach(),
            "_cd": cd,
            "_rgb": rgb,
            "_rgb_mean": rgb_parts["mean"],
            "_rgb_topk": rgb_parts["topk"],
            "loss_anc": zero.detach(),
            "_anc": zero,
            "loss_anc_cd": zero.detach(),
            "_anc_cd": zero,
            "loss_delta": zero.detach(),
            "_delta": zero,
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

        if centers is not None and self.lambda_delta > 0:
            B, N, _ = pred_xyz.shape
            R = centers.shape[1]
            if N % R != 0:
                raise ValueError(
                    f"pred points N={N} not divisible by num anchors R={R} "
                    "(cannot assign locals to centres for delta reg)"
                )
            K = N // R
            delta = pred_xyz.view(B, R, K, 3) - centers.unsqueeze(2)
            # Mean squared Euclidean radius of locals about their anchor.
            d_loss = delta.pow(2).sum(dim=-1).mean()
            total = total + self.lambda_delta * d_loss
            extras["loss_delta"] = d_loss.detach()
            extras["_delta"] = d_loss

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
        if self.lambda_delta > 0 and "_delta" in extras:
            terms["delta"] = self.lambda_delta * extras["_delta"]
        return terms
