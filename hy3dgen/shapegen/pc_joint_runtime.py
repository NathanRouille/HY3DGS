"""Joint-run scheduling and gradients on the shared GE parameters."""
import math

import torch


def make_lr_scheduler(optimizer, *, schedule, steps, warmup_steps=0, min_ratio=.01):
    if not 0 <= warmup_steps < steps or not 0 <= min_ratio <= 1:
        raise ValueError("Warmup must be shorter than the run; LR floor must be in [0,1]")
    if warmup_steps == 0 and min_ratio == .01:
        if schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=steps, eta_min=optimizer.param_groups[0]["lr"] * min_ratio)
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=0)
    def factor(update):
        if update < warmup_steps:
            return (update + 1) / warmup_steps
        if schedule == "constant":
            return 1.
        progress = min(1., (update - warmup_steps) / max(steps - warmup_steps, 1))
        return min_ratio + (1 - min_ratio) * .5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def shared_ge_gradient_metrics(model, recon_loss, flow_loss):
    """Weighted gradients, before clipping; do not modify parameter .grad."""
    params = tuple(p for p in model.ge.parameters() if p.requires_grad)
    if not params:
        raise ValueError("Shared GE has no trainable parameters")
    recon = torch.autograd.grad(recon_loss, params, retain_graph=True, allow_unused=True)
    flow = torch.autograd.grad(flow_loss, params, retain_graph=True, allow_unused=True)
    device = params[0].device
    rr, ff, dot = (torch.zeros((), device=device) for _ in range(3))
    for r, f in zip(recon, flow):
        if r is not None:
            rr += r.detach().float().square().sum()
        if f is not None:
            ff += f.detach().float().square().sum()
        if r is not None and f is not None:
            dot += (r.detach().float() * f.detach().float()).sum()
    rnorm, fnorm = rr.sqrt(), ff.sqrt()
    return {"shared_ge/recon_grad_norm": float(rnorm),
            "shared_ge/flow_grad_norm": float(fnorm),
            "shared_ge/flow_to_recon_ratio": float(fnorm / rnorm.clamp_min(1e-12)),
            "shared_ge/gradient_cosine": float(dot / (rnorm * fnorm).clamp_min(1e-12))}
