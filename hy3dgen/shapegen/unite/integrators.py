"""ODE / SDE integrators for flow sampling (from UNITE denoiser_utils/integrators.py)."""

from __future__ import annotations

import torch

try:
    from torchdiffeq import odeint
except ImportError:  # pragma: no cover - optional at import time
    odeint = None


class ode:
    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        timestep_shift,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"
        self.drift = drift
        self.t = torch.linspace(t0, t1, num_steps)
        if timestep_shift > 0:
            def compute_tm(t_n, ts):
                return ts * t_n / (1 + (ts - 1) * t_n)
            self.t = torch.tensor([compute_tm(float(t_n), timestep_shift) for t_n in self.t])
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        device = x[0].device if isinstance(x, tuple) else x.device
        t = self.t.to(device)

        if self.sampler_type == "euler" or odeint is None:
            if odeint is None and self.sampler_type != "euler":
                raise ImportError(
                    "torchdiffeq is required for non-euler ODE sampling (pip install torchdiffeq)"
                )
            x_cur = x
            for i in range(len(t) - 1):
                t_i = t[i]
                dt = t[i + 1] - t_i
                t_b = torch.ones(
                    x_cur.size(0) if not isinstance(x_cur, tuple) else x_cur[0].size(0),
                    device=device,
                ) * t_i
                v = self.drift(x_cur, t_b, model, **model_kwargs)
                x_cur = x_cur + dt * v if not isinstance(x_cur, tuple) else tuple(
                    xi + dt * vi for xi, vi in zip(x_cur, v)
                )
            return torch.stack([x, x_cur]) if not isinstance(x, tuple) else (x, x_cur)

        def _fn(t_step, x_in):
            t_b = (
                torch.ones(x_in[0].size(0), device=device) * t_step
                if isinstance(x_in, tuple)
                else torch.ones(x_in.size(0), device=device) * t_step
            )
            return self.drift(x_in, t_b, model, **model_kwargs)

        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        return odeint(_fn, x, t, method=self.sampler_type, atol=atol, rtol=rtol)
