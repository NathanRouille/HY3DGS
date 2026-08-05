"""Flow transport + sampling (adapted from UNITE modules/denoiser.py)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.stats import norm

from .integrators import ode
from .path import ICPlan


class Transport:
    def __init__(
        self,
        train_eps: float = 5e-2,
        sample_eps: float = 5e-2,
        *,
        use_lognorm: bool = True,
        lognorm_mu: float = 1.0,
        lognorm_sigma: float = 1.6,
    ):
        self.path_sampler = ICPlan()
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.use_lognorm = use_lognorm
        self.lognorm_mu = lognorm_mu
        self.lognorm_sigma = lognorm_sigma

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0, t1 = 0.0, 1.0
        eps = train_eps if not eval else sample_eps
        if sde:
            t0 = eps
            t1 = 1 - eps if last_step_size == 0 else 1 - last_step_size
        if reverse:
            t0, t1 = 1 - t0, 1 - t1
        return t0, t1

    def sample_logit_normal(self, mu, sigma, size=1):
        samples = norm.rvs(loc=mu, scale=sigma, size=size)
        samples = 1 / (1 + np.exp(-samples))
        return torch.tensor(samples, dtype=torch.float32)

    def sample(
        self,
        x1: torch.Tensor,
        *,
        sp_timesteps=None,
        timestep_shift: float = 0.0,
    ):
        x0 = torch.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        if self.use_lognorm:
            t = self.sample_logit_normal(self.lognorm_mu, self.lognorm_sigma, size=x1.shape[0])
            t = t * (t1 - t0) + t0
        else:
            t = torch.rand(x1.shape[0]) * (t1 - t0) + t0
        if sp_timesteps is not None:
            t = torch.rand(x1.shape[0]) * (sp_timesteps[1] - sp_timesteps[0]) + sp_timesteps[0]
        if timestep_shift > 0:
            t = timestep_shift * t / (1.0 + (timestep_shift - 1.0) * t)
        return t.to(x1.device), x0, x1

    def training_losses(
        self,
        model,
        x1: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        sp_timesteps=None,
        timestep_shift: float = 0.0,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if t is None:
            t, x0, x1 = self.sample(
                x1, sp_timesteps=sp_timesteps, timestep_shift=timestep_shift
            )
        else:
            x0 = torch.randn_like(x1)
        t, xt, _ut = self.path_sampler.plan(t, x0, x1)
        model_output = model(xt, t, **model_kwargs)
        return {
            "pred": model_output,
            "model_output": model_output,
            "sampled_t": t,
            "xt": xt,
            "x1": x1,
        }

    def get_drift(self):
        def body_fn(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            assert model_output.shape == x.shape
            return model_output
        return body_fn


class Sampler:
    def __init__(self, transport: Transport):
        self.transport = transport
        self.drift = transport.get_drift()

    def sample_ode(
        self,
        *,
        sampling_method: str = "euler",
        num_steps: int = 50,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        reverse: bool = False,
        timestep_shift: float = 0.0,
        t0: Optional[float] = None,
    ):
        if reverse:
            drift = lambda x, t, model, **kw: self.drift(
                x, torch.ones_like(t) * (1 - t), model, **kw
            )
        else:
            drift = self.drift
        interval_t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
        )
        # ``t0`` lets callers resume the ODE from a partially noised latent.
        if t0 is None:
            t0 = interval_t0
        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            timestep_shift=timestep_shift,
        )
        return _ode.sample
