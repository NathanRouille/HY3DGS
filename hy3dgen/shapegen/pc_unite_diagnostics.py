"""Latent-space diagnostics for ShapePCUnite evaluation.

Layout under ``{eval_out}/diagnostics/``:

  path_from_noise/     Curve A — ODE from t=0 vs linear interpolant (euclid + cos)
  oracle_endpoints/    Curve B — ODE started at each t0; endpoint vs z and vs mean
  velocity_along_ode/  Cosine of v_pred vs optimal (z−x_t)/(1−t) on the ODE path
  pca_trajectories/    PCA plots (per-object + global); reuses Curve B trajectories

Each curve stage writes ``global/`` (mean over objects) and ``per_object/``.

Optimization: one shared ODE pass per object produces Curve A (t0=0 trajectory),
Curve B endpoints (all t_start), the velocity curve (GE eval at ODE waypoints),
and PCA waypoints for a subset. When called from ``evaluate_pc_unite``, the
cfg=1.0 gen ODE (+ trajectory) and the eval oracle endpoint are reused.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

DEFAULT_T_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def token_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())


def _mean(vals: Sequence[float]) -> float:
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def _safe_stem(mesh: str, idx: int) -> str:
    return f"obj_{idx:04d}_{Path(mesh).stem[:12]}"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _write_csv(path: Path, rows: List[Dict], fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _avg_rows(rows: List[Dict], t_key: str) -> List[Dict]:
    by_t: Dict[float, List[Dict]] = {}
    for r in rows:
        by_t.setdefault(float(r[t_key]), []).append(r)
    out = []
    for t, group in sorted(by_t.items()):
        keys = {
            k
            for g in group
            for k, v in g.items()
            if k != t_key and isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        row: Dict[str, float] = {t_key: t}
        for k in sorted(keys):
            row[k] = _mean([float(g[k]) for g in group if k in g])
        out.append(row)
    return out


def _try_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        logger.warning("matplotlib missing — plots will be skipped")
        return None


def _plot_curves(
    path: Path,
    series: List[Tuple[str, List[float], List[float]]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    plt = _try_pyplot()
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, xs, ys in series:
        ax.plot(xs, ys, marker="o", lw=1.8, ms=4, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _flatten_latent(z: torch.Tensor):
    import numpy as np

    return z.detach().float().cpu().reshape(-1).numpy()


def _pca_fit_project(X, n_comp: int = 2):
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_comp, Vt.shape[0])
    components = Vt[:k]
    proj = Xc @ components.T
    var = (S**2) / max(X.shape[0] - 1, 1)
    ratio = var[:k] / max(var.sum(), 1e-12)
    return proj, ratio


def _estimate_z_mean(model, it: Dict, z_star: torch.Tensor, num_draws: int = 4) -> torch.Tensor:
    """Mean of several encodes (stochastic registers); falls back to z_star."""
    surface = it.get("surface")
    weak, keep = it.get("weak"), it.get("keep")
    zs = [z_star]
    if surface is not None and weak is not None:
        for _ in range(max(0, int(num_draws) - 1)):
            zi, _ = model.encode(surface, weak_context=weak, weak_context_keep=keep)
            zs.append(zi)
    return torch.stack([z.squeeze(0) for z in zs], dim=0).mean(dim=0, keepdim=True)


@torch.no_grad()
def _predict_velocity(model, x_t, t_scalar: float, weak, keep, cam=None) -> torch.Tensor:
    """GE velocity at ``(x_t, t)`` — same formula as ``sample_latents``."""
    b = x_t.shape[0]
    t = torch.full((b,), float(t_scalar), device=x_t.device, dtype=x_t.dtype)
    cond = model._resolve_cam_cond(
        cam, b, device=x_t.device, dtype=x_t.dtype, use_null=(cam is None)
    )
    x_pred = model.latent_norm(
        model._run_ge(x_t, t, context_embed=weak, context_keep=keep, cond_embed=cond)
    )
    eps = float(getattr(model.transport, "train_eps", 1e-3))
    denom = (1.0 - t.view(-1, 1, 1)).clamp_min(eps)
    return (x_pred - x_t) / denom


@torch.no_grad()
def _velocity_curve_on_traj(model, traj, t_grid, z, weak, keep, cam=None) -> List[Dict]:
    """Cosine(v_pred, (z−x_t)/(1−t)) at every ODE waypoint (incl. t→1)."""
    eps = float(getattr(model.transport, "train_eps", 1e-3))
    rows = []
    for ti in range(traj.shape[0]):
        t = float(t_grid[ti])
        x_t = traj[ti]
        v_pred = _predict_velocity(model, x_t, t, weak, keep, cam=cam)
        denom = max(1.0 - t, eps)
        u_star = (z - x_t) / denom
        rows.append(
            {
                "t": t,
                "velocity_cos": token_cosine(v_pred, u_star),
                "v_pred_norm": float(v_pred.norm()),
                "u_star_norm": float(u_star.norm()),
                "rel_v_to_ustar": rel_err(v_pred, u_star),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# PCA plotting
# ---------------------------------------------------------------------------


def _plot_pca_case(path, proj, kinds, t_starts, title: str) -> None:
    plt = _try_pyplot()
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, marker, color in (
        ("z_star", "*", "black"),
        ("eps", "X", "red"),
        ("z_mean", "D", "purple"),
    ):
        idx = [i for i, k in enumerate(kinds) if k == name]
        if idx:
            ax.scatter(
                proj[idx, 0], proj[idx, 1], c=color, marker=marker, s=120, zorder=5, label=name
            )
    cmap = plt.cm.viridis
    for t0 in t_starts:
        idx_ode = [i for i, k in enumerate(kinds) if k == f"ode_t0={t0:g}"]
        if len(idx_ode) >= 2:
            color = cmap(min(max(t0, 0.0), 1.0))
            ax.plot(
                proj[idx_ode, 0],
                proj[idx_ode, 1],
                "-",
                color=color,
                lw=1.5,
                alpha=0.85,
                label=f"ODE t0={t0:g}",
            )
            ax.scatter(proj[idx_ode[0], 0], proj[idx_ode[0], 1], c=[color], s=30, zorder=4)
            ax.scatter(
                proj[idx_ode[-1], 0],
                proj[idx_ode[-1], 1],
                c=[color],
                s=50,
                marker="o",
                edgecolors="k",
                zorder=4,
            )
    i_eps = next((i for i, k in enumerate(kinds) if k == "eps"), None)
    i_z = next((i for i, k in enumerate(kinds) if k == "z_star"), None)
    if i_eps is not None and i_z is not None:
        ax.plot(
            [proj[i_eps, 0], proj[i_z, 0]],
            [proj[i_eps, 1], proj[i_z, 1]],
            "--",
            color="gray",
            lw=1,
            alpha=0.6,
            label="linear bridge",
        )
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_pca_global(path, proj, meta, title: str) -> None:
    plt = _try_pyplot()
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.viridis
    for i, (_cid, kind, t) in enumerate(meta):
        if kind in ("z_star", "eps", "z_mean"):
            continue
        if isinstance(t, float) and t == t:
            ax.scatter(proj[i, 0], proj[i, 1], c=[cmap(min(max(t, 0.0), 1.0))], s=12, alpha=0.5)
    for name, marker, color in (
        ("z_star", "*", "black"),
        ("eps", "X", "red"),
        ("z_mean", "D", "purple"),
    ):
        idx = [i for i, m in enumerate(meta) if m[1] == name]
        if idx:
            ax.scatter(
                proj[idx, 0],
                proj[idx, 1],
                c=color,
                marker=marker,
                s=80,
                zorder=5,
                label=name,
            )
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main unified runner
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_eval_diagnostics(
    model,
    items: List[Dict],
    *,
    out_dir: Path,
    sample_steps: int,
    device: torch.device,
    t_grid: Optional[Sequence[float]] = None,
    pca_max_objects: int = 8,
    seed: int = 0,
    num_mean_draws: int = 4,
    waypoint_stride: int = 2,
    run_path: bool = True,
    run_oracle: bool = True,
    run_velocity: bool = True,
    run_pca: bool = True,
) -> Dict[str, Any]:
    """Write diagnostics under ``out_dir/diagnostics`` and return a summary dict."""
    del seed  # reserved for future multi-noise PCA draws
    diag_root = Path(out_dir) / "diagnostics"
    diag_root.mkdir(parents=True, exist_ok=True)
    t_grid = [float(t) for t in (t_grid if t_grid is not None else DEFAULT_T_GRID)]
    t_grid = sorted(set(t_grid))

    path_root = diag_root / "path_from_noise"
    orc_root = diag_root / "oracle_endpoints"
    vel_root = diag_root / "velocity_along_ode"
    pca_root = diag_root / "pca_trajectories"
    for root in (path_root, orc_root, vel_root, pca_root):
        (root / "per_object").mkdir(parents=True, exist_ok=True)
        (root / "global").mkdir(parents=True, exist_ok=True)

    usable = [(i, it) for i, it in enumerate(items) if it.get("weak") is not None]
    pca_budget = max(0, int(pca_max_objects)) if run_pca else 0

    all_path_rows: List[Dict] = []
    all_orc_rows: List[Dict] = []
    all_vel_rows: List[Dict] = []
    per_path: List[Dict] = []
    per_orc: List[Dict] = []
    per_vel: List[Dict] = []
    pca_cases: List[Dict] = []
    global_vecs: List = []
    global_meta: List = []

    for rank, (i, it) in enumerate(usable):
        z, noise = it["z"], it["noise"]
        weak, keep, cam = it["weak"], it["keep"], it.get("cam")
        stem = _safe_stem(it["mesh"], i)
        do_pca = rank < pca_budget
        cached_endpoints: Dict[float, torch.Tensor] = dict(it.get("diag_oracle_endpoints") or {})

        z_mean = _estimate_z_mean(model, it, z, num_draws=num_mean_draws)

        # --- noise→z ODE (Curve A / velocity / Curve B t=0 / PCA) ---
        need_z0 = run_path or run_oracle or run_velocity or do_pca
        need_traj0 = run_path or run_velocity or do_pca
        z_gen = it.get("diag_z_gen")
        traj0 = it.get("diag_traj0")
        t_grid0 = it.get("diag_t_grid0")
        if need_z0 and (z_gen is None or (need_traj0 and traj0 is None)):
            if need_traj0:
                z_gen, traj0, t_grid0 = model.sample_latents(
                    weak,
                    batch_size=1,
                    num_steps=sample_steps,
                    guidance_scale=1.0,
                    noise=noise,
                    context_keep=keep,
                    cam_cond=cam,
                    raw_camera_tokens=it.get("raw_camera_tokens"),
                    device=device,
                    dtype=z.dtype,
                    renorm_output=False,
                    return_trajectory=True,
                )
            else:
                z_gen = model.sample_latents(
                    weak,
                    batch_size=1,
                    num_steps=sample_steps,
                    guidance_scale=1.0,
                    noise=noise,
                    context_keep=keep,
                    cam_cond=cam,
                    raw_camera_tokens=it.get("raw_camera_tokens"),
                    device=device,
                    dtype=z.dtype,
                    renorm_output=False,
                )
                traj0, t_grid0 = None, None

        # --- velocity cosine along the actual ODE ---
        mean_vel_cos = float("nan")
        if run_velocity and traj0 is not None and t_grid0 is not None:
            vel_rows = _velocity_curve_on_traj(
                model, traj0, t_grid0, z, weak, keep, cam=cam
            )
            for row in vel_rows:
                all_vel_rows.append({**row, "mesh": it["mesh"], "obj": stem})
            mean_vel_cos = _mean([r["velocity_cos"] for r in vel_rows])
            obj_vel = vel_root / "per_object" / stem
            obj_vel.mkdir(parents=True, exist_ok=True)
            _write_csv(obj_vel / "velocity_curve.csv", vel_rows)
            ts = [r["t"] for r in vel_rows]
            _plot_curves(
                obj_vel / "velocity_cosine.png",
                [("velocity_cos", ts, [r["velocity_cos"] for r in vel_rows])],
                xlabel="t",
                ylabel="token cosine",
                title=f"{stem} v_pred vs (z−x_t)/(1−t)",
            )
            _plot_curves(
                obj_vel / "velocity_norms.png",
                [
                    ("||v_pred||", ts, [r["v_pred_norm"] for r in vel_rows]),
                    ("||u*||", ts, [r["u_star_norm"] for r in vel_rows]),
                ],
                xlabel="t",
                ylabel="L2 norm",
                title=f"{stem} velocity magnitudes",
            )
            vel_m = {
                "obj": stem,
                "mesh": it["mesh"],
                "mean_velocity_cos": mean_vel_cos,
                "velocity_cos@t0": next(
                    (r["velocity_cos"] for r in vel_rows if abs(r["t"]) < 1e-9),
                    float("nan"),
                ),
            }
            _write_json(obj_vel / "metrics.json", vel_m)
            per_vel.append(vel_m)

        path_rows = []
        if run_path and traj0 is not None and t_grid0 is not None:
            for ti in range(traj0.shape[0]):
                t = float(t_grid0[ti])
                x_ode = traj0[ti]
                t_b = t_grid0[ti].view(1, 1, 1)
                x_gt = t_b * z + (1.0 - t_b) * noise
                row = {
                    "t": t,
                    "rel_to_interpolant": rel_err(x_ode, x_gt),
                    "cos_to_interpolant": token_cosine(x_ode, x_gt),
                }
                path_rows.append(row)
                all_path_rows.append({**row, "mesh": it["mesh"], "obj": stem})

            obj_path_dir = path_root / "per_object" / stem
            obj_path_dir.mkdir(parents=True, exist_ok=True)
            _write_csv(obj_path_dir / "path_curve.csv", path_rows)
            ts = [r["t"] for r in path_rows]
            _plot_curves(
                obj_path_dir / "path_euclid.png",
                [("rel_to_interpolant", ts, [r["rel_to_interpolant"] for r in path_rows])],
                xlabel="t",
                ylabel="relative L2",
                title=f"{stem} Curve A (euclid vs interpolant)",
            )
            _plot_curves(
                obj_path_dir / "path_cosine.png",
                [("cos_to_interpolant", ts, [r["cos_to_interpolant"] for r in path_rows])],
                xlabel="t",
                ylabel="token cosine",
                title=f"{stem} Curve A (cosine vs interpolant)",
            )
            path_m = {
                "obj": stem,
                "mesh": it["mesh"],
                "mean_rel_to_interpolant": _mean([r["rel_to_interpolant"] for r in path_rows]),
                "mean_cos_to_interpolant": _mean([r["cos_to_interpolant"] for r in path_rows]),
                "endpoint_rel_to_z": rel_err(z_gen, z),
                "endpoint_cos_to_z": token_cosine(z_gen, z),
                "endpoint_rel_to_mean": rel_err(z_gen, z_mean),
                "endpoint_cos_to_mean": token_cosine(z_gen, z_mean),
            }
            _write_json(obj_path_dir / "metrics.json", path_m)
            per_path.append(path_m)

        # --- Curve B ---
        orc_rows = []
        pca_trajs: Dict[float, Tuple[torch.Tensor, torch.Tensor]] = {}
        if do_pca and traj0 is not None and t_grid0 is not None:
            pca_trajs[0.0] = (traj0, t_grid0)

        if run_oracle or do_pca:
            for t0 in t_grid:
                if abs(t0 - 0.0) < 1e-9:
                    z_out = z_gen
                    if do_pca and 0.0 not in pca_trajs and traj0 is not None:
                        pca_trajs[0.0] = (traj0, t_grid0)
                elif abs(t0 - 1.0) < 1e-9:
                    z_out = z
                    if do_pca:
                        t1 = torch.ones(1, device=z.device, dtype=z.dtype)
                        pca_trajs[1.0] = (z.unsqueeze(0), t1)
                else:
                    need_traj = do_pca
                    cached = None
                    if not need_traj:
                        for ct, cz in cached_endpoints.items():
                            if abs(float(ct) - float(t0)) < 1e-9:
                                cached = cz
                                break
                    if cached is not None:
                        z_out = cached
                    elif need_traj:
                        z_out, traj, tg = model.sample_latents(
                            weak,
                            batch_size=1,
                            num_steps=sample_steps,
                            guidance_scale=1.0,
                            z_init=z,
                            t_start=float(t0),
                            noise=noise,
                            context_keep=keep,
                            cam_cond=cam,
                            raw_camera_tokens=it.get("raw_camera_tokens"),
                            device=device,
                            dtype=z.dtype,
                            renorm_output=False,
                            return_trajectory=True,
                        )
                        pca_trajs[float(t0)] = (traj, tg)
                    else:
                        z_out = model.sample_latents(
                            weak,
                            batch_size=1,
                            num_steps=sample_steps,
                            guidance_scale=1.0,
                            z_init=z,
                            t_start=float(t0),
                            noise=noise,
                            context_keep=keep,
                            cam_cond=cam,
                            raw_camera_tokens=it.get("raw_camera_tokens"),
                            device=device,
                            dtype=z.dtype,
                            renorm_output=False,
                        )
                if run_oracle:
                    row = {
                        "t_start": float(t0),
                        "rel_to_z": rel_err(z_out, z),
                        "cos_to_z": token_cosine(z_out, z),
                        "rel_to_mean": rel_err(z_out, z_mean),
                        "cos_to_mean": token_cosine(z_out, z_mean),
                    }
                    orc_rows.append(row)
                    all_orc_rows.append({**row, "mesh": it["mesh"], "obj": stem})

        if run_oracle and orc_rows:
            obj_orc_dir = orc_root / "per_object" / stem
            obj_orc_dir.mkdir(parents=True, exist_ok=True)
            _write_csv(obj_orc_dir / "endpoint_curve.csv", orc_rows)
            ts = [r["t_start"] for r in orc_rows]
            _plot_curves(
                obj_orc_dir / "endpoint_euclid.png",
                [
                    ("rel_to_z", ts, [r["rel_to_z"] for r in orc_rows]),
                    ("rel_to_mean", ts, [r["rel_to_mean"] for r in orc_rows]),
                ],
                xlabel="t_start",
                ylabel="relative L2",
                title=f"{stem} Curve B (euclid)",
            )
            _plot_curves(
                obj_orc_dir / "endpoint_cosine.png",
                [
                    ("cos_to_z", ts, [r["cos_to_z"] for r in orc_rows]),
                    ("cos_to_mean", ts, [r["cos_to_mean"] for r in orc_rows]),
                ],
                xlabel="t_start",
                ylabel="token cosine",
                title=f"{stem} Curve B (cosine)",
            )
            orc_m = {
                "obj": stem,
                "mesh": it["mesh"],
                # t_start=0 reuses the pure-noise generated endpoint — NOT oracle@0.5.
                "t0_noise_endpoint_rel_to_z": next(
                    (r["rel_to_z"] for r in orc_rows if abs(r["t_start"]) < 1e-9), float("nan")
                ),
                "t0_noise_endpoint_cos_to_z": next(
                    (r["cos_to_z"] for r in orc_rows if abs(r["t_start"]) < 1e-9), float("nan")
                ),
                "t0_noise_endpoint_rel_to_mean": next(
                    (r["rel_to_mean"] for r in orc_rows if abs(r["t_start"]) < 1e-9), float("nan")
                ),
                "t0_noise_endpoint_cos_to_mean": next(
                    (r["cos_to_mean"] for r in orc_rows if abs(r["t_start"]) < 1e-9), float("nan")
                ),
                # Actual half-noise oracle (t_start=0.5) when present in the curve.
                "t0p5_oracle_rel_to_z": next(
                    (r["rel_to_z"] for r in orc_rows if abs(r["t_start"] - 0.5) < 1e-9),
                    float("nan"),
                ),
                "t0p5_oracle_cos_to_z": next(
                    (r["cos_to_z"] for r in orc_rows if abs(r["t_start"] - 0.5) < 1e-9),
                    float("nan"),
                ),
                "t0p5_oracle_rel_to_mean": next(
                    (r["rel_to_mean"] for r in orc_rows if abs(r["t_start"] - 0.5) < 1e-9),
                    float("nan"),
                ),
                "t0p5_oracle_cos_to_mean": next(
                    (r["cos_to_mean"] for r in orc_rows if abs(r["t_start"] - 0.5) < 1e-9),
                    float("nan"),
                ),
                # Backward-compat aliases (historically pointed at t_start=0 / noise endpoint).
                "endpoint_rel_to_z": next(
                    (r["rel_to_z"] for r in orc_rows if abs(r["t_start"]) < 1e-9), float("nan")
                ),
                "endpoint_cos_to_z": next(
                    (r["cos_to_z"] for r in orc_rows if abs(r["t_start"]) < 1e-9), float("nan")
                ),
                "endpoint_rel_to_mean": next(
                    (r["rel_to_mean"] for r in orc_rows if abs(r["t_start"]) < 1e-9), float("nan")
                ),
                "endpoint_cos_to_mean": next(
                    (r["cos_to_mean"] for r in orc_rows if abs(r["t_start"]) < 1e-9), float("nan")
                ),
                "endpoint_alias_note": (
                    "endpoint_* aliases are t_start=0 (noise endpoint); "
                    "use t0p5_oracle_* for half-noise oracle."
                ),
            }
            _write_json(obj_orc_dir / "metrics.json", orc_m)
            per_orc.append(orc_m)

        if do_pca and pca_trajs:
            case_id = stem
            case_vecs = []
            case_kinds = []
            for name, tens in (("z_star", z), ("eps", noise), ("z_mean", z_mean)):
                v = _flatten_latent(tens)
                case_vecs.append(v)
                case_kinds.append(name)
                global_vecs.append(v)
                global_meta.append((case_id, name, float("nan")))

            endpoint_rows = []
            for t0 in sorted(pca_trajs.keys()):
                traj, tg = pca_trajs[t0]
                idxs = list(range(0, traj.shape[0], max(1, waypoint_stride)))
                if idxs[-1] != traj.shape[0] - 1:
                    idxs.append(traj.shape[0] - 1)
                for ii in idxs:
                    vv = _flatten_latent(traj[ii])
                    case_vecs.append(vv)
                    case_kinds.append(f"ode_t0={t0:g}")
                    global_vecs.append(vv)
                    global_meta.append((case_id, f"ode@{t0:g}", float(tg[ii])))
                z_end = traj[-1]
                endpoint_rows.append(
                    {
                        "t_start": t0,
                        "rel_to_z": rel_err(z_end, z),
                        "cos_to_z": token_cosine(z_end, z),
                        "rel_to_mean": rel_err(z_end, z_mean),
                        "cos_to_mean": token_cosine(z_end, z_mean),
                    }
                )

            import numpy as np

            X = np.stack(case_vecs, axis=0)
            proj, ratio = _pca_fit_project(X, n_comp=2)
            obj_pca = pca_root / "per_object" / stem
            plots_dir = obj_pca / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            _plot_pca_case(
                plots_dir / f"{case_id}.png",
                proj,
                case_kinds,
                sorted(pca_trajs.keys()),
                title=f"{case_id}  PCA var={ratio[0]:.2f}/{ratio[1]:.2f}",
            )
            _write_json(obj_pca / "endpoints.json", endpoint_rows)
            pca_cases.append(
                {
                    "case_id": case_id,
                    "obj": stem,
                    "mesh": it["mesh"],
                    "endpoints": endpoint_rows,
                    "pca_explained_var": ratio.tolist(),
                }
            )

        logger.info(
            "[diag] %d/%d %s  mean_v_cos=%.4f  end_z=%.4f",
            rank + 1,
            len(usable),
            stem,
            mean_vel_cos,
            rel_err(z_gen, z) if z_gen is not None else float("nan"),
        )

    summary: Dict[str, Any] = {
        "t_grid": t_grid,
        "sample_steps": sample_steps,
        "num_objects": len(usable),
    }
    flat: Dict[str, Any] = {"num_objects": len(usable)}

    if run_velocity and per_vel:
        glob_vel = _avg_rows(all_vel_rows, "t")
        gdir = vel_root / "global"
        _write_csv(gdir / "velocity_curve.csv", glob_vel)
        _write_csv(gdir / "per_object.csv", per_vel)
        if glob_vel:
            _plot_curves(
                gdir / "velocity_cosine.png",
                [
                    (
                        "velocity_cos",
                        [r["t"] for r in glob_vel],
                        [r["velocity_cos"] for r in glob_vel],
                    )
                ],
                xlabel="t",
                ylabel="token cosine",
                title="Global v_pred vs (z−x_t)/(1−t)",
            )
            _plot_curves(
                gdir / "velocity_norms.png",
                [
                    (
                        "||v_pred||",
                        [r["t"] for r in glob_vel],
                        [r["v_pred_norm"] for r in glob_vel],
                    ),
                    (
                        "||u*||",
                        [r["t"] for r in glob_vel],
                        [r["u_star_norm"] for r in glob_vel],
                    ),
                ],
                xlabel="t",
                ylabel="L2 norm",
                title="Global velocity magnitudes",
            )
        vel_metrics = {
            "num_objects": len(per_vel),
            "mean_velocity_cos": _mean([p["mean_velocity_cos"] for p in per_vel]),
            "velocity_curve": glob_vel,
        }
        _write_json(gdir / "metrics.json", vel_metrics)
        summary["velocity_along_ode"] = vel_metrics
        flat["mean_velocity_cos"] = vel_metrics["mean_velocity_cos"]

    if run_path and per_path:
        glob_curve = _avg_rows(all_path_rows, "t")
        gdir = path_root / "global"
        _write_csv(gdir / "path_curve.csv", glob_curve)
        if glob_curve:
            _plot_curves(
                gdir / "path_euclid.png",
                [
                    (
                        "rel_to_interpolant",
                        [r["t"] for r in glob_curve],
                        [r["rel_to_interpolant"] for r in glob_curve],
                    )
                ],
                xlabel="t",
                ylabel="relative L2",
                title="Global Curve A (euclid vs interpolant)",
            )
            _plot_curves(
                gdir / "path_cosine.png",
                [
                    (
                        "cos_to_interpolant",
                        [r["t"] for r in glob_curve],
                        [r["cos_to_interpolant"] for r in glob_curve],
                    )
                ],
                xlabel="t",
                ylabel="token cosine",
                title="Global Curve A (cosine vs interpolant)",
            )
        path_metrics = {
            "num_objects": len(per_path),
            "mean_rel_to_interpolant": _mean([p["mean_rel_to_interpolant"] for p in per_path]),
            "mean_cos_to_interpolant": _mean([p["mean_cos_to_interpolant"] for p in per_path]),
            "endpoint_rel_to_z": _mean([p["endpoint_rel_to_z"] for p in per_path]),
            "endpoint_cos_to_z": _mean([p["endpoint_cos_to_z"] for p in per_path]),
            "endpoint_rel_to_mean": _mean([p["endpoint_rel_to_mean"] for p in per_path]),
            "endpoint_cos_to_mean": _mean([p["endpoint_cos_to_mean"] for p in per_path]),
            "path_curve": glob_curve,
        }
        _write_json(gdir / "metrics.json", path_metrics)
        summary["path_from_noise"] = path_metrics
        for k, v in path_metrics.items():
            if isinstance(v, (int, float)):
                flat[f"path/{k}"] = v

    if run_oracle and per_orc:
        glob_orc = _avg_rows(all_orc_rows, "t_start")
        gdir = orc_root / "global"
        _write_csv(gdir / "endpoint_curve.csv", glob_orc)
        if glob_orc:
            _plot_curves(
                gdir / "endpoint_euclid.png",
                [
                    (
                        "rel_to_z",
                        [r["t_start"] for r in glob_orc],
                        [r["rel_to_z"] for r in glob_orc],
                    ),
                    (
                        "rel_to_mean",
                        [r["t_start"] for r in glob_orc],
                        [r["rel_to_mean"] for r in glob_orc],
                    ),
                ],
                xlabel="t_start",
                ylabel="relative L2",
                title="Global Curve B (euclid)",
            )
            _plot_curves(
                gdir / "endpoint_cosine.png",
                [
                    (
                        "cos_to_z",
                        [r["t_start"] for r in glob_orc],
                        [r["cos_to_z"] for r in glob_orc],
                    ),
                    (
                        "cos_to_mean",
                        [r["t_start"] for r in glob_orc],
                        [r["cos_to_mean"] for r in glob_orc],
                    ),
                ],
                xlabel="t_start",
                ylabel="token cosine",
                title="Global Curve B (cosine)",
            )
        orc_metrics = {
            "num_objects": len(per_orc),
            "t0_noise_endpoint_rel_to_z": _mean(
                [p["t0_noise_endpoint_rel_to_z"] for p in per_orc]
            ),
            "t0_noise_endpoint_cos_to_z": _mean(
                [p["t0_noise_endpoint_cos_to_z"] for p in per_orc]
            ),
            "t0p5_oracle_rel_to_z": _mean([p["t0p5_oracle_rel_to_z"] for p in per_orc]),
            "t0p5_oracle_cos_to_z": _mean([p["t0p5_oracle_cos_to_z"] for p in per_orc]),
            # Legacy flat keys (t_start=0 / noise endpoint — do not treat as oracle@0.5).
            "endpoint_rel_to_z": _mean([p["endpoint_rel_to_z"] for p in per_orc]),
            "endpoint_cos_to_z": _mean([p["endpoint_cos_to_z"] for p in per_orc]),
            "endpoint_rel_to_mean": _mean([p["endpoint_rel_to_mean"] for p in per_orc]),
            "endpoint_cos_to_mean": _mean([p["endpoint_cos_to_mean"] for p in per_orc]),
            "endpoint_curve": glob_orc,
        }
        _write_json(gdir / "metrics.json", orc_metrics)
        summary["oracle_endpoints"] = orc_metrics
        for k, v in orc_metrics.items():
            if isinstance(v, (int, float)):
                flat[f"oracle/{k}"] = v
        # Explicit preferred keys for half-noise oracle.
        flat["oracle/t0p5_rel_to_z"] = orc_metrics["t0p5_oracle_rel_to_z"]
        flat["oracle/t0p5_cos_to_z"] = orc_metrics["t0p5_oracle_cos_to_z"]
        flat["oracle/t0_noise_endpoint_rel_to_z"] = orc_metrics[
            "t0_noise_endpoint_rel_to_z"
        ]

    if run_pca and pca_cases:
        import numpy as np

        gdir = pca_root / "global"
        if global_vecs:
            Xg = np.stack(global_vecs, axis=0)
            proj_g, ratio_g = _pca_fit_project(Xg, n_comp=2)
            _plot_pca_global(
                gdir / "overview.png",
                proj_g,
                global_meta,
                title=f"Global PCA  var={ratio_g[0]:.2f}/{ratio_g[1]:.2f}",
            )
        _write_json(pca_root / "cases.json", {"cases": pca_cases})
        pca_summary = {"num_cases": len(pca_cases), "num_objects": len(pca_cases)}
        _write_json(gdir / "metrics.json", pca_summary)
        summary["pca_trajectories"] = pca_summary
        flat["pca/num_objects"] = len(pca_cases)

    summary["flat"] = flat
    _write_json(diag_root / "summary.json", summary)
    logger.info("Diagnostics written to %s", diag_root)
    return summary
