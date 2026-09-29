"""Surflo global-camera branch for AdaLN conditioning (optional ablation).

Transfers Surflo's camera token projector + 4-block camera compressor from a
Surflo checkpoint, and adds a zero-initialized 512→width adapter so that at
init the branch contributes **exactly zero** to AdaLN (identity vs parent).

Importing the full ``surflo`` package is avoided (hy3dgs may lack
``torch_geometric``); only ``surflo.nn.compressor`` is loaded via a lightweight
namespace shim.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_DEFAULT_SURFLO_ROOT = "/export/home/nathan/Surflo"
_DEFAULT_SURFLO_CKPT = "/export/home/nathan/Surflo/checkpoints/surflo_v0.pt"


def _ensure_surflo_compressor_importable(surflo_root: str) -> None:
    """Register package namespaces without executing ``surflo/__init__.py``."""
    root = Path(surflo_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Surflo root not found: {root}")
    mapping = [
        ("surflo", root / "surflo"),
        ("surflo.nn", root / "surflo" / "nn"),
        ("surflo.nn.vggt", root / "surflo" / "nn" / "vggt"),
        ("surflo.nn.vggt.layers", root / "surflo" / "nn" / "vggt" / "layers"),
    ]
    for pkg, path in mapping:
        existing = sys.modules.get(pkg)
        if existing is not None and getattr(existing, "__path__", None):
            continue
        mod = types.ModuleType(pkg)
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]
        mod.__file__ = str(path / "__init__.py")
        sys.modules[pkg] = mod


def build_surflo_camera_compressor(surflo_root: str) -> nn.Module:
    """Construct Surflo's camera compressor (1 latent, 4 CA, no SA)."""
    _ensure_surflo_compressor_importable(surflo_root)
    from surflo.nn.compressor import Compressor  # noqa: WPS433

    return Compressor(
        embed_dim=512,
        depth=4,
        mlp_ratio=4.0,
        sa_to_ca_ratio=0,
        num_latent_tokens=1,
        elastic=False,
    )


class SurfloGlobalCameraBranch(nn.Module):
    """Raw VGGT camera tokens [B,S,2048] → AdaLN residual [B, width].

    Pipeline (matches Surflo SurfaceNet camera path):
      Linear(2048→512) → Compressor → [B,1,512] → squeeze → Linear(512→width)
    The final Linear is zero-initialized so the residual is 0 at start.
    """

    def __init__(
        self,
        *,
        width: int = 1024,
        surflo_root: str = _DEFAULT_SURFLO_ROOT,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.width = int(width)
        self.camera_token_projector = nn.Linear(2048, 512, bias=True)
        self.camera_tokens_compressor = build_surflo_camera_compressor(surflo_root)
        # Zero-init adapter: parent predictions unchanged until training moves this.
        self.adapter = nn.Linear(512, self.width, bias=True)
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for p in self.camera_token_projector.parameters():
            p.requires_grad = False
        for p in self.camera_tokens_compressor.parameters():
            p.requires_grad = False
        self.camera_token_projector.eval()
        self.camera_tokens_compressor.eval()

    def unfreeze_adapter(self) -> None:
        for p in self.adapter.parameters():
            p.requires_grad = True

    def train(self, mode: bool = True):
        """Keep frozen Surflo modules in eval; only adapter follows train/eval."""
        super().train(mode)
        self.camera_token_projector.eval()
        self.camera_tokens_compressor.eval()
        return self

    @classmethod
    def from_surflo_checkpoint(
        cls,
        ckpt_path: str = _DEFAULT_SURFLO_CKPT,
        *,
        width: int = 1024,
        surflo_root: str = _DEFAULT_SURFLO_ROOT,
        freeze_backbone: bool = True,
    ) -> "SurfloGlobalCameraBranch":
        branch = cls(
            width=width, surflo_root=surflo_root, freeze_backbone=False
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ema = ckpt.get("ema_state") if isinstance(ckpt, dict) else None
        if not isinstance(ema, dict):
            raise KeyError(
                f"Expected ema_state dict in {ckpt_path}, got keys="
                f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
            )
        proj_w = ema["ema_model.surface_net.camera_token_projector.weight"]
        proj_b = ema["ema_model.surface_net.camera_token_projector.bias"]
        branch.camera_token_projector.load_state_dict(
            {"weight": proj_w, "bias": proj_b}
        )
        pref = "ema_model.surface_net.camera_tokens_compressor."
        comp_sd = {k[len(pref) :]: v for k, v in ema.items() if k.startswith(pref)}
        missing, unexpected = branch.camera_tokens_compressor.load_state_dict(
            comp_sd, strict=False
        )
        if missing or unexpected:
            logger.warning(
                "Surflo camera compressor load: missing=%s unexpected=%s",
                missing,
                unexpected,
            )
        # Re-zero adapter after construction (already zero; keep explicit).
        nn.init.zeros_(branch.adapter.weight)
        nn.init.zeros_(branch.adapter.bias)
        if freeze_backbone:
            branch.freeze_backbone()
        branch.unfreeze_adapter()
        logger.info(
            "Loaded Surflo global-camera projector+compressor from %s "
            "(adapter zero-init → AdaLN residual 0)",
            ckpt_path,
        )
        return branch

    def forward(
        self,
        raw_camera_tokens: torch.Tensor,
        *,
        use_null: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            raw_camera_tokens: [B, S, 2048] last-layer VGGT camera tokens.
            use_null: if True, return zeros (CFG / weak-context dropout path).
        Returns:
            [B, width] residual added to time AdaLN conditioning.
        """
        if use_null or raw_camera_tokens is None:
            b = 1 if raw_camera_tokens is None else int(raw_camera_tokens.shape[0])
            device = (
                self.adapter.weight.device
                if raw_camera_tokens is None
                else raw_camera_tokens.device
            )
            dtype = self.adapter.weight.dtype
            return torch.zeros(b, self.width, device=device, dtype=dtype)

        cam = raw_camera_tokens
        if cam.dim() == 2:
            cam = cam.unsqueeze(0)
        if cam.dim() != 3 or cam.shape[-1] != 2048:
            raise ValueError(
                f"raw_camera_tokens expected [B,S,2048], got {tuple(cam.shape)}"
            )
        cam = cam.to(dtype=self.camera_token_projector.weight.dtype)
        with torch.set_grad_enabled(self.camera_token_projector.weight.requires_grad):
            tokens = self.camera_token_projector(cam)  # [B,S,512]
            compressed, _ = self.camera_tokens_compressor(tokens)  # [B,1,512]
        global_cam = compressed.squeeze(1)  # [B,512]
        return self.adapter(global_cam.to(dtype=self.adapter.weight.dtype))


def extract_raw_vggt_camera_tokens(
    batch: dict,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Stack per-sample cached VGGT camera tokens → [B, S, 2048]."""
    payloads = batch.get("vggt_cache")
    if not payloads:
        return None
    cams = []
    for payload in payloads:
        if not isinstance(payload, dict) or "camera_token" not in payload:
            return None
        cam = payload["camera_token"].to(device=device, dtype=torch.float32)
        if cam.dim() == 3 and cam.shape[0] == 1:
            cam = cam.squeeze(0)
        if cam.dim() != 2:
            raise ValueError(f"cached camera_token expected [S,C], got {tuple(cam.shape)}")
        cams.append(cam)
    # Pad S if needed (rare)
    max_s = max(c.shape[0] for c in cams)
    c_dim = cams[0].shape[-1]
    out = []
    for c in cams:
        if c.shape[0] < max_s:
            pad = c.new_zeros(max_s - c.shape[0], c_dim)
            c = torch.cat([c, pad], dim=0)
        out.append(c)
    return torch.stack(out, dim=0)
