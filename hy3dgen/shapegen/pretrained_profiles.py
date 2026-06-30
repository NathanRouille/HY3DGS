"""Architecture profiles for Hunyuan3D-pretrained ShapeGSAE experiments."""

from __future__ import annotations

from typing import Any, Dict, Optional

HUNYUAN_MINI_PROFILE: Dict[str, Any] = {
    "pretrained_repo": "tencent/Hunyuan3D-2mini",
    "pretrained_subfolder": "hunyuan3d-vae-v2-mini-withencoder",
    "num_latents": 512,
    "embed_dim": 64,
    "width": 1024,
    "heads": 16,
    "num_encoder_layers": 8,
    "num_decoder_layers": 16,
    "downsample_ratio": 20,
    "pc_size": 5120,
    "pc_sharpedge_size": 5120,
    "qk_norm": True,
    "point_feats": 7,
    "surface_channels": 10,
    "include_sharp_label": True,
    "shapevae_point_feats": 4,
    "use_safetensors": False,
}

_PROFILES = {
    "hunyuan_mini": HUNYUAN_MINI_PROFILE,
}


def get_pretrained_profile(name: str) -> Dict[str, Any]:
    if name not in _PROFILES:
        raise ValueError(
            f"Unknown pretrained_profile {name!r}; choices: {sorted(_PROFILES)}"
        )
    return dict(_PROFILES[name])


def apply_pretrained_profile(args) -> None:
    """Mutate argparse namespace when a pretrained profile is selected."""
    profile_name = getattr(args, "pretrained_profile", "none") or "none"
    if profile_name == "none":
        return

    profile = get_pretrained_profile(profile_name)
    arch_keys = (
        "num_latents",
        "embed_dim",
        "width",
        "heads",
        "num_encoder_layers",
        "num_decoder_layers",
        "downsample_ratio",
        "pc_size",
        "pc_sharpedge_size",
        "qk_norm",
        "point_feats",
        "surface_channels",
        "include_sharp_label",
        "shapevae_point_feats",
        "use_safetensors",
    )
    for key in arch_keys:
        if key in profile:
            setattr(args, key, profile[key])

    if not getattr(args, "pretrained_repo", None):
        args.pretrained_repo = profile["pretrained_repo"]
    if not getattr(args, "pretrained_subfolder", None):
        args.pretrained_subfolder = profile["pretrained_subfolder"]


def resolve_include_sharp_label(args) -> bool:
    """Return whether surfaces should include the Hunyuan sharp-edge label channel."""
    explicit = getattr(args, "include_sharp_label", None)
    if explicit is not None:
        return bool(explicit)
    profile = getattr(args, "pretrained_profile", "none") or "none"
    if profile != "none":
        return bool(get_pretrained_profile(profile).get("include_sharp_label", False))
    return False


def assert_arch_matches_profile(model, profile_name: str) -> None:
    """Validate that a built model matches the expected pretrained profile."""
    if profile_name == "none":
        return
    profile = get_pretrained_profile(profile_name)
    checks = (
        ("num_latents", model.num_latents),
        ("embed_dim", model.embed_dim),
    )
    for key, value in checks:
        expected = profile[key]
        if value != expected:
            raise ValueError(
                f"Model {key}={value} does not match profile {profile_name} ({expected})"
            )
