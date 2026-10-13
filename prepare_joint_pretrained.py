#!/usr/bin/env python3
"""Persist the original encoder weights locally; never download or resume a scene run."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def prepare(destination, source=None):
    destination = Path(destination)
    if destination.is_file():
        state = torch.load(destination, map_location="cpu", weights_only=True)
        if "encoder.input_proj.weight" not in state or not any(k.startswith("encoder.cross_attn.") for k in state):
            raise ValueError(f"Invalid persistent encoder checkpoint: {destination}")
        print(f"Persistent pretrained encoder ready: {destination}", flush=True)
        return
    if source is None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        revision = "f90a0f7df7d5e6f71109cf333f6a95a0ae3194a6"
        filename = "hunyuan3d-vae-v2-mini-withencoder/model.fp16.ckpt"
        try:
            source = hf_hub_download("tencent/Hunyuan3D-2mini", filename,
                revision=revision, local_files_only=True)
        except LocalEntryNotFoundError:
            # HF_HOME may already point at the new, initially empty group cache.
            cached = Path.home() / ".cache/huggingface/hub/models--tencent--Hunyuan3D-2mini/snapshots" / revision / filename
            if not cached.is_file():
                raise FileNotFoundError("Original cached Hunyuan weights missing; pass --source /path/to/model.fp16.ckpt")
            source = cached
    raw = torch.load(source, map_location="cpu", weights_only=True)
    raw = raw.get("state_dict", raw)
    state = {k: v.clone() for k, v in raw.items()
             if k.startswith(("encoder.input_proj.", "encoder.cross_attn."))}
    if "encoder.input_proj.weight" not in state or not any(k.startswith("encoder.cross_attn.") for k in state):
        raise ValueError("Source contains no compatible pretrained point encoder")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    torch.save(state, temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.replace(destination)
    destination.with_suffix(".json").write_text(json.dumps({"source": str(source),
        "repo": "tencent/Hunyuan3D-2mini", "subfolder": "hunyuan3d-vae-v2-mini-withencoder",
        "keys": sorted(state), "sha256": digest}, indent=2))
    print(f"Saved {len(state)} original pretrained encoder tensors to {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", default="runs/model_cache/hunyuan_encoder_cross_attn.pt")
    parser.add_argument("--source", help="Optional original Hunyuan VAE file; default uses the existing HF cache offline")
    args = parser.parse_args()
    prepare(args.output, args.source)
