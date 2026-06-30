#!/usr/bin/env python3
"""Phase 0: Hunyuan ShapeVAE mesh reconstruction sanity check on ShapeNet.

Loads the official 7-channel SharpEdge surface (xyz | normals | sharp_label),
runs ShapeVAE encode/decode/latents2mesh, and exports meshes plus diagnostics.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import trimesh

from hy3dgen.shapegen.models.autoencoders import ShapeVAE
from hy3dgen.shapegen.pipelines import export_to_trimesh
from hy3dgen.shapegen.pretrained_profiles import HUNYUAN_MINI_PROFILE
from hy3dgen.shapegen.surface_loaders import _get_vertex_colors, load_surface_sharpegde, normalize_mesh
from train_gs_ae import resolve_category_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _sample_stem(mesh_path: str) -> str:
    """Unique name for exports (handles ShapeNet .../<id>/models/model_normalized.obj)."""
    p = Path(mesh_path)
    if p.name == "model_normalized.obj" and p.parent.name == "models":
        return p.parent.parent.name
    return p.parent.name


def discover_mesh_paths(
    data_dir: str,
    categories: Optional[Set[str]] = None,
    max_items: Optional[int] = None,
) -> List[str]:
    data_path = Path(data_dir).resolve()
    mesh_paths: List[str] = []
    for folder in sorted(data_path.iterdir()):
        if not folder.is_dir():
            continue
        if categories is not None:
            cat_id = folder.name.split("_", 1)[0]
            if cat_id not in categories:
                continue
        obj_path = folder / "model_normalized.obj"
        if obj_path.exists():
            mesh_paths.append(str(obj_path.resolve()))
    if max_items is not None:
        mesh_paths = mesh_paths[:max_items]
    return mesh_paths


def _load_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh", merge_primitives=True)
    if isinstance(mesh, trimesh.scene.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def _load_surface_and_gt_mesh(
    mesh_path: str,
    *,
    pc_size: int,
    pc_sharpedge_size: int,
) -> tuple[torch.Tensor, trimesh.Trimesh]:
    """Return VAE input surface and loader-normalized GT mesh (same coordinate frame)."""
    mesh = _load_mesh(mesh_path)
    surface, gt_mesh = load_surface_sharpegde(
        mesh,
        num_points=pc_size,
        num_sharp_points=pc_sharpedge_size,
    )
    return surface, gt_mesh


def _export_mesh(mesh: trimesh.Trimesh, path: Path) -> str:
    """Export raw recon geometry exactly like minimal_vae_demo (no fix_normals, GLB)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    return str(path)


def _export_textured_gt_mesh(mesh_path: str, path: Path) -> Dict[str, str]:
    """Export normalized GT OBJ with per-sample MTL + texture (CloudCompare-safe).

    Writes ``{stem}.obj``, ``{stem}.mtl``, and ``{stem}_texture.<ext>`` so multi-sample
    runs do not clobber shared ``material_0.png`` / ``material.mtl`` files.
    """
    stem = path.stem
    out_dir = path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_full = _load_mesh(mesh_path)
    try:
        mesh_full = trimesh.util.concatenate(mesh_full.dump())
    except Exception:
        mesh_full = trimesh.util.concatenate(mesh_full)
    mesh_full = normalize_mesh(mesh_full)
    try:
        mesh_full.fix_normals()
    except Exception:
        logger.warning("fix_normals failed for %s; exporting as-is", stem)

    has_uv_texture = (
        isinstance(mesh_full.visual, trimesh.visual.texture.TextureVisuals)
        and getattr(mesh_full.visual, "uv", None) is not None
        and mesh_full.visual.material is not None
        and getattr(mesh_full.visual.material, "image", None) is not None
    )
    if not has_uv_texture:
        vc = (_get_vertex_colors(mesh_full) * 255.0).round().astype(np.uint8)
        mesh_full.visual = trimesh.visual.ColorVisuals(
            mesh=mesh_full,
            vertex_colors=vc,
        )

    tmp_dir = out_dir / f".__mesh_export_{stem}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    paths: Dict[str, str] = {"gt_mesh": str(path)}
    try:
        tmp_obj = tmp_dir / "mesh.obj"
        mesh_full.export(tmp_obj)

        mtl_src: Optional[Path] = None
        tex_src: Optional[Path] = None
        for f in tmp_dir.iterdir():
            if f.suffix.lower() == ".mtl":
                mtl_src = f
            elif f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tga"):
                tex_src = f

        mtl_dst = out_dir / f"{stem}.mtl"
        tex_dst: Optional[Path] = None
        if tex_src is not None:
            tex_dst = out_dir / f"{stem}_texture{tex_src.suffix.lower()}"
            shutil.copy2(tex_src, tex_dst)
            paths["gt_texture"] = str(tex_dst)

        if mtl_src is not None:
            mtl_text = mtl_src.read_text()
            if tex_dst is not None:
                tex_name = tex_dst.name
                mtl_text = re.sub(
                    r"(?m)^(map_Kd\s+)\S+",
                    lambda m, n=tex_name: f"{m.group(1)}{n}",
                    mtl_text,
                )
            mtl_dst.write_text(mtl_text)
            paths["gt_mtl"] = str(mtl_dst)

        obj_text = tmp_obj.read_text()
        if mtl_dst.exists():
            obj_text = re.sub(
                r"(?m)^mtllib\s+\S+",
                f"mtllib {mtl_dst.name}",
                obj_text,
                count=1,
            )
        path.write_text(obj_text)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return paths


def _mesh_fill_face_count(mesh_path: str) -> int:
    mesh = _load_mesh(mesh_path)
    try:
        mesh_full = trimesh.util.concatenate(mesh.dump())
    except Exception:
        mesh_full = trimesh.util.concatenate(mesh)
    origin_num = mesh_full.faces.shape[0]
    mesh_fill = trimesh.Trimesh(
        vertices=mesh_full.vertices,
        faces=mesh_full.faces[origin_num:],
    )
    return int(mesh_fill.faces.shape[0])


def _surface_diagnostics(surface: torch.Tensor) -> Dict:
    arr = surface.squeeze(0).float().cpu().numpy()
    n_uniform = arr.shape[0] // 2
    uniform = arr[:n_uniform]
    sharp = arr[n_uniform:]
    return {
        "surface_shape": list(surface.shape),
        "num_channels": int(arr.shape[1]),
        "uniform_label_mean": float(uniform[:, 6].mean()) if arr.shape[1] >= 7 else None,
        "sharp_label_mean": float(sharp[:, 6].mean()) if arr.shape[1] >= 7 else None,
    }


@torch.no_grad()
def reconstruct_mesh(
    vae: ShapeVAE,
    surface: torch.Tensor,
    *,
    sample_posterior: bool,
    device: torch.device,
    dtype: torch.dtype,
    mc_kwargs: Dict,
) -> trimesh.Trimesh:
    surface = surface.to(device=device, dtype=dtype)
    latents = vae.encode(surface, sample_posterior=sample_posterior)
    latents = vae.decode(latents)
    mesh_out = vae.latents2mesh(latents, **mc_kwargs)
    mesh = export_to_trimesh(mesh_out)[0]
    return mesh


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Hunyuan ShapeVAE mesh reconstruction")
    p.add_argument("--data_dir", required=True, help="ShapeNet split directory (e.g. val/)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=15)
    p.add_argument("--indices", type=str, default=None,
                   help="Comma-separated dataset indices to evaluate")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--categories", type=str, default=None)
    p.add_argument(
        "--sample_posterior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, sample from VAE posterior (matches minimal_vae_demo); "
             "else use posterior mode (mean).",
    )
    p.add_argument("--pretrained_repo", type=str, default=HUNYUAN_MINI_PROFILE["pretrained_repo"])
    p.add_argument("--pretrained_subfolder", type=str,
                   default=HUNYUAN_MINI_PROFILE["pretrained_subfolder"])
    p.add_argument("--pc_size", type=int, default=HUNYUAN_MINI_PROFILE["pc_size"])
    p.add_argument("--pc_sharpedge_size", type=int,
                   default=HUNYUAN_MINI_PROFILE["pc_sharpedge_size"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--octree_resolution", type=int, default=256)
    p.add_argument("--mc_level", type=float, default=0.0)
    p.add_argument("--num_chunks", type=int, default=20000)
    p.add_argument("--bounds", type=float, default=1.01)
    p.add_argument(
        "--use_safetensors",
        action=argparse.BooleanOptionalAction,
        default=HUNYUAN_MINI_PROFILE.get("use_safetensors", False),
        help="Load HF weights as safetensors (Hunyuan3D-2mini ships model.fp16.ckpt only).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = resolve_category_ids(args.categories)
    mesh_paths = discover_mesh_paths(args.data_dir, categories=categories)
    logger.info("Found %d meshes in %s", len(mesh_paths), args.data_dir)

    all_indices = list(range(len(mesh_paths)))
    if args.shuffle:
        random.shuffle(all_indices)
    if args.indices:
        sample_indices = [int(x.strip()) for x in args.indices.split(",")]
    else:
        sample_indices = all_indices[: min(args.num_samples, len(all_indices))]

    logger.info("Loading ShapeVAE from %s / %s", args.pretrained_repo, args.pretrained_subfolder)
    vae = ShapeVAE.from_pretrained(
        args.pretrained_repo,
        subfolder=args.pretrained_subfolder,
        device=device,
        dtype=dtype,
        use_safetensors=args.use_safetensors,
        pc_size=args.pc_size,
        pc_sharpedge_size=args.pc_sharpedge_size,
    )
    vae.eval()

    gt_dir = output_dir / "gt"
    recon_dir = output_dir / "recon"
    gt_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    mc_kwargs = dict(
        output_type="trimesh",
        bounds=args.bounds,
        mc_level=args.mc_level,
        num_chunks=args.num_chunks,
        octree_resolution=args.octree_resolution,
        mc_algo="mc",
        enable_pbar=False,
    )

    results = []
    for rank, idx in enumerate(sample_indices):
        mesh_path = mesh_paths[idx]
        stem = _sample_stem(mesh_path)
        logger.info("[%d/%d] %s", rank + 1, len(sample_indices), stem)

        mesh_fill_faces = _mesh_fill_face_count(mesh_path)
        surface, _ = _load_surface_and_gt_mesh(
            mesh_path,
            pc_size=args.pc_size,
            pc_sharpedge_size=args.pc_sharpedge_size,
        )
        diag = _surface_diagnostics(surface)
        diag["mesh_fill_faces"] = mesh_fill_faces
        diag["mesh_path"] = mesh_path
        diag["sample_posterior"] = bool(args.sample_posterior)

        gt_paths = _export_textured_gt_mesh(mesh_path, gt_dir / f"{stem}_gt.obj")
        diag.update(gt_paths)

        try:
            recon = reconstruct_mesh(
                vae,
                surface,
                sample_posterior=args.sample_posterior,
                device=device,
                dtype=dtype,
                mc_kwargs=mc_kwargs,
            )
            recon_path = recon_dir / f"{stem}_recon.glb"
            diag["recon_mesh"] = _export_mesh(recon, recon_path)
            diag["output_mesh"] = diag["recon_mesh"]  # backward-compatible alias
            diag["status"] = "ok"
        except Exception as exc:
            logger.exception("Failed on %s: %s", mesh_path, exc)
            diag["status"] = "error"
            diag["error"] = str(exc)

        results.append(diag)

    summary = {
        "pretrained_repo": args.pretrained_repo,
        "pretrained_subfolder": args.pretrained_subfolder,
        "sample_posterior": bool(args.sample_posterior),
        "pc_size": args.pc_size,
        "pc_sharpedge_size": args.pc_sharpedge_size,
        "num_evaluated": len(results),
        "num_ok": sum(1 for r in results if r.get("status") == "ok"),
        "samples": results,
    }
    diag_path = output_dir / "diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote diagnostics to %s (%d/%d ok)", diag_path, summary["num_ok"], len(results))


if __name__ == "__main__":
    main()
