"""Surface + render dataset for ShapePCUnite joint training.

Surfaces are returned in the **camera frame** of the chosen G-Objaverse view so
they share coordinates with VGGT depth-unprojected patch centres (OpenCV-like:
x right, y down, z forward).

Training can expand to ``(mesh, view)`` pairs (Design A): each sample uses that
view's RGB / cache / ``c2w`` camera-frame surface. Eval typically keeps a single
fixed ``view_idx`` for comparability.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import Dataset

from .gobjaverse_gt import (
    GObjaverseGTSource,
    gobjaverse_gt_source_from_manifest,
    gobjaverse_view_paths,
    intrinsics_from_meta,
    list_available_gobjaverse_views,
    parse_view_indices,
    read_gobjaverse_view_meta,
    unity_c2w_from_meta,
    _read_depth_exr,
    _read_rgb_image,
)
from .surface_loaders import RGBSharpEdgeSurfaceLoader
from .vggt_context import CachedVGGTContextStore, world_to_camera_torch

logger = logging.getLogger(__name__)


class RenderViewLoader:
    """Load a G-Objaverse RGBD view for a mesh (optional default ``view_idx``)."""

    def __init__(
        self,
        gt_source: GObjaverseGTSource,
        *,
        view_idx: int = 0,
    ):
        self.gt_source = gt_source
        self.view_idx = int(view_idx)

    def load_view(
        self, mesh_path: str, view_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        vid = self.view_idx if view_idx is None else int(view_idx)
        render_dir = self.gt_source.render_dir_for_mesh(mesh_path)
        rgb_path, nd_path, json_path = gobjaverse_view_paths(render_dir, vid)
        meta = read_gobjaverse_view_meta(json_path)
        h, w = self.gt_source.height, self.gt_source.width
        max_depth = float(meta.get("max_depth", 5.0))
        c2w_unity = unity_c2w_from_meta(meta)

        rgb_np = _read_rgb_image(rgb_path, h, w)
        depth_np = _read_depth_exr(
            nd_path, c2w_unity[:3, 3], max_depth=max_depth, height=h, width=w
        )
        fx, fy, cx, cy = intrinsics_from_meta(meta, h, w)

        return {
            "rgb": torch.from_numpy(rgb_np).float().permute(2, 0, 1),  # CHW
            "depth": torch.from_numpy(depth_np[..., 0]).float(),
            "intrinsics": torch.tensor([fx, fy, cx, cy], dtype=torch.float32),
            "c2w": torch.from_numpy(c2w_unity.astype("float32")),
            "view_idx": torch.tensor(vid, dtype=torch.long),
        }


class SurfaceRenderDataset(Dataset):
    """Mesh surface (camera frame) + render view (+ optional VGGT cache).

    When ``view_indices`` has more than one entry, the dataset expands to one
    sample per ``(mesh_path, view_idx)`` pair (shuffled independently by the
    DataLoader).
    """

    def __init__(
        self,
        data_dir: str,
        mesh_paths: List[str],
        *,
        pc_size: int = 5120,
        pc_sharpedge_size: int = 5120,
        seed: Optional[int] = None,
        include_sharp_label: bool = False,
        gt_source: Optional[GObjaverseGTSource] = None,
        view_idx: int = 0,
        view_indices: Optional[List[int]] = None,
        vggt_cache: Optional[CachedVGGTContextStore] = None,
        gobjaverse_normalization: bool = True,
        surface_in_camera_frame: bool = True,
        align_mode: str = "cross",
        filter_missing_views: bool = True,
    ):
        self.mesh_paths = mesh_paths
        self.include_sharp_label = include_sharp_label
        self.view_indices = (
            [int(v) for v in view_indices]
            if view_indices is not None
            else [int(view_idx)]
        )
        if not self.view_indices:
            raise ValueError("view_indices must be non-empty")
        # Backward-compat: single default view used when callers ignore samples.
        self.view_idx = int(self.view_indices[0])
        self.vggt_cache = vggt_cache
        self.gobjaverse_normalization = bool(gobjaverse_normalization)
        # Default True: put GT xyz into the same camera frame as VGGT PE.
        self.surface_in_camera_frame = bool(surface_in_camera_frame)
        self.align_mode = str(align_mode)
        self.surface_loader = RGBSharpEdgeSurfaceLoader(
            num_uniform_points=pc_size,
            num_sharp_points=pc_sharpedge_size,
            seed=seed,
            include_sharp_label=include_sharp_label,
        )
        self.render_loader = RenderViewLoader(gt_source, view_idx=self.view_idx) if gt_source else None
        if not self.mesh_paths:
            raise FileNotFoundError(f"No meshes under {data_dir}")

        self.samples: List[Tuple[str, int]] = self._build_samples(
            filter_missing=filter_missing_views
        )
        logger.info(
            "SurfaceRenderDataset: %d meshes × %d view(s) → %d samples, "
            "render=%s, vggt_cache=%s, gobjaverse_norm=%s, surface_frame=%s",
            len(self.mesh_paths),
            len(self.view_indices),
            len(self.samples),
            gt_source is not None,
            vggt_cache is not None,
            self.gobjaverse_normalization and gt_source is not None,
            "camera" if self.surface_in_camera_frame else "object",
        )

    def _build_samples(self, *, filter_missing: bool) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        skipped_render = 0
        skipped_cache = 0
        for path in self.mesh_paths:
            available: Optional[Set[int]] = None
            if filter_missing and self.render_loader is not None:
                try:
                    rd = self.render_loader.gt_source.render_dir_for_mesh(path)
                    available = set(list_available_gobjaverse_views(rd))
                except Exception as e:
                    logger.warning("No render dir for %s: %s", path, e)
                    available = set()
            for vid in self.view_indices:
                if available is not None and vid not in available:
                    skipped_render += 1
                    continue
                # When a cache root is configured, only keep pairs that are cached
                # so multi-view batches never mix present/missing weak context.
                if self.vggt_cache is not None and not self.vggt_cache.has(path, vid):
                    skipped_cache += 1
                    continue
                samples.append((path, int(vid)))
        if skipped_render:
            logger.info("Skipped %d (mesh, view) pairs with missing renders", skipped_render)
        if skipped_cache:
            logger.info(
                "Skipped %d (mesh, view) pairs missing from VGGT cache "
                "(cache all views before multi-view train)",
                skipped_cache,
            )
        if not samples:
            raise FileNotFoundError(
                f"No (mesh, view) samples from {len(self.mesh_paths)} meshes "
                f"and views {self.view_indices}"
                + (
                    " — cache is empty for these views; run cache_vggt_features.py first"
                    if self.vggt_cache is not None
                    else ""
                )
            )
        return samples

    def _surface_meta(self, path: str) -> Optional[Dict]:
        if not self.gobjaverse_normalization or self.render_loader is None:
            return None
        try:
            return self.render_loader.gt_source.load_meta(path)
        except Exception as e:
            logger.warning("No G-Objaverse meta for %s: %s", path, e)
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        n = len(self.samples)
        for attempt in range(n):
            path, view_idx = self.samples[(idx + attempt) % n]
            try:
                surface = self.surface_loader(
                    path, gobjaverse_meta=self._surface_meta(path)
                ).squeeze(0)
                out: Dict = {
                    "surface": surface,
                    "mesh_path": path,
                    "surface_frame": "object",
                    "view_idx": torch.tensor(view_idx, dtype=torch.long),
                }
                if self.render_loader is not None and self.render_loader.gt_source.has_gt(path):
                    view = self.render_loader.load_view(path, view_idx=view_idx)
                    out.update(view)
                    if self.surface_in_camera_frame:
                        # Transform xyz only; normals/rgb stay as stored channels.
                        xyz = surface[:, :3]
                        xyz_cam = world_to_camera_torch(xyz, view["c2w"])
                        surface = surface.clone()
                        surface[:, :3] = xyz_cam
                        # Rotate normals into camera frame too (same R).
                        if surface.shape[-1] >= 6:
                            nrm = surface[:, 3:6]
                            R = view["c2w"][:3, :3]
                            surface[:, 3:6] = nrm @ R
                        out["surface"] = surface
                        out["surface_frame"] = "camera"
                if self.vggt_cache is not None:
                    cached = self.vggt_cache.load(path, view_idx)
                    if cached is not None:
                        out["vggt_cache"] = cached
                # Cross / fair_gobK: /mean(GT depth) + Hunyuan bbox from cache stats
                if (
                    self.surface_in_camera_frame
                    and "depth" in out
                    and "vggt_cache" in out
                    and isinstance(out["vggt_cache"], dict)
                    and "align_stats" in out["vggt_cache"]
                ):
                    from hy3dgen.shapegen.cam_align import align_gt_xyz, gt_depth_mean_z

                    mz = gt_depth_mean_z(
                        out["depth"], out["intrinsics"], out.get("rgb")
                    )
                    surf = out["surface"].clone()
                    surf[:, :3] = align_gt_xyz(
                        surf[:, :3],
                        mean_z_gt_depth=mz,
                        align_stats=out["vggt_cache"]["align_stats"],
                        mode=self.align_mode,
                    )
                    out["surface"] = surf
                    out["align_mode"] = self.align_mode
                    out["gt_depth_mean_z"] = torch.tensor(mz, dtype=torch.float32)
                return out
            except Exception as e:
                logger.warning("Skipping %s view %d: %s", path, view_idx, e)
        raise RuntimeError(f"All {n} (mesh, view) samples failed to load")


def collate_surface_render(batch: List[Dict]) -> Dict:
    out: Dict = {
        "surface": torch.stack([b["surface"] for b in batch], dim=0),
        "mesh_path": [b["mesh_path"] for b in batch],
        "surface_frame": [b.get("surface_frame", "object") for b in batch],
        "view_idx": torch.stack(
            [
                b["view_idx"]
                if torch.is_tensor(b["view_idx"])
                else torch.tensor(int(b["view_idx"]), dtype=torch.long)
                for b in batch
            ],
            dim=0,
        ),
    }
    if "rgb" in batch[0]:
        out["rgb"] = torch.stack([b["rgb"] for b in batch], dim=0)
        out["depth"] = torch.stack([b["depth"] for b in batch], dim=0)
        out["intrinsics"] = torch.stack([b["intrinsics"] for b in batch], dim=0)
        out["c2w"] = torch.stack([b["c2w"] for b in batch], dim=0)
    if all("vggt_cache" in b for b in batch):
        out["vggt_cache"] = [b["vggt_cache"] for b in batch]
    return out


def build_surface_render_dataset(
    data_dir: str,
    *,
    max_items: Optional[int] = None,
    categories: Optional[Set[str]] = None,
    include_sharp_label: bool = False,
    use_experiment_manifest: bool = True,
    manifest: Optional[Dict] = None,
    render_root: Optional[str] = None,
    view_idx: int = 0,
    view_indices: Optional[List[int]] = None,
    num_views: Optional[int] = None,
    vggt_cache_root: Optional[str] = None,
    pc_size: int = 5120,
    pc_sharpedge_size: int = 5120,
    seed: Optional[int] = None,
    gobjaverse_normalization: bool = True,
    surface_in_camera_frame: bool = True,
    world_scale: float = 1.0,  # deprecated — ignored (kept for CLI compat)
    align_mode: str = "cross",
    filter_missing_views: bool = True,
) -> SurfaceRenderDataset:
    from train_pc_ae import discover_mesh_paths

    if world_scale != 1.0:
        logger.warning(
            "world_scale=%s is deprecated and ignored; surfaces live in camera "
            "frame without an extra rescale.",
            world_scale,
        )
    if align_mode not in ("cross", "fair_gobK"):
        raise ValueError(f"align_mode must be 'cross' or 'fair_gobK', got {align_mode!r}")

    resolved_views = (
        [int(v) for v in view_indices]
        if view_indices is not None
        else parse_view_indices(view_idx=view_idx, num_views=num_views, view_indices=None)
    )

    mesh_paths = discover_mesh_paths(
        data_dir,
        categories=categories,
        max_items=max_items,
        use_experiment_manifest=use_experiment_manifest,
    )
    gt_source = None
    if manifest is not None and manifest.get("gt_source") == "gobjaverse":
        gt_source = gobjaverse_gt_source_from_manifest(manifest, render_root=render_root)
    cache = CachedVGGTContextStore(vggt_cache_root) if vggt_cache_root else None
    return SurfaceRenderDataset(
        data_dir,
        mesh_paths,
        pc_size=pc_size,
        pc_sharpedge_size=pc_sharpedge_size,
        seed=seed,
        include_sharp_label=include_sharp_label,
        gt_source=gt_source,
        view_idx=resolved_views[0],
        view_indices=resolved_views,
        vggt_cache=cache,
        gobjaverse_normalization=gobjaverse_normalization,
        surface_in_camera_frame=surface_in_camera_frame,
        align_mode=align_mode,
        filter_missing_views=filter_missing_views,
    )
