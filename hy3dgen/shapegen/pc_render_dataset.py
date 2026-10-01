"""Surface + render dataset for ShapePCUnite joint training.

Surfaces are returned in the **camera frame** of the chosen G-Objaverse view so
they share coordinates with VGGT depth-unprojected patch centres (OpenCV-like:
x right, y down, z forward).

Sampling modes:
  - ``views_per_sample=1`` (Design A): expand to ``(mesh, view)`` pairs; each
    sample uses that view's RGB / cache / ``c2w`` camera-frame surface.
  - ``views_per_sample>1``: one sample per mesh; at load time randomly (or
    stably) pick N views from the train pool. View 0 of the pick is the
    reference camera for GT mesh + multi-view VGGT.
  - ``use_joint_vggt_cache``: load precomputed joint VGGT. If
    ``len(view_indices) == views_per_sample``, the ordered tuple is fixed
    (legacy). If the pool is larger, train randomly samples an ordered
    ``views_per_sample``-tuple (``random.sample``); eval should pass a fixed
    pair with ``view_sample_mode='first'``.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
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
from .internscenes_gt import (
    InternScenesGTSource,
    InternScenesNPSurfaceLoader,
    discover_internscenes_room_paths,
    load_internscenes_view,
)
from .surface_loaders import RGBSharpEdgeSurfaceLoader
from .vggt_context import (
    CachedVGGTContextStore,
    ordered_view_tuples,
    world_to_camera_torch,
)

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
        if getattr(self.gt_source, "KIND", None) == "internscenes":
            return load_internscenes_view(mesh_path, vid)
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

    When ``views_per_sample==1`` and ``view_indices`` has more than one entry,
    the dataset expands to one sample per ``(mesh_path, view_idx)`` pair.

    When ``views_per_sample>1``, one sample per mesh; N views are chosen from
    the pool at ``__getitem__`` time (random or first-N).
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
        views_per_sample: int = 1,
        view_sample_mode: str = "random",
        vggt_cache: Optional[CachedVGGTContextStore] = None,
        use_joint_vggt_cache: bool = False,
        gobjaverse_normalization: bool = True,
        surface_in_camera_frame: bool = True,
        align_mode: str = "cross",
        filter_missing_views: bool = True,
        strict_load: bool = False,
        internscenes_pack: bool = False,
    ):
        self.mesh_paths = mesh_paths
        self.internscenes_pack = bool(internscenes_pack)
        self.include_sharp_label = include_sharp_label
        self.view_indices = (
            [int(v) for v in view_indices]
            if view_indices is not None
            else [int(view_idx)]
        )
        if not self.view_indices:
            raise ValueError("view_indices must be non-empty")
        self.views_per_sample = max(int(views_per_sample), 1)
        if self.views_per_sample > len(self.view_indices):
            raise ValueError(
                f"views_per_sample={self.views_per_sample} > "
                f"len(view_indices)={len(self.view_indices)}"
            )
        if view_sample_mode not in ("random", "first"):
            raise ValueError(
                f"view_sample_mode must be 'random' or 'first', got {view_sample_mode!r}"
            )
        self.view_sample_mode = str(view_sample_mode)
        self.seed = None if seed is None else int(seed)
        # When True, load failures raise instead of silently substituting another mesh.
        self.strict_load = bool(strict_load)
        # Backward-compat: single default view used when callers ignore samples.
        self.view_idx = int(self.view_indices[0])
        self.vggt_cache = vggt_cache
        self.use_joint_vggt_cache = bool(use_joint_vggt_cache)
        if self.use_joint_vggt_cache:
            if self.vggt_cache is None:
                raise ValueError("use_joint_vggt_cache requires vggt_cache_root")
            if self.views_per_sample < 2:
                raise ValueError(
                    "use_joint_vggt_cache requires views_per_sample >= 2"
                )
            if len(self.view_indices) < self.views_per_sample:
                raise ValueError(
                    "use_joint_vggt_cache requires len(view_indices) >= "
                    f"views_per_sample (got {len(self.view_indices)} vs "
                    f"{self.views_per_sample})"
                )
        self.gobjaverse_normalization = bool(gobjaverse_normalization)
        # Default True: put GT xyz into the same camera frame as VGGT PE.
        self.surface_in_camera_frame = bool(surface_in_camera_frame)
        self.align_mode = str(align_mode)
        if self.internscenes_pack:
            self.surface_loader = InternScenesNPSurfaceLoader(
                num_uniform_points=pc_size,
                num_sharp_points=pc_sharpedge_size,
                seed=seed,
                include_sharp_label=include_sharp_label,
            )
        else:
            self.surface_loader = RGBSharpEdgeSurfaceLoader(
                num_uniform_points=pc_size,
                num_sharp_points=pc_sharpedge_size,
                seed=seed,
                include_sharp_label=include_sharp_label,
            )
        self.render_loader = RenderViewLoader(gt_source, view_idx=self.view_idx) if gt_source else None
        if not self.mesh_paths:
            raise FileNotFoundError(f"No meshes under {data_dir}")

        # Per-mesh available views from the pool (after filter).
        self._mesh_available: Dict[str, List[int]] = {}
        self.samples: List[Tuple[str, int]] = self._build_samples(
            filter_missing=filter_missing_views
        )
        logger.info(
            "SurfaceRenderDataset: %d meshes, pool=%d view(s), views_per_sample=%d "
            "(%s) → %d samples, align_mode=%s, render=%s, vggt_cache=%s, "
            "joint_vggt_cache=%s, "
            "gobjaverse_norm=%s, surface_frame=%s, seed=%s, strict_load=%s",
            len(self.mesh_paths),
            len(self.view_indices),
            self.views_per_sample,
            self.view_sample_mode,
            len(self.samples),
            self.align_mode,
            gt_source is not None,
            vggt_cache is not None,
            self.use_joint_vggt_cache,
            self.gobjaverse_normalization and gt_source is not None,
            "camera" if self.surface_in_camera_frame else "object",
            self.seed,
            self.strict_load,
        )

    def _available_pool_views(self, path: str, available: Optional[Set[int]]) -> List[int]:
        """Pool views that exist on disk for this mesh (order = view_indices)."""
        out: List[int] = []
        for vid in self.view_indices:
            if available is not None and vid not in available:
                continue
            out.append(int(vid))
        return out

    def _build_samples(self, *, filter_missing: bool) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        skipped_render = 0
        skipped_cache = 0
        skipped_too_few = 0
        multi = self.views_per_sample > 1
        # Multi-view PE is online unless joint cache is enabled.
        require_cache = (
            (not multi) and self.vggt_cache is not None
        ) or (
            multi and self.use_joint_vggt_cache and self.vggt_cache is not None
        )
        joint_fixed = (
            self.use_joint_vggt_cache
            and len(self.view_indices) == self.views_per_sample
        )
        joint_pool = self.use_joint_vggt_cache and not joint_fixed

        for path in self.mesh_paths:
            available: Optional[Set[int]] = None
            if filter_missing and self.render_loader is not None:
                try:
                    gs = self.render_loader.gt_source
                    if getattr(gs, "KIND", None) == "internscenes":
                        available = gs.list_available_views(path, self.view_indices)
                    else:
                        rd = gs.render_dir_for_mesh(path)
                        available = set(list_available_gobjaverse_views(rd))
                except Exception as e:
                    logger.warning("No render dir for %s: %s", path, e)
                    available = set()
            pool = self._available_pool_views(path, available)
            self._mesh_available[path] = pool

            if multi:
                if len(pool) < self.views_per_sample:
                    skipped_too_few += 1
                    continue
                if self.use_joint_vggt_cache:
                    if joint_fixed:
                        if not all(v in pool for v in self.view_indices):
                            skipped_too_few += 1
                            continue
                        needed = [list(self.view_indices)]
                    else:
                        # All ordered k-tuples among available pool views.
                        needed = ordered_view_tuples(
                            pool, tuple_size=self.views_per_sample
                        )
                    missing = [
                        t for t in needed if not self.vggt_cache.has_joint(path, t)
                    ]
                    if missing:
                        skipped_cache += 1
                        continue
                # Sentinel view_idx=-1 → choose N views in __getitem__.
                samples.append((path, -1))
                continue

            for vid in pool:
                if require_cache and not self.vggt_cache.has(path, vid):
                    skipped_cache += 1
                    continue
                samples.append((path, int(vid)))
            skipped_render += max(0, len(self.view_indices) - len(pool))

        if skipped_render:
            logger.info("Skipped %d (mesh, view) pairs with missing renders", skipped_render)
        if skipped_cache:
            logger.info(
                "Skipped %d (mesh, view) pairs missing from VGGT cache "
                + (
                    "(cache all ordered joint pairs with "
                    "cache_vggt_features.py --joint_pairs)"
                    if joint_pool
                    else (
                        "(cache joint views before multi-view train)"
                        if self.use_joint_vggt_cache
                        else "(cache all views before multi-view train)"
                    )
                ),
                skipped_cache,
            )
        if skipped_too_few:
            logger.info(
                "Skipped %d meshes with fewer than %d available pool views",
                skipped_too_few,
                self.views_per_sample,
            )
        if not samples:
            raise FileNotFoundError(
                f"No samples from {len(self.mesh_paths)} meshes "
                f"and views {self.view_indices}"
                + (
                    " — cache is empty for these views; run cache_vggt_features.py first"
                    if require_cache
                    else ""
                )
            )
        return samples

    def _choose_views(self, path: str, idx: int) -> List[int]:
        pool = list(self._mesh_available.get(path) or self.view_indices)
        k = self.views_per_sample
        if len(pool) < k:
            raise RuntimeError(
                f"{path}: only {len(pool)} pool views, need views_per_sample={k}"
            )
        # Fixed ordered tuple: exact view_indices (eval / legacy joint), or
        # view_sample_mode=first → first-k of the available pool.
        if self.view_sample_mode == "first":
            if (
                self.use_joint_vggt_cache
                and len(self.view_indices) == self.views_per_sample
            ):
                return list(self.view_indices)
            return pool[:k]
        # Fresh random ordered k-tuple each __getitem__ (covers all P(n,k)).
        del idx  # unused; keeps signature stable for callers
        return random.sample(pool, k)

    def _surface_meta(self, path: str) -> Optional[Dict]:
        if not self.gobjaverse_normalization or self.render_loader is None:
            return None
        try:
            return self.render_loader.gt_source.load_meta(path)
        except Exception as e:
            logger.warning("No G-Objaverse meta for %s: %s", path, e)
            return None

    def _surface_to_camera(
        self, surface: torch.Tensor, c2w: torch.Tensor
    ) -> torch.Tensor:
        xyz = surface[:, :3]
        xyz_cam = world_to_camera_torch(xyz, c2w)
        out = surface.clone()
        out[:, :3] = xyz_cam
        if out.shape[-1] >= 6:
            nrm = out[:, 3:6]
            R = c2w[:3, :3]
            out[:, 3:6] = nrm @ R
        return out

    def _apply_gt_align(
        self,
        surface: torch.Tensor,
        *,
        views: List[Dict],
        c2w_ref: torch.Tensor,
        vggt_cache: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """Normalize GT xyz for ``align_mode``. Returns (surface, mean_z_or_None)."""
        if not self.surface_in_camera_frame:
            return surface, None

        if self.align_mode == "c_meanrms":
            from hy3dgen.shapegen.cam_align import (
                align_gt_xyz,
                compute_c_meanrms_gt_stats,
            )

            depths = []
            rgbs = []
            Ks = []
            c2ws = []
            for v in views:
                d = v["depth"]
                depths.append(d.numpy() if torch.is_tensor(d) else np.asarray(d))
                rgb = v["rgb"]
                if torch.is_tensor(rgb):
                    rgbs.append(rgb.permute(1, 2, 0).numpy())
                else:
                    rgbs.append(np.asarray(rgb))
                Ks.append(v["intrinsics"].numpy() if torch.is_tensor(v["intrinsics"]) else v["intrinsics"])
                c2ws.append(
                    v["c2w"].numpy() if torch.is_tensor(v["c2w"]) else np.asarray(v["c2w"])
                )
            depth_masks = None
            if views and all("depth_mask" in v for v in views):
                depth_masks = []
                for v in views:
                    m = v["depth_mask"]
                    depth_masks.append(
                        m.numpy() if torch.is_tensor(m) else np.asarray(m)
                    )
            stats = compute_c_meanrms_gt_stats(
                depths,
                rgbs,
                Ks,
                c2ws,
                c2w_ref.numpy() if torch.is_tensor(c2w_ref) else np.asarray(c2w_ref),
                erode_iters=1,
                depth_masks=depth_masks,
            )
            surf = surface.clone()
            surf[:, :3] = align_gt_xyz(
                surf[:, :3],
                mean_z_gt_depth=1.0,
                align_stats={"c_meanrms": stats},
                mode="c_meanrms",
            )
            return surf, None

        # cross / fair_gobK: need cache align_stats + GT-depth mean_z
        if (
            vggt_cache is None
            or not isinstance(vggt_cache, dict)
            or "align_stats" not in vggt_cache
        ):
            return surface, None

        from hy3dgen.shapegen.cam_align import align_gt_xyz, gt_depth_mean_z

        ref = views[0]
        mz = gt_depth_mean_z(ref["depth"], ref["intrinsics"], ref.get("rgb"))
        surf = surface.clone()
        surf[:, :3] = align_gt_xyz(
            surf[:, :3],
            mean_z_gt_depth=mz,
            align_stats=vggt_cache["align_stats"],
            mode=self.align_mode,
        )
        return surf, float(mz)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        n = len(self.samples)
        if self.strict_load:
            path, view_idx = self.samples[idx % n]
            try:
                if self.views_per_sample > 1:
                    chosen = self._choose_views(path, idx)
                    return self._load_multi_view_sample(path, chosen)
                return self._load_single_view_sample(path, int(view_idx))
            except Exception as e:
                raise RuntimeError(
                    f"strict_load: failed to load sample idx={idx} "
                    f"path={path} view={view_idx}: {e}"
                ) from e
        for attempt in range(n):
            path, view_idx = self.samples[(idx + attempt) % n]
            try:
                if self.views_per_sample > 1:
                    chosen = self._choose_views(path, idx + attempt)
                    return self._load_multi_view_sample(path, chosen)
                return self._load_single_view_sample(path, int(view_idx))
            except Exception as e:
                logger.warning("Skipping %s view %s: %s", path, view_idx, e)
        raise RuntimeError(f"All {n} samples failed to load")

    def _load_single_view_sample(self, path: str, view_idx: int) -> Dict:
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
                surface = self._surface_to_camera(surface, view["c2w"])
                out["surface"] = surface
                out["surface_frame"] = "camera"
            cached = None
            if self.vggt_cache is not None:
                cached = self.vggt_cache.load(path, view_idx)
                if cached is not None:
                    out["vggt_cache"] = cached
            surf, mz = self._apply_gt_align(
                out["surface"],
                views=[view] if "depth" in out else [],
                c2w_ref=view["c2w"],
                vggt_cache=cached if isinstance(cached, dict) else None,
            )
            out["surface"] = surf
            out["align_mode"] = self.align_mode
            if mz is not None:
                out["gt_depth_mean_z"] = torch.tensor(mz, dtype=torch.float32)
        return out

    def _load_multi_view_sample(self, path: str, view_ids: List[int]) -> Dict:
        """One object + N views; ref = view_ids[0] (VGGT + GT camera frame)."""
        if self.render_loader is None:
            raise RuntimeError("Multi-view samples require a render GT source")
        surface = self.surface_loader(
            path, gobjaverse_meta=self._surface_meta(path)
        ).squeeze(0)
        views = [self.render_loader.load_view(path, view_idx=int(v)) for v in view_ids]
        ref = views[0]
        if self.surface_in_camera_frame:
            surface = self._surface_to_camera(surface, ref["c2w"])
            surface_frame = "camera"
        else:
            surface_frame = "object"

        surface, mz = self._apply_gt_align(
            surface,
            views=views,
            c2w_ref=ref["c2w"],
            vggt_cache=None,
        )
        joint_cached = None
        if self.use_joint_vggt_cache and self.vggt_cache is not None:
            joint_cached = self.vggt_cache.load_joint(path, view_ids)
            if joint_cached is None:
                raise FileNotFoundError(
                    f"Missing joint VGGT cache for {path} views {view_ids}"
                )

        rgb_views = torch.stack([v["rgb"] for v in views], dim=0)  # [S,3,H,W]
        depth_views = torch.stack([v["depth"] for v in views], dim=0)
        intrinsics_views = torch.stack([v["intrinsics"] for v in views], dim=0)
        c2w_views = torch.stack([v["c2w"] for v in views], dim=0)
        view_idx_t = torch.tensor([int(v) for v in view_ids], dtype=torch.long)

        out: Dict = {
            "surface": surface,
            "mesh_path": path,
            "surface_frame": surface_frame,
            # Ref view fields (compat with single-view collate / debug).
            "rgb": ref["rgb"],
            "depth": ref["depth"],
            "intrinsics": ref["intrinsics"],
            "c2w": ref["c2w"],
            "view_idx": view_idx_t[0],
            "rgb_views": rgb_views,
            "depth_views": depth_views,
            "intrinsics_views": intrinsics_views,
            "c2w_views": c2w_views,
            "view_indices": view_idx_t,
            "align_mode": self.align_mode,
        }
        if joint_cached is not None:
            out["vggt_cache"] = joint_cached
        if mz is not None:
            out["gt_depth_mean_z"] = torch.tensor(mz, dtype=torch.float32)
        return out


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
    if all("rgb_views" in b for b in batch):
        # [B,S,3,H,W] — S must match across the batch (same views_per_sample).
        out["rgb_views"] = torch.stack([b["rgb_views"] for b in batch], dim=0)
        out["depth_views"] = torch.stack([b["depth_views"] for b in batch], dim=0)
        out["intrinsics_views"] = torch.stack(
            [b["intrinsics_views"] for b in batch], dim=0
        )
        out["c2w_views"] = torch.stack([b["c2w_views"] for b in batch], dim=0)
        out["view_indices"] = torch.stack([b["view_indices"] for b in batch], dim=0)
    if all("vggt_cache" in b for b in batch):
        out["vggt_cache"] = [b["vggt_cache"] for b in batch]
    elif any("vggt_cache" in b for b in batch):
        raise RuntimeError("Inconsistent vggt_cache across batch items")
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
    views_per_sample: int = 1,
    view_sample_mode: str = "random",
    vggt_cache_root: Optional[str] = None,
    use_joint_vggt_cache: bool = False,
    pc_size: int = 5120,
    pc_sharpedge_size: int = 5120,
    seed: Optional[int] = None,
    gobjaverse_normalization: bool = True,
    surface_in_camera_frame: bool = True,
    world_scale: float = 1.0,  # deprecated — ignored (kept for CLI compat)
    align_mode: str = "cross",
    filter_missing_views: bool = True,
    strict_load: bool = False,
) -> SurfaceRenderDataset:
    from train_pc_ae import discover_mesh_paths

    if world_scale != 1.0:
        logger.warning(
            "world_scale=%s is deprecated and ignored; surfaces live in camera "
            "frame without an extra rescale.",
            world_scale,
        )
    if align_mode not in ("cross", "fair_gobK", "c_meanrms"):
        raise ValueError(
            f"align_mode must be 'cross', 'fair_gobK', or 'c_meanrms', got {align_mode!r}"
        )

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
        views_per_sample=views_per_sample,
        view_sample_mode=view_sample_mode,
        vggt_cache=cache,
        use_joint_vggt_cache=use_joint_vggt_cache,
        gobjaverse_normalization=gobjaverse_normalization,
        surface_in_camera_frame=surface_in_camera_frame,
        align_mode=align_mode,
        filter_missing_views=filter_missing_views,
        strict_load=strict_load,
        internscenes_pack=False,
    )


def build_internscenes_render_dataset(
    pack_root: str,
    *,
    split: str = "train",
    room_ids: Optional[List[str]] = None,
    max_items: Optional[int] = None,
    include_sharp_label: bool = False,
    view_indices: Optional[List[int]] = None,
    views_per_sample: int = 1,
    view_sample_mode: str = "random",
    vggt_cache_root: Optional[str] = None,
    use_joint_vggt_cache: bool = False,
    pc_size: int = 5120,
    pc_sharpedge_size: int = 5120,
    seed: Optional[int] = None,
    surface_in_camera_frame: bool = True,
    align_mode: str = "c_meanrms",
    filter_missing_views: bool = True,
    strict_load: bool = False,
) -> SurfaceRenderDataset:
    """InternScenes bathroom pack (or any ``pack/rooms/<id>/`` layout)."""
    if align_mode not in ("cross", "fair_gobK", "c_meanrms"):
        raise ValueError(f"bad align_mode {align_mode!r}")
    if not view_indices:
        raise ValueError("InternScenes dataset requires explicit view_indices")
    resolved_views = [int(v) for v in view_indices]
    room_paths = discover_internscenes_room_paths(
        pack_root,
        split=split,
        room_ids=room_ids,
        max_items=max_items,
    )
    gt_source = InternScenesGTSource(pack_root)
    cache = CachedVGGTContextStore(vggt_cache_root) if vggt_cache_root else None
    return SurfaceRenderDataset(
        pack_root,
        room_paths,
        pc_size=pc_size,
        pc_sharpedge_size=pc_sharpedge_size,
        seed=seed,
        include_sharp_label=include_sharp_label,
        gt_source=gt_source,
        view_idx=resolved_views[0],
        view_indices=resolved_views,
        views_per_sample=views_per_sample,
        view_sample_mode=view_sample_mode,
        vggt_cache=cache,
        use_joint_vggt_cache=use_joint_vggt_cache,
        gobjaverse_normalization=False,
        surface_in_camera_frame=surface_in_camera_frame,
        align_mode=align_mode,
        filter_missing_views=filter_missing_views,
        strict_load=strict_load,
        internscenes_pack=True,
    )
