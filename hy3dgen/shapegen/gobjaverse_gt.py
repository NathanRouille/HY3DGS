"""G-Objaverse pre-rendered RGBD + camera loader for ShapeGSAE training.

G-Objaverse uses a Unity-based rendering system (Y-up, left-handed). Camera
poses in each view JSON are stored as column-basis ``(x, y, z, origin)`` in the
**render-normalized object frame**:

    v_render = (v_raw - centroid) * scale

where ``scale`` and ``bbox`` come from the view JSON (see RichDreamer
``process_unity_dataset.py``). We convert Unity cameras to the OpenGL-style
``c2w`` expected by :class:`~hy3dgen.shapegen.gs_renderer.GaussianRenderer`.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import trimesh

logger = logging.getLogger(__name__)

GOBJAVERSE_NUM_VIEWS: int = 40
GOBJAVERSE_GT_TAG: str = "gobjaverse_unity_rgbd"
GOBJAVERSE_VIEW_LAYOUT: str = "gobjaverse40"


def gobjaverse_eval_view_indices(num_loaded_views: int) -> List[int]:
    """Spread eval views for G-Objaverse periodic / offline evaluation.

    The first views in the G-Objaverse layout are nearly identical (small
    azimuth steps), so scoring only views 0..5 overestimates quality.  This
    picks ~10 well-separated views from a 38-view training set:

    * Every 5th view in the first 25 (1-based views 1, 6, 11, 16, 21)
    * Views 26 and 27
    * Views 28, 34, and 38 from the tail

    Indices are 0-based; view numbers above are 1-based as stored on disk.
    """
    if num_loaded_views <= 0:
        return []

    indices: List[int] = []
    # First 25 views (0..24): every 5th starting at 0
    indices.extend(range(0, min(25, num_loaded_views), 5))
    # Views 26-27 (indices 25-26)
    for i in (25, 26):
        if i < num_loaded_views:
            indices.append(i)
    # Views 28, 34, 38 → indices 27, 33, 37
    for i in (27, 33, 37):
        if i < num_loaded_views:
            indices.append(i)

    seen: set[int] = set()
    out: List[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out

# RichDreamer depth_warp_example near-plane heuristic (sqrt(3)/2 in unit cube).
_GOBJAVERSE_NEAR_MARGIN: float = math.sqrt(3.0) * 0.5


def _require_cv2():
    try:
        import cv2  # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for G-Objaverse GT loading "
            "(pip install opencv-python-headless)."
        ) from exc
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    return cv2


def read_gobjaverse_view_meta(json_path: Union[str, Path]) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_gobjaverse_coord_fix(vertices: np.ndarray) -> np.ndarray:
    """Undo the scene-graph rotation that trimesh applies but G-Objaverse ignores.

    GLB scene graphs typically include a Z-up-to-Y-up rotation (R_x(-90deg)).
    ``trimesh.to_geometry()`` faithfully applies this, but G-Objaverse / Unity
    rendering keeps the original Z-up mesh coordinates (with just the scale).
    We undo the rotation with R_x(+90deg): ``new_Y = -old_Z``, ``new_Z = old_Y``.
    """
    out = vertices.copy()
    old_y = vertices[:, 1].copy()
    old_z = vertices[:, 2].copy()
    out[:, 1] = -old_z
    out[:, 2] = old_y
    return out


def normalize_mesh_gobjaverse(mesh: trimesh.Trimesh, meta: Dict) -> trimesh.Trimesh:
    """Map Objaverse vertices into G-Objaverse render-normalized space.

    Applies the coordinate-system fix (undoing the scene-graph rotation) before
    centering and scaling. Preserves material / UV visuals for texture sampling.
    """
    mesh = mesh.copy()
    visual = mesh.visual
    mesh.vertices = apply_gobjaverse_coord_fix(mesh.vertices)
    scale = float(meta["scale"][0])
    centroid = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    mesh.vertices = (mesh.vertices.astype(np.float64) - centroid) * scale
    try:
        mesh.visual = visual
    except Exception:
        pass
    return mesh


def unity_c2w_from_meta(meta: Dict) -> np.ndarray:
    """Build 4×4 camera-to-world matrix from a G-Objaverse view JSON (Unity basis)."""
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = np.asarray(meta["x"], dtype=np.float64)
    c2w[:3, 1] = np.asarray(meta["y"], dtype=np.float64)
    c2w[:3, 2] = np.asarray(meta["z"], dtype=np.float64)
    c2w[:3, 3] = np.asarray(meta["origin"], dtype=np.float64)
    return c2w


def unity_c2w_to_opengl(c2w_unity: np.ndarray) -> torch.Tensor:
    """Convert G-Objaverse Unity camera pose for :class:`GaussianRenderer`.

    RichDreamer ``depth_warp_example.read_camera_matrix_single`` stores
    ``(x, -y, -z)``; ``convert_pose`` right-multiplies ``diag(1,-1,-1)`` so
    warping extrinsics use the raw Unity ``(x, y, z)`` basis. ``GaussianRenderer``
    applies the same ``gl_to_cv`` flip internally, so we pass ``(x, -y, -z)``.
    """
    c2w_gl = np.array(c2w_unity, dtype=np.float64, copy=True)
    c2w_gl[:3, 1] *= -1.0
    c2w_gl[:3, 2] *= -1.0
    return torch.from_numpy(c2w_gl.astype(np.float32))


def intrinsics_from_meta(
    meta: Dict,
    height: int,
    width: int,
) -> Tuple[float, float, float, float]:
    """Pinhole intrinsics from G-Objaverse ``x_fov`` / ``y_fov`` (radians)."""
    x_fov = float(meta["x_fov"])
    y_fov = float(meta.get("y_fov", meta["x_fov"]))
    fx = width / (2.0 * math.tan(x_fov * 0.5))
    fy = height / (2.0 * math.tan(y_fov * 0.5))
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    return fx, fy, cx, cy


def _read_rgb_image(path: Path, height: int, width: int) -> np.ndarray:
    cv2 = _require_cv2()
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read RGB image: {path}")
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        # NOTE: every G-Objaverse render is saved as RGBA -- this branch is the
        # one actually exercised in practice. It previously composited the raw
        # BGR channels (from cv2.imread) directly onto white *without* the
        # BGR -> RGB flip applied below in the 3-channel branch, silently
        # swapping Red and Blue for every GT image ever loaded for training.
        rgb = img[..., :3].astype(np.float32)[..., ::-1] / 255.0  # BGR → RGB
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        rgb = rgb * alpha + (1.0 - alpha)  # composite on white background
    else:
        rgb = img[..., :3].astype(np.float32) / 255.0
        rgb = rgb[..., ::-1]  # BGR → RGB
    if rgb.shape[0] != height or rgb.shape[1] != width:
        cv2 = _require_cv2()
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return rgb


def _read_depth_exr(
    path: Path,
    camera_origin: np.ndarray,
    max_depth: float,
    height: int,
    width: int,
) -> np.ndarray:
    """Read metric depth from ``*_nd.exr`` (4th channel), with near-plane masking."""
    cv2 = _require_cv2()
    nd = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if nd is None:
        raise FileNotFoundError(f"Failed to read depth EXR: {path}")
    if nd.shape[-1] < 4:
        raise ValueError(f"Expected 4-channel nd.exr at {path}, got shape {nd.shape}")
    depth = nd[..., 3:4].astype(np.float32)

    cam_dist = float(np.linalg.norm(camera_origin))
    near_distance = cam_dist - _GOBJAVERSE_NEAR_MARGIN
    depth = depth.copy()
    depth[depth < near_distance] = 0.0
    depth[depth > max_depth] = 0.0

    if depth.shape[0] != height or depth.shape[1] != width:
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    return depth


def gobjaverse_view_paths(render_dir: Path, view_idx: int) -> Tuple[Path, Path, Path]:
    stem = f"{view_idx:05d}"
    view_dir = render_dir / stem
    return (
        view_dir / f"{stem}.png",
        view_dir / f"{stem}_nd.exr",
        view_dir / f"{stem}.json",
    )


def gobjaverse_render_dir(render_root: Union[str, Path], gobjaverse_id: str) -> Path:
    return Path(render_root) / gobjaverse_id


def gobjaverse_gt_source_from_manifest(
    manifest: Dict,
    *,
    render_root: Optional[str] = None,
    height: int = 512,
    width: int = 512,
) -> GObjaverseGTSource:
    """Build a :class:`GObjaverseGTSource` from an experiment manifest dict."""
    root = (
        render_root
        or manifest.get("render_root")
        or os.environ.get("GOBJAVERSE_RENDER_ROOT")
    )
    if not root:
        raise ValueError(
            "G-Objaverse render_root not set in manifest or GOBJAVERSE_RENDER_ROOT."
        )
    mesh_map = manifest.get("mesh_to_gobjaverse_id") or {}
    return GObjaverseGTSource(
        render_root=str(root),
        mesh_to_gobjaverse_id=mesh_map,
        height=int(manifest.get("render_height", height)),
        width=int(manifest.get("render_width", width)),
    )


class GObjaverseGTSource:
    """Load pre-rendered G-Objaverse RGBD views + cameras for a mesh."""

    def __init__(
        self,
        render_root: str,
        mesh_to_gobjaverse_id: Dict[str, str],
        *,
        height: int = 512,
        width: int = 512,
        view_indices: Optional[List[int]] = None,
        num_views: Optional[int] = None,
    ):
        self.render_root = Path(render_root)
        self.mesh_to_gobjaverse_id = {
            os.path.realpath(k): v for k, v in mesh_to_gobjaverse_id.items()
        }
        self.height = int(height)
        self.width = int(width)

        if view_indices is not None:
            self.view_indices = [int(i) for i in view_indices]
        elif num_views is not None:
            n = min(int(num_views), GOBJAVERSE_NUM_VIEWS)
            self.view_indices = list(range(n))
        else:
            self.view_indices = list(range(GOBJAVERSE_NUM_VIEWS))

        self.num_views = len(self.view_indices)
        self._tag = GOBJAVERSE_GT_TAG
        self._meta_cache: Dict[str, Dict] = {}

    def resolve_gobjaverse_id(self, mesh_path: str) -> str:
        key = os.path.realpath(mesh_path)
        if key not in self.mesh_to_gobjaverse_id:
            raise KeyError(
                f"No G-Objaverse id for mesh {mesh_path}. "
                "Add it to manifest mesh_to_gobjaverse_id."
            )
        return self.mesh_to_gobjaverse_id[key]

    def render_dir_for_mesh(self, mesh_path: str) -> Path:
        return gobjaverse_render_dir(self.render_root, self.resolve_gobjaverse_id(mesh_path))

    def has_gt(self, mesh_path: str) -> bool:
        try:
            render_dir = self.render_dir_for_mesh(mesh_path)
        except KeyError:
            return False
        rgb, _, _ = gobjaverse_view_paths(render_dir, self.view_indices[0])
        return rgb.is_file()

    def load_meta(self, mesh_path: str) -> Dict:
        key = os.path.realpath(mesh_path)
        if key in self._meta_cache:
            return self._meta_cache[key]
        render_dir = self.render_dir_for_mesh(mesh_path)
        _, _, json_path = gobjaverse_view_paths(render_dir, 0)
        meta = read_gobjaverse_view_meta(json_path)
        self._meta_cache[key] = meta
        return meta

    def load(
        self, mesh_path: str
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[Dict]]:
        render_dir = self.render_dir_for_mesh(mesh_path)
        rgbs: List[torch.Tensor] = []
        depths: List[torch.Tensor] = []
        c2ws: List[torch.Tensor] = []
        view_params: List[Dict] = []

        for view_idx in self.view_indices:
            rgb_path, nd_path, json_path = gobjaverse_view_paths(render_dir, view_idx)
            meta = read_gobjaverse_view_meta(json_path)
            max_depth = float(meta.get("max_depth", 5.0))

            rgb_np = _read_rgb_image(rgb_path, self.height, self.width)
            c2w_unity = unity_c2w_from_meta(meta)
            depth_np = _read_depth_exr(
                nd_path,
                c2w_unity[:3, 3],
                max_depth=max_depth,
                height=self.height,
                width=self.width,
            )

            fx, fy, cx, cy = intrinsics_from_meta(meta, self.height, self.width)
            fov_x_deg = math.degrees(float(meta["x_fov"]))
            fov_y_deg = math.degrees(float(meta.get("y_fov", meta["x_fov"])))

            rgbs.append(torch.from_numpy(rgb_np).float())
            depths.append(torch.from_numpy(depth_np).float())
            c2ws.append(unity_c2w_to_opengl(c2w_unity))
            view_params.append(
                {
                    "view_idx": int(view_idx),
                    "fx": float(fx),
                    "fy": float(fy),
                    "cx": float(cx),
                    "cy": float(cy),
                    "fov_x_deg": float(fov_x_deg),
                    "fov_y_deg": float(fov_y_deg),
                    "max_depth": float(max_depth),
                    "camera_distance": float(np.linalg.norm(c2w_unity[:3, 3])),
                    "gobjaverse_id": self.resolve_gobjaverse_id(mesh_path),
                }
            )

        return rgbs, depths, c2ws, view_params
