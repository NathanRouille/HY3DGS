"""Aspect-preserving letterbox onto a white canvas (no anisotropic stretch)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def letterbox_rgb(
    image: Image.Image | np.ndarray,
    *,
    target_w: int,
    target_h: int,
    fill: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Fit ``image`` inside ``(target_w, target_h)`` with white padding.

    Preserves aspect ratio; never stretches. Useful before feeding square
    G-Objaverse renders into landscape-trained models (NOVA3R 518x392, Surflo).
    """
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            arr = np.clip(image, 0.0, 1.0)
            if arr.max() <= 1.0 + 1e-6:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = image
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        image = Image.fromarray(arr[..., :3], mode="RGB")
    else:
        image = image.convert("RGB")

    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source size {image.size}")

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), fill)
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def make_view_sheet(
    originals: list[Image.Image],
    letterboxed: list[Image.Image],
    labels: list[str],
) -> Image.Image:
    """Side-by-side original | letterbox rows for visual QA."""
    rows = []
    for orig, lb, lab in zip(originals, letterboxed, labels):
        o = orig.convert("RGB")
        l = lb.convert("RGB")
        # Match heights for the row
        h = max(o.height, l.height)
        def _pad_h(im: Image.Image, height: int) -> Image.Image:
            if im.height == height:
                return im
            c = Image.new("RGB", (im.width, height), (240, 240, 240))
            c.paste(im, (0, (height - im.height) // 2))
            return c

        o, l = _pad_h(o, h), _pad_h(l, h)
        gap = Image.new("RGB", (16, h), (200, 200, 200))
        row = Image.new("RGB", (o.width + gap.width + l.width, h), (255, 255, 255))
        row.paste(o, (0, 0))
        row.paste(gap, (o.width, 0))
        row.paste(l, (o.width + gap.width, 0))
        # Label bar
        bar = Image.new("RGB", (row.width, 28), (30, 30, 30))
        rows.append(bar)
        rows.append(row)
        rows.append(Image.new("RGB", (row.width, 8), (255, 255, 255)))

    width = max(r.width for r in rows)
    height = sum(r.height for r in rows)
    out = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    return out
