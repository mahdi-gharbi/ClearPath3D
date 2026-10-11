"""Render a 2D map (and optional path / markers) to an RGB image. Display only; x right, y up."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from clearpath3d.mapping.projection import FREE, OCCUPIED, Map2D

COLORS = {
    "unknown": (150, 150, 160),
    "free": (245, 245, 240),
    "occupied": (30, 30, 35),
    "path": (220, 40, 40),
    "start": (40, 160, 60),
    "goal": (40, 90, 220),
    "marker": (240, 150, 0),
}


def render_map(
    m: Map2D,
    path_m: list[list[float]] | None = None,
    start_xy=None,
    goal_xy=None,
    markers_xy: list | None = None,
    scale: int = 4,
) -> np.ndarray:
    img = np.empty(m.state.shape + (3,), dtype=np.uint8)
    img[:] = COLORS["unknown"]
    img[m.state == FREE] = COLORS["free"]
    img[m.state == OCCUPIED] = COLORS["occupied"]

    def put(xy, color, r=1):
        i, j = m.cell_of(np.asarray(xy))
        img[max(i - r, 0) : i + r + 1, max(j - r, 0) : j + r + 1] = color

    if path_m:
        P = np.asarray(path_m)
        for a, b in zip(P[:-1], P[1:], strict=True):
            for t in np.linspace(0, 1, int(np.linalg.norm(b - a) / m.resolution_m * 2) + 2):
                put(a + (b - a) * t, COLORS["path"], 0)
    for xy in markers_xy or []:
        put(xy, COLORS["marker"], 1)
    if start_xy is not None:
        put(start_xy, COLORS["start"], 2)
    if goal_xy is not None:
        put(goal_xy, COLORS["goal"], 2)
    # array [i=x, j=y] -> image rows = -y, cols = x
    out = np.flipud(np.transpose(img, (1, 0, 2)))
    return np.repeat(np.repeat(out, scale, axis=0), scale, axis=1)


def save_png(img: np.ndarray, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_image(str(path), o3d.geometry.Image(np.ascontiguousarray(img)))
