"""Frontiers (observed-free cells touching unknown) and the region reachable through known free space."""

from __future__ import annotations

import heapq
import math

import numpy as np
from scipy import ndimage

from clearpath3d.mapping.projection import FREE, UNKNOWN, Map2D


def frontier_cells(m: Map2D, min_cluster: int = 3) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (mask, clusters). A frontier cell is FREE with a 4-neighbour UNKNOWN cell."""
    unknown = m.state == UNKNOWN
    near_unknown = ndimage.binary_dilation(
        unknown, structure=ndimage.generate_binary_structure(2, 1)
    )
    mask = (m.state == FREE) & near_unknown
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    clusters = [np.argwhere(labels == k) for k in range(1, n + 1)]
    clusters = [c for c in clusters if len(c) >= min_cluster]
    keep = np.zeros_like(mask)
    for c in clusters:
        keep[c[:, 0], c[:, 1]] = True
    return keep, clusters


def travel_distance(blocked: np.ndarray, start: tuple[int, int], res: float) -> np.ndarray:
    """Shortest 8-connected distance (m) from ``start`` over non-blocked cells (no corner cutting).

    ``inf`` = not reachable through known free space for this robot footprint. Used to rank and
    filter candidates; the executed path is re-planned with A* and its real length is reported.
    """
    nx, ny = blocked.shape
    dist = np.full((nx, ny), np.inf)
    if blocked[start]:
        return dist
    dist[start] = 0.0
    heap = [(0.0, start)]
    sq2 = math.sqrt(2.0)
    while heap:
        d, (i, j) = heapq.heappop(heap)
        if d > dist[i, j]:
            continue
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            a, b = i + di, j + dj
            if not (0 <= a < nx and 0 <= b < ny) or blocked[a, b]:
                continue
            if di and dj and (blocked[i + di, j] or blocked[i, j + dj]):
                continue
            step = res * (sq2 if di and dj else 1.0)
            nd = d + step
            if nd < dist[a, b]:
                dist[a, b] = nd
                heapq.heappush(heap, (nd, (a, b)))
    return dist
