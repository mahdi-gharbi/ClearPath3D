"""Change detection on the voxel evidence: which voxels flipped state in a frame, and why.

- ``freed``:    occupied (L >= l_occ) before, free (L <= l_free) after — re-observed as empty;
- ``appeared``: free before, occupied after — something now stands where free space was observed.

Each voxel keeps its last *stable* state (free or occupied); an event is emitted when the current state
is the opposite stable state, even if the transition took several frames. Voxels that were never
observed cannot "change". Expiry never creates a change to free.
Events are clustered into connected components so they can be inspected individually.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class ChangeEvent:
    kind: str  # freed | appeared
    frame_id: str
    timestamp_s: float
    voxel_count: int
    center_m: list[float]
    min_m: list[float]
    max_m: list[float]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


STATE_UNKNOWN, STATE_FREE, STATE_OCC = 0, 1, 2


class ChangeDetector:
    def __init__(self, n_voxels: int, shape, grid_min, res: float, l_occ: float, l_free: float):
        self.stable = np.zeros(n_voxels, dtype=np.int8)
        self.shape, self.grid_min, self.res = shape, np.asarray(grid_min), res
        self.l_occ, self.l_free = l_occ, l_free

    def update(
        self,
        log_odds: np.ndarray,
        frame_id: str,
        t: float,
        min_voxels: int = 4,
        mask: np.ndarray | None = None,
    ) -> list[ChangeEvent]:
        """``mask`` (bool, per voxel) restricts where events are reported; states are tracked everywhere."""
        cur = np.full(len(log_odds), STATE_UNKNOWN, dtype=np.int8)
        cur[log_odds <= self.l_free] = STATE_FREE
        cur[log_odds >= self.l_occ] = STATE_OCC
        freed = (self.stable == STATE_OCC) & (cur == STATE_FREE)
        appeared = (self.stable == STATE_FREE) & (cur == STATE_OCC)
        definite = cur != STATE_UNKNOWN
        self.stable[definite] = cur[definite]
        if mask is not None:
            freed &= mask
            appeared &= mask
        return _events(
            freed, appeared, self.shape, self.grid_min, self.res, frame_id, t, min_voxels
        )


def detect_changes(
    before: np.ndarray,
    after: np.ndarray,
    shape: tuple[int, int, int],
    grid_min: np.ndarray,
    res: float,
    l_occ: float,
    l_free: float,
    frame_id: str,
    t: float,
    min_voxels: int = 4,
) -> list[ChangeEvent]:
    freed = (before >= l_occ) & (after <= l_free)
    appeared = (before <= l_free) & (after >= l_occ)
    return _events(freed, appeared, shape, grid_min, res, frame_id, t, min_voxels)


def _events(freed, appeared, shape, grid_min, res, frame_id, t, min_voxels) -> list[ChangeEvent]:
    events = []
    for kind, mask in (("freed", freed), ("appeared", appeared)):
        if not mask.any():
            continue
        vol = mask.reshape(shape)
        labels, n = ndimage.label(vol, structure=np.ones((3, 3, 3)))
        for k in range(1, n + 1):
            idx = np.argwhere(labels == k)
            if len(idx) < min_voxels:
                continue
            lo = grid_min + idx.min(axis=0) * res
            hi = grid_min + (idx.max(axis=0) + 1) * res
            events.append(
                ChangeEvent(
                    kind,
                    frame_id,
                    t,
                    int(len(idx)),
                    ((lo + hi) / 2).round(3).tolist(),
                    lo.round(3).tolist(),
                    hi.round(3).tolist(),
                )
            )
    return events
