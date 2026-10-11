"""2D detections lifted to 3D and attached to geometric tracks (optional; geometry never depends on it).

Lifting modes (the plan's warning: do not project a whole image box without filtering the background):
- ``mask``:              pixels of the instance mask (optionally eroded to drop mixed boundary pixels);
- ``bbox``:              every valid pixel inside the 2D box — the naive baseline, kept for the ablation;
- ``bbox_depth_filter``: pixels inside the box restricted to the nearest dominant depth mode — an
                         approximation for detectors without masks (documented, tested, not exact).

Labels are attached to tracks by 3D overlap and accumulated as score-weighted votes; a track gets a
``semantic_label`` only with enough votes and a clear majority. A cluster without a supporting detection
keeps ``semantic_label = None``: no label is invented from geometry alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy import ndimage

from clearpath3d.contracts import RGBDFrame
from clearpath3d.geometry.frames import transform_points
from clearpath3d.geometry.projection import backproject

LIFT_MODES = ("mask", "bbox", "bbox_depth_filter")


@dataclass
class Detection2D:
    label: str
    score: float
    bbox: tuple[int, int, int, int]  # u0, v0, u1, v1 (inclusive-exclusive pixel bounds)
    mask: np.ndarray | None = None  # bool [H, W]
    source: str = "detector"


class Detector(Protocol):
    name: str

    def detect(self, frame: RGBDFrame) -> list[Detection2D]: ...


@dataclass
class Lifted:
    label: str
    score: float
    frame_id: str
    points_map: np.ndarray  # [N, 3]
    pixels: np.ndarray  # [N, 2] (v, u)
    min_m: np.ndarray
    max_m: np.ndarray
    mode: str


def _nearest_depth_mode(z: np.ndarray, bin_m: float = 0.05, min_share: float = 0.15) -> np.ndarray:
    """Keep the nearest depth mode holding at least ``min_share`` of the box pixels (+- 2 bins)."""
    if len(z) == 0:
        return np.zeros(0, bool)
    bins = np.floor((z - z.min()) / bin_m).astype(int)
    counts = np.bincount(bins)
    smooth = np.convolve(counts, np.ones(3), mode="same")
    ok = np.nonzero(smooth >= min_share * len(z))[0]
    peak = int(ok[0]) if len(ok) else int(np.argmax(smooth))
    # grow the mode while counts stay substantial
    lo = hi = peak
    while lo > 0 and counts[lo - 1] > 0.05 * smooth[peak]:
        lo -= 1
    while hi < len(counts) - 1 and counts[hi + 1] > 0.05 * smooth[peak]:
        hi += 1
    return (bins >= lo) & (bins <= hi)


def lift(
    frame: RGBDFrame, det: Detection2D, mode: str = "mask", erode_px: int = 2, min_points: int = 30
) -> Lifted | None:
    if frame.T_map_camera is None:
        return None  # no pose: cannot place it in map
    if mode not in LIFT_MODES:
        raise ValueError(mode)
    h, w = frame.depth_m.shape
    u0, v0, u1, v1 = det.bbox
    sel = np.zeros((h, w), bool)
    if mode == "mask":
        if det.mask is None:
            return None
        sel = det.mask.copy()
        if erode_px > 0:
            sel = ndimage.binary_erosion(sel, iterations=erode_px)
    else:
        sel[max(v0, 0) : min(v1, h), max(u0, 0) : min(u1, w)] = True
    sel &= frame.depth_valid
    pts, pix = backproject(frame.depth_m, sel, frame.intrinsics)
    if mode == "bbox_depth_filter" and len(pts):
        keep = _nearest_depth_mode(pts[:, 2])
        pts, pix = pts[keep], pix[keep]
    if len(pts) < min_points:
        return None
    pm = transform_points(frame.T_map_camera, pts)
    lo, hi = np.percentile(pm, 2, axis=0), np.percentile(pm, 98, axis=0)  # robust extent
    return Lifted(det.label, det.score, frame.frame_id, pm, pix, lo, hi, mode)


def _overlap_fraction(a_lo, a_hi, b_lo, b_hi) -> float:
    """Intersection volume over the smaller box volume (robust to partial views)."""
    inter = np.clip(np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo), 0, None).prod()
    va = np.clip(a_hi - a_lo, 1e-6, None).prod()
    vb = np.clip(b_hi - b_lo, 1e-6, None).prod()
    return float(inter / min(va, vb))


@dataclass
class LabelVotes:
    votes: dict[str, float] = field(default_factory=dict)
    count: int = 0
    evidence: list[tuple[str, str, float]] = field(default_factory=list)  # (frame_id, label, score)

    def add(self, label: str, score: float, frame_id: str) -> None:
        self.votes[label] = self.votes.get(label, 0.0) + score
        self.count += 1
        self.evidence.append((frame_id, label, round(score, 3)))

    def decide(self, min_votes: int = 2, min_share: float = 0.6) -> str | None:
        if self.count < min_votes or not self.votes:
            return None
        label, v = max(self.votes.items(), key=lambda kv: kv[1])
        return label if v / sum(self.votes.values()) >= min_share else None


class SemanticFusion:
    """Accumulates lifted detections onto tracks (by 3D overlap of the track's visible box)."""

    def __init__(
        self,
        min_overlap: float = 0.3,
        min_votes: int = 2,
        min_share: float = 0.6,
        pad_m: float = 0.05,
    ):
        """``pad_m`` inflates both boxes before the overlap test: a surface seen face-on has zero thickness."""
        self.min_overlap, self.min_votes, self.min_share, self.pad_m = (
            min_overlap,
            min_votes,
            min_share,
            pad_m,
        )
        self.by_track: dict[int, LabelVotes] = {}
        self.unmatched: list[Lifted] = []

    def update(self, tracks, lifted: list[Lifted]) -> None:
        for L in lifted:
            best, best_ov = None, self.min_overlap
            for t in tracks:
                c, e = np.asarray(t.x[:3]), np.asarray(t.extent)
                pad = self.pad_m
                ov = _overlap_fraction(
                    L.min_m - pad, L.max_m + pad, c - e / 2 - pad, c + e / 2 + pad
                )
                if ov >= best_ov:
                    best, best_ov = t, ov
            if best is None:
                self.unmatched.append(L)
                continue
            self.by_track.setdefault(best.track_id, LabelVotes()).add(L.label, L.score, L.frame_id)

    def label_of(self, track_id: int) -> str | None:
        v = self.by_track.get(track_id)
        return None if v is None else v.decide(self.min_votes, self.min_share)
