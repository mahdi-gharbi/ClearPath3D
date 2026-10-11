"""Ground-plane detection with explicit acceptance criteria.

The largest plane is *not* assumed to be the floor. Up to ``max_candidates`` planes are extracted
by sequential RANSAC; each is accepted only if it is (1) within ``max_tilt_deg`` of the gravity
prior, (2) at a plausible height, and (3) supported by enough points *and* observed area.
Points must be expressed in a gravity-aligned frame (z up), typically ``base_link``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GroundParams:
    distance_threshold_m: float = 0.03
    max_tilt_deg: float = 12.0
    expected_height_m: float = 0.0  # floor height in the input frame (base_link: 0)
    height_tolerance_m: float = 0.10
    min_inliers: int = 200
    min_support_area_m2: float = 0.25
    area_cell_m: float = 0.05
    max_candidates: int = 4
    ransac_iterations: int = 400
    seed: int = 0


@dataclass
class PlaneCandidate:
    normal: np.ndarray  # unit, oriented with n_z >= 0
    offset: float
    inlier_count: int
    residual_m: float
    support_area_m2: float
    tilt_deg: float
    height_at_origin_m: float | None
    accepted: bool
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "normal": [round(float(x), 4) for x in self.normal],
            "offset": round(self.offset, 4),
            "inliers": self.inlier_count,
            "area_m2": round(self.support_area_m2, 3),
            "tilt_deg": round(self.tilt_deg, 2),
            "height_m": None
            if self.height_at_origin_m is None
            else round(self.height_at_origin_m, 3),
            "accepted": self.accepted,
            "reasons": self.reasons,
        }


@dataclass
class GroundResult:
    plane: PlaneCandidate | None
    inlier_mask: np.ndarray  # over input points; all False if no ground
    candidates: list[PlaneCandidate]


def _fit_plane_svd(pts: np.ndarray) -> tuple[np.ndarray, float]:
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    return n, float(-n @ c)


def _ransac_plane(
    pts: np.ndarray, thr: float, iters: int, rng: np.random.Generator
) -> tuple[np.ndarray, float] | None:
    n_pts = len(pts)
    if n_pts < 3:
        return None
    idx = rng.integers(0, n_pts, size=(iters, 3))
    a, b, c = pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]
    normals = np.cross(b - a, c - a)
    norm = np.linalg.norm(normals, axis=1)
    ok = norm > 1e-9
    if not ok.any():
        return None
    normals = normals[ok] / norm[ok, None]
    offsets = -np.einsum("ij,ij->i", normals, a[ok])
    best_count, best = -1, None
    chunk = 64
    for s in range(0, len(normals), chunk):
        dist = np.abs(pts @ normals[s : s + chunk].T + offsets[s : s + chunk])
        counts = (dist < thr).sum(axis=0)
        j = int(np.argmax(counts))
        if counts[j] > best_count:
            best_count, best = int(counts[j]), (normals[s + j], float(offsets[s + j]))
    return best


def observed_area(pts_xy: np.ndarray, cell: float) -> float:
    if len(pts_xy) == 0:
        return 0.0
    cells = np.unique(np.floor(pts_xy / cell).astype(np.int64), axis=0)
    return float(len(cells) * cell * cell)


def detect_ground(points: np.ndarray, p: GroundParams) -> GroundResult:
    rng = np.random.default_rng(p.seed)
    remaining = np.arange(len(points))
    candidates: list[PlaneCandidate] = []
    masks: list[np.ndarray] = []
    up = np.array([0.0, 0.0, 1.0])
    for _ in range(p.max_candidates):
        if len(remaining) < max(3, p.min_inliers // 4):
            break
        sub = points[remaining]
        hyp = _ransac_plane(sub, p.distance_threshold_m, p.ransac_iterations, rng)
        if hyp is None:
            break
        n, d = hyp
        inl = np.abs(sub @ n + d) < p.distance_threshold_m
        if inl.sum() < 3:
            break
        n, d = _fit_plane_svd(sub[inl])
        dist = sub @ n + d
        inl = np.abs(dist) < p.distance_threshold_m
        resid = float(np.sqrt(np.mean(dist[inl] ** 2))) if inl.any() else float("inf")
        tilt = float(np.degrees(np.arccos(np.clip(abs(n @ up), -1, 1))))
        # Area of the inliers projected on the plane (in-plane coordinates).
        e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        in_pts = sub[inl]
        area = observed_area(np.stack([in_pts @ e1, in_pts @ e2], axis=1), p.area_cell_m)
        # Height is only meaningful for near-horizontal planes (|n_z| > cos 60 deg).
        height = -d / n[2] if abs(n[2]) > 0.5 else None
        reasons = []
        if tilt > p.max_tilt_deg:
            reasons.append(f"tilt {tilt:.1f} deg > {p.max_tilt_deg}")
        if height is None or abs(height - p.expected_height_m) > p.height_tolerance_m:
            h = "n/a" if height is None else f"{height:.2f} m"
            reasons.append(
                f"height {h} not within {p.height_tolerance_m} m of {p.expected_height_m} m"
            )
        if inl.sum() < p.min_inliers:
            reasons.append(f"inliers {int(inl.sum())} < {p.min_inliers}")
        if area < p.min_support_area_m2:
            reasons.append(f"observed area {area:.2f} m2 < {p.min_support_area_m2}")
        candidates.append(
            PlaneCandidate(n, d, int(inl.sum()), resid, area, tilt, height, not reasons, reasons)
        )
        full = np.zeros(len(points), dtype=bool)
        full[remaining[inl]] = True
        masks.append(full)
        remaining = remaining[~inl]

    accepted = [i for i, c in enumerate(candidates) if c.accepted]
    if not accepted:
        return GroundResult(None, np.zeros(len(points), dtype=bool), candidates)
    best = max(accepted, key=lambda i: candidates[i].inlier_count)
    plane = candidates[best]
    # Final labelling against all points (a plane extracted later may own points near the floor).
    mask = np.abs(points @ plane.normal + plane.offset) < p.distance_threshold_m
    return GroundResult(plane, mask, candidates)
