"""Rigid transforms. Convention: ``T_A_B`` maps points in frame B to frame A (see docs/coordinates.md)."""

from __future__ import annotations

import numpy as np

# Columns are the camera_optical axes expressed in base_link (optical z forward -> base x, ...).
R_BASE_OPTICAL = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert(T_A_B: np.ndarray) -> np.ndarray:
    """Return ``T_B_A``. Uses the rigid-body closed form, not a general matrix inverse."""
    R = T_A_B[:3, :3]
    t = T_A_B[:3, 3]
    return make_transform(R.T, -R.T @ t)


def compose(*transforms: np.ndarray) -> np.ndarray:
    """``compose(T_A_B, T_B_C) == T_A_C``."""
    out = np.eye(4)
    for T in transforms:
        out = out @ T
    return out


def transform_points(T_A_B: np.ndarray, points_B: np.ndarray) -> np.ndarray:
    """Apply ``T_A_B`` to an (N, 3) array of points expressed in B."""
    pts = np.asarray(points_B, dtype=np.float64)
    return pts @ T_A_B[:3, :3].T + T_A_B[:3, 3]


def rotate_vectors(T_A_B: np.ndarray, vectors_B: np.ndarray) -> np.ndarray:
    """Rotate direction vectors (no translation)."""
    return np.asarray(vectors_B, dtype=np.float64) @ T_A_B[:3, :3].T


def is_rigid(T: np.ndarray, atol: float = 1e-6) -> bool:
    T = np.asarray(T)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    R = T[:3, :3]
    return (
        np.allclose(R.T @ R, np.eye(3), atol=atol)
        and abs(np.linalg.det(R) - 1.0) < atol
        and np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=atol)
    )


def assert_rigid(T: np.ndarray, name: str = "T", atol: float = 1e-6) -> None:
    if not is_rigid(T, atol):
        raise ValueError(
            f"{name} is not a proper rigid transform (orthonormal R, det=+1, last row 0001)"
        )


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def camera_mount(
    height_m: float, pitch_down_rad: float, forward_m: float = 0.0, lateral_m: float = 0.0
) -> np.ndarray:
    """``T_base_camera`` for a forward-looking camera pitched down by ``pitch_down_rad``."""
    R = rot_y(pitch_down_rad) @ R_BASE_OPTICAL
    return make_transform(R, [forward_m, lateral_m, height_m])


def planar_pose(x: float, y: float, yaw: float) -> np.ndarray:
    """``T_map_base`` for a robot on the floor (z = 0) at (x, y) with heading ``yaw``."""
    return make_transform(rot_z(yaw), [x, y, 0.0])
