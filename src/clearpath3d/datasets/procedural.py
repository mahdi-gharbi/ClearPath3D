"""Procedural room layouts for held-out evaluation (geometry varies with the seed).

Each seed gives: room size, partition position, door position/width, 1-4 boxes, an optional solid
cabinet (occluder) and a scripted scan along the room's centre line. Objects are kept out of the robot's
scripted corridor so the scripted motion itself never passes through an obstacle. The *pipeline* never sees
these parameters; they only exist in ``ground_truth/`` through the rendered scene.
"""

from __future__ import annotations

import math

import numpy as np

from clearpath3d.datasets.synthetic import (
    WALL_H,
    WALL_T,
    SceneObject,
    SceneSpec,
    TrajectoryPoint,
    _box,
)

ROBOT_CORRIDOR_HALF_WIDTH = 0.6  # objects keep this clearance from the scripted path line


def _walls(rx: float, ry: float, px: float, door_y0: float, door_w: float) -> list[SceneObject]:
    t, h = WALL_T, WALL_H
    beige, grey = (0.85, 0.80, 0.70), (0.72, 0.72, 0.75)
    return [
        SceneObject("floor", "floor", (-t, -t, -0.05), (rx + t, ry + t, 0.0), (0.45, 0.42, 0.38)),
        SceneObject("wall_south", "wall", (-t, -t, 0), (rx + t, 0.0, h), beige),
        SceneObject("wall_north", "wall", (-t, ry, 0), (rx + t, ry + t, h), beige),
        SceneObject("wall_west", "wall", (-t, 0.0, 0), (0.0, ry, h), beige),
        SceneObject("wall_east", "wall", (rx, 0.0, 0), (rx + t, ry, h), beige),
        SceneObject("partition_a", "wall", (px, 0.0, 0), (px + t, door_y0, h), grey),
        SceneObject("partition_b", "wall", (px, door_y0 + door_w, 0), (px + t, ry, h), grey),
    ]


def _overlaps(a_lo, a_hi, b_lo, b_hi, gap: float) -> bool:
    return all(a_lo[k] - gap < b_hi[k] and b_lo[k] - gap < a_hi[k] for k in range(2))


def build_procedural(seed: int, n_frames_hint: int = 28) -> SceneSpec:
    rng = np.random.default_rng(10_000 + seed)
    rx = float(rng.uniform(5.5, 7.0))
    ry = float(rng.uniform(3.6, 5.0))
    px = float(rng.uniform(0.6, 0.7) * rx)
    door_w = float(rng.choice([rng.uniform(0.75, 1.1), rng.uniform(0.3, 0.45)], p=[0.75, 0.25]))
    yc = ry / 2 + float(rng.uniform(-0.3, 0.3))  # scripted path line
    door_y0 = float(np.clip(yc - door_w / 2 + rng.uniform(-0.15, 0.15), 0.3, ry - door_w - 0.3))
    objs = _walls(rx, ry, px, door_y0, door_w)

    placed: list[tuple] = []

    def free_spot(sx: float, sy: float, tries: int = 200):
        for _ in range(tries):
            cx = float(rng.uniform(0.5 + sx / 2, px - 0.3 - sx / 2))
            cy = float(rng.uniform(0.3 + sy / 2, ry - 0.3 - sy / 2))
            lo, hi = (cx - sx / 2, cy - sy / 2), (cx + sx / 2, cy + sy / 2)
            if lo[1] < yc + ROBOT_CORRIDOR_HALF_WIDTH and hi[1] > yc - ROBOT_CORRIDOR_HALF_WIDTH:
                continue
            if any(_overlaps(lo, hi, a, b, 0.3) for a, b in placed):
                continue
            placed.append((lo, hi))
            return cx, cy
        return None

    if rng.random() < 0.6:
        sx, sy = float(rng.uniform(0.8, 1.3)), float(rng.uniform(0.4, 0.6))
        spot = free_spot(sx, sy)
        if spot:
            cx, cy = spot
            objs.append(
                SceneObject(
                    "cabinet",
                    "furniture",
                    (cx - sx / 2, cy - sy / 2, 0.0),
                    (cx + sx / 2, cy + sy / 2, float(rng.uniform(0.6, 0.85))),
                    (0.45, 0.30, 0.20),
                )
            )
    for k in range(int(rng.integers(1, 5))):
        sx, sy, sz = (
            float(rng.uniform(0.25, 0.5)),
            float(rng.uniform(0.2, 0.4)),
            float(rng.uniform(0.15, 0.4)),
        )
        spot = free_spot(sx, sy)
        if spot:
            objs.append(_box(f"box_{k}", spot[0], spot[1], sx, sy, sz))

    # Scripted scan: sweep at the start, move along the centre line, sweep, end facing the door.
    x0, x1 = 0.6, px - 0.7
    way = [
        (x0, yc, 40),
        (x0, yc, -40),
        (x0 + 0.6, yc, -10),
        (x0 + 0.6, yc, 80),
        (x0 + 0.6, yc, -80),
        (x1, yc, 0),
        (x1, yc, -100),
        (x1, yc, 100),
        (x1, yc, 0),
    ]
    steps = [6, 4, 3, 5, 6, 3, 4, 3]
    pts = [way[0]]
    for (a, b, c), (d, e, f), n in zip(way[:-1], way[1:], steps, strict=True):
        for i in range(1, n + 1):
            u = i / n
            pts.append((a + (d - a) * u, b + (e - b) * u, c + (f - c) * u))
    traj = [TrajectoryPoint(i * 0.2, x, y, math.radians(yaw)) for i, (x, y, yaw) in enumerate(pts)]
    desc = (
        f"procedural seed {seed}: room {rx:.2f}x{ry:.2f} m, door {door_w:.2f} m at y={door_y0:.2f}, "
        f"{len(objs) - 7} objects"
    )
    spec = SceneSpec(f"proc_{seed:03d}", desc, objs, traj, seed=seed)
    spec.layout = {
        "rx": rx,
        "ry": ry,
        "partition_x": px,
        "door_y0": door_y0,
        "door_w": door_w,
        "path_y": yc,
    }
    return spec
