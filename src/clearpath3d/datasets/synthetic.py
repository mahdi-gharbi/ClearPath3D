"""Reproducible synthetic RGB-D scenes rendered with Open3D ``RaycastingScene``.

Occlusions are real (ray casting against triangle meshes). Depth noise and dropout are simple
parametric perturbations; they are *not* a faithful model of any commercial sensor.

Output layout (strict separation):

    <out>/manifest.json
    <out>/sensor_export/{calibration.json, frames.json, frames/NNNNNN.npz}   # pipeline input
    <out>/ground_truth/{objects.json, frames/NNNNNN.npz, scene_mesh.ply}     # evaluator only
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d

from clearpath3d.contracts import Intrinsics
from clearpath3d.geometry.frames import camera_mount, compose, planar_pose
from clearpath3d.geometry.projection import ray_directions_optical

GENERATOR_VERSION = "synthetic-0.1"
INVALID_ID = o3d.t.geometry.RaycastingScene.INVALID_ID


@dataclass
class SceneObject:
    name: str
    category: str  # floor | wall | table | box
    min_m: tuple[float, float, float]
    max_m: tuple[float, float, float]
    color: tuple[float, float, float]
    # Piecewise-constant motion: list of (t_start_s, (dx, dy, dz)) applied from t_start.
    motion: list[tuple[float, tuple[float, float, float]]] = field(default_factory=list)

    def offset_at(self, t: float) -> np.ndarray:
        off = np.zeros(3)
        for t0, d in self.motion:
            if t >= t0:
                off = np.asarray(d, dtype=np.float64)
        return off

    @property
    def is_static(self) -> bool:
        return not self.motion


@dataclass
class TrajectoryPoint:
    t: float
    x: float
    y: float
    yaw: float


@dataclass
class Perturbations:
    depth_noise_sigma0_m: float = 0.002
    depth_noise_quadratic: float = 0.0015  # sigma(z) = sigma0 + k * z^2
    dropout_prob: float = 0.01
    hole_count: int = 0  # rectangular dropout holes per frame
    hole_size_px: int = 20
    max_range_m: float = 8.0


@dataclass
class CameraSpec:
    width: int = 320
    height: int = 240
    hfov_deg: float = 62.0
    mount_height_m: float = 0.8
    mount_pitch_down_deg: float = 15.0
    mount_forward_m: float = 0.1

    def intrinsics(self) -> Intrinsics:
        f = (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)
        return Intrinsics(
            fx=f,
            fy=f,
            cx=(self.width - 1) / 2,
            cy=(self.height - 1) / 2,
            width=self.width,
            height=self.height,
        )

    def T_base_camera(self) -> np.ndarray:
        return camera_mount(
            self.mount_height_m, math.radians(self.mount_pitch_down_deg), self.mount_forward_m
        )


@dataclass
class SceneSpec:
    name: str
    description: str
    objects: list[SceneObject]
    trajectory: list[TrajectoryPoint]
    camera: CameraSpec = field(default_factory=CameraSpec)
    perturbations: Perturbations = field(default_factory=Perturbations)
    seed: int = 0
    poses_supplied: bool = True


# ------------------------------------------------------------------------- scene library

ROOM_X, ROOM_Y, WALL_H, WALL_T = 6.0, 4.0, 2.5, 0.1
PARTITION_X = 3.9  # partition wall occupies x in [3.9, 4.0]


def _room(door_y0: float, door_width: float) -> list[SceneObject]:
    grey, beige = (0.72, 0.72, 0.75), (0.85, 0.80, 0.70)
    t, h = WALL_T, WALL_H
    objs = [
        SceneObject(
            "floor", "floor", (-t, -t, -0.05), (ROOM_X + t, ROOM_Y + t, 0.0), (0.45, 0.42, 0.38)
        ),
        SceneObject("wall_south", "wall", (-t, -t, 0), (ROOM_X + t, 0.0, h), beige),
        SceneObject("wall_north", "wall", (-t, ROOM_Y, 0), (ROOM_X + t, ROOM_Y + t, h), beige),
        SceneObject("wall_west", "wall", (-t, 0.0, 0), (0.0, ROOM_Y, h), beige),
        SceneObject("wall_east", "wall", (ROOM_X, 0.0, 0), (ROOM_X + t, ROOM_Y, h), beige),
        SceneObject(
            "partition_a", "wall", (PARTITION_X, 0.0, 0), (PARTITION_X + t, door_y0, h), grey
        ),
        SceneObject(
            "partition_b",
            "wall",
            (PARTITION_X, door_y0 + door_width, 0),
            (PARTITION_X + t, ROOM_Y, h),
            grey,
        ),
    ]
    return objs


def _table(cx: float, cy: float, name: str = "table") -> list[SceneObject]:
    wood = (0.55, 0.35, 0.20)
    lx, ly, top_z, top_t, leg = 1.2, 0.7, 0.75, 0.04, 0.05
    objs = [
        SceneObject(
            f"{name}_top",
            "table",
            (cx - lx / 2, cy - ly / 2, top_z - top_t),
            (cx + lx / 2, cy + ly / 2, top_z),
            wood,
        )
    ]
    for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        x = cx + sx * (lx / 2 - leg)
        y = cy + sy * (ly / 2 - leg)
        objs.append(
            SceneObject(
                f"{name}_leg{i}",
                "table",
                (x - leg / 2, y - leg / 2, 0.0),
                (x + leg / 2, y + leg / 2, top_z - top_t),
                wood,
            )
        )
    return objs


def _box(name: str, cx: float, cy: float, sx=0.4, sy=0.3, sz=0.3, motion=None) -> SceneObject:
    return SceneObject(
        name,
        "box",
        (cx - sx / 2, cy - sy / 2, 0.0),
        (cx + sx / 2, cy + sy / 2, sz),
        (0.76, 0.58, 0.36),
        motion=motion or [],
    )


def _sweep(
    x: float, y: float, yaw0: float, yaw1: float, n: int, dt: float = 0.2
) -> list[TrajectoryPoint]:
    return [
        TrajectoryPoint(i * dt, x, y, yaw0 + (yaw1 - yaw0) * i / max(n - 1, 1)) for i in range(n)
    ]


def _path(
    waypoints: list[tuple[float, float, float]], steps: list[int], dt: float = 0.2
) -> list[TrajectoryPoint]:
    """Piecewise-linear trajectory through (x, y, yaw_deg) waypoints; ``steps[i]`` frames per leg."""
    pts = [waypoints[0]]
    for (x0, y0, a0), (x1, y1, a1), n in zip(waypoints[:-1], waypoints[1:], steps, strict=True):
        for k in range(1, n + 1):
            f = k / n
            pts.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, a0 + (a1 - a0) * f))
    return [TrajectoryPoint(i * dt, x, y, math.radians(a)) for i, (x, y, a) in enumerate(pts)]


def _scan_path(density: int = 1) -> list[TrajectoryPoint]:
    """Scripted scan; ``density`` multiplies the number of frames (same path, smaller inter-frame motion)."""
    steps = [6, 4, 3, 5, 6, 3, 3]
    return _path(
        [
            (0.6, 2.3, 40),
            (0.6, 2.3, -40),
            (1.2, 2.35, -10),
            (1.2, 2.35, 80),
            (1.2, 2.35, -80),
            (3.2, 2.35, 0),
            (3.2, 2.35, -100),
            (3.2, 2.35, 100),
        ],
        [n * density for n in steps],
        dt=0.2 / density,
    )


def build_scene(name: str, **overrides) -> SceneSpec:
    """Return a parametrised scene. ``overrides`` may set camera/perturbation fields or seed.

    ``proc_NNN`` names give procedural layouts (geometry from NNN, see ``datasets/procedural.py``).
    """
    if name.startswith("proc_"):
        from clearpath3d.datasets.procedural import build_procedural

        spec = build_procedural(int(name.split("_")[1]))
        return _apply_overrides(spec, overrides)
    if name == "room_clear":
        spec = SceneSpec(
            name,
            "Room with a 0.9 m door to a clear corridor, a table and two boxes.",
            _room(2.0, 0.9) + _table(2.4, 3.0) + [_box("box_a", 2.6, 1.2), _box("box_b", 1.8, 1.7)],
            _sweep(0.5, 2.2, math.radians(-10), math.radians(10), 5),
        )
    elif name == "dominant_wall":
        spec = SceneSpec(
            name,
            "Camera 1.6 m from a wall: the wall plane is far larger than the visible floor.",
            _room(2.0, 0.9) + [_box("box_a", 3.3, 0.5)],
            _sweep(2.3, 0.95, 0.0, 0.0, 1),
        )
    elif name == "no_ground":
        spec = SceneSpec(
            name,
            "Camera pitched up towards a wall: no floor in view.",
            _room(2.0, 0.9),
            _sweep(2.4, 1.0, 0.0, 0.0, 1),
            camera=CameraSpec(mount_pitch_down_deg=-28.0),
        )
    elif name == "narrow_door":
        spec = SceneSpec(
            name,
            "Same room with a 0.35 m door, narrower than the reference robot (0.5 m).",
            _room(2.2, 0.35) + _table(2.0, 3.2) + [_box("box_a", 2.5, 0.8)],
            _path(
                [
                    (0.6, 2.375, 30),
                    (0.6, 2.375, -30),
                    (1.6, 2.375, 0),
                    (2.7, 2.375, 0),
                    (2.7, 2.375, 35),
                    (2.7, 2.375, -35),
                ],
                [4, 3, 4, 2, 3],
            ),
        )
    elif name == "hidden_box":
        spec = SceneSpec(
            name,
            "room_scan trajectory; a small box hides behind a solid cabinet and is never observed.",
            _room(2.0, 0.9)
            + [
                SceneObject(
                    "cabinet", "furniture", (1.8, 2.95, 0.0), (3.0, 3.45, 0.75), (0.45, 0.30, 0.20)
                ),
                _box("box_a", 2.6, 1.2),
                _box("box_b", 1.8, 1.7),
                _box("hidden", 2.4, 3.7, 0.3, 0.3, 0.25),
            ],
            _scan_path(),
        )
    elif name == "moving_box":
        spec = SceneSpec(
            name,
            "box_b is moved at t=1.3 s from (1.8, 1.7) to (2.1, 0.75); its old place is "
            "re-observed only from frame 12 on.",
            _room(2.0, 0.9)
            + _table(2.4, 3.0)
            + [_box("box_a", 2.6, 1.2), _box("box_b", 1.8, 1.7, motion=[(1.3, (0.3, -0.95, 0.0))])],
            _path(
                [
                    (0.6, 2.3, -40),
                    (0.6, 2.3, 40),
                    (1.2, 2.35, 60),
                    (1.2, 2.35, -50),
                    (1.25, 2.35, -60),
                ],
                [6, 4, 5, 6],
            ),
        )
    elif name == "sensor_dropout":
        spec = SceneSpec(
            name,
            "room_scan with heavy sensor dropout: 15 % random pixels, 10 rectangular holes per frame, "
            "3x depth noise. Holes must never become free space.",
            _room(2.0, 0.9) + _table(2.4, 3.0) + [_box("box_a", 2.6, 1.2), _box("box_b", 1.8, 1.7)],
            _scan_path(),
            perturbations=Perturbations(
                depth_noise_sigma0_m=0.006,
                depth_noise_quadratic=0.0045,
                dropout_prob=0.15,
                hole_count=10,
                hole_size_px=40,
            ),
        )
    elif name == "low_overlap":
        spec = SceneSpec(
            name,
            "Pose jumps too large for local registration: 70 deg turns and a 0.8 m jump between frames.",
            _room(2.0, 0.9) + _table(2.4, 3.0) + [_box("box_a", 2.6, 1.2), _box("box_b", 1.8, 1.7)],
            [
                TrajectoryPoint(0.0, 0.6, 2.3, 0.0),
                TrajectoryPoint(0.2, 0.6, 2.3, math.radians(10)),
                TrajectoryPoint(0.4, 0.6, 2.3, math.radians(80)),
                TrajectoryPoint(0.6, 0.6, 2.3, math.radians(-60)),
                TrajectoryPoint(0.8, 1.4, 2.3, math.radians(-60)),
                TrajectoryPoint(1.0, 1.45, 2.3, math.radians(-55)),
            ],
        )
    elif name == "room_scan_dense":
        spec = SceneSpec(
            name,
            "room_scan path with 4x more frames (3-8 deg per frame): for frame-to-frame odometry.",
            _room(2.0, 0.9) + _table(2.4, 3.0) + [_box("box_a", 2.6, 1.2), _box("box_b", 1.8, 1.7)],
            _scan_path(density=4),
        )
    elif name == "room_scan":
        spec = SceneSpec(
            name,
            "room_clear layout explored by a 30-frame scan (rotations + forward motion).",
            _room(2.0, 0.9) + _table(2.4, 3.0) + [_box("box_a", 2.6, 1.2), _box("box_b", 1.8, 1.7)],
            _scan_path(),
        )
    else:
        raise KeyError(f"unknown scene {name!r}; available: {SCENES}")
    return _apply_overrides(spec, overrides)


def _apply_overrides(spec: SceneSpec, overrides: dict) -> SceneSpec:
    for k, v in overrides.items():
        if k == "seed":
            spec.seed = int(v)
        elif hasattr(spec.camera, k):
            setattr(spec.camera, k, v)
        elif hasattr(spec.perturbations, k):
            setattr(spec.perturbations, k, v)
        else:
            raise KeyError(f"unknown override {k!r}")
    return spec


SCENES = (
    "room_clear",
    "dominant_wall",
    "no_ground",
    "narrow_door",
    "room_scan",
    "hidden_box",
    "moving_box",
    "sensor_dropout",
    "low_overlap",
    "room_scan_dense",
)


# ------------------------------------------------------------------------- rendering


def _object_mesh(obj: SceneObject, t: float) -> o3d.geometry.TriangleMesh:
    lo = np.asarray(obj.min_m) + obj.offset_at(t)
    size = np.asarray(obj.max_m) - np.asarray(obj.min_m)
    m = o3d.geometry.TriangleMesh.create_box(*size)
    m.translate(lo)
    return m


def mesh_hash(spec: SceneSpec, t: float = 0.0) -> str:
    h = hashlib.sha256()
    for obj in spec.objects:
        m = _object_mesh(obj, t)
        h.update(obj.name.encode())
        h.update(np.round(np.asarray(m.vertices), 6).tobytes())
        h.update(np.asarray(m.triangles).astype(np.int64).tobytes())
    return h.hexdigest()


@dataclass
class RenderedFrame:
    depth_clean: np.ndarray  # float32 z-depth, 0 where no hit within range
    depth_noisy: np.ndarray  # float32, 0 = no measurement
    color: np.ndarray  # uint8 [H, W, 3]
    instance: np.ndarray  # int32 object index, -1 = no hit
    T_map_camera: np.ndarray
    T_map_base: np.ndarray


class Renderer:
    def __init__(self, spec: SceneSpec):
        self.spec = spec
        self.intr = spec.camera.intrinsics()
        self.T_base_camera = spec.camera.T_base_camera()
        self._dirs = ray_directions_optical(self.intr).reshape(-1, 3)
        self._cache_key: tuple | None = None
        self._scene = None

    def _scene_at(self, t: float):
        key = tuple(tuple(o.offset_at(t)) for o in self.spec.objects)
        if key != self._cache_key:
            scene = o3d.t.geometry.RaycastingScene()
            self._geom_to_obj = {}
            for i, obj in enumerate(self.spec.objects):
                gid = scene.add_triangles(
                    o3d.t.geometry.TriangleMesh.from_legacy(_object_mesh(obj, t))
                )
                self._geom_to_obj[int(gid)] = i
            self._scene, self._cache_key = scene, key
        return self._scene

    def render(self, tp: TrajectoryPoint, rng: np.random.Generator) -> RenderedFrame:
        intr, pert = self.intr, self.spec.perturbations
        T_map_base = planar_pose(tp.x, tp.y, tp.yaw)
        T_map_camera = compose(T_map_base, self.T_base_camera)
        R, origin = T_map_camera[:3, :3], T_map_camera[:3, 3]
        dirs_map = self._dirs @ R.T  # unit z component in the optical frame -> t_hit == z-depth
        rays = np.concatenate([np.broadcast_to(origin, dirs_map.shape), dirs_map], axis=1)
        ans = self._scene_at(tp.t).cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
        t_hit = ans["t_hit"].numpy().astype(np.float64)
        gids = ans["geometry_ids"].numpy().astype(np.int64)
        normals = ans["primitive_normals"].numpy().astype(np.float64)

        hit = np.isfinite(t_hit) & (gids != INVALID_ID) & (t_hit <= pert.max_range_m)
        lut = np.full(max(self._geom_to_obj) + 1, -1, dtype=np.int32)
        for g, i in self._geom_to_obj.items():
            lut[g] = i
        instance = np.where(hit, lut[np.where(hit, gids, 0)], -1).astype(np.int32)
        depth_clean = np.where(hit, t_hit, 0.0)

        # Colour: material albedo with Lambertian shading w.r.t. the viewing ray.
        albedo = np.array([o.color for o in self.spec.objects] + [(0.0, 0.0, 0.0)])
        unit = dirs_map / np.linalg.norm(dirs_map, axis=1, keepdims=True)
        shade = 0.35 + 0.65 * np.abs(np.einsum("ij,ij->i", np.nan_to_num(normals), unit))
        color = albedo[instance] * np.where(hit, shade, 0.0)[:, None]

        # Perturbations: depth noise grows with z^2, random pixel dropout, rectangular holes.
        sigma = pert.depth_noise_sigma0_m + pert.depth_noise_quadratic * depth_clean**2
        noisy = depth_clean + rng.normal(0.0, 1.0, size=depth_clean.shape) * sigma
        drop = rng.random(depth_clean.shape) < pert.dropout_prob
        noisy = np.where(hit & ~drop & (noisy > 0), noisy, 0.0)
        H, W = intr.height, intr.width
        noisy = noisy.reshape(H, W)
        for _ in range(pert.hole_count):
            v0 = int(rng.integers(0, max(1, H - pert.hole_size_px)))
            u0 = int(rng.integers(0, max(1, W - pert.hole_size_px)))
            noisy[v0 : v0 + pert.hole_size_px, u0 : u0 + pert.hole_size_px] = 0.0

        return RenderedFrame(
            depth_clean.reshape(H, W).astype(np.float32),
            noisy.astype(np.float32),
            np.clip(color * 255.0 + 0.5, 0, 255).astype(np.uint8).reshape(H, W, 3),
            instance.reshape(H, W),
            T_map_camera,
            T_map_base,
        )


# ------------------------------------------------------------------------- export


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def generate(spec: SceneSpec, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    sensor, gt = out_dir / "sensor_export", out_dir / "ground_truth"
    (sensor / "frames").mkdir(parents=True, exist_ok=True)
    (gt / "frames").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)
    r = Renderer(spec)
    intr = r.intr
    calibration_id = f"{spec.name}-cam-v1"

    _write_json(
        sensor / "calibration.json",
        {
            "calibration_id": calibration_id,
            "intrinsics": intr.model_dump(),
            "T_base_camera": r.T_base_camera.tolist(),
            "registered_color_depth": True,
        },
    )

    frames_meta = []
    for i, tp in enumerate(spec.trajectory):
        fr = r.render(tp, rng)
        fid = f"{i:06d}"
        np.savez_compressed(
            sensor / "frames" / f"{fid}.npz", color_rgb=fr.color, depth=fr.depth_noisy
        )
        np.savez_compressed(
            gt / "frames" / f"{fid}.npz",
            depth_clean=fr.depth_clean,
            instance=fr.instance,
            T_map_camera=fr.T_map_camera,
            T_map_base=fr.T_map_base,
        )
        meta = {"frame_id": fid, "timestamp_s": tp.t, "file": f"frames/{fid}.npz"}
        if spec.poses_supplied:
            meta["T_map_camera"] = fr.T_map_camera.tolist()
            meta["pose_source"] = "supplied"
        else:
            meta["pose_source"] = "unavailable"
        frames_meta.append(meta)
    _write_json(sensor / "frames.json", {"sequence_id": spec.name, "frames": frames_meta})

    _write_json(
        gt / "objects.json",
        {
            "floor_plane_map": {"normal": [0.0, 0.0, 1.0], "offset": 0.0},
            "objects": [
                {"index": i, **asdict(o), "static": o.is_static} for i, o in enumerate(spec.objects)
            ],
        },
    )
    full = o3d.geometry.TriangleMesh()
    for o in spec.objects:
        full += _object_mesh(o, 0.0)
    o3d.io.write_triangle_mesh(str(gt / "scene_mesh_t0.ply"), full)

    manifest = {
        "dataset": "clearpath3d-synthetic",
        "generator_version": GENERATOR_VERSION,
        "scene": spec.name,
        "description": spec.description,
        "seed": spec.seed,
        "mesh_hash_t0": mesh_hash(spec, 0.0),
        "open3d_version": o3d.__version__,
        "license": "Generated data, same license as the repository (MIT).",
        "depth_convention": "z",  # z-depth along the optical axis
        "depth_unit": "m",
        "depth_scale": 1.0,  # stored_value * depth_scale = metres
        "invalid_depth_value": 0.0,
        "pose_convention": "T_map_camera maps camera_optical points to map; column vectors",
        "poses_supplied": spec.poses_supplied,
        "camera": asdict(spec.camera),
        "perturbations": asdict(spec.perturbations),
        "trajectory": [asdict(t) for t in spec.trajectory],
        "transformations_applied": [],
        "usage_limits": (
            "Synthetic benchmark. Noise/dropout are parametric, not a sensor model. "
            "ground_truth/ is for evaluation only and must never be read by the pipeline."
        ),
        "num_frames": len(spec.trajectory),
    }
    _write_json(out_dir / "manifest.json", manifest)
    return out_dir
