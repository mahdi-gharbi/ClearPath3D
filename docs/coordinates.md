# Coordinate conventions

This document is the single source of truth for frames, units and transforms in ClearPath3D.
Every dataset adapter converts its native conventions **once** into these contracts; the rest of
the core never re-interprets them.

## Transform notation

`T_A_B` is a 4x4 homogeneous matrix (float64) that maps a point expressed in frame **B** into
frame **A**. Points are homogeneous column vectors.

```text
p_A   = T_A_B @ p_B
T_A_C = T_A_B @ T_B_C          # inner frame names must match
T_B_A = inverse(T_A_B)         # = [R^T, -R^T t; 0 0 0 1]
```

Read the name right-to-left: `T_map_camera` takes camera-frame points to the map.
The translation column of `T_A_B` is the origin of B expressed in A (e.g. the translation of
`T_map_camera` is the camera position in the map).

Implementation: `clearpath3d.geometry.frames` (`make_transform`, `invert`, `compose`,
`transform_points`, `assert_rigid`). `assert_rigid` checks orthonormality and `det(R)=+1`.
A valid rotation is not necessarily the *right* convention, so the test-suite also renders an
asymmetric synthetic scene and checks that a box on the robot's left appears on the left of the
image and back-projects to the left of the robot (`tests/unit/test_asymmetric_scene.py`).

## Frames

| Frame | Axes | Notes |
|---|---|---|
| `map` | x, y horizontal, **z up** | Local global frame. In synthetic scenes the floor is the plane z = 0. |
| `base_link` | **x forward, y left, z up** (REP-103) | Robot base, origin on the floor under the robot centre. |
| `camera_optical` | **x right (image u), y down (image v), z forward** | Standard pinhole optical frame. |

The fixed rotation between `base_link` and an optical frame looking straight ahead is

```text
R_base_optical = [[ 0,  0,  1],     # optical z (forward) -> base x
                  [-1,  0,  0],     # optical x (right)   -> base -y
                  [ 0, -1,  0]]     # optical y (down)    -> base -z
```

(the columns are the optical axes expressed in `base_link`). A camera mounted at height `h`,
pitched down by angle `θ` (positive = looking down) is

```text
T_base_camera = Trans(x_m, y_m, h) · RotY(θ) · R_base_optical
```

`RotY(θ)` with positive θ rotates base x towards -z, i.e. tilts the view downwards.
`T_base_camera` is an **explicit calibration** (`calibration_id` in each frame); it is never
inferred from the viewer orientation. Viewer rotations (Three.js y-up, RViz) are display-only
and never applied to data.

## Units and time

- Distances in metres, angles in radians, time in seconds (float64).
- Source timestamps are kept as provided by the dataset (`timestamp_s`); nothing is re-based.

## Pinhole model and depth

Intrinsics: `fx, fy, cx, cy, width, height` (pixels). Pixel `(u, v)` = (column, row); integer
pixel coordinates refer to the pixel centre, i.e. no +0.5 offset is applied anywhere.

```text
z = depth_m[v, u]                   # z-depth along the optical axis, NOT the ray length
x = (u - cx) * z / fx
y = (v - cy) * z / fy
u = fx * x / z + cx ,   v = fy * y / z + cy
```

If a source produces Euclidean distance along the ray (`depth_convention: ray`), the adapter
converts it once with `projection.ray_distance_to_z_depth`:
`z = r / sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2)`.

The synthetic simulator casts rays with direction `d = [(u-cx)/fx, (v-cy)/fy, 1]` (unnormalised,
unit z component), so Open3D's `t_hit` is directly the z-depth. This is verified by a test that
renders a fronto-parallel wall and checks the depth image is constant.

Colour and depth are assumed registered (same intrinsics, same optical frame). An adapter whose
sensor is not registered must register it before producing an `RGBDFrame`.

## Invalid depth

`depth_valid[v,u]` is False for 0, NaN, ±inf and values outside `[min_depth_m, max_depth_m]`.
An invalid pixel is **no measurement**: it never produces a point and never produces free space
("missing depth is not free space"). Frames whose image size does not match the intrinsics are
rejected. Depth unit errors (e.g. millimetres stored as metres) are caught by the manifest's
`depth_scale` plus a plausibility check on the median valid depth.

## Poses

`RGBDFrame.T_map_camera` is `None` unless `pose_source` is `supplied` (from the dataset /
simulator, allowed by the protocol) or `estimated` (by a registration module). Reports always
carry `pose_source`. Single-frame analysis that has no pose works in `base_link` using the
calibration only, and says so in its output (`frame_id`).

## Gravity prior and its limit

Single-frame ground detection uses `base_link` z as "up" (via `T_base_camera`). This assumes the
robot base is level; on slopes or with a mis-calibrated mount the angular gate (default 12°) will
reject the true floor and the result is `ground_status = unavailable`, never a guessed floor.
