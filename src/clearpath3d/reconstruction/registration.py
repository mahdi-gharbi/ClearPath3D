"""Frame-to-keyframe RGB-D registration (local odometry), ICP point-to-plane with a robust kernel.

This is *local odometry*, not SLAM: there is no loop closure, drift accumulates. Scores (fitness, inlier
RMSE) measure geometric agreement; they are not a probability that a pose is correct.

Gauge: the map frame is the robot base at the first frame (``T_map_camera_0 = T_base_camera``), so the
map stays z-up through the mount calibration. Estimated trajectories are compared to reference poses
after aligning the first frame (documented in the evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from clearpath3d.contracts import PoseSource, RGBDFrame
from clearpath3d.geometry.frames import invert
from clearpath3d.geometry.projection import backproject


@dataclass
class ICPParams:
    # coarse -> fine. The 0.2 m level widens the basin: with (0.08, 0.04, 0.02) a 13 deg turn converged to
    # a wrong minimum (0.49 m error); with the 0.2 m level the same pair is within 1 mm / 0.02 deg.
    voxel_sizes_m: tuple[float, ...] = (0.2, 0.08, 0.04, 0.02)
    max_iterations: tuple[int, ...] = (30, 30, 20, 15)
    corr_factor: float = 3.0  # max correspondence distance = factor * voxel
    method: str = "point_to_plane"  # point_to_plane | point_to_point
    robust_kernel: str = "tukey"  # tukey | none
    tukey_k_factor: float = 1.0  # k = factor * voxel
    init: str = "constant_velocity"  # constant_velocity | identity
    min_fitness: float = 0.3
    max_inlier_rmse_m: float = 0.03
    max_translation_m: float = 0.5  # per frame, motion plausibility
    max_rotation_deg: float = 35.0
    max_range_m: float = 4.0
    max_consecutive_rejections: int = (
        10  # room_scan_dense: 5 -> lost at frame 55; 10 -> recovers (83/121)
    )
    # A converged ICP with high fitness can still be wrong (degenerate geometry, e.g. sliding along a wall:
    # measured 0.25 m errors at fitness 0.98). Two extra gates:
    max_deviation_from_prediction_m: float = 0.10  # vs every motion hypothesis, per elapsed frame
    # Degeneracy gate: smallest eigenvalue of the normalised point-to-plane information sum(n n^T)/N over the
    # final correspondences (1 = isotropic). Sliding along a wall + floor gives < 0.025; well-constrained
    # frames 0.13-0.29 on room_scan_dense, where this threshold was chosen (then checked on other sequences).
    # Open3D's get_information_matrix_from_point_clouds was NOT usable: its translation block is N*I.
    min_p2plane_information: float = 0.03
    global_fallback: bool = True  # FPFH + RANSAC initialisation when the local ICP is rejected
    fpfh_voxel_m: float = 0.05


@dataclass
class RegistrationResult:
    accepted: bool
    # (fields below kept in order of construction)
    T_prev_cur: np.ndarray
    fitness: float
    inlier_rmse_m: float
    translation_m: float
    rotation_deg: float
    reasons: list[str] = field(default_factory=list)
    min_p2plane_information: float = float("nan")


def _cloud(frame: RGBDFrame, max_range: float) -> o3d.geometry.PointCloud:
    valid = frame.depth_valid & (frame.depth_m <= max_range)
    pts, _ = backproject(frame.depth_m, valid, frame.intrinsics)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))


def _rot_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def p2plane_information(src, tgt, T: np.ndarray, vox: float, corr: float) -> float:
    """Smallest eigenvalue of sum(n n^T)/N over nearest-neighbour correspondences at pose ``T``."""
    from scipy.spatial import cKDTree

    from clearpath3d.geometry.frames import transform_points

    s = np.asarray(src.voxel_down_sample(vox).points)
    t = tgt.voxel_down_sample(vox)
    t.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * vox, max_nn=30))
    tp, tn = np.asarray(t.points), np.asarray(t.normals)
    if len(s) == 0 or len(tp) == 0:
        return 0.0
    dist, idx = cKDTree(tp).query(transform_points(T, s), distance_upper_bound=corr)
    ok = np.isfinite(dist)
    if ok.sum() < 30:
        return 0.0
    n = tn[idx[ok]]
    return float(np.linalg.eigvalsh(n.T @ n / ok.sum())[0])


def global_init(
    src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud, voxel: float, seed: int = 0
) -> np.ndarray:
    """FPFH features + RANSAC (seeded) coarse alignment, used only as an initial guess for ICP."""
    reg = o3d.pipelines.registration
    o3d.utility.random.seed(seed)

    def feat(c):
        d = c.voxel_down_sample(voxel)
        d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel, max_nn=30))
        return d, reg.compute_fpfh_feature(
            d, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel, max_nn=100)
        )

    s, fs = feat(src)
    t, ft = feat(tgt)
    g = reg.registration_ransac_based_on_feature_matching(
        s,
        t,
        fs,
        ft,
        True,
        1.5 * voxel,
        reg.TransformationEstimationPointToPoint(False),
        3,
        [
            reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            reg.CorrespondenceCheckerBasedOnDistance(1.5 * voxel),
        ],
        reg.RANSACConvergenceCriteria(100000, 0.999),
    )
    return np.asarray(g.transformation)


def register(
    src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud, init: np.ndarray, p: ICPParams
) -> RegistrationResult:
    """Estimate ``T_tgt_src`` (maps source-camera points into the target camera)."""
    reg = o3d.pipelines.registration
    T = init.copy()
    res = None
    for vox, iters in zip(p.voxel_sizes_m, p.max_iterations, strict=True):
        s, t = src.voxel_down_sample(vox), tgt.voxel_down_sample(vox)
        if len(s.points) < 30 or len(t.points) < 30:
            return RegistrationResult(False, init, 0.0, float("inf"), 0.0, 0.0, ["too few points"])
        param = o3d.geometry.KDTreeSearchParamHybrid(radius=2 * vox, max_nn=30)
        s.estimate_normals(param)
        t.estimate_normals(param)
        kernel = reg.TukeyLoss(k=p.tukey_k_factor * vox) if p.robust_kernel == "tukey" else None
        if p.method == "point_to_plane":
            est = (
                reg.TransformationEstimationPointToPlane(kernel)
                if kernel
                else reg.TransformationEstimationPointToPlane()
            )
        else:
            est = reg.TransformationEstimationPointToPoint()
        res = reg.registration_icp(
            s, t, p.corr_factor * vox, T, est, reg.ICPConvergenceCriteria(max_iteration=iters)
        )
        T = res.transformation
    min_info = p2plane_information(
        src, tgt, T, 2 * p.voxel_sizes_m[-1], p.corr_factor * 2 * p.voxel_sizes_m[-1]
    )
    tr = float(np.linalg.norm(T[:3, 3]))
    rot = _rot_deg(T[:3, :3])
    reasons = []
    if res.fitness < p.min_fitness:
        reasons.append(f"fitness {res.fitness:.2f} < {p.min_fitness}")
    if res.inlier_rmse > p.max_inlier_rmse_m:
        reasons.append(f"inlier rmse {res.inlier_rmse:.3f} > {p.max_inlier_rmse_m}")
    if tr > p.max_translation_m:
        reasons.append(f"translation {tr:.2f} m > {p.max_translation_m}")
    if rot > p.max_rotation_deg:
        reasons.append(f"rotation {rot:.1f} deg > {p.max_rotation_deg}")
    if p.min_p2plane_information and min_info < p.min_p2plane_information:
        reasons.append(f"translation weakly constrained (p2plane information {min_info:.3f})")
    return RegistrationResult(
        not reasons,
        np.asarray(T),
        float(res.fitness),
        float(res.inlier_rmse),
        tr,
        rot,
        reasons,
        min_info,
    )


class ICPOdometry:
    """Wraps a sequence of frames (poses ignored) and yields frames with ``pose_source=estimated``.

    Rejected frames are yielded with ``pose_source=unavailable`` (no pose): downstream modules then keep
    the last map and say so. After too many consecutive rejections, tracking is declared lost.
    """

    def __init__(
        self,
        T_base_camera: np.ndarray,
        p: ICPParams | None = None,
        T_map_camera_0: np.ndarray | None = None,
    ):
        """``T_map_camera_0``: optional initial pose anchor (e.g. the first supplied pose only). Without it,
        the map is the robot base at the first frame (``T_base_camera``)."""
        self.p = p or ICPParams()
        self.T_map_camera_0 = T_map_camera_0
        self.T_map_cam_kf: np.ndarray | None = None  # pose of the reference (last accepted) frame
        self.kf_cloud: o3d.geometry.PointCloud | None = None
        self.last_rel = np.eye(4)
        self.T_base_camera = T_base_camera
        self.rejections = 0
        self.lost = False
        self.log: list[dict] = []

    def process(self, frame: RGBDFrame) -> RGBDFrame:
        cloud = _cloud(frame, self.p.max_range_m)
        if self.T_map_cam_kf is None:
            T0 = self.T_base_camera if self.T_map_camera_0 is None else self.T_map_camera_0
            self.T_map_cam_kf, self.kf_cloud = T0.copy(), cloud
            self.log.append(
                {"frame_id": frame.frame_id, "accepted": True, "reason": "gauge (first frame)"}
            )
            return _with_pose(frame, self.T_map_cam_kf, PoseSource.estimated)
        if self.lost:
            self.log.append(
                {"frame_id": frame.frame_id, "accepted": False, "reason": "tracking lost"}
            )
            return _with_pose(frame, None, PoseSource.unavailable)
        k = self.rejections + 1  # frames elapsed since the keyframe
        # Motion hypotheses: constant velocity (composed over k frames) and "no motion". Velocity can change
        # abruptly (start/stop, turn -> drive), so ICP is tried from both and the motion gate accepts a result
        # close to either hypothesis. ICP estimates T_kf_cur (current camera -> keyframe camera).
        hyps = [("zero", np.eye(4))]
        if self.p.init == "constant_velocity":
            hyps.insert(0, ("constant_velocity", np.linalg.matrix_power(self.last_rel, k)))
        tried = [(name, H, register(cloud, self.kf_cloud, H, self.p)) for name, H in hyps]
        used_global = False
        if self.p.global_fallback and not any(r.accepted for _, _, r in tried):
            G = global_init(cloud, self.kf_cloud, self.p.fpfh_voxel_m)
            tried.append(("global", G, register(cloud, self.kf_cloud, G, self.p)))
            used_global = True

        def deviation(r: RegistrationResult) -> float:
            return min(float(np.linalg.norm((invert(H) @ r.T_prev_cur)[:3, 3])) for _, H in hyps)

        for _, _, r in tried:
            if (
                r.accepted
                and p_dev(self.p)
                and deviation(r) > self.p.max_deviation_from_prediction_m * k
            ):
                r.accepted = False
                r.reasons.append(f"{deviation(r):.2f} m from every motion hypothesis")
        ok = [t for t in tried if t[2].accepted]
        init_name, _, r = max(ok or tried, key=lambda t: t[2].fitness)
        entry = {
            "frame_id": frame.frame_id,
            "accepted": r.accepted,
            "init": init_name,
            "global_fallback": used_global,
            "fitness": round(r.fitness, 4),
            "inlier_rmse_m": round(r.inlier_rmse_m, 5),
            "translation_m": round(r.translation_m, 4),
            "rotation_deg": round(r.rotation_deg, 2),
            "min_p2plane_information": round(r.min_p2plane_information, 4),
            "reasons": r.reasons,
        }
        self.log.append(entry)
        if not r.accepted:
            self.rejections += 1
            if self.rejections >= self.p.max_consecutive_rejections:
                self.lost = True
            return _with_pose(frame, None, PoseSource.unavailable)
        # per-frame motion estimate (k-th root approximated by the motion over k frames when k == 1)
        self.last_rel = (
            r.T_prev_cur if self.rejections == 0 else _fractional(r.T_prev_cur, self.rejections + 1)
        )
        self.rejections = 0
        T = self.T_map_cam_kf @ r.T_prev_cur
        self.T_map_cam_kf, self.kf_cloud = T, cloud
        return _with_pose(frame, T, PoseSource.estimated)


def _fractional(T: np.ndarray, k: int) -> np.ndarray:
    """Approximate per-frame motion from a k-frame motion (axis-angle and translation divided by k)."""
    from scipy.spatial.transform import Rotation

    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(Rotation.from_matrix(T[:3, :3]).as_rotvec() / k).as_matrix()
    out[:3, 3] = T[:3, 3] / k
    return out


def p_dev(p: ICPParams) -> bool:
    return p.max_deviation_from_prediction_m is not None and p.max_deviation_from_prediction_m > 0


def _with_pose(frame: RGBDFrame, T: np.ndarray | None, src: PoseSource) -> RGBDFrame:
    return RGBDFrame(
        frame.sequence_id,
        frame.frame_id,
        frame.timestamp_s,
        frame.color_rgb,
        frame.depth_m,
        frame.depth_valid,
        frame.intrinsics,
        T,
        src,
        frame.calibration_id,
        T_base_camera=frame.T_base_camera,
        metadata=frame.metadata,
    )


def trajectory_errors(est: list[np.ndarray | None], ref: list[np.ndarray]) -> dict:
    """ATE/RPE after aligning the first frame (est[0] := ref[0]). Frames without an estimate are skipped."""
    A = ref[0] @ invert(est[0])
    pairs = [(A @ e, r) for e, r in zip(est, ref, strict=True) if e is not None]
    te = np.array([np.linalg.norm(e[:3, 3] - r[:3, 3]) for e, r in pairs])
    rpe_t, rpe_r = [], []
    idx = [k for k, e in enumerate(est) if e is not None]
    for a, b in zip(idx[:-1], idx[1:], strict=True):
        de = invert(est[a]) @ est[b]
        dr = invert(ref[a]) @ ref[b]
        err = invert(dr) @ de
        rpe_t.append(np.linalg.norm(err[:3, 3]))
        rpe_r.append(_rot_deg(err[:3, :3]))
    return {
        "alignment": "first frame",
        "frames_estimated": len(pairs),
        "frames_total": len(est),
        "ate_rmse_m": round(float(np.sqrt(np.mean(te**2))), 4) if len(te) else None,
        "ate_max_m": round(float(te.max()), 4) if len(te) else None,
        "rpe_trans_rmse_m": round(float(np.sqrt(np.mean(np.square(rpe_t)))), 4) if rpe_t else None,
        "rpe_rot_rmse_deg": round(float(np.sqrt(np.mean(np.square(rpe_r)))), 3) if rpe_r else None,
    }
