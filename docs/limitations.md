# Known limitations (kept current)

- **Synthetic sensor model.** Gaussian noise growing with z² plus random/rectangular dropout. It is
  not a model of a RealSense or any other device. No multipath, flying pixels, IR shadows or
  rolling shutter.
- **Flat-colour rendering.** Colour is albedo × Lambertian shading; no textures. Fine for geometry,
  useless for evaluating a 2D detector.
- **Level-base assumption.** Single-frame ground detection uses `base_link` z as gravity. A tilted
  robot or wrong mount calibration leads to `ground_status=unavailable` (by design: no guess).
- **Visible boxes only.** Obstacle boxes cover the observed points; a box seen from one side is
  thinner than the real object. Walls that touch form a single cluster.
- **Point-level ground metrics are approximate:** each 3 cm voxel is labelled through one
  representative pixel.
- **Supplied poses only (M2).** Reconstruction uses dataset/simulator poses; no odometry is evaluated.
- **TSDF is not free space.** The TSDF mesh is for presentation; free space will come from the ray-based map (M3).
- **Redwood sample**: 5 frames, ~10 cm of motion, no mount calibration (no ground/traversability),
  licence not verified. It checks formats and conventions, not generalisation.
- **Open3D pinned to 0.19.0** because of a TSDF regression in the 0.20.0 Windows wheel.
- **Stale free space.** Space observed free stays free until re-observed or expired; an object moved into it
  unseen is a false free (measured: 48 cells in moving_box at frame 12).
- **5 cm voxels.** Surfaces on voxel boundaries spread over two cells; narrow passages lose up to ~5 cm per side.
- **Conservative free criterion.** Cells near the far edge of the view or only seen at floor level stay
  unknown (free coverage 44–64 % in the synthetic scenes).
- **Tracking is geometric.** Fragments of walls/furniture can become tracks; identities are not guaranteed
  across long occlusions; a teleported object starts a new track. `occluded` also covers out-of-view.
- **Robot model.** Cylinder on a flat floor; no slopes, friction or dynamics. Paths are geometric simulations.
- **Speed.** ~0.6–0.7 s per 320×240 frame on a laptop CPU; not real time.
- **ROS2 not verified.** The adapter code exists and its pure conversions are unit-tested, but it has not run
  in ROS 2 yet (ROS not installed). Do not claim ROS integration until `ros2_ws/verify_log.txt` shows it.
- **Next-best-view**: evaluated on 2 synthetic scenes × 3 noise seeds only; coarse rays (24×16, 5 cm steps)
  can skip voxel corners; the hidden box was found in 1/3 runs; ~1–2.5 s per decision.
- **Held-out evaluation is synthetic only**: one procedural generator family (axis-aligned boxes, flat colours,
  one camera). Real sequences with ground truth have not been benchmarked.
- **ICP odometry**: local only (no loop closure); depth-only geometry degenerates along walls/floors, where it
  refuses frames and can lose tracking (83/121 frames on room_scan_dense). Thresholds tuned on that sequence.
- **Conservative planner**: extra dilation near obstacles and unknown reduces reachable viewpoints (the hidden
  box of `hidden_box` is no longer found by NBV in the reference runs).
- **Viewer** is a local development tool (no auth, CORS limited to localhost:5173).
- **Semantics**: COCO classes only; evaluated with an oracle on synthetic data and only for multi-view
  consistency on 5 real frames (one false positive seen). No accuracy claim on real data.
- **Real data**: one TUM robot sequence, no geometric ground truth of the scene; maps are compared with the
  mocap-pose map, not with the world. Far Kinect readings create noisy occupied cells. Depth-only ICP fails there.
- **Blind zone**: the camera does not see ~1 m in front of the robot; the simulated driver turns to look before
  moving, but an object appearing inside the blind zone after the last look cannot be seen by this sensor.
