// Typed client for the ClearPath3D API (proxied by Vite on /v1).

export type Robot = { radius_m: number; height_m: number; margin_m: number };

export type Track = {
  track_id: number;
  center_m: number[];
  visible_extent_m: number[];
  lifecycle: string;
  evidence_frame_ids: string[];
};

export type Snapshot = {
  replay_id: string;
  sequence_id: string;
  frame_index: number;
  num_frames: number;
  playing: boolean;
  finished: boolean;
  T_map_base: number[][] | null;
  T_map_camera: number[][] | null;
  robot: Robot;
  pose_mode: string;
  odometry_lost: boolean | null;
  map: { counts: Record<string, number>; version: number; floor_z_m: number; resolution_m: number } | null;
  map_unavailable_reason: string | null;
  changes: { kind: string; frame_id: string; voxel_count: number; center_m: number[] }[];
  snapshot: {
    frame_id: string;
    timestamp_s: number;
    pose_status: string;
    ground_status: string;
    warnings: string[];
    timings_ms: Record<string, number>;
    tracks: Track[];
    occupancy_version: number;
  } | null;
};

export type MapGrid = { nx: number; ny: number; ox: number; oy: number; res: number; data: Int8Array; version: number };

export type PlanResult = {
  success: boolean;
  cause: string;
  message: string;
  path_m: number[][];
  length_m: number;
  margin_cells: number;
  note?: string;
};

export type WorldObject = { name: string; category: string; center_m: number[]; size_m: number[] };
export type TimelineStep = { step: number; frame_id: string; origin: string; timestamp_s: number; counts: Record<string, number> | null };

async function grid(r: Response): Promise<MapGrid | null> {
  if (r.status === 404) return null;
  const h = r.headers;
  return {
    nx: +h.get("x-nx")!, ny: +h.get("x-ny")!, ox: +h.get("x-origin-x")!, oy: +h.get("x-origin-y")!,
    res: +h.get("x-resolution")!, version: +(h.get("x-version") ?? h.get("x-step") ?? 0),
    data: new Int8Array(await r.arrayBuffer()),
  };
}

const post = (url: string, body: unknown) =>
  fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });

export type Candidate = { x: number; y: number; yaw: number; path_length_m: number; rotation_rad: number; gain_voxels: number; score: number };
export type Viewpoints = { status: string; message: string; best: Candidate | null; top: Candidate[]; candidates: number; frontier_cells: number; note: string };

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export const api = {
  health: () => fetch("/v1/health").then((r) => json<{ scenes: string[] }>(r)),
  create: (scene: string, robot: Robot, pose_mode = "supplied") =>
    fetch("/v1/replays", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ scene, robot, pose_mode }),
    }).then((r) => json<{ replay_id: string; num_frames: number }>(r)),
  command: (rid: string, command: string, steps = 1, fps = 2) =>
    fetch(`/v1/replays/${rid}/commands`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ command, steps, fps }),
    }).then((r) => json<unknown>(r)),
  snapshot: (rid: string) => fetch(`/v1/replays/${rid}/snapshot`).then((r) => json<Snapshot>(r)),
  map: async (rid: string): Promise<MapGrid | null> => {
    const r = await fetch(`/v1/replays/${rid}/artifacts/map2d`);
    if (r.status === 404) return null;
    const h = r.headers;
    return {
      nx: +h.get("x-nx")!,
      ny: +h.get("x-ny")!,
      ox: +h.get("x-origin-x")!,
      oy: +h.get("x-origin-y")!,
      res: +h.get("x-resolution")!,
      version: +h.get("x-version")!,
      data: new Int8Array(await r.arrayBuffer()),
    };
  },
  points: async (rid: string) => {
    const r = await fetch(`/v1/replays/${rid}/artifacts/points`);
    if (!r.ok) return null;
    const n = +r.headers.get("x-count")!;
    const buf = await r.arrayBuffer();
    return { xyz: new Float32Array(buf, 0, n * 3), rgb: new Uint8Array(buf, n * 12, n * 3), bytes: buf.byteLength };
  },
  voxels: async (rid: string) => {
    const r = await fetch(`/v1/replays/${rid}/artifacts/voxels`);
    const n = +r.headers.get("x-count")!;
    const buf = await r.arrayBuffer();
    return { xyz: new Float32Array(buf, 0, n * 3), size: +r.headers.get("x-voxel-size")!, bytes: buf.byteLength };
  },
  plan: (rid: string, start: number[], goal: number[], robot: Robot) =>
    fetch(`/v1/replays/${rid}/plans`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ start, goal, robot }),
    }).then((r) => json<PlanResult>(r)),
  viewpoints: (rid: string) => fetch(`/v1/replays/${rid}/viewpoints`).then((r) => json<Viewpoints>(r)),
  world: (rid: string) => fetch(`/v1/replays/${rid}/world`).then((r) => json<{ objects: WorldObject[]; note: string }>(r)),
  moveObject: (rid: string, name: string, x: number, y: number) => post(`/v1/replays/${rid}/world/move`, { name, x, y }).then((r) => json<unknown>(r)),
  drive: (rid: string, goal: number[], robot: Robot, final_yaw: number | null) =>
    post(`/v1/replays/${rid}/drive`, { goal, robot, final_yaw }).then((r) => json<unknown>(r)),
  driveStop: (rid: string) => post(`/v1/replays/${rid}/drive/stop`, {}).then((r) => json<unknown>(r)),
  timeline: (rid: string) => fetch(`/v1/replays/${rid}/timeline`).then((r) => json<{ steps: TimelineStep[] }>(r)),
  mapAt: (rid: string, step: number) => fetch(`/v1/replays/${rid}/timeline/${step}/map2d`).then(grid),
  evidence: (rid: string, id: string) => fetch(`/v1/replays/${rid}/evidence/${id}`).then((r) => json<any>(r)),
};
