import { useCallback, useEffect, useRef, useState } from "react";
import { api, type MapGrid, type PlanResult, type Robot, type Snapshot, type TimelineStep, type Viewpoints, type WorldObject } from "./api";
import { Map2D } from "./Map2D";
import { Scene3D } from "./Scene3D";

type ClickMode = "start" | "goal" | "why" | "move";
type DriveState = { status: string; outcome?: string; path_m?: number[][]; replans?: number; cause?: string; message?: string; steps?: number };

function diffCodes(a: MapGrid, b: MapGrid): Int8Array {
  const out = new Int8Array(b.data.length);
  for (let k = 0; k < out.length; k++) {
    if (a.data[k] === b.data[k]) continue;
    out[k] = b.data[k] === 100 ? 1 : b.data[k] === 0 ? 2 : 3;
  }
  return out;
}

export default function App() {
  const [scenes, setScenes] = useState<string[]>([]);
  const [scene, setScene] = useState("room_scan");
  const [robot, setRobot] = useState<Robot>({ radius_m: 0.25, height_m: 0.4, margin_m: 0.05 });
  const [rid, setRid] = useState<string | null>(null);
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [grid, setGrid] = useState<MapGrid | null>(null);
  const [voxels, setVoxels] = useState<{ xyz: Float32Array; size: number } | null>(null);
  const [points, setPoints] = useState<{ xyz: Float32Array; rgb: Uint8Array } | null>(null);
  const [bytes, setBytes] = useState(0);
  const [mode, setMode] = useState<ClickMode>("goal");
  const [start, setStart] = useState<number[] | null>(null);
  const [goal, setGoal] = useState<number[] | null>(null);
  const [plan, setPlan] = useState<PlanResult | null>(null);
  const [why, setWhy] = useState<any>(null);
  const [whyXY, setWhyXY] = useState<number[] | null>(null);
  const [vp, setVp] = useState<Viewpoints | null>(null);
  const [vpBusy, setVpBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(2);
  const [poseMode, setPoseMode] = useState("supplied");
  const [show, setShow] = useState({ voxels: true, points: true, tracks: true, ground: true, unknown: true });
  const pending = useRef(false);
  const [objects, setObjects] = useState<WorldObject[]>([]);
  const [moving, setMoving] = useState<string | null>(null);
  const [drive, setDrive] = useState<DriveState | null>(null);
  const [steps, setSteps] = useState<TimelineStep[]>([]);
  const [viewStep, setViewStep] = useState<number | null>(null); // null = live map
  const [viewGrid, setViewGrid] = useState<MapGrid | null>(null);
  const [diffBase, setDiffBase] = useState<{ step: number; grid: MapGrid } | null>(null);

  useEffect(() => {
    api.health().then((h) => setScenes(h.scenes)).catch((e) => setError(String(e)));
  }, []);

  const refresh = useCallback(async (id: string) => {
    if (pending.current) return; // UI refresh is rate-limited: skip while a refresh is in flight
    pending.current = true;
    try {
      const [s, g, v, p] = await Promise.all([api.snapshot(id), api.map(id), api.voxels(id), api.points(id)]);
      setSnap(s);
      setGrid(g);
      setVoxels(v);
      setPoints(p);
      setBytes((g ? g.data.byteLength : 0) + v.bytes + (p ? p.bytes : 0));
      api.timeline(id).then((t) => setSteps(t.steps)).catch(() => undefined);
      api.world(id).then((w) => setObjects(w.objects)).catch(() => setObjects([]));
    } catch (e) {
      setError(String(e));
    } finally {
      pending.current = false;
    }
  }, []);

  // SSE: small metadata events; artefacts are fetched separately, at most every 500 ms.
  useEffect(() => {
    if (!rid) return;
    const es = new EventSource(`/v1/replays/${rid}/events`);
    let timer: number | null = null;
    const schedule = () => {
      if (timer === null) timer = window.setTimeout(() => ((timer = null), refresh(rid)), 500);
    };
    ["step", "reset", "end", "pause"].forEach((k) => es.addEventListener(k, schedule));
    es.addEventListener("drive", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setDrive((prev) => ({ ...prev, ...d }));
      schedule();
    });
    return () => {
      es.close();
      if (timer !== null) clearTimeout(timer);
    };
  }, [rid, refresh]);

  const create = async () => {
    setError(null);
    setPlan(null);
    setWhy(null);
    setVp(null);
    const r = await api.create(scene, robot, poseMode).catch((e) => (setError(String(e)), null));
    if (!r) return;
    setRid(r.replay_id);
    await api.command(r.replay_id, "step", 1);
    await refresh(r.replay_id);
    const T = (await api.snapshot(r.replay_id)).T_map_base;
    if (T) setStart([T[0][3], T[1][3]]);
  };

  const cmd = async (c: string, steps = 1) => {
    if (!rid) return;
    await api.command(rid, c, steps, fps).catch((e) => setError(String(e)));
    if (c !== "play") refresh(rid);
  };

  const runPlan = async () => {
    if (!rid || !start || !goal) return;
    setPlan(await api.plan(rid, start, goal, robot).catch((e) => (setError(String(e)), null)));
  };

  const nextBestView = async () => {
    if (!rid) return;
    setVpBusy(true);
    const r = await api.viewpoints(rid).catch((e) => (setError(String(e)), null));
    setVpBusy(false);
    setVp(r);
    if (r?.best) setGoal([r.best.x, r.best.y]);
  };

  // Live sliders: re-plan automatically (debounced) when the robot changes and a plan exists.
  useEffect(() => {
    if (!rid || !start || !goal || !plan) return;
    const t = window.setTimeout(runPlan, 300);
    return () => clearTimeout(t);
  }, [robot]); // eslint-disable-line react-hooks/exhaustive-deps

  const driveToGoal = async () => {
    if (!rid || !goal) return;
    setDrive({ status: "starting" });
    const yaw = vp?.best && goal[0] === vp.best.x && goal[1] === vp.best.y ? vp.best.yaw : null;
    await api.drive(rid, goal, robot, yaw).catch((e) => (setError(String(e)), setDrive(null)));
  };

  const showStep = async (k: number | null) => {
    setViewStep(k);
    if (rid && k !== null) setViewGrid(await api.mapAt(rid, k));
  };

  const shownGrid = viewStep === null ? grid : viewGrid;
  const diff = diffBase && shownGrid && diffBase.grid.data.length === shownGrid.data.length ? diffCodes(diffBase.grid, shownGrid) : null;
  const diffCounts = diff ? [1, 2, 3].map((c) => diff.reduce((n, v) => n + (v === c ? 1 : 0), 0)) : null;

  const setBefore = () => {
    const k = viewStep ?? steps.length - 1;
    const g = viewStep === null ? grid : viewGrid;
    if (g) setDiffBase({ step: k, grid: g });
  };

  const inspectChange = async (k: number) => {
    if (!rid) return;
    setWhy(await api.evidence(rid, `change:${k}`).then((c) => ({ change: c })).catch((e) => ({ error: String(e) })));
  };

  const onMapClick = async (xy: [number, number]) => {
    if (mode === "move" && rid && moving) {
      await api.moveObject(rid, moving, xy[0], xy[1]).catch((e) => setError(String(e)));
      setMoving(null);
      setMode("goal");
      setObjects((await api.world(rid)).objects);
      return;
    }
    if (mode === "start") setStart(xy);
    else if (mode === "goal") setGoal(xy);
    else if (rid) {
      setWhyXY(xy);
      setWhy(await api.evidence(rid, `point:${xy[0].toFixed(3)}:${xy[1].toFixed(3)}`).catch((e) => ({ error: String(e) })));
    }
  };

  const s = snap?.snapshot;
  const T = snap?.T_map_base ?? null;
  return (
    <div className="app">
      <header>
        <strong>ClearPath3D</strong>
        <select value={scene} onChange={(e) => setScene(e.target.value)}>
          {scenes.map((n) => (
            <option key={n}>{n}</option>
          ))}
        </select>
        <label className="slider">R {robot.radius_m.toFixed(2)} m <input type="range" min={0.08} max={0.45} step={0.01} value={robot.radius_m} onChange={(e) => setRobot({ ...robot, radius_m: +e.target.value })} /></label>
        <label className="slider">H {robot.height_m.toFixed(2)} m <input type="range" min={0.15} max={1.2} step={0.05} value={robot.height_m} onChange={(e) => setRobot({ ...robot, height_m: +e.target.value })} /></label>
        <label className="slider">margin {robot.margin_m.toFixed(2)} m <input type="range" min={0} max={0.2} step={0.01} value={robot.margin_m} onChange={(e) => setRobot({ ...robot, margin_m: +e.target.value })} /></label>
        <label>poses <select value={poseMode} onChange={(e) => setPoseMode(e.target.value)}>
          <option value="supplied">supplied</option>
          <option value="icp">estimated (ICP)</option>
        </select></label>
        <button onClick={create}>Load replay</button>
        <span className="sep" />
        <button disabled={!rid} onClick={() => cmd("step")}>Step</button>
        <button disabled={!rid} onClick={() => cmd("play")}>Play</button>
        <button disabled={!rid} onClick={() => cmd("pause")}>Pause</button>
        <button disabled={!rid} onClick={() => (cmd("reset"), setPlan(null), setVp(null))}>Reset</button>
        <label>fps <input type="number" min={0.5} max={10} step={0.5} value={fps} onChange={(e) => setFps(+e.target.value)} /></label>
        {snap && (
          <span className="badges">
            <span className="badge">frame {snap.frame_index}/{snap.num_frames}</span>
            <span className={`badge pose-${s?.pose_status}`}>pose: {s?.pose_status ?? "?"}</span>
            <span className={`badge ground-${s?.ground_status}`}>ground: {s?.ground_status ?? "?"}</span>
            {snap.odometry_lost && <span className="badge pose-unavailable">tracking lost: map frozen</span>}
          </span>
        )}
      </header>
      {error && <div className="error" onClick={() => setError(null)}>{error}</div>}
      <main>
        <section className="view3d">
          <div className="toggles">
            {(["voxels", "points", "tracks", "ground", "unknown"] as const).map((k) => (
              <label key={k}>
                <input type="checkbox" checked={show[k]} onChange={(e) => setShow({ ...show, [k]: e.target.checked })} /> {k}
              </label>
            ))}
          </div>
          <Scene3D voxels={voxels} points={points} tracks={s?.tracks ?? []} path={plan?.success ? plan.path_m : null}
            T_map_base={T} robot={robot} floorZ={snap?.map?.floor_z_m ?? 0} show={show} />
          <div className="legend">3D: occupied voxels coloured by height, current frame points, wireframe = visible extent of tracks. Display rotated z-up &rarr; y-up (data unchanged).</div>
        </section>
        <section className="side">
          <div className="modes">
            Click on map:
            {(["start", "goal", "why"] as const).map((m) => (
              <button key={m} className={mode === m ? "active" : ""} onClick={() => setMode(m)}>{m === "why" ? "why?" : `set ${m}`}</button>
            ))}
            <button disabled={!start || !goal || !rid} onClick={runPlan}>Plan</button>
            <button disabled={!rid || vpBusy} onClick={nextBestView}>{vpBusy ? "computing..." : "Next best view"}</button>
          </div>
          <Map2D grid={shownGrid} path={drive?.status === "moving" && drive.path_m ? drive.path_m : plan?.success ? plan.path_m : null} start={start} goal={goal}
            diff={diff} objects={mode === "move" ? objects : []}
            robotXY={T ? [T[0][3], T[1][3]] : null} tracks={s?.tracks ?? []} showUnknown={show.unknown}
            onClick={onMapClick} selected={mode === "why" ? whyXY : null}
            viewpoints={vp ? [...(vp.best ? [vp.best] : []), ...vp.top.slice(1, 8)] : []} />
          <div className="legend">white = observed free &middot; black = occupied &middot; grey = unknown &middot; orange arrows = proposed viewpoints &middot; R={robot.radius_m} m, H={robot.height_m} m, margin={robot.margin_m} m, cell={snap?.map?.resolution_m ?? "?"} m</div>
          <div className="panel">
            <b>Demo (simulated robot and world)</b>
            <div className="row">
              <button disabled={!rid || !goal || objects.length === 0 || drive?.status === "moving"} onClick={driveToGoal}>Drive to goal</button>
              <button disabled={!rid || drive?.status !== "moving"} onClick={() => rid && api.driveStop(rid)}>Stop</button>
              {drive && (
                <span className={drive.cause ? "warn" : "muted"}>
                  {drive.status}{drive.outcome ? `: ${drive.outcome}` : ""}{drive.replans !== undefined ? ` · replans ${drive.replans}` : ""}{drive.steps !== undefined ? ` · ${drive.steps} steps` : ""}
                  {drive.cause ? ` · ${drive.cause}` : ""}{drive.message ? ` · ${drive.message}` : ""}
                </span>
              )}
            </div>
            {objects.length > 0 ? (
              <div className="row">
                Move (simulator):
                {objects.map((o) => (
                  <button key={o.name} className={moving === o.name ? "active" : ""} onClick={() => (setMoving(o.name), setMode("move"))}>{o.name}</button>
                ))}
                {moving && <span className="muted">then click the target on the map</span>}
              </div>
            ) : (
              <div className="muted">No simulator for this replay (recorded data): drive and world edits are unavailable.</div>
            )}
            <div className="muted">Simulator controls act on the true world; perception only sees the newly rendered images.</div>
            {steps.length > 0 && (
              <>
                <div className="row">
                  Timeline:
                  <input type="range" min={0} max={steps.length - 1} value={viewStep ?? steps.length - 1} onChange={(e) => showStep(+e.target.value)} style={{ flex: 1 }} />
                  <button className={viewStep === null ? "active" : ""} onClick={() => showStep(null)}>live</button>
                </div>
                <div className="muted">
                  {viewStep === null ? "live map" : `step ${viewStep} · frame ${steps[viewStep]?.frame_id} · ${steps[viewStep]?.origin}`}
                  {" · "}<span className="link" onClick={setBefore}>set as before</span>
                  {diffBase && <> · <span className="link" onClick={() => setDiffBase(null)}>clear diff</span></>}
                </div>
                {diffBase && diffCounts && (
                  <div>
                    diff vs step {diffBase.step}: <span style={{ color: "#e03131" }}>+occupied {diffCounts[0]}</span> &middot;{" "}
                    <span style={{ color: "#2f9e44" }}>+free {diffCounts[1]}</span> &middot;{" "}
                    <span style={{ color: "#f08c00" }}>+unknown {diffCounts[2]}</span>
                  </div>
                )}
              </>
            )}
          </div>
          {plan && (
            <div className={`panel ${plan.success ? "ok" : "fail"}`}>
              <b>{plan.success ? `Path ${plan.length_m.toFixed(2)} m` : `No path: ${plan.cause}`}</b>
              <div>{plan.message}</div>
              {plan.success && plan.margin_cells > 0 && <div>{plan.margin_cells} path cells inside the safety margin band</div>}
              <div className="muted">{plan.note}</div>
            </div>
          )}
          {vp && (
            <div className={`panel ${vp.best ? "ok" : "fail"}`}>
              <b>Next best view: {vp.status}</b>
              <div>{vp.message}</div>
              {vp.best && (
                <div>
                  at ({vp.best.x.toFixed(2)}, {vp.best.y.toFixed(2)}) yaw {((vp.best.yaw * 180) / Math.PI).toFixed(0)}&deg; &middot; predicted gain{" "}
                  {vp.best.gain_voxels} voxels &middot; travel {vp.best.path_length_m.toFixed(2)} m &middot; score {vp.best.score.toFixed(3)} (goal set; press Plan)
                </div>
              )}
              <div className="muted">{vp.candidates} reachable candidates &middot; {vp.frontier_cells} frontier cells &middot; {vp.note}</div>
            </div>
          )}
          {why?.change && (
            <div className="panel">
              <b>change {why.change.kind} @ frame {why.change.frame_id}</b>
              <div>{why.change.voxel_count} voxels, box {JSON.stringify(why.change.min_m)} &rarr; {JSON.stringify(why.change.max_m)}</div>
              <div>now: {why.change.voxels_now.occupied} occupied, {why.change.voxels_now.free} free &middot; last frames: {why.change.last_frames_in_region.join(", ")}</div>
            </div>
          )}
          {why && !why.change && (
            <div className="panel">
              <b>why? {why.error ? "" : `cell ${why.cell} -> ${why.state}`}</b>
              {why.error ? <div>{why.error}</div> : (
                <>
                  <div>ground support: {String(why.ground_support)} &middot; band free fraction: {why.band_free_fraction}</div>
                  <div className="muted">{why.criterion}</div>
                  <table>
                    <thead><tr><th>z (m)</th><th>log-odds</th><th>obs</th><th>last frame</th><th>source</th></tr></thead>
                    <tbody>
                      {why.voxels.map((v: any) => (
                        <tr key={v.height_above_floor_m}>
                          <td>{v.height_above_floor_m}</td><td>{v.log_odds}</td><td>{v.observations}</td><td>{v.last_frame ?? "-"}</td><td>{v.last_update}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}
            </div>
          )}
          {snap && (
            <div className="panel">
              <b>Map</b> {snap.map ? Object.entries(snap.map.counts).map(([k, v]) => `${k} ${v}`).join(" · ") : snap.map_unavailable_reason}
              <div>tracks: {s?.tracks.length ?? 0} &middot; changes: {snap.changes.length} &middot; transfer last refresh: {(bytes / 1024).toFixed(0)} KiB</div>
              {snap.changes.slice(-5).map((c, k, arr) => (
                <div key={k} className="muted link" onClick={() => inspectChange(snap.changes.length - arr.length + k)}>
                  change {c.kind} @ frame {c.frame_id}: {c.voxel_count} voxels at ({c.center_m.map((x) => x.toFixed(2)).join(", ")}) &mdash; inspect
                </div>
              ))}
              <b>Timings (ms)</b>
              <div className="muted">{s && Object.entries(s.timings_ms).map(([k, v]) => `${k} ${v.toFixed(0)}`).join(" · ")}</div>
              {s?.warnings.map((w, k) => <div key={k} className="warn">warning: {w}</div>)}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
