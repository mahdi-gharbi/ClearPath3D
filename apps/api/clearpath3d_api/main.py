"""ClearPath3D local API (FastAPI). Thin layer over ``clearpath3d.runtime.session.ReplaySession``.

Run:  .venv/Scripts/python -m uvicorn clearpath3d_api.main:app --app-dir apps/api --port 8010
Long computations never run on the event loop: session methods are called via ``run_in_threadpool``
and each session processes frames on its own single worker thread.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from clearpath3d.config import load_config
from clearpath3d.datasets.synthetic import SCENES, build_scene, generate
from clearpath3d.planning.astar import RobotSpec
from clearpath3d.runtime.replay import ReplayConfig
from clearpath3d.runtime.session import ReplaySession
from clearpath3d_api.sim import Driver, SimWorld

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "synthetic"

app = FastAPI(title="ClearPath3D API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
SESSIONS: dict[str, ReplaySession] = {}
WORLDS: dict[str, SimWorld] = {}
DRIVERS: dict[str, Driver] = {}


class RobotIn(BaseModel):
    radius_m: float = Field(0.25, gt=0, le=2)
    height_m: float = Field(0.40, gt=0, le=3)
    margin_m: float = Field(0.05, ge=0, le=1)

    def spec(self) -> RobotSpec:
        return RobotSpec(self.radius_m, self.height_m, self.margin_m)


class ReplayIn(BaseModel):
    scene: str | None = Field(None, description=f"synthetic scene: {', '.join(SCENES)}")
    sequence_path: str | None = Field(
        None, description="existing sequence directory (manifest.json)"
    )
    config: str = "configs/synthetic.yaml"
    robot: RobotIn = RobotIn()
    tsdf: bool = False
    pose_mode: Literal["supplied", "icp"] = "supplied"


class CommandIn(BaseModel):
    command: Literal["play", "pause", "step", "reset"]
    steps: int = Field(1, ge=1, le=1000)
    fps: float = Field(2.0, gt=0, le=30)


class PlanIn(BaseModel):
    start: tuple[float, float]
    goal: tuple[float, float]
    robot: RobotIn | None = None


def _session(rid: str) -> ReplaySession:
    s = SESSIONS.get(rid)
    if s is None:
        raise HTTPException(404, f"replay {rid} not found")
    return s


@app.get("/v1/health")
def health() -> dict:
    return {"status": "ok", "replays": len(SESSIONS), "scenes": list(SCENES)}


@app.post("/v1/replays")
async def create_replay(body: ReplayIn) -> dict:
    if (body.scene is None) == (body.sequence_path is None):
        raise HTTPException(422, "give exactly one of scene or sequence_path")
    if body.scene is not None:
        if body.scene not in SCENES:
            raise HTTPException(422, f"unknown scene {body.scene}")
        root = DATA / body.scene
        if not (root / "manifest.json").exists():
            await run_in_threadpool(generate, build_scene(body.scene), root)
    else:
        root = Path(body.sequence_path)
        if not (root / "manifest.json").exists():
            raise HTTPException(422, "sequence_path has no manifest.json")
    cfg = ReplayConfig(
        pipeline=load_config(REPO / body.config), robot=body.robot.spec(), tsdf_enabled=body.tsdf
    )
    rid = uuid.uuid4().hex[:10]
    SESSIONS[rid] = await run_in_threadpool(ReplaySession, rid, root, cfg, 1000, body.pose_mode)
    if body.scene is not None:
        d = cfg.pipeline.depth
        WORLDS[rid] = SimWorld(body.scene, d.min_depth_m, d.max_depth_m)
        DRIVERS[rid] = Driver(SESSIONS[rid], WORLDS[rid])
    return {
        "replay_id": rid,
        "sequence": SESSIONS[rid].seq.sequence_id,
        "num_frames": len(SESSIONS[rid].seq),
    }


@app.post("/v1/replays/{rid}/commands")
async def command(rid: str, body: CommandIn) -> dict:
    s = _session(rid)
    if body.command == "step":
        out = await run_in_threadpool(s.step, body.steps)
        return {"stepped": len(out), "frame_index": s.index, "finished": s.finished}
    if body.command == "play":
        s.play(body.fps)
    elif body.command == "pause":
        s.pause()
    else:
        await run_in_threadpool(s.reset)
    return {"ok": True, "frame_index": s.index}


@app.get("/v1/replays/{rid}/snapshot")
async def snapshot(rid: str) -> dict:
    return await run_in_threadpool(_session(rid).snapshot)


@app.get("/v1/replays/{rid}/events")
async def events(rid: str, request: Request, after: int = 0) -> StreamingResponse:
    """Server-sent events: small JSON metadata only (versions, timings, warnings), never point clouds."""
    s = _session(rid)

    async def gen():
        last = after
        while not await request.is_disconnected():
            evs = await run_in_threadpool(s.events_after, last, 10.0)
            if not evs:
                yield ": keep-alive\n\n"
                continue
            for e in evs:
                last = e.seq
                yield f"id: {e.seq}\nevent: {e.kind}\ndata: {json.dumps(e.data)}\n\n"
            await asyncio.sleep(0)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/v1/replays/{rid}/artifacts/{key}")
async def artifact(rid: str, key: Literal["map2d", "points", "voxels", "blocked"]) -> Response:
    s = _session(rid)
    fn = {
        "map2d": s.map_bytes,
        "points": s.points_bytes,
        "voxels": s.occupied_voxels_bytes,
        "blocked": s.blocked_bytes,
    }[key]
    res = await run_in_threadpool(fn)
    if res is None:
        raise HTTPException(404, f"{key} not available yet")
    data, headers = res
    headers["X-Bytes"] = str(len(data))
    return Response(content=data, media_type="application/octet-stream", headers=headers)


@app.post("/v1/replays/{rid}/plans")
async def plans(rid: str, body: PlanIn) -> dict:
    s = _session(rid)
    robot = body.robot.spec() if body.robot else None
    return await run_in_threadpool(s.plan, body.start, body.goal, robot)


@app.get("/v1/replays/{rid}/viewpoints")
async def viewpoints(rid: str, top_k: int = 10) -> dict:
    """Next-best-view proposal from the observed map (may take a few seconds; runs off the event loop)."""
    return await run_in_threadpool(_session(rid).viewpoints, top_k)


@app.get("/v1/replays/{rid}/evidence/{evidence_id}")
async def evidence(rid: str, evidence_id: str) -> dict:
    s = _session(rid)
    try:
        return await run_in_threadpool(s.evidence, evidence_id)
    except (KeyError, ValueError) as e:
        raise HTTPException(404, str(e)) from e


class MoveIn(BaseModel):
    name: str
    x: float
    y: float


class DriveIn(BaseModel):
    goal: tuple[float, float]
    robot: RobotIn | None = None
    final_yaw: float | None = None
    step_m: float = Field(0.25, gt=0.05, le=1.0)


def _world(rid: str) -> SimWorld:
    _session(rid)
    w = WORLDS.get(rid)
    if w is None:
        raise HTTPException(404, "no simulator for this replay (only synthetic scenes have one)")
    return w


@app.get("/v1/replays/{rid}/world")
async def world(rid: str) -> dict:
    """SIMULATOR controls (full world). Shown for the demo only; perception never reads this."""
    w = _world(rid)
    return {
        "objects": w.objects(),
        "sim_time_s": round(w.t, 3),
        "rendered_views": w.n_rendered,
        "note": "simulator ground truth, for demo controls only",
    }


@app.post("/v1/replays/{rid}/world/move")
async def world_move(rid: str, body: MoveIn) -> dict:
    try:
        return _world(rid).move_object(body.name, body.x, body.y)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/v1/replays/{rid}/drive")
async def drive(rid: str, body: DriveIn) -> dict:
    """Start a simulated drive to ``goal`` (re-plans at every step); progress is streamed as SSE 'drive'."""
    _world(rid)
    drv = DRIVERS[rid]
    if drv.running:
        raise HTTPException(409, "a drive is already running")
    robot = body.robot.spec() if body.robot else None
    s = SESSIONS[rid]
    asyncio.get_running_loop().run_in_executor(
        None,
        lambda: drv.run(body.goal, robot or s.cfg.robot, body.step_m, final_yaw=body.final_yaw),
    )
    return {"started": True}


@app.post("/v1/replays/{rid}/drive/stop")
async def drive_stop(rid: str) -> dict:
    _world(rid)
    DRIVERS[rid].stop_flag.set()
    return {"stopping": True}


@app.get("/v1/replays/{rid}/timeline")
async def timeline(rid: str) -> dict:
    return {"steps": await run_in_threadpool(_session(rid).timeline)}


@app.get("/v1/replays/{rid}/timeline/{step}/map2d")
async def timeline_map(rid: str, step: int) -> Response:
    res = await run_in_threadpool(_session(rid).map_bytes_at, step)
    if res is None:
        raise HTTPException(404, "no map at this step")
    data, headers = res
    return Response(content=data, media_type="application/octet-stream", headers=headers)


@app.get("/v1/replays/{rid}/diff")
async def diff(rid: str, a: int, b: int) -> dict:
    try:
        return await run_in_threadpool(_session(rid).map_diff, a, b)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@app.delete("/v1/replays/{rid}")
async def delete(rid: str) -> dict:
    if rid in DRIVERS:
        DRIVERS.pop(rid).stop_flag.set()
    WORLDS.pop(rid, None)
    s = SESSIONS.pop(rid, None)
    if s is None:
        raise HTTPException(404, f"replay {rid} not found")
    await run_in_threadpool(s.close)
    return {"deleted": rid}
