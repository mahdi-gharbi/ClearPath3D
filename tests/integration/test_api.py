"""API: replay lifecycle, binary artefacts, plans, evidence, deterministic reset, SSE metadata."""

import json
import threading

import numpy as np
import pytest
from clearpath3d_api.main import SESSIONS, app
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
    for s in list(SESSIONS.values()):
        s.close()
    SESSIONS.clear()


@pytest.fixture(scope="module")
def rid(client):
    r = client.post("/v1/replays", json={"scene": "narrow_door"})
    assert r.status_code == 200, r.text
    return r.json()["replay_id"]


def test_create_validation(client):
    assert client.post("/v1/replays", json={}).status_code == 422
    assert client.post("/v1/replays", json={"scene": "nope"}).status_code == 422
    assert client.get("/v1/replays/doesnotexist/snapshot").status_code == 404


def test_step_snapshot_and_artifacts(client, rid):
    r = client.post(f"/v1/replays/{rid}/commands", json={"command": "step", "steps": 4})
    assert r.json()["stepped"] == 4
    snap = client.get(f"/v1/replays/{rid}/snapshot").json()
    assert snap["frame_index"] == 4 and snap["snapshot"]["pose_status"] == "supplied"
    assert snap["map"] is not None and snap["map"]["counts"]["free"] > 0
    m = client.get(f"/v1/replays/{rid}/artifacts/map2d")
    nx, ny = int(m.headers["x-nx"]), int(m.headers["x-ny"])
    grid = np.frombuffer(m.content, dtype=np.int8).reshape(nx, ny)
    assert set(np.unique(grid)) <= {-1, 0, 100}
    p = client.get(f"/v1/replays/{rid}/artifacts/points")
    n = int(p.headers["x-count"])
    assert len(p.content) == n * 12 + n * 3 and n < 20000  # downsampled, not the raw frame
    v = client.get(f"/v1/replays/{rid}/artifacts/voxels")
    assert len(v.content) == int(v.headers["x-count"]) * 12


def test_plan_and_evidence(client, rid):
    r = client.post(
        f"/v1/replays/{rid}/plans", json={"start": [0.6, 2.375], "goal": [3.95, 2.375]}
    ).json()
    assert r["success"] is False and r["cause"] in {"goal_footprint_blocked", "goal_unknown"}
    e = client.get(f"/v1/replays/{rid}/evidence/point:1.5:2.4").json()
    assert e["state"] in {"free", "occupied", "unknown"} and "criterion" in e and e["voxels"]
    assert client.get(f"/v1/replays/{rid}/evidence/bogus:1").status_code == 404
    vp = client.get(f"/v1/replays/{rid}/viewpoints").json()
    assert vp["status"] in {"ok", "no_informative_candidate", "no_reachable_candidate"}
    if vp["status"] == "ok":
        assert vp["best"]["gain_voxels"] >= 1 and vp["top"][0]["score"] == vp["best"]["score"]


def test_reset_is_deterministic(client, rid):
    client.post(f"/v1/replays/{rid}/commands", json={"command": "reset"})
    client.post(f"/v1/replays/{rid}/commands", json={"command": "step", "steps": 3})
    a = client.get(f"/v1/replays/{rid}/artifacts/map2d").content
    client.post(f"/v1/replays/{rid}/commands", json={"command": "reset"})
    assert client.get(f"/v1/replays/{rid}/snapshot").json()["frame_index"] == 0
    client.post(f"/v1/replays/{rid}/commands", json={"command": "step", "steps": 3})
    b = client.get(f"/v1/replays/{rid}/artifacts/map2d").content
    assert a == b


def test_events_carry_metadata_only(rid):
    s = SESSIONS[rid]
    seen = []
    t = threading.Thread(target=lambda: seen.extend(s.events_after(10**9 - 1, timeout_s=0.1)))
    t.start()
    t.join()
    evs = s.events_after(0, timeout_s=0.1)
    steps = [e for e in evs if e.kind == "step"]
    assert steps and all(len(json.dumps(e.data)) < 4000 for e in steps)
    assert {"occupancy_version", "frame_id", "pose_source"} <= set(steps[-1].data)


def test_icp_pose_mode_reports_estimated_poses(client):
    r = client.post("/v1/replays", json={"scene": "narrow_door", "pose_mode": "icp"})
    assert r.status_code == 200, r.text
    rid2 = r.json()["replay_id"]
    client.post(f"/v1/replays/{rid2}/commands", json={"command": "step", "steps": 3})
    snap = client.get(f"/v1/replays/{rid2}/snapshot").json()
    assert snap["pose_mode"] == "icp"
    assert snap["snapshot"]["pose_status"] in {"estimated", "unavailable"}
    assert snap["odometry_lost"] in (True, False)
    assert (
        client.post("/v1/replays", json={"scene": "narrow_door", "pose_mode": "magic"}).status_code
        == 422
    )
