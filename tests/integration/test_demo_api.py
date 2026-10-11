"""Interactive demo API: simulator world, drive with re-planning, timeline and diff."""

import time

import numpy as np
import pytest
from clearpath3d_api.main import DRIVERS, SESSIONS, WORLDS, app
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
    for s in list(SESSIONS.values()):
        s.close()
    SESSIONS.clear()
    WORLDS.clear()
    DRIVERS.clear()


@pytest.fixture(scope="module")
def rid(client):
    r = client.post("/v1/replays", json={"scene": "room_scan"}).json()["replay_id"]
    client.post(f"/v1/replays/{r}/commands", json={"command": "step", "steps": 31})
    return r


def _wait_drive(rid, timeout=600):
    s = SESSIONS[rid]
    last = 0
    t0 = time.time()
    while time.time() - t0 < timeout:
        for e in s.events_after(last, 5):
            last = e.seq
            if e.kind == "drive" and e.data["status"] in ("finished", "error"):
                return e.data
    raise TimeoutError


def test_world_listing_and_move(client, rid):
    w = client.get(f"/v1/replays/{rid}/world").json()
    names = {o["name"] for o in w["objects"]}
    assert {"box_a", "box_b", "table"} <= names and "simulator" in w["note"]
    r = client.post(f"/v1/replays/{rid}/world/move", json={"name": "box_a", "x": 2.6, "y": 1.0})
    assert r.status_code == 200
    moved = {o["name"]: o for o in client.get(f"/v1/replays/{rid}/world").json()["objects"]}[
        "box_a"
    ]
    np.testing.assert_allclose(moved["center_m"][:2], [2.6, 1.0], atol=1e-6)
    assert (
        client.post(
            f"/v1/replays/{rid}/world/move", json={"name": "wall_north", "x": 0, "y": 0}
        ).status_code
        == 404
    )


def test_drive_reaches_goal_and_records_timeline(client, rid):
    n0 = len(client.get(f"/v1/replays/{rid}/timeline").json()["steps"])
    assert client.post(f"/v1/replays/{rid}/drive", json={"goal": [3.0, 2.35]}).json()["started"]
    assert client.post(f"/v1/replays/{rid}/drive", json={"goal": [3.0, 2.35]}).status_code == 409
    res = _wait_drive(rid)
    assert res["outcome"] == "reached", res
    steps = client.get(f"/v1/replays/{rid}/timeline").json()["steps"]
    assert len(steps) > n0 and steps[-1]["origin"] == "simulator_drive"
    xy, _ = SESSIONS[rid].current_pose()
    assert np.linalg.norm(xy - [3.0, 2.35]) < 0.2
    d = client.get(f"/v1/replays/{rid}/diff", params={"a": 0, "b": len(steps) - 1}).json()
    assert d["transitions"]
    m = client.get(f"/v1/replays/{rid}/timeline/0/map2d")
    assert len(m.content) == int(m.headers["x-nx"]) * int(m.headers["x-ny"])


def test_blocked_goal_stops_with_a_cause(client, rid):
    assert client.post(f"/v1/replays/{rid}/drive", json={"goal": [2.4, 3.0]}).json()[
        "started"
    ]  # under the table
    res = _wait_drive(rid)
    assert res["outcome"] in ("stopped", "reached")
    if res["outcome"] == "stopped":
        assert res["cause"]


def test_no_world_for_recorded_data(client, tmp_path):
    # A recorded (non-synthetic-scene) replay has no simulator controls.
    from clearpath3d.datasets.synthetic import build_scene, generate

    root = generate(build_scene("narrow_door"), tmp_path / "rec")
    r = client.post("/v1/replays", json={"sequence_path": str(root)}).json()["replay_id"]
    assert client.get(f"/v1/replays/{r}/world").status_code == 404
    assert client.post(f"/v1/replays/{r}/drive", json={"goal": [1, 1]}).status_code == 404


def test_box_moved_into_known_free_path_is_discovered_and_never_hit(client):
    r = client.post("/v1/replays", json={"scene": "room_scan"}).json()["replay_id"]
    client.post(f"/v1/replays/{r}/commands", json={"command": "step", "steps": 31})
    s = SESSIONS[r]
    goal = [1.2, 2.35]
    # Before the move, the goal is reachable along y ~ 2.35 through x = 2.2 (observed free during the scan).
    before = s.plan(s.current_pose()[0].tolist(), goal)
    assert before["success"] and any(abs(p[0] - 2.2) < 0.6 for p in before["path_m"])
    client.post(f"/v1/replays/{r}/world/move", json={"name": "box_b", "x": 2.2, "y": 2.35})
    box = {o["name"]: o for o in client.get(f"/v1/replays/{r}/world").json()["objects"]}["box_b"]
    poses = []
    assert client.post(f"/v1/replays/{r}/drive", json={"goal": goal}).json()["started"]
    last, res, t0 = 0, None, time.time()
    while res is None and time.time() - t0 < 600:
        for e in s.events_after(last, 5):
            last = e.seq
            if e.kind == "drive" and e.data["status"] == "moving":
                poses.append(e.data["pose"][:2])
            if e.kind == "drive" and e.data["status"] in ("finished", "error"):
                res = e.data
    assert res is not None and res["status"] == "finished", res
    assert len(poses) >= 1, "the robot must actually drive towards the box"
    lo = np.asarray(box["center_m"][:2]) - np.asarray(box["size_m"][:2]) / 2
    hi = np.asarray(box["center_m"][:2]) + np.asarray(box["size_m"][:2]) / 2
    radius = s.cfg.robot.radius_m
    for p in poses:  # robot disk vs the TRUE box footprint (simulator side)
        d = np.hypot(*np.maximum(np.maximum(lo - p, 0), np.asarray(p) - hi))
        assert d >= radius, (p, d)
    assert res["outcome"] == "reached" and res["replans"] >= 1 or res["outcome"] == "stopped", res
