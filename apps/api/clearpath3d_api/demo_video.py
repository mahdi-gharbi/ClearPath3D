"""Render the portfolio demo video (MP4) from the real system state — no screen recording needed.

Story (plan section 1): the camera explores the room and the map builds up; a box is moved by the simulator
into the corridor the robot had seen free; the robot drives to a goal, looks before moving, discovers the box
and re-plans; a before/after diff shows what changed; finally the next-best-view is proposed.

Each frame = camera RGB (what the robot sees) | depth | 2D map (robot, plan, simulator-truth box outline
dashed in magenta) + caption. Every pixel comes from the pipeline outputs or the simulator renders.

    .venv/Scripts/python -m clearpath3d_api.demo_video --out artifacts/demo/demo_video.mp4
Requires: pip install -r requirements-video.txt (imageio, imageio-ffmpeg).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from clearpath3d.config import load_config
from clearpath3d.datasets.synthetic import build_scene, generate
from clearpath3d.mapping.projection import FREE, OCCUPIED, UNKNOWN
from clearpath3d.mapping.render import render_map
from clearpath3d.runtime.replay import ReplayConfig
from clearpath3d.runtime.session import ReplaySession
from clearpath3d_api.sim import Driver, SimWorld

W, H = 1600, 720
PANEL_H = 600


def _font(size: int):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE, F_TEXT, F_SMALL = _font(30), _font(22), _font(17)


def _depth_rgb(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    d = np.clip(depth / 6.0, 0, 1)
    rgb = np.stack([(1 - d) * 255, (0.5 - np.abs(d - 0.5)) * 400, d * 255], axis=-1).clip(0, 255).astype(np.uint8)
    rgb[~valid] = 0  # holes = no measurement (black), never free
    return rgb


def _fit(img: np.ndarray, w: int, h: int) -> Image.Image:
    im = Image.fromarray(img)
    im.thumbnail((w, h))
    canvas = Image.new("RGB", (w, h), (24, 26, 30))
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def _map_panel(session: ReplaySession, world: SimWorld, path, goal, diff_base=None, viewpoint=None) -> np.ndarray:
    m = session.map2d()
    if m is None:
        return np.full((PANEL_H, 640, 3), 60, np.uint8)
    pose = session.current_pose()
    img = render_map(m, path, pose[0] if pose else None, goal, scale=4).copy()
    if diff_base is not None:  # colour cells that changed since the "before" snapshot
        cur = m.state
        for code, col in ((OCCUPIED, (224, 49, 49)), (FREE, (47, 158, 68)), (UNKNOWN, (240, 140, 0))):
            mask = (diff_base != cur) & (cur == code)
            mm = np.repeat(np.repeat(np.flipud(mask.T), 4, 0), 4, 1)
            img[mm] = col
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    ny = m.state.shape[1]

    def px(x, y):
        return ((x - m.origin_xy[0]) / m.resolution_m * 4, (ny - (y - m.origin_xy[1]) / m.resolution_m) * 4)

    for o in world.objects():  # simulator truth, dashed outline (display only)
        if o["category"] != "box":
            continue
        c, s = o["center_m"], o["size_m"]
        x0, y0 = px(c[0] - s[0] / 2, c[1] + s[1] / 2)
        x1, y1 = px(c[0] + s[0] / 2, c[1] - s[1] / 2)
        for k in range(int(x0), int(x1), 6):
            dr.line([(k, y0), (min(k + 3, x1), y0)], fill=(174, 62, 201), width=2)
            dr.line([(k, y1), (min(k + 3, x1), y1)], fill=(174, 62, 201), width=2)
        for k in range(int(y0), int(y1), 6):
            dr.line([(x0, k), (x0, min(k + 3, y1))], fill=(174, 62, 201), width=2)
            dr.line([(x1, k), (x1, min(k + 3, y1))], fill=(174, 62, 201), width=2)
    if pose:  # robot heading
        (x, y), yaw = pose
        cx, cy = px(x, y)
        dr.line([(cx, cy), (cx + 22 * math.cos(yaw), cy - 22 * math.sin(yaw))], fill=(112, 72, 232), width=4)
    if viewpoint is not None:
        cx, cy = px(viewpoint["x"], viewpoint["y"])
        dr.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=(240, 140, 0))
        dr.line([(cx, cy), (cx + 30 * math.cos(viewpoint["yaw"]), cy - 30 * math.sin(viewpoint["yaw"]))],
                fill=(240, 140, 0), width=4)
    return np.asarray(im)


def _compose(session, world, title, caption, path=None, goal=None, diff_base=None, viewpoint=None, status=""):
    f = session.last_frame
    frame = Image.new("RGB", (W, H), (18, 20, 24))
    dr = ImageDraw.Draw(frame)
    dr.text((24, 14), title, fill=(240, 240, 240), font=F_TITLE)
    if f is not None:
        frame.paste(_fit(f.color_rgb, 460, 345), (20, 70))
        frame.paste(_fit(_depth_rgb(f.depth_m, f.depth_valid), 460, 345), (20, 70 + 345 + 10))
        dr.text((28, 74), "camera RGB", fill=(255, 255, 255), font=F_SMALL)
        dr.text((28, 74 + 355), "depth (black = no measurement)", fill=(255, 255, 255), font=F_SMALL)
    frame.paste(_fit(_map_panel(session, world, path, goal, diff_base, viewpoint), 1080, PANEL_H - 20), (500, 70))
    dr.text((500, 50), "2D map: white = observed free · black = occupied · grey = UNKNOWN · "
            "magenta dashed = true box (simulator) · green = robot · blue = goal", fill=(200, 200, 200), font=F_SMALL)
    dr.rectangle([0, H - 52, W, H], fill=(36, 40, 48))
    dr.text((24, H - 42), caption, fill=(255, 220, 120), font=F_TEXT)
    if status:
        dr.text((W - 24 - dr.textlength(status, font=F_SMALL), 22), status, fill=(170, 200, 255), font=F_SMALL)
    return np.asarray(frame)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/demo/demo_video.mp4")
    ap.add_argument("--fps", type=float, default=4.0)
    args = ap.parse_args()
    cfg = load_config("configs/synthetic.yaml")
    root = generate(build_scene("room_scan"), Path("data/synthetic/room_scan"))
    session = ReplaySession("video", root, ReplayConfig(pipeline=cfg, tsdf_enabled=False))
    d = cfg.depth
    world = SimWorld("room_scan", d.min_depth_m, d.max_depth_m)
    frames: list[np.ndarray] = []

    def hold(img, seconds):
        frames.extend([img] * max(1, int(seconds * args.fps)))

    title = "ClearPath3D — can the robot pass, what blocks it, where should it look?"
    n = len(session.seq)
    for i in range(n):
        session.step(1)
        hold(_compose(session, world, title, "1/5  The camera explores: the map is built from depth rays; "
                      "unseen space stays UNKNOWN (grey), never free.", status=f"frame {i + 1}/{n} · pose: supplied"), 1.0)
        if i == 17:  # mid-exploration: unknown space left -> where to look next?
            vp = session.viewpoints()
            best = vp.get("best")
            cap = (f"2/5  Where to look next? Best reachable view (orange): predicted gain {best['gain_voxels']} "
                   "unknown voxels, from the observed map only." if best else f"2/5  Next-best-view: {vp['status']}")
            hold(_compose(session, world, title, cap, viewpoint=best, status=f"next-best-view: {vp['status']}"), 6)
    before = session.map2d().state.copy()
    goal = (1.2, 2.35)
    r = session.plan(session.current_pose()[0].tolist(), goal)
    hold(_compose(session, world, title, "3/5  Plan to the goal through OBSERVED free space only (red path, "
                  "robot radius 0.25 m).", path=r["path_m"], goal=goal, status="A*"), 5)
    world.move_object("box_b", 2.2, 2.35)
    hold(_compose(session, world, title, "4/5  The simulator moves a box onto that corridor. The map is stale: "
                  "it still says free there.", path=r["path_m"], goal=goal, status="simulator edit (truth, dashed)"), 5)

    drv = Driver(session, world)
    events = []
    orig_emit = session.emit

    def tap(kind, data=None):
        orig_emit(kind, data)
        if kind == "drive":
            events.append(dict(data or {}))
            st = data.get("status")
            if st in ("moving", "turning"):
                p = data.get("path_m") or next((e["path_m"] for e in reversed(events) if e.get("path_m")), None)
                txt = ("4/5  Look before moving (1 m camera blind zone): the box is seen, the plan changes."
                       if st == "turning" else "4/5  Drive one step, observe, re-plan on the new map.")
                hold(_compose(session, world, title, txt, path=p, goal=goal,
                              status=f"{st} · replans {data.get('replans', '-')}"), 1.0)

    session.emit = tap
    res = drv.run(goal, session.cfg.robot)
    session.emit = orig_emit
    hold(_compose(session, world, title, f"4/5  Goal {res['status']}: {res['steps']} steps, {res['replans']} re-plans; "
                  "the robot disk never touched the true box.", goal=goal, status=f"drive {res['status']}"), 5)
    hold(_compose(session, world, title, "5/5  What changed since before the drive? red = became occupied, "
                  "green = became free, orange = became unknown.", goal=goal, diff_base=before,
                  status="before/after diff"), 7)
    session.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=args.fps, codec="libx264", quality=7, macro_block_size=8)
    print(f"wrote {out} ({len(frames)} frames, {len(frames) / args.fps:.1f} s); drive result: {res}")


if __name__ == "__main__":
    main()
