import { useEffect, useRef } from "react";
import type { MapGrid } from "./api";

type Props = {
  grid: MapGrid | null;
  path: number[][] | null;
  start: number[] | null;
  goal: number[] | null;
  robotXY: number[] | null;
  tracks: { center_m: number[]; visible_extent_m: number[]; lifecycle: string }[];
  showUnknown: boolean;
  onClick: (xy: [number, number]) => void;
  selected: number[] | null;
  viewpoints?: { x: number; y: number; yaw: number }[];
  diff?: Int8Array | null; // per cell: 0 same, 1 became occupied, 2 became free, 3 became unknown
  objects?: { name: string; center_m: number[]; size_m: number[] }[]; // simulator truth (demo controls)
};

const DIFF_COLORS: Record<number, number[]> = { 1: [224, 49, 49], 2: [47, 158, 68], 3: [240, 140, 0] };

const SCALE = 4; // pixels per cell

// Map frame: x right, y up. Canvas: row 0 at the top, so y is flipped (display only).
export function Map2D({ grid, path, start, goal, robotXY, tracks, showUnknown, onClick, selected, viewpoints = [], diff = null, objects = [] }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = ref.current;
    if (!c || !grid) return;
    c.width = grid.nx * SCALE;
    c.height = grid.ny * SCALE;
    const ctx = c.getContext("2d")!;
    const img = ctx.createImageData(grid.nx, grid.ny);
    for (let i = 0; i < grid.nx; i++) {
      for (let j = 0; j < grid.ny; j++) {
        const v = grid.data[i * grid.ny + j];
        const p = ((grid.ny - 1 - j) * grid.nx + i) * 4;
        const d = diff ? diff[i * grid.ny + j] : 0;
        const col = d ? DIFF_COLORS[d] : v === 100 ? [25, 25, 30] : v === 0 ? [246, 246, 240] : showUnknown ? [150, 152, 165] : [246, 246, 240];
        img.data.set([...col, 255], p);
      }
    }
    const tmp = document.createElement("canvas");
    tmp.width = grid.nx;
    tmp.height = grid.ny;
    tmp.getContext("2d")!.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, c.width, c.height);

    const px = (x: number, y: number): [number, number] => [
      ((x - grid.ox) / grid.res) * SCALE,
      (grid.ny - (y - grid.oy) / grid.res) * SCALE,
    ];
    ctx.setLineDash([4, 3]);
    for (const o of objects) {
      const [x0, y0] = px(o.center_m[0] - o.size_m[0] / 2, o.center_m[1] + o.size_m[1] / 2);
      ctx.strokeStyle = "#ae3ec9";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x0, y0, (o.size_m[0] / grid.res) * SCALE, (o.size_m[1] / grid.res) * SCALE);
      ctx.fillStyle = "#ae3ec9";
      ctx.fillText(o.name, x0 + 2, y0 - 3);
    }
    ctx.setLineDash([]);
    for (const t of tracks) {
      const [x0, y0] = px(t.center_m[0] - t.visible_extent_m[0] / 2, t.center_m[1] + t.visible_extent_m[1] / 2);
      ctx.strokeStyle = t.lifecycle === "confirmed" ? "#e8590c" : "#adb5bd";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x0, y0, (t.visible_extent_m[0] / grid.res) * SCALE, (t.visible_extent_m[1] / grid.res) * SCALE);
    }
    if (path && path.length > 1) {
      ctx.strokeStyle = "#e03131";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      path.forEach((p, k) => (k ? ctx.lineTo(...px(p[0], p[1])) : ctx.moveTo(...px(p[0], p[1]))));
      ctx.stroke();
    }
    const dot = (p: number[] | null, color: string, r = 6) => {
      if (!p) return;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(...px(p[0], p[1]), r, 0, Math.PI * 2);
      ctx.fill();
    };
    viewpoints.forEach((v, k) => {
      const [x, y] = px(v.x, v.y);
      const L = k === 0 ? 22 : 12;
      ctx.strokeStyle = k === 0 ? "#f08c00" : "rgba(240,140,0,0.45)";
      ctx.lineWidth = k === 0 ? 3 : 1.5;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + L * Math.cos(v.yaw), y - L * Math.sin(v.yaw));
      ctx.stroke();
      ctx.fillStyle = ctx.strokeStyle;
      ctx.beginPath();
      ctx.arc(x, y, k === 0 ? 5 : 3, 0, Math.PI * 2);
      ctx.fill();
    });
    dot(robotXY, "#7048e8", 5);
    dot(start, "#2f9e44");
    dot(goal, "#1c7ed6");
    if (selected) {
      const [sx, sy] = px(selected[0], selected[1]);
      ctx.strokeStyle = "#f08c00";
      ctx.lineWidth = 2;
      ctx.strokeRect(sx - SCALE, sy - SCALE, SCALE * 2, SCALE * 2);
    }
  }, [grid, path, start, goal, robotXY, tracks, showUnknown, selected, viewpoints, diff, objects]);

  if (!grid) return <div className="empty">No 2D map yet (no floor established).</div>;
  return (
    <canvas
      ref={ref}
      className="map2d"
      onClick={(e) => {
        const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
        const sx = ((e.clientX - r.left) / r.width) * grid.nx;
        const sy = ((e.clientY - r.top) / r.height) * grid.ny;
        onClick([grid.ox + sx * grid.res, grid.oy + (grid.ny - sy) * grid.res]);
      }}
    />
  );
}
