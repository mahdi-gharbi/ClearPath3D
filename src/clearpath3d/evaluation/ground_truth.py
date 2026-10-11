"""Ground-truth access for the evaluator ONLY. Pipeline modules must not import this module."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GTFrame:
    depth_clean: np.ndarray
    instance: np.ndarray
    T_map_camera: np.ndarray
    T_map_base: np.ndarray


class GroundTruth:
    def __init__(self, root: Path | str):
        self.root = Path(root) / "ground_truth"
        data = json.loads((self.root / "objects.json").read_text())
        self.objects = data["objects"]
        self.floor_plane_map = data["floor_plane_map"]
        self.categories = np.array([o["category"] for o in self.objects])

    def frame(self, frame_id: str) -> GTFrame:
        with np.load(self.root / "frames" / f"{frame_id}.npz") as z:
            return GTFrame(z["depth_clean"], z["instance"], z["T_map_camera"], z["T_map_base"])

    def physical_objects(self) -> dict[str, list[int]]:
        """Group mesh parts into physical objects (table top + legs -> 'table')."""
        groups: dict[str, list[int]] = {}
        for o in self.objects:
            if o["category"] in ("floor", "wall"):
                continue
            key = o["name"].split("_")[0] if o["category"] == "table" else o["name"]
            groups.setdefault(key, []).append(o["index"])
        return groups

    def full_aabb(self, indices: list[int], t: float) -> tuple[np.ndarray, np.ndarray]:
        los, his = [], []
        for i in indices:
            o = self.objects[i]
            off = np.zeros(3)
            for t0, d in o["motion"]:
                if t >= t0:
                    off = np.asarray(d, dtype=np.float64)
            los.append(np.asarray(o["min_m"]) + off)
            his.append(np.asarray(o["max_m"]) + off)
        return np.min(los, axis=0), np.max(his, axis=0)
