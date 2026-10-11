"""The pipeline must not read ground truth: static import check + run with ground_truth/ removed."""

import ast
import shutil
from pathlib import Path

import numpy as np

from clearpath3d.datasets.base import SensorSequence
from clearpath3d.runtime.pipeline import analyze_frame

SRC = Path(__file__).resolve().parents[2] / "src" / "clearpath3d"
PIPELINE_PACKAGES = [
    "geometry",
    "datasets/base.py",
    "perception",
    "runtime",
    "reconstruction",
    "mapping",
    "planning",
    "contracts.py",
    "config.py",
]


def _imports(path: Path) -> set[str]:
    mods = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
        elif isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
    return mods


def test_pipeline_modules_do_not_import_evaluation_or_gt():
    files = []
    for p in PIPELINE_PACKAGES:
        q = SRC / p
        files += list(q.rglob("*.py")) if q.is_dir() else [q]
    for f in files:
        for m in _imports(f):
            assert "evaluation" not in m and "ground_truth" not in m, f"{f} imports {m}"
        assert "ground_truth" not in f.read_text() or f.name == "base.py", f
    # base.py mentions ground_truth only in its docstring; it must not build a path to it.
    assert '"ground_truth"' not in (SRC / "datasets" / "base.py").read_text()


def test_pipeline_runs_identically_without_ground_truth(scene_dir, cfg, tmp_path):
    src = scene_dir("room_clear")
    dst = tmp_path / "sensor_only"
    shutil.copytree(src, dst)
    shutil.rmtree(dst / "ground_truth")
    a = analyze_frame(SensorSequence(src)[0], cfg)
    b = analyze_frame(SensorSequence(dst)[0], cfg)
    np.testing.assert_array_equal(a.labels, b.labels)
    assert a.snapshot.detections == b.snapshot.detections
