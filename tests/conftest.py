from pathlib import Path

import pytest

from clearpath3d.config import load_config
from clearpath3d.datasets.synthetic import build_scene, generate

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg():
    return load_config(REPO / "configs" / "synthetic.yaml")


@pytest.fixture(scope="session")
def scene_dir(tmp_path_factory):
    """Generate (once per session) a synthetic scene at test resolution; returns a factory."""
    cache: dict[str, Path] = {}

    def make(name: str, **overrides) -> Path:
        key = name + repr(sorted(overrides.items()))
        if key not in cache:
            spec = build_scene(name, **overrides)
            cache[key] = generate(spec, tmp_path_factory.mktemp(name))
        return cache[key]

    return make
