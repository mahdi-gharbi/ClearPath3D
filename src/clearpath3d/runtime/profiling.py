"""Process-level memory and latency helpers (measured, never estimated)."""

from __future__ import annotations

import platform

import numpy as np
import psutil


def memory_mb() -> dict[str, float]:
    """Current RSS and, on Windows, the process peak working set (includes Open3D's C++ heap)."""
    info = psutil.Process().memory_info()
    out = {"rss_mb": round(info.rss / 2**20, 1)}
    peak = getattr(info, "peak_wset", None)
    if peak is not None:
        out["peak_working_set_mb"] = round(peak / 2**20, 1)
    return out


def percentiles(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {"n": 0}
    a = np.asarray(values_ms, dtype=np.float64)
    return {
        "n": int(a.size),
        "p50_ms": round(float(np.percentile(a, 50)), 3),
        "p95_ms": round(float(np.percentile(a, 95)), 3),
        "max_ms": round(float(a.max()), 3),
    }


def hardware() -> dict[str, str | int | float]:
    return {
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": f"{platform.system()} {platform.release()}",
    }
