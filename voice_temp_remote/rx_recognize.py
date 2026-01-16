"""DTW-based template matching for command recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class RecognitionResult:
    command: str
    distance: float


def dtw_distance(x: np.ndarray, y: np.ndarray, band: int | None = None) -> float:
    n, m = x.shape[0], y.shape[0]
    if band is None:
        band = max(n, m)
    band = max(band, abs(n - m))
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m + 1, i + band + 1)
        for j in range(j_start, j_end):
            dist = np.linalg.norm(x[i - 1] - y[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m])


def recognize_command(
    feat: np.ndarray, templates: Dict[str, Iterable[np.ndarray]], band: int | None = None
) -> RecognitionResult:
    best_cmd = "unknown"
    best_dist = float("inf")
    for cmd, tmpl_list in templates.items():
        for tmpl in tmpl_list:
            dist = dtw_distance(feat, tmpl, band=band)
            if dist < best_dist:
                best_dist = dist
                best_cmd = cmd
    return RecognitionResult(command=best_cmd, distance=best_dist)
