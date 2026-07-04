"""Rank-preserving per-batch score calibration."""
from __future__ import annotations
from typing import Sequence
import numpy as np
BATCH_CALIBRATION_METHODS = ("none", "clip_below")
def _clamp(v): return max(0.0, min(1.0, float(v)))
def apply_clip_below(scores, human_low=0.05, human_hi=0.49):
    a = np.asarray(scores, dtype=np.float64); n = len(a)
    if n == 0: return a
    order = np.argsort(-a); out = np.empty(n)
    if n == 1: out[order[0]] = human_hi; return out
    for i, idx in enumerate(order):
        out[idx] = human_hi - (i/(n-1))*(human_hi-human_low)
    return out
def apply_batch_calibration(scores, method="clip_below"):
    name = str(method or "none").strip().lower()
    cal = [_clamp(v) for v in scores] if name in ("", "none") else apply_clip_below(scores)
    return [round(_clamp(float(v)), 6) for v in cal]
