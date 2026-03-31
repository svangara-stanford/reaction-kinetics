"""Cycle phase segments: early/late rising and early/late falling from drive_sign."""

import numpy as np


def compute_cycle_segment_labels(drive_sign: np.ndarray) -> np.ndarray:
    """
    Assign each timestep a segment label from contiguous blocks of rising (+1) and falling (-1).
    Each block is split into early (first half) and late (second half).
    Returns (T,) array of "early_rising" | "late_rising" | "early_falling" | "late_falling" | "unknown".
    """
    T = len(drive_sign)
    out = np.full(T, "unknown", dtype=object)
    if T == 0:
        return out
    sign = np.asarray(drive_sign, dtype=np.int64)
    i = 0
    while i < T:
        s = i
        while i < T and sign[i] == sign[s]:
            i += 1
        e = i
        n = e - s
        if n == 0:
            continue
        half = n // 2
        if sign[s] == 1:
            out[s : s + half] = "early_rising"
            out[s + half : e] = "late_rising"
        elif sign[s] == -1:
            out[s : s + half] = "early_falling"
            out[s + half : e] = "late_falling"
    return out
