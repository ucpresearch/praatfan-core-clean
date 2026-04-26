"""
Baseline Burg parity via the praatfan_rust backend (uses Rust _resample
+ Rust Burg). Compare aggregate numbers to the Python pure-NumPy backend.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

import praatfan_rust


REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
INCLUDE = [
    "one_two_three_four_five.wav",
    "one_two_three_four_five_16k.wav",
    "one_two_three_four_five_32float.wav",
    "one_two_three_four_five-gain5.wav",
    "one_two_three_four_five.flac",
]


def _praat_formant(path):
    snd = parselmouth.Sound(str(path))
    fmt = call(snd, "To Formant (burg)", 0.005, 5, 5500.0, 0.025, 50.0)
    n = call(fmt, "Get number of frames")
    rows = []
    for i in range(1, n + 1):
        t = call(fmt, "Get time from frame number", i)
        row = {"t": t}
        for k in (1, 2, 3):
            row[f"f{k}"] = call(fmt, "Get value at time", k, t, "Hertz", "Linear")
        rows.append(row)
    return rows


def _rust_formant(path):
    snd = praatfan_rust.Sound.from_file_channel(str(path), 0)
    fmt = snd.to_formant_burg(0.005, 5, 5500.0, 0.025, 50.0)
    times = np.asarray(fmt.xs())
    rows = []
    for i, t in enumerate(times):
        row = {"t": float(t)}
        for k in (1, 2, 3):
            try:
                v = fmt.get_value_at_time(k, float(t))
            except Exception:
                v = float('nan')
            row[f"f{k}"] = float(v) if v is not None and not math.isnan(v) else float('nan')
        rows.append(row)
    return rows


def main():
    diffs = {"f1": [], "f2": [], "f3": []}
    for name in INCLUDE:
        path = FIXTURES / name
        praat = _praat_formant(path)
        ours = _rust_formant(path)
        n = min(len(praat), len(ours))
        for i in range(n):
            if abs(praat[i]["t"] - ours[i]["t"]) > 1e-4:
                continue
            for k in ("f1", "f2", "f3"):
                p = praat[i][k]; o = ours[i][k]
                if math.isnan(p) or (o is None) or math.isnan(o):
                    continue
                diffs[k].append(abs(o - p))

    for k in ("f1", "f2", "f3"):
        d = np.array(diffs[k])
        print(f"{k.upper()}: n={len(d)} mean={np.mean(d):5.2f} p50={np.percentile(d, 50):5.2f} p95={np.percentile(d, 95):5.2f} p99={np.percentile(d, 99):6.2f} max={np.max(d):7.1f}")


if __name__ == "__main__":
    main()
