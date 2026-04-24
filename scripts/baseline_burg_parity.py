"""
Baseline parity measurement: praatfan standalone Formant vs parselmouth.

Produces:
  - scripts/baseline_burg_parity.csv  (per-frame F1/2/3, B1/2/3, both sides)
  - stdout summary: mean/max/p95/p99 error per formant and bandwidth, split
    by boundary vs interior frames.

Run from repo root:
    python scripts/baseline_burg_parity.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

from praatfan.sound import Sound as PFSound
from praatfan.formant import sound_to_formant_burg

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
OUT_CSV = REPO / "scripts" / "baseline_burg_parity.csv"

# Standard Praat defaults for "To Formant (burg)"
TIME_STEP = 0.005
MAX_N_FORMANTS = 5
MAX_FORMANT = 5500.0
WINDOW = 0.025
PRE_EMPH = 50.0

# Only process mono fixtures. tam-ha* has unicode that trips parselmouth in
# some setups; skip those two to keep the baseline stable.
INCLUDE = [
    "one_two_three_four_five.wav",
    "one_two_three_four_five_16k.wav",
    "one_two_three_four_five_32float.wav",
    "one_two_three_four_five-gain5.wav",
    "one_two_three_four_five.flac",
]


def _praat_formant(path: Path):
    snd = parselmouth.Sound(str(path))
    fmt = call(snd, "To Formant (burg)",
               TIME_STEP, MAX_N_FORMANTS, MAX_FORMANT, WINDOW, PRE_EMPH)
    n = call(fmt, "Get number of frames")
    rows = []
    for i in range(1, n + 1):
        t = call(fmt, "Get time from frame number", i)
        row = {"t": t}
        for k in (1, 2, 3):
            row[f"f{k}"] = call(fmt, "Get value at time", k, t, "Hertz", "Linear")
            row[f"b{k}"] = call(fmt, "Get bandwidth at time", k, t, "Hertz", "Linear")
        rows.append(row)
    return rows


def _praatfan_formant(path: Path):
    snd = PFSound.from_file(str(path))
    fmt = sound_to_formant_burg(
        snd,
        time_step=TIME_STEP,
        max_num_formants=MAX_N_FORMANTS,
        max_formant_hz=MAX_FORMANT,
        window_length=WINDOW,
        pre_emphasis_from=PRE_EMPH,
    )
    rows = []
    for frame in fmt.frames:
        row = {"t": frame.time}
        for k in (1, 2, 3):
            fp = frame.get_formant(k)
            row[f"f{k}"] = fp.frequency if fp else math.nan
            row[f"b{k}"] = fp.bandwidth if fp else math.nan
        rows.append(row)
    return rows


def _pair_by_time(a, b, tol=1e-4):
    """Pair frames that match on time (both are on the same Praat-style grid)."""
    # Praat frame timings are deterministic given parameters; assume they line
    # up one-to-one. If lengths differ, note it and match by nearest time.
    if len(a) == len(b):
        for ai, bi in zip(a, b):
            yield ai, bi
        return
    # Mismatch: align by nearest time from smaller onto larger.
    short, long = (a, b) if len(a) < len(b) else (b, a)
    b_times = np.array([r["t"] for r in long])
    for ai in short:
        j = int(np.argmin(np.abs(b_times - ai["t"])))
        if abs(b_times[j] - ai["t"]) < tol:
            yield (ai, long[j]) if short is a else (long[j], ai)


def _summary(label, diffs):
    diffs = np.asarray([d for d in diffs if not math.isnan(d)])
    if diffs.size == 0:
        print(f"  {label:14s}: no valid frames")
        return
    abs_d = np.abs(diffs)
    print(f"  {label:14s}: n={diffs.size:4d} "
          f"mean={abs_d.mean():8.3f} "
          f"p50={np.median(abs_d):8.3f} "
          f"p95={np.percentile(abs_d, 95):8.3f} "
          f"p99={np.percentile(abs_d, 99):8.3f} "
          f"max={abs_d.max():8.3f}")


def main():
    all_rows = []
    per_file = {}

    for name in INCLUDE:
        path = FIXTURES / name
        if not path.exists():
            print(f"skip missing {name}")
            continue
        print(f"== {name}")
        praat_rows = _praat_formant(path)
        ours_rows = _praatfan_formant(path)
        print(f"   praat frames={len(praat_rows)}, ours frames={len(ours_rows)}")

        diffs = {k: [] for k in ("f1", "f2", "f3", "b1", "b2", "b3")}
        for pr, our in _pair_by_time(praat_rows, ours_rows):
            row = {
                "file": name,
                "t": pr["t"],
                "praat_f1": pr["f1"], "our_f1": our["f1"],
                "praat_f2": pr["f2"], "our_f2": our["f2"],
                "praat_f3": pr["f3"], "our_f3": our["f3"],
                "praat_b1": pr["b1"], "our_b1": our["b1"],
                "praat_b2": pr["b2"], "our_b2": our["b2"],
                "praat_b3": pr["b3"], "our_b3": our["b3"],
            }
            all_rows.append(row)
            for k in diffs:
                p = pr[k]
                o = our[k]
                if math.isnan(p) or math.isnan(o):
                    continue
                diffs[k].append(o - p)

        per_file[name] = diffs
        for k in ("f1", "f2", "f3"):
            _summary(f"{k} err Hz", diffs[k])
        for k in ("b1", "b2", "b3"):
            _summary(f"{k} err Hz", diffs[k])

    # Aggregate
    print("\n== AGGREGATE (all files)")
    agg = {k: [] for k in ("f1", "f2", "f3", "b1", "b2", "b3")}
    for d in per_file.values():
        for k, v in d.items():
            agg[k].extend(v)
    for k in ("f1", "f2", "f3"):
        _summary(f"{k} err Hz", agg[k])
    for k in ("b1", "b2", "b3"):
        _summary(f"{k} err Hz", agg[k])

    # CSV
    with open(OUT_CSV, "w", newline="") as fh:
        fieldnames = [
            "file", "t",
            "praat_f1", "our_f1", "praat_f2", "our_f2", "praat_f3", "our_f3",
            "praat_b1", "our_b1", "praat_b2", "our_b2", "praat_b3", "our_b3",
        ]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {OUT_CSV} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
