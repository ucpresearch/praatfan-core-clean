"""
Triage: where in the clip do the large F1/F2/F3 errors occur?

Hypothesis A: errors concentrate at boundary frames (first/last ~window-lengths).
Hypothesis B: errors are uniform across the clip (suggests algorithm, not boundary).
Hypothesis C: errors concentrate in low-energy frames (silence / inter-word).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import parselmouth

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "scripts" / "baseline_burg_parity.csv"
FIXTURES = REPO / "tests" / "fixtures"


def _energy_at(snd_path, times, window=0.025):
    s = parselmouth.Sound(str(snd_path))
    samples = s.values[0]  # (n_ch, n_samples) → mono
    sr = s.sampling_frequency
    energies = []
    half = int(round(window * sr))
    for t in times:
        c = int(round(t * sr))
        a = max(0, c - half)
        b = min(len(samples), c + half)
        seg = samples[a:b]
        if len(seg) == 0:
            energies.append(0.0)
        else:
            energies.append(float(np.sqrt(np.mean(seg ** 2))))
    return np.array(energies)


def main():
    rows = []
    with open(CSV) as fh:
        r = csv.DictReader(fh)
        for row in r:
            rows.append(row)

    # Group by file
    by_file = {}
    for row in rows:
        by_file.setdefault(row["file"], []).append(row)

    for fname, file_rows in by_file.items():
        print(f"\n== {fname}")
        snd = parselmouth.Sound(str(FIXTURES / fname))
        dur = snd.duration
        times = np.array([float(r["t"]) for r in file_rows])
        errs = {k: np.array([
            abs(float(r[f"our_{k}"]) - float(r[f"praat_{k}"]))
            if r[f"our_{k}"] not in ("", "nan") and r[f"praat_{k}"] not in ("", "nan")
            else math.nan
            for r in file_rows
        ]) for k in ("f1", "f2", "f3", "b1", "b2", "b3")}

        energies = _energy_at(FIXTURES / fname, times)

        # Split by region
        edge_thresh = 0.05  # first/last 50 ms
        is_edge = (times < edge_thresh) | (times > dur - edge_thresh)
        is_low_energy = energies < np.percentile(energies, 20)

        for k in ("f1", "f2", "f3"):
            e = errs[k]
            valid = ~np.isnan(e)
            if not valid.any():
                continue
            edge_valid = valid & is_edge
            interior_valid = valid & ~is_edge
            low_valid = valid & is_low_energy & ~is_edge
            hi_valid = valid & ~is_low_energy & ~is_edge

            def _s(mask):
                return (mask.sum(),
                        float(np.nanmean(e[mask])) if mask.any() else math.nan,
                        float(np.nanmax(e[mask])) if mask.any() else math.nan)

            n_e, mean_e, max_e = _s(edge_valid)
            n_i, mean_i, max_i = _s(interior_valid)
            n_l, mean_l, max_l = _s(low_valid)
            n_h, mean_h, max_h = _s(hi_valid)
            print(f"  {k}: edge n={n_e:3d} mean={mean_e:8.2f} max={max_e:8.2f} | "
                  f"interior n={n_i:3d} mean={mean_i:8.2f} max={max_i:8.2f} | "
                  f"low-e n={n_l:3d} mean={mean_l:7.2f} max={max_l:7.2f} | "
                  f"hi-e n={n_h:3d} mean={mean_h:7.2f} max={max_h:7.2f}")

        # Worst 5 frames
        agg = np.nanmax(np.stack([errs["f1"], errs["f2"], errs["f3"]]), axis=0)
        worst_idx = np.argsort(-np.where(np.isnan(agg), -1, agg))[:5]
        print("  worst 5 frames (max over F1/F2/F3 error):")
        for i in worst_idx:
            if np.isnan(agg[i]):
                continue
            print(f"    t={times[i]:.3f} energy={energies[i]:.5f} "
                  f"F1={errs['f1'][i]:7.2f} F2={errs['f2'][i]:7.2f} F3={errs['f3'][i]:7.2f}")


if __name__ == "__main__":
    main()
