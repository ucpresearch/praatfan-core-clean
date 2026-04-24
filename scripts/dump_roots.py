"""
Dump ALL roots (not just filtered formants) at a problematic frame.
Shows magnitude, angle, frequency, bandwidth for every pole.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from praatfan.sound import Sound as PFSound
from praatfan.formant import (
    _burg_lpc,
    _gaussian_window,
    _lpc_roots,
    _resample,
)

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
FILE = FIXTURES / "one_two_three_four_five.wav"

MAX_N_FORMANTS = 5
MAX_FORMANT = 5500.0
WINDOW = 0.025
PRE_EMPH = 50.0

TIMES = [0.311, 0.356, 0.361, 0.371]


def main():
    snd = PFSound.from_file(str(FILE))
    dur = snd.duration

    target_rate = 2 * MAX_FORMANT
    samples = _resample(snd.samples, snd.sample_rate, target_rate)
    sr = target_rate

    # Pre-emphasis
    alpha = float(np.exp(-2 * np.pi * PRE_EMPH / sr))
    x = np.zeros(len(samples))
    x[0] = samples[0]
    for i in range(1, len(samples)):
        x[i] = samples[i] - alpha * samples[i - 1]

    # Physical window
    phys = 2 * WINDOW
    wn = int(round(phys * sr))
    if wn % 2 == 0:
        wn += 1
    half = wn // 2
    win = _gaussian_window(wn)

    order = 2 * MAX_N_FORMANTS

    for t in TIMES:
        center = int(round(t * sr))
        a_start = center - half
        a_end = a_start + wn
        if a_start < 0 or a_end > len(x):
            frame = np.zeros(wn)
            ss = max(0, a_start); ee = min(len(x), a_end)
            ds = ss - a_start
            frame[ds:ds+(ee-ss)] = x[ss:ee]
        else:
            frame = x[a_start:a_end].copy()

        win_frame = frame * win
        a = _burg_lpc(win_frame, order)
        roots = _lpc_roots(a)

        print(f"\n=== t={t:.3f}  order={order}  roots={len(roots)}")
        entries = []
        for z in roots:
            r = abs(z)
            theta = np.angle(z)
            freq = theta * sr / (2 * np.pi)
            bw = -np.log(r) * sr / np.pi if r > 0 else float("inf")
            entries.append((z, r, theta, freq, bw))
        # sort by |freq|
        entries.sort(key=lambda e: abs(e[3]))
        for z, r, theta, freq, bw in entries:
            tag = ""
            if freq <= 0:
                tag += " (neg/dc — skipped: imag<=0 check)"
            if 0 < freq < 50:
                tag += " (skipped: min_freq)"
            if freq > MAX_FORMANT - 50:
                tag += " (skipped: max_freq)"
            if bw <= 0:
                tag += " (skipped: bw<=0)"
            print(f"  z=({z.real:+.4f}{z.imag:+.4f}j) |z|={r:.4f} theta={theta:+.4f} "
                  f"F={freq:+8.2f} B={bw:8.2f}{tag}")


if __name__ == "__main__":
    main()
