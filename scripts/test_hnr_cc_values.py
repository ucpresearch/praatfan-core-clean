#!/usr/bin/env python3
"""
Compare HNR CC values between praatfan and parselmouth
using the corrected window formula: (ppw + 1) / min_pitch
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import parselmouth
from parselmouth.praat import call
# Use the Python implementation directly to test it
from praatfan.sound import Sound as PraatfanSound

audio_path = "tests/fixtures/one_two_three_four_five.wav"

# Load with both
snd_pm = parselmouth.Sound(audio_path)
snd_pf = PraatfanSound.from_file(audio_path)

time_step = 0.01
min_pitch = 75.0
silence_threshold = 0.1
ppw = 1.0

# Get parselmouth HNR CC
hnr_pm = call(snd_pm, "To Harmonicity (cc)", time_step, min_pitch, silence_threshold, ppw)
n_frames_pm = call(hnr_pm, "Get number of frames")

# Get praatfan HNR CC
hnr_pf = snd_pf.to_harmonicity_cc(
    time_step=time_step,
    min_pitch=min_pitch,
    silence_threshold=silence_threshold,
    periods_per_window=ppw
)

print("=" * 70)
print("HNR CC Value Comparison")
print("=" * 70)
print(f"Parselmouth frames: {n_frames_pm}")
print(f"Praatfan frames: {hnr_pf.n_frames}")

if n_frames_pm != hnr_pf.n_frames:
    print("\n⚠️  Frame count mismatch! Comparing what we can...")

# Compare frame by frame
print(f"\n{'Frame':>6} {'t_pm':>10} {'t_pf':>10} {'HNR_pm':>10} {'HNR_pf':>10} {'diff':>10}")
print("-" * 70)

errors = []
for i in range(1, min(n_frames_pm + 1, hnr_pf.n_frames + 1, 50)):
    t_pm = call(hnr_pm, "Get time from frame number", i)
    t_pf = hnr_pf.times[i - 1]  # praatfan is 0-indexed

    hnr_val_pm = call(hnr_pm, "Get value in frame", i)
    hnr_val_pf = hnr_pf.values[i - 1]

    # Handle NaN/undefined
    if hnr_val_pm is None or (isinstance(hnr_val_pm, float) and np.isnan(hnr_val_pm)):
        hnr_val_pm = float('nan')
    if np.isnan(hnr_val_pf):
        hnr_val_pf = float('nan')

    if not np.isnan(hnr_val_pm) and not np.isnan(hnr_val_pf):
        diff = hnr_val_pf - hnr_val_pm
        errors.append(abs(diff))
        diff_str = f"{diff:>10.2f}"
    elif np.isnan(hnr_val_pm) and np.isnan(hnr_val_pf):
        diff_str = "both NaN"
    else:
        diff_str = "one NaN"

    print(f"{i:>6} {t_pm:>10.4f} {t_pf:>10.4f} {hnr_val_pm:>10.2f} {hnr_val_pf:>10.2f} {diff_str}")

if errors:
    print(f"\n--- Error Statistics (voiced frames) ---")
    print(f"Count: {len(errors)}")
    print(f"Mean error: {np.mean(errors):.4f} dB")
    print(f"Max error: {np.max(errors):.4f} dB")
    print(f"Min error: {np.min(errors):.4f} dB")
    print(f"Std error: {np.std(errors):.4f} dB")
