#!/usr/bin/env python3
"""
Compare Pitch CC strengths between praatfan and parselmouth.
HNR is derived from these strengths via: HNR = 10 * log10(r / (1-r))
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import parselmouth
from parselmouth.praat import call
from praatfan.sound import Sound as PraatfanSound

audio_path = "tests/fixtures/one_two_three_four_five.wav"

# Load with both
snd_pm = parselmouth.Sound(audio_path)
snd_pf = PraatfanSound.from_file(audio_path)

# Parameters matching HNR CC defaults
time_step = 0.01
pitch_floor = 75.0
pitch_ceiling = 600.0

# Get Pitch CC from parselmouth (hidden command)
# To Pitch (cc): time_step, floor, max_candidates, very_accurate, silence_threshold,
#                voicing_threshold, octave_cost, octave_jump_cost, vuv_cost, ceiling
pitch_pm = call(snd_pm, "To Pitch (cc)", time_step, pitch_floor, 15, False,
                0.03, 0.45, 0.01, 0.35, 0.14, pitch_ceiling)

n_frames_pm = call(pitch_pm, "Get number of frames")
print(f"Parselmouth Pitch CC: {n_frames_pm} frames")

# Get Pitch CC from praatfan
pitch_pf = snd_pf.to_pitch(
    time_step=time_step,
    pitch_floor=pitch_floor,
    pitch_ceiling=pitch_ceiling,
    method="cc"
)
print(f"Praatfan Pitch CC: {pitch_pf.n_frames} frames")

# Get arrays from parselmouth
pm_array = pitch_pm.selected_array

# Compare frame by frame
print(f"\n{'Frame':>6} {'t_pm':>8} {'t_pf':>8} {'F0_pm':>8} {'F0_pf':>8} {'str_pm':>8} {'str_pf':>8} {'str_diff':>10}")
print("-" * 85)

strength_errors = []
for i in range(1, min(n_frames_pm + 1, pitch_pf.n_frames + 1, 50)):
    t_pm = call(pitch_pm, "Get time from frame number", i)
    f0_pm = pm_array['frequency'][i-1]
    str_pm = pm_array['strength'][i-1]

    # Praatfan is 0-indexed
    t_pf = pitch_pf.times()[i - 1]
    f0_pf = pitch_pf.values()[i - 1]
    str_pf = pitch_pf.strengths()[i - 1]

    # Format output
    f0_pm_str = f"{f0_pm:.1f}" if f0_pm > 0 else "unvoiced"
    f0_pf_str = f"{f0_pf:.1f}" if not np.isnan(f0_pf) else "unvoiced"
    str_pm_str = f"{str_pm:.4f}"
    str_pf_str = f"{str_pf:.4f}" if not np.isnan(str_pf) else "N/A"

    if not np.isnan(str_pf):
        diff = str_pf - str_pm
        strength_errors.append(abs(diff))
        diff_str = f"{diff:+.4f}"
    else:
        diff_str = "N/A"

    print(f"{i:>6} {t_pm:>8.4f} {t_pf:>8.4f} {f0_pm_str:>8} {f0_pf_str:>8} {str_pm_str:>8} {str_pf_str:>8} {diff_str:>10}")

if strength_errors:
    print(f"\n--- Strength Error Statistics ---")
    print(f"Count: {len(strength_errors)}")
    print(f"Mean: {np.mean(strength_errors):.6f}")
    print(f"Max: {np.max(strength_errors):.6f}")
    print(f"Std: {np.std(strength_errors):.6f}")

# Now let's manually compute what HNR should be from these strengths
print("\n\n--- HNR derived from Pitch CC strengths ---")
print(f"{'Frame':>6} {'r_pm':>8} {'r_pf':>8} {'HNR_pm':>10} {'HNR_pf':>10} {'diff':>10}")
print("-" * 60)

for i in range(1, min(n_frames_pm + 1, pitch_pf.n_frames + 1, 50)):
    str_pm = pm_array['strength'][i-1]
    str_pf = pitch_pf.strengths()[i - 1]

    # Compute HNR from strength: HNR = 10 * log10(r / (1-r))
    if str_pm > 0 and str_pm < 1:
        hnr_pm = 10 * np.log10(str_pm / (1 - str_pm))
        hnr_pm = max(-200, min(100, hnr_pm))  # Clamp
    else:
        hnr_pm = -200

    if not np.isnan(str_pf) and str_pf > 0 and str_pf < 1:
        hnr_pf = 10 * np.log10(str_pf / (1 - str_pf))
        hnr_pf = max(-200, min(100, hnr_pf))
    else:
        hnr_pf = -200

    diff = hnr_pf - hnr_pm if abs(hnr_pm) < 199 and abs(hnr_pf) < 199 else float('nan')

    str_pm_str = f"{str_pm:.4f}"
    str_pf_str = f"{str_pf:.4f}" if not np.isnan(str_pf) else "N/A"
    diff_str = f"{diff:+.2f}" if not np.isnan(diff) else "N/A"

    print(f"{i:>6} {str_pm_str:>8} {str_pf_str:>8} {hnr_pm:>10.2f} {hnr_pf:>10.2f} {diff_str:>10}")
