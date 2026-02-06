#!/usr/bin/env python3
"""
Test computing HNR CC by deriving from Pitch CC (like HNR AC derives from Pitch AC).
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import parselmouth
from parselmouth.praat import call
from praatfan.sound import Sound as PraatfanSound
from praatfan.pitch import sound_to_pitch
from praatfan.harmonicity import strength_to_hnr

audio_path = "tests/fixtures/one_two_three_four_five.wav"

snd_pm = parselmouth.Sound(audio_path)
snd_pf = PraatfanSound.from_file(audio_path)

time_step = 0.01
min_pitch = 75.0
silence_threshold = 0.1
ppw = 1.0

# Get parselmouth HNR CC
hnr_pm = call(snd_pm, "To Harmonicity (cc)", time_step, min_pitch, silence_threshold, ppw)
n_frames_pm = call(hnr_pm, "Get number of frames")

# Compute Pitch CC with praatfan
# Key: use (ppw + 1) periods_per_window
pitch_pf = sound_to_pitch(
    snd_pf,
    time_step=time_step,
    pitch_floor=min_pitch,
    pitch_ceiling=600.0,
    method="cc",
    periods_per_window=ppw + 1.0,  # Add 1 for "forward" CC
    apply_octave_cost=False  # Get raw correlation strength
)

print(f"Parselmouth HNR CC: {n_frames_pm} frames")
print(f"Praatfan Pitch CC: {pitch_pf.n_frames} frames")

# Compute HNR from Pitch CC
print(f"\n{'Frame':>5} {'t_pm':>8} {'t_pf':>8} {'voiced':>7} {'strength':>10} {'HNR_calc':>10} {'HNR_pm':>10}")
print("-" * 75)

errors = []
for i in range(min(n_frames_pm, pitch_pf.n_frames)):
    t_pm = call(hnr_pm, "Get time from frame number", i + 1)
    t_pf = pitch_pf.times()[i]

    frame = pitch_pf.frames[i]
    voiced = frame.voiced
    strength = frame.strength

    # Compute HNR from strength
    if voiced:
        # Need to get the actual correlation strength, not the unvoiced penalty
        # For CC, check if the selected candidate is voiced (freq > 0)
        hnr_calc = strength_to_hnr(strength) if 0 < strength < 1 else -200.0
    else:
        hnr_calc = -200.0

    # Get parselmouth HNR
    hnr_pm_val = call(hnr_pm, "Get value in frame", i + 1)
    if hnr_pm_val is None or np.isnan(hnr_pm_val):
        hnr_pm_val = -200.0

    diff = hnr_calc - hnr_pm_val if abs(hnr_calc) < 199 and abs(hnr_pm_val) < 199 else float('nan')
    if not np.isnan(diff):
        errors.append(abs(diff))

    voiced_str = "YES" if voiced else "no"
    diff_str = f"{diff:+.2f}" if not np.isnan(diff) else "N/A"

    print(f"{i+1:>5} {t_pm:>8.4f} {t_pf:>8.4f} {voiced_str:>7} {strength:>10.4f} {hnr_calc:>10.2f} {hnr_pm_val:>10.2f} {diff_str}")

    if i >= 49:
        break

if errors:
    print(f"\n--- Error Statistics (both voiced) ---")
    print(f"Count: {len(errors)}")
    print(f"Mean: {np.mean(errors):.4f} dB")
    print(f"Max: {np.max(errors):.4f} dB")
