#!/usr/bin/env python3
"""
Compare Pitch AC vs CC strengths between praatfan and parselmouth.
Investigate where the strength differences come from.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import parselmouth
from parselmouth.praat import call
from praatfan.sound import Sound as PraatfanSound
from praatfan.pitch import sound_to_pitch

audio_path = "tests/fixtures/one_two_three_four_five.wav"

snd_pm = parselmouth.Sound(audio_path)
snd_pf = PraatfanSound.from_file(audio_path)

time_step = 0.01
min_pitch = 75.0

print("=" * 80)
print("Comparing Pitch AC strengths")
print("=" * 80)

# Pitch AC - parselmouth
pitch_ac_pm = call(snd_pm, "To Pitch (ac)", time_step, min_pitch, 15, False,
                   0.03, 0.45, 0.01, 0.35, 0.14, 600)
pm_ac_arr = pitch_ac_pm.selected_array

# Pitch AC - praatfan (with apply_octave_cost=False to get raw strength)
pitch_ac_pf = sound_to_pitch(snd_pf, time_step=time_step, pitch_floor=min_pitch,
                              pitch_ceiling=600.0, method="ac", apply_octave_cost=False)

print(f"\n{'Frame':>5} {'F0_pm':>8} {'F0_pf':>8} {'str_pm':>8} {'str_pf':>8} {'str_diff':>10}")
print("-" * 55)

ac_errors = []
for i in range(min(50, len(pm_ac_arr), pitch_ac_pf.n_frames)):
    f0_pm = pm_ac_arr['frequency'][i]
    str_pm = pm_ac_arr['strength'][i]

    f0_pf = pitch_ac_pf.values()[i]
    str_pf = pitch_ac_pf.strengths()[i]

    f0_pm_str = f"{f0_pm:.1f}" if f0_pm > 0 else "unv"
    f0_pf_str = f"{f0_pf:.1f}" if f0_pf > 0 else "unv"

    # Only compare when both are voiced
    if f0_pm > 0 and f0_pf > 0 and str_pm > 0 and str_pm < 1 and str_pf > 0 and str_pf < 1:
        diff = str_pf - str_pm
        ac_errors.append(abs(diff))
        diff_str = f"{diff:+.4f}"
    else:
        diff_str = "N/A"

    print(f"{i+1:>5} {f0_pm_str:>8} {f0_pf_str:>8} {str_pm:>8.4f} {str_pf:>8.4f} {diff_str:>10}")

if ac_errors:
    print(f"\n--- AC Strength Error (both voiced, valid strength) ---")
    print(f"Count: {len(ac_errors)}")
    print(f"Mean: {np.mean(ac_errors):.6f}")
    print(f"95th percentile: {np.percentile(ac_errors, 95):.6f}")
    print(f"Max: {np.max(ac_errors):.6f}")

print("\n" + "=" * 80)
print("Comparing Pitch CC strengths")
print("=" * 80)

# Pitch CC - parselmouth
pitch_cc_pm = call(snd_pm, "To Pitch (cc)", time_step, min_pitch, 15, False,
                   0.03, 0.45, 0.01, 0.35, 0.14, 600)
pm_cc_arr = pitch_cc_pm.selected_array

# Pitch CC - praatfan
pitch_cc_pf = sound_to_pitch(snd_pf, time_step=time_step, pitch_floor=min_pitch,
                              pitch_ceiling=600.0, method="cc", apply_octave_cost=False)

print(f"\n{'Frame':>5} {'F0_pm':>8} {'F0_pf':>8} {'str_pm':>8} {'str_pf':>8} {'str_diff':>10}")
print("-" * 55)

cc_errors = []
for i in range(min(50, len(pm_cc_arr), pitch_cc_pf.n_frames)):
    f0_pm = pm_cc_arr['frequency'][i]
    str_pm = pm_cc_arr['strength'][i]

    f0_pf = pitch_cc_pf.values()[i]
    str_pf = pitch_cc_pf.strengths()[i]

    f0_pm_str = f"{f0_pm:.1f}" if f0_pm > 0 else "unv"
    f0_pf_str = f"{f0_pf:.1f}" if f0_pf > 0 else "unv"

    # Only compare when both are voiced
    if f0_pm > 0 and f0_pf > 0 and str_pm > 0 and str_pm < 1 and str_pf > 0 and str_pf < 1:
        diff = str_pf - str_pm
        cc_errors.append(abs(diff))
        diff_str = f"{diff:+.4f}"
    else:
        diff_str = "N/A"

    print(f"{i+1:>5} {f0_pm_str:>8} {f0_pf_str:>8} {str_pm:>8.4f} {str_pf:>8.4f} {diff_str:>10}")

if cc_errors:
    print(f"\n--- CC Strength Error (both voiced, valid strength) ---")
    print(f"Count: {len(cc_errors)}")
    print(f"Mean: {np.mean(cc_errors):.6f}")
    print(f"95th percentile: {np.percentile(cc_errors, 95):.6f}")
    print(f"Max: {np.max(cc_errors):.6f}")

print("\n" + "=" * 80)
print("Summary: Which method has better strength match?")
print("=" * 80)
if ac_errors and cc_errors:
    print(f"AC 95th percentile strength error: {np.percentile(ac_errors, 95):.6f}")
    print(f"CC 95th percentile strength error: {np.percentile(cc_errors, 95):.6f}")
