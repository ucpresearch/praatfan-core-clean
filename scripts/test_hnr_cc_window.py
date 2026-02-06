#!/usr/bin/env python3
"""
Test HNR CC window size and forward cross-correlation behavior.

Compare praatfan vs parselmouth to determine:
1. What window duration does HNR CC actually use?
2. How does "forward cross-correlation" affect the window/frame positioning?
"""

import numpy as np
import parselmouth
from parselmouth.praat import call

# Load test audio
audio_path = "tests/fixtures/one_two_three_four_five.wav"
snd_pm = parselmouth.Sound(audio_path)

print("=" * 70)
print("HNR CC: Investigating window size and forward cross-correlation")
print("=" * 70)

# Default parameters
time_step = 0.01
min_pitch = 75.0
silence_threshold = 0.1
periods_per_window = 1.0  # Praat's stated default for CC

# Get parselmouth HNR CC
hnr_pm = call(snd_pm, "To Harmonicity (cc)", time_step, min_pitch, silence_threshold, periods_per_window)

n_frames_pm = call(hnr_pm, "Get number of frames")
t1_pm = call(hnr_pm, "Get time from frame number", 1)
t_last_pm = call(hnr_pm, "Get time from frame number", n_frames_pm)

print(f"\nParselmouth HNR CC results:")
print(f"  Number of frames: {n_frames_pm}")
print(f"  First frame time (t1): {t1_pm:.6f} s")
print(f"  Last frame time: {t_last_pm:.6f} s")
print(f"  Duration: {snd_pm.duration:.6f} s")

# Calculate what window duration would give this frame count
duration = snd_pm.duration

print(f"\n--- Window duration analysis ---")

# Test different window duration hypotheses
for periods in [1.0, 2.0, 3.0, 4.5]:
    window_dur = periods / min_pitch
    # Standard formula: n_frames = floor((duration - window) / step) + 1
    expected_frames = int(np.floor((duration - window_dur) / time_step + 1e-9)) + 1
    # Centered t1
    centered_t1 = (duration - (expected_frames - 1) * time_step) / 2.0
    # Left-aligned t1 (window/2 from start)
    left_t1 = window_dur / 2.0

    match_frames = "✓" if expected_frames == n_frames_pm else "✗"
    match_t1_centered = "✓" if abs(centered_t1 - t1_pm) < 1e-6 else "✗"
    match_t1_left = "✓" if abs(left_t1 - t1_pm) < 1e-6 else "✗"

    print(f"\n{periods} periods (window = {window_dur*1000:.2f} ms):")
    print(f"  Expected frames: {expected_frames} {match_frames}")
    print(f"  Centered t1: {centered_t1:.6f} s {match_t1_centered}")
    print(f"  Left-aligned t1: {left_t1:.6f} s {match_t1_left}")

# Now test with "forward" interpretation
# Forward CC might mean: window starts at t, not centered at t
print(f"\n--- Forward cross-correlation hypotheses ---")

for periods in [1.0, 2.0, 3.0]:
    window_dur = periods / min_pitch

    # Forward: frame at t analyzes [t, t + window]
    # First valid frame: t >= 0, t + window <= duration
    # So: 0 <= t <= duration - window
    # n_frames = floor((duration - window) / step) + 1
    expected_frames_fwd = int(np.floor((duration - window_dur) / time_step + 1e-9)) + 1

    # Forward t1 options:
    # Option A: t1 = 0 (start immediately)
    t1_fwd_a = 0.0
    # Option B: t1 = half period (some offset)
    t1_fwd_b = (1.0 / min_pitch) / 2.0
    # Option C: centered in available range
    t1_fwd_c = (duration - window_dur - (expected_frames_fwd - 1) * time_step) / 2.0

    match_a = "✓" if abs(t1_fwd_a - t1_pm) < 1e-6 else f"✗ (diff={t1_fwd_a - t1_pm:.6f})"
    match_b = "✓" if abs(t1_fwd_b - t1_pm) < 1e-6 else f"✗ (diff={t1_fwd_b - t1_pm:.6f})"
    match_c = "✓" if abs(t1_fwd_c - t1_pm) < 1e-6 else f"✗ (diff={t1_fwd_c - t1_pm:.6f})"

    print(f"\n{periods} periods forward (window = {window_dur*1000:.2f} ms):")
    print(f"  Expected frames: {expected_frames_fwd} {'✓' if expected_frames_fwd == n_frames_pm else '✗'}")
    print(f"  t1 = 0: {t1_fwd_a:.6f} s {match_a}")
    print(f"  t1 = half_period: {t1_fwd_b:.6f} s {match_b}")
    print(f"  t1 = centered_fwd: {t1_fwd_c:.6f} s {match_c}")

# Test different periods_per_window values to see if parselmouth uses the parameter
print(f"\n--- Does parselmouth use periods_per_window parameter? ---")

for ppw in [1.0, 2.0, 3.0, 4.5, 6.0]:
    hnr_test = call(snd_pm, "To Harmonicity (cc)", time_step, min_pitch, silence_threshold, ppw)
    n_frames_test = call(hnr_test, "Get number of frames")
    t1_test = call(hnr_test, "Get time from frame number", 1)

    # Calculate expected window duration
    window_dur = ppw / min_pitch
    expected_frames = int(np.floor((duration - window_dur) / time_step + 1e-9)) + 1

    match = "✓" if expected_frames == n_frames_test else "✗"
    print(f"  periods_per_window={ppw}: {n_frames_test} frames (expected {expected_frames} {match}), t1={t1_test:.6f}")

# Get some actual HNR values for comparison
print(f"\n--- Sample HNR values (first 10 voiced frames) ---")
print(f"{'Frame':>6} {'Time':>10} {'HNR (dB)':>10}")
count = 0
for i in range(1, min(n_frames_pm + 1, 200)):
    t = call(hnr_pm, "Get time from frame number", i)
    hnr = call(hnr_pm, "Get value at time", t, "Cubic")
    if hnr is not None and not np.isnan(hnr) and hnr > -200:
        print(f"{i:>6} {t:>10.4f} {hnr:>10.2f}")
        count += 1
        if count >= 10:
            break
