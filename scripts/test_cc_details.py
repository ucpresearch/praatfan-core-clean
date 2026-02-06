#!/usr/bin/env python3
"""
Investigate the three potential sources of CC discrepancy:
1. Parabolic interpolation for sub-sample precision
2. Different boundary handling
3. Slight differences in frame centering
"""

import sys
sys.path.insert(0, '/home/urielc/local/decfiles/private/Dev/git/praatfan-core-clean/src')

import numpy as np
import parselmouth
from parselmouth.praat import call
from praatfan.sound import Sound

# Load test sound
sound_path = "/home/urielc/local/decfiles/private/Dev/git/praatfan-core-clean/tests/fixtures/one_two_three_four_five.wav"
pm_sound = parselmouth.Sound(sound_path)
my_sound = Sound.from_file(sound_path)

sample_rate = my_sound.sample_rate
samples = my_sound.samples
duration = my_sound.duration

# Parameters
min_pitch = 75.0
max_pitch = 600.0
time_step = 0.01

# Get parselmouth Pitch CC
pm_pitch = call(pm_sound, "To Pitch (cc)", time_step, min_pitch, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, max_pitch)

pm_n_frames = call(pm_pitch, "Get number of frames")
print(f"=== Parselmouth Pitch CC ===")
print(f"Frames: {pm_n_frames}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {duration:.6f}s")
print()

# Get Praat frame times and values
# For CC, we can't get strength directly, but we can get F0 values
pm_times = []
pm_values = []
for i in range(1, pm_n_frames + 1):
    t = call(pm_pitch, "Get time from frame number", i)
    f0 = call(pm_pitch, "Get value in frame", i, "Hertz")
    pm_times.append(t)
    pm_values.append(f0 if f0 else 0.0)

# To get strength, we need to use Harmonicity CC
# The strength for CC comes from the cross-correlation value
pm_harm = call(pm_sound, "To Harmonicity (cc)", time_step, min_pitch, 0.1, 1.0)
pm_harm_n = call(pm_harm, "Get number of frames")
pm_harm_times = []
pm_harm_values = []
for i in range(1, pm_harm_n + 1):
    t = call(pm_harm, "Get time from frame number", i)
    v = call(pm_harm, "Get value in frame", i)
    pm_harm_times.append(t)
    pm_harm_values.append(v if v else -200.0)

print(f"Harmonicity CC frames: {pm_harm_n}")
print()

# Convert HNR back to correlation r
# HNR = 10 * log10(r / (1 - r))
# 10^(HNR/10) = r / (1 - r)
# r = 10^(HNR/10) / (1 + 10^(HNR/10))
def hnr_to_r(hnr):
    if hnr < -100:
        return 0.0
    ratio = 10 ** (hnr / 10)
    return ratio / (1 + ratio)

# ============================================================
# INVESTIGATION 1: Frame centering
# ============================================================
print("=" * 70)
print("INVESTIGATION 1: Frame Centering")
print("=" * 70)

# CC window duration = 2 / min_pitch
window_duration = 2.0 / min_pitch
window_samples = int(round(window_duration * sample_rate))
if window_samples % 2 == 0:
    window_samples += 1
half_window = window_samples // 2

print(f"Window duration: {window_duration:.6f}s")
print(f"Window samples: {window_samples}")
print(f"Half window: {half_window}")
print()

# Test the frame timing formula
# Centered: t1 = (duration - (n-1) * time_step) / 2
n_frames_centered = int(np.floor((duration - window_duration) / time_step + 1e-9)) + 1
t1_centered = (duration - (n_frames_centered - 1) * time_step) / 2.0

print(f"Centered formula:")
print(f"  n_frames = {n_frames_centered}")
print(f"  t1 = {t1_centered:.6f}s")
print(f"  Praat t1 = {pm_times[0]:.6f}s")
print(f"  Difference: {abs(t1_centered - pm_times[0]):.9f}s")
print()

# Check if frame times match exactly
print("Frame time comparison (first 5 frames):")
for i in range(min(5, len(pm_times))):
    my_t = t1_centered + i * time_step
    pm_t = pm_times[i]
    diff = abs(my_t - pm_t)
    print(f"  Frame {i+1}: my={my_t:.6f}s, pm={pm_t:.6f}s, diff={diff:.9f}s")
print()

# ============================================================
# INVESTIGATION 2: Boundary handling
# ============================================================
print("=" * 70)
print("INVESTIGATION 2: Boundary Handling")
print("=" * 70)

# Find voiced frames with known HNR
voiced_frames = []
for i in range(len(pm_times)):
    if pm_values[i] > 0:
        # Find corresponding HNR frame
        t = pm_times[i]
        # Find closest harmonicity frame
        hnr = -200.0
        for j, ht in enumerate(pm_harm_times):
            if abs(ht - t) < 0.001:  # Within 1ms
                hnr = pm_harm_values[j]
                break
        r = hnr_to_r(hnr)
        voiced_frames.append((i, t, pm_values[i], r, hnr))

print(f"Found {len(voiced_frames)} voiced frames with HNR data")
print()

if voiced_frames:
    # Use a frame that's well within the signal (not near edges)
    mid_voiced = voiced_frames[len(voiced_frames) // 2]
    test_idx, test_t, test_f0, test_r, test_hnr = mid_voiced

    print(f"Testing frame {test_idx+1} at t={test_t:.6f}s")
    print(f"  Expected F0: {test_f0:.2f} Hz")
    print(f"  Expected HNR: {test_hnr:.2f} dB")
    print(f"  Expected r (from HNR): {test_r:.6f}")
    print()

    # Extract frame - test different boundary approaches
    center_sample = int(round(test_t * sample_rate))

    print(f"Center sample: {center_sample}")
    print(f"Start sample: {center_sample - half_window}")
    print(f"End sample: {center_sample - half_window + window_samples}")
    print(f"Total samples in file: {len(samples)}")
    print()

    # Method A: round() for center
    start_a = int(round(test_t * sample_rate)) - half_window
    frame_a = samples[start_a:start_a + window_samples].copy()

    # Method B: floor() for center
    start_b = int(np.floor(test_t * sample_rate)) - half_window
    frame_b = samples[start_b:start_b + window_samples].copy()

    # Method C: Different half_window calculation
    half_window_alt = (window_samples - 1) // 2
    start_c = int(round(test_t * sample_rate)) - half_window_alt
    frame_c = samples[start_c:start_c + window_samples].copy()

    print(f"Boundary methods:")
    print(f"  A (round, half={half_window}): start={start_a}")
    print(f"  B (floor, half={half_window}): start={start_b}")
    print(f"  C (round, half={half_window_alt}): start={start_c}")
    print()

# ============================================================
# INVESTIGATION 3: Parabolic interpolation
# ============================================================
print("=" * 70)
print("INVESTIGATION 3: Parabolic Interpolation")
print("=" * 70)

if voiced_frames:
    # Use the test frame from Investigation 2
    frame = frame_a  # Use method A

    # Forward cross-correlation without interpolation
    min_lag = int(np.ceil(sample_rate / max_pitch))
    max_lag = int(np.floor(sample_rate / min_pitch))

    # Compute FCC for all lags
    fcc = np.zeros(max_lag + 1)
    for lag in range(min_lag, min(max_lag + 1, len(frame) // 2)):
        first_period = frame[:lag]
        second_period = frame[lag:2*lag]
        if len(second_period) == lag:
            corr = np.sum(first_period * second_period)
            e1 = np.sum(first_period * first_period)
            e2 = np.sum(second_period * second_period)
            if e1 > 0 and e2 > 0:
                fcc[lag] = corr / np.sqrt(e1 * e2)

    # Find the best lag (peak)
    best_lag = min_lag
    for lag in range(min_lag + 1, min(max_lag, len(fcc) - 1)):
        if fcc[lag] > fcc[lag-1] and fcc[lag] > fcc[lag+1]:
            if fcc[lag] > fcc[best_lag]:
                best_lag = lag

    print(f"Best lag (integer): {best_lag}")
    print(f"  Frequency: {sample_rate / best_lag:.4f} Hz")
    print(f"  Strength (no interp): {fcc[best_lag]:.6f}")
    print()

    # Parabolic interpolation
    if best_lag > min_lag and best_lag < len(fcc) - 1:
        r_prev = fcc[best_lag - 1]
        r_curr = fcc[best_lag]
        r_next = fcc[best_lag + 1]

        print(f"Values around peak:")
        print(f"  r[{best_lag-1}] = {r_prev:.6f}")
        print(f"  r[{best_lag}] = {r_curr:.6f}")
        print(f"  r[{best_lag+1}] = {r_next:.6f}")
        print()

        denom = r_prev - 2*r_curr + r_next
        if abs(denom) > 1e-10:
            # Parabolic interpolation for lag
            delta = 0.5 * (r_prev - r_next) / denom
            refined_lag = best_lag + delta

            # Parabolic interpolation for strength
            refined_strength = r_curr - 0.25 * (r_prev - r_next) * delta

            print(f"Parabolic interpolation:")
            print(f"  Delta: {delta:.6f}")
            print(f"  Refined lag: {refined_lag:.6f}")
            print(f"  Refined frequency: {sample_rate / refined_lag:.4f} Hz")
            print(f"  Refined strength: {refined_strength:.6f}")
            print()

            print(f"Comparison with Praat:")
            print(f"  Praat F0: {test_f0:.4f} Hz")
            print(f"  My F0 (no interp): {sample_rate / best_lag:.4f} Hz")
            print(f"  My F0 (interp): {sample_rate / refined_lag:.4f} Hz")
            print(f"  F0 error (no interp): {abs(sample_rate / best_lag - test_f0):.4f} Hz")
            print(f"  F0 error (interp): {abs(sample_rate / refined_lag - test_f0):.4f} Hz")
            print()
            print(f"  Praat r (from HNR): {test_r:.6f}")
            print(f"  My r (no interp): {fcc[best_lag]:.6f}")
            print(f"  My r (interp): {refined_strength:.6f}")
            print(f"  r error (no interp): {abs(fcc[best_lag] - test_r):.6f}")
            print(f"  r error (interp): {abs(refined_strength - test_r):.6f}")

# ============================================================
# COMPREHENSIVE TEST: Multiple frames
# ============================================================
print()
print("=" * 70)
print("COMPREHENSIVE TEST: Multiple Voiced Frames")
print("=" * 70)

errors_no_interp = []
errors_with_interp = []
f0_errors_no_interp = []
f0_errors_with_interp = []

for idx, t, pm_f0, pm_r, pm_hnr in voiced_frames[:20]:  # Test first 20 voiced frames
    if pm_r < 0.5:  # Skip low correlation frames
        continue

    # Extract frame
    center_sample = int(round(t * sample_rate))
    start_sample = center_sample - half_window

    if start_sample < 0 or start_sample + window_samples > len(samples):
        continue  # Skip boundary frames for now

    frame = samples[start_sample:start_sample + window_samples].copy()

    # Compute FCC
    fcc = np.zeros(max_lag + 1)
    for lag in range(min_lag, min(max_lag + 1, len(frame) // 2)):
        first_period = frame[:lag]
        second_period = frame[lag:2*lag]
        if len(second_period) == lag:
            corr = np.sum(first_period * second_period)
            e1 = np.sum(first_period * first_period)
            e2 = np.sum(second_period * second_period)
            if e1 > 0 and e2 > 0:
                fcc[lag] = corr / np.sqrt(e1 * e2)

    # Find peak
    best_lag = min_lag
    for lag in range(min_lag + 1, min(max_lag, len(fcc) - 1)):
        if fcc[lag] > fcc[lag-1] and fcc[lag] > fcc[lag+1]:
            if fcc[lag] > fcc[best_lag]:
                best_lag = lag

    if best_lag <= min_lag or best_lag >= len(fcc) - 1:
        continue

    # No interpolation
    my_f0_no_interp = sample_rate / best_lag
    my_r_no_interp = fcc[best_lag]

    # With interpolation
    r_prev = fcc[best_lag - 1]
    r_curr = fcc[best_lag]
    r_next = fcc[best_lag + 1]

    denom = r_prev - 2*r_curr + r_next
    if abs(denom) > 1e-10:
        delta = 0.5 * (r_prev - r_next) / denom
        if abs(delta) < 1:
            refined_lag = best_lag + delta
            refined_r = r_curr - 0.25 * (r_prev - r_next) * delta
            my_f0_interp = sample_rate / refined_lag
            my_r_interp = refined_r
        else:
            my_f0_interp = my_f0_no_interp
            my_r_interp = my_r_no_interp
    else:
        my_f0_interp = my_f0_no_interp
        my_r_interp = my_r_no_interp

    errors_no_interp.append(abs(my_r_no_interp - pm_r))
    errors_with_interp.append(abs(my_r_interp - pm_r))
    f0_errors_no_interp.append(abs(my_f0_no_interp - pm_f0))
    f0_errors_with_interp.append(abs(my_f0_interp - pm_f0))

print(f"Tested {len(errors_no_interp)} voiced frames")
print()

if errors_no_interp:
    print("Correlation r errors:")
    print(f"  No interpolation: mean={np.mean(errors_no_interp):.6f}, max={np.max(errors_no_interp):.6f}")
    print(f"  With interpolation: mean={np.mean(errors_with_interp):.6f}, max={np.max(errors_with_interp):.6f}")
    print()
    print("F0 errors (Hz):")
    print(f"  No interpolation: mean={np.mean(f0_errors_no_interp):.4f}, max={np.max(f0_errors_no_interp):.4f}")
    print(f"  With interpolation: mean={np.mean(f0_errors_with_interp):.4f}, max={np.max(f0_errors_with_interp):.4f}")

# ============================================================
# SHOW DETAILED FRAME-BY-FRAME COMPARISON
# ============================================================
print()
print("=" * 70)
print("DETAILED FRAME-BY-FRAME (first 10 high-correlation frames)")
print("=" * 70)

count = 0
for idx, t, pm_f0, pm_r, pm_hnr in voiced_frames:
    if pm_r < 0.9:  # Only high correlation frames
        continue
    if count >= 10:
        break

    # Extract frame
    center_sample = int(round(t * sample_rate))
    start_sample = center_sample - half_window

    if start_sample < 0 or start_sample + window_samples > len(samples):
        continue

    frame = samples[start_sample:start_sample + window_samples].copy()

    # Compute FCC
    fcc = np.zeros(max_lag + 1)
    for lag in range(min_lag, min(max_lag + 1, len(frame) // 2)):
        first_period = frame[:lag]
        second_period = frame[lag:2*lag]
        if len(second_period) == lag:
            corr = np.sum(first_period * second_period)
            e1 = np.sum(first_period * first_period)
            e2 = np.sum(second_period * second_period)
            if e1 > 0 and e2 > 0:
                fcc[lag] = corr / np.sqrt(e1 * e2)

    # Find peak
    best_lag = min_lag
    for lag in range(min_lag + 1, min(max_lag, len(fcc) - 1)):
        if fcc[lag] > fcc[lag-1] and fcc[lag] > fcc[lag+1]:
            if fcc[lag] > fcc[best_lag]:
                best_lag = lag

    if best_lag <= min_lag or best_lag >= len(fcc) - 1:
        continue

    # With interpolation
    r_prev = fcc[best_lag - 1]
    r_curr = fcc[best_lag]
    r_next = fcc[best_lag + 1]

    denom = r_prev - 2*r_curr + r_next
    if abs(denom) > 1e-10:
        delta = 0.5 * (r_prev - r_next) / denom
        refined_lag = best_lag + delta
        refined_r = r_curr - 0.25 * (r_prev - r_next) * delta
    else:
        refined_lag = best_lag
        refined_r = r_curr

    my_f0 = sample_rate / refined_lag

    print(f"Frame {idx+1} t={t:.4f}s:")
    print(f"  F0: pm={pm_f0:.2f} Hz, my={my_f0:.2f} Hz, err={abs(my_f0-pm_f0):.2f} Hz")
    print(f"  r:  pm={pm_r:.6f}, my={refined_r:.6f}, err={abs(refined_r-pm_r):.6f}")
    print()
    count += 1
