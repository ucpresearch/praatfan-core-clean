#!/usr/bin/env python3
"""
Investigate the systematic -1.72 Hz offset in CC F0 values.
"""

import sys
sys.path.insert(0, '/home/urielc/local/decfiles/private/Dev/git/praatfan-core-clean/src')

import numpy as np
import parselmouth
from parselmouth.praat import call
from praatfan.sound import Sound

sound_path = "/home/urielc/local/decfiles/private/Dev/git/praatfan-core-clean/tests/fixtures/one_two_three_four_five.wav"
pm_sound = parselmouth.Sound(sound_path)
my_sound = Sound.from_file(sound_path)

sample_rate = my_sound.sample_rate
samples = my_sound.samples

min_pitch = 75.0
max_pitch = 600.0
time_step = 0.01

window_duration = 2.0 / min_pitch
window_samples = int(round(window_duration * sample_rate))
if window_samples % 2 == 0:
    window_samples += 1
half_window = window_samples // 2

min_lag = int(np.ceil(sample_rate / max_pitch))
max_lag = int(np.floor(sample_rate / min_pitch))

pm_pitch = call(pm_sound, "To Pitch (cc)", time_step, min_pitch, 15, "no",
                0.03, 0.45, 0.01, 0.35, 0.14, max_pitch)

def compute_full_xcorr(frame, min_lag, max_lag):
    """Full frame cross-correlation at each lag."""
    fcc = np.zeros(max_lag + 1)
    n = len(frame)
    for lag in range(min_lag, min(max_lag + 1, n)):
        x1 = frame[:n-lag]
        x2 = frame[lag:]
        corr = np.sum(x1 * x2)
        e1 = np.sum(x1 * x1)
        e2 = np.sum(x2 * x2)
        if e1 > 0 and e2 > 0:
            fcc[lag] = corr / np.sqrt(e1 * e2)
    return fcc

print("Investigating systematic offset in CC F0")
print("=" * 70)

# Look at a specific frame with ~2 Hz error
test_t = 0.2560  # Error was 1.99 Hz
pm_f0 = call(pm_pitch, "Get value at time", test_t, "Hertz", "linear")

print(f"\nTest frame at t={test_t}s")
print(f"Praat F0: {pm_f0:.4f} Hz")

center = int(round(test_t * sample_rate))
start = center - half_window
frame = samples[start:start + window_samples].copy()

fcc = compute_full_xcorr(frame, min_lag, max_lag)

# Expected lag from Praat F0
expected_lag = sample_rate / pm_f0
print(f"Expected lag: {expected_lag:.2f} samples")

# Find the peak
best_lag = min_lag
for lag in range(min_lag + 1, min(max_lag, len(fcc) - 1)):
    if fcc[lag] > fcc[lag-1] and fcc[lag] > fcc[lag+1]:
        if fcc[lag] > fcc[best_lag]:
            best_lag = lag

print(f"\nBest integer lag: {best_lag}")
print(f"Frequency at integer lag: {sample_rate / best_lag:.4f} Hz")

# Show FCC values around the expected lag
int_expected = int(round(expected_lag))
print(f"\nFCC values around expected lag ({int_expected}):")
for lag in range(int_expected - 3, int_expected + 4):
    if lag >= min_lag and lag < len(fcc):
        freq = sample_rate / lag
        marker = " <-- expected" if lag == int_expected else (" <-- best" if lag == best_lag else "")
        print(f"  lag {lag}: r={fcc[lag]:.6f}, f={freq:.2f} Hz{marker}")

# Try different interpolation methods
print("\nPARABOLIC INTERPOLATION:")
r_prev = fcc[best_lag - 1]
r_curr = fcc[best_lag]
r_next = fcc[best_lag + 1]

print(f"  r[{best_lag-1}] = {r_prev:.6f}")
print(f"  r[{best_lag}] = {r_curr:.6f}")
print(f"  r[{best_lag+1}] = {r_next:.6f}")

# Standard parabolic
denom = r_prev - 2*r_curr + r_next
if abs(denom) > 1e-10:
    delta = 0.5 * (r_prev - r_next) / denom
    refined_lag = best_lag + delta
    refined_f0 = sample_rate / refined_lag
    print(f"\n  Standard parabolic:")
    print(f"    delta = {delta:.6f}")
    print(f"    refined_lag = {refined_lag:.4f}")
    print(f"    F0 = {refined_f0:.4f} Hz")
    print(f"    Error = {abs(refined_f0 - pm_f0):.4f} Hz")

# What if we interpolate in frequency domain instead of lag domain?
f_prev = sample_rate / (best_lag - 1)
f_curr = sample_rate / best_lag
f_next = sample_rate / (best_lag + 1)

denom_f = r_prev - 2*r_curr + r_next
if abs(denom_f) > 1e-10:
    delta_f = 0.5 * (r_prev - r_next) / denom_f
    # Convert delta in frequency terms
    df = (f_prev - f_next) / 2  # approximate frequency step
    refined_f0_freq = f_curr + delta_f * df
    print(f"\n  Frequency-domain interpolation:")
    print(f"    F0 = {refined_f0_freq:.4f} Hz")
    print(f"    Error = {abs(refined_f0_freq - pm_f0):.4f} Hz")

# What if there's a sinc interpolation being used?
def sinc_interpolate(r, lag, depth=70):
    """Windowed sinc interpolation."""
    result = 0.0
    lag_int = int(np.floor(lag))
    for i in range(max(0, lag_int - depth), min(len(r), lag_int + depth + 1)):
        phi = lag - i
        if abs(phi) < 1e-10:
            sinc = 1.0
        else:
            sinc = np.sin(np.pi * phi) / (np.pi * phi)
        if depth > 0:
            window = 0.5 + 0.5 * np.cos(np.pi * phi / depth)
        else:
            window = 1.0
        result += r[i] * sinc * window
    return result

# Try finding peak with sinc interpolation
print("\n  Testing with sinc interpolation:")
best_r_sinc = 0
best_lag_sinc = best_lag
for offset in np.linspace(-0.5, 0.5, 101):
    test_lag = best_lag + offset
    r_val = sinc_interpolate(fcc, test_lag)
    if r_val > best_r_sinc:
        best_r_sinc = r_val
        best_lag_sinc = test_lag

refined_f0_sinc = sample_rate / best_lag_sinc
print(f"    refined_lag = {best_lag_sinc:.4f}")
print(f"    F0 = {refined_f0_sinc:.4f} Hz")
print(f"    Error = {abs(refined_f0_sinc - pm_f0):.4f} Hz")

# Check if the issue is consistent across frames
print("\n" + "=" * 70)
print("SYSTEMATIC ANALYSIS ACROSS FRAMES")
print("=" * 70)

lag_errors = []  # How far off is our best integer lag from expected?
interp_effects = []  # Does interpolation help or hurt?

pm_n_frames = call(pm_pitch, "Get number of frames")
for i in range(1, pm_n_frames + 1):
    t = call(pm_pitch, "Get time from frame number", i)
    pm_f0 = call(pm_pitch, "Get value in frame", i, "Hertz")

    if not pm_f0 or np.isnan(pm_f0) or pm_f0 <= 0:
        continue

    center = int(round(t * sample_rate))
    start = center - half_window
    if start < 0 or start + window_samples > len(samples):
        continue

    frame = samples[start:start + window_samples].copy()
    fcc = compute_full_xcorr(frame, min_lag, max_lag)

    # Find best integer lag
    best_lag = min_lag
    for lag in range(min_lag + 1, min(max_lag, len(fcc) - 1)):
        if fcc[lag] > fcc[lag-1] and fcc[lag] > fcc[lag+1]:
            if fcc[lag] > fcc[best_lag]:
                best_lag = lag

    if best_lag <= min_lag or best_lag >= len(fcc) - 1:
        continue

    expected_lag = sample_rate / pm_f0
    lag_error = best_lag - expected_lag
    lag_errors.append(lag_error)

    # F0 without interpolation
    f0_no_interp = sample_rate / best_lag

    # F0 with interpolation
    r_prev = fcc[best_lag - 1]
    r_curr = fcc[best_lag]
    r_next = fcc[best_lag + 1]
    denom = r_prev - 2*r_curr + r_next
    if abs(denom) > 1e-10:
        delta = 0.5 * (r_prev - r_next) / denom
        if abs(delta) < 1:
            f0_interp = sample_rate / (best_lag + delta)
        else:
            f0_interp = f0_no_interp
    else:
        f0_interp = f0_no_interp

    err_no_interp = abs(f0_no_interp - pm_f0)
    err_interp = abs(f0_interp - pm_f0)
    interp_effects.append(err_no_interp - err_interp)  # Positive = interp helps

print(f"\nLag error (best_lag - expected_lag):")
print(f"  Mean: {np.mean(lag_errors):.3f} samples")
print(f"  Std:  {np.std(lag_errors):.3f} samples")

print(f"\nInterpolation effect (error_reduction):")
print(f"  Mean: {np.mean(interp_effects):.3f} Hz (positive = interp helps)")
print(f"  Frames where interp helps: {sum(1 for e in interp_effects if e > 0)}/{len(interp_effects)}")
