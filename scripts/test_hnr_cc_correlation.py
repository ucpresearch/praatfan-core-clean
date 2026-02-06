#!/usr/bin/env python3
"""
Debug HNR CC correlation values to understand why values > 1.0 occur.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import parselmouth
from parselmouth.praat import call

audio_path = "tests/fixtures/one_two_three_four_five.wav"
snd_pm = parselmouth.Sound(audio_path)

# Load samples directly
samples = np.array(snd_pm.values.flatten())
sample_rate = snd_pm.sampling_frequency
duration = snd_pm.duration

# HNR CC parameters
time_step = 0.01
min_pitch = 75.0
max_pitch = 600.0
silence_threshold = 0.1
ppw = 1.0

# Window calculation (from our finding: (ppw + 1) / min_pitch)
window_duration = 2.0 / min_pitch  # 2-period window
window_samples = int(round(window_duration * sample_rate))
if window_samples % 2 == 0:
    window_samples += 1
half_window = window_samples // 2

min_lag = int(np.ceil(sample_rate / max_pitch))
max_lag = int(np.floor(sample_rate / min_pitch))

# Frame timing
n_frames = int(np.floor((duration - window_duration) / time_step + 1e-9)) + 1
t1 = (duration - (n_frames - 1) * time_step) / 2.0

global_peak = np.max(np.abs(samples))

print(f"Sample rate: {sample_rate}")
print(f"Window samples: {window_samples}")
print(f"Min lag: {min_lag}, Max lag: {max_lag}")
print(f"N frames: {n_frames}")

# Get parselmouth HNR CC for comparison
hnr_pm = call(snd_pm, "To Harmonicity (cc)", time_step, min_pitch, silence_threshold, ppw)

print(f"\n{'Frame':>5} {'time':>8} {'intensity':>10} {'best_r':>8} {'best_lag':>8} {'HNR_calc':>10} {'HNR_pm':>10}")
print("-" * 80)

for i in range(min(20, n_frames)):
    t = t1 + i * time_step
    center = int(round(t * sample_rate))
    start = center - half_window
    end = start + window_samples

    # Extract frame
    if start < 0 or end > len(samples):
        continue
    frame = samples[start:end].copy()

    # Check intensity
    local_peak = np.max(np.abs(frame))
    local_intensity = local_peak / (global_peak + 1e-30)

    # Compute cross-correlation for all lags
    n = len(frame)
    r_array = np.zeros(max_lag + 1)

    for lag in range(min_lag, min(max_lag + 1, n)):
        x1 = frame[:n-lag]
        x2 = frame[lag:]
        corr = np.sum(x1 * x2)
        e1 = np.sum(x1 * x1)
        e2 = np.sum(x2 * x2)
        if e1 > 0 and e2 > 0:
            r_array[lag] = corr / np.sqrt(e1 * e2)

    # Find best peak (without interpolation first)
    best_r = 0.0
    best_lag = 0
    for lag in range(min_lag + 1, min(max_lag, len(r_array) - 1)):
        if r_array[lag] > r_array[lag-1] and r_array[lag] > r_array[lag+1]:
            if r_array[lag] > best_r:
                best_r = r_array[lag]
                best_lag = lag

    # Now try with parabolic interpolation
    best_r_interp = 0.0
    for lag in range(min_lag + 1, min(max_lag, len(r_array) - 1)):
        if r_array[lag] > r_array[lag-1] and r_array[lag] > r_array[lag+1]:
            r_prev = r_array[lag-1]
            r_curr = r_array[lag]
            r_next = r_array[lag+1]

            denom = r_prev - 2*r_curr + r_next
            if abs(denom) > 1e-10:
                delta = 0.5 * (r_prev - r_next) / denom
                if abs(delta) < 1:
                    refined_r = r_curr - 0.25 * (r_prev - r_next) * delta
                    if refined_r > best_r_interp:
                        best_r_interp = refined_r
                elif r_curr > best_r_interp:
                    best_r_interp = r_curr
            elif r_curr > best_r_interp:
                best_r_interp = r_curr

    # Compute HNR
    if best_r_interp > 0 and best_r_interp < 1:
        hnr_calc = 10 * np.log10(best_r_interp / (1 - best_r_interp))
        hnr_calc = max(-200, min(100, hnr_calc))
    elif best_r_interp >= 1:
        hnr_calc = 100.0  # Clamped (should not happen!)
    else:
        hnr_calc = -200.0

    # Get parselmouth value
    hnr_pm_val = call(hnr_pm, "Get value in frame", i + 1)
    if hnr_pm_val is None or np.isnan(hnr_pm_val):
        hnr_pm_str = "-200.00"
    else:
        hnr_pm_str = f"{hnr_pm_val:.2f}"

    flag = " ⚠️ >1!" if best_r_interp > 1.0 else ""
    print(f"{i+1:>5} {t:>8.4f} {local_intensity:>10.4f} {best_r_interp:>8.4f} {best_lag:>8} {hnr_calc:>10.2f} {hnr_pm_str:>10}{flag}")

print("\n\n--- Checking raw r values for frame 15 (first voiced in PM) ---")
i = 14  # 0-indexed
t = t1 + i * time_step
center = int(round(t * sample_rate))
start = center - half_window
end = start + window_samples
frame = samples[start:end].copy()

print(f"\nFrame 15 at t={t:.4f}:")
print(f"  r values around max_lag region:")
for lag in range(max(min_lag, max_lag - 10), min(max_lag + 5, len(frame))):
    x1 = frame[:len(frame)-lag]
    x2 = frame[lag:]
    if len(x1) > 0 and len(x2) > 0:
        corr = np.sum(x1 * x2)
        e1 = np.sum(x1 * x1)
        e2 = np.sum(x2 * x2)
        if e1 > 0 and e2 > 0:
            r = corr / np.sqrt(e1 * e2)
            print(f"  lag={lag}: r={r:.4f}")
