#!/usr/bin/env python3
"""
Compare intermediate values for formant computation.
This outputs values that can be compared with the Rust version.
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal

def gaussian_window(n):
    if n <= 1:
        return np.array([1.0])
    alpha = 12.0
    mid = (n - 1) / 2.0
    i = np.arange(n)
    x = (i - mid) / mid
    return np.exp(-alpha * x * x)

def burg_lpc(samples, order):
    n = len(samples)
    if n <= order:
        return np.zeros(order + 1)

    a = np.zeros(order + 1)
    a[0] = 1.0

    ef = samples.copy()
    eb = samples.copy()

    for k in range(1, order + 1):
        num = 0.0
        den = 0.0
        for i in range(k, n):
            num += ef[i] * eb[i - 1]
            den += ef[i] ** 2 + eb[i - 1] ** 2

        if den < 1e-30:
            break

        reflection = -2.0 * num / den

        ef_new = np.zeros(n)
        eb_new = np.zeros(n)
        for i in range(k, n):
            ef_new[i] = ef[i] + reflection * eb[i - 1]
            eb_new[i] = eb[i - 1] + reflection * ef[i]
        ef = ef_new
        eb = eb_new

        a_new = np.zeros(order + 1)
        a_new[0] = 1.0
        for i in range(1, k):
            a_new[i] = a[i] + reflection * a[k - i]
        a_new[k] = reflection
        a = a_new

    return a

def main():
    path = "tests/fixtures/tam-haʃaʁav-haɡadol-mono.wav"

    sample_rate, data = wavfile.read(path)
    # Normalize to float
    if data.dtype == np.int16:
        samples = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        samples = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.float32:
        samples = data.astype(np.float64)
    else:
        samples = data.astype(np.float64)

    duration = len(samples) / sample_rate

    print("=== Intermediate value comparison ===")
    print(f"Sound: {len(samples)} samples, {sample_rate} Hz, {duration:.6f}s")

    max_formant_hz = 5500.0
    target_rate = 2.0 * max_formant_hz

    print("\n--- Resampling ---")
    print(f"Original rate: {sample_rate} Hz")
    print(f"Target rate: {target_rate} Hz")

    # Use scipy.signal.resample (what the Python implementation uses)
    new_length = int(len(samples) * target_rate / sample_rate)
    print(f"Expected resampled length: {new_length}")

    # Also try linear resampling for comparison
    ratio = target_rate / sample_rate
    resampled_linear = np.array([
        samples[min(int(i / ratio), len(samples) - 1)] * (1 - (i / ratio - int(i / ratio))) +
        samples[min(int(i / ratio) + 1, len(samples) - 1)] * (i / ratio - int(i / ratio))
        for i in range(new_length)
    ])

    # scipy resample (what Python implementation actually uses)
    resampled = signal.resample(samples, new_length)

    print(f"Resampled length: {len(resampled)}")
    print(f"First 5 resampled (scipy): {list(resampled[:5])}")
    print(f"First 5 resampled (linear): {list(resampled_linear[:5])}")

    # Pre-emphasis
    dt = 1.0 / target_rate
    alpha = np.exp(-2.0 * np.pi * 50.0 * dt)
    print(f"\n--- Pre-emphasis ---")
    print(f"Alpha: {alpha:.10f}")

    pre_emphasized = np.zeros(len(resampled))
    pre_emphasized[0] = resampled[0]
    for i in range(1, len(resampled)):
        pre_emphasized[i] = resampled[i] - alpha * resampled[i - 1]
    print(f"First 5 pre-emphasized: {list(pre_emphasized[:5])}")

    # Window parameters
    window_length = 0.025
    physical_window_duration = 2.0 * window_length
    window_samples = int(round(physical_window_duration * target_rate))
    if window_samples % 2 == 0:
        window_samples += 1
    half_window = window_samples // 2

    print(f"\n--- Window parameters ---")
    print(f"Physical window duration: {physical_window_duration:.6f}s")
    print(f"Window samples: {window_samples}")
    print(f"Half window: {half_window}")

    # Frame timing
    time_step = window_length / 4.0
    n_frames = int(np.floor((duration - physical_window_duration) / time_step)) + 1
    if n_frames < 1:
        n_frames = 1
    t1 = (duration - (n_frames - 1) * time_step) / 2.0

    print(f"\n--- Frame timing ---")
    print(f"Time step: {time_step:.6f}s")
    print(f"Number of frames: {n_frames}")
    print(f"First frame time (t1): {t1:.6f}s")

    # Generate window
    window = gaussian_window(window_samples)
    print(f"\n--- Gaussian window ---")
    print(f"Window center value: {window[half_window]:.10f}")
    print(f"Window edge value: {window[0]:.10f}")

    # Process frame 50
    frame_i = min(50, n_frames - 1)
    t = t1 + frame_i * time_step
    center_sample = int(round(t * target_rate))
    start_sample = center_sample - half_window

    print(f"\n--- Frame {frame_i} at t={t:.6f}s ---")
    print(f"Center sample: {center_sample}")
    print(f"Start sample: {start_sample}")

    # Extract frame
    if start_sample < 0 or start_sample + window_samples > len(pre_emphasized):
        frame_samples = np.zeros(window_samples)
        src_start = max(0, start_sample)
        src_end = min(len(pre_emphasized), start_sample + window_samples)
        dst_start = src_start - start_sample
        dst_end = dst_start + (src_end - src_start)
        frame_samples[dst_start:dst_end] = pre_emphasized[src_start:src_end]
    else:
        frame_samples = pre_emphasized[start_sample:start_sample + window_samples].copy()

    # Apply window
    windowed = frame_samples * window

    print(f"First 5 windowed: {list(windowed[:5])}")
    print(f"Windowed center: {windowed[half_window]:.10f}")

    # LPC
    lpc_order = 10
    lpc_coeffs = burg_lpc(windowed, lpc_order)

    print(f"\n--- LPC coefficients ---")
    for i, c in enumerate(lpc_coeffs):
        print(f"  a[{i}] = {c:.10f}")

    # Companion matrix eigenvalues
    order = lpc_order
    companion = np.zeros((order, order))
    for i in range(order):
        companion[0, i] = -lpc_coeffs[i + 1]
    for i in range(1, order):
        companion[i, i - 1] = 1.0

    eigenvalues = np.linalg.eigvals(companion)

    print(f"\n--- Eigenvalues (raw) ---")
    for i, e in enumerate(eigenvalues):
        freq = np.angle(e) * target_rate / (2 * np.pi)
        r = np.abs(e)
        bw = -np.log(r) * target_rate / np.pi if r > 0 else float('inf')
        print(f"  e[{i}] = ({e.real:.6f}, {e.imag:.6f}i) -> freq={freq:.1f} Hz, bw={bw:.1f} Hz")

    # Filter for upper half-plane and valid frequencies
    min_freq = 50.0
    max_freq = max_formant_hz - 50.0

    formants = []
    for e in eigenvalues:
        if e.imag <= 0:
            continue
        r = np.abs(e)
        theta = np.angle(e)
        freq = theta * target_rate / (2 * np.pi)
        bw = -np.log(r) * target_rate / np.pi if r > 0 else float('inf')

        if min_freq <= freq <= max_freq and bw > 0:
            formants.append((freq, bw))

    formants.sort(key=lambda x: x[0])

    print(f"\n--- Formants (frame {frame_i}) ---")
    for i, (freq, bw) in enumerate(formants):
        print(f"  F{i+1} = {freq:.1f} Hz (BW={bw:.1f} Hz)")

    # Now compare with actual praatfan formant computation
    print(f"\n--- Using praatfan formant API ---")
    import sys
    sys.path.insert(0, 'src')
    from praatfan.sound import Sound
    from praatfan.formant import sound_to_formant_burg

    sound = Sound.from_file(path)
    formant = sound_to_formant_burg(sound, 0.0, 5, max_formant_hz, window_length, 50.0)
    print(f"Total frames: {formant.n_frames}")

    if frame_i < formant.n_frames:
        frame = formant.frames[frame_i]
        print(f"Frame {frame_i} formants:")
        for i, fp in enumerate(frame.formants):
            print(f"  F{i+1} = {fp.frequency:.1f} Hz (BW={fp.bandwidth:.1f} Hz)")

    # Mean F1
    f1 = formant.formant_values(1)
    valid = f1[~np.isnan(f1)]
    mean_f1 = np.mean(valid) if len(valid) > 0 else float('nan')
    print(f"\nMean F1 (Python): {mean_f1:.1f} Hz")


if __name__ == "__main__":
    main()
