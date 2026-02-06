#!/usr/bin/env python3
"""
Compare Parselmouth, Python, and Rust formant results.
"""

import numpy as np
import subprocess
import sys
sys.path.insert(0, 'src')

from praatfan.sound import Sound
from praatfan.formant import sound_to_formant_burg

def main():
    path = "tests/fixtures/tam-haʃaʁav-haɡadol-mono.wav"

    # --- Parselmouth (ground truth) ---
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(path)
    pm_formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    n_frames = call(pm_formant, "Get number of frames")

    f1_praat = []
    for i in range(1, n_frames + 1):
        t = call(pm_formant, "Get time from frame number", i)
        f1 = call(pm_formant, "Get value at time", 1, t, "Hertz", "Linear")
        f1_praat.append(f1)
    f1_praat = np.array(f1_praat)

    # --- Python ---
    sound = Sound.from_file(path)
    formant = sound_to_formant_burg(sound, 0.0, 5, 5500.0, 0.025, 50.0)
    f1_python = formant.formant_values(1)

    # --- Rust (run subprocess and parse output) ---
    result = subprocess.run(
        ['cargo', 'run', '--release', '--example', 'dump_f1_json'],
        cwd='rust',
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        # Try creating the example first
        rust_code = '''
use praatfan::Sound;

fn main() {
    let path = "../tests/fixtures/tam-haʃaʁav-haɡadol-mono.wav";
    let sound = Sound::from_file(path).expect("Failed to load");
    let formant = sound.to_formant_burg(0.0, 5, 5500.0, 0.025, 50.0);
    let f1 = formant.formant_values(1);
    for v in f1.iter() {
        println!("{}", v);
    }
}
'''
        with open('rust/examples/dump_f1_json.rs', 'w') as f:
            f.write(rust_code)

        result = subprocess.run(
            ['cargo', 'run', '--release', '--example', 'dump_f1_json'],
            cwd='rust',
            capture_output=True,
            text=True
        )

    f1_rust = np.array([float(line) for line in result.stdout.strip().split('\n') if line])

    # --- Comparison ---
    print("=" * 60)
    print("FORMANT F1 COMPARISON")
    print("=" * 60)

    print(f"\nFrame counts: Praat={len(f1_praat)}, Python={len(f1_python)}, Rust={len(f1_rust)}")

    # Ensure same length
    min_len = min(len(f1_praat), len(f1_python), len(f1_rust))
    f1_praat = f1_praat[:min_len]
    f1_python = f1_python[:min_len]
    f1_rust = f1_rust[:min_len]

    # Valid (voiced) frames
    valid_mask = ~np.isnan(f1_praat) & ~np.isnan(f1_python) & ~np.isnan(f1_rust)
    n_valid = np.sum(valid_mask)

    print(f"Voiced frames (all three have values): {n_valid}/{min_len}")

    # Statistics
    print(f"\n--- Mean F1 ---")
    mean_praat = np.nanmean(f1_praat)
    mean_python = np.nanmean(f1_python)
    mean_rust = np.nanmean(f1_rust)
    print(f"  Parselmouth: {mean_praat:.1f} Hz")
    print(f"  Python:      {mean_python:.1f} Hz (diff from Praat: {mean_python - mean_praat:+.1f} Hz)")
    print(f"  Rust:        {mean_rust:.1f} Hz (diff from Praat: {mean_rust - mean_praat:+.1f} Hz)")

    # Frame-by-frame differences
    diff_py_praat = f1_python[valid_mask] - f1_praat[valid_mask]
    diff_rs_praat = f1_rust[valid_mask] - f1_praat[valid_mask]
    diff_rs_py = f1_rust[valid_mask] - f1_python[valid_mask]

    print(f"\n--- Python vs Parselmouth ---")
    print(f"  Mean diff: {np.mean(diff_py_praat):+.2f} Hz")
    print(f"  Std diff:  {np.std(diff_py_praat):.2f} Hz")
    print(f"  Max diff:  {np.max(np.abs(diff_py_praat)):.2f} Hz")
    within_1hz = np.sum(np.abs(diff_py_praat) < 1.0)
    within_10hz = np.sum(np.abs(diff_py_praat) < 10.0)
    print(f"  Within 1 Hz:  {within_1hz}/{n_valid} ({100*within_1hz/n_valid:.1f}%)")
    print(f"  Within 10 Hz: {within_10hz}/{n_valid} ({100*within_10hz/n_valid:.1f}%)")

    print(f"\n--- Rust vs Parselmouth ---")
    print(f"  Mean diff: {np.mean(diff_rs_praat):+.2f} Hz")
    print(f"  Std diff:  {np.std(diff_rs_praat):.2f} Hz")
    print(f"  Max diff:  {np.max(np.abs(diff_rs_praat)):.2f} Hz")
    within_1hz = np.sum(np.abs(diff_rs_praat) < 1.0)
    within_10hz = np.sum(np.abs(diff_rs_praat) < 10.0)
    print(f"  Within 1 Hz:  {within_1hz}/{n_valid} ({100*within_1hz/n_valid:.1f}%)")
    print(f"  Within 10 Hz: {within_10hz}/{n_valid} ({100*within_10hz/n_valid:.1f}%)")

    print(f"\n--- Rust vs Python ---")
    print(f"  Mean diff: {np.mean(diff_rs_py):+.2f} Hz")
    print(f"  Std diff:  {np.std(diff_rs_py):.2f} Hz")
    print(f"  Max diff:  {np.max(np.abs(diff_rs_py)):.2f} Hz")
    within_1hz = np.sum(np.abs(diff_rs_py) < 1.0)
    within_10hz = np.sum(np.abs(diff_rs_py) < 10.0)
    print(f"  Within 1 Hz:  {within_1hz}/{n_valid} ({100*within_1hz/n_valid:.1f}%)")
    print(f"  Within 10 Hz: {within_10hz}/{n_valid} ({100*within_10hz/n_valid:.1f}%)")

    # Error distribution histogram
    print(f"\n--- Error Distribution (Rust vs Python) ---")
    bins = [0, 1, 5, 10, 25, 50, 100, float('inf')]
    hist, _ = np.histogram(np.abs(diff_rs_py), bins=bins)
    for i in range(len(bins)-1):
        upper = f"{bins[i+1]}" if bins[i+1] != float('inf') else "inf"
        bar = "█" * int(hist[i] * 40 / n_valid) if n_valid > 0 else ""
        print(f"  {bins[i]:5.0f}-{upper:>4}: {bar} {hist[i]} ({100*hist[i]/n_valid:.1f}%)")

if __name__ == "__main__":
    main()
