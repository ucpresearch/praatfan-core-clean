#!/usr/bin/env python3
"""
Calculate error percentiles for Python and Rust vs Parselmouth.
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

    # --- Rust ---
    # Create the example if needed
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

    # Ensure same length
    min_len = min(len(f1_praat), len(f1_python), len(f1_rust))
    f1_praat = f1_praat[:min_len]
    f1_python = f1_python[:min_len]
    f1_rust = f1_rust[:min_len]

    # Valid (voiced) frames
    valid_mask = ~np.isnan(f1_praat) & ~np.isnan(f1_python) & ~np.isnan(f1_rust)
    n_valid = np.sum(valid_mask)

    # Absolute errors
    err_py_praat = np.abs(f1_python[valid_mask] - f1_praat[valid_mask])
    err_rs_praat = np.abs(f1_rust[valid_mask] - f1_praat[valid_mask])
    err_rs_py = np.abs(f1_rust[valid_mask] - f1_python[valid_mask])

    print("=" * 70)
    print("FORMANT F1 ERROR STATISTICS")
    print("=" * 70)
    print(f"\nTotal frames: {min_len}")
    print(f"Voiced frames (all three have values): {n_valid}")

    print("\n" + "=" * 70)
    print("PYTHON vs PARSELMOUTH")
    print("=" * 70)
    print(f"  Mean error:        {np.mean(err_py_praat):8.2f} Hz")
    print(f"  Std error:         {np.std(err_py_praat):8.2f} Hz")
    print(f"  Median error:      {np.median(err_py_praat):8.2f} Hz")
    print(f"  Max error:         {np.max(err_py_praat):8.2f} Hz")
    print(f"  90th percentile:   {np.percentile(err_py_praat, 90):8.2f} Hz")
    print(f"  95th percentile:   {np.percentile(err_py_praat, 95):8.2f} Hz")
    print(f"  99th percentile:   {np.percentile(err_py_praat, 99):8.2f} Hz")

    within_1 = np.sum(err_py_praat < 1.0)
    within_10 = np.sum(err_py_praat < 10.0)
    within_50 = np.sum(err_py_praat < 50.0)
    print(f"\n  Within 1 Hz:       {within_1:4d}/{n_valid} ({100*within_1/n_valid:5.1f}%)")
    print(f"  Within 10 Hz:      {within_10:4d}/{n_valid} ({100*within_10/n_valid:5.1f}%)")
    print(f"  Within 50 Hz:      {within_50:4d}/{n_valid} ({100*within_50/n_valid:5.1f}%)")

    print("\n" + "=" * 70)
    print("RUST vs PARSELMOUTH")
    print("=" * 70)
    print(f"  Mean error:        {np.mean(err_rs_praat):8.2f} Hz")
    print(f"  Std error:         {np.std(err_rs_praat):8.2f} Hz")
    print(f"  Median error:      {np.median(err_rs_praat):8.2f} Hz")
    print(f"  Max error:         {np.max(err_rs_praat):8.2f} Hz")
    print(f"  90th percentile:   {np.percentile(err_rs_praat, 90):8.2f} Hz")
    print(f"  95th percentile:   {np.percentile(err_rs_praat, 95):8.2f} Hz")
    print(f"  99th percentile:   {np.percentile(err_rs_praat, 99):8.2f} Hz")

    within_1 = np.sum(err_rs_praat < 1.0)
    within_10 = np.sum(err_rs_praat < 10.0)
    within_50 = np.sum(err_rs_praat < 50.0)
    print(f"\n  Within 1 Hz:       {within_1:4d}/{n_valid} ({100*within_1/n_valid:5.1f}%)")
    print(f"  Within 10 Hz:      {within_10:4d}/{n_valid} ({100*within_10/n_valid:5.1f}%)")
    print(f"  Within 50 Hz:      {within_50:4d}/{n_valid} ({100*within_50/n_valid:5.1f}%)")

    print("\n" + "=" * 70)
    print("RUST vs PYTHON")
    print("=" * 70)
    print(f"  Mean error:        {np.mean(err_rs_py):8.2f} Hz")
    print(f"  Std error:         {np.std(err_rs_py):8.2f} Hz")
    print(f"  Median error:      {np.median(err_rs_py):8.2f} Hz")
    print(f"  Max error:         {np.max(err_rs_py):8.2f} Hz")
    print(f"  90th percentile:   {np.percentile(err_rs_py, 90):8.2f} Hz")
    print(f"  95th percentile:   {np.percentile(err_rs_py, 95):8.2f} Hz")
    print(f"  99th percentile:   {np.percentile(err_rs_py, 99):8.2f} Hz")

    within_1 = np.sum(err_rs_py < 1.0)
    within_10 = np.sum(err_rs_py < 10.0)
    within_50 = np.sum(err_rs_py < 50.0)
    print(f"\n  Within 1 Hz:       {within_1:4d}/{n_valid} ({100*within_1/n_valid:5.1f}%)")
    print(f"  Within 10 Hz:      {within_10:4d}/{n_valid} ({100*within_10/n_valid:5.1f}%)")
    print(f"  Within 50 Hz:      {within_50:4d}/{n_valid} ({100*within_50/n_valid:5.1f}%)")

if __name__ == "__main__":
    main()
