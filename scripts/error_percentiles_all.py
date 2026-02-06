#!/usr/bin/env python3
"""
Calculate error percentiles for Python and Rust vs Parselmouth for all formants.
"""

import numpy as np
import subprocess
import sys
sys.path.insert(0, 'src')

from praatfan.sound import Sound
from praatfan.formant import sound_to_formant_burg

def get_rust_formants(formant_num):
    """Get formant values from Rust implementation."""
    rust_code = f'''
use praatfan::Sound;

fn main() {{
    let path = "../tests/fixtures/tam-haʃaʁav-haɡadol-mono.wav";
    let sound = Sound::from_file(path).expect("Failed to load");
    let formant = sound.to_formant_burg(0.0, 5, 5500.0, 0.025, 50.0);
    let f = formant.formant_values({formant_num});
    for v in f.iter() {{
        println!("{{}}", v);
    }}
}}
'''
    with open(f'rust/examples/dump_f{formant_num}.rs', 'w') as f:
        f.write(rust_code)

    result = subprocess.run(
        ['cargo', 'run', '--release', '--example', f'dump_f{formant_num}'],
        cwd='rust',
        capture_output=True,
        text=True
    )
    return np.array([float(line) for line in result.stdout.strip().split('\n') if line])


def print_stats(name, errors, n_valid):
    """Print statistics for an error array."""
    print(f"\n{'=' * 70}")
    print(f"{name}")
    print('=' * 70)
    print(f"  Mean error:        {np.mean(errors):8.2f} Hz")
    print(f"  Std error:         {np.std(errors):8.2f} Hz")
    print(f"  Median error:      {np.median(errors):8.2f} Hz")
    print(f"  Max error:         {np.max(errors):8.2f} Hz")
    print(f"  90th percentile:   {np.percentile(errors, 90):8.2f} Hz")
    print(f"  95th percentile:   {np.percentile(errors, 95):8.2f} Hz")
    print(f"  99th percentile:   {np.percentile(errors, 99):8.2f} Hz")

    within_1 = np.sum(errors < 1.0)
    within_10 = np.sum(errors < 10.0)
    within_50 = np.sum(errors < 50.0)
    within_100 = np.sum(errors < 100.0)
    print(f"\n  Within 1 Hz:       {within_1:4d}/{n_valid} ({100*within_1/n_valid:5.1f}%)")
    print(f"  Within 10 Hz:      {within_10:4d}/{n_valid} ({100*within_10/n_valid:5.1f}%)")
    print(f"  Within 50 Hz:      {within_50:4d}/{n_valid} ({100*within_50/n_valid:5.1f}%)")
    print(f"  Within 100 Hz:     {within_100:4d}/{n_valid} ({100*within_100/n_valid:5.1f}%)")


def main():
    path = "tests/fixtures/tam-haʃaʁav-haɡadol-mono.wav"

    # --- Parselmouth (ground truth) ---
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(path)
    pm_formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    n_frames = call(pm_formant, "Get number of frames")

    # --- Python ---
    sound = Sound.from_file(path)
    formant_py = sound_to_formant_burg(sound, 0.0, 5, 5500.0, 0.025, 50.0)

    print("=" * 70)
    print("FORMANT ERROR STATISTICS (vs Parselmouth)")
    print("=" * 70)
    print(f"\nFile: {path}")
    print(f"Total frames: {n_frames}")

    # Process each formant (F1-F5)
    for fn in range(1, 6):
        print(f"\n\n{'#' * 70}")
        print(f"# FORMANT F{fn}")
        print('#' * 70)

        # Get Parselmouth values
        f_praat = []
        for i in range(1, n_frames + 1):
            t = call(pm_formant, "Get time from frame number", i)
            f = call(pm_formant, "Get value at time", fn, t, "Hertz", "Linear")
            f_praat.append(f)
        f_praat = np.array(f_praat)

        # Get Python values
        f_python = formant_py.formant_values(fn)

        # Get Rust values
        f_rust = get_rust_formants(fn)

        # Ensure same length
        min_len = min(len(f_praat), len(f_python), len(f_rust))
        f_praat = f_praat[:min_len]
        f_python = f_python[:min_len]
        f_rust = f_rust[:min_len]

        # Valid (voiced) frames - where all three have values
        valid_mask = ~np.isnan(f_praat) & ~np.isnan(f_python) & ~np.isnan(f_rust)
        n_valid = np.sum(valid_mask)

        print(f"\nFrames with valid F{fn} in all three: {n_valid}/{min_len}")

        if n_valid == 0:
            print(f"  No valid frames for F{fn}")
            continue

        # Mean values
        print(f"\nMean F{fn}:")
        print(f"  Parselmouth: {np.nanmean(f_praat[valid_mask]):8.1f} Hz")
        print(f"  Python:      {np.nanmean(f_python[valid_mask]):8.1f} Hz")
        print(f"  Rust:        {np.nanmean(f_rust[valid_mask]):8.1f} Hz")

        # Absolute errors
        err_py_praat = np.abs(f_python[valid_mask] - f_praat[valid_mask])
        err_rs_praat = np.abs(f_rust[valid_mask] - f_praat[valid_mask])
        err_rs_py = np.abs(f_rust[valid_mask] - f_python[valid_mask])

        print_stats(f"PYTHON vs PARSELMOUTH (F{fn})", err_py_praat, n_valid)
        print_stats(f"RUST vs PARSELMOUTH (F{fn})", err_rs_praat, n_valid)
        print_stats(f"RUST vs PYTHON (F{fn})", err_rs_py, n_valid)


if __name__ == "__main__":
    main()
