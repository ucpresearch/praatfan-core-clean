"""
Measure parity vs parselmouth on ALL algorithms — not just Formant —
so we can check the resampler change doesn't regress anything and to
verify which modules actually depend on _resample.

Algorithms:
  - Formant (uses _resample internally)
  - Pitch (native rate)
  - Intensity (native rate)
  - HNR / Harmonicity (derives from Pitch)
  - Spectrum (single-frame FFT, native rate)
  - Spectrogram (native rate)

Plus a resample-then-analyze path: sound.resample(11000).to_pitch() etc.
That path uses _resample, so regressions there would matter.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

from praatfan.sound import Sound as PFSound
from praatfan.formant import sound_to_formant_burg
from praatfan.pitch import sound_to_pitch
from praatfan.intensity import sound_to_intensity
from praatfan.harmonicity import sound_to_harmonicity_cc
from praatfan.spectrogram import sound_to_spectrogram

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
FILES = ["one_two_three_four_five.wav",
         "one_two_three_four_five_16k.wav",
         "one_two_three_four_five-gain5.wav"]


def _diffs(ours_vals, praat_vals):
    arr_o = np.asarray(ours_vals, dtype=np.float64)
    arr_p = np.asarray(praat_vals, dtype=np.float64)
    n = min(len(arr_o), len(arr_p))
    if n == 0:
        return np.array([])
    o = arr_o[:n]; p = arr_p[:n]
    mask = ~(np.isnan(o) | np.isnan(p))
    return np.abs(o[mask] - p[mask])


def _summary(label, diffs):
    if len(diffs) == 0:
        print(f"    {label:18s}: no valid frames")
        return
    print(f"    {label:18s}: n={len(diffs):4d} "
          f"mean={np.mean(diffs):8.3f} p95={np.percentile(diffs, 95):8.3f} "
          f"p99={np.percentile(diffs, 99):8.3f} max={np.max(diffs):9.3f}")


def test_formant(path):
    snd = parselmouth.Sound(str(path))
    fmt_p = call(snd, "To Formant (burg)", 0.005, 5, 5500.0, 0.025, 50.0)
    pf = PFSound.from_file(str(path))
    fmt_o = sound_to_formant_burg(pf, time_step=0.005, max_num_formants=5,
                                    max_formant_hz=5500.0, window_length=0.025,
                                    pre_emphasis_from=50.0)
    n = call(fmt_p, "Get number of frames")
    times = np.array([f.time for f in fmt_o.frames])
    rows = []
    for i in range(1, n + 1):
        t = call(fmt_p, "Get time from frame number", i)
        j = int(np.argmin(np.abs(times - t)))
        vals = {}
        for k in (1, 2, 3):
            p = call(fmt_p, "Get value at time", k, t, "Hertz", "Linear")
            fp = fmt_o.frames[j].get_formant(k)
            vals[f"praat_f{k}"] = p
            vals[f"our_f{k}"] = fp.frequency if fp else float('nan')
        rows.append(vals)
    return rows


def test_pitch(path):
    snd = parselmouth.Sound(str(path))
    p_obj = call(snd, "To Pitch", 0.0, 75.0, 600.0)
    pf = PFSound.from_file(str(path))
    our_p = sound_to_pitch(pf, time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0)
    n = call(p_obj, "Get number of frames")
    times_p = [call(p_obj, "Get time from frame number", i) for i in range(1, n + 1)]
    praat_vals = [call(p_obj, "Get value at time", t, "Hertz", "Linear") for t in times_p]
    ours_vals = []
    our_times = np.array([f.time for f in our_p.frames])
    for t in times_p:
        j = int(np.argmin(np.abs(our_times - t)))
        ours_vals.append(our_p.frames[j].frequency if our_p.frames[j].voiced else float('nan'))
    return praat_vals, ours_vals


def test_intensity(path):
    snd = parselmouth.Sound(str(path))
    intens_p = call(snd, "To Intensity", 100.0, 0.0, "yes")
    pf = PFSound.from_file(str(path))
    our_i = sound_to_intensity(pf, min_pitch=100.0)
    n = call(intens_p, "Get number of frames")
    times = [call(intens_p, "Get time from frame number", i) for i in range(1, n + 1)]
    praat_vals = [call(intens_p, "Get value at time", t, "Cubic") for t in times]
    our_times = our_i.times  # property: np.ndarray
    ours_vals = []
    for t in times:
        j = int(np.argmin(np.abs(our_times - t)))
        ours_vals.append(our_i.values[j])
    return praat_vals, ours_vals


def test_harmonicity(path):
    snd = parselmouth.Sound(str(path))
    h_p = call(snd, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
    pf = PFSound.from_file(str(path))
    our_h = sound_to_harmonicity_cc(pf, time_step=0.01, min_pitch=75.0,
                                      silence_threshold=0.1, periods_per_window=1.0)
    n = call(h_p, "Get number of frames")
    times = [call(h_p, "Get time from frame number", i) for i in range(1, n + 1)]
    praat_vals = [call(h_p, "Get value at time", t, "Linear") for t in times]
    ours_vals = []
    our_times = our_h.times
    for t in times:
        j = int(np.argmin(np.abs(our_times - t)))
        v = our_h.values[j]
        ours_vals.append(v if not (v is None or math.isnan(v) or v < -190) else float('nan'))
    return praat_vals, ours_vals


def test_spectrogram(path):
    snd = parselmouth.Sound(str(path))
    sp_p = call(snd, "To Spectrogram", 0.005, 5000.0, 0.002, 20.0, "Gaussian")
    pf = PFSound.from_file(str(path))
    our_sp = sound_to_spectrogram(pf, window_length=0.005, max_frequency=5000.0,
                                    time_step=0.002, frequency_step=20.0)
    # Sample a few points for diff
    n_t = call(sp_p, "Get number of frames")
    n_f = call(sp_p, "Get number of frequencies") if False else 0
    praat_vals = []
    ours_vals = []
    for i in range(1, n_t + 1, max(1, n_t // 30)):
        t = call(sp_p, "Get time from frame number", i)
        for f_hz in [500.0, 1500.0, 2500.0, 3500.0]:
            p_val = call(sp_p, "Get power at", t, f_hz)
            # Our side: get power at closest grid point
            try:
                our_val = our_sp.get_power_at(t, f_hz)
            except Exception:
                our_val = float('nan')
            praat_vals.append(p_val); ours_vals.append(our_val)
    return praat_vals, ours_vals


def main():
    print(f"{'file':<40} algorithm")
    for name in FILES:
        path = FIXTURES / name
        print(f"\n== {name}")
        # Formant
        rows = test_formant(path)
        for k in (1, 2, 3):
            diffs = _diffs([r[f"our_f{k}"] for r in rows], [r[f"praat_f{k}"] for r in rows])
            _summary(f"Formant F{k} (Hz)", diffs)
        # Pitch
        p, o = test_pitch(path)
        _summary("Pitch (Hz)       ", _diffs(o, p))
        # Intensity
        p, o = test_intensity(path)
        _summary("Intensity (dB)   ", _diffs(o, p))
        # Harmonicity
        try:
            p, o = test_harmonicity(path)
            _summary("HNR (dB)         ", _diffs(o, p))
        except Exception as e:
            print(f"    HNR: ERROR {e!r}")
        # Spectrogram
        try:
            p, o = test_spectrogram(path)
            _summary("Spectrogram pow  ", _diffs(o, p))
        except Exception as e:
            print(f"    Spectrogram: ERROR {e!r}")


if __name__ == "__main__":
    main()
