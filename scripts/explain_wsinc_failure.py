"""
Two questions to answer:
  (A) Does wsinc have a phase shift vs Praat? If yes, why?
  (B) Why did silent regions break so badly?
"""
from __future__ import annotations

import math
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy import signal as ss
from pathlib import Path

from praatfan.formant import _resample_wsinc

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def phase_report(label, out, praat):
    n = min(len(out), len(praat))
    d = out[:n] - praat[:n]
    fp = np.fft.rfft(praat[:n])
    fo = np.fft.rfft(out[:n])
    freqs = np.fft.rfftfreq(n, d=1/11000.0)
    # Only bins with significant praat magnitude
    m = np.abs(fp) > np.percentile(np.abs(fp), 50)
    ph = np.unwrap(np.angle(fo[m] / fp[m]))
    slope, intercept = np.polyfit(freqs[m], ph, 1)
    tau_samples = -slope / (2 * np.pi) * 11000
    resid = ph - (slope * freqs[m] + intercept)
    print(f"  {label:20s}: mean_abs={np.mean(np.abs(d)):.3e}  delay={tau_samples:+.4f} samples  "
          f"resid_phase_max={np.max(np.abs(resid)):.3e} rad")


def silence_report(label, out, praat):
    # Input is silent for first ~1031 input samples = ~472 output samples
    silent_region = slice(0, 400)
    os = out[silent_region]
    ps = praat[silent_region]
    print(f"  {label:20s}: silent[0:400] rms={np.sqrt(np.mean(os**2)):.3e} "
          f"praat_rms={np.sqrt(np.mean(ps**2)):.3e} "
          f"Nyquist_bin={np.abs(np.fft.rfft(os)[-1]):.3e} "
          f"(praat_Nyq={np.abs(np.fft.rfft(ps)[-1]):.3e})")


def main():
    snd = parselmouth.Sound(str(FIXTURES / "one_two_three_four_five.wav"))
    orig = snd.values[0].copy()
    praat = call(snd, "Resample", 11000.0, 50).values[0]

    new_length = int(len(orig) * 11000 / 24000)

    # Various resamplers
    # scipy no pad
    scipy_raw = ss.resample(orig, new_length)

    # scipy pow-2 pad
    pad = 1
    while pad < len(orig) * 2: pad *= 2
    padded = np.zeros(pad); padded[:len(orig)] = orig
    new_pad = int(pad * 11000 / 24000)
    scipy_pow2 = ss.resample(padded, new_pad)[:new_length]

    # windowed-sinc
    wsinc = _resample_wsinc(orig, 24000.0, 11000.0, precision=50)

    print("=== Phase analysis (vs Praat) ===")
    phase_report("scipy raw", scipy_raw, praat)
    phase_report("scipy pow-2", scipy_pow2, praat)
    phase_report("wsinc prec=50", wsinc, praat)

    print()
    print("=== Silence-region behavior (first 400 samples, input is silent there) ===")
    silence_report("scipy raw", scipy_raw, praat)
    silence_report("scipy pow-2", scipy_pow2, praat)
    silence_report("wsinc prec=50", wsinc, praat)

    # Deeper: what IS in Praat's silence-region output? Is it FFT-wraparound of later content?
    print()
    print("=== Where does Praat's silence-region content come from? ===")
    # If we ZERO OUT the signal after time 0.1s in the input, does the silence
    # region in Praat's output disappear?
    print("Test: zero out input after sample 2400 (= first 100ms only), resample, check silent region.")
    orig_truncated = orig.copy()
    orig_truncated[2400:] = 0.0  # zero out after 100ms
    snd_trunc = parselmouth.Sound(orig_truncated, sampling_frequency=24000.0)
    praat_trunc = call(snd_trunc, "Resample", 11000.0, 50).values[0]
    print(f"   Praat trunc silence[0:400] rms: {np.sqrt(np.mean(praat_trunc[:400]**2)):.3e}")
    print(f"   (original silence[0:400] rms: {np.sqrt(np.mean(praat[:400]**2)):.3e})")
    # If zeroing later content reduces silence-region noise, FFT is propagating it

    # What about with zeroed-later signal and scipy?
    scipy_trunc_raw = ss.resample(orig_truncated, new_length)
    print(f"   scipy raw trunc silence rms: {np.sqrt(np.mean(scipy_trunc_raw[:400]**2)):.3e}")


if __name__ == "__main__":
    main()
