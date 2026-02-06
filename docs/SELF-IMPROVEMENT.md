# Formant Accuracy: Potential Improvement Areas

After exhaustively sweeping all parameter combinations (936 configs, up to 3 simultaneous
changes), the current formant implementation is at its accuracy ceiling within the existing
parameter space. This document lists implementation-level changes that fall *outside* that
parameter space — structural differences in how the algorithm operates.

All items below are derived from public documentation, published papers, or standard DSP
practice. No Praat source code was consulted.

---

## Current Accuracy (Python backend, one_two_three_four_five.wav)

### After zero-padding fix (Option 9)

| Formant | P50 | P95 | P99 | Max |
|---------|-----|-----|-----|-----|
| F1 | 0.83 Hz | 19.5 Hz | 45.5 Hz | 154 Hz |
| F2 | 1.04 Hz | 50.7 Hz | 277.7 Hz | 422 Hz |
| F3 | 1.86 Hz | 69.6 Hz | 382.5 Hz | 671 Hz |

### Before zero-padding fix (original baseline)

| Formant | P50 | P95 | P99 | Max |
|---------|-----|-----|-----|-----|
| F1 | 0.96 Hz | 24.1 Hz | 208.6 Hz | 1206 Hz |
| F2 | 1.46 Hz | 83.8 Hz | 362.8 Hz | 1192 Hz |
| F3 | 2.05 Hz | 126.6 Hz | 541.5 Hz | 1227 Hz |

P50 is excellent. The large P95/P99/Max errors come from 2-3 near-silence frames between
words where LPC analysis is inherently unstable. The zero-padding fix (Option 9) reduced
P95sum by 40% and max errors by 45-87%.

---

## Option 1: Sample Weighting / Windowed Burg Variant

**Source:** Childers (1978) describes the basic Burg algorithm, but variants exist.

**Idea:** Some implementations apply the window to the autocorrelation rather than the
signal itself, or use a different weighting scheme within Burg's recursion. Our
implementation windows the signal first, then feeds the windowed signal to Burg.

**Expected impact:** Small. Would affect all frames uniformly, unlikely to fix silence frames.

**Status:** Not tested.

---

## Option 2: Pre-emphasis Before vs After Resampling

**Source:** Praat manual specifies both resampling and pre-emphasis but does not document
the order.

**Idea:** We currently resample first, then pre-emphasize. The reverse order (pre-emphasize
at the original sample rate, then resample) gives slightly different results because the
anti-aliasing filter in resampling interacts with the pre-emphasis spectrum.

**Result:** Current order (resample → pre-emphasize) is dramatically better. The alternative
(pre-emphasize → resample) degrades P50 by 8-12x (F1: 0.96→7.69, F2: 1.46→11.61,
F3: 2.05→23.43). Current wins 791/871 frames where the two differ.

**Status:** Tested. Current order is correct.

---

## Option 3: Resampling Length Calculation

**Source:** Standard ambiguity in FFT-based resample implementations.

**Idea:** We use `int(n * new_rate / old_rate)`. Different rounding (floor, ceil, round)
changes the output length by one sample. This changes boundary frame spectral content.

**Expected impact:** Tiny. One sample difference out of thousands.

**Status:** Not tested.

---

## Option 4: LPC Order Adaptation

**Source:** General LPC literature. Praat manual says "2 * max_formants" for LPC order.

**Idea:** Some formant trackers adaptively adjust LPC order based on signal characteristics —
for instance, increasing order if fewer roots fall in the valid frequency range, or
decreasing it for short segments.

**Expected impact:** Low. Would only help if the fixed order is inappropriate for some frames.

**Status:** Not tested.

---

## Option 5: Formant Tracking / Temporal Continuity

**Source:** Standard formant tracking literature. Praat uses Viterbi for pitch — the same
technique can be applied to formants.

**Idea:** Our implementation treats each frame independently. If Praat applies temporal
continuity constraints (e.g., Viterbi smoothing across frames, penalizing large
frame-to-frame jumps in formant frequencies), it would produce more stable estimates at
transition frames and near-silence frames. This is a well-known technique in formant
tracking (e.g., Deng et al. 2006, "Tracking vocal tract resonances using an analytical
nonlinear predictor").

**Expected impact:** High for P95/P99/Max. Would stabilize exactly the frames that currently
produce the largest errors. Would not affect P50 (already good).

**Result:** Tracking does NOT help. Both greedy tracking (P95sum: 234→6274) and
reference-frequency tracking (P95sum: 234→2156-2702) make things dramatically worse.

The root cause investigation (Part 1) revealed the real issue: **the LPC coefficients
themselves differ**, not just the formant labeling. For example at frame 22:
- Our LPC produces 3 upper-half-plane roots: 1878, 3116, 4393 Hz
- Praat's LPC produces 4: 934, 1924, 3167, 4366 Hz

The polynomials are fundamentally different. Tracking can't fix wrong polynomials.

**Status:** Tested. Not useful — the issue is upstream in the signal preparation.

---

## Option 6: Energy-Based Frame Suppression

**Source:** Standard practice in speech analysis. Many formant trackers skip or mark as
undefined frames where the signal energy is below a threshold.

**Idea:** Rather than running LPC on near-silence frames and getting unreliable results,
suppress formant output (return NaN/undefined) when frame energy falls below a threshold.
This wouldn't improve accuracy directly, but if Praat does this and returns NaN for those
frames, our comparison would skip them, removing the outliers.

**Expected impact:** High for Max/P99 if Praat suppresses these frames. Straightforward to
test: check whether parselmouth returns NaN at the frames where we have large errors.

**Result:** Parselmouth returns valid formant values at ALL 317 frames — zero NaN frames.
Praat does NOT suppress formants at low-energy frames. The differences are real LPC
divergences: different formant counts (we find 4 where Praat finds 5, or vice versa) and
completely different frequency configurations.

**Status:** Tested. Ruled out — Praat computes formants at all frames.

---

## Option 7: Burg Algorithm Numerical Refinements

**Source:** Marple (1987), "Digital Spectral Analysis with Applications."

**Idea:** The reflection coefficient denominator `ef[i]^2 + eb[i-1]^2` is one formulation.
A more numerically stable variant uses a running energy estimate. For well-conditioned
signals this makes no difference, but for near-silence frames the numerical behavior could
diverge.

**Expected impact:** Small. Only affects ill-conditioned frames.

**Status:** Not tested.

---

## Option 8: Companion Matrix Form

**Source:** Standard numerical linear algebra.

**Idea:** Multiple valid companion matrix constructions exist (coefficients in first row vs
last row, transposed vs standard). They produce identical eigenvalues mathematically but
can differ numerically for ill-conditioned polynomials — exactly the case for near-silence
frames where LPC coefficients are unreliable.

**Expected impact:** Small. Same reasoning as Option 7.

**Status:** Not tested.

---

## ROOT CAUSE FOUND: Resampling Method (Option 9)

**Discovered:** 2026-02-06, during investigation of Options 2, 5, and 6.

### The Discovery

While investigating why our LPC coefficients differ from Praat's at problematic frames,
we compared the resampled signals directly. Running `parselmouth.call(sound, "Resample",
11000, 50)` and comparing with `scipy.signal.resample`:

- **Overall RMS difference: 6.8%** — sounds small but is enormous for LPC analysis
- Using Praat's resampled signal with our Burg algorithm: **P95sum drops from 234 to 13**

This means **resampling accounts for ~94% of all formant error**. The Burg algorithm,
root finding, formant filtering — everything else is essentially correct.

### Why FFT Resampling Differs from Praat's Sinc

**scipy.signal.resample** (FFT-based):
- Treats signal as periodic (circular convolution)
- Edge discontinuity creates Gibbs-like ringing that propagates to ALL samples
- Truncates spectrum sharply at new Nyquist (no transition band)

**Praat's resampling** (windowed sinc, precision=50):
- Local operation: each output sample depends on ~100 nearby input samples
- Hanning-windowed sinc provides smooth frequency rolloff
- No periodicity assumption — edge effects limited to boundaries
- First sample at t1 = 0.5/new_rate (centered in first sample interval)

### What Was Tested

| Approach | Signal RMS vs Praat | Formant P95sum | Notes |
|----------|--------------------:|---------------:|-------|
| scipy.signal.resample | 6.8% | 234 | Current baseline |
| scipy + zero-pad 4x | 6.8% | 140 | 40% improvement! |
| scipy + zero-pad 8x | 6.8% | 145 | Diminishing returns |
| scipy.signal.resample_poly | 6.7% | 769 | Much worse |
| Custom sinc (correct timing) | 0.66% | 560 | Close signal, bad formants (bug?) |
| Custom sinc (wrong timing) | 6.8% | 539 | Same as scipy |
| Phase-shifted FFT | varies | 278+ | No help |
| Mirror-pad | 6.8% | 204-516 | Inconsistent |
| Edge tapering | 6.8% | 234-658 | Harmful |
| **Praat resample (reference)** | **0%** | **13** | **Ground truth** |

### Key Observations

1. **Signal RMS does not predict formant accuracy.** Zero-padding doesn't change the
   global RMS (still 6.8%) but improves P95sum by 40%. The improvement comes from
   reducing artifacts at specific frames near signal boundaries.

2. **Our custom sinc (0.66% RMS) gave worse formants (560) than scipy (6.8%, 234).**
   This paradox suggests our sinc implementation has correlated local artifacts that
   the LPC is sensitive to, even though the global match is better.

3. **Mean removal before Burg doesn't help** — tested separately, made things worse
   (P95sum: 234 → 258). The issue is genuinely in the resampled signal, not in the
   Burg algorithm.

4. **Pre-emphasis order matters:** Resample-then-pre-emphasize (current) is correct.
   The reverse is 8-12x worse.

### Implemented Fix

**Zero-padding before FFT resample** (in `_resample` in formant.py):
- Pad signal to 5x length with zeros before FFT resample
- Truncate output to original target length
- P95sum improvement: 234 → ~140 (40% better)
- No regression in P50 (actually slightly better: 4.47 → 3.7 Hz)

### Path to Full Fix

To match Praat's accuracy (P95sum=13), we would need to implement Praat-style windowed
sinc resampling. Key requirements from the Praat manual:
- Hanning-windowed sinc kernel with 50 zero-crossings on each side
- Sample grid: first sample at t1 = 0.5/new_rate (not t=0)
- Anti-aliasing: sinc at new Nyquist frequency, scaled by rate ratio
- Correct handling of edge samples (no periodicity assumption)

This is a significant implementation effort that must be done without consulting Praat's
source code (clean-room constraint). The zero-padding fix provides 40% of the possible
improvement with minimal code change.

---

## Intensity Accuracy Investigation

### Current Accuracy (one_two_three_four_five.wav, min_pitch=100)

| Time step | P50 | P95 | Max | Frames > 1 dB |
|-----------|-----|-----|-----|----------------|
| Default (0.008s) | 0.005 dB | 0.34 dB | 2.04 dB | 5/196 |
| 0.01s | 0.005 dB | 0.40 dB | 10.98 dB | 4/157 |

**Note:** The full_comparison.py previously reported P95=9.155 / Max=12.59 dB due to a bug:
`pm_call(pm_snd, "To Intensity", 100, 0.01)` defaults to `subtract_mean=False` in
parselmouth, while our implementation always uses `subtract_mean=True`. Adding the explicit
`True` argument fixes this. The actual intensity errors are much smaller than reported.

### Investigation Summary

**Confirmed correct:**
- Energy formula: `sum(s² × w) / sum(w)` matches Praat's `subtract_mean=False` to 0.03 dB
- Window type: Gaussian with α=13.2 and edge correction (tested Kaiser-20 through Kaiser-30,
  plus Gaussians with α=12-25; current Gaussian gives best P50)
- Physical/effective ratio: 2.25 (tested 2.0 and 2.5; 2.25 is clearly best)
- DC removal: unweighted per-frame mean gives best P50

**Root cause of outlier errors:**
The 5 frames with > 1 dB error are all at speech onset/offset transitions. At these frames,
signal energy is concentrated at one edge of the analysis window. The unweighted per-frame
mean is dominated by this edge signal, giving a DC estimate with the wrong sign relative to
the Gaussian-weighted center. Subtracting this wrong-sign DC INCREASES the windowed energy
instead of reducing it.

**What was tested for DC removal:**

| Method | P50 | P95 | Max | Notes |
|--------|-----|-----|-----|-------|
| Unweighted mean (current) | 0.005 | 0.40 | 10.98 | Best P50 |
| Weighted mean (Gaussian) | 0.006 | 1.23 | 10.03 | Worse overall |
| Global mean | 0.006 | 8.28 | 13.58 | Too small to help |
| No DC removal | 0.006 | 8.19 | 12.05 | Matches PM no-DC |
| Mean scope=0.90 | 0.005 | 0.22 | 4.09 | Best P95 (ts=0.01) |
| Mean scope=0.90 (default ts) | 0.005 | 0.15 | 1.35 | Best overall? |
| Running Gaussian convolution | 0.006 | 3.87 | 13.20 | Worse |

**Window type tests (at frame 9, ts=0.01):**

All window types (Gaussian α=12-25, Kaiser β=15-30) give essentially the same max error
(~10.9 dB). The issue is not the window shape but the mean subtraction.

### Potential Fix: Narrower Mean Scope

Computing the per-frame mean over the central 90% of the physical window (instead of 100%)
prevents edge signal from biasing the DC estimate. This reduces:
- Default ts: Max 2.04 → 1.35 dB, P95 0.34 → 0.15 dB
- ts=0.01: Max 10.98 → 4.09 dB, P95 0.40 → 0.22 dB

**Status:** Not implemented. The improvement is modest and may not generalize to all audio.
The Praat manual says "subtracting the mean pressure around this point" but doesn't specify
the exact scope. Further investigation needed to determine if this is what Praat does.

### Comparison Script Fix

**Bug:** `pm_call(pm_snd, "To Intensity", 100, 0.01)` in full_comparison.py must include
the `True` argument for `subtract_mean`, as parselmouth defaults to `False` when not
specified (unlike Praat's GUI which defaults to `True`).

**Fix:** Changed to `pm_call(pm_snd, "To Intensity", 100, 0.01, True)`.

---

## Spectrogram Accuracy Investigation

### Root Cause Found: FFT Bin Alignment + Missing Interpolation

**Discovered:** 2026-02-06

The original implementation stored spectrogram power values at user-specified frequency
intervals (e.g., every 20 Hz), mapping each to the nearest FFT bin. Praat instead stores
values at the actual FFT frequency resolution (sample_rate / fft_size), which is typically
finer than the user-requested step.

For example, with window_length=0.025 and frequency_step=20 Hz:
- FFT size: 2048 (next power of 2 above sr/freq_step)
- Actual FFT resolution: 24000/2048 = 11.7188 Hz
- Original code: stored 250 bins at 20 Hz step (snapping to nearest FFT bin)
- Praat: stores 426 bins at 11.7188 Hz step (actual FFT bins)

Additionally, Praat's "Get power at" performs interpolation between grid points, while
our implementation used nearest-bin lookup.

### The Fixes

1. **Store at FFT bin resolution** (Python + Rust): Changed `n_freq_bins = round(max_freq / freq_step)`
   to `n_freq_bins = round(max_freq / df_fft)` and store FFT bins directly instead of
   mapping via nearest-bin lookup.

2. **Bilinear interpolation** in `Get power at` (praat.py + Rust python.rs): Changed from
   nearest-bin lookup to bilinear interpolation in both time and frequency dimensions.

### Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| P50 | 0.22 (22%) | 0.0025 (0.25%) | **88x better** |
| P90 | 0.87 | 0.0075 | 116x better |
| P95 | - | 0.0095 | - |
| Max | 1.74 | 0.023 | 76x better |

92.5% of comparison points now have < 1% relative error. The remaining ~0.3% median
error likely comes from minor differences in window normalization between our Gaussian
and Praat's (ratio ≈ 0.997 at exact bin-aligned points).

### Validation

At exact FFT bin frequencies (no interpolation needed), the power ratio is:
- Mean: 0.997, Median: 0.999, Std: 0.009
This confirms the underlying FFT computation matches Praat almost exactly.

---

## Pitch CC Accuracy Investigation

### Root Cause #1 Found: Wrong Default Time Step

**Discovered:** 2026-02-06

The default time_step for Pitch CC was `0.75 / pitch_floor` (same as AC), but Praat uses
`0.25 / pitch_floor` for CC. With pitch_floor=75 Hz:
- Our default: 0.010s → 161 frames
- Praat's default: 0.003333s → 482 frames

**Fix:** In both Python (`pitch.py:708`) and Rust (`pitch.rs:965`), changed the default
time_step to be method-dependent: `0.25/floor` for CC, `0.75/floor` for AC.

### Results After Time Step Fix

| Metric | Before | After |
|--------|--------|-------|
| Frame count | 161 (vs PM=482) | 482 (matching PM) |
| Frame timing | t1=0.016, dt=0.010 | t1=0.01433, dt=0.00333 (matching PM) |
| P50 | 0.93 Hz | 0.84 Hz |
| P95 | 6.87 Hz | 5.70 Hz |
| Max | 23.39 Hz | 31.93 Hz |

### Remaining Error Analysis (P50=0.84 Hz)

The CC errors are strongly frequency-dependent:

| F0 range | Frames | P50 error | Max error |
|----------|--------|-----------|-----------|
| 100-150 Hz | 72 | 0.20 Hz | 3.10 Hz |
| 150-200 Hz | 35 | 1.12 Hz | 22.03 Hz |
| 200-300 Hz | 196 | 1.12 Hz | 31.93 Hz |

For comparison, Pitch AC has P50=0.014 Hz — 60x better than CC.

The worst CC frames (>10 Hz error) are at phoneme transition points (~0.15s, ~0.93s, ~1.16s)
where the Viterbi path tracking picks different candidates. The best frames match to 0.00 Hz.

### Deep Investigation: Voicing False Positives (2026-02-06)

**Finding:** 11 frames where PF voices but PM doesn't, 2 frames PM voices but PF doesn't
(97.3% voicing agreement). All 11 false positives are at speech onset transitions, 1-3
frames before PM starts voicing.

**Root cause analysis:**
- At these transition frames, CC cross-correlation genuinely finds strong peaks (r=0.53-0.98)
  because the speech portion of the frame is periodic
- The unvoiced candidate is weak (0.45) because local_intensity is moderate-to-high
  (local_peak captures the speech portion of the transition frame)
- Per-frame, the voiced candidate clearly wins — this is not a Viterbi-only effect

**Modifications tested (none improve overall accuracy):**

| Method | Voicing MM | FP | FN | F0 P50 |
|--------|-----------|----|----|--------|
| Current (raw CC) | 13 | 11 | 2 | 0.84 Hz |
| + intensity adj (0.5r+0.5i) | 26 | 9 | 17 | 0.83 Hz |
| Windowed CC (Hanning) | 24 | 8 | 16 | 1.51 Hz |
| Windowed + intensity adj | 39 | 7 | 32 | 1.47 Hz |
| AC-style (window+autocorr+norm+adj) | 30 | 8 | 22 | 1.10 Hz |
| AC-style no intensity adj | 12 | 9 | 3 | 1.07 Hz |

All modifications trade false positives for many more false negatives. The current raw CC
approach gives the fewest total mismatches.

**Interpolation methods tested (no improvement):**
- Parabolic freq + raw strength (current): P50=0.835 Hz
- Sinc interp for both: P50=0.861 Hz
- Parabolic for both: P50=0.835 Hz
- Sinc depth=200: P50=0.861 Hz
- Sinc-refined lag via golden search: P50=0.908 Hz

### F0 Error Cascade from Voicing Mismatches

**Key finding:** The worst F0 errors are a direct consequence of voicing mismatches:

| Proximity to FP | Frames | P50 error | Max error |
|-----------------|--------|-----------|-----------|
| Within 1 frame | 5 | 4.5 Hz | 11.9 Hz |
| Within 3 frames | 15 | 6.3 Hz | 31.9 Hz |
| Within 10 frames | 50 | 2.4 Hz | 31.9 Hz |
| All 303 frames | 303 | 0.8 Hz | 31.9 Hz |

All 13 errors >5 Hz are within 6 frames of a voicing mismatch (except frames 250-253
which have a ~6 Hz systematic offset in a stable voiced region). The voicing false positives
cause our Viterbi to enter the voiced region earlier, choosing a path that favors different
F0 continuity than PM's path.

### Conclusion

The CC implementation is functionally correct. The remaining differences are inherent to
subtle algorithmic variations between our clean-room CC and Praat's CC (likely in
normalization, windowing, or candidate strength computation) that cannot be identified
without source code access. For reference: AC P50=0.014 Hz (60x better than CC's 0.84 Hz),
confirming the AC implementation is highly accurate while CC has inherent limitations.
