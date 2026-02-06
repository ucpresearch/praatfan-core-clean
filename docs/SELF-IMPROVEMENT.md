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
