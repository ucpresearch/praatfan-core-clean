# Resampler & Burg parity investigation

Branch: `burg-parity-investigation` (10 commits ahead of `main`, unmerged).

## Goal

Close the standalone Burg formant parity gap vs parselmouth before
attempting FormantPath. Working hypothesis going in (per
`docs/TRANSFERABLE_FINDINGS.md`): the resampler is the dominant gap
source, and Praat's `Sound: Resample` is a windowed-sinc interpolator.

## Starting state

`one_two_three_four_five.*` fixtures, 5 files, `To Formant (burg)` with
defaults `0.005, 5, 5500.0, 0.025, 50.0` — comparing every frame to
parselmouth.

```
F1: mean 6.06 Hz, p50 0.79, p95 19.5, p99 67.3, max 1689
F2: mean 14.07,   p50 1.06, p95 52.6, p99 280,   max 1145
F3: mean 18.08,   p50 1.87, p95 69.5, p99 382,   max 1408
B1: mean 12.13,   p99 192,  max 1424
```

p50 was tight; tails were brutal. Classic boundary/edge-frame signature
for some frames, formant-assignment-shift for others.

## What we tried

### 1. Bandwidth formula DP22 ✓ already correct

GPL doc raised this as an open question. Black-box test: synth signals
from known LPC poles, compare bandwidth.

**Result:** our existing `-log(|z|) * sample_rate / π` already matches
Praat's K=2 convention. No change needed.

Script: `scripts/test_bandwidth_formula.py`

### 2. Window function tuning ✓ α=12 already optimal

Swept Gaussian α ∈ {4, 6, 8, 10, 12, 14, 16, 20, 24}, with/without
edge-subtraction, plus Hann/Hamming/Blackman.

**Result:** α=12 wins by a wide margin. Hann/Hamming/Blackman are
20× worse. Window is not the gap.

Script: `scripts/tune_window.py`

### 3. Frame timing ✓ bit-exact

Compared frame timestamps directly: max diff 2.2e-16. Not the gap.

Script: `scripts/check_frame_times.py`

### 4. min_freq filter on extracted formants ✗ already optimal at 50

Diagnosed worst frame (`gain5.wav` at t=1.431, F1 err 1689 Hz):

- Our LPC root sits at 48.32 Hz → filtered out by `min_freq=50`
- Praat's LPC root sits at 50.4 Hz → passes
- All other formants then shift by one slot → F1 reports 1739
  (which IS Praat's F2)

Tested min_freq ∈ {0, 10, 20, 30, 40, 45, 48, 50, 55, 60} +
conditional rescue rules ("only allow <50 Hz when fewer than 5
formants found").

**Result:** 50 wins globally. Lower lets near-DC poles displace real
formants in many other frames; higher misses legitimate formants. The
2.1 Hz pole shift between us and Praat at this single frame is a
resampler-induced numerical disagreement, not a filter problem.

Scripts: `scripts/test_minfreq.py`, `scripts/test_minfreq_fine.py`,
`scripts/test_conditional_filter.py`

### 5. Direct windowed-sinc resampler ✗ broke Burg parity

Wrote `_resample_wsinc` matching Praat's documented algorithm (sinc ×
half-Hann, support `±(precision+0.5)*step` after Agent D's correction).
Kernel shape was confirmed via 200×-oversampled impulse-response
extraction + least-squares window fit (Hann MSE 1.5e-6; alternatives
≥10× worse).

Point-level match vs `parselmouth.call(Resample, ..., 50)`:
| precision | mean diff | max diff | Burg F1 mean |
|-----------|-----------|----------|--------------|
| 50        | 2.3e-4    | 4.0e-3   | **25.8**     |
| 500       | 9.2e-5    | 4.6e-4   | 18.2         |
| 1000      | 6.4e-5    | 2.1e-4   | 17.6         |
| 2000      | 5.4e-5    | 8.7e-5   | —            |

**The point-diff to Praat dropped 5–40× compared to scipy** — but
Burg parity got dramatically *worse*. Why:

- Our wsinc has finite kernel support (±109 input samples at prec=50).
- Praat's actual impulse response has a 1/d long tail extending
  thousands of samples (probed via `scripts/explain_wsinc_failure.py`).
  Confirmed: zeroing input samples after 2400 reduces silence-region
  output RMS from 5.2e-5 → 5.5e-8 (1000× drop), so silent regions are
  driven by content propagating from later in the signal.
- That ~5e-5 Nyquist-frequency baseline in silent regions is what
  Burg LPC fits in those frames. Without it, our wsinc outputs
  mathematically zero, Burg hits its `den < 1e-30` break, and pole
  configurations go chaotic.

Partitioned by frame energy (RMS percentile):
| region | frames | wsinc F1 | scipy F1 |
|--------|--------|----------|----------|
| voice  | 795    | 3.5      | 1.1      |
| mid    | 474    | 40.5     | 9.6      |
| silent | 316    | **60.0** | 6.9      |

wsinc wins on F2/F3 in voice frames (11.7 vs 18.5; 19.8 vs 24.8) but
loses catastrophically in mid/silent frames.

Scripts: `scripts/boundary_hypotheses.py`, `kernel_hypotheses.py`,
`timing_hypotheses.py`, `decisive_resample_probe*.py`,
`compare_wsinc_vs_scipy.py`, `explain_wsinc_failure.py`.

Four parallel agents explored boundary handling, kernel shape, sample
timing, and impulse-response probing. Outcome:
- timing formula `(m+0.5)*ratio − 0.5` confirmed
- boundary-depth variants tie within 0.3% (the GPL doc's asymmetric
  formula actually *hurts* under half-Hann)
- window-shape tweaks move point-diff <10%
- kernel is `sinc × Hann` with support `±(prec+0.5)*step` (Agent D)

### 6. FFT-convolution of our explicit Hann-windowed kernel ✗ same wall

Hypothesized: an infinite-support FFT-conv of our Hann-windowed sinc
would reproduce both Praat's main lobe AND its 1/d tail.

Implementation: `scipy.signal.fftconvolve` with kernel of
`window_factor × signal_length` half-width, then cubic-spline
interpolation at fractional output positions.

| window_factor | point mean | point max | Burg F1 mean | silent-rms |
|---------------|------------|-----------|--------------|------------|
| 0.5           | 4.7e-5     | 5.4e-4    | 19.7         | 4.1e-6     |
| 1.0           | 5.4e-5     | 5.5e-4    | 14.2         | 1.3e-5     |
| 2.0           | 6.0e-5     | 5.6e-4    | 12.1         | 2.3e-5     |
| 5.0           | —          | —         | 11.7         | ~3e-5      |
| 10.0          | —          | —         | **11.7** ⟂  | ~4e-5      |

Praat silent-rms ≈ 5.2e-5.

**Best point-level match achieved: 5× better than direct wsinc, 40×
better than scipy pow-2. Burg parity plateaus at 11.7 vs scipy
pow-2's 4.79.** Even with infinite-support Hann, the silent-region
baseline never reaches Praat's level. The remaining gap is the
**spectrum shape near Nyquist**: scipy uses a brick-wall LPF
(rectangular in frequency → unwindowed sinc in time, infinite support
naturally) — our Hann-tapered kernel has smooth rolloff. Burg cares
more about that spectrum-shape near Nyquist than about point-level
accuracy.

Script: `scripts/try_fftconv_kernel.py`

### 7. FFT-padding strategies ✓ pow-2 is the win

Compared 5x linear pad vs no pad vs centered pad vs reflect/edge/symmetric
pads vs **next pow-2 above 2× input length**.

Pow-2 padding is **quantitatively closer to Praat in every frequency
band** (per-band ratio `pow-2_err / 5x_err` between 0.62 and 0.75).
Most likely explanation: Praat also uses pow-2 FFT internally; our
pow-2-padded input goes through the same-size FFT, while 5×-linear pad
forces mixed-radix FFT with slightly different numerical behavior.

Tested time-correction (fractional shift to cancel 0.27-sample delay
vs Praat) and exact-multiple padding (rate drift = 0). Both make Burg
parity *worse*. The pow-2 win isn't about timing or rate drift —
scipy's implicit timing already matches Burg's frame grid expectation.

Scripts: `scripts/spectral_gap.py`, `test_padding_phase.py`,
`test_pow2_sizes.py`, `test_delay_correction.py`, `why_pow2.py`.

## What we kept

`src/praatfan/formant.py`:

- **`_resample`** uses scipy FFT resample with input padded to next
  power-of-2 above 2× input length. Reproduces Praat's long impulse
  response and silent-region Nyquist baseline that Burg depends on.
- **`_resample_wsinc`** retained as a refined clean-room reference
  implementation (kernel = sinc × Hann, support `±(precision+0.5)*step`,
  timing `(m+0.5)*ratio − 0.5`). Matches Praat's *kernel shape* — useful
  for future integer-ratio paths or if we ever pair it with FFT-domain
  filtering to get the spectral shape right too.

Scripts under `scripts/`: full hypothesis-test harness, baseline,
regression checks.

## Current Burg parity (5-fixture aggregate, 1585 frames)

```
       before    after   delta
F1 m   6.06      4.79    -21%
F1 p95 19.5      15.2    -22%
F1 p99 67        38.7    -42%
F1 max 1689      1689    0    (single-frame pole-shift, resampler-driven)
F2 m   14.07     12.81   -9%
F2 p99 280       277     -1%
F3 m   18.08     16.97   -6%
F3 p99 382       382     0
B1 m   12.13     9.94    -18%
B1 p99 192       124     -35%
```

## What we ruled out

| source              | gap?       | evidence                              |
|---------------------|------------|---------------------------------------|
| bandwidth formula   | no         | already K=2; matches Praat            |
| window function     | no         | α=12 already optimal                  |
| frame timing        | no         | bit-exact                             |
| min_freq filter     | no         | 50 globally optimal                   |
| pre-emphasis order  | no         | resample-then-preemph is right        |
| direct wsinc        | won't help | same kernel shape but no long tail    |
| boundary depth      | <0.3%      | the doc's asymmetric formula hurts    |
| sample timing       | <0.1%      | `(m+0.5)*ratio−0.5` confirmed         |

## Where we're standing

- Resampler is the only knob that's still moving the numbers.
- scipy FFT pow-2 is the best `_resample` we've found (F1 p99 38.7,
  21% better than baseline).
- The remaining gap is **structural**: Praat's resampler matches
  scipy's FFT brick-wall behavior near Nyquist (long 1/d kernel tail)
  but with finer phase. We can't replicate the exact phase without
  reverse-engineering Praat's internal FFT.
- The single max F1 outlier (1689 Hz) is a frame where our pole sits
  at 48.3 Hz vs Praat's 50.4 Hz — same physical near-DC resonance,
  ~2 Hz angular shift from resample differences. Not closable without
  matching Praat's phase exactly. Could be papered over with a
  formant-tracking heuristic (continuity across frames) but that's
  outside the clean-room recipe.
- Other algorithms unaffected: Pitch / Intensity / HNR / Spectrogram
  are bit-identical with old vs new resampler when run on native-rate
  audio (verified). On resample-then-analyze paths they're tied or
  marginally better with pow-2.

## Open paths if we want to keep pushing

1. **FFT-domain brick-wall + windowed-sinc hybrid** — apply a
   scipy-style brick-wall LPF (gives the long tail) then an explicit
   windowed-sinc fractional-delay step (gives the precise main lobe).
   Two stages. Untested.
2. **Match Praat's exact FFT internals** — likely Praat uses a
   specific FFT library (probably from Numerical Recipes or its own
   port). Without source access, we'd need to probe its rounding
   behavior on synthesised inputs. High effort, low expected payoff.
3. **Single-frame post-processing** — formant-continuity tracker that
   detects shift-by-one slots when an F1 candidate sits just below
   the 50 Hz floor. Fixes the t=1.431 max outlier; doesn't help the
   bulk distribution.
4. **Move on to FormantPath** — per `TRANSFERABLE_FINDINGS.md`'s
   impact ordering, FormantPath Viterbi gains are 10× larger than
   anything left to extract from standalone Formant.

## Branch contents

10 commits on `burg-parity-investigation`, none on `main`:

```
5fd8efc Full-algorithm parity baseline + resampler regression check
2eca1f7 FFT-convolution of Hann-windowed-sinc kernel hits same Burg-parity wall
7ef8093 Refined wsinc kernel per Agent D: sinc × Hann over ±(prec+0.5)×step
e51e601 Explain why wsinc broke: phase + silence mechanics
2cd804a Why pow-2 wins: output is genuinely closer to Praat at sample level
ba10d91 Time correction isn't the fix — scipy's alignment already matches Burg
c2b1f78 Resample: pow-2 FFT size cuts F1 mean 6.06→4.79, p99 67→39
1eb906d Filter tuning dead end: the max error is resample-driven
252cbe6 Window α=12 optimal, frame timing bit-exact
0ab266b Burg parity investigation: baseline, bandwidth check, resampler study
```

Single-line `_resample` change in `src/praatfan/formant.py`. The
`_resample_wsinc` function and ~20 investigation scripts under
`scripts/` are net-new and can be merged or kept on the branch.
