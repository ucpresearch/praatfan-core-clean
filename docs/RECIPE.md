# Detailed Implementation Recipe

Comprehensive guide for clean room implementation of Praat's acoustic analysis algorithms. This document expands on `docs/RECIPE.md` with detailed formulas, edge cases, and implementation notes.

**Clean Room Principle:** This document distinguishes between:
- ✅ **DOCUMENTED** - Explicitly stated in papers or manual
- 🔬 **DETERMINE** - Has a specification; find the exact value via black-box testing
- 📚 **STANDARD** - Standard technique from cited reference (e.g., Numerical Recipes)

## Table of Contents

1. [General Principles](#general-principles)
2. [Phase 1: Foundation](#phase-1-foundation)
3. [Phase 2: Spectrum & Spectral Moments](#phase-2-spectrum--spectral-moments)
4. [Phase 3: Intensity](#phase-3-intensity)
5. [Phase 4: Pitch](#phase-4-pitch)
6. [Phase 5: Harmonicity](#phase-5-harmonicity)
7. [Phase 6: Formant](#phase-6-formant)
8. [Phase 7: Spectrogram](#phase-7-spectrogram)
9. [Appendix: Resampling](#appendix-resampling)

---

## General Principles

### Algorithm Dependencies - Implement in This Order

Some algorithms depend on others. Implement and validate dependencies first.

```
┌─────────────────────────────────────────────────────────────────────┐
│  FOUNDATION LAYER (implement first)                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Spectrum ─────────► Spectral Moments                               │
│      │                                                              │
│      └────────────► Spectrogram (windowed Spectrum)                 │
│                                                                     │
│  Intensity (independent)                                            │
├─────────────────────────────────────────────────────────────────────┤
│  PITCH LAYER (implement after foundation)                           │
├─────────────────────────────────────────────────────────────────────┤
│  Pitch ────────────► Harmonicity AC (derived from Pitch AC)         │
│      │                                                              │
│      └────────────► Harmonicity CC (derived from Pitch CC)          │
├─────────────────────────────────────────────────────────────────────┤
│  COMPLEX LAYER (can be parallel with Pitch)                         │
├─────────────────────────────────────────────────────────────────────┤
│  Formant (independent but complex - many sub-steps)                 │
└─────────────────────────────────────────────────────────────────────┘
```

**Critical dependency:** Harmonicity is NOT a separate algorithm. It is computed directly from Pitch's correlation strength `r` using: `HNR = 10 × log₁₀(r / (1-r))` (Praat manual, Harmonicity.html). If Pitch is wrong, Harmonicity will be wrong. If Pitch strength values are correct, Harmonicity is trivial.

### Before Implementing Any Algorithm

1. Read the relevant documentation in `RESOURCES.md`
2. Identify constants that need to be determined via testing
3. Implement and validate against parselmouth output

### Determinable Constants

When documentation gives a **specification** but not an exact **value**:

| Specification | How to Determine |
|---------------|------------------|
| "sidelobes below -120 dB" | Compute FFT of window, find coefficient achieving spec |
| "Kaiser-20; -190 dB sidelobes" | β ≈ 20 (from name), verify sidelobes via FFT |
| Frame timing approach | Test left-aligned vs centered vs right-aligned |
| DC removal method | Test weighted vs unweighted mean |

### Standard Techniques (Implicit in Citations)

When Praat manual cites "Numerical Recipes" or similar, these standard techniques are included:

- **Root finding:** Companion matrix eigenvalues (Numerical Recipes Ch. 9.5)
- **Root polishing:** Newton-Raphson iteration (Numerical Recipes)
- **Polynomial evaluation:** Horner's method (standard)
- **Bessel functions:** Series expansion (standard)
- **Burg's algorithm:** Childers (1978), pp. 252-255

### Frame Timing Convention

✅ **DOCUMENTED:** Number of frames formula exists in various manual pages.

🔬 **DETERMINE:** The exact first-frame time (t1) is not documented. Standard STFT literature offers:

- **Left-aligned:** `t₁ = window/2`
- **Centered:** `t₁ = midpoint - (N-1)×step/2`
- **Right-aligned:** `t₁ = duration - window/2 - (N-1)×step`

**Approach:** Implement all three options, compare against parselmouth output to determine which Praat uses.

**General formula pattern:**
```
numberOfFrames = floor((duration - windowDuration) / timeStep) + 1
t1 = ???  // Determine via black-box testing
```

---

## Phase 1: Foundation

### 1.1 Sound Type

✅ **DOCUMENTED:** Standard audio representation

```rust
struct Sound {
    samples: Vec<f64>,      // Mono samples
    sample_rate: f64,       // Hz
}
```

**Constraint:** Mono only. For multi-channel files, require explicit channel selection or error.

**Time conventions (standard DSP):**
- `dx` = sample period = 1/sample_rate
- `nx` = number of samples
- `x1` = time of first sample (typically 0.5 × dx, centered on first sample)

### 1.2 FFT Wrapper

✅ **DOCUMENTED:** Standard DFT definition

Use `rustfft` crate or similar. Verify normalization conventions.

### 1.3 Window Functions

#### Hanning Window

✅ **DOCUMENTED:** Standard formula
```
w[i] = 0.5 - 0.5 × cos(2π × i / (N-1))
```

#### Gaussian Window (for pitch, formant, spectrogram)

✅ **DOCUMENTED specification:** "sidelobes below -120 dB"

🔬 **DETERMINE:** The exact coefficient in the Gaussian formula that achieves -120 dB sidelobes.

**Approach:**
1. Start with standard Gaussian: `w[i] = exp(-α × (i - center)² / N²)`
2. Vary α and compute FFT to find sidelobe level
3. Find the α value that achieves ≤ -120 dB sidelobes
4. Verify against parselmouth output

**From Boersma (1993) postscript:**
```
w(t) = (exp(-12 × (t/T - 0.5)²) - e^(-12)) / (1 - e^(-12))
```
where T is the physical window duration and t ∈ [0, T]. This provides a starting point.

#### Kaiser-Bessel Window (for intensity)

✅ **DOCUMENTED specification:** "Kaiser-20; sidelobes below -190 dB"

🔬 **DETERMINE:** The exact β parameter. "Kaiser-20" suggests β ≈ 20.

**Approach:**
1. Start with β = 20
2. Compute Kaiser window and its FFT
3. Verify sidelobes ≤ -190 dB
4. Adjust if necessary and verify against parselmouth

📚 **STANDARD:** Bessel I₀ function - use standard series expansion from numerical analysis texts.

---

## Phase 2: Spectrum & Spectral Moments

### 2.1 Spectrum

✅ **DOCUMENTED:** Praat manual - fully specified

**Fourier transform definition:**
```
X(f) = ∫₀ᵀ x(t) e^(-2πift) dt
```

**Discrete version:**
```
X[k] = Σₙ x[n] × e^(-2πikn/N) × Δt
```

✅ **DOCUMENTED:** Multiply FFT output by `Δt` (sample period) - this follows from the integral definition.

**Storage:** Only positive frequencies (0 to Nyquist). Negative frequencies are conjugate symmetric for real signals.

### 2.2 Spectral Moments

✅ **DOCUMENTED:** Praat manual - complete formulas (100% documented)

**Centre of gravity:**
```
f_c = ∫ f × |S(f)|^p df / ∫ |S(f)|^p df
```

**Central moment n:**
```
μ_n = ∫ (f - f_c)^n × |S(f)|^p df / ∫ |S(f)|^p df
```

**Derived measures:**
- Standard deviation = √(μ₂)
- Skewness = μ₃ / μ₂^1.5
- Kurtosis = μ₄ / μ₂² - 3 (excess kurtosis)

### 2.3 Band Energy

✅ **DOCUMENTED:** `E = ∫_{f_min}^{f_max} |S(f)|² df`

🔬 **DETERMINE:** Whether a factor of 2 is needed for one-sided spectrum (to account for negative frequencies).

**Approach:** Compare computed band energy against parselmouth for a known signal.

---

## Phase 3: Intensity

### 3.1 Algorithm Overview

✅ **DOCUMENTED:** Praat manual

> "The values in the sound are first squared, then convolved with a Gaussian analysis window (Kaiser-20; sidelobes below -190 dB). The effective duration of this analysis window is 3.2 / pitchFloor."

### 3.2 Window Type Clarification

The documentation says "Gaussian analysis window (Kaiser-20)".

🔬 **DETERMINE:** Whether this is a Gaussian window, a Kaiser window, or something else that achieves the Kaiser-20 specification.

**Approach:** Test both window types against parselmouth output.

### 3.3 Physical vs Effective Duration

✅ **DOCUMENTED:** Effective duration = 3.2 / pitch_floor

🔬 **DETERMINE:** The ratio of physical to effective duration. For Gaussian-like windows, physical is typically 2× effective.

**Approach:** Test physical = 2× effective, verify against parselmouth.

### 3.4 DC Removal Method

✅ **DOCUMENTED:** From Praat manual, Intro 6.2:
> "first subtracting the mean pressure around this point, and then applying the Gaussian window"

The order is: subtract mean THEN apply window.

🔬 **DETERMINE:** Whether the mean is weighted or unweighted.

**Options:**
1. **Unweighted mean:** `mean = sum(samples) / n`
2. **Weighted mean:** `mean = sum(samples × window) / sum(window)`

**Approach:** Test both against parselmouth. The order "subtract THEN window" suggests unweighted, but verify empirically.

### 3.5 Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Effective window duration | 3.2 / min_pitch | ✅ Documented |
| Physical window duration | 🔬 | Determine (likely 2× effective) |
| Time step (default) | 0.8 / min_pitch | ✅ Documented (1/4 of effective) |
| Kaiser β | 🔬 | Determine (approximately 20) |

### 3.6 dB Conversion

✅ **DOCUMENTED:** Standard formula
```
intensity_db = 10 × log₁₀(mean_square / reference²)
```

🔬 **DETERMINE:** The reference pressure value (standard is 2×10⁻⁵ Pa, but verify).

---

## Phase 4: Pitch (Boersma 1993)

### 4.1 Primary Source

✅ **DOCUMENTED:** Boersma, P. (1993). "Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound."

This paper provides nearly complete algorithm specification.

### 4.2 Algorithm Overview

✅ **DOCUMENTED in paper:**
1. Preprocessing (optional soft upsampling)
2. Global peak computation
3. Frame-by-frame autocorrelation analysis
4. Sinc interpolation for sub-sample precision
5. Candidate scoring
6. Viterbi path finding

### 4.3 Autocorrelation Normalization (Eq. 9)

✅ **DOCUMENTED:**
```
r_x(τ) ≈ r_a(τ) / r_w(τ)
```

Divide the windowed signal's autocorrelation by the window's autocorrelation.

### 4.4 Window Autocorrelation

✅ **DOCUMENTED for Hanning window (Eq. 8):**
```
r_w(τ) = (1 - |τ|/T) × (2/3 + 1/3×cos(2πτ/T)) + (1/2π) × sin(2π|τ|/T)
```

For Gaussian window: compute numerically or derive analytically.

### 4.5 Sinc Interpolation (Eq. 22)

✅ **DOCUMENTED:**
```
sinc_window(φ, N) = [sin(πφ) / (πφ)] × [0.5 + 0.5×cos(πφ/N)]
```

✅ **DOCUMENTED:** N = min(500, samples_to_half_window) - paper gives this bound.

🔬 **DETERMINE:** The actual interpolation depth used in practice (paper says up to 500, but smaller values may be used).

### 4.6 Candidate Strength Formulas

✅ **DOCUMENTED (Eq. 23) - Unvoiced:**
```
R_unvoiced = voicingThreshold + max(0, 2 - (localPeak/globalPeak)/silenceThreshold) × (1 + voicingThreshold)
```

✅ **DOCUMENTED (Eq. 24) - Voiced:**
```
R_voiced = r(τ_max) - octaveCost × log₂(minimumPitch × τ_max)
```

### 4.7 Viterbi Transition Costs (Eq. 27)

✅ **DOCUMENTED:**
```
transitionCost(F1, F2) =
  0                                    if F1=0 AND F2=0 (both unvoiced)
  voicedUnvoicedCost                   if F1=0 XOR F2=0 (voicing change)
  octaveJumpCost × |log₂(F1/F2)|       if F1≠0 AND F2≠0 (pitch jump)
```

✅ **DOCUMENTED (manual):** Time step correction:
> "corrected for the time step: multiply by 0.01 s / TimeStep"

### 4.8 Parameters

| Parameter | Default | Source |
|-----------|---------|--------|
| time_step | 0.0 (auto: 0.75/floor) | ✅ Documented |
| pitch_floor | 75 Hz | ✅ Documented |
| pitch_ceiling | 600 Hz | ✅ Documented |
| max_candidates | 15 | ✅ Documented |
| voicing_threshold | 0.45 | ✅ Documented |
| silence_threshold | 0.03 | ✅ Documented |
| octave_cost | 0.01 | ✅ Documented |
| octave_jump_cost | 0.35 | ✅ Documented |
| voiced_unvoiced_cost | 0.14 | ✅ Documented |

### 4.9 Window Duration

✅ **DOCUMENTED:** Window = 3 periods = 3 / pitch_floor for autocorrelation method.

🔬 **DETERMINE:** Physical vs effective window ratio (likely 2× for Gaussian).

### 4.10 Gaussian Window Formula

✅ **DOCUMENTED in Boersma (1993) postscript:**
```
w(t) = (exp(-12 × (t/T - 0.5)²) - e^(-12)) / (1 - e^(-12))
```

This provides the starting point; verify achieves -120 dB sidelobes.

---

## Phase 5: Harmonicity (HNR)

### ⚠️ CRITICAL: Harmonicity Depends on Pitch

**Do NOT implement Harmonicity as a standalone algorithm.** Both AC and CC harmonicity methods use the Pitch algorithm internally and extract HNR from the correlation strength.

**Implementation approach:**
1. Implement Pitch (AC method) first → validate fully
2. Harmonicity AC = apply HNR formula to Pitch AC strengths
3. Implement Pitch (CC method) → validate fully
4. Harmonicity CC = apply HNR formula to Pitch CC strengths

If your Pitch implementation produces correct `strength` values, Harmonicity is just one line of code.

### 5.1 HNR Formula

✅ **DOCUMENTED (Praat manual, Harmonicity.html):**
> "if 99% of the energy of the signal is in the periodic part, and 1% is noise, the HNR is 10*log10(99/1) = 20 dB"

This generalizes to:
```
HNR (dB) = 10 × log₁₀(r / (1 - r))
```

where `r` is the normalized autocorrelation (correlation strength) at the detected pitch period.

### 5.2 Practical Bounds

🔬 **DETERMINE:** How Praat handles r → 0 and r → 1 (avoiding infinity).

**Approach:** Test with pure tones (r ≈ 1) and noise (r ≈ 0) to see what values Praat returns. Some clamping must occur.

### 5.3 Unvoiced Frame Handling

🔬 **DETERMINE:** What value is returned for unvoiced frames.

**Approach:** Check parselmouth output for unvoiced regions.

### 5.4 AC Method

✅ **DOCUMENTED (Praat manual, Sound: To Harmonicity (ac)):**
> "The algorithm performs an acoustic periodicity detection on the basis of an accurate autocorrelation method, as described in Boersma (1993)."

This means: run Pitch with AC method, extract strength values, apply HNR formula.

✅ **DOCUMENTED:** Parameters include `periods_per_window` (4.5 standard, 6.0 for high precision).

### 5.5 CC Method

✅ **DOCUMENTED (Praat manual, Sound: To Harmonicity (cc)):**
The CC method uses "forward cross-correlation" internally via the Pitch algorithm.

**Implementation:** Run Pitch with CC method (method=3, FCC_ACCURATE), extract strength values, apply HNR formula.

📚 **STANDARD:** Normalized cross-correlation formula:
```
r[τ] = Σᵢ x[i] × y[i+τ] / √(Σᵢ x[i]² × Σᵢ y[i+τ]²)
```

### 5.6 Debugging Harmonicity

**If HNR values don't match parselmouth:**

1. **First check:** Does your Pitch F0 match? If not, fix Pitch first.
2. **Second check:** Does your Pitch `strength` match? Query with `Get strength in frame`.
3. **Third check:** Only if F0 and strength both match, check HNR formula and clamping.

The HNR formula is trivial. 99% of harmonicity bugs are actually Pitch bugs.

### 5.8 Parameters

| Parameter | Default | Source |
|-----------|---------|--------|
| time_step | 0.01 s | ✅ Documented |
| min_pitch | 75 Hz | ✅ Documented |
| silence_threshold | 0.1 | ✅ Documented |
| periods_per_window | 4.5 (AC, minimum 3.0), 1.0 (CC) | ✅ Documented |

---

## Phase 6: Formant (Burg LPC)

### 6.1 Algorithm Overview

✅ **DOCUMENTED:** Praat manual

1. Resample to 2 × formant_ceiling Hz
2. Pre-emphasize
3. For each frame: window, LPC, find roots, extract formants

### 6.2 Window Length Parameter

✅ **DOCUMENTED:**
> "The actual length is twice this value, because Praat uses a Gaussian-like analysis window"

So `window_length = 0.025` means **0.050 seconds actual window**.

### 6.3 Pre-emphasis Filter

✅ **DOCUMENTED:**
```
α = exp(-2π × F_pre × Δt)
x'[i] = x[i] - α × x[i-1]
```

### 6.4 Resampling

✅ **DOCUMENTED:** Resample to `2 × max_formant_hz`

See Appendix: Resampling for interpolation approach.

### 6.5 Gaussian Window

✅ **DOCUMENTED specification:** "sidelobes below -120 dB"

🔬 **DETERMINE:** Exact Gaussian coefficient (same as pitch - see Section 1.3).

### 6.6 Burg's Algorithm

📚 **STANDARD:** Childers (1978), pp. 252-255 - explicitly cited by Praat manual.

Burg's method estimates LPC coefficients by minimizing forward and backward prediction errors. The algorithm is fully specified in Childers.

**LPC order:** `2 × max_formants` (e.g., 10 for 5 formants) - ✅ DOCUMENTED

### 6.7 Polynomial Construction

📚 **STANDARD:** Convert LPC coefficients to polynomial for root finding.

LPC equation: `x[n] = Σ(k=1 to p) a[k] × x[n-k]`

Polynomial: `z^p - a[1]z^(p-1) - ... - a[p] = 0`

🔬 **DETERMINE:** The exact sign convention (positive or negative coefficients).

### 6.8 Root Finding

📚 **STANDARD:** Numerical Recipes, Chapter 9.5 - explicitly cited.

**Companion matrix eigenvalues** is a standard technique for polynomial root finding.

### 6.9 Root Polishing

📚 **STANDARD:** Newton-Raphson iteration from Numerical Recipes.

🔬 **DETERMINE:** Number of iterations needed for convergence.

**Approach:** Start with standard Newton-Raphson, increase iterations until formant accuracy matches parselmouth.

### 6.10 Unstable Root Handling

📚 **STANDARD:** Roots outside the unit circle indicate instability. Standard LPC practice is to reflect them inside.

🔬 **DETERMINE:** The exact reflection formula.

**Options:**
1. `z → z / |z|²` (preserves angle)
2. `z → 1 / z*` (conjugate inverse)
3. `z → z / |z|` (move to unit circle)

**Approach:** Test each against parselmouth.

### 6.11 Extracting Formants from Roots

📚 **STANDARD (Markel & Gray 1976):**
```
frequency = |arg(z)| × sample_rate / (2π)
bandwidth = -ln|z| × sample_rate / π
```

### 6.12 Formant Filtering

✅ **DOCUMENTED:**
> "All formants below 50 Hz and all formants above Formant ceiling minus 50 Hz, are therefore removed."

### 6.13 Frame Sample Extraction

🔬 **DETERMINE:** The exact sample extraction formula for each frame.

**Approach:** For a frame at time t, test different windowing approaches:
1. Centered at t
2. Left-aligned at t
3. Various half-window offset conventions

Compare frame-by-frame against parselmouth.

### 6.14 Parameters

| Parameter | Default | Source |
|-----------|---------|--------|
| time_step | 0.0 (auto: 25% of window) | ✅ Documented |
| max_formants | 5 | ✅ Documented |
| max_formant_hz | 5500 Hz | ✅ Documented |
| window_length | 0.025 s (actual = 0.050 s) | ✅ Documented |
| pre_emphasis_from | 50 Hz | ✅ Documented |

---

## Phase 7: Spectrogram

### 7.1 Algorithm Overview

✅ **DOCUMENTED:** Standard STFT

For each frame:
1. Extract windowed segment
2. Apply window function
3. Compute FFT
4. Store power spectrum

### 7.2 Window Duration

✅ **DOCUMENTED:**
> "The Gaussian window... analyzes a factor of 2 slower than the other window shapes, because the analysis is actually performed on twice as many samples per frame."

So for Gaussian: physical window = 2× effective.

### 7.3 Gaussian Bandwidth

✅ **DOCUMENTED:**
```
Gaussian -3dB bandwidth = 1.2982804 / window_length
```

### 7.4 Oversampling Limits

✅ **DOCUMENTED:**
- Time step: never less than `1/(8√π) × window_length`
- Frequency step: never less than `(√π)/8 / window_length`

### 7.5 Parameters

| Parameter | Default | Source |
|-----------|---------|--------|
| window_length | 0.005 s | ✅ Documented |
| max_frequency | 5000 Hz | ✅ Documented |
| time_step | 0.002 s | ✅ Documented |
| frequency_step | 20 Hz | ✅ Documented |
| window_shape | Gaussian | ✅ Documented |

### 7.6 Output Units

🔬 **DETERMINE:** The exact output units (likely Pa²/Hz for power spectral density).

---

## Appendix: Resampling

### A.1 Overview

✅ **DOCUMENTED:** Praat manual

> "If Precision is greater than 1, the method is sin(x)/x ('sinc') interpolation, with a depth equal to Precision"

### A.2 Sinc Interpolation

📚 **STANDARD:** Windowed sinc interpolation is standard DSP.

✅ **DOCUMENTED (Boersma 1993, Eq. 22):** The windowed sinc formula:
```
sinc_window(φ, N) = [sin(πφ) / (πφ)] × [0.5 + 0.5×cos(πφ/N)]
```

This formula is documented for autocorrelation interpolation but applies generally.

### A.3 Anti-aliasing

✅ **DOCUMENTED:**
> "If Sampling frequency is less than the sampling frequency of the selected sound, an anti-aliasing low-pass filtering is performed prior to resampling"

🔬 **DETERMINE:** The exact lowpass filter implementation (FFT-based vs time-domain).

### A.4 Time Alignment

🔬 **DETERMINE:** How output samples are aligned in time.

**Approach:** Compare sample positions between Praat-resampled and your implementation.

---

## Validation Checklist

### Debugging Methodology

When validating against parselmouth, follow this order:

**Step 1: Match the count**

First, ensure you produce the **same number of values** (frames) as parselmouth. If you have 158 frames but parselmouth has 159, something is wrong with your frame timing formula. Do not proceed to value comparison until the counts match.

**Step 2: Match individual values, not averages**

Precision means that **for each time point**, your value should be close to the parselmouth value at that same time point.

- ✅ **Correct:** For frame at t=0.125s, your F1=523.4 Hz vs parselmouth F1=523.5 Hz (error: 0.1 Hz)
- ❌ **Wrong:** "Average F1 error across all frames is 0.5 Hz" (this hides systematic errors)

Compare frame-by-frame. If frame 47 has a large error but others are fine, investigate frame 47 specifically. Averaging can mask problems where some frames are completely wrong while others compensate.

**Step 3: Verify intermediate values before final output**

For multi-step algorithms, verify intermediate values when parselmouth exposes them:

| Algorithm | Intermediate values to check |
|-----------|------------------------------|
| **Pitch** | 1. Voicing decision, 2. Strength (`Get strength in frame`), 3. F0 |
| **Formant** | 1. Formant count per frame, 2. Individual F1/F2/F3, 3. Bandwidths |
| **Harmonicity** | Validate Pitch first! HNR = trivial formula applied to pitch strength |
| **Spectrum** | 1. Bin count/dx, 2. Band energies, 3. Moments |

**Critical for Harmonicity:** If HNR is wrong, don't debug HNR—debug Pitch. The HNR formula is trivial.

**Step 4: Investigate outliers**

If 95% of frames match but 5% have large errors, those outliers often reveal edge cases:
- Frames near signal boundaries
- Frames during unvoiced regions
- Frames at phoneme transitions
- Numerical edge cases (very small denominators, etc.)

### For Each Algorithm

- [ ] Frame count matches exactly
- [ ] Frame times match (same t1 and time step)
- [ ] Values within tolerance **at each individual frame**
- [ ] Edge cases handled (silence, very short signals)
- [ ] Parameters produce same effect as Praat defaults

### Tolerance Targets

| Algorithm | Target |
|-----------|--------|
| Formant | F1, F2, F3 within 1 Hz |
| Pitch | Within 0.01 Hz |
| Intensity | Within 0.001 dB |
| Spectrum | Relative error < 1e-10 |
| Spectral moments | Exact match |
| HNR | Within 0.01 dB |

### Common Issues to Investigate via Black-Box Testing

1. **Frame timing:** Test left/center/right alignment options
2. **Window coefficients:** Test range of values achieving sidelobe specification
3. **DC removal:** Test weighted vs unweighted mean
4. **Physical vs effective duration:** Test 2× ratio assumption
5. **Root polishing:** Test with increasing iterations until accuracy achieved
6. **Sign conventions:** Test +/- variations in polynomial construction
7. **Reflection formulas:** Test different unstable root handling methods

---

## Decision Points: Explicit Choices for Black-Box Testing

This section enumerates **every specific choice** that must be determined via black-box testing. For each decision point, we list all reasonable options, how to test them, and what symptoms indicate the wrong choice.

---

### DP1: Frame Timing (t1)

**Applies to:** All frame-based analyses (Pitch, Formant, Intensity, Harmonicity, Spectrogram)

**Specification:** Documentation gives number of frames formula but not the first frame time.

**Options to test:**

| Option | Formula | Description |
|--------|---------|-------------|
| A. Left-aligned | `t1 = windowDuration / 2` | First frame at half-window from start |
| B. Centered | `t1 = (duration - (nFrames-1) × timeStep) / 2` | Frames symmetric around signal midpoint |
| C. Right-aligned | `t1 = duration - windowDuration/2 - (nFrames-1) × timeStep` | Last frame at half-window from end |

**How to test:**
1. Compute t1 using each formula
2. Compare your frame times against parselmouth frame times
3. The correct option will have frame times matching within floating-point tolerance

**Symptoms of wrong choice:**
- Frame count matches but values are systematically offset
- First/last frames have large errors, middle frames are fine
- Values seem "shifted" by one or more frames

---

### DP2: Gaussian Window Coefficient

**Applies to:** Pitch, Formant, Spectrogram (any algorithm using Gaussian window)

**Specification:** "sidelobes below -120 dB"

**Options to test:**

| Option | Formula | Notes |
|--------|---------|-------|
| A. α = 12 | `(exp(-12 × (t/T - 0.5)²) - e^(-12)) / (1 - e^(-12))` | From Boersma (1993) postscript |
| B. α = 48 | `(exp(-48 × ((i-mid)/N)²) - e^(-12)) / (1 - e^(-12))` | Different normalization |
| C. Find empirically | Vary α, compute FFT, measure sidelobes | Any α achieving -120 dB is valid |

**How to test:**
1. Generate window with each α value
2. Compute FFT (zero-pad heavily for resolution)
3. Measure max sidelobe relative to main lobe
4. Verify sidelobes ≤ -120 dB
5. Compare analysis output against parselmouth

**Symptoms of wrong choice:**
- Spectral leakage artifacts
- Slightly different formant/pitch values across all frames
- Values close but not within tolerance

**Note:** Options A and B may be equivalent under different normalization conventions. The key is achieving the -120 dB specification.

---

### DP3: Kaiser Window β (for Intensity)

**Applies to:** Intensity analysis

**Specification:** "Kaiser-20; sidelobes below -190 dB"

**Options to test:**

| Option | β value | Notes |
|--------|---------|-------|
| A. β = 20 | Exactly 20 | Literal interpretation of "Kaiser-20" |
| B. β = 2π² ≈ 19.74 | ~19.74 | Mathematical derivation |
| C. β = 2π² + 0.5 ≈ 20.24 | ~20.24 | Adjusted for -190 dB |

**How to test:**
1. Generate Kaiser window with each β
2. Compute FFT, measure sidelobes
3. Verify sidelobes ≤ -190 dB
4. Compare intensity values against parselmouth

**Symptoms of wrong choice:**
- Intensity values differ by small constant offset
- Ripple in intensity contour that shouldn't be there

---

### DP4: Physical vs Effective Window Duration

**Applies to:** All windowed analyses

**Specification:** Documentation often gives "effective" duration; physical may be 2×.

**Options to test:**

| Option | Physical duration | Notes |
|--------|-------------------|-------|
| A. 1× | physical = effective | Window parameter is actual duration |
| B. 2× | physical = 2 × effective | Common for Gaussian windows |

**How to test:**
1. Compute number of samples in window using each option
2. Compare frame count and values against parselmouth

**Symptoms of wrong choice:**
- Frame count mismatch
- Values systematically wrong (using wrong amount of signal)

**Known from documentation:**
- Formant: "actual length is twice this value" → **Option B confirmed**
- Spectrogram with Gaussian: "twice as many samples" → **Option B confirmed**
- Intensity: effective = 3.2/pitch_floor, physical likely 2× → **Test Option B first**

---

### DP5: DC Removal Method (for Intensity)

**Applies to:** Intensity analysis

**Specification:** "subtracting the mean... then applying the window"

**Options to test:**

| Option | Method | Formula |
|--------|--------|---------|
| A. Unweighted mean | Subtract simple mean before windowing | `mean = Σx / n` |
| B. Weighted mean | Subtract window-weighted mean | `mean = Σ(x × w) / Σw` |
| C. No DC removal | Don't subtract mean | Just window and square |

**How to test:**
1. Implement intensity with each DC removal method
2. Compare frame-by-frame against parselmouth
3. Test with signals that have DC offset

**Symptoms of wrong choice:**
- Intensity values differ, especially for signals with DC offset
- Constant offset in dB values

**Hint:** Documentation says "subtracting... then applying" which suggests Option A (unweighted), since weighted mean requires the window to be applied first.

---

### DP6: LPC Polynomial Sign Convention

**Applies to:** Formant analysis

**Specification:** Standard LPC equation, but sign convention varies in literature.

**Options to test:**

| Option | Polynomial construction |
|--------|------------------------|
| A. Negative | `poly[i+1] = -coeffs[i]` |
| B. Positive | `poly[i+1] = +coeffs[i]` |

**How to test:**
1. Build polynomial with each sign convention
2. Find roots
3. Extract formants
4. Compare against parselmouth

**Symptoms of wrong choice:**
- Completely wrong formant values
- Roots outside expected frequency range
- Negative frequencies

---

### DP7: Unstable Root Reflection

**Applies to:** Formant analysis (LPC roots outside unit circle)

**Specification:** Standard practice to reflect unstable roots inside unit circle.

**Options to test:**

| Option | Formula | Effect |
|--------|---------|--------|
| A. Preserve angle | `z → z / |z|²` | Keeps frequency, changes bandwidth |
| B. Conjugate inverse | `z → 1 / conj(z)` | Equivalent to A for roots on real axis |
| C. Move to circle | `z → z / |z|` | Sets bandwidth to zero |

**How to test:**
1. Find a frame where LPC produces unstable roots
2. Apply each reflection formula
3. Compare resulting formant frequencies and bandwidths

**Symptoms of wrong choice:**
- Bandwidth errors on specific frames
- Occasional large formant errors (frames with unstable roots)

**Note:** Options A and B are mathematically equivalent: `1/conj(z) = z/|z|²`

---

### DP8: Root Polishing Iterations

**Applies to:** Formant analysis

**Specification:** Newton-Raphson polishing is standard practice (Numerical Recipes).

**Options to test:**

| Option | Iterations | Notes |
|--------|------------|-------|
| A. None | 0 | Skip polishing |
| B. Light | 5-10 | Quick refinement |
| C. Medium | 20-40 | Standard |
| D. Heavy | 80+ | High precision |

**How to test:**
1. Start with no polishing, measure accuracy
2. Increase iterations until formant accuracy stops improving
3. Find minimum iterations needed for target tolerance

**Symptoms of insufficient polishing:**
- Most frames match, but ~3-5% have errors >1 Hz
- Errors appear random, not systematic

---

### DP9: Band Energy Factor of 2

**Applies to:** Spectrum band energy calculation

**Specification:** One-sided spectrum stores only positive frequencies.

**Options to test:**

| Option | Formula | Rationale |
|--------|---------|-----------|
| A. No factor | `E = Σ |X[k]|² × df` | Energy only in stored bins |
| B. Factor of 2 | `E = 2 × Σ |X[k]|² × df` | Account for negative frequencies |
| C. Factor of 2 except DC/Nyquist | `E = |X[0]|² + 2×Σ|X[k]|² + |X[N/2]|²` | Proper one-sided conversion |

**How to test:**
1. Compute band energy with each formula
2. Compare against parselmouth `Get band energy`

**Symptoms of wrong choice:**
- Band energy off by factor of ~2
- Or off by small amount (DC/Nyquist handling)

---

### DP10: HNR Clamping Bounds

**Applies to:** Harmonicity (AC and CC)

**Specification:** HNR formula gives ±∞ at r=1 and r=0.

**Options to test:**

| Option | Bounds | Notes |
|--------|--------|-------|
| A. ±100 dB | Clamp to [-100, 100] | Conservative |
| B. ±150 dB | Clamp to [-150, 150] | Wide range |
| C. ±200 dB | Clamp to [-200, 200] | Very wide |
| D. Observe | Check what parselmouth returns for extremes | Empirical |

**How to test:**
1. Create pure tone (r ≈ 1) → should give high positive HNR
2. Create white noise (r ≈ 0) → should give large negative HNR
3. Check what values parselmouth returns
4. Use those as your clamp bounds

---

### DP11: Unvoiced Frame HNR Value

**Applies to:** Harmonicity

**Specification:** Not documented what value to return for unvoiced frames.

**Options to test:**

| Option | Value | Notes |
|--------|-------|-------|
| A. Very negative | -200 dB | Indicates "no harmonicity" |
| B. NaN/undefined | Return None/NaN | Indicates "undefined" |
| C. Zero | 0 dB | Equal harmonic/noise |

**How to test:**
1. Find unvoiced region in test audio
2. Check what parselmouth returns for those frames
3. Match that behavior

---

### DP12-14: CC Method Implementation

**⚠️ IMPORTANT: CC Harmonicity is NOT a standalone algorithm.**

The correct understanding:

1. **Harmonicity CC uses Pitch CC internally** (Praat manual confirms this)
2. Pitch CC = the pitch algorithm with CC/cross-correlation method
3. HNR is extracted from pitch correlation strength using `10 × log₁₀(r / (1-r))`

**Correct implementation approach:**

1. Implement the Pitch algorithm with support for CC method
2. The CC method computes normalized cross-correlation instead of autocorrelation
3. Extract the correlation strength `r` at the detected pitch period
4. Apply HNR formula: `HNR = 10 × log₁₀(r / (1-r))`

**What to determine via black-box testing:**

| Decision | Options | How to test |
|----------|---------|-------------|
| Window duration for CC | `periodsPerWindow / pitchFloor` | Compare frame count |
| Frame timing formula | See DP1 | Compare frame times |
| Local mean handling | Weighted vs unweighted | Compare correlation values |

**Do NOT attempt to implement CC as a standalone cross-correlation of adjacent segments.** The CC method is part of the Pitch algorithm, not a separate computation.

**Debugging approach:**
1. First validate Pitch CC gives correct F0 values
2. Then check Pitch CC strength values via `Get strength in frame`
3. Only if both match, verify HNR formula

---

### DP15: Sinc Interpolation Depth

**Applies to:** Pitch (autocorrelation peak refinement), Resampling

**Specification:** Boersma (1993) says N = min(500, samples_to_half_window)

**Options to test:**

| Option | Depth | Notes |
|--------|-------|-------|
| A. N = 70 | 70 samples each side | Commonly used |
| B. N = 500 | 500 samples each side | Paper maximum |
| C. Adaptive | min(500, available_samples) | Paper formula |

**How to test:**
1. Implement with each depth
2. Compare pitch values against parselmouth
3. Deeper interpolation should give higher precision

**Symptoms of insufficient depth:**
- Pitch values slightly off (sub-Hz errors)
- Errors larger for high-frequency signals

---

## Summary Table

| Decision Point | Options | Priority |
|----------------|---------|----------|
| DP1: Frame timing | Left / Center / Right | HIGH - affects all analyses |
| DP2: Gaussian α | 12 / 48 / empirical | MEDIUM - affects pitch, formant |
| DP3: Kaiser β | 20 / 19.74 / 20.24 | LOW - affects intensity only |
| DP4: Physical/effective ratio | 1× / 2× | HIGH - documented for some |
| DP5: DC removal | Unweighted / Weighted / None | MEDIUM - affects intensity |
| DP6: LPC signs | Negative / Positive | HIGH - breaks formant if wrong |
| DP7: Root reflection | z/|z|² / 1/conj(z) / z/|z| | LOW - rare edge case |
| DP8: Root polishing | 0 / 10 / 40 / 80 iterations | MEDIUM - affects ~3% of frames |
| DP9: Band energy factor | 1× / 2× / 2× except edges | MEDIUM - affects spectrum |
| DP10: HNR bounds | ±100 / ±150 / ±200 dB | LOW - edge case |
| DP11: Unvoiced HNR | -200 / NaN / 0 | LOW - unvoiced frames |
| DP12-14: CC method | Via Pitch CC, not standalone | HIGH - see note below |
| DP15: Sinc depth | 70 / 500 / adaptive | LOW - sub-Hz precision |

**Note on DP12-14:** CC Harmonicity should NOT be implemented as a standalone cross-correlation algorithm. It uses Pitch CC internally. Implement Pitch with CC method support, then extract HNR from pitch strength values.

---

## Testing Order Recommendation

1. **First:** DP1 (frame timing) - must match before comparing values
2. **Second:** DP4 (physical/effective) - affects frame count
3. **Then by algorithm dependency:**
   - Spectrum: DP9
   - Intensity: DP3, DP5
   - Pitch (AC): DP2, DP15
   - **Harmonicity AC: DP10, DP11** (after Pitch AC is validated)
   - Pitch (CC): same as AC + verify CC-specific behavior
   - **Harmonicity CC: DP10, DP11** (after Pitch CC is validated)
   - Formant: DP2, DP6, DP7, DP8
