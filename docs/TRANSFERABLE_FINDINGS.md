# Transferable findings from praatfan-core-rs

Items from the GPL sibling (`praatfan-core-rs`) that this clean-room MIT
project can pick up.

> **Clean-room caveat (read first).** The GPL sibling investigated all
> of these items by reading Praat's C++ source. Most of the answers it
> found can in principle be rediscovered from Praat's *behaviour* alone,
> but anything Praat-source-derived is unsafe for this project to copy
> verbatim. This document is a curated **list of hypotheses worth
> testing black-box** plus a few crate suggestions that don't touch
> Praat at all.
>
> Wording rules I've tried to follow:
> - Hypotheses are stated as "candidate A vs candidate B"; I do not say
>   which one Praat uses, only that the experiment will tell you.
> - Where a numerical magnitude or symptom is named (e.g. "an outlier
>   token's F2 errs by ~1800 Hz"), it's a measurable fact about
>   Praat's externally observable output, not an implementation detail.
> - Where I include a code sketch, the algorithm is independently
>   motivated (sampling theorem, basic DSP) — internal magic constants
>   from Praat's source are deliberately left as `?` for you to find
>   experimentally.

---

## Free wins (no Praat source involvement)

### `faer` for eigendecomposition (root finding)

- Crate: <https://crates.io/crates/faer> (MIT)
- Pure Rust, WASM-compatible. Replaces a custom Hessenberg + Francis QR
  with a well-tested equivalent. Precision ≈ LAPACK `dhseqr`.
- Minimal feature set:
  ```toml
  faer = { version = "0.24", default-features = false, features = ["std", "linalg"] }
  ```
- API: `faer::Mat::<f64>::from_fn(n, n, |i, j| ...).eigenvalues()?`.
- No Praat connection. Just a better eigensolver crate.

### `speexdsp-rs` FFTPACK port

This crate is independently published MIT-licensed Rust, derived from
Xiph/Vorbis's public-domain `smallft` C code. Whether it happens to
match any other FFT implementation's bit pattern is something you'd
have to verify yourself — I'm not asserting parity with anything in
particular.

- Source: <https://github.com/rust-av/speexdsp-rs> — files
  `fft/src/{smallft,dradb,dradf}.rs`.
- It's `f32` out of the box. If you need higher precision, vendor and
  promote `f32 → f64` throughout. Trivial mechanical change.
- **Two bugs in the upstream MIT code** found by static review +
  comparing forward/backward FFT round-trip to numpy on synthetic
  inputs (entirely independent of Praat):
  1. `dradf.rs:15` — operator-precedence error from c2rust output. The
     index `ch[t1 << (1 + t3 - 1)]` was meant to be `ch[(t1 << 1) + t3
     - 1]`. The shift-by-`(1+t3-1)` form clearly can't be intended:
     the index would explode for any nontrivial `t3`. Verify by
     observing that the unfixed kernel fails a self-roundtrip.
  2. Truncated trig constants. The upstream f32 source uses literals
     like `0.866`, `6.283`, `0.70710677`. For any f64 use, those need
     to be the closest-f64 representations of `sin(60°)`, `2π`, and
     `sqrt(0.5)` respectively. Source-evident the moment you compare
     them to `std::f64::consts::TAU` etc.
- Verify after fixing: `drftf1` forward FFT on a synthetic signal
  should match `numpy.fft.rfft` to ~f64 machine precision (5e-16) and a
  round-trip at n=32 should recover the original within ~1e-13.

If you decide an FFTPACK-style FFT is the right backend for any of
your DSP, this crate (with these fixes) is a clean MIT starting
point. I'm not telling you which DSP needs it.

### Regression-baseline pattern

- Pin a CSV of `(filename, t_mid_s, praat_f1/2/3, ours_f1/2/3)` from
  parselmouth calls on public audio (e.g. Hillenbrand 1995). Run a
  check after any precision-touching change. The GPL repo has a
  reference `scripts/{generate,check}_h95_*.py`; copying the *idea*
  is fine, the *files* are GPL.

### Feature-flag non-WASM deps

If you ever need real LAPACK bindings or anything else native-only,
gate it behind a Cargo feature so the WASM build stays pure-Rust.

---

## Decision points

Open questions for the clean-room project to answer via black-box
experiments. The format is: an **observation**, the **hypotheses worth
testing**, and the **experiment that distinguishes them**. I'm not
saying which hypothesis is correct.

### DP: is `Sound: Resample` a single-stage or two-stage operation?

(Likely the **first** thing to test if standalone Burg formant parity
isn't tracking — `docs/RESAMPLER_INVESTIGATION.md` documents what
single-stage attempts look like at the parity wall.)

**Observations.**
1. Praat's resampled output in **silent input regions** is not zero
   — there's a small RMS baseline (~5e-5 on typical 16-bit audio).
2. The impulse response (delta-in, observe out) has a long `1/d`-like
   tail extending thousands of samples, plus a precise main lobe.

**Hypotheses to weigh.**
- *A. Pure compact-support windowed-sinc.* Compatible with the precise
  main lobe; **incompatible** with observation 1 (silent regions
  would be exactly zero).
- *B. Pure FFT brick-wall lowpass.* Compatible with observation 1 (a
  rectangular cutoff in frequency produces a `1/d` sinc tail in time);
  but the main lobe phase differs from windowed-sinc.
- *C. A two-stage composition* — a brick-wall LPF followed by a
  windowed-sinc fractional-delay step. Could explain both observations.

**Experiment.** Implement A, B, and C; for each, compare to
`parselmouth.praat.call(sound, "Resample...", new_rate, 50)`
sample-by-sample on a known signal AND on a sample-aligned silent
signal. Whichever produces both the long tail in silent regions AND
matches main-lobe phase to ~1e-5 wins.

**Rough sketch of C** (knobs marked `?` are the things you'll need to
find experimentally — padding amount, cutoff bin convention, sinc
window depth, etc.):

```python
def _resample_two_stage(signal, src_rate, dst_rate):
    if dst_rate < src_rate:
        # Stage 1: FFT brick-wall LPF. Padding amount, cutoff bin formula,
        # and edge handling are all DPs — match to Praat by experiment.
        nfft = ?  # next power of two above n + ?·boundary_pad
        padded = _pad_with_some_strategy(signal, nfft)
        spectrum = np.fft.rfft(padded)
        cutoff = ?  # function of dst_rate / src_rate and nfft
        spectrum[cutoff:] = 0
        filtered = np.fft.irfft(spectrum, nfft)[: len(signal)]
    else:
        filtered = signal  # upsample skips AA filter? — also a DP

    # Stage 2: windowed-sinc fractional interpolation
    return _windowed_sinc(filtered, src_rate, dst_rate, depth=?)
```

The relative magnitudes the GPL sibling observed: scipy's pure-FFT
resampler hits an F1-mean parity floor around 5 Hz; a pure
windowed-sinc with the right kernel goes much tighter on voiced
frames but explodes on silent frames; a two-stage version trades the
two off and can do better than either alone. Use these as
sanity-checks while iterating.

### DP: sinc-interpolator window depth near signal boundaries

**Observation.** Resample output near the start and end of the signal
diverges from Praat ~10× more than mid-signal output, even after
matching everything else.

**Hypotheses.** The raised-cosine window's effective depth at sample
position `x`, with neighbour indices `left` and `right`:
- *A.* Symmetric: `leftDepth = rightDepth = f(maxDepth)` for some
  constant function (i.e., the window is the same shape regardless
  of where `x` sits in `[left, right]`).
- *B.* Asymmetric and `x`-dependent: the window stretches further on
  whichever side has more samples to draw on (so its width depends
  on the distance from `x` to each endpoint).

Plus version-dependence: parselmouth bundles a specific Praat version,
so the answer might differ from what newer Praat ships.

**Experiment.** Synthesise a known signal, ask Praat to resample
it, and at a boundary sample compute what each candidate would
predict. Only one will match.

### DP: bandwidth formula from LPC root magnitude

**Observation.** Standard DSP references give the resonance bandwidth
as `K · log(|z|⁻¹) · sample_rate / (2π)` for some convention `K`.
Different references pick different `K`s.

**Experiment.** Synthesise a sound from a known-pole LPC polynomial,
run `Sound: To Formant (burg)`, and solve for `K`.

**Why it matters.** Picking the wrong convention silently halves or
doubles your bandwidths, which is mostly cosmetic for standalone
formant display but breaks downstream use of bandwidth as a
weighting `σ`.

### DP: does FormantPath's default-path extract equal `Sound: To Formant (burg)`?

**Observation.** Take a FormantPath with no `Path finder` call, run
`Extract Formant` (which returns the middle candidate's Formant), and
compare frame-by-frame to a standalone `Sound: To Formant (burg)` at
the same ceiling. They're not equal.

**Hypotheses for what could differ between the two pipelines.** Which
of the following Praat treats the two pipelines differently on:
candidate frame-grid alignment (timestamps), pre-emphasis order,
windowing convention, frame-mean handling, sample-extraction offset
near integer/half-integer indices, edge-padding behaviour. There
might be more than one of these in play.

**Experiment.** Each hypothesis is independently testable: design a
synthetic input that's sensitive to one knob (e.g., a constant DC
offset to test mean handling, a half-sample-aligned timestamp to test
sample-extraction offset). Pin down which knobs matter, then which
direction matches.

The fact that the two pipelines diverge at all is essential context
for any FormantPath recipe; the specific cause(s) are yours to find.

### DP: what quantity does Praat use for per-frame intensity?

**Observation.** FormantPath's `Path finder` exposes an
`intensityModulationStepSize` parameter, so it's modulating cost by
some per-frame intensity scalar. Praat doesn't expose that scalar
directly through parselmouth's API.

**Hypothesis space.** Anything the parselmouth user can compute from
the signal: `max(sample²)`, RMS of the windowed frame, peak power, or
some derived quantity from the LPC analysis itself.

**Experiment.** Construct test signals with frames whose hypothesis
quantities have known relative orderings, run `Path finder` with
non-trivial intensity modulation, and see which hypothesis predicts
the selected paths. Be ready for the answer to be something less
obvious than RMS.

### DP: FormantPath Viterbi track-count cap

**Observation.** On certain Hillenbrand 1995 vowel tokens (notably
high-F2 vowels like /i/), it's possible for an implementation that
"looks right" by every other criterion to still produce a >1000 Hz F2
error at the vowel midpoint, because its Viterbi picks a low-ceiling
candidate whose F2 is a spurious narrow-bandwidth resonance.

**Hypothesis space.** When the Viterbi computes the per-transition
frequency-change cost, how many formant tracks does it sum over? Two
plausible candidates:
- *A.* All the tracks the Formant frames have (e.g., 5).
- *B.* Capped at the length of the user-supplied per-track parameters
  vector (e.g., 4 when the user passes `"3 3 3 3"` to `Path finder`).
- (Other caps are conceivable too — e.g., per-frame
  `numberOfFormants`. List them all and test.)

These differ by at most one track, but that one track's
frequency-change contribution is large enough on susceptible tokens
to flip the path.

**Experiment.** Pick `w32iy` (or any high-F2 vowel where the lowest
ceiling has a bandwidth-narrow spurious resonance below F2). Ask
Praat to run `Path finder` with default weights and `parameters =
"3 3 3 3"`. Then implement A, B, ... and see which gives Praat's
extracted F2 (~2740 Hz on this token) vs the spurious value
(~946 Hz).

### DP: FormantPath Viterbi path-selection semantics

The Viterbi DP's internal `delta`/`psi` matrices aren't exposed by
parselmouth, so the cost-function and back-tracking rules aren't
directly observable. You're going to be partially in the dark here.

Two viable strategies:

1. **Try for exact parity** by reverse-engineering from I/O. Newer
   Praat (≥6.2, depending on what your parselmouth bundles) exposes
   `Get stress of candidate`, `List stress of candidates`, `Get
   candidate in frame`, and `To Matrix (stress)` / `To Matrix (qsums)`.
   With those you can drive a reasonably constrained inverse problem
   for the cost structure.
2. **Settle for functional parity.** Implement a standard Viterbi
   with the documented cost components (stress, Q-factor, frequency
   change, ceiling change, intensity modulation). Expect ~70–90%
   path agreement; the residual disagreements come from internal
   tie-breaking and accumulator rules you can't probe black-box.

Both are defensible. Choose based on what your downstream consumers
need.

---

## What to start with (impact order, GPL-sibling experience)

Numbers below are observable consequences (parity vs parselmouth on
the Hillenbrand 1995 set, 1609 tokens) — the implementation choices
that produce them are yours to discover.

1. **Resampler structure** (§ DP: `Sound: Resample`). Almost certainly
   the first lever. Single-stage approaches plateau at F1 mean ~5 Hz
   and produce chaotic poles in silent frames; the GPL sibling found
   a two-stage decomposition does much better. If your standalone
   Burg parity isn't tracking, start here.
2. **FormantPath Viterbi track-count cap** (§ DP). On the GPL project
   this single-knob change took F2 mean parity from 2.01 Hz to 0.14
   Hz and F2 max from 1795 Hz to 50 Hz across 1609 tokens — the
   biggest single FormantPath parity move.
3. **FormantPath ≠ standalone Formant** (§ DP). A handful of
   structural divergences. Worth ~2–40 Hz per frame on FormantPath
   output when any one of them is wrong.
4. **Bandwidth K convention** (§ DP). Quietly multiplies/divides all
   bandwidths by 2 if wrong; downstream stress fits silently break.
5. **Viterbi internals** (§ DP) — diminishing returns; ~10–30% of
   path disagreement lives here and may not be fully closable.
6. **Sinc boundary depth** (§ DP). Mostly cosmetic for formant parity;
   meaningful for `Resample` bit-equivalence.
7. **Frame intensity quantity** (§ DP). Modest; only affects cases
   with non-default `intensityModulationStepSize`.

Everything else (FFTPACK choice, eigensolver choice, Kahan summation
in inner loops) is hygiene — useful for confidence and code quality
but unlikely to move your h95 parity numbers at all.
