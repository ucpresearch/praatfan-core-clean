# BUG / update request: AC harmonicity saturates at ~100 dB; low `periods_per_window` produces invalid correlations

Status: **FIXED** 2026-08-26 (see "Resolution" at the end). Filed 2026-08-26
as a clean-room update request.

## ⚠️ Clean-room constraints for whoever fixes this

This report was written to be safe under this repo's clean-room policy: every
fact below is (a) a black-box observation of program behavior, (b) analysis of
THIS repo's own MIT code, or (c) a pointer to a permitted public source. The
implementer must work ONLY from those categories.

Additionally: a prior fix attempt was REVERTED because its derivation
consulted Praat's GPL source. Do NOT consult: Praat's source code, the
`praatfan_gpl` implementation (praatfan-core-rs), that reverted commit or any
reflog remnant of it, the -rs repo's 2026-08-26 CHANGELOG entry, or the
coordinating session's scratchpad/memory notes. If you find yourself knowing
"what Praat does inside," stop and re-derive from the permitted sources.

## Symptom (black-box, measured)

1. `to_harmonicity_ac` emits exactly `99.99999964022885` dB on large fractions
   of conversational speech frames — e.g. ~62% of voiced frames on Buckeye
   files at `periods_per_window=1.0`, `time_step=0.005`, `floor=60`. That
   value is the clip constant: in float64,
   `10*log10((1-1e-10)/1e-10) == 99.99999964022885`, so any correlation
   `r >= 1-1e-10` collapses onto it (see the clamp in `harmonicity.py`).
2. At `periods_per_window=1.0`, the internal correlation maximum `r` reaches
   values far above 1 (observed up to ~15) on a large share of voiced frames.
   An `r > 1` is not a valid normalized correlation; downstream HNR from such
   frames is meaningless regardless of clamping.
3. At `periods_per_window=4.5` the >1 overshoot is tiny (max observed
   ~1.0000003) and occurs on genuinely near-periodic signal (a calibration
   tone), where a very large HNR is legitimate; there the defect is only how
   the boundary is REPRESENTED (a repeated magic constant).

## Own-code analysis (unrestricted — this is our MIT code)

The AC method normalizes the signal autocorrelation by the analysis window's
own autocorrelation: `r_norm(tau) = (r_x(tau)/r_x(0)) / (r_w(tau)/r_w(0))`.
For any finite window, `r_w(tau)` decays toward zero as `tau` approaches the
window length, so the quotient becomes ill-conditioned at long lags: division
by a vanishing denominator amplifies noise without bound. Our search for the
correlation maximum currently extends to `sample_rate/pitch_floor` samples,
which at low `periods_per_window` lies deep inside that ill-conditioned
region — that is where the `r > 1` values arise. This diagnosis follows
entirely from this repo's own code plus elementary DSP; no external source is
needed to see it or to choose a remedy for it.

## Black-box facts about the reference (observed via program behavior)

- Real Praat's `Sound: To Harmonicity (ac)...` REFUSES `periods per window`
  below 3.0 (observable by attempting it in Praat's UI/scripting; error is
  user-visible). Its form default is 4.5.
- parselmouth's `Sound.to_harmonicity_ac()` DEFAULTS to 1.0 and does not
  refuse it — it bypasses the check real Praat applies. (This bypass is the
  origin of an old, incorrect belief in downstream repos that "Praat uses
  1.0".) 1.0 is Praat's default for the CC method, not AC.
- On frames where our output pins at ~100 dB at ppw=4.5, real Praat
  (via parselmouth, at Praat-legal parameters) reports its own distinctive
  large boundary value rather than a clamped ~100.

## Permitted public sources

- Boersma (1993), "Accurate short-term analysis of the fundamental frequency
  and the harmonics-to-noise ratio of a sampled sound" — the published
  description of the AC method, including the window-normalized
  autocorrelation and HNR = 10*log10(r/(1-r)).
- Praat's public manual pages for Harmonicity / `To Harmonicity (ac)...`
  (behavioral documentation, permitted by this repo's policy), including
  whatever they state about parameter ranges and how silent/degenerate frames
  are represented.
- Black-box comparison against parselmouth output at Praat-legal parameters,
  which is also the acceptance test.

## Acceptance criteria

1. No output frame carries the clip constant; degenerate/boundary frames are
   represented deliberately (choose the representation from the public manual
   and/or black-box observation, and document the choice).
2. At `periods_per_window=4.5` on natural speech, agreement with parselmouth
   (excluding frames where Praat emits its own boundary representation)
   should improve over the current ~0.91 Pearson r.
3. `periods_per_window` values in the ill-conditioned regime either produce
   mathematically valid correlations or are refused/handled explicitly —
   justify the choice from the own-code analysis above, not from any
   knowledge of the reference's internals.
4. `praatfan` (Python) and `praatfan_rust` (rust/) stay in lockstep at their
   current ~1e-9 dB agreement or better; pitch/intensity/formant outputs
   unchanged.
5. Downstream note: formantwise's conditioning currently calls AC harmonicity
   at ppw=1.0 and its gate threshold was calibrated on the saturated outputs —
   any praatfan fix changes those values, so formantwise adoption is a
   separate, gated decision (documented in that repo); do not co-develop.

## Sequencing note (added 2026-08-26, coordination with formantwise)

The SIMPLEST acceptable resolution is likely: refuse `periods_per_window < 3.0`
for the AC method (black-box-observable real-Praat behavior; clean-room-safe
for any implementer) + a deliberate boundary representation at legal
parameters — which makes the ill-conditioned regime unreachable and removes
the need for the deeper math remedy.

**Do NOT land the refusal before formantwise stops calling ac at ppw=1.0.**
formantwise 0.3.0's prepass conditioning calls exactly that; adding the
refusal first breaks every formantwise run. Order: formantwise 0.3.1 switches
its conditioning to the CC method at 1.0 (Praat's own CC default) and
recalibrates its gate; THEN this refusal can land here.

## Boundary-representation decision (maintainer, 2026-08-26)

Perfect/near-perfect periodicity (r at the boundary) will be represented as
**NaN / undefined**, not a numeric marker (neither the clamp constant nor
±150-style values, and distinct from the unvoiced representation). Rationale:
HNR is genuinely unbounded as r→1; NaN is the honest float for "undefined",
and it composes correctly downstream (threshold comparisons reject NaN, so
e.g. calibration tones drop out of vowel-frame gates by construction — the
desirable outcome). This refines acceptance criterion 1: the "deliberate
representation" is NaN, documented in the API docs, applied consistently in
praatfan and praatfan_rust. Consumers that want to treat perfect periodicity
as "maximally voiced" must handle NaN explicitly — note it in the CHANGELOG.

---

## Resolution (2026-08-26)

Fixed in `src/praatfan/harmonicity.py` and `rust/src/harmonicity.rs`, with the
diagnostic plumbed through `rust/src/python.rs` (PyO3) and `rust/src/pipe.rs`
(JSON pipe). Derived only from this repo's own code, elementary DSP, the
public Praat manual, and black-box runs; no Praat, praatfan_gpl, or reverted-
commit source was consulted.

### What was wrong

Two independent defects, one masking the other.

1. **Ill-conditioning below 3 periods per window.** The lag search runs to
   `sample_rate/pitch_floor` while the window is only
   `periods_per_window/pitch_floor` long, so the longest lag searched sits at
   `1/periods_per_window` of the window length. Measured `r_w(tau_max)/r_w(0)`
   on a Hanning window (16 kHz, floor 60 Hz): **0.472** at ppw 3.0, **0.167**
   at 2.0, **-4.2e-17** at 1.0 — zero to within rounding, and of arbitrary
   sign. Max observed correlation on conversational speech follows exactly:
   **1.13 / 1.76 / 35.9**. Confirmed by locating the argmax: **100 %** of
   overshooting frames peak in the top quarter of the lag range (median
   `r_w` ratio there at ppw 1.0 is 0.0001).

2. **The clamp hid it.** `r` was clamped to `[1e-10, 1-1e-10]`, so every
   invalid correlation collapsed onto 99.99999964022885 dB — an arithmetic
   boundary wearing the costume of a measurement.

### Black-box findings that shaped the fix

- Real Praat refuses ppw < 3.0: `call(snd, "To Harmonicity (ac)...", ...)`
  raises *"Number of periods per window must be at least 3.0."* for 0.5, 1.0,
  2.0 and 2.9, and succeeds at 3.0. parselmouth's `to_harmonicity_ac()`
  wrapper bypasses that check. Praat does **not** refuse low ppw for the CC
  method, whose standard value is 1.0 — the AC bound is specific to the
  method that divides by the window autocorrelation.
- Praat reports exactly **-200 dB** for pure silence and for a DC signal.
- Praat has **no** distinctive high-periodicity boundary constant. A pure
  sine yields 123.23755252 dB, the Buckeye name-masking tone 128.64 /
  129.04596913 — different numbers, i.e. its own f64 residue, not a clamp.
- Praat manual, `Sound: To Harmonicity (ac)...`: *"4.5 is best for speech:
  HNR values up to 37 dB are guaranteed to be detected reliably; 6 periods
  per window raises this figure to more than 60 dB"* — a documented,
  ppw-dependent reliability ceiling, consistent with the conditioning above.
- Our estimator is **not** biased in the useful range. On synthetic
  sine+noise at known HNR (ppw 4.5) we read 30.27 / 19.99 / 10.13 / 0.65 dB
  for true 30 / 20 / 10 / 0 — closer to truth than Praat's 30.45 / 20.21 /
  10.35 / 1.43.

### The fix

- **`periods_per_window < 3.0` (AC only) now raises a `FutureWarning`** and
  will become an error, per the sequencing note above (formantwise 0.3.1
  must move off ppw=1.0 first). Same message from pure Python, PyO3, and the
  unified selector; the pipe writes it to stderr. `AC_MIN_PERIODS_PER_WINDOW`
  carries the derivation.
- **The clamp is gone.** `strength_to_hnr` now applies one documented policy,
  shared by the AC and CC paths:
  - `r <= 0` → `UNDEFINED_HNR` (-200 dB), matching Praat's black-box value
    for silence and DC.
  - `1 - r <= f64 EPS` → **`nan`**. The correlation carries no information
    about its distance from 1, so no HNR is invented. This covers `r > 1`.
  - otherwise → `10*log10(r/(1-r))`.
- The upper test is `1 - r <= EPS`, **not** `r >= 1`: the last representable
  double below 1 returns 156.5356 dB while a one-ULP-larger `r` returns nan,
  and the two backends' correlations differ at exactly that scale — the
  strict form made them disagree on 59 frames of a 30 s excerpt.

### Acceptance criteria

| # | Criterion | Result |
|---|---|---|
| 1 | No frame carries the clip constant | **Met.** Zero occurrences on every fixture at every ppw; boundaries documented in `strength_to_hnr` (DP10/DP11) and covered by anti-plateau tests in both languages. |
| 2 | ppw=4.5 agreement improves over ~0.91 | **Met.** Buckeye s0101a 30 s speech-dense: Pearson **0.9408 → 0.9745**, MAD **12.02 → 0.94 dB**, p95 **60.50 → 4.82 dB**. |
| 3 | Ill-conditioned ppw handled explicitly | **Met.** Warned now, error later; invalid correlations surface as nan instead of a saturated dB value. Justified from the own-code conditioning measurements above. |
| 4 | Python/Rust lockstep, other outputs unchanged | **Met.** Max AC difference at ppw>=3 is **5.1e-9 dB** (mostly ~1e-13); CC nan sets now match exactly. Pitch/intensity/formant untouched. |
| 5 | Downstream formantwise | Untouched, as instructed. The refusal is deliberately still a warning so formantwise 0.3.0 keeps running; its values at ppw=1.0 do change (invalid frames are now nan). |

### Tests

- `tests/test_harmonicity_boundaries.py` — 44 tests: boundary policy, ULP
  robustness, anti-plateau, warning on/off by ppw and by method, real-audio
  contours, and recovery of a known synthetic HNR (so the policy cannot be
  satisfied by returning nan everywhere).
- `rust/src/harmonicity.rs::boundary_tests` — 7 mirrored unit tests.
- Full suite: **188 Python passed**, **19 Rust passed**, WASM target builds.

### Known, pre-existing, out of scope

`tests/fixtures/tam-haʃaʁav-haɡadol-mono.wav` (48 kHz) sits at Pearson 0.638
against parselmouth at ppw=4.5 — **unchanged** by this fix (0.6381 before and
after) and unrelated to the clamp (it had zero clipped frames either way).
Median absolute difference is only 0.48 dB, but ~5 % of frames run 10-16 dB
low in short runs (t≈1.52-1.545, 1.715-1.74 s), pulling Pearson to 0.737 when
excluded. That is a distinct AC-strength accuracy question deserving its own
investigation.
