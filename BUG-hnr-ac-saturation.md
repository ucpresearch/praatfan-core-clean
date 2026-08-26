# BUG / update request: AC harmonicity saturates at ~100 dB; low `periods_per_window` produces invalid correlations

Status: OPEN. Filed 2026-08-26 as a clean-room update request.

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
