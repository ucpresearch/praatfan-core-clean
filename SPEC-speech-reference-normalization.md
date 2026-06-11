# Shared design: speech-referenced amplitude normalization (2026-06-11)

Common spec referenced by the four per-repo bug requests in this
directory. Each request is self-contained; this file holds the shared
API contract and the estimator definition so the Python packages
(praatfan, praatfan-rust, praatfan-gpl, and sibling analysis packages)
implement one interoperable concept.

## Revision history

- **v2 (2026-06-11, CURRENT)** — estimator standards moved from the
  sample domain to the **frame domain** (percentile over per-frame
  peak |x|, scale = median per-frame RMS). Rationale below ("Why the
  frame domain"). Implementations of v1 (praatfan-core-clean's
  `speech_reference.py`, 98th-percentile |x| over speech-masked
  samples) should migrate: v1 and v2 references are NOT
  interchangeable (v2's default sits systematically lower — a
  typical-speech peak rather than a near-max one), so the whole stack
  must be on one version before references are passed across
  packages.
- **v1 (2026-06-11, retired)** — `reference_peak` = 98th percentile of
  |x| over speech-masked samples. Holds against isolated events but
  has a robustness cliff at (100−p)% = 2% of speech-masked time:
  cumulative loud non-speech (laughter, claps) above that fraction
  captures the percentile entirely (measured: 8 s of burst against
  5 min of speech moves the reference +4400%). Conversational corpora
  reach 2% cumulative laughter routinely.

## Why the frame domain (v2)

A reference estimator has two design levers that the sample domain
welds together:

1. **Peak-ness** — how peak-like is the measured quantity? Raw |x|
   samples are dominated by the low-amplitude bulk of the waveform
   (speech crest factors are large), so a sample-domain estimator must
   use a HIGH percentile to measure anything peak-like…
2. **Robustness** — …and the contamination cliff is (100−p)% of the
   masked material, so high p = fragile. Sample-domain p=98 ⇒ 2%
   cliff; lowering p buys robustness but the quantity slides down
   into the waveform body (crest-factor- and vowel-content-dependent).

The frame construction decouples them: **max|x| within each frame**
does the peak-finding (each frame's value is a genuine local peak,
glottal-pulse scale), and a **modest percentile across frames** does
the robustness (cliff = (100−p)% of speech-masked TIME). At the
default p=75 the cliff is 25% contamination — effectively unreachable
— while the measured quantity is still a true peak.

`reference_percentile` remains the one real semantic choice:
- p = 75 (default): "typical speech peak" — robustness-first.
- p = 90–95: "loud-speech peak" — closer to Praat's original
  global-peak semantics (existing silence_threshold intuitions
  transfer better), cliff at 5–10%, still ≫ the v1 margin.
Document the cliff as (100−p)% of speech-masked time wherever the
parameter is exposed.

## The shared bug, in one paragraph

Several analysis paths reference per-frame amplitude against a
**whole-file amplitude statistic** (Praat's `global_peak`, or a
whole-input z-norm in a downstream feature-extraction pipeline). Praat
designed this for utterance-length sounds where the global peak ≈ the
speech peak. On long conversational recordings (10-minute Buckeye
files), the statistic is set by whatever is loudest anywhere — a click,
laugh, or interviewer burst — and is diluted by silence density.
Consequences measured downstream (2026-06-11): per-60s-chunk vs
whole-file normalization relabels 29–72% of self-supervised feature
codes, driven by local silence
density rather than speech level; and the pitch/HNR voicing gate
(local_peak / global_peak vs silence_threshold) classifies quiet voiced
frames as silent when anything loud exists elsewhere in the file —
which then NaN-gates formants in downstream boundary refinement.

## The two-call contract

**Call 1 — estimator** (new, pure function, no analysis side effects):

```
estimate_speech_reference(samples, sample_rate, *,
                          frame_s=0.05, hop_s=0.01,
                          speech_floor_db=30.0,
                          reference_percentile=75.0)   # over FRAME PEAKS (v2)
  -> SpeechReference {
       speech_mask:    per-frame bool (hop_s grid) — "this looks like speech"
       mean, std:      z-norm-style standards (median frame mean / RMS)
       reference_peak: peak-style standard for Praat-like consumers
       frame_times:    per-frame centers (s)
     }
```

Definition (v2; signal-only — no annotation input, so no train-time
peek concerns):
1. Frame the signal (`frame_s` windows, `hop_s` hop); per-frame RMS.
2. `speech_mask` = frames whose log-RMS is within `speech_floor_db`
   of the 95th-percentile log-RMS. (Silence/leakage drops out; loud
   bursts stay in but cannot dominate step 3.)
3. Standards from **frame-level robust statistics over speech-masked
   frames** (see "Why the frame domain" above):
   - `reference_peak` = `reference_percentile`-ile (default 75) of
     **per-frame peak |x|** (max within frame, percentile across
     frames). Consumers replacing Praat's `global_peak` should expect
     `local_intensity` values to scale up accordingly; the
     silence-threshold semantics keep working because BOTH quiet-frame
     and threshold semantics are now speech-relative.
   - `std` = median per-frame RMS (the z-norm scale; equals the
     typical speech RMS, burst-invariant by construction)
   - `mean` = median per-frame mean (DC reference)
   Packages that only need the peak (praatfan*) may omit `mean`/`std`
   from their return type, but must not redefine the mask or the peak.
   Reference implementation: this package's `speech_reference` module.

Acceptance for the estimator itself (both synthetic checks must pass):
- isolated event: +0.5 s 0.9-amplitude burst on a quiet 10-min
  recording with 5 min of speech → standards move <5%;
- cumulative events (the v1 killer): +8 s of burst against 5 min of
  speech (~2.6% of speech-masked time) → standards still move <5%.

### v1 → v2 migration (praatfan-core-clean, and its Rust port)

The implemented v1 estimator (98th percentile over speech-masked
SAMPLES) needs three changes, everything else stays:
1. Compute per-frame peak |x| (max within each speech-masked frame)
   and take `reference_percentile` over THOSE, instead of a percentile
   over the pooled samples. (The sample-level mask machinery can go.)
2. Default `reference_percentile` 98.0 → 75.0; document the
   contamination cliff as (100−p)% of speech-masked time.
3. Apply identically in the pure-Python and Rust implementations
   (keep the half-away-from-zero rounding parity convention already
   established there).
The `resolve_reference_peak` / `reference_peak`-parameter plumbing on
the analysis calls is unaffected — only the internal default
estimator changes.

**Call 2 — modified analysis call** (new variant; the existing call is
left byte-identical for Praat-parity / back-compat):

The existing analysis entry points (`to_pitch_*`, `to_harmonicity_*`,
extraction) gain a variant accepting `reference_peak: float | None`.
Semantics:
- `reference_peak=<float>`: use it wherever the whole-file
  peak/statistic is used today (pitch `local_intensity` denominator,
  silence threshold, z-norm scale, …).
- **Default (`None`): the NON-MAX alternative** — internally run
  Call 1 and use its `reference_peak`. I.e. the new variant is
  robust-by-default; callers wanting exact Praat behaviour keep using
  the original call.

Rationale for two calls instead of one flag: the estimator runs once
per recording and its output is shared across pitch, HNR, and
downstream normalization — recomputing it inside every analysis
call would be wasteful and risks divergent definitions per package.

## Acceptance criteria (all repos)

- Original entry points byte-identical (existing parity/golden tests
  untouched).
- New variant with explicit `reference_peak=x` is deterministic in x
  and monotone in the obvious sense (scaling the input and the
  reference together is a no-op).
- Synthetic regression test: a 10-min file of quiet speech + one loud
  1 s burst. Original call: voicing/codes change file-wide vs the
  burst-free version. New variant: changes confined to the burst's
  neighbourhood.
- Cross-package: praatfan / praatfan-rust / praatfan-gpl and sibling
  packages implement the same estimator definition (same constants) so
  a single reference can pass through the whole stack.
