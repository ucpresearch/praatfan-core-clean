# BUG: whole-file amplitude reference contaminates voicing on long recordings

> **SPEC REVISION v2 (2026-06-11):** the estimator definition in the
> companion `SPEC-speech-reference-normalization.md` changed AFTER the
> initial implementation pass: `reference_peak` is now a percentile
> (default 75) over PER-FRAME peak |x|, not the 98th percentile over
> pooled speech-masked samples. The v1 form has a contamination cliff
> at 2% of speech-masked time (8 s of cumulative laughter/bursts
> against 5 min of speech moves the reference +4400%); v2 moves the
> cliff to (100-p)%. See the spec's "Revision history" + "v1 -> v2
> migration" sections. The `reference_peak` parameter plumbing on the
> analysis calls is unaffected.


**Repo:** `praatfan-core-clean`
**Packages affected:** `praatfan` (pure Python), `praatfan-rust` (MIT Rust + bindings)
**Severity:** medium (correctness on long conversational audio; silent)
**Companion spec:** `SPEC-speech-reference-normalization.md` (two-call API + estimator definition)

## Bug

Pitch (and therefore harmonicity, which routes through the same
machinery) references every frame's amplitude against a whole-file
statistic:

- `src/praatfan/pitch.py:800`:
  `global_peak = np.percentile(np.abs(samples), 99.99)`
- `src/praatfan/pitch.py:826`:
  `local_intensity = local_peak / (global_peak + 1e-30)` — feeds the
  per-candidate intensity adjustment and, downstream, the path
  finder's `silence_threshold` comparison.
- `rust/src/pitch.rs:1100` + `:1153`: same construction in the Rust
  port.
- `src/praatfan/harmonicity.py:181,212,242,275`: exposes
  `silence_threshold` (default 0.1) into the same mechanism.

The 99.99th-percentile softening (vs Praat's strict max) helps against
single-sample clicks but not real events: on a 10-minute 16 kHz file
the percentile is the top ~1k samples, and a 0.5 s burst is 8k samples
— it fully saturates the statistic.

## Impact

On utterance-length sounds (Praat's design regime) global peak ≈
speech peak and all is well. On long conversational recordings:

- One loud event anywhere (click, laugh, interviewer, mic bump)
  raises `global_peak`, lowering every frame's `local_intensity`
  file-wide → quiet voiced frames cross under `silence_threshold` →
  forced unvoiced.
- Downstream, where formant tracks are voicing-gated, frames killed
  here surface as NaN formants exactly in quiet speech, where boundary
  refinement needs them most.
- The effect is invisible: no warning, no parameter the caller can
  currently use to scope the reference to speech.

Note `silence_threshold=0.0` is NOT a workaround: `local_intensity`
also enters the voiced-candidate strength adjustment
(`_find_autocorrelation_peaks(..., local_intensity=...,
apply_intensity_adjustment=...)`), not just the silence gate.

## Request

Per `SPEC-speech-reference-normalization.md`:

1. New `estimate_speech_reference(samples, sample_rate, ...)` →
   `(speech_mask, reference_peak, frame_times)`.
2. New variants of `to_pitch_ac` / `to_pitch_cc` /
   `to_harmonicity_{ac,cc}` accepting `reference_peak: float | None`,
   substituting it for the internal whole-file statistic. Default
   `None` = run the estimator internally (robust-by-default in the
   NEW call only).
3. The existing entry points stay byte-identical — your Praat-parity
   tests should not change at all.
4. Implement in BOTH the pure-Python and Rust paths with identical
   constants, and keep them bit-consistent with each other (the
   existing py↔rs parity harness should cover the new variant too).

## Repro sketch

```python
import numpy as np
from praatfan import Sound
sr = 16000
speech = 0.02 * np.sin(2*np.pi*120*np.arange(sr*600)/sr)  # quiet "voiced" 10 min
burst = speech.copy(); burst[sr*300:sr*300+sr//2] = 0.9    # one 0.5 s event
v0 = Sound(speech, sr).to_pitch_ac(...).values()
v1 = Sound(burst,  sr).to_pitch_ac(...).values()
# BUG: v1 goes unvoiced far from the burst; with reference_peak from the
# estimator, differences should be confined to the burst neighbourhood.
```
