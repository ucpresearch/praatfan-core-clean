# BUG (**FIXED** 2026-08-26): identical `to_harmonicity_ac` call — praatfan_gpl says "unvoiced", praatfan/praatfan_rust say "~100 dB". Please confirm and fix.

Filed 2026-08-26. Companion minimal repro to `BUG-hnr-ac-saturation.md` (which
carries the own-code analysis and the clean-room constraints — read its
warning section before working on this).

## The call (X)

```python
from praatfan import Sound          # PRAATFAN_BACKEND set per backend below
s = Sound("s0101a_10s.wav")         # first 10 s of Buckeye s0101a, 16 kHz mono
h = s.to_harmonicity_ac(time_step=0.005, minimum_pitch=60.0,
                        periods_per_window=1.0)
```

Same file, same parameters, three backends. (`praatfan_gpl` was driven through
`praatfan-core-rs`'s `praatfan-gpl-pipe` binary, `harmonicity_ac` with the
identical parameters; parselmouth at the same call gives the same picture.)

## What comes back (Y vs Z)

| t (s) | praatfan | praatfan_rust | **praatfan_gpl** |
|---|---|---|---|
| 0.1675 | 99.99999964023 | 99.99999964023 | **−200.000** |
| 4.5575 | 99.99999964023 | 99.99999964023 | **−200.000** |
| 5.0825 | 99.99999964023 | 99.99999964023 | **−200.000** |
| 5.4175 | 99.99999964023 | 99.99999964023 | **−200.000** |
| 5.9375 | 99.99999964023 | 99.99999964023 | **−200.000** |
| 7.0925 | 99.99999964023 | 99.99999964023 | **−200.000** |
| 7.8525 | 99.99999964023 | 99.99999964023 | **−200.000** |

- 55 of 1,994 frames (2.8 %) on this quiet excerpt read exactly
  `99.99999964022885` from praatfan/praatfan_rust; on speech-dense material
  the rate reaches tens of percent of voiced frames.
- At every one of those frames, praatfan_gpl reports **−200 dB (unvoiced)**.
  The two answers could not disagree harder: "essentially perfect
  periodicity" vs "no periodicity at all".
- praatfan_gpl's values on the excerpt span −15.9 … 11.0 dB — physically
  ordinary — with **zero** frames above 90 dB.
- praatfan and praatfan_rust agree with each other to 0.0 on this stream
  (the bug is shared by the pair, as their lockstep requires).
- `99.99999964022885 == 10*log10((1-1e-10)/1e-10)` exactly — a clamp
  boundary, not a measurement.

## Please confirm and fix

Confirm: run the snippet on any conversational recording; the constant
appears wherever the disagreement does. Fix: in praatfan (Python) and
praatfan_rust together, under the clean-room process and acceptance criteria
in `BUG-hnr-ac-saturation.md` — permitted sources only; do not consult
Praat's source, praatfan_gpl's code, or the reverted prior attempt. The
Praat-faithful references (praatfan_gpl's pipe, parselmouth) are for
BLACK-BOX comparison of outputs only.

Repro artifacts: session scratchpad `hnrbug/` (excerpt wav, both JSON dumps,
the comparison one-liner).

---

## Resolution (2026-08-26)

Confirmed and fixed; see the "Resolution" section of
`BUG-hnr-ac-saturation.md` for the full derivation and acceptance table.

Reproduced exactly: **55 of 1 994** frames on the 10 s excerpt carried
99.99999964022885 dB, at ppw=1.0, floor 60, time_step 0.005.

Root cause of the disagreement in the table above: at ppw=1.0 the window is
`1.0/60` s long while the lag search runs to `1/60` s, so the longest lag
searched sits at the *end* of the window, where the window's own
autocorrelation has decayed to `-4.2e-17`. Dividing by that produced
correlations up to **35.9**; the old clamp then folded every one of them onto
the ~100 dB constant. praatfan_gpl's -200 and our ~100 were two different
renderings of the same non-measurement.

After the fix, at ppw=1.0 those frames are **`nan`** (552 of 5 994 on a
speech-dense 30 s excerpt), a `FutureWarning` explains why, and the call will
become an error once formantwise moves off ppw=1.0. `praatfan` and
`praatfan_rust` produce identical nan sets and agree to ~1e-13 dB on finite
frames at ppw>=3.
