# Implementation decisions: speech-referenced normalization (normative)

Companion to `SPEC-speech-reference-normalization.md`. That document defines the two-call
contract; this one pins every choice it left open, so that praatfan,
praatfan-rust, praatfan-gpl, and sibling packages produce **the same
`reference_peak` from the same samples** and expose **the same API
surface** to the unified praatfan selector (which downstream consumers
actually call).

Status: normative. Any deviation discovered during implementation must
be reflected here **before** the deviating code merges in any repo —
otherwise the cross-package acceptance criterion fails silently.

---

## 1. Estimator: exact algorithm

`estimate_speech_reference(samples, sample_rate, *, frame_s=0.05,
hop_s=0.01, speech_floor_db=30.0, reference_percentile=75.0)`

Input: mono samples, float64 (convert on entry). All arithmetic f64.

The norming standards are **frame-level robust statistics** over
speech-masked frames (median per-frame RMS / mean, and a percentile of
per-frame peak |x|) — NOT sample-level moments/percentiles, which stay
burst-dominated (a 0.5 s, 40× burst is ~5% of speech samples but ~99% of
the energy, so a sample-std or high sample-percentile lands inside the
burst). Frame-level statistics give a short burst only a few frames of
leverage.

### 1.1 Framing

- `n_frame = round(frame_s * sample_rate)`, `n_hop = round(hop_s *
  sample_rate)` (round-half-away-from-zero on the product; with the
  defaults at common rates these are exact integers anyway).
- Frame `i` covers samples `[i*n_hop, i*n_hop + n_frame)`,
  starting at sample 0. Full frames only:
  `n_frames = 1 + floor((N - n_frame) / n_hop)` for `N >= n_frame`.
  Trailing samples not covered by any full frame are simply never
  speech-masked.
- Short-signal case `N < n_frame` (`N > 0`): one all-speech frame
  covering all `N` samples, with standards taken directly from the
  samples — `std = stddev(x)` (floored to 1.0), `mean = mean(x)`,
  `reference_peak = max(|x|)`, `speech_fraction = 1.0`.
- Empty case `N == 0`: empty `speech_mask` / `frame_times`, `mean = 0.0`,
  `std = 1.0`, `reference_peak = 0.0`, `speech_fraction = 0.0`.
- `frame_times[i] = (i*n_hop + n_frame/2) / sample_rate`
  (short-signal case: `N / (2*sample_rate)`).

### 1.2 Per-frame level

- `rms[i] = sqrt(mean(x[frame]^2) + 1e-20)`. The `1e-20` floor keeps a
  silent frame finite (and so includable in 1.3); it is negligible for
  any real speech frame.
- `db[i] = 20 * log10(rms[i] + 1e-20)`. With the floor every frame has a
  finite dB level, so there is no zero-frame exclusion.
- `peak[i] = max(|x[frame]|)` and `mean[i] = mean(x[frame])` (signed,
  the DC reference) are also collected per frame.

### 1.3 Speech mask

- `p95_db` = 95th percentile (see §2 for the percentile definition) of
  `db[i]` over **all** frames. The 95 is a fixed constant of the
  contract, not a parameter.
- `speech_mask[i] = db[i] >= p95_db - speech_floor_db` — inclusive `>=`.
  Frames above `p95_db` (bursts) are included; they get bounded leverage
  in 1.4, not exclusion.
- Fallback: if no frame passes (cannot happen with the `1e-20` floor —
  the frame achieving `p95_db` always passes its own threshold — but
  defensively), set the mask all-True.

### 1.4 Norming standards (frame-level, over speech-masked frames)

Let `S = { i : speech_mask[i] }`.

- `std` = `median(rms[i] for i in S)` — the z-norm scale (the typical
  speech RMS). Floor to `1.0` if it comes out `<= 0`.
- `mean` = `median(mean[i] for i in S)` — the DC reference.
- `reference_peak` = `reference_percentile`-ile (§2, default 75) of
  `peak[i]` over `S` — a TYPICAL-speech peak, deliberately not the
  loudest. The contamination cliff is (100−p)% of speech-masked time
  (p=75 ⇒ 25%), vs the retired v1 sample-domain p=98 whose cliff was
  only 2%.
- `speech_fraction` = `|S| / n_frames`.
- All-zero signal: every `peak[i] = 0`, so `reference_peak = 0.0`
  (mask is all-True via the floor, `std = 1.0`). No exception.

### 1.5 Return value

`SpeechReference` with exactly these field names in every package:

| field             | type                     | content                          |
|-------------------|--------------------------|----------------------------------|
| `speech_mask`     | bool array, len n_frames | per-frame "looks like speech"    |
| `frame_times`     | f64 array, len n_frames  | frame centers in seconds         |
| `mean`            | float                    | median per-frame mean over `S`   |
| `std`             | float                    | median per-frame RMS over `S`    |
| `reference_peak`  | float                    | the norming peak standard        |
| `speech_fraction` | float                    | `|S| / n_frames`                 |

Python: a small frozen dataclass (or PyO3 class) with those
attributes. Rust: a struct with those field names.

---

## 2. Percentile definition (everywhere)

NumPy's default — linear interpolation, Hyndman & Fan type 7. For
sorted ascending `a[0..n-1]` and percentile `q`:

```
h = (n - 1) * q / 100
l = floor(h)
result = a[l] + (h - l) * (a[l+1] - a[l])   # second term absent when h == l
```

`n == 1` → the single value. Rust implementations must implement this
formula directly (full sort or select; either is fine — the result is
a function of the sorted values only, so it is exactly reproducible).

---

## 3. API surface

### 3.1 Estimator exposure

- Pure Python: `praatfan.estimate_speech_reference(...)` (also the
  unified-level export).
- Rust crates: `praatfan_rust.estimate_speech_reference(...)`,
  `praatfan_gpl.estimate_speech_reference(...)` (module-level
  functions taking `(samples, sample_rate)` + the keyword params).
- The **unified** `praatfan.estimate_speech_reference` dispatches to
  the active backend's native estimator when present
  (`hasattr`-guarded), falling back to the pure-Python one. §5 is what
  makes this dispatch safe.

### 3.2 Analysis variants: separate named methods

New methods, not a sentinel kwarg on the existing ones (the existing
entry points must stay byte-identical, and `Option<f64>`/`float | None`
maps cleanly across PyO3 and native Rust, where a three-state sentinel
does not):

```
Sound.to_pitch_ac_referenced(<same params as to_pitch_ac>, *,
                             reference_peak: float | None = None)
Sound.to_pitch_cc_referenced(...)
Sound.to_harmonicity_ac_referenced(...)
Sound.to_harmonicity_cc_referenced(...)
```

Same method names at every layer: unified selector, pure-Python
backend, praatfan_rust PyO3, praatfan_gpl PyO3. This is what the
selector's `hasattr` feature-detection keys on — do not vary the name
per package.

Semantics:

- `reference_peak=<float>`: must be finite and `> 0`, else
  `ValueError` (Python) / `Err` (Rust). Used wherever the whole-file
  amplitude statistic is used today — in praatfan/praatfan-rust that
  is `global_peak` in the `local_intensity` denominator (which feeds
  both the silence gate and the voiced-candidate intensity
  adjustment). No other behavioral change.
- `reference_peak=None` (default): run the estimator internally with
  default parameters and use its `reference_peak`. If that comes back
  `<= 0` (all-zero signal), fall back to the legacy whole-file
  statistic for this call — i.e. behave exactly like the original
  entry point (which on a silent file is all-unvoiced anyway).

### 3.3 Backends that cannot comply

When the unified selector's active backend lacks the native method —
parselmouth (always; you cannot substitute `global_peak` inside real
Praat) or an old praatfan-gpl wheel — the unified `*_referenced`
methods emit a **loud warning** (Python `warnings.warn`, non-silenced
category, message stating that `reference_peak` is being IGNORED and
which backends support it) and then run the legacy whole-file-statistic
path. The warning must fire on **every** affected call, not once per
process — a corpus run that silently reverts to the whole-file
statistic reintroduces the exact bug this exists to fix, so the
degradation has to stay visible in the log for each file processed.

### 3.4 Out of scope

The parselmouth-compat `call()` layer does not expose any of this (it
mirrors real Praat commands only).

---

## 4. Cross-implementation consistency requirements

- **Explicit-float path**: with the same `reference_peak` value passed
  in, a given package's variant is exactly deterministic, and
  scale-invariant: scaling samples and reference together by a power
  of two (exact in f64) must give bit-identical output.
- **Estimator agreement**: across packages, `reference_peak` from the
  same f64 samples must agree to relative `1e-9`, and `speech_mask`
  must be identical except on frames whose `db` lies within `1e-9` dB
  of the threshold. (Bit-exactness is not required — summation order
  in the RMS may differ — but the percentile step is exact given the
  same mask, so in practice agreement is much tighter.)
- The existing py↔rs parity harness in praatfan-core-clean covers the
  new variants; praatfan-gpl validates against the same shared
  fixture below.

---

## 5. Shared synthetic regression (identical in every repo)

Deterministic, no RNG:

```python
sr = 16000
t = np.arange(sr * 600) / sr                 # 10 minutes
speech = 0.02 * np.sin(2 * np.pi * 120 * t)  # quiet "voiced"
burst = speech.copy()
burst[sr*300 : sr*300 + sr//2] = 0.9         # one 0.5 s event at t=300
```

Assertions:

1. **Legacy** `to_pitch_ac`: voicing decisions on `burst` differ from
   `speech` at frames more than 10 s away from the burst (the bug,
   demonstrated).
2. **New variant, default `None`**: voicing differences between
   `burst` and `speech` are confined to frames within ±2 s of the
   burst.
3. **Explicit reference**: passing the *same* float to both signals →
   identical output outside the ±2 s neighbourhood.
4. **Scale invariance**: `(8*x, reference_peak=8*r)` is bit-identical
   to `(x, reference_peak=r)`.
5. **Estimator sanity (isolated event)** on `burst`: `reference_peak`
   within 10% of the burst-free value (the burst is ~50 frames out of
   ~60k and cannot reach the 75th percentile of per-frame peaks).
6. **Estimator sanity (cumulative events)** — the v1 killer: replace
   the single 0.5 s burst with ~8 s of cumulative burst (~2.6% of
   speech-masked time) and the standards still move <5%. v1's
   sample-domain p=98 fails this (2% cliff); v2's frame-domain p=75
   (25% cliff) passes.

---

## 6. Change control

If implementation in any repo forces a change to §1–§3 (e.g. an edge
case not covered here), update this file first and notify the other
repos; the per-repo bug request (`BUG-speech-reference-normalization.md`)
stays as-is and inherits from here.

## 7. Reference implementation status (praatfan-core-clean)

praatfan-core-clean implements the **v2 (frame-domain)** estimator
matching the v2 shared design (`SPEC-speech-reference-normalization.md`);
this package's `speech_reference` module is the reference definition. As
migrated and verified here (2026-06-12):

- Pure-Python estimator `praatfan/speech_reference.py` and Rust port
  `rust/src/speech_reference.rs` produce **bit-identical** `reference_peak`
  and identical `speech_mask` / `frame_times` on the test signals
  (observed rel diff 0.0, not merely the 1e-9 the contract requires).
- `n_frame`/`n_hop` use `floor(x + 0.5)` (Python) / `f64::round` (Rust)
  on the `seconds × rate` product, both half-away-from-zero, both
  floored to ≥1.
- `reference_peak` is the `reference_percentile`-ile (default 75) of
  the **per-frame peak |x|** over speech-masked frames (the v1
  sample-level union machinery is gone). Both ports also return the
  z-norm standards `mean` (median per-frame mean) and `std` (median
  per-frame RMS) plus `speech_fraction`, even though §1.5 lets
  peak-only consumers omit them — keeping the dataclass shape identical
  to the shared reference for interop.
- A silent frame is kept finite in dB by a `+1e-20` floor (so no
  zero-frame exclusion); its per-frame peak is 0, so an all-zero signal
  still yields `reference_peak = 0.0`.
- New methods land as `to_pitch_ac_referenced`, `to_pitch_cc_referenced`,
  `to_harmonicity_ac_referenced`, `to_harmonicity_cc_referenced` on the
  unified `Sound`, each backend adapter, and the praatfan_rust PyO3
  class — same names everywhere, `hasattr`-guarded so old wheels degrade
  via the §3.3 warning.
- Unsupported-backend warning category is `ReferencePeakIgnoredWarning`
  (subclass of `UserWarning`), registered with a `filterwarnings("always")`
  so it fires per call.

A sibling repo may differ in internal mechanism but must match the
observable contract (§1–§5) and the method names in §3.2.
