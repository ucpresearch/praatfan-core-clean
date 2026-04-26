# Recipe: FormantPath

Clean-room recipe sketch for Praat's `FormantPath` (multi-ceiling Burg
formant analysis with Viterbi-optimal ceiling path selection).

> **Clean-room status of this document.** Everything in this recipe is:
> 1. Documented in the Praat manual or in cited papers (Boersma &
>    Weenink, Weenink 2015).
> 2. Observable black-box from parselmouth (parameter names, return
>    types, behavior on test signals).
> 3. A decision point (DP) — an open question to answer by black-box
>    experiment, with hypotheses framed as "A vs B" and the experiment
>    that distinguishes them.
>
> If you write code based on this document and want to confirm its
> clean-room provenance, every concrete spec below should be traceable
> to one of (1) the cited public references, (2) parselmouth output,
> or (3) an experiment YOU ran. Anything that isn't is a bug — please
> remove and re-derive it independently.

---

## Allowed sources

- Praat manual: <https://www.fon.hum.uva.nl/praat/manual/FormantPath.html>
- Parselmouth API + `parselmouth.praat.call(...)` for driving Praat
- Cited DSP / numerical-methods literature
- Published papers (especially Weenink 2015, the FormantPath origin)

**Not allowed:** reading Praat's C++ source.

**Prerequisites:** complete the standalone Burg `Formant` recipe in
[`RECIPE.md`](RECIPE.md) first; FormantPath builds on it.

---

## References

1. **Praat:** Boersma, P. & Weenink, D. (2024). *Praat: doing phonetics
   by computer.* <https://www.fon.hum.uva.nl/praat/>
2. **FormantPath origin:** Weenink, D. (2015). "Improved formant
   frequency measurements of short segments." *Proceedings of ICPhS
   2015*.
3. **Optimal-ceiling background:** Escudero, P., Boersma, P., Rauber,
   A. S., & Bion, R. A. H. (2009). "A cross-dialect acoustic
   description of vowels: Brazilian and European Portuguese." *J.
   Acoust. Soc. Am. 126*(3), 1379–1393.
4. **Parselmouth:** Jadoul, Y., Thompson, B., & de Boer, B. (2018).
   "Introducing Parselmouth: A Python interface to Praat." *Journal of
   Phonetics, 71*, 1–15.
5. **Viterbi DP:** Forney, G. D. (1973). "The Viterbi Algorithm."
   *Proceedings of the IEEE, 61*(3), 268–278.
6. **Linear least squares with SVD:** Press, W. H. et al., *Numerical
   Recipes* (Cambridge), Ch. 15.

---

## 1. What FormantPath is (high-level, manual-derivable)

A FormantPath is a bundle of `N = 2 · numberOfStepsUpDown + 1` Burg
formant analyses computed at log-spaced ceiling frequencies, plus a
per-frame integer "path" pointing at one candidate per frame. Two
operations are documented:

- **`Path finder`** assigns the path by minimising a sum of cost
  components (stress, Q-factor, frequency change between adjacent
  frames, ceiling change, intensity-modulated weighting).
- **`Extract Formant`** materialises the path into a regular Formant
  object whose values are taken frame-by-frame from the selected
  candidate.

### Praat scripting API (observable)

| Method | Signature | Output |
|---|---|---|
| Construct | `Sound: To FormantPath (burg)... timeStep maxFormants middleCeiling windowLength preEmph ceilingStep stepsUpDown` | FormantPath |
| Path finder | `FormantPath: Path finder... qW fcW sW ccW intensityMod pathWindowLength parameters$ power` | modifies path |
| Extract | `FormantPath: Extract Formant` | Formant |
| Set path | `Set path... tmin tmax candidate` | modifies path |
| Stress queries (Praat ≥ 6.2) | `Get/List stress of candidate(s)`, `Get/Set optimal path` | various |
| Candidate query | `Get candidate in frame... iframe` | integer |

### Parselmouth caveat

Parselmouth 0.4.x ships Praat 6.1.38, which **doesn't expose** the
stress/candidate query commands as scripting calls. In that version
the only way to validate your implementation against Praat's
FormantPath is to compare the **extracted Formant** frame-by-frame.
If you can use a parselmouth that bundles Praat ≥ 6.2, the stress
queries are a much stronger verification harness.

---

## 2. Constructor parameters ✅ DOCUMENTED

From the Praat manual:

| Name | Unit | Praat default | Meaning |
|---|---|---|---|
| `timeStep` | s | 0.005 | time between frames |
| `maximumNumberOfFormants` | unitless | 5 | formants per frame |
| `middleCeiling` | Hz | 5500 (F) / 5000 (M) | center ceiling |
| `windowLength` | s | 0.025 | analysis window |
| `preEmphasisFrequency` | Hz | 50 | pre-emphasis filter |
| `ceilingStepSize` | unitless | 0.05 | log step for ceilings |
| `numberOfStepsUpDown` | unitless | 4 | steps each side of middle |

### Ceiling generation ✅ DOCUMENTED

```
N = numberOfStepsUpDown
for i in 0 .. 2*N:
    ceiling[i] = middleCeiling * exp((i - N) * ceilingStepSize)
```

The array is monotonically ascending; `ceiling[N]` is the middle.
With default `N = 4` you get 9 candidates.

### Constraint

`ceiling[last] ≤ sampleRate / 2` — Praat raises an error otherwise.

---

## 3. Per-candidate Burg analysis

For each ceiling, run a standalone Burg formant analysis (your existing
`RECIPE.md` §Formant) at `max_formant_hz = ceiling[i]`. Each candidate
yields a per-frame `(frequencies[1..maxFormants], bandwidths[1..maxFormants])`
plus some per-frame scalar — see DPs below.

### DP-FP-1: do all candidates share one frame grid?

**Observation.** When you run `Sound: To Formant (burg)` standalone at
each ceiling, each Formant gets its own frame grid because resampling
to `2 · ceiling` produces a sound with a slightly different `dx` per
ceiling. Yet `Extract Formant` from a FormantPath returns one Formant
with a single grid that all candidates' frames are aligned to.

**Hypothesis.** Praat must force all candidates' Burg analyses onto a
common frame grid. The two natural choices:
- *A.* The grid of the middle ceiling (the common reference).
- *B.* Some other shared reference (e.g., the original sound's grid).

**Experiment.** Construct a FormantPath, get its `(t1, dx, nx)` via
parselmouth's Sampled queries, and compare to the standalone Burg
Formant at each ceiling. Whichever standalone matches is the
reference.

### DP-FP-2: is FormantPath's per-candidate Burg the same as standalone Burg?

**Observation (essential, do this experiment first).** Build a
FormantPath without calling `Path finder`, run `Extract Formant` (which
returns the middle candidate verbatim), and compare frame-by-frame to
a standalone `Sound: To Formant (burg)` at the middle ceiling. **They
do not match** — the differences are systematic, not numerical noise.

**Hypotheses for what could differ in the two pipelines** (test each
in isolation; more than one may apply):
- Frame-grid alignment (DP-FP-1 above).
- Pre-emphasis ordering relative to resampling.
- Whether per-frame DC is removed before windowing.
- Sample-extraction offset convention near integer/half-integer
  fractional indices.
- Edge-of-signal sample handling (clamp, zero-pad, reflect).

**Experiment.** Each hypothesis admits a synthetic input that's
selectively sensitive to that knob:
- DC: add a constant offset to a known signal; mean-removal
  pipelines will produce identical formants, non-removal pipelines
  will shift them.
- Sample-extraction offset: place a frame timestamp at exact
  integer/half-integer fractional indices and compare.
- Edge handling: extract formants from the very first / very last
  frame on a windowed-down signal.

Pin down each knob one at a time. The cumulative effect when any are
wrong is several Hz to tens of Hz of formant divergence.

### DP-FP-3: bandwidth-from-pole convention

See `TRANSFERABLE_FINDINGS.md` §"DP: bandwidth formula from LPC root
magnitude". The formula has two scaling conventions in the literature.
Run that experiment first because the wrong choice silently breaks the
stress fit downstream.

### DP-FP-4: per-frame intensity scalar

`Path finder` exposes an `intensityModulationStepSize` argument so the
DP works on a per-frame intensity. That value isn't exposed by
parselmouth directly. See `TRANSFERABLE_FINDINGS.md` §"DP: what
quantity does Praat use for per-frame intensity?".

### Default path ✅ DOCUMENTED

Before `Path finder` runs, the path is initialised to the middle
candidate at every frame. `Extract Formant` at this point returns the
middle candidate's Formant frame-by-frame.

---

## 4. `Path finder` — Viterbi over the trellis

A standard Viterbi DP over a `(N candidates) × (T frames)` trellis,
with cost components documented in the Praat manual.

### 4.1 Parameters ✅ DOCUMENTED (from Path finder form)

| Name | Praat default | Description |
|---|---|---|
| `qWeight` | 1.0 | weight on Q-factor |
| `frequencyChangeWeight` | 1.0 | weight on inter-frame freq change |
| `stressWeight` | 1.0 | weight on stress |
| `ceilingChangeWeight` | 1.0 | weight on ceiling jumps |
| `intensityModulationStepSize` | 5.0 dB | sigmoid half-width |
| `pathWindowLength` | 0.035 s | stress-fit window |
| `parameters` | `[3, 3, 3, 3]` | Legendre orders per track |
| `power` | 1.25 | stress exponent |

### 4.2 Cost components — observable structure, internal details DPs

The manual tells you which signals shape the cost; it doesn't tell you
the exact formulas, normalisation, or accumulation rules. Treat each
component as a DP.

**Per-state (static) cost** combines, at each `(candidate, frame)`:

- A **stress** term — penalises candidates whose formant tracks don't
  fit a low-order polynomial well over a `pathWindowLength` window
  (see §4.3 below).
- A **Q-factor** term — rewards candidates with sharper resonances
  (small bandwidth relative to frequency).
- An **intensity modulation** scalar that scales the above two
  contributions on a per-frame basis when frame-energy varies.

**Per-transition cost** combines, on each `(candidate j at t-1) →
(candidate i at t)` arrow:

- A **frequency-change** term — penalises large frequency shifts
  between corresponding formants of adjacent frames.
- A **ceiling-change** term — penalises switches between distant
  ceilings.

#### DP-FP-5: how are stress and Q-sum normalised before weighting?

**Observation.** All weights default to 1.0; saturation is gentle (a
stress of, say, 50 doesn't dominate one of 5). So there's a
normalisation step that maps raw values to a bounded range.

**Hypothesis.** Saturating divide: `min(raw / cutoff, 1.0)` for some
cutoff per quantity.

**Experiment.** Sweep `stressWeight` (with all others at 0) over a
test signal and observe at what magnitude the chosen path stops
changing — that's the saturation threshold. Same trick for Q-factor.

#### DP-FP-6: how is the frequency-change cost normalised?

Same shape as DP-FP-5; sweep `frequencyChangeWeight` alone.

#### DP-FP-7: track-count cap in transition-cost loop

See `TRANSFERABLE_FINDINGS.md` §"DP: FormantPath Viterbi track count
cap". This one is a single-knob change with very large parity
consequences — definitely test it.

### 4.3 Stress via Legendre polynomial fit 📚 STANDARD + 🔬 DETERMINE

For each candidate `i` at frame `t`:

1. Take the formant tracks (one polynomial per track, one fit per
   frame) over a `pathWindowLength` window centred at `t`.
2. For each track, fit a Legendre polynomial of order
   `parameters[k] - 1` to that track's `(time, frequency)` data over
   the window, using weighted least squares with weight per data
   point coming from the bandwidth (see DP-FP-3 for convention).
3. Combine the per-track residuals and parameter-variance estimates
   into a single scalar "stress" for `(candidate i, frame t)`.

**Standard parts**:
- Legendre basis on `[-1, 1]` with the Bonnet recursion (Numerical
  Recipes Ch. 5).
- Weighted-least-squares solve via SVD (Numerical Recipes Ch. 15).
- Standard chi-squared / weighted-residual sums.

**DP parts**:

#### DP-FP-8: exact stress formula

The literature pins `stress = f(parameter_variance, chi_squared,
power, dof)` for some `f` involving the `power` parameter. Several
plausible forms exist (e.g., `sqrt(var^power · chisq/dof)`,
`(var^power · chisq)^(1/2)`, etc.).

**Experiment.** If you have access to a Praat ≥ 6.2 via parselmouth,
`call(fp, "Get stress of candidate", ...)` gives you a black-box stress
oracle. Pick a known-input case where you can compute `var`, `chisq`,
`dof`, `power` independently, and fit `f` to Praat's output. If you're
stuck on Praat 6.1.38, you can extract a `Stress`-style Matrix via
`call(fp, "To Matrix (stress)", ...)` (available in 6.1.38) which
gives you the same oracle on a per-cell basis.

#### DP-FP-9: SVD tolerance for the LS solve

**Hypothesis.** Either a fixed tolerance (e.g., `1e-5`, `1e-12`) or a
data-dependent one (`n * eps`).

**Experiment.** Construct a near-rank-deficient design matrix and see
when Praat's reported stress changes vs your reproduction.

### 4.4 Intensity modulation ✅ DOCUMENTED (Praat manual)

The manual says intensity modulation re-weights the per-state cost
based on each frame's energy: louder frames carry more cost weight
than quiet ones, governed by a sigmoid whose half-width is
`intensityModulationStepSize` dB.

#### DP-FP-10: exact sigmoid argument

Plausible forms: `(dBi - dBmid) / step`, where `dBi` is the
candidate's frame intensity in dB and `dBmid` is some reference.
Candidates for `dBmid`: median dB across the signal, mean dB, dB at
geometric-mean intensity of `[min, max]`. The reference cancels in
practice (only differences matter), but the *shape* of the sigmoid
argument may not.

**Experiment.** Construct a signal with two energy levels (loud + quiet
sections), and `Path finder` should pick the loud-section path more
strongly. Probe at varying `intensityModulationStepSize` to recover
the sigmoid shape.

### 4.5 Viterbi mechanics 📚 STANDARD + ⚠️ INTERNAL

The forward pass is the standard Forney Viterbi: `delta[i][t] = static
cost(i,t) + min_j(delta[j][t-1] + transition(j→i))`, plus path
back-tracking via `psi`.

#### DP-FP-11: Viterbi tie-breaking, sign, and accumulation rules

⚠️ Some of these are internal DP details that may NOT be observable
black-box. Praat's reference implementation has at least two
non-obvious behaviours that the GPL sibling found by source reading;
they may or may not show up in your I/O probes:

1. *Possible double-counting of static cost at non-initial frames.*
2. *Possible argmax-at-end (vs argmin-at-end) for back-tracking
   start.*

**Strategy.** Don't try to match these from black-box first. Build a
straightforward Viterbi with the cost components you've recovered
from DP-FP-5..10, accept ~70–90% path agreement with Praat, and
revisit only if your downstream needs absolute parity. The GPL
sibling's experience: closing the last 10–30% by reverse-engineering
internals didn't move final formant accuracy enough to justify the
investigation effort, at least at single-Hz tolerance.

If you do need exact parity and have parselmouth ≥ Praat 6.2, the
`To Matrix (stress)` / `To Matrix (qsums)` exports plus
`Get candidate in frame` give you enough oracles to invert the cost
function reasonably.

---

## 5. `Extract Formant` ✅ DOCUMENTED

For each frame `t`, copy the `path[t]`-th candidate's frame contents
into the output Formant. The output's `(t1, dx, nx, maxFormants)`
match the candidates' shared grid (see DP-FP-1).

---

## 6. Verification recipe

### 6.1 Ceiling parity (trivially testable)

```python
expected = [middleCeiling * exp((i - N) * ceilingStepSize)
            for i in range(2*N + 1)]
# Compare to your FormantPath.ceilings — exact-match expected.
```

### 6.2 Per-candidate Formant parity vs **the FormantPath** (not standalone)

For each ceiling `i`, run `Path finder` with weights chosen to force
the path to candidate `i` everywhere (e.g., set all weights to 0
except for one that strongly prefers candidate `i`'s ceiling — this
takes some experimentation), `Extract Formant`, and compare to your
candidate `i`'s formants. **Do not** compare to a standalone `To
Formant (burg)` at the same ceiling — those will systematically
differ (DP-FP-2).

If `parselmouth.praat.call` ≥ 6.2 is available, `Get candidate in
frame` lets you read out the path directly without forcing.

### 6.3 Extracted Formant after path finding (primary gate)

```python
import parselmouth as pm
from parselmouth.praat import call
snd = pm.Sound("audio.wav")
fp  = call(snd, "To FormantPath (burg)",
           0.005, 5, 5500.0, 0.025, 50.0, 0.05, 4)
call(fp, "Path finder",
     0.5, 0.5, 0.5, 0.5, 5.0, 0.035, "3 3 3 3", 1.25)
formant = call(fp, "Extract Formant")
# Query at every frame midpoint, compare to your implementation.
```

Acceptance target (suggested): match Praat's extracted F1/F2/F3 to
≤ 1 Hz on 99 % of frames on a substantial test set (e.g., Hillenbrand
1995's 1609 vowel tokens). The remaining 1 % will be tokens where the
Viterbi makes a different choice from Praat's; that's expected unless
you've fully closed DP-FP-7 and DP-FP-11.

### 6.4 Stress matrix parity (newer Praat only)

```python
m = call(fp, "To Matrix (stress)", 0.035, "3 3 3 3", 1.25)
# Available in 6.1.38; exposes per-(candidate, frame) stress.
```

This is by far the strongest single oracle for DP-FP-8/9 and your
overall FormantModeler implementation.

---

## 7. Gotchas

1. **`Get value at time` returns NaN outside the frame range.** Clamp
   `t` to `[t1, t1 + (nx-1)·dx]` before querying short-clip formants.
2. **1-based vs 0-based.** Praat is 1-based throughout (frames,
   candidates, tracks, parameter indices). Internally use whatever you
   like, but make sure your public API matches Praat semantics.
3. **`pathWindowLength` × `parameters` interaction.** When
   `stressWeight > 0`, Praat validates that `pathWindowLength`
   provides enough data points for `max(parameters)` coefficients; too
   few points and `Path finder` errors out. Either keep
   `pathWindowLength ≥ max(parameters) · timeStep`, or skip stress
   computation entirely when `stressWeight == 0`.

---

## 8. Decision-point summary

| # | Question | Distinguishing experiment |
|---|---|---|
| FP-1 | Frame grid shared from middle ceiling? | Compare FormantPath grid to per-ceiling standalone |
| FP-2 | Per-candidate Burg differs from standalone? | Default-path Extract vs `Sound: To Formant (burg)` |
| FP-3 | Bandwidth K convention | Synth signal with known poles |
| FP-4 | Frame-intensity quantity | Vary signal energy, observe path |
| FP-5 | Stress / Q-sum normalisation cutoffs | Sweep weights, observe saturation |
| FP-6 | Frequency-change cost normalisation | Sweep `frequencyChangeWeight` alone |
| FP-7 | Track-count cap in transition cost | High-F2 vowel test |
| FP-8 | Exact stress formula | Probe via `To Matrix (stress)` oracle |
| FP-9 | SVD tolerance | Near-rank-deficient design |
| FP-10 | Intensity-sigmoid argument shape | Two-energy-level signal |
| FP-11 | Viterbi internal accumulation/back-tracking | Possibly unobservable; see body |

---

[praat-manual]: https://www.fon.hum.uva.nl/praat/manual/FormantPath.html
