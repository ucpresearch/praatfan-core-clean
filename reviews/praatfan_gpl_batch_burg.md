# Feature request: batch `to_formant_burg` for multiple ceilings

**Requested by:** praatfan-core-clean (unified selector API)
**Affected project:** `praatfan-core-rs` / `praatfan_gpl` (PyO3 bindings +
pure-Rust crate)
**Related request:** `reviews/praatfan_batch_burg.md` — same feature, already
implemented in praatfan-core-clean (pure Python + praatfan_rust).

## Summary

Add a batched variant of `Sound.to_formant_burg` that accepts a list of
`maximum_formant` ceilings and returns a parallel list of `Formant` objects
from a single Python → Rust (or Rust → Rust) call.

No algorithmic change. Each entry in the batch does exactly what the singular
call does today. The motivation is amortizing FFI overhead and unlocking
per-ceiling parallelism via rayon.

## Context

Downstream pipelines sweep Burg ceilings (e.g. an 11-point grid 4500–6500 Hz)
across many short speech regions per file, with all other parameters
(`time_step`, `max_number_of_formants`, `window_length`, `pre_emphasis_from`)
fixed. A typical workload calls `to_formant_burg` tens of thousands of times
per file, most on sub-sounds of ~200–400 ms. Each call pays Python ↔ Rust FFI
overhead and runs serially even though the ceilings are independent. A batch
entry point amortizes FFI per region and lets the backend parallelize the
per-ceiling work internally. Preliminary estimates on a representative
pipeline show extract-path wall time dropping by roughly 2× when the batch
path is available with rayon-parallel per-ceiling evaluation.

praatfan-core-clean's unified selector has a matching entry point
(`Sound.to_formant_burg_multi`) and will transparently pick up
`praatfan_gpl`'s native implementation via `hasattr`:

```python
# src/praatfan/selector.py  (PraatfanCoreSound adapter)
def to_formant_burg_multi(self, maximum_formants, time_step=0.0,
                          max_number_of_formants=5, window_length=0.025,
                          pre_emphasis_from=50.0):
    ceilings = [float(hz) for hz in maximum_formants]
    if hasattr(self._inner, "to_formant_burg_multi"):
        results = self._inner.to_formant_burg_multi(
            ceilings, time_step, max_number_of_formants,
            window_length, pre_emphasis_from,
        )
        return [UnifiedFormant(r, self.BACKEND) for r in results]
    # ... falls back to a Python loop over singular calls
```

So adding the method is purely additive and requires no coordination on the
consumer side — installing a newer praatfan_gpl automatically upgrades the
batch path from loop to native rayon.

## Proposed API

### PyO3 (Python)

```python
class Sound:
    def to_formant_burg_multi(
        self,
        maximum_formants: Sequence[float],
        time_step: float = 0.0,
        max_number_of_formants: int = 5,
        window_length: float = 0.025,
        pre_emphasis_from: float = 50.0,
    ) -> list[Formant]:
        """Run Burg LPC for each ceiling in *maximum_formants*.

        Equivalent to calling :meth:`to_formant_burg` once per ceiling,
        but amortizes FFI overhead and parallelizes per-ceiling work via
        rayon. Returns a list of Formant objects in the same order as
        *maximum_formants*. Empty input → empty output.
        """
```

Argument order: `maximum_formants` first as positional (no default); all other
kwargs with defaults matching singular `to_formant_burg`. This matches the
PyO3 signature used in praatfan_rust:

```rust
#[pyo3(signature = (maximum_formants, time_step=0.0, max_number_of_formants=5,
                    window_length=0.025, pre_emphasis_from=50.0))]
fn to_formant_burg_multi(
    &self,
    py: Python<'_>,
    maximum_formants: Vec<f64>,
    time_step: f64,
    max_number_of_formants: usize,
    window_length: f64,
    pre_emphasis_from: f64,
) -> Vec<PyFormant> {
    let formants = py.allow_threads(|| {
        self.inner.to_formant_burg_multi(
            time_step, max_number_of_formants,
            &maximum_formants, window_length, pre_emphasis_from,
        )
    });
    formants.into_iter().map(|inner| PyFormant { inner }).collect()
}
```

Key details:

- `py.allow_threads(...)` around the rayon call releases the GIL so other
  Python threads run during the batch. Standard PyO3 practice; not a
  correctness requirement but expected hygiene.
- `Vec<f64>` at the PyO3 boundary is passed as `&[f64]` to the core function.

### Pure Rust

```rust
impl Sound {
    pub fn to_formant_burg_multi(
        &self,
        time_step: f64,
        max_num_formants: usize,
        maximum_formants: &[f64],
        window_length: f64,
        pre_emphasis_from: f64,
    ) -> Vec<Formant>;
}
```

`&[f64]` (not `Vec<f64>`) so Rust callers depending on the crate directly can
pass any slice/array without allocating.

### Core implementation

```rust
pub fn sound_to_formant_burg_multi(
    sound: &Sound,
    time_step: f64,
    max_num_formants: usize,
    maximum_formants: &[f64],
    window_length: f64,
    pre_emphasis_from: f64,
) -> Vec<Formant> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        maximum_formants
            .par_iter()
            .map(|&hz| sound_to_formant_burg(
                sound, time_step, max_num_formants, hz,
                window_length, pre_emphasis_from,
            ))
            .collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        maximum_formants
            .iter()
            .map(|&hz| sound_to_formant_burg(
                sound, time_step, max_num_formants, hz,
                window_length, pre_emphasis_from,
            ))
            .collect()
    }
}
```

The `cfg(target_arch = "wasm32")` guard matters because default rayon relies
on OS threads. In a WASM build without `wasm-bindgen-rayon`, `par_iter` panics
at runtime. The sequential fallback keeps the function callable from any
downstream crate's browser WASM build without thread-pool setup. Callers who
want parallel WASM can opt into `wasm-bindgen-rayon` at their own crate level
— no change needed here.

### Cargo.toml

Add rayon as a native-only dep:

```toml
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
rayon = "1"
```

## Non-goals

- No change to the singular `to_formant_burg` signature or semantics.
- No change to the Burg algorithm, resample step, or pre-emphasis math.
- No batched variants for pitch / intensity / HNR — those aren't swept the
  same way. Could be added later if similar patterns emerge.
- Not asking for batching across multiple Sounds — each region's sub-sound is
  still one call, just with N ceilings.

## Test

Bit-parity with repeated singular calls, which is exactly the check
praatfan-core-clean already runs (`tests/test_formant_multi.py`):

```python
ceilings = [4500, 4700, 4900, ..., 6500]
refs = [
    sound.to_formant_burg(maximum_formant=hz, window_length=0.025, ...)
    for hz in ceilings
]
batch = sound.to_formant_burg_multi(
    maximum_formants=ceilings, window_length=0.025, ...)
for r, b in zip(refs, batch):
    for k in range(1, 5):
        assert np.array_equal(b.formant_values(k),  r.formant_values(k),
                              equal_nan=True)
        assert np.array_equal(b.bandwidth_values(k), r.bandwidth_values(k),
                              equal_nan=True)
```

Empty input and single-ceiling cases must also behave as expected (empty list
/ one-element list equivalent to the singular call).

## Reference implementation

Already landed in praatfan-core-clean as of 2026-04-19. Relevant files for
cross-reference:

- `rust/Cargo.toml` — target-gated `rayon = "1"` dep.
- `rust/src/formant.rs` — `sound_to_formant_burg_multi`.
- `rust/src/sound.rs` — `Sound::to_formant_burg_multi` wrapper.
- `rust/src/python.rs` — PyO3 `to_formant_burg_multi` on `PySound`.
- `src/praatfan/selector.py` — unified selector entry point +
  `PraatfanCoreSound.to_formant_burg_multi` with `hasattr`-based native
  dispatch.
- `tests/test_formant_multi.py` — parametrized parity tests across all
  available backends (praatfan, praatfan_rust, parselmouth, praatfan_gpl).

Parity confirmed across all four backends on the praatfan-core-clean side;
adding the native method on praatfan_gpl will upgrade the dispatch from
Python-loop fallback to native-rayon automatically.

## Contact

Questions: `uriel.cohen.priva@gmail.com`.
