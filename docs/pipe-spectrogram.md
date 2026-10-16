# Pipe spec: `spectrogram` analysis, and the embedded-V8 bundle contract

Status: agreed between the two pipe engines (`praatfan-open-pipe`, MIT, and
`praatfan-gpl-pipe`, GPL) on 2026-10-16; not yet implemented. Both engines
implement this text independently. It describes requests, responses and
file layouts only; no implementation is shared.

## 1. Request

```json
{"type": "spectrogram",
 "window_length": 0.005, "max_frequency": 5000.0,
 "time_step": 0.002, "frequency_step": 20.0,
 "window_shape": "gaussian",
 "dtype": "f64le", "start_time": null, "end_time": null}
```

| key | default | meaning |
|---|---|---|
| `window_length` | 0.005 | effective window length, s (as Praat's "Sound: To Spectrogram") |
| `max_frequency` | 5000 | ceiling, Hz. `<= 0` or above Nyquist is clamped to Nyquist (not an error) |
| `time_step` | 0.002 | requested time step, s (the step actually used is reported in the result) |
| `frequency_step` | 20 | requested frequency step, Hz (likewise) |
| `window_shape` | `"gaussian"` | `"gaussian"` or `"hanning"`; anything else is an error |
| `dtype` | `"f64le"` | `"f64le"` or `"f32le"`: encoding of `power_b64` |
| `start_time`, `end_time` | absent | restrict the *returned* frames to those whose time lies in `[start_time, end_time]` (either may be absent). The grid is the full-sound grid, so returned times are identical to the corresponding frames of an unrestricted request |

Every key is optional. Unknown keys are errors, as for every analysis. The
sound is the request's single channel; multichannel files need `channel`, as
for every analysis. Callers asking for "narrow" vs "wide" band pass a
`window_length` (the R package maps `"wide"` to 0.005 s and `"narrow"` to
0.025 s); the engine sees only the number.

## 2. Result

```json
{"type": "spectrogram",
 "n_times": 735, "n_freqs": 250,
 "time_step": 0.002, "frequency_step": 20.0,
 "times": [...], "freqs": [...],
 "layout": "time-major", "dtype": "f64le",
 "power_b64": "..."}
```

- `times` (frame centres, s) and `freqs` (bin centres, Hz) are the
  authoritative grid. `time_step`/`frequency_step` echo the steps actually
  used.
- `power_b64`: base64 (standard alphabet, padded, no line breaks) of
  `n_times * n_freqs` values in `dtype`, little-endian, time-major: the value
  for frame `t`, bin `f` is element `t * n_freqs + f`.
- Values are linear power, as Praat's Spectrogram object holds them; no dB
  conversion in the engine.
- A sound shorter than the analysis window gives an empty result, not an
  error: `n_times: 0`, `times: []`, `power_b64: ""`, response `ok: true`.
  `freqs` and `n_freqs` are still reported.
- Parity is per engine, not across engines. The GPL engine targets Praat's
  `To Spectrogram` exactly. The open engine is a standard Gaussian/Hanning
  STFT with the same time grid at common settings but its own frequency
  grid and values (within a few percent of Praat's; it does not chase
  parity — use the GPL engine for that). Callers must read the grid from
  `times`/`freqs`, never assume it; only the response *shape* is uniform.

### Errors and edge cases

`ok: false` (with `error`) when:

- `window_length`, `time_step` or `frequency_step` is not finite and > 0;
- `max_frequency`, `start_time` or `end_time` is not finite;
- `start_time > end_time`;
- `window_shape` is not `gaussian`/`hanning`, or `dtype` not `f64le`/`f32le`.

`ok: true`:

- explicit `null` for `start_time`/`end_time` means absent;
- `max_frequency` of 0, negative, or above Nyquist is clamped to Nyquist
  (parselmouth rejects `<= 0`, so black-box comparisons pass it Nyquist for
  that case);
- a time range that selects no frames gives the empty result above.

Cross-build parity (native vs wasm32, different libm) is close but not
bit-identical; compare with relative tolerances, not equality.

Size note: the defaults give 500 frames/s × 250 bins = 125k values/s, i.e.
1 MB/s of f64 (1.3 MB/s of base64). Long files should use `f32le` and/or a
time range.

## 3. Additive pipe features (both engines)

- Response `"warnings"`: array of strings, omitted when empty — messages the
  binary would otherwise print only to stderr.
- In-memory transports (WASM builds), each returning the same JSON text as
  the binary would:
  - `pipe_request_bytes(Uint8Array file_bytes, String json)` — `wav_path`
    optional, used only as a format hint (extension) and name;
  - `pipe_request_samples(Float64Array samples, Number sample_rate, String
    json)` — refuses `wav_path` and `channel`, and a `sample_rate` that is
    not finite and > 0.
  - Undecodable bytes give `ok: false`, naming `wav_path` if given, else
    `"<bytes>"`.

## 4. Embedded-V8 bundle contract (for the R package's `"wasm"` engines)

- wasm-bindgen `--target no-modules`: defines a global `wasm_bindgen`,
  initialised with `wasm_bindgen.initSync({module: <wasm bytes>})`.
- Exports `version()` (the crate version string), `pipe_request_bytes`,
  `pipe_request_samples`.
- Runs in a bare V8: no `TextEncoder`/`TextDecoder`, `atob`, `console` of its
  own. The host loads praatfan-core-clean's `rust/wasm-v8/shim.js` (MIT)
  first; it provides `TextEncoder`, `TextDecoder`, a `console` fallback,
  `praatfan_b64decode` and `praatfan_b64bytes`. This contract is kept stable;
  additions only.
- Plain wasm32 target features (rustc defaults): no `simd128`, no relaxed
  SIMD. R's V8 package links the system V8 on Linux (12.4 on the reference
  machine, via libnode) and older ones exist in the wild.
- A Rust panic traps (wasm32 aborts); hosts catch the trap, report it, and
  discard the instance.
- No builder paths in the `.wasm` (`--remap-path-prefix`).

### GPL release assets (downloaded on demand by the R package)

The GPL bundle is not shipped inside the MIT R package; the R package
downloads it on request (`pf_setup(engine = "gpl_wasm")`), as
`pf_setup(engine = "gpl")` already does for `praatfan-gpl-pipe`.

- `praatfan-gpl-wasm-v8.zip`: `praatfan_gpl.js`, `praatfan_gpl_bg.wasm`,
  `LICENSE` (GPL-3.0), `PROVENANCE.txt` (commit, rustc and wasm-bindgen
  versions).
- `SHA256SUMS`: one file covering every release asset (the zip and the
  `praatfan-gpl-pipe-*` binaries), `sha256sum` format; the R package verifies
  every download against it.
