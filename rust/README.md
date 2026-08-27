# praatfan-rust

Fast Rust implementation of Praat's core acoustic analysis algorithms, with
Python bindings. MIT-licensed.

`praatfan_rust` is the compiled backend for
[`praatfan`](https://pypi.org/project/praatfan/) — a clean-room
reimplementation of Praat's acoustic analysis, written from published papers
and Praat's public documentation rather than from Praat's GPL source, so it can
be used under a permissive licence.

```bash
pip install praatfan-rust
```

It can be used directly, or installed alongside `praatfan` and selected
automatically through that package's unified API.

## What it computes

| Analysis | Methods |
|---|---|
| Pitch (F0) | autocorrelation, cross-correlation |
| Formants | Burg LPC, single and batched over ceilings |
| Intensity | — |
| Harmonicity (HNR) | autocorrelation, cross-correlation |
| Spectrum | FFT, plus centre of gravity / spread / skewness / kurtosis |
| Spectrogram | STFT |

Audio is read via [Symphonia](https://github.com/pdeljanov/symphonia) (WAV,
FLAC, MP3, OGG, AIFF, CAF and more). NIST SPHERE, including
shorten-compressed, decodes through a statically linked
[desphere](https://github.com/ucpresearch/desphere) — no extra dependency and
no external binary. Mono only; multi-channel files require an explicit channel
selection.

## Example

```python
import praatfan_rust

sound = praatfan_rust.Sound("audio.wav")

pitch = sound.to_pitch_ac(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0)
formant = sound.to_formant_burg(time_step=0.0, max_number_of_formants=5,
                                maximum_formant=5500.0)
harmonicity = sound.to_harmonicity_ac(0.01, 75.0, 0.1, 4.5)
```

## Reading harmonicity values

Two entries in an HNR contour are markers rather than measurements:

- `-200` — no periodicity was found (silence, DC, unvoiced).
- `NaN` — the correlation is not resolvable from 1, so no HNR can be derived.

Filter both before averaging, e.g. `v[np.isfinite(v) & (v > -200)]`. The
autocorrelation method also requires `periods_per_window >= 3.0`; below that
the estimator is ill-conditioned and emits a `FutureWarning` (this will become
an error in a future release). `4.5` is the standard value for the AC method;
`1.0` is the CC default, not an AC one.

## Also available

- **WebAssembly** — the same core runs in the browser via `wasm-pack`.
- **`praatfan-open-pipe`** — a JSON-over-stdin/stdout binary used by the R
  package, so R and shell callers pay the audio load once per file.

## Links

- Source, documentation and issues:
  <https://github.com/ucpresearch/praatfan-core-clean>
- Pure-Python sibling: <https://pypi.org/project/praatfan/>

## Licence

MIT.
