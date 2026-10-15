# praatfan for R

Praat-compatible acoustic analysis in R: pitch (F0), formants (Burg LPC),
intensity, harmonicity (HNR), spectral moments, and band energy, with values
validated frame-by-frame against Praat. The Rust engine ships inside the
package compiled to WebAssembly and runs in R through the `V8` package, so it
works as soon as the package is installed; the native binary
(`praatfan-open-pipe`, JSON stdin/stdout) is an optional speed-up with the
same numbers. No Python, pip, or conda required.

## Install

```r
# install.packages("remotes")
remotes::install_github("ucpresearch/praatfan-core-clean", subdir = "r-package")
```

That's it — with no native binary installed, analysis runs in the WASM
engine inside R (a message says so once per session). For native speed —
several threads per file — install the binary with `pf_setup()`; once it is
present it is used automatically. `pf_setup()` also pre-provisions cluster
nodes, pins a version with `ref =`, and installs the GPL engine.

`pf_setup()` tries, in order: a pre-built binary from GitHub Releases, a git
clone built with cargo, and a local source tree. Point the package at an
existing binary instead with `PRAATFAN_PIPE=/path/to/praatfan-open-pipe`.

## Use

Every function takes an audio file (WAV, FLAC, MP3, OGG, AIFF, NIST SPHERE
incl. shorten) and returns a tidy data.frame. Mono only; pass `channel` to
pick one channel of a multi-channel file.

```r
p <- pf_pitch("audio.wav")                    # time, f0 (NA = unvoiced), strength, voiced
f <- pf_formants("audio.wav",                 # time, n_formants, F1..F5, B1..B5
                 max_formant_hz = 5000)
i <- pf_intensity("audio.wav")                # time, intensity (dB)
h <- pf_harmonicity("audio.wav")              # time, hnr (dB; -200 = no periodicity, NA = undefined)

# Praat-parity spectral scalars at chosen times
m <- pf_spectral_moments("audio.wav", times = c(0.10, 0.15, 0.20))
b <- pf_band_energy("audio.wav", times = c(0.10, 0.15), f_min = 0, f_max = 1000)
```

Analysis parameters mirror Praat's commands (`pf_pitch` also exposes the
Boersma 1993 tuning set: `voicing_threshold`, `octave_cost`, ...). Omitted
parameters take Praat's defaults; misspelled parameter names are errors, not
silent fallbacks.

To compute several measures from one file load, use the batch interface:

```r
res <- pf_analyze("audio.wav", list(
  list(type = "pitch_ac"),
  list(type = "formant_burg", max_formant_hz = 5000),
  list(type = "spectral_moments", times = I(c(0.10, 0.15)))
))
```

(Wrap vector fields like `times` in `I()` so length-1 vectors stay JSON
arrays.)

## Engines

Three interchangeable engines speak the same JSON schema; every analysis
function works identically with any of them:

| Engine | Runs as | License | Notes |
|--------|---------|---------|-------|
| `"open"` | `praatfan-open-pipe` binary | MIT | Clean-room implementation, validated frame-by-frame against Praat |
| `"wasm"` | the open engine as WebAssembly, in `V8` | MIT | Ships with the package; same numbers as `"open"` (to floating-point rounding), one thread per call |
| `"gpl"` | `praatfan-gpl-pipe` binary | GPL | Tracks Praat's values even more closely |

With no engine chosen, `"open"` is used when its binary is available and
`"wasm"` otherwise.

```r
pf_setup(engine = "gpl")     # install alongside the default engine
pf_engine("gpl")             # use it for the rest of the session
pf_engine()                  # query the current engine

# Or per call, e.g. to compare engines side by side:
a <- pf_analyze("audio.wav", list(list(type = "formant_burg")), engine = "open")
b <- pf_analyze("audio.wav", list(list(type = "formant_burg")), engine = "gpl")
```

Selection precedence: `pf_engine(...)` / `pf_analyze(engine = ...)` >
`PRAATFAN_ENGINE` environment variable > `"open"` if its binary is available,
else `"wasm"`. `PRAATFAN_PIPE` (an explicit binary path) always wins for the
native engines and makes the open binary available. The engine choice affects only
which subprocess runs; this package itself remains MIT either way.

## Notes

- For generic spectra/spectrograms, R is already well served (seewave,
  phonTools, wrassp); this package deliberately carries only the
  Praat-parity values those tools can't replicate.
- The `"open"` engine's binary uses all CPU cores for each file. When you
  run analyses in parallel yourself (`parallel::mclapply`, `future`, ...),
  limit its threads with `Sys.setenv(RAYON_NUM_THREADS = 1)` (or cores ÷
  workers) before the first call. Results are identical at any thread count.
- The `"wasm"` engine runs one thread per call; to use more cores, analyse
  several files at once in separate R processes (PSOCK clusters,
  `future::plan(multisession)`). Each process builds its own engine.
- `pf_uninstall()` removes the installed binary and cached sources.
