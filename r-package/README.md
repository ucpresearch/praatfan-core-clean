# praatfan for R

Praat-compatible acoustic analysis in R: pitch (F0), formants (Burg LPC),
intensity, harmonicity (HNR), spectral moments, and band energy, with values
validated frame-by-frame against Praat. All analysis runs in a small native
Rust binary (`praatfan-open-pipe`, JSON stdin/stdout) — no Python, pip, or
conda required.

## Install

```r
# install.packages("remotes")
remotes::install_github("ucpresearch/praatfan-core-clean", subdir = "r-package")
```

That's it — the native engine binary is fetched automatically on first use
(disable with `options(praatfan.auto_setup = FALSE)`). To fetch it
explicitly — e.g. to pre-provision a cluster node, pin a version with
`ref =`, or install the GPL engine — call `pf_setup()`.

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
h <- pf_harmonicity("audio.wav")              # time, hnr (dB)

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

Two interchangeable native engines speak the same JSON schema; every
analysis function works identically with either:

| Engine | Binary | License | Notes |
|--------|--------|---------|-------|
| `"open"` (default) | `praatfan-open-pipe` | MIT | Clean-room implementation, validated frame-by-frame against Praat |
| `"gpl"` | `praatfan-gpl-pipe` | GPL | Tracks Praat's values even more closely |

```r
pf_setup(engine = "gpl")     # install alongside the default engine
pf_engine("gpl")             # use it for the rest of the session
pf_engine()                  # query the current engine

# Or per call, e.g. to compare engines side by side:
a <- pf_analyze("audio.wav", list(list(type = "formant_burg")), engine = "open")
b <- pf_analyze("audio.wav", list(list(type = "formant_burg")), engine = "gpl")
```

Selection precedence: `PRAATFAN_PIPE` (explicit binary path, always wins) >
`pf_engine(...)` / `pf_analyze(engine = ...)` > `PRAATFAN_ENGINE`
environment variable > default `"open"`. The engine choice affects only
which subprocess runs; this package itself remains MIT either way.

## Notes

- For generic spectra/spectrograms, R is already well served (seewave,
  phonTools, wrassp); this package deliberately carries only the
  Praat-parity values those tools can't replicate.
- `pf_uninstall()` removes the installed binary and cached sources.
