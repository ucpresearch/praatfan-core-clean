# praatfan

A clean-room reimplementation of Praat's acoustic analysis algorithms, with a unified API that supports multiple backends.

> **Note:** Latest release **v0.1.14**. The API is stabilizing but may still
> change.
>
> **v0.1.15 (unreleased) changes analysis values — read before upgrading.**
> Every backend (pure Python, Rust, WASM) changes the same way:
> - **Intensity:** frames are now centred, with Praat's frame count (often
>   one more frame than before), and use the documented Kaiser-20 window.
>   Per-frame values move by up to a few dB, toward Praat's. Silence is
>   still `-inf`.
> - **Pitch:** each frame's local average is subtracted before
>   autocorrelation (Boersma 1993). White noise and DC offsets are no longer
>   reported as voiced, and voicing decisions on speech change in a few frames.
>   Pitch and harmonicity frame counts can differ by one at exact boundaries.
> - **Harmonicity:** the pitch tracker runs with Boersma's HNR settings
>   (voicing threshold 0, no path finder). Noise now gets a finite HNR instead
>   of -200, and a DC signal reads -200 instead of `nan`.
> - **Formant:** audio below twice the formant ceiling (e.g. 8 kHz with
>   5500 Hz) is now resampled up to twice the ceiling, as Praat does.
>   Previously it was analysed as if the ceiling were its Nyquist frequency.
> - **Spectrogram:** a maximum frequency above Nyquist is clamped to
>   Nyquist (pure Python used to raise an error).
> - **`Sound.resample`:** now resamples up as well as down; a higher rate
>   used to return a copy at the old rate. Also available in the WASM build.
>
> **v0.1.14** makes the Rust backend give the same numbers on every machine:
> its maths functions and FFTs no longer depend on the host's maths library or
> on which CPU features it detects at run time. Values move from v0.1.13 only
> in the last digits: at most ~1.5e-11 relative (formants by up to ~1e-8 Hz,
> F0 by ~1e-12 Hz). Single-threaded pitch and harmonicity analysis is up to ~1.8×
> slower; with all cores the cost is ~1.2–1.3×. Every binary now ships its
> third-party license notices. See [Performance and threading](#performance-and-threading).
>
> **v0.1.13** updates NIST SPHERE decoding to desphere 0.1.1: malformed files
> now fail with a clear error instead of being read incorrectly. Analysis
> values are unchanged.
>
> **v0.1.12 — read before upgrading:** unvoiced F0 is now `nan` on every
> backend (`Pitch.values()`, and `call(..., "Get value in frame")`), where some
> backends returned 0. It also adds the `praatfan_wasm` backend (the Rust
> engine as WebAssembly, `pip install "praatfan[wasm]"`), the `spectrogram`
> analysis in the JSON pipe and R package, and a `SHA256SUMS` file on each
> release.
>
> **v0.1.11** runs the Rust backend's analyses on all CPU cores. Results are
> bit-identical to v0.1.10; see [Performance and threading](#performance-and-threading)
> for how to limit the thread count. It also adds a native Windows ARM64
> build of `praatfan-open-pipe`.
>
> **v0.1.10 changed harmonicity output values — read before upgrading from
> v0.1.9 or earlier.**
> `To Harmonicity (ac)` / `To Harmonicity (cc)` now report `nan` for frames
> whose normalized correlation is not resolvable from 1 in double precision.
> Earlier releases clamped those frames to a fixed ~100 dB (AC) or to -200 dB
> (CC), which made an arithmetic boundary look like a measurement. Frames with
> no periodicity at all still report -200 dB. Downstream code that averages or
> thresholds HNR must now handle `nan` (e.g. `numpy.nanmean`).
>
> `periods_per_window < 3.0` on the AC method now raises a `FutureWarning`; the
> autocorrelation estimator is ill-conditioned below that value and it will
> become an error in a later release. Praat's standard value for this method is
> 4.5; use the cross-correlation method if you need 1.0.
>
> Also new in v0.1.10: the `praatfan-open-pipe` JSON binary and the R package
> that wraps it. The v0.1.9 NIST SPHERE support is unchanged.

## Installation

### From PyPI (recommended)

```bash
# Pure Python implementation (works everywhere)
pip install praatfan

# Rust-accelerated backend (optional, faster)
pip install praatfan-rust

# + NIST SPHERE support (TIMIT/WSJ/Switchboard, incl. shorten)
pip install "praatfan[sphere]"
```

Both packages are on PyPI: [praatfan](https://pypi.org/project/praatfan/) | [praatfan-rust](https://pypi.org/project/praatfan-rust/)

Available `praatfan-rust` wheels (single abi3 wheel per platform, CPython 3.9+):
- **Linux**: x86_64, aarch64
- **macOS**: ARM64, x86_64 (Intel)
- **Windows**: x86_64, ARM64

### NIST SPHERE (`.sph`) files

`Sound.from_file` reads SPHERE audio (incl. shorten-compressed) when the optional
[`desphere`](https://pypi.org/project/desphere/) package is installed —
`pip install "praatfan[sphere]"` pulls it with a Rust accelerator. The
`praatfan-rust` backend has SPHERE support compiled in, so it needs nothing extra.
Without desphere, libsndfile still handles uncompressed SPHERE.

### From source

```bash
# Pure Python
pip install git+https://github.com/ucpresearch/praatfan-core-clean.git

# Rust backend (requires Rust toolchain)
cd rust && pip install maturin && maturin develop --features python
```

### R

An R package wrapping a native Rust binary (`praatfan-open-pipe`, JSON
stdin/stdout — no Python required) lives in [`r-package/`](r-package/):

```r
remotes::install_github("ucpresearch/praatfan-core-clean", subdir = "r-package")
library(praatfan)
f0  <- pf_pitch("audio.wav")  # engine binary auto-installs on first use
fmt <- pf_formants("audio.wav", max_formant_hz = 5000)
```

See [`r-package/README.md`](r-package/README.md) for the full function list
(`pf_pitch`, `pf_formants`, `pf_intensity`, `pf_harmonicity`,
`pf_spectral_moments`, `pf_band_energy`, and batch `pf_analyze`). The same
`praatfan-open-pipe` binary is usable from any language that can spawn a
process and speak JSON (`praatfan-open-pipe --help` documents the schema).

## Quick Start

```python
from praatfan import Sound

# Load audio
sound = Sound("audio.wav")

# Or from numpy array
import numpy as np
samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
sound = Sound(samples, sampling_frequency=16000)

# Analyze
pitch = sound.to_pitch_ac()
formant = sound.to_formant_burg()
intensity = sound.to_intensity()

# Access results (unified API)
print(pitch.values())      # F0 in Hz (numpy array)
print(formant.formant_values(1))  # F1 in Hz
print(intensity.values())  # dB
```

## Backend System

praatfan supports multiple acoustic analysis backends through a unified API. Your code works the same regardless of which backend is active.

### Available Backends

| Backend | Package | License | Description |
|---------|---------|---------|-------------|
| `praatfan` | Built-in | MIT | Pure Python clean-room implementation |
| `praatfan_rust` | `praatfan_rust` (from wheels or maturin) | MIT | Rust implementation with PyO3 bindings |
| `parselmouth` | `praat-parselmouth` | GPL | Python bindings to Praat |
| `praatfan_gpl` | `praatfan_gpl` | MIT | Separate Rust implementation |
| `praatfan_wasm` | `praatfan[wasm]` (adds `mini-racer`) | MIT | The `praatfan_rust` engine compiled to WebAssembly, run in an embedded V8; pipe analyses only, the rest falls back to pure Python |

### Backend Selection

Backends are selected automatically in this order of preference:
1. `praatfan_rust` (if installed)
2. `praatfan_gpl` (if installed)
3. `praatfan_wasm` (if `praatfan[wasm]` is installed)
4. `praatfan` (always available)

`parselmouth` is never selected automatically; request it by name
(`set_backend("parselmouth")` or `PRAATFAN_BACKEND=parselmouth`).

You can override this through:

#### 1. Environment Variable

```bash
export PRAATFAN_BACKEND=parselmouth
python my_script.py
```

#### 2. Configuration File

Create `~/.praatfan/config.toml` or `./praatfan.toml`:

```toml
backend = "parselmouth"
```

#### 3. Runtime Selection

```python
from praatfan import set_backend, get_backend, get_available_backends

# Check what's available
print(get_available_backends())  # ['praatfan', 'parselmouth']

# Check current backend
print(get_backend())  # 'praatfan'

# Switch backend
set_backend('parselmouth')

# New Sound objects use the new backend
sound = Sound("audio.wav")  # Uses parselmouth
```

### Installing Additional Backends

#### parselmouth (GPL)

```bash
pip install praat-parselmouth
```

Note: parselmouth is GPL-licensed. If you need a permissive license, use the built-in `praatfan` backend.

#### praatfan_rust

The Rust-accelerated backend. Install from PyPI or build from source:

```bash
# From PyPI (recommended)
pip install praatfan-rust

# Or build from source (requires Rust toolchain)
cd rust && pip install maturin && maturin develop --features python
```

#### praatfan_gpl

A separate Rust implementation:

```bash
pip install praatfan-gpl
```

## Unified API

All backends return results with the same API, so you can switch backends without changing your analysis code.

### Sound

```python
from praatfan import Sound

# Load from file
sound = Sound("audio.wav")

# From numpy array
sound = Sound(samples, sampling_frequency=16000)

# Properties
sound.n_samples          # Number of samples
sound.sampling_frequency # Sample rate in Hz
sound.duration           # Duration in seconds
sound.values             # Samples as numpy array
```

### Pitch Analysis

```python
# Autocorrelation method (default)
pitch = sound.to_pitch_ac(
    time_step=0.0,      # 0 = auto
    pitch_floor=75.0,   # Min F0 in Hz
    pitch_ceiling=600.0 # Max F0 in Hz
)

# Cross-correlation method
pitch = sound.to_pitch_cc(time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0)

# Results
pitch.xs()        # Time values (numpy array)
pitch.values()    # F0 values in Hz (NaN for unvoiced)
pitch.strengths() # Voicing strength
pitch.n_frames    # Number of frames
pitch.time_step   # Time step between frames
pitch.backend     # Which backend produced this
```

### Formant Analysis

```python
formant = sound.to_formant_burg(
    time_step=0.0,              # 0 = auto
    max_number_of_formants=5,   # Max formants to find
    maximum_formant=5500.0,     # Max frequency in Hz
    window_length=0.025,        # Window in seconds
    pre_emphasis_from=50.0      # Pre-emphasis frequency
)

# Results
formant.xs()                  # Time values
formant.formant_values(1)     # F1 values (1-indexed)
formant.formant_values(2)     # F2 values
formant.bandwidth_values(1)   # B1 bandwidths
formant.n_frames              # Number of frames
```

### Intensity Analysis

```python
intensity = sound.to_intensity(
    minimum_pitch=100.0,  # Determines window size
    time_step=0.0         # 0 = auto
)

# Results
intensity.xs()      # Time values
intensity.values()  # Intensity in dB
intensity.n_frames  # Number of frames
```

### Harmonicity (HNR) Analysis

```python
# Autocorrelation method
harmonicity = sound.to_harmonicity_ac(
    time_step=0.01,
    minimum_pitch=75.0,
    silence_threshold=0.1,
    periods_per_window=4.5
)

# Cross-correlation method
harmonicity = sound.to_harmonicity_cc(
    time_step=0.01,
    minimum_pitch=75.0,
    silence_threshold=0.1,
    periods_per_window=1.0
)

# Results
harmonicity.xs()      # Time values
harmonicity.values()  # HNR in dB (-200 = no periodicity, nan = undefined)
harmonicity.n_frames  # Number of frames
```

### Spectrum Analysis

```python
spectrum = sound.to_spectrum(fast=True)

# Results
spectrum.xs()                       # Frequency values
spectrum.values()                   # Complex spectrum
spectrum.get_center_of_gravity()    # Spectral centroid
spectrum.n_bins                     # Number of frequency bins
spectrum.df                         # Frequency resolution
```

### Spectrogram Analysis

```python
spectrogram = sound.to_spectrogram(
    window_length=0.005,
    maximum_frequency=5000.0,
    time_step=0.002,
    frequency_step=20.0
)

# Results
spectrogram.xs()      # Time values
spectrogram.ys()      # Frequency values
spectrogram.values()  # Power values (2D: freq x time)
spectrogram.n_times   # Number of time frames
spectrogram.n_freqs   # Number of frequency bins
```

### Per-Window Spectral Analysis

Extract spectral features at specific time points (useful for analyzing spectral characteristics aligned with other measurements like formants or pitch):

```python
import numpy as np

# Extract a portion of the sound
part = sound.extract_part(0.1, 0.2)  # 100ms segment

# Get spectrum at a specific time
spectrum = sound.get_spectrum_at_time(0.15, window_length=0.025)
cog = spectrum.get_center_of_gravity()

# Batch extraction of spectral moments at multiple time points
times = np.array([0.1, 0.15, 0.2, 0.25])
moments = sound.get_spectral_moments_at_times(times, window_length=0.025)
# Returns dict with: 'times', 'center_of_gravity', 'standard_deviation',
#                    'skewness', 'kurtosis'

# Batch extraction of band energy
energy = sound.get_band_energy_at_times(times, f_min=0, f_max=1000)
```

## Migrating from parselmouth

There are two migration paths depending on your needs:

### Option 1: Drop-in replacement (minimal code changes)

Use `praatfan` with the `call()` function for **minimal code changes**. The `Sound()` constructor and `call()` function match parselmouth's API:

```python
# Before (parselmouth)
import parselmouth
from parselmouth.praat import call

snd = parselmouth.Sound("audio.wav")
pitch = call(snd, "To Pitch (ac)", 0, 75, 600)
f0 = call(pitch, "Get value in frame", 10, "Hertz")

# After (praatfan) - just change the imports
from praatfan import Sound, call

snd = Sound("audio.wav")  # Same constructor syntax as parselmouth
pitch = call(snd, "To Pitch (ac)", 0, 75, 600)
f0 = call(pitch, "Get value in frame", 10, "Hertz")
```

You can also switch backends at runtime:

```python
from praatfan import Sound, call, set_backend

set_backend("parselmouth")  # Use parselmouth under the hood
# or "praatfan", "praatfan_rust", "praatfan_gpl"

snd = Sound("audio.wav")
# ... rest of code unchanged
```

### Option 2: Clean API (more Pythonic)

Use `praatfan` with the direct methods for a **cleaner, more Pythonic API**:

```python
# Before (parselmouth)
import parselmouth

snd = parselmouth.Sound("audio.wav")
pitch = snd.to_pitch_ac()
f0 = pitch.selected_array['frequency']

formant = snd.to_formant_burg()

# After (praatfan) - cleaner API
from praatfan import Sound

snd = Sound("audio.wav")
pitch = snd.to_pitch()
f0 = pitch.values()  # Returns numpy array directly

formant = snd.to_formant_burg()
f1_values = formant.formant_values(1)  # All F1 values as numpy array
```

### Which to choose?

| Use case | Recommendation |
|----------|----------------|
| Migrating existing parselmouth scripts with `call()` | Option 1 (drop-in replacement) |
| Writing new code | Option 2 (cleaner API) |
| Need to switch between backends at runtime | Either (both support `set_backend()`) |
| Want parselmouth accuracy with MIT license | Either |

### API differences summary

| Feature | parselmouth | praatfan |
|---------|-------------|----------|
| Load from file | `Sound("path")` | `Sound("path")` |
| Load from array | `Sound(samples, sr)` | `Sound(samples, sampling_frequency=sr)` |
| call() function | `parselmouth.praat.call` | `praatfan.call` |
| Frame indexing in call() | 1-based | 1-based |
| Backend switching | No | Yes (`set_backend()`) |

## Performance and threading

The Rust backend (`praatfan_rust`) spreads the per-frame work of pitch,
harmonicity, intensity, formant and spectrogram analysis, and the resampler
behind formant analysis, across all CPU cores. Results are bit-identical to a
single-threaded run at any thread count.

- **Thread count:** set the `RAYON_NUM_THREADS` environment variable before the
  first analysis in a process (e.g. `RAYON_NUM_THREADS=4`). If you already
  parallelise across files (`multiprocessing`, joblib, several R sessions),
  set it to `1`, or to cores ÷ workers, so the two levels don't oversubscribe
  the CPU.
- **Python threads:** the GIL is released while an analysis runs, so calls
  from several Python threads run concurrently.
- **`multiprocessing`:** safe with the `fork` start method, including after
  the parent process has already run analyses.
- **Same numbers on every machine (since v0.1.14):** the Rust backend's
  numerics use no host maths library and no CPU-specific code paths.
  Transcendental functions come from the pure-Rust `libm` crate and FFTs from
  a vendored FFTPACK port. A given input therefore gives the same bits
  whichever CPU features the machine has (checked across glibc's FMA, AVX2
  and AVX code paths). Builds for different CPU architectures have not been
  compared bit for bit.
- **WebAssembly** builds are single-threaded.

## Why Multiple Backends?

1. **License flexibility**: The built-in `praatfan` backend is MIT-licensed, while `parselmouth` is GPL. Choose based on your project's licensing needs.

2. **Performance**: The Rust backends (`praatfan-rust`, `praatfan-core`) offer better performance for batch processing.

3. **Validation**: Compare results across backends to validate your analysis pipeline.

4. **Portability**: The pure Python backend works everywhere without compilation.

## Accuracy

The clean-room implementation (`praatfan`) matches parselmouth closely:

| Analysis | Typical Difference |
|----------|-------------------|
| Pitch (F0) | < 0.01 Hz |
| Formants | < 1 Hz (high-energy frames) |
| Intensity | < 0.3 dB |
| HNR | < 0.5 dB |

## Citing praatfan

If you use praatfan in published work, please cite it and state the version
you used (`praatfan.__version__` in Python, `packageVersion("praatfan")` in
R):

```bibtex
@misc{cohenpriva2026praatfan,
  author       = {Cohen Priva, Uriel},
  title        = {{Praatfan}: A clean-room reimplementation of {Praat}'s acoustic analyses},
  year         = {2026},
  note         = {Software, version X.Y.Z},
  howpublished = {\url{https://github.com/ucpresearch/praatfan-core-clean}}
}
```

Please also cite **Praat**. praatfan's code shares nothing with Praat's,
but the analyses it implements, their parameters and their defaults are
Praat's, and praatfan was validated against Praat's output. Praat's
authors ask for this form (fill in the version and date, see
[praat.org](https://www.praat.org)):

```bibtex
@misc{boersma2026praat,
  author       = {Boersma, Paul and Weenink, David},
  title        = {{Praat}: Doing phonetics by computer},
  year         = {2026},
  note         = {Computer program, version X.Y.Z, retrieved DD Month YYYY},
  howpublished = {\url{https://www.praat.org}}
}
```

If you report pitch or harmonicity, the algorithm's own paper is
Boersma (1993), "Accurate short-term analysis of the fundamental frequency
and the harmonics-to-noise ratio of a sampled sound", *Proceedings of the
Institute of Phonetic Sciences* 17: 97-110.

In R, `citation("praatfan")` prints these entries, with praatfan's
installed version filled in. GitHub's "Cite this repository" button (from
`CITATION.cff`) gives a `@software` variant for biblatex users.

## License

MIT License - see LICENSE file.

Compiled forms of the Rust backend (the `praatfan-rust` wheels, the
`praatfan-open-pipe` binaries and the WebAssembly bundles) also contain
third-party code under permissive or file-level licenses (MIT, Apache-2.0,
BSD, Zlib, Unicode-3.0, MPL-2.0 for the unmodified `symphonia` audio
decoders). The full list with license texts is `rust/LICENSE-THIRD-PARTY`,
shipped inside each wheel and bundle; `praatfan-open-pipe --licenses` prints
it.

The `praatfan` and `praatfan-rust` backends are clean-room implementations that do not contain any GPL code.

Note: If you use the `parselmouth` backend, your application may be subject to GPL requirements.
