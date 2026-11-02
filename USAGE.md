# praatfan Usage Guide

This guide covers how to use praatfan from Python and JavaScript/TypeScript (via WASM).

## Overview

praatfan provides acoustic analysis functions equivalent to Praat:

| Analysis | Description |
|----------|-------------|
| **Pitch** | Fundamental frequency (F0) detection using AC or CC methods |
| **Formant** | Formant frequencies and bandwidths using Burg's LPC method |
| **Intensity** | RMS energy contour in dB |
| **Harmonicity** | Harmonics-to-noise ratio (HNR) |
| **Spectrum** | Single-frame FFT with spectral moments |
| **Spectrogram** | Time-frequency representation (STFT) |

---

## Installation

### Python

```bash
# From PyPI (recommended)
pip install praatfan              # Pure Python (works everywhere)
pip install "praatfan[wasm]"      # + the Rust engine as WebAssembly (no compiled wheel needed)
pip install praatfan-rust         # Rust backend (optional, fastest)
pip install "praatfan[sphere]"    # + NIST SPHERE (TIMIT/WSJ/Switchboard) support
```

#### NIST SPHERE files (`.sph`)

`Sound.from_file` reads SPHERE audio (incl. shorten-compressed) when the optional
[`desphere`](https://pypi.org/project/desphere/) package is installed; without it,
libsndfile still handles uncompressed SPHERE, and other `.sph` files raise the
usual unsupported-format error. The `praatfan[sphere]` extra pulls `desphere[fast]`
(with a small Rust accelerator for large shorten files); output is byte-identical
to the pure-Python decoder. To stay pure-Python, require plain `desphere` in your
own dependencies instead — praatfan uses whichever is installed. The `praatfan-rust`
backend has SPHERE support compiled in, so it needs nothing extra.

```bash
# Or build Rust backend from source
cd rust
pip install maturin
maturin develop --features python
```

### JavaScript/TypeScript (WASM)

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for web browsers
cd rust
wasm-pack build --target web --features wasm

# Build for Node.js
wasm-pack build --target nodejs --features wasm
```

The built package will be in `rust/pkg/`.

`Sound.from_wav()` accepts WAV **and** NIST SPHERE bytes (including
shorten-compressed): the SPHERE decoder is statically linked into the WASM
bundle, so `.sph` files are handled fully client-side with nothing sent to a
server. No extra package or configuration is required.

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
praatfan_rust = { path = "../praatfan-core-clean/rust" }
```

### R

```r
remotes::install_github("ucpresearch/praatfan-core-clean", subdir = "r-package")
library(praatfan)
# Works immediately: the engine ships as WebAssembly and runs via V8.
# pf_setup() installs the faster native praatfan-open-pipe binary
# (or set PRAATFAN_PIPE to one); once present it is used automatically.
```

Analysis functions return tidy data.frames:

```r
p <- pf_pitch("audio.wav")                    # time, f0 (NA = unvoiced), strength, voiced
f <- pf_formants("audio.wav")                 # time, n_formants, F1..F5, B1..B5
i <- pf_intensity("audio.wav")                # time, intensity (dB)
h <- pf_harmonicity("audio.wav")              # time, hnr (dB)
m <- pf_spectral_moments("audio.wav", times = c(0.10, 0.15))
b <- pf_band_energy("audio.wav", times = c(0.10, 0.15), f_min = 0, f_max = 1000)
plot(pf_spectrogram("audio.wav"))             # "wide" (default), "narrow", or seconds
```

See `r-package/README.md` for details, including the batch interface
(`pf_analyze`) that computes several measures from one file load.

### Any language (JSON pipe)

The `praatfan-open-pipe` binary reads one JSON request on stdin and writes
one JSON response to stdout — usable from any language that can spawn a
process:

```bash
cd rust && cargo build --release --features pipe --bin praatfan-open-pipe

echo '{"wav_path":"audio.wav","analyses":[{"type":"pitch_ac"}]}' \
  | rust/target/release/praatfan-open-pipe
```

`praatfan-open-pipe --help` documents the full request/response schema.

---

## Python Usage

All examples use the unified `praatfan.Sound`, which works the same on every
backend (see [Backends and auto-selection](#backends-and-auto-selection-python)).
Result objects return whole contours as numpy arrays; per-frame and per-time
queries go through the parselmouth-style `call()`.

### Creating Sound Objects

```python
import numpy as np
import praatfan
from praatfan import Sound, call

# From an audio file (WAV, FLAC, MP3, OGG, ...)
sound = Sound("path/to/audio.wav")

# From a numpy array
samples = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
sound = Sound(samples, sampling_frequency=44100)

# One channel of a multi-channel file: read it yourself, pass the channel
import soundfile as sf
data, sr = sf.read("stereo.wav")
sound = Sound(data[:, 0], sampling_frequency=sr)   # left channel
```

**Sound properties:**

```python
sound.n_samples           # Number of samples
sound.sampling_frequency  # Sample rate in Hz
sound.duration            # Duration in seconds
sound.values              # Samples as a numpy array (a property, not a method)
sound.backend             # Which backend this Sound uses
```

**Resampling and excerpts:**

```python
s16 = sound.resample(16000)        # up or down; the same rate returns a copy
part = sound.extract_part(0.1, 0.4)
```

### Pitch Analysis

Detect fundamental frequency (F0) with the autocorrelation (AC) or
cross-correlation (CC) method.

```python
pitch = sound.to_pitch_ac(
    time_step=0.0,        # 0 = auto (0.75 / pitch_floor)
    pitch_floor=75.0,     # Minimum pitch in Hz
    pitch_ceiling=600.0,  # Maximum pitch in Hz
)
pitch = sound.to_pitch_cc(time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0)
# Optional Boersma (1993) tuning: voicing_threshold, silence_threshold,
# octave_cost, octave_jump_cost, voiced_unvoiced_cost
```

**Pitch results:**

```python
pitch.n_frames            # Number of frames
pitch.time_step           # Time step between frames
times = pitch.xs()        # Frame times
f0 = pitch.values()       # F0 in Hz; NaN for unvoiced frames
strength = pitch.strengths()  # Strength of the selected candidate per frame

# Per-frame and per-time queries (frame numbers are 1-based, as in Praat)
f0_10 = call(pitch, "Get value in frame", 10, "Hertz")
f0_t = call(pitch, "Get value at time", 0.5, "Hertz", "Linear")
```

### Formant Analysis

Formant frequencies and bandwidths by Burg's LPC method.

```python
formant = sound.to_formant_burg(
    time_step=0.0,              # 0 = auto (window_length / 4)
    max_number_of_formants=5,
    maximum_formant=5500.0,     # 5000 Hz for a male speaker, 5500 Hz for a female speaker
    window_length=0.025,        # seconds
    pre_emphasis_from=50.0,     # Hz
)
```

**Formant results:**

```python
times = formant.xs()
f1 = formant.formant_values(1)    # F1 per frame (NaN where absent)
f2 = formant.formant_values(2)
b1 = formant.bandwidth_values(1)  # Bandwidth of F1

f1_t = call(formant, "Get value at time", 1, 0.5, "Hertz", "Linear")
b1_t = call(formant, "Get bandwidth at time", 1, 0.5, "Hertz", "Linear")
n_10 = call(formant, "Get number of formants", 10)
```

### Intensity Analysis

```python
intensity = sound.to_intensity(
    minimum_pitch=100.0,  # Sets the window length (3.2 / minimum_pitch effective)
    time_step=0.0,        # 0 = auto (0.8 / minimum_pitch)
)
times = intensity.xs()
db = intensity.values()   # dB; -inf for digital silence
db_t = call(intensity, "Get value at time", 0.5, "Cubic")
```

### Harmonicity Analysis

Harmonics-to-noise ratio (HNR).

```python
hnr = sound.to_harmonicity_ac(
    time_step=0.01,
    minimum_pitch=75.0,
    silence_threshold=0.1,
    periods_per_window=4.5,   # AC needs at least 3.0
)
hnr = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=75.0,
                              silence_threshold=0.1, periods_per_window=1.0)

times = hnr.xs()
values = hnr.values()   # dB; -200 = no periodicity; NaN = correlation not resolvable from 1
```

### Spectrum Analysis

A single FFT over the whole sound, with spectral moments.

```python
spectrum = sound.to_spectrum(fast=True)   # fast=True pads to a power of two
spectrum.n_bins                 # Number of bins
spectrum.df                     # Bin spacing in Hz
freqs = spectrum.xs()

cog = spectrum.get_center_of_gravity(power=2.0)
sd = spectrum.get_standard_deviation(power=2.0)
skew = spectrum.get_skewness(power=2.0)
kurt = spectrum.get_kurtosis(power=2.0)
energy = spectrum.get_band_energy(f_min=0.0, f_max=1000.0)
```

For moments or band energy at many time points, use
`sound.get_spectral_moments_at_times(times)` and
`sound.get_band_energy_at_times(times, f_min, f_max)`.

### Spectrogram Analysis

A standard short-time Fourier transform (not Praat's exact grid).

```python
spectrogram = sound.to_spectrogram(
    window_length=0.005,       # seconds
    maximum_frequency=5000.0,  # Hz; clamped to the Nyquist frequency
    time_step=0.002,           # seconds
    frequency_step=20.0,       # Hz
)
spectrogram.n_times, spectrogram.n_freqs
times = spectrogram.xs()
freqs = spectrogram.ys()
power = spectrogram.values()   # 2-D array, shape (n_freqs, n_times)
p = call(spectrogram, "Get power at", 0.5, 1000.0)
```

---

## JavaScript/WASM Usage

### Setup

```javascript
// ES modules (web)
import init, { Sound } from './pkg/praatfan_rust.js';

async function main() {
    await init();  // Initialize WASM module

    // Now you can use Sound and other types
}

// Node.js
const { Sound } = require('./pkg/praatfan_rust.js');
```

### Creating Sound Objects

```javascript
// From raw samples (Float64Array)
const samples = new Float64Array([...]);
const sound = new Sound(samples, 44100);

// From WAV file bytes (Uint8Array)
const response = await fetch('audio.wav');
const wavBytes = new Uint8Array(await response.arrayBuffer());
const sound = Sound.from_wav(wavBytes);

// From specific channel of multi-channel WAV
const sound = Sound.from_wav_channel(wavBytes, 0);  // left channel
```

**Sound properties:**

```javascript
sound.n_samples()     // Number of samples
sound.sample_rate()   // Sample rate in Hz
sound.duration()      // Duration in seconds
sound.samples()       // Float64Array of samples
```

### Pitch Analysis

```javascript
// Autocorrelation method
const pitch = sound.to_pitch_ac(
    0.0,    // time_step (0 = auto)
    75.0,   // pitch_floor
    600.0   // pitch_ceiling
);

// Cross-correlation method
const pitch = sound.to_pitch_cc(0.0, 75.0, 600.0);
```

**Pitch methods:**

```javascript
pitch.n_frames()              // Number of frames
pitch.time_step()             // Time step between frames
pitch.pitch_floor()           // Minimum pitch
pitch.pitch_ceiling()         // Maximum pitch

const times = pitch.times();       // Float64Array of time points
const values = pitch.values();     // Float64Array of F0 (NaN for unvoiced)
const strengths = pitch.strengths(); // Float64Array of correlation strengths

// Access individual frames
const time = pitch.get_time_from_frame(10);
const f0 = pitch.get_value_in_frame(10);  // NaN if unvoiced
```

### Formant Analysis

```javascript
const formant = sound.to_formant_burg(
    0.0,     // time_step
    5,       // max_num_formants
    5500.0,  // max_formant_hz
    0.025,   // window_length
    50.0     // pre_emphasis_from
);
```

**Formant methods:**

```javascript
formant.n_frames()            // Number of frames
formant.time_step()           // Time step
formant.max_num_formants()    // Maximum formants per frame

const times = formant.times();           // Float64Array
const f1 = formant.formant_values(1);    // Float64Array of F1
const f2 = formant.formant_values(2);    // Float64Array of F2
const b1 = formant.bandwidth_values(1);  // Float64Array of B1

// Access individual frames
const time = formant.get_time_from_frame(10);
const f1 = formant.get_value_at_frame(10, 1);     // F1 at frame 10
const b1 = formant.get_bandwidth_at_frame(10, 1); // B1 at frame 10
```

### Intensity Analysis

```javascript
const intensity = sound.to_intensity(
    100.0,  // min_pitch
    0.0     // time_step
);
```

**Intensity methods:**

```javascript
intensity.n_frames()          // Number of frames
intensity.time_step()         // Time step

const times = intensity.times();   // Float64Array
const values = intensity.values(); // Float64Array (dB)

intensity.get_minimum()       // Min intensity
intensity.get_maximum()       // Max intensity
intensity.get_mean()          // Mean intensity

const time = intensity.get_time_from_frame(10);
const db = intensity.get_value_in_frame(10);
```

### Harmonicity Analysis

```javascript
// Autocorrelation method
const harmonicity = sound.to_harmonicity_ac(
    0.01,   // time_step
    75.0,   // min_pitch
    0.1,    // silence_threshold
    4.5     // periods_per_window
);

// Cross-correlation method
const harmonicity = sound.to_harmonicity_cc(0.01, 75.0, 0.1, 1.0);
```

**Harmonicity methods:**

```javascript
harmonicity.n_frames()        // Number of frames
harmonicity.time_step()       // Time step

const times = harmonicity.times();   // Float64Array
const values = harmonicity.values(); // Float64Array (dB; -200 = no periodicity, NaN = undefined)

const time = harmonicity.get_time_from_frame(10);
const hnr = harmonicity.get_value_in_frame(10);
```

### Spectrum Analysis

```javascript
const spectrum = sound.to_spectrum(true);  // fast=true
```

**Spectrum methods:**

```javascript
spectrum.n_bins()             // Number of frequency bins
spectrum.df()                 // Frequency resolution (Hz)
spectrum.max_frequency()      // Maximum frequency

const real = spectrum.real();  // Float64Array
const imag = spectrum.imag();  // Float64Array

const freq = spectrum.get_freq_from_bin(100);

// Spectral moments
const cog = spectrum.get_center_of_gravity(2.0);
const std = spectrum.get_standard_deviation(2.0);
const skew = spectrum.get_skewness(2.0);
const kurt = spectrum.get_kurtosis(2.0);

// Band energy
const energy = spectrum.get_band_energy(0.0, 1000.0);
```

### Spectrogram Analysis

```javascript
const spectrogram = sound.to_spectrogram(
    0.005,   // window_length
    5000.0,  // max_frequency
    0.002,   // time_step
    20.0     // frequency_step
);
```

**Spectrogram methods:**

```javascript
spectrogram.n_times()         // Number of time frames
spectrogram.n_freqs()         // Number of frequency bins
spectrogram.time_step()       // Time step
spectrogram.freq_step()       // Frequency step
spectrogram.time_min()        // Start time
spectrogram.time_max()        // End time
spectrogram.freq_min()        // Minimum frequency
spectrogram.freq_max()        // Maximum frequency

const times = spectrogram.times();        // Float64Array
const freqs = spectrogram.frequencies();  // Float64Array
const values = spectrogram.values();      // Float64Array (flat, freq × time)

const time = spectrogram.get_time_from_frame(10);
const freq = spectrogram.get_freq_from_bin(50);
const power = spectrogram.get_value_at(10, 50);  // time_frame, freq_bin
```

---

## API Comparison: Python vs JavaScript

| Feature | Python | JavaScript |
|---------|--------|------------|
| Create from samples | `Sound(samples, sampling_frequency=sr)` | `new Sound(samples, sr)` |
| Create from WAV file | `Sound("path.wav")` | `Sound.from_wav(bytes)` |
| Get samples | `sound.values` (property) | `sound.samples()` |
| Sample rate | `sound.sampling_frequency` | `sound.sample_rate()` |
| Pitch F0 values | `pitch.values()` | `pitch.values()` |
| Formant values | `formant.formant_values(n)` | `formant.formant_values(n)` |
| Time points | `.xs()` | `.times()` |

---

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

### Backends and auto-selection (Python)

With no `set_backend()`, `PRAATFAN_BACKEND` or config file, the first
available of these is used:

| Backend | Install | Speed |
|---------|---------|-------|
| `praatfan_rust` | `pip install praatfan-rust` | fastest (native, all cores) |
| `praatfan_gpl` | `pip install praatfan-gpl` | native |
| `praatfan_wasm` | `pip install "praatfan[wasm]"` | the `praatfan_rust` engine compiled to WebAssembly and run in an embedded V8 ([mini-racer](https://pypi.org/project/mini-racer/)); one thread per call, several times faster than pure Python |
| `praatfan` | always installed | pure-Python reference implementation |

`parselmouth` is never auto-selected; request it by name
(`set_backend("parselmouth")` or `PRAATFAN_BACKEND=parselmouth`).

`praatfan_wasm` runs the analyses the JSON pipe carries — pitch (AC/CC),
formant (Burg), intensity, harmonicity (AC/CC), spectral moments and band
energy at times — in WebAssembly, with the same numbers as `praatfan_rust`
(bit for bit since 0.1.14). Everything else (`to_spectrum`, `to_spectrogram`,
the `*_referenced` analyses, `extract_part`, `resample`) is served by the
pure-Python implementation, and `Pitch.raw_ac_strengths()` raises
`NotImplementedError`. Unvoiced F0 is `NaN`, as on every backend. Each
process gets its own
engine; `multiprocessing` works with every start method (a process forked
after its parent has used the engine routes its calls to a helper process,
since V8 itself does not survive `fork`).

## Common Workflows

### Extract F1/F2 vowel formants

**Python:**
```python
import praatfan
import numpy as np

sound = praatfan.Sound("vowel.wav")
formant = sound.to_formant_burg(max_number_of_formants=5, maximum_formant=5500)

# Get F1 and F2 trajectories
f1 = formant.formant_values(1)
f2 = formant.formant_values(2)
times = formant.xs()

# Filter out undefined values
valid = ~(np.isnan(f1) | np.isnan(f2))
print(f"Mean F1: {np.mean(f1[valid]):.0f} Hz")
print(f"Mean F2: {np.mean(f2[valid]):.0f} Hz")
```

**JavaScript:**
```javascript
const sound = Sound.from_wav(wavBytes);
const formant = sound.to_formant_burg(0, 5, 5500, 0.025, 50);

const f1 = formant.formant_values(1);
const f2 = formant.formant_values(2);

// Filter and compute mean
let sumF1 = 0, sumF2 = 0, count = 0;
for (let i = 0; i < f1.length; i++) {
    if (!isNaN(f1[i]) && !isNaN(f2[i])) {
        sumF1 += f1[i];
        sumF2 += f2[i];
        count++;
    }
}
console.log(`Mean F1: ${(sumF1/count).toFixed(0)} Hz`);
console.log(`Mean F2: ${(sumF2/count).toFixed(0)} Hz`);
```

### Pitch tracking with voicing detection

**Python:**
```python
import numpy as np
import praatfan

sound = praatfan.Sound("speech.wav")
pitch = sound.to_pitch_ac(pitch_floor=75, pitch_ceiling=300)

f0 = pitch.values()
times = pitch.xs()

# Find voiced regions
voiced = ~np.isnan(f0)
print(f"Voiced frames: {voiced.sum()} / {len(f0)}")
print(f"Mean F0 (voiced): {np.nanmean(f0):.1f} Hz")
```

**JavaScript:**
```javascript
const sound = Sound.from_wav(wavBytes);
const pitch = sound.to_pitch_ac(0, 75, 300);

const f0 = pitch.values();
const times = pitch.times();

let voicedCount = 0, sum = 0;
for (let i = 0; i < f0.length; i++) {
    if (!isNaN(f0[i])) {
        voicedCount++;
        sum += f0[i];
    }
}
console.log(`Voiced frames: ${voicedCount} / ${f0.length}`);
console.log(`Mean F0 (voiced): ${(sum/voicedCount).toFixed(1)} Hz`);
```

---

## Typical Parameter Values

| Analysis | Parameter | Typical Value | Notes |
|----------|-----------|---------------|-------|
| Pitch | `pitch_floor` | 75 Hz | Lower for male voices |
| Pitch | `pitch_ceiling` | 300-600 Hz | Higher for female/child voices |
| Formant | `maximum_formant` | 5000 Hz (male), 5500 Hz (female) | Adjust for speaker |
| Formant | `max_number_of_formants` | 5 | Standard for vowel analysis |
| Formant | `window_length` | 0.025 s | 25 ms is typical |
| Intensity | `minimum_pitch` | 100 Hz | Determines window size |
| Harmonicity | `periods_per_window` | 4.5 (AC), 1.0 (CC) | **AC requires >= 3.0** (FutureWarning below it, error in a future release); 1.0 is the CC value, not an AC value |

---

## Notes

- **Mono audio only**: praatfan only supports mono audio. Use `from_file_channel()` or `from_wav_channel()` for stereo files.
- **Undefined values**: Unvoiced frames return `NaN` for pitch. Harmonicity uses
  two distinct markers: `-200` dB where no periodicity was found (unvoiced,
  silence, DC), and `NaN` where the correlation is not resolvable from 1 and no
  HNR can be derived. Filter with `np.isfinite(v) & (v > -200)`.
- **Time step = 0**: When `time_step` is 0, an automatic value is computed based on the analysis type.
- **Parselmouth compatibility**: The Python API is designed to be compatible with parselmouth's API, making migration straightforward.
