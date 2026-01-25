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
# Install from source using maturin
cd rust
pip install maturin
maturin develop --features python

# Or build a wheel
maturin build --features python --release
pip install target/wheels/praatfan-*.whl
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

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
praatfan = { path = "../praatfan-core-clean/rust" }
```

---

## Python Usage

### Creating Sound Objects

```python
import praatfan
import numpy as np

# From a WAV file
sound = praatfan.Sound("path/to/audio.wav")

# From a numpy array
samples = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
sound = praatfan.Sound(samples, sampling_frequency=44100)

# From a specific channel of a multi-channel file
sound = praatfan.Sound.from_file_channel("stereo.wav", channel=0)  # left channel
```

**Sound properties:**

```python
sound.n_samples          # Number of samples
sound.sampling_frequency # Sample rate in Hz
sound.duration           # Duration in seconds
sound.values()           # Samples as numpy array
sound.xs()               # Time values for each sample
```

### Pitch Analysis

Detect fundamental frequency (F0) using autocorrelation or cross-correlation methods.

```python
# Autocorrelation method (default, more accurate for clean speech)
pitch = sound.to_pitch_ac(
    time_step=0.0,        # 0 = auto (0.75 / pitch_floor)
    pitch_floor=75.0,     # Minimum pitch in Hz
    pitch_ceiling=600.0   # Maximum pitch in Hz
)

# Cross-correlation method (better for noisy speech)
pitch = sound.to_pitch_cc(time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0)

# Alias for to_pitch_ac
pitch = sound.to_pitch()
```

**Pitch properties and methods:**

```python
pitch.n_frames                    # Number of analysis frames
pitch.time_step                   # Time step between frames

# Get all values as numpy arrays
times = pitch.xs()                # Time points for all frames
frequencies = pitch.values()      # F0 values (NaN for unvoiced frames)
frequencies = pitch.to_array()    # Same as values()

# Parselmouth-compatible selected_array property
data = pitch.selected_array       # Dict with 'frequency' and 'strength' keys
f0 = data['frequency']            # numpy array of F0 values
strength = data['strength']       # numpy array of correlation strengths

# Access individual frames (0-indexed)
f0 = pitch.get_value_in_frame(10)         # F0 at frame 10
f0 = pitch.get_value_at_time(0.5)         # F0 at t=0.5s (interpolated)
strength = pitch.get_strength_at_time(0.5) # Strength at t=0.5s
```

### Formant Analysis

Extract formant frequencies and bandwidths using Burg's LPC method.

```python
formant = sound.to_formant_burg(
    time_step=0.0,              # 0 = auto
    max_number_of_formants=5,   # Maximum formants to track
    maximum_formant=5500.0,     # Max formant frequency (5500 for male, 5000 for female)
    window_length=0.025,        # Window length in seconds
    pre_emphasis_from=50.0      # Pre-emphasis frequency in Hz
)
```

**Formant properties and methods:**

```python
formant.n_frames                  # Number of analysis frames
formant.time_step                 # Time step between frames

# Get formant values as numpy arrays (formant_number: 1=F1, 2=F2, etc.)
times = formant.xs()              # Time points
f1 = formant.to_array(1)          # F1 values for all frames
f2 = formant.to_array(2)          # F2 values
b1 = formant.to_bandwidth_array(1) # B1 (bandwidth of F1)

# Access individual frames
f1 = formant.get_value_in_frame(1, 10)      # F1 at frame 10
f1 = formant.get_value_at_time(1, 0.5)      # F1 at t=0.5s
b1 = formant.get_bandwidth_in_frame(1, 10)  # B1 at frame 10
b1 = formant.get_bandwidth_at_time(1, 0.5)  # B1 at t=0.5s
n = formant.get_number_of_formants(10)      # Number of formants at frame 10
```

### Intensity Analysis

Compute RMS energy contour in dB.

```python
intensity = sound.to_intensity(
    minimum_pitch=100.0,  # Determines window size
    time_step=0.0         # 0 = auto
)
```

**Intensity properties and methods:**

```python
intensity.n_frames                # Number of frames
intensity.time_step               # Time step

times = intensity.xs()            # Time points
values = intensity.values()       # Intensity in dB
values = intensity.to_array()     # Same as values()

# Statistics
intensity.get_minimum()           # Min intensity in dB
intensity.get_maximum()           # Max intensity in dB
intensity.get_mean()              # Mean intensity in dB

# Access at specific time
db = intensity.get_value_at_time(0.5)
```

### Harmonicity Analysis

Compute harmonics-to-noise ratio (HNR).

```python
# Autocorrelation method
harmonicity = sound.to_harmonicity_ac(
    time_step=0.01,           # Time step in seconds
    minimum_pitch=75.0,       # Minimum pitch in Hz
    silence_threshold=0.1,    # Silence threshold (0-1)
    periods_per_window=4.5    # Periods per analysis window
)

# Cross-correlation method
harmonicity = sound.to_harmonicity_cc(
    time_step=0.01,
    minimum_pitch=75.0,
    silence_threshold=0.1,
    periods_per_window=1.0
)

# Alias for to_harmonicity_ac
harmonicity = sound.to_harmonicity()
```

**Harmonicity properties and methods:**

```python
harmonicity.n_frames              # Number of frames
harmonicity.time_step             # Time step

times = harmonicity.xs()          # Time points
values = harmonicity.values()     # HNR in dB (-200 for silent/unvoiced)
values = harmonicity.to_array()   # Same as values()

hnr = harmonicity.get_value_at_time(0.5)
```

### Spectrum Analysis

Compute single-frame FFT with spectral moments.

```python
spectrum = sound.to_spectrum(fast=True)  # fast=True uses power-of-2 FFT
```

**Spectrum properties and methods:**

```python
spectrum.n_bins                   # Number of frequency bins
spectrum.df                       # Frequency resolution (Hz)
spectrum.maximum_frequency        # Maximum frequency (Nyquist)

freqs = spectrum.xs()             # Frequency values for all bins
real_part = spectrum.real()       # Real parts
imag_part = spectrum.imag()       # Imaginary parts

# Spectral moments
cog = spectrum.get_center_of_gravity(power=2.0)  # Centroid in Hz
std = spectrum.get_standard_deviation(power=2.0) # Spread in Hz
skew = spectrum.get_skewness(power=2.0)          # Skewness
kurt = spectrum.get_kurtosis(power=2.0)          # Kurtosis

# Band energy
energy = spectrum.get_band_energy(band_floor=0.0, band_ceiling=1000.0)
```

### Spectrogram Analysis

Compute time-frequency representation (STFT).

```python
spectrogram = sound.to_spectrogram(
    window_length=0.005,      # Window length in seconds
    maximum_frequency=5000.0, # Maximum frequency in Hz
    time_step=0.002,          # Time step in seconds
    frequency_step=20.0       # Frequency step in Hz
)
```

**Spectrogram properties and methods:**

```python
spectrogram.n_times               # Number of time frames
spectrogram.n_freqs               # Number of frequency bins
spectrogram.time_step             # Time step
spectrogram.frequency_step        # Frequency step

times = spectrogram.xs()          # Time values
freqs = spectrogram.ys()          # Frequency values
power = spectrogram.values()      # Power values (flat array, freq × time)

# Access at specific location
power = spectrogram.get_power_at(time=0.5, frequency=1000.0)
```

---

## JavaScript/WASM Usage

### Setup

```javascript
// ES modules (web)
import init, { Sound } from './pkg/praatfan.js';

async function main() {
    await init();  // Initialize WASM module

    // Now you can use Sound and other types
}

// Node.js
const { Sound } = require('./pkg/praatfan.js');
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
const values = harmonicity.values(); // Float64Array (dB, -200 for silent)

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
| Get samples | `sound.values()` | `sound.samples()` |
| Sample rate | `sound.sampling_frequency` | `sound.sample_rate()` |
| Pitch F0 values | `pitch.values()` or `pitch.selected_array['frequency']` | `pitch.values()` |
| Formant values | `formant.to_array(n)` | `formant.formant_values(n)` |
| Time points | `.xs()` | `.times()` |

---

## Common Workflows

### Extract F1/F2 vowel formants

**Python:**
```python
import praatfan
import numpy as np

sound = praatfan.Sound("vowel.wav")
formant = sound.to_formant_burg(max_number_of_formants=5, maximum_formant=5500)

# Get F1 and F2 trajectories
f1 = formant.to_array(1)
f2 = formant.to_array(2)
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
sound = praatfan.Sound("speech.wav")
pitch = sound.to_pitch_ac(pitch_floor=75, pitch_ceiling=300)

f0 = pitch.selected_array['frequency']
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
| Formant | `maximum_formant` | 5500 Hz (male), 5000 Hz (female) | Adjust for speaker |
| Formant | `max_number_of_formants` | 5 | Standard for vowel analysis |
| Formant | `window_length` | 0.025 s | 25 ms is typical |
| Intensity | `minimum_pitch` | 100 Hz | Determines window size |
| Harmonicity | `periods_per_window` | 4.5 (AC), 1.0 (CC) | Trade-off: precision vs time resolution |

---

## Notes

- **Mono audio only**: praatfan only supports mono audio. Use `from_file_channel()` or `from_wav_channel()` for stereo files.
- **NaN values**: Unvoiced frames return `NaN` for pitch and `-200` for harmonicity.
- **Time step = 0**: When `time_step` is 0, an automatic value is computed based on the analysis type.
- **Parselmouth compatibility**: The Python API is designed to be compatible with parselmouth's API, making migration straightforward.
