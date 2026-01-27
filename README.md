# praatfan

A clean-room reimplementation of Praat's acoustic analysis algorithms, with a unified API that supports multiple backends.

> **Note:** This is a pre-release (v0.1.0). The API is stabilizing but may still change.

## Installation

### Option 1: Pure Python (from source)

```bash
pip install git+https://github.com/ucpresearch/praatfan-core-clean.git
```

This installs the pure Python implementation. Works everywhere, no compilation needed.

### Option 2: Pre-built Rust wheels (faster)

Pre-compiled wheels with Rust acceleration are available from [GitHub Releases](https://github.com/ucpresearch/praatfan-core-clean/releases/tag/v0.1.0).

```bash
# Linux x86_64, Python 3.12
pip install "https://github.com/ucpresearch/praatfan-core-clean/releases/download/v0.1.0/praatfan_rust-0.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

# macOS ARM64 (Apple Silicon), Python 3.9 (default macOS Python)
pip install "https://github.com/ucpresearch/praatfan-core-clean/releases/download/v0.1.0/praatfan_rust-0.1.0-cp39-cp39-macosx_11_0_arm64.whl"

# Windows x86_64, Python 3.12
pip install "https://github.com/ucpresearch/praatfan-core-clean/releases/download/v0.1.0/praatfan_rust-0.1.0-cp312-cp312-win_amd64.whl"
```

Available wheels:
- **Linux x86_64**: Python 3.9, 3.10, 3.11, 3.12
- **macOS ARM64**: Python 3.9, 3.10, 3.11, 3.12
- **Windows x86_64**: Python 3.9, 3.10, 3.11, 3.12

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

### Backend Selection

Backends are selected automatically in this order of preference:
1. `praatfan_gpl` (if installed)
2. `praatfan_rust` (if installed)
3. `praatfan` (always available)
4. `parselmouth` (if installed)

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

The Rust-accelerated backend. Install from [pre-built wheels](#option-2-pre-built-rust-wheels-faster), or build from source (requires Rust toolchain):

```bash
cd rust
pip install maturin
maturin develop --features python
```

#### praatfan_gpl

A separate Rust implementation. Install from [GitHub Releases](https://github.com/ucpresearch/praatfan-core-rs/releases):

```bash
# Linux x86_64, Python 3.12
pip install https://github.com/ucpresearch/praatfan-core-rs/releases/download/v0.1.1/praatfan_gpl-0.1.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# macOS Apple Silicon, Python 3.12
pip install https://github.com/ucpresearch/praatfan-core-rs/releases/download/v0.1.1/praatfan_gpl-0.1.1-cp312-cp312-macosx_11_0_arm64.whl

# Windows x86_64, Python 3.12
pip install https://github.com/ucpresearch/praatfan-core-rs/releases/download/v0.1.1/praatfan_gpl-0.1.1-cp312-cp312-win_amd64.whl
```

See the [praatfan-core-rs releases](https://github.com/ucpresearch/praatfan-core-rs/releases) for other Python versions and platforms.

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
harmonicity.values()  # HNR in dB
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

## License

MIT License - see LICENSE file.

The `praatfan` and `praatfan-rust` backends are clean-room implementations that do not contain any GPL code.

Note: If you use the `parselmouth` backend, your application may be subject to GPL requirements.
