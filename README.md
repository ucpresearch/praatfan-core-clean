# praatfan

A clean-room reimplementation of Praat's acoustic analysis algorithms, with a unified API that supports multiple backends.

## Installation

```bash
pip install git+https://github.com/ucpresearch/praatfan-core-clean.git
```

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
| `praatfan-rust` | Built-in (requires build) | MIT | Rust implementation with PyO3 bindings |
| `parselmouth` | `praat-parselmouth` | GPL | Python bindings to Praat |
| `praatfan-core` | `praatfan-core` | MIT | Separate Rust implementation |

### Backend Selection

Backends are selected automatically in this order of preference:
1. `praatfan-rust` (if compiled)
2. `praatfan` (always available)
3. `parselmouth` (if installed)
4. `praatfan-core` (if installed)

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

#### praatfan-rust

If pre-compiled wheels are available (check releases), pip installs them automatically - no Rust toolchain needed:

```bash
pip install praatfan[rust]  # If wheels are published
```

To build from source (requires Rust toolchain):

```bash
cd rust
pip install maturin
maturin develop --features python
```

#### praatfan-core

```bash
pip install git+https://github.com/ucpresearch/praatfan-core.git
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

## Migrating from parselmouth

If you're currently using parselmouth directly, here's how to migrate:

### Before (parselmouth)

```python
import parselmouth

snd = parselmouth.Sound("audio.wav")
pitch = snd.to_pitch_ac()
f0 = pitch.selected_array['frequency']

formant = snd.to_formant_burg()
# Getting F1 requires Praat calls
from parselmouth.praat import call
f1_at_time = call(formant, "Get value at time", 1, 0.5, "Hertz", "Linear")
```

### After (praatfan)

```python
from praatfan import Sound

snd = Sound("audio.wav")
pitch = snd.to_pitch_ac()
f0 = pitch.values()  # Unified API

formant = snd.to_formant_burg()
f1_values = formant.formant_values(1)  # All F1 values as array
```

### Keep using parselmouth as backend

If you want the unified API but still use parselmouth under the hood:

```python
from praatfan import Sound, set_backend

set_backend('parselmouth')
# Now all analysis uses parselmouth internally
# but you get the unified result API
```

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
