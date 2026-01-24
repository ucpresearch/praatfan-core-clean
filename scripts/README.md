# Validation Scripts

This document explains how to validate your clean room implementation against parselmouth (Python bindings for Praat).

## Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy soundfile praat-parselmouth matplotlib

# Install praatfan in development mode
pip install -e src/
```

## Validation Order (Follow Dependencies)

Implement and validate in this order:

```
1. Spectrum         (foundation)
2. Spectral Moments (uses Spectrum)
3. Intensity        (independent)
4. Pitch (AC)       (foundation for Harmonicity)
5. Harmonicity (AC) (depends on Pitch AC - just a formula!)
6. Pitch (CC)       (foundation for Harmonicity CC)
7. Harmonicity (CC) (depends on Pitch CC - just a formula!)
8. Formant          (complex, independent)
9. Spectrogram      (uses windowed Spectrum)
```

**Critical:** Harmonicity is NOT a separate algorithm. It uses Pitch internally. If Harmonicity is wrong, debug Pitch, not Harmonicity.

## Key Validation Rules

### Step 1: Match the Count First

Before comparing values, ensure you have the same number of frames:

```
Your implementation: 158 frames
Parselmouth: 159 frames
→ STOP. Fix frame timing before comparing values.
```

### Step 2: Verify Intermediate Values

For multi-step algorithms, check intermediate values first:

| Algorithm | Check in this order |
|-----------|---------------------|
| **Pitch** | 1. Voicing decisions, 2. Strengths (`Get strength in frame`), 3. F0 |
| **Formant** | 1. Formant count per frame, 2. Individual F1/F2/F3, 3. Bandwidths |
| **Harmonicity** | Validate Pitch first! HNR is just a formula on pitch strength |

### Step 3: Compare Frame by Frame

Don't use averages - compare each frame individually:

```python
for i, (my_val, praat_val) in enumerate(zip(my_values, praat_values)):
    error = abs(my_val - praat_val)
    if error > tolerance:
        print(f"Frame {i}: my={my_val:.4f}, praat={praat_val:.4f}, error={error:.4f}")
```

### Step 4: Investigate Outliers

If 95% match but 5% have large errors, those outliers reveal edge cases.

## Tolerance Targets

| Algorithm | Target |
|-----------|--------|
| Formant F1/F2/F3 | within 1 Hz |
| Pitch F0 | within 0.01 Hz |
| Intensity | within 0.001 dB |
| Spectrum | relative error < 1e-10 |
| HNR | within 0.01 dB |

---

## Getting Ground Truth with parselmouth

### Formant Analysis

```python
import parselmouth
from parselmouth.praat import call
import numpy as np

# Load audio
snd = parselmouth.Sound("tests/fixtures/one_two_three_four_five.wav")

# Run formant analysis with default parameters
formant = call(snd, "To Formant (burg)",
    0.01,    # time_step
    5,       # max_formants
    5500.0,  # max_formant_hz
    0.025,   # window_length
    50.0     # pre_emphasis_from
)

# Get number of frames
n_frames = call(formant, "Get number of frames")

# Extract values for each frame
for i in range(1, n_frames + 1):
    t = call(formant, "Get time from frame number", i)

    # Get F1, F2, F3 at this frame
    f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
    f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
    f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")

    # NaN indicates undefined (no formant found)
    if not np.isnan(f1):
        print(f"t={t:.3f}: F1={f1:.1f} Hz, F2={f2:.1f} Hz, F3={f3:.1f} Hz")
```

### Pitch Analysis

```python
snd = parselmouth.Sound("tests/fixtures/one_two_three_four_five.wav")

pitch = call(snd, "To Pitch",
    0.01,    # time_step (0.0 = auto: 0.75/pitch_floor)
    75.0,    # pitch_floor
    600.0    # pitch_ceiling
)

n_frames = call(pitch, "Get number of frames")

for i in range(1, n_frames + 1):
    t = call(pitch, "Get time from frame number", i)
    f0 = call(pitch, "Get value in frame", i, "Hertz")  # Note: "in frame", not "at time"
    strength = call(pitch, "Get strength in frame", i)   # For HNR calculation

    if f0 is None or f0 == 0:
        print(f"t={t:.3f}: unvoiced")
    else:
        print(f"t={t:.3f}: F0={f0:.2f} Hz, strength={strength:.4f}")
```

### Intensity Analysis

```python
snd = parselmouth.Sound("tests/fixtures/one_two_three_four_five.wav")

intensity = call(snd, "To Intensity",
    100.0,   # min_pitch
    0.01     # time_step (0.0 = auto: 0.8/min_pitch)
)

n_frames = call(intensity, "Get number of frames")

for i in range(1, n_frames + 1):
    t = call(intensity, "Get time from frame number", i)
    db = call(intensity, "Get value at time", t, "Cubic")
    print(f"t={t:.3f}: {db:.2f} dB")
```

### Harmonicity (HNR) Analysis

```python
snd = parselmouth.Sound("tests/fixtures/one_two_three_four_five.wav")

# AC method (autocorrelation)
harmonicity = call(snd, "To Harmonicity (ac)",
    0.01,    # time_step
    75.0,    # min_pitch
    0.1,     # silence_threshold
    4.5      # periods_per_window
)

# CC method (cross-correlation)
# harmonicity = call(snd, "To Harmonicity (cc)",
#     0.01,    # time_step
#     75.0,    # min_pitch
#     0.1,     # silence_threshold
#     1.0      # periods_per_window
# )

n_frames = call(harmonicity, "Get number of frames")

for i in range(1, n_frames + 1):
    t = call(harmonicity, "Get time from frame number", i)
    hnr = call(harmonicity, "Get value at time", t, "Cubic")

    if hnr < -100:  # Very negative = unvoiced
        print(f"t={t:.3f}: unvoiced")
    else:
        print(f"t={t:.3f}: HNR={hnr:.1f} dB")
```

### Spectrum Analysis

```python
snd = parselmouth.Sound("tests/fixtures/one_two_three_four_five.wav")

spectrum = call(snd, "To Spectrum", "yes")  # "yes" = fast (FFT)

# Spectral moments
cog = call(spectrum, "Get centre of gravity", 2.0)  # power = 2.0
std = call(spectrum, "Get standard deviation", 2.0)
skew = call(spectrum, "Get skewness", 2.0)
kurt = call(spectrum, "Get kurtosis", 2.0)

print(f"Centre of gravity: {cog:.1f} Hz")
print(f"Standard deviation: {std:.1f} Hz")
print(f"Skewness: {skew:.4f}")
print(f"Kurtosis: {kurt:.4f}")

# Band energy
energy = call(spectrum, "Get band energy", 0.0, 1000.0)
print(f"Energy 0-1000 Hz: {energy:.6e} Pa²·s")
```

### Spectrogram Analysis

```python
snd = parselmouth.Sound("tests/fixtures/one_two_three_four_five.wav")

spectrogram = call(snd, "To Spectrogram",
    0.005,      # window_length
    5000.0,     # max_frequency
    0.002,      # time_step
    20.0,       # frequency_step
    "Gaussian"  # window_shape
)

# Get dimensions
n_times = call(spectrogram, "Get number of frames")
n_freqs = call(spectrogram, "Get number of frequencies")

print(f"Spectrogram: {n_times} time frames × {n_freqs} frequency bins")

# Get power at specific time and frequency
power = call(spectrogram, "Get power at", 0.5, 1000.0)  # time, frequency
print(f"Power at t=0.5s, f=1000Hz: {power:.6e} Pa²/Hz")
```

---

## Test Audio Files

The `tests/fixtures/` directory should contain:

- `one_two_three_four_five.wav` - Speech sample saying "one two three four five"

All test files must be **mono** (single channel). Multi-channel files are not supported.
