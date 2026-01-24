# API Specification

Target API for praatfan-core-clean. This defines the public interface that implementations should provide.

## Core Types

### Sound

```rust
pub struct Sound {
    // Internal: samples and sample_rate
}

impl Sound {
    /// Load from audio file (WAV, FLAC). Mono only - error on multi-channel.
    pub fn from_file(path: &Path) -> Result<Sound>;

    /// Create from raw samples
    pub fn from_samples(samples: &[f64], sample_rate: f64) -> Sound;

    /// Total duration in seconds
    pub fn duration(&self) -> f64;

    /// Sample rate in Hz
    pub fn sample_rate(&self) -> f64;

    /// Access to raw samples
    pub fn samples(&self) -> &[f64];

    /// Analysis methods
    pub fn to_pitch(&self, time_step: f64, pitch_floor: f64, pitch_ceiling: f64) -> Pitch;
    pub fn to_intensity(&self, min_pitch: f64, time_step: f64) -> Intensity;
    pub fn to_formant_burg(&self, time_step: f64, max_formants: f64, max_formant_hz: f64,
                           window_length: f64, pre_emphasis_from: f64) -> Formant;
    pub fn to_harmonicity_ac(&self, time_step: f64, min_pitch: f64,
                              silence_threshold: f64, periods_per_window: f64) -> Harmonicity;
    pub fn to_harmonicity_cc(&self, time_step: f64, min_pitch: f64,
                              silence_threshold: f64, periods_per_window: f64) -> Harmonicity;
    pub fn to_spectrum(&self) -> Spectrum;
    pub fn to_spectrogram(&self, time_step: f64, max_frequency: f64,
                          window_length: f64, frequency_step: f64) -> Spectrogram;
}
```

### Pitch

```rust
pub struct Pitch {
    // Frames with F0 candidates and selected path
}

impl Pitch {
    /// Get F0 value at specific time
    /// Returns None for unvoiced frames
    pub fn get_value_at_time(&self, time: f64, unit: PitchUnit,
                              interpolation: Interpolation) -> Option<f64>;

    /// Time of first frame
    pub fn start_time(&self) -> f64;

    /// Time of last frame
    pub fn end_time(&self) -> f64;

    /// Time step between frames
    pub fn time_step(&self) -> f64;

    /// Number of frames
    pub fn num_frames(&self) -> usize;

    /// Get time of frame by index (0-based)
    pub fn get_time_from_frame(&self, frame: usize) -> f64;
}

pub enum PitchUnit {
    Hertz,
    Mel,
    Semitones,
    SemitonesRe100Hz,
    SemitonesRe200Hz,
    SemitonesRe440Hz,
    Erb,
}

pub enum PitchMethod {
    Autocorrelation,    // AC - 3 periods window
    CrossCorrelation,   // CC - 1 period window
}
```

### Intensity

```rust
pub struct Intensity {
    // Frames with dB values
}

impl Intensity {
    /// Get intensity value at specific time in dB
    pub fn get_value_at_time(&self, time: f64, interpolation: Interpolation) -> Option<f64>;

    /// Time bounds
    pub fn start_time(&self) -> f64;
    pub fn end_time(&self) -> f64;
    pub fn time_step(&self) -> f64;
    pub fn num_frames(&self) -> usize;
    pub fn get_time_from_frame(&self, frame: usize) -> f64;
}
```

### Formant

```rust
pub struct Formant {
    // Frames with formant frequencies and bandwidths
}

impl Formant {
    /// Get formant frequency at time
    /// formant_number: 1 = F1, 2 = F2, etc.
    pub fn get_value_at_time(&self, formant_number: u32, time: f64,
                              unit: FrequencyUnit, interpolation: Interpolation) -> Option<f64>;

    /// Get formant bandwidth at time
    pub fn get_bandwidth_at_time(&self, formant_number: u32, time: f64,
                                  unit: FrequencyUnit, interpolation: Interpolation) -> Option<f64>;

    /// Maximum number of formants per frame
    pub fn max_formants(&self) -> u32;

    /// Time bounds
    pub fn start_time(&self) -> f64;
    pub fn end_time(&self) -> f64;
    pub fn time_step(&self) -> f64;
    pub fn num_frames(&self) -> usize;
    pub fn get_time_from_frame(&self, frame: usize) -> f64;
}

pub enum FrequencyUnit {
    Hertz,
    Bark,
    Mel,
    Erb,
}
```

### Harmonicity

```rust
pub struct Harmonicity {
    // Frames with HNR values in dB
}

impl Harmonicity {
    /// Get HNR value at specific time in dB
    /// Returns very negative value (e.g., -200) for unvoiced
    pub fn get_value_at_time(&self, time: f64, interpolation: Interpolation) -> Option<f64>;

    /// Time bounds
    pub fn start_time(&self) -> f64;
    pub fn end_time(&self) -> f64;
    pub fn time_step(&self) -> f64;
    pub fn num_frames(&self) -> usize;
    pub fn get_time_from_frame(&self, frame: usize) -> f64;
}
```

### Spectrum

```rust
pub struct Spectrum {
    // FFT result: real and imaginary parts
}

impl Spectrum {
    /// Spectral moments
    pub fn get_center_of_gravity(&self, power: f64) -> f64;
    pub fn get_standard_deviation(&self, power: f64) -> f64;
    pub fn get_skewness(&self, power: f64) -> f64;
    pub fn get_kurtosis(&self, power: f64) -> f64;

    /// Band energy
    pub fn get_band_energy(&self, freq_min: f64, freq_max: f64) -> f64;

    /// Frequency bounds
    pub fn freq_min(&self) -> f64;  // 0.0
    pub fn freq_max(&self) -> f64;  // Nyquist
    pub fn freq_step(&self) -> f64; // 1/duration
    pub fn num_bins(&self) -> usize;
}
```

### Spectrogram

```rust
pub struct Spectrogram {
    // 2D array: (n_freqs, n_times) power values
}

impl Spectrogram {
    /// Access to power values (Pa²/Hz)
    pub fn values(&self) -> &Array2<f64>;  // or Vec<Vec<f64>>

    /// Frequency bounds
    pub fn freq_min(&self) -> f64;
    pub fn freq_max(&self) -> f64;

    /// Time bounds
    pub fn time_min(&self) -> f64;
    pub fn time_max(&self) -> f64;

    /// Frame/bin accessors
    pub fn get_time_from_frame(&self, frame: usize) -> f64;
    pub fn get_freq_from_bin(&self, bin: usize) -> f64;
    pub fn num_frames(&self) -> usize;
    pub fn num_freq_bins(&self) -> usize;
}
```

### Interpolation

```rust
pub enum Interpolation {
    Nearest,
    Linear,
    Cubic,
    Sinc70,   // 70-point sinc
    Sinc700,  // 700-point sinc
}
```

### Window Shapes

```rust
pub enum WindowShape {
    Rectangular,
    Triangular,
    Parabolic,
    Hanning,
    Hamming,
    Gaussian,
    Kaiser,
}
```

## Default Parameters

### Pitch

| Parameter | Default | Notes |
|-----------|---------|-------|
| time_step | 0.0 | Auto: 0.75/pitch_floor |
| pitch_floor | 75.0 Hz | Determines window length |
| pitch_ceiling | 600.0 Hz | Maximum F0 |

### Intensity

| Parameter | Default | Notes |
|-----------|---------|-------|
| min_pitch | 100.0 Hz | Determines window length |
| time_step | 0.0 | Auto: 0.8/min_pitch |

### Formant

| Parameter | Default | Notes |
|-----------|---------|-------|
| time_step | 0.0 | Auto: 25% of window |
| max_formants | 5.0 | Number of formants |
| max_formant_hz | 5500.0 | Formant ceiling (5000 for male) |
| window_length | 0.025 | Effective; actual = 2× |
| pre_emphasis_from | 50.0 Hz | Pre-emphasis frequency |

### Harmonicity

| Parameter | Default | Notes |
|-----------|---------|-------|
| time_step | 0.01 s | Frame interval |
| min_pitch | 75.0 Hz | Determines window |
| silence_threshold | 0.1 | Relative to global max |
| periods_per_window | 4.5 | 6.0 for high precision |

### Spectrogram

| Parameter | Default | Notes |
|-----------|---------|-------|
| time_step | 0.002 s | 2 ms |
| max_frequency | 5000.0 Hz | Upper frequency limit |
| window_length | 0.005 s | 5 ms for broadband |
| frequency_step | 20.0 Hz | Frequency resolution |

## Frequency Scale Conversions

```rust
// Hz to Mel (Praat formula)
fn hz_to_mel(hz: f64) -> f64 {
    550.0 * (1.0 + hz / 550.0).ln()
}

// Hz to Bark (Traunmüller formula)
fn hz_to_bark(hz: f64) -> f64 {
    let x = hz / 600.0;
    7.0 * (x + (x * x + 1.0).sqrt()).ln()
}

// Hz to ERB (Glasberg & Moore 1990)
fn hz_to_erb(hz: f64) -> f64 {
    21.4 * (0.00437 * hz + 1.0).log10()
}

// Hz to semitones
fn hz_to_semitones(hz: f64, reference: f64) -> f64 {
    12.0 * (hz / reference).log2()
}
```

## Error Handling

```rust
pub enum PraatError {
    Io(std::io::Error),
    InvalidParameter(String),
    Analysis(String),
    UndefinedValue { time: f64, reason: String },
    MultiChannelNotSupported,  // Require mono
}

pub type Result<T> = std::result::Result<T, PraatError>;
```
