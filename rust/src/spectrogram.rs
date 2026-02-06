//! Spectrogram - Time-frequency representation.
//!
//! This module computes spectrograms using the Short-Time Fourier Transform (STFT).
//! A spectrogram shows how the spectral content of a signal changes over time.
//!
//! # Documentation Sources
//!
//! - Praat manual: Sound: To Spectrogram...
//! - Standard STFT definition from signal processing textbooks
//!
//! # Key Documented Facts
//!
//! From the Praat manual:
//!
//! - **Gaussian window**: "analyzes a factor of 2 slower... twice as many samples"
//!   This means the physical window is 2× the effective (user-specified) window length.
//!
//! - **Gaussian -3dB bandwidth**: 1.2982804 / window_length
//!
//! - **Time step minimum**: never less than 1/(8√π) × window_length
//!
//! - **Frequency step minimum**: never less than (√π)/8 / window_length
//!
//! # Algorithm Overview
//!
//! The STFT computes a sequence of FFTs on overlapping windowed segments:
//!
//! 1. **Windowing**: Multiply each segment by a window function (Gaussian or Hanning)
//!    to reduce spectral leakage from edge discontinuities.
//!
//! 2. **FFT**: Compute the Fourier transform of each windowed segment.
//!
//! 3. **Power**: Compute |X(f)|² for each frequency bin.
//!
//! 4. **Grid**: Arrange power values on a time-frequency grid.
//!
//! # Time-Frequency Trade-off
//!
//! The window length controls the trade-off between time and frequency resolution:
//!
//! - **Shorter windows**: Better time resolution, worse frequency resolution
//! - **Longer windows**: Better frequency resolution, worse time resolution
//!
//! This is a fundamental property of the Fourier transform (uncertainty principle).

use ndarray::Array2;
use rustfft::{num_complex::Complex, FftPlanner};

use crate::sound::Sound;

/// Window shape for spectrogram analysis.
///
/// The window function determines the frequency response characteristics
/// of the analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowShape {
    /// Gaussian window.
    ///
    /// - Physical window is 2× effective window length
    /// - Provides excellent frequency resolution
    /// - Has very low sidelobes (-60 to -100 dB depending on α)
    /// - Standard choice for spectrograms in speech analysis
    Gaussian,

    /// Hanning window.
    ///
    /// - Physical window equals effective window length
    /// - Good balance of main lobe width and sidelobe level
    /// - Sidelobes at -31 dB
    Hanning,
}

/// Time-frequency representation (power spectral density over time).
///
/// The spectrogram stores power values on a 2D grid:
/// - Rows: frequency bins (0 to max_frequency)
/// - Columns: time frames
///
/// Power values are typically displayed on a logarithmic (dB) scale
/// and color-coded in spectrogram visualizations.
#[derive(Debug, Clone)]
pub struct Spectrogram {
    /// Power values (n_freqs × n_times).
    ///
    /// values[freq_bin, time_frame] = |X(f,t)|²
    ///
    /// These are linear power values; convert to dB with 10×log₁₀(power).
    values: Array2<f64>,

    /// Start time in seconds.
    time_min: f64,

    /// End time in seconds.
    time_max: f64,

    /// Minimum frequency in Hz (typically 0).
    freq_min: f64,

    /// Maximum frequency in Hz (user-specified).
    freq_max: f64,

    /// Time step between frames in seconds.
    time_step: f64,

    /// Frequency step between bins in Hz.
    freq_step: f64,

    /// Time of first frame center in seconds.
    ///
    /// Frames are centered in the signal, so t1 > 0 due to the
    /// half-window margin needed at the start.
    t1: f64,
}

impl Spectrogram {
    /// Create a new Spectrogram object.
    ///
    /// # Arguments
    ///
    /// * `values` - 2D array of power values (n_freqs × n_times)
    /// * `time_min` - Start time in seconds (typically 0)
    /// * `time_max` - End time in seconds (signal duration)
    /// * `freq_min` - Minimum frequency in Hz (typically 0)
    /// * `freq_max` - Maximum frequency in Hz
    /// * `time_step` - Time step between frames
    /// * `freq_step` - Frequency step between bins
    /// * `t1` - Time of first frame center
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        values: Array2<f64>,
        time_min: f64,
        time_max: f64,
        freq_min: f64,
        freq_max: f64,
        time_step: f64,
        freq_step: f64,
        t1: f64,
    ) -> Self {
        Self {
            values,
            time_min,
            time_max,
            freq_min,
            freq_max,
            time_step,
            freq_step,
            t1,
        }
    }

    /// Get the power values (n_freqs × n_times).
    ///
    /// Access individual values with values[[freq_bin, time_frame]].
    #[inline]
    pub fn values(&self) -> &Array2<f64> {
        &self.values
    }

    /// Number of time frames.
    #[inline]
    pub fn n_times(&self) -> usize {
        self.values.ncols()
    }

    /// Number of frequency bins.
    #[inline]
    pub fn n_freqs(&self) -> usize {
        self.values.nrows()
    }

    /// Start time in seconds.
    #[inline]
    pub fn time_min(&self) -> f64 {
        self.time_min
    }

    /// End time in seconds.
    #[inline]
    pub fn time_max(&self) -> f64 {
        self.time_max
    }

    /// Minimum frequency in Hz.
    #[inline]
    pub fn freq_min(&self) -> f64 {
        self.freq_min
    }

    /// Maximum frequency in Hz.
    #[inline]
    pub fn freq_max(&self) -> f64 {
        self.freq_max
    }

    /// Time step between frames.
    #[inline]
    pub fn time_step(&self) -> f64 {
        self.time_step
    }

    /// Frequency step between bins.
    #[inline]
    pub fn freq_step(&self) -> f64 {
        self.freq_step
    }

    /// Get time for a frame index (0-based).
    ///
    /// Time = t1 + frame × time_step
    #[inline]
    pub fn get_time_from_frame(&self, frame: usize) -> f64 {
        self.t1 + frame as f64 * self.time_step
    }

    /// Get frequency for a bin index (0-based).
    ///
    /// Frequency = freq_min + bin × freq_step
    #[inline]
    pub fn get_freq_from_bin(&self, bin_index: usize) -> f64 {
        self.freq_min + bin_index as f64 * self.freq_step
    }

    /// Get array of all time points.
    pub fn times(&self) -> Vec<f64> {
        (0..self.n_times())
            .map(|i| self.get_time_from_frame(i))
            .collect()
    }

    /// Get array of all frequency points.
    pub fn frequencies(&self) -> Vec<f64> {
        (0..self.n_freqs())
            .map(|i| self.get_freq_from_bin(i))
            .collect()
    }
}

/// Generate Gaussian window for spectrogram.
///
/// The Gaussian window provides excellent frequency resolution with very low
/// sidelobes. The window is energy-normalized to ensure consistent power
/// measurements across different window sizes.
///
/// # Formula
///
/// ```text
/// w(n) = exp(-α × ((n - center) / center)²)
/// ```
///
/// where center = (N-1)/2 is the window center.
///
/// # Arguments
///
/// * `n` - Number of samples in the window
/// * `alpha` - Shape parameter controlling the window width.
///   Higher α = narrower main lobe, lower sidelobes.
///   α = 12.0 is typical for spectrograms.
///
/// # Returns
///
/// Energy-normalized window coefficients (sum of squares = 1)
fn gaussian_window(n: usize, alpha: f64) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0];
    }

    // Window center
    let mid = (n - 1) as f64 / 2.0;

    // Generate unnormalized Gaussian
    let window: Vec<f64> = (0..n)
        .map(|i| {
            // Map to [-1, 1] range centered at mid
            let x = (i as f64 - mid) / mid;
            // Gaussian: exp(-α × x²)
            (-alpha * x * x).exp()
        })
        .collect();

    // Energy normalization: divide by sqrt(sum of squares)
    // This ensures consistent power measurements regardless of window size
    let energy: f64 = window.iter().map(|&w| w * w).sum();
    let norm = energy.sqrt();

    window.iter().map(|&w| w / norm).collect()
}

/// Generate Hanning window.
///
/// The Hanning (or Hann) window is a raised cosine window that provides
/// a good trade-off between main lobe width and sidelobe level.
///
/// # Formula
///
/// ```text
/// w(n) = 0.5 - 0.5 × cos(2πn / (N-1))
/// ```
///
/// # Arguments
///
/// * `n` - Number of samples in the window
fn hanning_window(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0];
    }

    (0..n)
        .map(|i| {
            0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
        })
        .collect()
}

/// Compute spectrogram from sound using Short-Time Fourier Transform.
///
/// This is the main entry point for spectrogram computation, using a Gaussian
/// window by default.
///
/// # Arguments
///
/// * `sound` - Sound object to analyze
/// * `window_length` - Effective window length in seconds (typical: 0.005).
///   For Gaussian windows, the physical window is 2× this value.
/// * `max_frequency` - Maximum frequency in Hz (typical: 5000).
///   Higher values show more of the spectrum but require more computation.
/// * `time_step` - Time step between frames in seconds (typical: 0.002).
///   Smaller values give finer time resolution.
/// * `frequency_step` - Frequency resolution in Hz (typical: 20).
///   Smaller values give finer frequency resolution.
///
/// # Returns
///
/// Spectrogram object containing the time-frequency power representation.
pub fn sound_to_spectrogram(
    sound: &Sound,
    window_length: f64,
    max_frequency: f64,
    time_step: f64,
    frequency_step: f64,
) -> Spectrogram {
    sound_to_spectrogram_with_shape(
        sound,
        window_length,
        max_frequency,
        time_step,
        frequency_step,
        WindowShape::Gaussian,
    )
}

/// Compute spectrogram with explicit window shape.
///
/// # Algorithm
///
/// 1. **Frame timing**: Calculate frame positions centered in the signal
///
/// 2. **Window generation**: Create window function (Gaussian or Hanning)
///
/// 3. **For each frame**:
///    - Extract samples centered at frame time
///    - Apply window function
///    - Zero-pad to FFT size
///    - Compute FFT
///    - Compute power: |X(f)|²
///    - Store power at desired frequency bins
///
/// # Arguments
///
/// * `sound` - Sound object to analyze
/// * `window_length` - Effective window length in seconds
/// * `max_frequency` - Maximum frequency in Hz
/// * `time_step` - Time step between frames in seconds
/// * `frequency_step` - Frequency resolution in Hz
/// * `window_shape` - Window function to use (Gaussian or Hanning)
pub fn sound_to_spectrogram_with_shape(
    sound: &Sound,
    window_length: f64,
    max_frequency: f64,
    time_step: f64,
    frequency_step: f64,
    window_shape: WindowShape,
) -> Spectrogram {
    let samples = sound.samples();
    let sample_rate = sound.sample_rate();
    let duration = sound.duration();

    // Physical window duration
    // For Gaussian: physical = 2× effective (documented in Praat manual)
    // For Hanning: physical = effective
    let physical_window_duration = match window_shape {
        WindowShape::Gaussian => 2.0 * window_length,
        WindowShape::Hanning => window_length,
    };

    // Number of samples in physical window
    let mut window_samples = (physical_window_duration * sample_rate).round() as usize;
    if window_samples % 2 == 0 {
        window_samples += 1;  // Ensure odd for symmetric window
    }
    let half_window = window_samples / 2;

    // Generate window function
    let window = match window_shape {
        // α = 12.0 gives good frequency resolution with reasonable sidelobes
        WindowShape::Gaussian => gaussian_window(window_samples, 12.0),
        WindowShape::Hanning => hanning_window(window_samples),
    };

    // Frame timing: centered in signal
    // First frame must be at least half_window from start
    // Last frame must be at least half_window from end
    let n_frames = ((duration - physical_window_duration) / time_step).floor() as usize + 1;
    let n_frames = n_frames.max(1);

    // Center frames symmetrically in signal
    let t1 = (duration - (n_frames - 1) as f64 * time_step) / 2.0;

    // FFT size determination
    // Must be large enough for:
    // - The window samples (obviously)
    // - Desired frequency resolution: sample_rate / fft_size ≤ frequency_step
    let min_fft_size = (sample_rate / frequency_step).ceil() as usize;
    let mut fft_size = 1;
    while fft_size < window_samples.max(min_fft_size) {
        fft_size *= 2;  // Use power of 2 for efficient FFT
    }

    // Actual frequency resolution from FFT (may be finer than user-requested)
    // Store at FFT bin resolution, matching Praat's behavior
    let df_fft = sample_rate / fft_size as f64;

    // Number of frequency bins from 0 to max_frequency at FFT resolution
    let n_freq_bins = (max_frequency / df_fft).round() as usize;

    // Set up FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    // Initialize output array: (n_freq_bins rows, n_frames columns)
    let mut values = Array2::<f64>::zeros((n_freq_bins, n_frames));

    let n_samples = samples.len();
    let samples_slice = samples.as_slice().unwrap();

    // Process each frame
    for i in 0..n_frames {
        let t = t1 + i as f64 * time_step;

        // Extract frame centered at time t
        let center = (t * sample_rate).round() as isize;
        let start = center - half_window as isize;
        let end = start + window_samples as isize;

        // Handle boundaries with zero-padding
        let mut frame = vec![0.0; window_samples];
        if start < 0 || end > n_samples as isize {
            // Partial overlap with signal boundaries
            let src_start = 0.max(start) as usize;
            let src_end = (n_samples as isize).min(end) as usize;
            let dst_start = (src_start as isize - start) as usize;
            let dst_end = dst_start + (src_end - src_start);
            frame[dst_start..dst_end].copy_from_slice(&samples_slice[src_start..src_end]);
        } else {
            // Full overlap - copy all samples
            let start = start as usize;
            let end = end as usize;
            frame.copy_from_slice(&samples_slice[start..end]);
        }

        // Apply window function
        // This reduces spectral leakage from edge discontinuities
        let windowed: Vec<f64> = frame
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect();

        // Zero-pad to FFT size
        // Zero-padding provides frequency domain interpolation
        let mut buffer: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); fft_size];
        for (j, &s) in windowed.iter().enumerate() {
            buffer[j] = Complex::new(s, 0.0);
        }

        // Compute FFT: X[k] = Σ x[n] × e^(-2πikn/N)
        fft.process(&mut buffer);

        // Compute power spectrum: |X(f)|² = Re² + Im²
        // Only need positive frequencies (0 to Nyquist)
        let power: Vec<f64> = buffer[..fft_size / 2 + 1]
            .iter()
            .map(|c| c.norm_sqr())  // |c|² = Re² + Im²
            .collect();

        // Store power directly at FFT bin resolution
        for j in 0..n_freq_bins.min(power.len()) {
            values[[j, i]] = power[j];
        }
    }

    Spectrogram::new(
        values,
        0.0,           // time_min
        duration,      // time_max
        0.0,           // freq_min
        max_frequency, // freq_max
        time_step,
        df_fft,        // actual FFT frequency resolution
        t1,
    )
}
