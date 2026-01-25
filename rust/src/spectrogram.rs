//! Spectrogram - Time-frequency representation.
//!
//! Documentation sources:
//! - Praat manual: Sound: To Spectrogram...
//! - Standard STFT definition
//!
//! Key documented facts:
//! - Gaussian window: "analyzes a factor of 2 slower... twice as many samples"
//! - Gaussian -3dB bandwidth: 1.2982804 / window_length
//! - Time step minimum: never less than 1/(8√π) × window_length
//! - Frequency step minimum: never less than (√π)/8 / window_length

use ndarray::Array2;
use rustfft::{num_complex::Complex, FftPlanner};

use crate::sound::Sound;

/// Window shape for spectrogram analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowShape {
    /// Gaussian window (physical window is 2x effective).
    Gaussian,
    /// Hanning window.
    Hanning,
}

/// Time-frequency representation (power spectral density over time).
#[derive(Debug, Clone)]
pub struct Spectrogram {
    /// Power values (n_freqs × n_times).
    values: Array2<f64>,
    /// Start time in seconds.
    time_min: f64,
    /// End time in seconds.
    time_max: f64,
    /// Minimum frequency in Hz.
    freq_min: f64,
    /// Maximum frequency in Hz.
    freq_max: f64,
    /// Time step between frames.
    time_step: f64,
    /// Frequency step between bins.
    freq_step: f64,
    /// Time of first frame center.
    t1: f64,
}

impl Spectrogram {
    /// Create a new Spectrogram object.
    ///
    /// # Arguments
    ///
    /// * `values` - 2D array of power values (n_freqs × n_times)
    /// * `time_min` - Start time in seconds
    /// * `time_max` - End time in seconds
    /// * `freq_min` - Minimum frequency in Hz
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
    #[inline]
    pub fn get_time_from_frame(&self, frame: usize) -> f64 {
        self.t1 + frame as f64 * self.time_step
    }

    /// Get frequency for a bin index (0-based).
    #[inline]
    pub fn get_freq_from_bin(&self, bin_index: usize) -> f64 {
        self.freq_min + bin_index as f64 * self.freq_step
    }

    /// Get array of time points.
    pub fn times(&self) -> Vec<f64> {
        (0..self.n_times())
            .map(|i| self.get_time_from_frame(i))
            .collect()
    }

    /// Get array of frequency points.
    pub fn frequencies(&self) -> Vec<f64> {
        (0..self.n_freqs())
            .map(|i| self.get_freq_from_bin(i))
            .collect()
    }
}

/// Generate Gaussian window for spectrogram.
///
/// Uses the same Gaussian formula as Intensity analysis.
fn gaussian_window(n: usize, alpha: f64) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0];
    }

    let mid = (n - 1) as f64 / 2.0;

    let window: Vec<f64> = (0..n)
        .map(|i| {
            let x = (i as f64 - mid) / mid;
            (-alpha * x * x).exp()
        })
        .collect();

    // Energy normalization: divide by sqrt(sum of squares)
    let energy: f64 = window.iter().map(|&w| w * w).sum();
    let norm = energy.sqrt();

    window.iter().map(|&w| w / norm).collect()
}

/// Generate Hanning window.
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
/// # Arguments
///
/// * `sound` - Sound object
/// * `window_length` - Effective window length in seconds (default 0.005)
/// * `max_frequency` - Maximum frequency in Hz (default 5000)
/// * `time_step` - Time step in seconds (default 0.002)
/// * `frequency_step` - Frequency step in Hz (default 20)
///
/// # Returns
///
/// Spectrogram object
///
/// Note:
/// For Gaussian windows, the physical window is twice the effective
/// window length (documented: "analyzes a factor of 2 slower").
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

    // For Gaussian window, physical window is 2x effective window
    let physical_window_duration = match window_shape {
        WindowShape::Gaussian => 2.0 * window_length,
        WindowShape::Hanning => window_length,
    };

    // Number of samples in physical window
    let mut window_samples = (physical_window_duration * sample_rate).round() as usize;
    if window_samples % 2 == 0 {
        window_samples += 1;
    }
    let half_window = window_samples / 2;

    // Generate window function
    let window = match window_shape {
        WindowShape::Gaussian => gaussian_window(window_samples, 12.0),
        WindowShape::Hanning => hanning_window(window_samples),
    };

    // Frame timing - centered
    let n_frames = ((duration - physical_window_duration) / time_step).floor() as usize + 1;
    let n_frames = n_frames.max(1);
    let t1 = (duration - (n_frames - 1) as f64 * time_step) / 2.0;

    // Frequency bins from 0 to max_frequency (exclusive at max)
    let n_freq_bins = (max_frequency / frequency_step) as usize;

    // FFT size - must be large enough for desired frequency resolution
    let min_fft_size = (sample_rate / frequency_step).ceil() as usize;
    let mut fft_size = 1;
    while fft_size < window_samples.max(min_fft_size) {
        fft_size *= 2;
    }

    // Frequency resolution from FFT
    let df_fft = sample_rate / fft_size as f64;

    // Set up FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    // Compute spectrogram
    // Values array is (n_freq_bins, n_frames)
    let mut values = Array2::<f64>::zeros((n_freq_bins, n_frames));

    let n_samples = samples.len();
    let samples_slice = samples.as_slice().unwrap();

    for i in 0..n_frames {
        let t = t1 + i as f64 * time_step;

        // Extract frame centered at t
        let center = (t * sample_rate).round() as isize;
        let start = center - half_window as isize;
        let end = start + window_samples as isize;

        // Handle boundaries
        let mut frame = vec![0.0; window_samples];
        if start < 0 || end > n_samples as isize {
            let src_start = 0.max(start) as usize;
            let src_end = (n_samples as isize).min(end) as usize;
            let dst_start = (src_start as isize - start) as usize;
            let dst_end = dst_start + (src_end - src_start);
            frame[dst_start..dst_end].copy_from_slice(&samples_slice[src_start..src_end]);
        } else {
            let start = start as usize;
            let end = end as usize;
            frame.copy_from_slice(&samples_slice[start..end]);
        }

        // Apply window
        let windowed: Vec<f64> = frame
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect();

        // Zero-pad to FFT size
        let mut buffer: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); fft_size];
        for (j, &s) in windowed.iter().enumerate() {
            buffer[j] = Complex::new(s, 0.0);
        }

        // Compute FFT
        fft.process(&mut buffer);

        // Compute power: |X(f)|^2
        let power: Vec<f64> = buffer[..fft_size / 2 + 1]
            .iter()
            .map(|c| c.norm_sqr())
            .collect();

        // Extract power at desired frequency bins
        for j in 0..n_freq_bins {
            let freq = j as f64 * frequency_step;
            let fft_bin = (freq / df_fft).round() as usize;
            if fft_bin < power.len() {
                values[[j, i]] = power[fft_bin];
            }
        }
    }

    Spectrogram::new(
        values,
        0.0,
        duration,
        0.0,
        max_frequency,
        time_step,
        frequency_step,
        t1,
    )
}
