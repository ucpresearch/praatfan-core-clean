//! Intensity - RMS energy contour in dB.
//!
//! This module computes the loudness (intensity) of an audio signal over time,
//! expressed in decibels (dB) relative to a reference sound pressure level.
//!
//! # Documentation Sources
//!
//! - Praat manual: Sound: To Intensity...
//! - Praat manual: Intro 6.2. Configuring the intensity contour
//!
//! # Key Documented Facts
//!
//! From the Praat manual:
//! - Window: "Gaussian analysis window (Kaiser-20; sidelobes below -190 dB)"
//! - Effective duration: 3.2 / min_pitch
//! - Default time step: 0.8 / min_pitch (1/4 of effective duration)
//! - DC removal: "subtracting the mean... then applying the window"
//!
//! # Decision Points Determined via Testing
//!
//! These values were found through black-box testing against parselmouth:
//! - **DP3**: Window type - Gaussian with α=13.2 gave best match
//! - **DP4**: Physical vs effective window ratio = 2.25×
//! - **DP5**: DC removal = unweighted mean subtraction before windowing
//!
//! # Algorithm Overview
//!
//! For each analysis frame:
//! 1. Extract samples centered at frame time
//! 2. Subtract DC (mean) from the samples
//! 3. Apply Gaussian window
//! 4. Compute weighted mean square (RMS squared)
//! 5. Convert to dB relative to reference pressure

use ndarray::Array1;

use crate::sound::Sound;

/// Interpolation method for getting values at specific times.
///
/// Different interpolation methods offer trade-offs between smoothness
/// and computational cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// Nearest neighbor interpolation.
    ///
    /// Returns the value of the nearest frame. Fast but produces
    /// step-like output when querying between frames.
    Nearest,

    /// Linear interpolation.
    ///
    /// Interpolates linearly between two adjacent frames. Good balance
    /// of smoothness and speed.
    Linear,

    /// Cubic (Catmull-Rom) interpolation.
    ///
    /// Uses four surrounding frames for smooth interpolation. Produces
    /// continuous first derivatives, suitable for smooth contours.
    Cubic,
}

/// Intensity contour (loudness over time).
///
/// Values are in dB relative to a reference pressure (2×10⁻⁵ Pa, the
/// standard reference for sound pressure level in air).
///
/// # Typical Values
///
/// - -∞ dB: Complete silence
/// - ~30 dB: Whisper
/// - ~60 dB: Normal conversation
/// - ~80 dB: Loud speech
/// - ~100+ dB: Very loud sounds (shouting, music)
#[derive(Debug, Clone)]
pub struct Intensity {
    /// Time points in seconds.
    ///
    /// Each time corresponds to the center of an analysis frame.
    times: Array1<f64>,

    /// Intensity values in dB.
    ///
    /// Values are relative to reference pressure squared (4×10⁻¹⁰ Pa²).
    /// Negative infinity indicates complete silence (zero energy).
    values: Array1<f64>,

    /// Time step between frames.
    ///
    /// Default: 0.8 / min_pitch (about 10.7 ms for 75 Hz min_pitch)
    time_step: f64,

    /// Minimum pitch used for analysis.
    ///
    /// This parameter determines the window size: lower min_pitch means
    /// longer windows, which produces smoother intensity contours but
    /// less temporal precision.
    min_pitch: f64,
}

impl Intensity {
    /// Create a new Intensity object.
    pub fn new(times: Array1<f64>, values: Array1<f64>, time_step: f64, min_pitch: f64) -> Self {
        Self {
            times,
            values,
            time_step,
            min_pitch,
        }
    }

    /// Get the time points in seconds.
    #[inline]
    pub fn times(&self) -> &Array1<f64> {
        &self.times
    }

    /// Get the intensity values in dB.
    #[inline]
    pub fn values(&self) -> &Array1<f64> {
        &self.values
    }

    /// Get the number of frames.
    #[inline]
    pub fn n_frames(&self) -> usize {
        self.times.len()
    }

    /// Get the time step between frames.
    #[inline]
    pub fn time_step(&self) -> f64 {
        self.time_step
    }

    /// Get the minimum pitch used for analysis.
    #[inline]
    pub fn min_pitch(&self) -> f64 {
        self.min_pitch
    }

    /// Get intensity value at a specific time.
    ///
    /// # Arguments
    ///
    /// * `time` - Time in seconds
    /// * `interpolation` - Interpolation method to use
    ///
    /// # Returns
    ///
    /// Intensity in dB, or None if the time is outside the analysis range.
    ///
    /// # Boundary Behavior
    ///
    /// Returns None if time is more than half a time step before the first
    /// frame or after the last frame. Within this boundary, edge values
    /// are extrapolated using the specified interpolation method.
    pub fn get_value_at_time(&self, time: f64, interpolation: Interpolation) -> Option<f64> {
        if self.n_frames() == 0 {
            return None;
        }

        // Compute floating-point index into the frame array
        let t0 = self.times[0];
        let idx_float = (time - t0) / self.time_step;

        // Check bounds: allow ±0.5 frame tolerance at edges
        if idx_float < -0.5 || idx_float > self.n_frames() as f64 - 0.5 {
            return None;
        }

        match interpolation {
            Interpolation::Nearest => {
                // Round to nearest frame index
                let idx = idx_float.round() as usize;
                let idx = idx.min(self.n_frames() - 1);
                Some(self.values[idx])
            }
            Interpolation::Linear => {
                // Linear interpolation between two adjacent frames
                let idx = idx_float.floor() as isize;

                // Handle edge cases
                if idx < 0 {
                    return Some(self.values[0]);
                }
                let idx = idx as usize;
                if idx >= self.n_frames() - 1 {
                    return Some(self.values[self.n_frames() - 1]);
                }

                // Fractional position between frames
                let frac = idx_float - idx as f64;

                // Linear interpolation: v = v0 × (1-t) + v1 × t
                Some(self.values[idx] * (1.0 - frac) + self.values[idx + 1] * frac)
            }
            Interpolation::Cubic => {
                // Cubic (Catmull-Rom) interpolation using 4 surrounding points
                let idx = idx_float.floor() as isize;
                let frac = idx_float - idx as f64;

                // Get 4 surrounding point indices with boundary clamping
                let n = self.n_frames();
                let i0 = 0.max(idx - 1) as usize;
                let i1 = 0.max(idx).min(n as isize - 1) as usize;
                let i2 = 0.max(idx + 1).min(n as isize - 1) as usize;
                let i3 = (idx + 2).min(n as isize - 1) as usize;

                let y0 = self.values[i0];
                let y1 = self.values[i1];
                let y2 = self.values[i2];
                let y3 = self.values[i3];

                // Catmull-Rom spline formula
                // This produces a smooth curve that passes through all control points
                // with continuous first derivatives
                let t = frac;
                let t2 = t * t;
                let t3 = t2 * t;

                let result = 0.5
                    * ((2.0 * y1)                                    // Constant term
                        + (-y0 + y2) * t                             // Linear term
                        + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t2 // Quadratic term
                        + (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t3);    // Cubic term
                Some(result)
            }
        }
    }
}

/// Generate Gaussian window matching Praat's intensity analysis.
///
/// The Gaussian window is defined as:
/// ```text
/// w(x) = (exp(-α × x²) - exp(-α)) / (1 - exp(-α))
/// ```
///
/// where x ranges from -1 to 1 across the window.
///
/// The subtraction of exp(-α) and division by (1 - exp(-α)) ensures that:
/// - The window value is exactly 0 at the edges (x = ±1)
/// - The window value is exactly 1 at the center (x = 0)
///
/// # Arguments
///
/// * `n` - Number of samples in the window
/// * `alpha` - Shape parameter controlling the window width
///   - Higher α = narrower main lobe, lower sidelobes
///   - α = 13.2 was determined via black-box testing to match Praat
///
/// # Returns
///
/// Vector of window coefficients
fn gauss_window(n: usize, alpha: f64) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0];
    }

    // Mid-point of window (center)
    let mid = (n - 1) as f64 / 2.0;

    // Edge value: exp(-α) at x = ±1
    let exp_edge = (-alpha).exp();

    // Normalization factor: 1 - exp(-α)
    let norm = 1.0 - exp_edge;

    (0..n)
        .map(|i| {
            // Map sample index to range [-1, 1]
            let x = (i as f64 - mid) / mid;

            // Gaussian with edge subtraction and normalization
            let exp_term = (-alpha * x * x).exp();
            (exp_term - exp_edge) / norm
        })
        .collect()
}

/// Compute intensity from sound.
///
/// This function computes the loudness contour of an audio signal by
/// calculating windowed RMS energy in dB for each analysis frame.
///
/// # Algorithm
///
/// From Praat manual documentation:
///
/// 1. **Frame Timing**: Frames are placed at regular intervals starting
///    from half the window duration into the signal.
///
/// 2. **Sample Extraction**: For each frame, extract samples centered
///    at the frame time.
///
/// 3. **DC Removal**: Subtract the mean (DC component) from the samples.
///    This prevents DC offset from inflating the intensity measurement.
///
/// 4. **Windowing**: Apply a Gaussian window to the samples. The window
///    smoothly tapers the signal to zero at the edges, reducing spectral
///    leakage and ensuring the measurement reflects the frame center.
///
/// 5. **RMS Calculation**: Compute the weighted mean square:
///    ```text
///    mean_square = Σ (sample² × window) / Σ window
///    ```
///
/// 6. **dB Conversion**: Convert to decibels relative to reference pressure:
///    ```text
///    intensity_dB = 10 × log₁₀(mean_square / p_ref²)
///    ```
///    where p_ref = 2×10⁻⁵ Pa (standard reference for SPL in air).
///
/// # Arguments
///
/// * `sound` - Sound object to analyze
/// * `min_pitch` - Minimum pitch in Hz. This determines the window size:
///   - Effective duration = 3.2 / min_pitch
///   - Physical duration = 7.2 / min_pitch (ratio 2.25× determined via testing)
///   Lower min_pitch = longer windows = smoother but less time-precise contour.
/// * `time_step` - Time step in seconds. Use 0 for automatic (0.8 / min_pitch).
///
/// # Returns
///
/// Intensity object containing time points and dB values.
pub fn sound_to_intensity(sound: &Sound, min_pitch: f64, time_step: f64) -> Intensity {
    let samples = sound.samples();
    let sample_rate = sound.sample_rate();
    let duration = sound.duration();

    // Default time step (documented: 0.8 / min_pitch = 1/4 of effective duration)
    // For 75 Hz min_pitch: 0.8/75 ≈ 10.7 ms
    let time_step = if time_step <= 0.0 {
        0.8 / min_pitch
    } else {
        time_step
    };

    // Physical window duration
    // Documented effective duration is 3.2 / min_pitch
    // Physical/effective ratio of 2.25 was determined via black-box testing
    // Physical = 3.2 × 2.25 = 7.2 / min_pitch
    let physical_window_duration = 7.2 / min_pitch;
    let half_window_duration = physical_window_duration / 2.0;

    // Convert window duration to samples
    let mut window_samples = (physical_window_duration * sample_rate).round() as usize;

    // Ensure odd number of samples for symmetric window centered on a sample
    if window_samples % 2 == 0 {
        window_samples += 1;
    }

    let half_window_samples = window_samples / 2;

    // Generate Gaussian window
    // α = 13.2 was determined via black-box testing to match Praat's output
    let window = gauss_window(window_samples, 13.2);

    // Frame timing: left-aligned, starting at half window duration
    // First frame centered at t1 = half_window_duration
    // Last possible frame at t_max = duration - half_window_duration
    let t1 = half_window_duration;
    let t_max = duration - half_window_duration;

    // Calculate number of frames (at least 1)
    let n_frames = ((t_max - t1) / time_step + 1e-9).floor() as usize + 1;
    let n_frames = n_frames.max(1);

    let mut times = Vec::with_capacity(n_frames);
    let mut values = Vec::with_capacity(n_frames);

    // Reference pressure squared: (2×10⁻⁵ Pa)² = 4×10⁻¹⁰ Pa²
    // This is the standard reference for sound pressure level (SPL) in air
    // 0 dB SPL corresponds to the threshold of human hearing at 1 kHz
    let p_ref = 4e-10;

    let n_samples = samples.len();

    // Process each frame
    for i in 0..n_frames {
        // Frame center time
        let t = t1 + i as f64 * time_step;
        times.push(t);

        // Find center sample index
        let center_sample = (t * sample_rate).round() as isize;

        // Calculate start of window region
        let start_sample = center_sample - half_window_samples as isize;

        // Extract frame samples with boundary handling
        // Samples outside the signal range are treated as zero
        let mut frame_samples = vec![0.0; window_samples];
        for j in 0..window_samples {
            let src_idx = start_sample + j as isize;
            if src_idx >= 0 && (src_idx as usize) < n_samples {
                frame_samples[j] = samples[src_idx as usize];
            }
            // Samples outside bounds remain 0 (zero-padding at edges)
        }

        // DC removal: subtract mean from samples
        // This removes any DC offset that would inflate the intensity
        // and ensures we measure only the AC (varying) component
        let mean: f64 = frame_samples.iter().sum::<f64>() / frame_samples.len() as f64;
        for s in frame_samples.iter_mut() {
            *s -= mean;
        }

        // Compute weighted mean square
        // This is the windowed RMS² value:
        // mean_square = Σ(s² × w) / Σw
        let window_sum: f64 = window.iter().sum();
        let mean_square: f64 = frame_samples
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * s * w)  // s² × w
            .sum::<f64>()
            / window_sum;

        // Convert to dB: 10 × log₁₀(mean_square / p_ref²)
        // NEG_INFINITY for silent frames (avoids log(0) = -∞)
        let intensity_db = if mean_square <= 0.0 {
            f64::NEG_INFINITY
        } else {
            10.0 * (mean_square / p_ref).log10()
        };

        values.push(intensity_db);
    }

    Intensity::new(
        Array1::from_vec(times),
        Array1::from_vec(values),
        time_step,
        min_pitch,
    )
}
