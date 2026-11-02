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

use crate::parallel::map_frames;
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

/// Modified Bessel function of the first kind, order 0, by its power series
/// I0(x) = sum_k ((x/2)^k / k!)^2 (standard numerical analysis). Mirrors
/// `_bessel_i0` in `src/praatfan/intensity.py` term for term.
fn bessel_i0(x: f64) -> f64 {
    let half = 0.5 * x;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 1.0;
    loop {
        let r = half / k;
        term *= r * r;
        sum += term;
        if term < sum * 1e-17 {
            return sum;
        }
        k += 1.0;
    }
}

/// Kaiser window with shape parameter `beta` over `n` samples:
/// `w[i] = I0(beta * sqrt(1 - x^2)) / I0(beta)`, x from -1 to 1.
/// Praat's manual names the intensity window "Kaiser-20" (beta = 20).
fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0];
    }
    let mid = (n - 1) as f64 / 2.0;
    let norm = bessel_i0(beta);
    (0..n)
        .map(|i| {
            let x = (i as f64 - mid) / mid;
            let arg = (1.0 - x * x).max(0.0).sqrt();
            bessel_i0(beta * arg) / norm
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

    // Default time step (documented: 0.8 / min_pitch = 1/4 of effective duration)
    // For 75 Hz min_pitch: 0.8/75 ≈ 10.7 ms
    let time_step = if time_step <= 0.0 {
        0.8 / min_pitch
    } else {
        time_step
    };

    // Window (2026-11-02): the documented Kaiser-20 window over a physical
    // duration of 2x the documented effective duration (3.2 / min_pitch).
    // Earlier versions used a Gaussian (alpha 13.2) over 7.2 / min_pitch,
    // fitted against a grid that was 4 ms off. See src/praatfan/intensity.py.
    let physical_window_duration = 6.4 / min_pitch;

    // Convert window duration to samples
    let mut window_samples = (physical_window_duration * sample_rate).round() as usize;

    // Ensure odd number of samples for symmetric window centered on a sample
    if window_samples % 2 == 0 {
        window_samples += 1;
    }

    let half_window_samples = window_samples / 2;

    let window = kaiser_window(window_samples, 20.0);

    // Frame timing (black-box, 2026-11-02): frames are centred in the sound,
    // their count set by a span of 6.4 / min_pitch. The duration is taken as
    // n_samples * (1 / sample_rate): at exact frame boundaries only that form
    // reproduced Praat's count.
    let duration = samples.len() as f64 * (1.0 / sample_rate);
    let frame_span = 6.4 / min_pitch;
    let n_frames = ((duration - frame_span) / time_step).floor();
    let n_frames = if n_frames >= 0.0 { n_frames as usize + 1 } else { 1 };
    let t1 = (duration - (n_frames - 1) as f64 * time_step) / 2.0;

    // Reference pressure squared: (2×10⁻⁵ Pa)² = 4×10⁻¹⁰ Pa²
    // This is the standard reference for sound pressure level (SPL) in air
    // 0 dB SPL corresponds to the threshold of human hearing at 1 kHz
    let p_ref = 4e-10;

    let n_samples = samples.len();

    // Process each frame (in parallel, order-preserving; bit-identical to a
    // serial loop since each frame's arithmetic is unchanged)
    // Weight normalisation, identical for every frame
    let window_sum: f64 = window.iter().sum();

    let frames: Vec<(f64, f64)> = map_frames(n_frames, || vec![0.0; window_samples], |frame_samples, i| {
        // Frame center time
        let t = t1 + i as f64 * time_step;

        // Centre on the sample nearest t; sample i sits at (i + 0.5) / sr.
        let center_sample = (t * sample_rate - 0.5).round() as isize;

        // Calculate start of window region
        let start_sample = center_sample - half_window_samples as isize;

        // Extract frame samples with boundary handling
        // Samples outside the signal range are treated as zero
        // (scratch is reused, so every element is written)
        for j in 0..window_samples {
            let src_idx = start_sample + j as isize;
            frame_samples[j] = if src_idx >= 0 && (src_idx as usize) < n_samples {
                samples[src_idx as usize]
            } else {
                0.0 // zero-padding at edges
            };
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
        let mean_square: f64 = frame_samples
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * s * w)  // s² × w
            .sum::<f64>()
            / window_sum;

        // Convert to dB: 10 × log₁₀(mean_square / p_ref²). Digital silence is
        // -inf (Praat floors at -300 dB; we keep -inf, which callers rely on).
        let intensity_db = if mean_square <= 0.0 {
            f64::NEG_INFINITY
        } else {
            10.0 * libm::log10(mean_square / p_ref)
        };

        (t, intensity_db)
    });
    let (times, values): (Vec<f64>, Vec<f64>) = frames.into_iter().unzip();

    Intensity::new(
        Array1::from_vec(times),
        Array1::from_vec(values),
        time_step,
        min_pitch,
    )
}
