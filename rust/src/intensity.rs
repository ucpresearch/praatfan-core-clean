//! Intensity - RMS energy contour in dB.
//!
//! Documentation sources:
//! - Praat manual: Sound: To Intensity...
//! - Praat manual: Intro 6.2. Configuring the intensity contour
//!
//! Key documented facts:
//! - Window: "Gaussian analysis window (Kaiser-20; sidelobes below -190 dB)"
//! - Effective duration: 3.2 / min_pitch
//! - Default time step: 0.8 / min_pitch (1/4 of effective duration)
//! - DC removal: "subtracting the mean... then applying the window"
//!
//! Decision points determined via testing:
//! - DP3: Window type - Gaussian α=13.2 gave best results
//! - DP4: Physical vs effective window ratio = 2.25×
//! - DP5: DC removal = unweighted mean subtraction before windowing

use ndarray::Array1;

use crate::sound::Sound;

/// Interpolation method for getting values at specific times.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// Nearest neighbor interpolation.
    Nearest,
    /// Linear interpolation.
    Linear,
    /// Cubic (Catmull-Rom) interpolation.
    Cubic,
}

/// Intensity contour (loudness over time).
///
/// Values are in dB relative to a reference pressure.
#[derive(Debug, Clone)]
pub struct Intensity {
    /// Time points in seconds.
    times: Array1<f64>,
    /// Intensity values in dB.
    values: Array1<f64>,
    /// Time step between frames.
    time_step: f64,
    /// Minimum pitch used for analysis.
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
    /// * `interpolation` - Interpolation method
    ///
    /// # Returns
    ///
    /// Intensity in dB, or None if outside range
    pub fn get_value_at_time(&self, time: f64, interpolation: Interpolation) -> Option<f64> {
        if self.n_frames() == 0 {
            return None;
        }

        let t0 = self.times[0];
        let idx_float = (time - t0) / self.time_step;

        if idx_float < -0.5 || idx_float > self.n_frames() as f64 - 0.5 {
            return None;
        }

        match interpolation {
            Interpolation::Nearest => {
                let idx = idx_float.round() as usize;
                let idx = idx.min(self.n_frames() - 1);
                Some(self.values[idx])
            }
            Interpolation::Linear => {
                let idx = idx_float.floor() as isize;
                if idx < 0 {
                    return Some(self.values[0]);
                }
                let idx = idx as usize;
                if idx >= self.n_frames() - 1 {
                    return Some(self.values[self.n_frames() - 1]);
                }
                let frac = idx_float - idx as f64;
                Some(self.values[idx] * (1.0 - frac) + self.values[idx + 1] * frac)
            }
            Interpolation::Cubic => {
                let idx = idx_float.floor() as isize;
                let frac = idx_float - idx as f64;

                // Get 4 surrounding points for cubic interpolation
                let n = self.n_frames();
                let i0 = 0.max(idx - 1) as usize;
                let i1 = 0.max(idx).min(n as isize - 1) as usize;
                let i2 = 0.max(idx + 1).min(n as isize - 1) as usize;
                let i3 = (idx + 2).min(n as isize - 1) as usize;

                let y0 = self.values[i0];
                let y1 = self.values[i1];
                let y2 = self.values[i2];
                let y3 = self.values[i3];

                // Cubic interpolation (Catmull-Rom spline)
                let t = frac;
                let t2 = t * t;
                let t3 = t2 * t;

                let result = 0.5
                    * ((2.0 * y1)
                        + (-y0 + y2) * t
                        + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t2
                        + (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t3);
                Some(result)
            }
        }
    }
}

/// Generate Gaussian window matching Praat's intensity analysis.
fn gauss_window(n: usize, alpha: f64) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0];
    }

    let mid = (n - 1) as f64 / 2.0;
    let exp_edge = (-alpha).exp();
    let norm = 1.0 - exp_edge;

    (0..n)
        .map(|i| {
            let x = (i as f64 - mid) / mid; // Range [-1, 1]
            let exp_term = (-alpha * x * x).exp();
            (exp_term - exp_edge) / norm
        })
        .collect()
}

/// Compute intensity from sound.
///
/// Algorithm (from Praat manual):
/// 1. Square the signal values
/// 2. Convolve with Gaussian window (after DC removal if enabled)
/// 3. Convert to dB
///
/// # Arguments
///
/// * `sound` - Sound object
/// * `min_pitch` - Minimum pitch in Hz (determines window size)
/// * `time_step` - Time step in seconds (0 = auto: 0.8 / min_pitch)
///
/// # Returns
///
/// Intensity object
pub fn sound_to_intensity(sound: &Sound, min_pitch: f64, time_step: f64) -> Intensity {
    let samples = sound.samples();
    let sample_rate = sound.sample_rate();
    let duration = sound.duration();

    // Default time step (documented)
    let time_step = if time_step <= 0.0 {
        0.8 / min_pitch
    } else {
        time_step
    };

    // Window duration (determined via black-box testing: ratio = 2.25)
    let physical_window_duration = 7.2 / min_pitch;
    let half_window_duration = physical_window_duration / 2.0;

    // Number of samples in window
    let mut window_samples = (physical_window_duration * sample_rate).round() as usize;
    if window_samples % 2 == 0 {
        window_samples += 1; // Ensure odd for symmetric window
    }

    let half_window_samples = window_samples / 2;

    // Generate window (alpha = 13.2 determined via black-box testing)
    let window = gauss_window(window_samples, 13.2);

    // Frame timing (left-aligned)
    let t1 = half_window_duration;
    let t_max = duration - half_window_duration;

    let n_frames = ((t_max - t1) / time_step + 1e-9).floor() as usize + 1;
    let n_frames = n_frames.max(1);

    let mut times = Vec::with_capacity(n_frames);
    let mut values = Vec::with_capacity(n_frames);

    // Reference pressure squared (standard: 2e-5 Pa)
    let p_ref = 4e-10; // (2e-5)²

    let n_samples = samples.len();

    for i in 0..n_frames {
        let t = t1 + i as f64 * time_step;
        times.push(t);

        // Find center sample index
        let center_sample = (t * sample_rate).round() as isize;

        // Extract window region
        let start_sample = center_sample - half_window_samples as isize;

        // Extract frame samples with boundary handling
        let mut frame_samples = vec![0.0; window_samples];
        for j in 0..window_samples {
            let src_idx = start_sample + j as isize;
            if src_idx >= 0 && (src_idx as usize) < n_samples {
                frame_samples[j] = samples[src_idx as usize];
            }
        }

        // Subtract mean (DC removal)
        let mean: f64 = frame_samples.iter().sum::<f64>() / frame_samples.len() as f64;
        for s in frame_samples.iter_mut() {
            *s -= mean;
        }

        // Compute weighted mean square
        let window_sum: f64 = window.iter().sum();
        let mean_square: f64 = frame_samples
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * s * w)
            .sum::<f64>()
            / window_sum;

        // Convert to dB
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
