//! Harmonicity - Harmonics-to-noise ratio (HNR) contour.
//!
//! CRITICAL: Harmonicity is NOT a standalone algorithm.
//! It uses Pitch internally and extracts HNR from the correlation strength.
//!
//! Documentation sources:
//! - Praat manual: Harmonicity
//!   "if 99% of the energy of the signal is in the periodic part, and 1% is noise,
//!    the HNR is 10*log10(99/1) = 20 dB"
//! - Praat manual: Sound: To Harmonicity (ac)...
//!   "The algorithm performs an acoustic periodicity detection on the basis of
//!    an accurate autocorrelation method, as described in Boersma (1993)."
//!
//! HNR Formula (from Praat manual):
//!     HNR (dB) = 10 × log₁₀(r / (1 - r))
//!
//! where r is the normalized autocorrelation (pitch strength) at the pitch period.

use ndarray::Array1;

use crate::pitch::{sound_to_pitch_internal, FrameTiming, PitchMethod};
use crate::sound::Sound;

/// Interpolation method for harmonicity values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarmonicityInterpolation {
    /// Nearest neighbor interpolation.
    Nearest,
    /// Linear interpolation.
    Linear,
    /// Cubic (Catmull-Rom) interpolation.
    Cubic,
}

/// Harmonicity (HNR) contour.
///
/// Values are in dB. Higher values indicate more periodic (harmonic) signal.
/// Typical values for speech: 10-20 dB for vowels.
#[derive(Debug, Clone)]
pub struct Harmonicity {
    /// Time points in seconds.
    times: Array1<f64>,
    /// HNR values in dB.
    values: Array1<f64>,
    /// Time step between frames.
    time_step: f64,
    /// Minimum pitch used for analysis.
    min_pitch: f64,
}

impl Harmonicity {
    /// Create a new Harmonicity object.
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

    /// Get the HNR values in dB.
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

    /// Get HNR value at a specific time.
    ///
    /// # Arguments
    ///
    /// * `time` - Time in seconds
    /// * `interpolation` - Interpolation method
    ///
    /// # Returns
    ///
    /// HNR in dB, or None if outside range
    pub fn get_value_at_time(
        &self,
        time: f64,
        interpolation: HarmonicityInterpolation,
    ) -> Option<f64> {
        if self.n_frames() == 0 {
            return None;
        }

        let t0 = self.times[0];
        let idx_float = (time - t0) / self.time_step;

        if idx_float < -0.5 || idx_float > self.n_frames() as f64 - 0.5 {
            return None;
        }

        match interpolation {
            HarmonicityInterpolation::Nearest => {
                let idx = idx_float.round() as usize;
                let idx = idx.min(self.n_frames() - 1);
                Some(self.values[idx])
            }
            HarmonicityInterpolation::Linear => {
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
            HarmonicityInterpolation::Cubic => {
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

/// Convert pitch strength (correlation) to HNR in dB.
///
/// Formula (from Praat manual):
///     HNR = 10 × log₁₀(r / (1 - r))
///
/// # Arguments
///
/// * `r` - Pitch strength (normalized autocorrelation, 0-1)
///
/// # Returns
///
/// HNR in dB
///
/// Note:
/// - r = 0.5 → HNR = 0 dB (equal harmonic and noise energy)
/// - r = 0.99 → HNR = 20 dB
/// - r = 0.999 → HNR = 30 dB
/// Clamping is needed for r near 0 or 1.
#[inline]
pub fn strength_to_hnr(r: f64) -> f64 {
    // Clamp r to valid range [epsilon, 1-epsilon]
    // The autocorrelation normalization can produce values slightly > 1
    // due to parabolic interpolation and approximations
    let r = r.clamp(1e-10, 1.0 - 1e-10);
    10.0 * (r / (1.0 - r)).log10()
}

/// Compute harmonicity using autocorrelation method.
///
/// This function:
/// 1. Computes Pitch using AC method
/// 2. Extracts strength values from Pitch
/// 3. Applies HNR formula to get dB values
///
/// # Arguments
///
/// * `sound` - Sound object
/// * `time_step` - Time step in seconds
/// * `min_pitch` - Minimum pitch in Hz
/// * `silence_threshold` - Silence threshold (0-1)
/// * `periods_per_window` - Number of periods per window
///
/// # Returns
///
/// Harmonicity object
pub fn sound_to_harmonicity_ac(
    sound: &Sound,
    time_step: f64,
    min_pitch: f64,
    silence_threshold: f64,
    periods_per_window: f64,
) -> Harmonicity {
    // Step 1: Compute pitch using AC method with harmonicity windowing
    // Harmonicity uses periods_per_window (default 4.5) for longer windows
    // and left-aligned frame timing
    // Disable octave cost to get raw correlation strength for HNR
    let pitch = sound_to_pitch_internal(
        sound,
        time_step,
        min_pitch,
        600.0, // Standard ceiling
        PitchMethod::Ac,
        0.45,              // voicing_threshold
        silence_threshold, // Use provided silence threshold
        0.01,              // octave_cost
        0.35,              // octave_jump_cost
        0.14,              // voiced_unvoiced_cost
        periods_per_window,
        FrameTiming::Left,
        false, // apply_octave_cost = false for raw correlation
    );

    // Step 2: Extract times and convert strengths to HNR
    let times = pitch.times();
    let hnr_values: Vec<f64> = pitch
        .frames()
        .iter()
        .map(|frame| {
            if frame.voiced() {
                strength_to_hnr(frame.strength())
            } else {
                -200.0 // Unvoiced value
            }
        })
        .collect();

    Harmonicity::new(
        times,
        Array1::from_vec(hnr_values),
        pitch.time_step(),
        min_pitch,
    )
}

/// Compute harmonicity using cross-correlation method.
///
/// This function computes HNR directly from cross-correlation for all frames,
/// not just pitch-voiced frames.
///
/// # Arguments
///
/// * `sound` - Sound object
/// * `time_step` - Time step in seconds
/// * `min_pitch` - Minimum pitch in Hz
/// * `silence_threshold` - Silence threshold (0-1)
/// * `periods_per_window` - Number of periods per window (Praat default: 1.0)
///
/// # Returns
///
/// Harmonicity object
pub fn sound_to_harmonicity_cc(
    sound: &Sound,
    time_step: f64,
    min_pitch: f64,
    silence_threshold: f64,
    _periods_per_window: f64,
) -> Harmonicity {
    let samples = sound.samples();
    let sample_rate = sound.sample_rate();
    let duration = sound.duration();

    let max_pitch = 600.0;

    // Frame timing matches Pitch CC (2-period window)
    let window_duration = 2.0 / min_pitch;
    let mut window_samples = (window_duration * sample_rate).round() as usize;
    if window_samples % 2 == 0 {
        window_samples += 1;
    }
    let half_window = window_samples / 2;

    let min_lag = (sample_rate / max_pitch).ceil() as usize;
    let max_lag = (sample_rate / min_pitch).floor() as usize;

    // Centered frame timing (same as Pitch CC)
    let n_frames = ((duration - window_duration) / time_step + 1e-9).floor() as usize + 1;
    let n_frames = n_frames.max(1);
    let t1 = (duration - (n_frames - 1) as f64 * time_step) / 2.0;

    let global_peak = samples.iter().map(|&s| s.abs()).fold(0.0f64, f64::max);

    let n_samples = samples.len();
    let samples_slice = samples.as_slice().unwrap();

    let mut times = Vec::with_capacity(n_frames);
    let mut hnr_values = Vec::with_capacity(n_frames);

    for i in 0..n_frames {
        let t = t1 + i as f64 * time_step;
        times.push(t);

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

        // Check for silence
        let local_peak = frame.iter().map(|&s| s.abs()).fold(0.0f64, f64::max);
        let local_intensity = local_peak / (global_peak + 1e-30);

        if local_intensity < silence_threshold * 0.01 {
            hnr_values.push(-200.0);
            continue;
        }

        // Compute full-frame cross-correlation and find peaks
        let n = frame.len();
        let mut r_array = vec![0.0; max_lag + 1];

        for lag in min_lag..=max_lag.min(n - 1) {
            let x1 = &frame[..n - lag];
            let x2 = &frame[lag..];

            let corr: f64 = x1.iter().zip(x2.iter()).map(|(&a, &b)| a * b).sum();
            let e1: f64 = x1.iter().map(|&x| x * x).sum();
            let e2: f64 = x2.iter().map(|&x| x * x).sum();

            if e1 > 0.0 && e2 > 0.0 {
                r_array[lag] = corr / (e1 * e2).sqrt();
            }
        }

        // Find best peak with parabolic interpolation
        let mut best_r = 0.0;
        for lag in (min_lag + 1)..max_lag.min(r_array.len() - 1) {
            if r_array[lag] > r_array[lag - 1] && r_array[lag] > r_array[lag + 1] {
                // Parabolic interpolation for refined strength
                let r_prev = r_array[lag - 1];
                let r_curr = r_array[lag];
                let r_next = r_array[lag + 1];

                let denom = r_prev - 2.0 * r_curr + r_next;
                if denom.abs() > 1e-10 {
                    let delta = 0.5 * (r_prev - r_next) / denom;
                    if delta.abs() < 1.0 {
                        let refined_r = r_curr - 0.25 * (r_prev - r_next) * delta;
                        if refined_r > best_r {
                            best_r = refined_r;
                        }
                    } else if r_curr > best_r {
                        best_r = r_curr;
                    }
                } else if r_curr > best_r {
                    best_r = r_curr;
                }
            }
        }

        let hnr = if best_r > 0.0 {
            strength_to_hnr(best_r)
        } else {
            -200.0
        };

        hnr_values.push(hnr);
    }

    Harmonicity::new(
        Array1::from_vec(times),
        Array1::from_vec(hnr_values),
        time_step,
        min_pitch,
    )
}
