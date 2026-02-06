//! Harmonicity - Harmonics-to-noise ratio (HNR) contour.
//!
//! This module computes the harmonicity (harmonics-to-noise ratio) of a signal,
//! which measures how much of the signal's energy is in periodic (harmonic)
//! components versus noise.
//!
//! # Critical Architecture Note
//!
//! **Harmonicity is NOT a standalone algorithm.** It internally uses the Pitch
//! module to compute autocorrelation or cross-correlation, then applies the
//! HNR formula to the correlation strength values.
//!
//! # Documentation Sources
//!
//! - Praat manual: Harmonicity
//!   "if 99% of the energy of the signal is in the periodic part, and 1% is noise,
//!    the HNR is 10*log10(99/1) = 20 dB"
//!
//! - Praat manual: Sound: To Harmonicity (ac)...
//!   "The algorithm performs an acoustic periodicity detection on the basis of
//!    an accurate autocorrelation method, as described in Boersma (1993)."
//!
//! # HNR Formula
//!
//! From the Praat manual (Harmonicity.html):
//!
//! ```text
//! HNR (dB) = 10 × log₁₀(r / (1 - r))
//! ```
//!
//! where r is the normalized autocorrelation (pitch strength) at the pitch period.
//!
//! # Interpretation
//!
//! The formula derives from the relationship between correlation and signal content:
//! - r represents the proportion of energy in periodic (harmonic) components
//! - (1-r) represents the proportion of energy in aperiodic (noise) components
//! - HNR is the ratio of harmonic to noise energy in decibels
//!
//! # Typical Values
//!
//! | r value | HNR (dB) | Interpretation |
//! |---------|----------|----------------|
//! | 0.5     | 0 dB     | Equal harmonic and noise energy |
//! | 0.9     | ~9.5 dB  | Moderate periodicity |
//! | 0.99    | 20 dB    | High periodicity (clean vowel) |
//! | 0.999   | 30 dB    | Very high periodicity |

use ndarray::Array1;

use crate::pitch::{sound_to_pitch_internal, FrameTiming, PitchMethod};
use crate::sound::Sound;

/// Interpolation method for harmonicity values.
///
/// Different interpolation methods offer trade-offs between smoothness
/// and computational cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarmonicityInterpolation {
    /// Nearest neighbor interpolation.
    ///
    /// Returns the value of the nearest frame. Fast but produces
    /// step-like output when querying between frames.
    Nearest,

    /// Linear interpolation.
    ///
    /// Interpolates linearly between two adjacent frames.
    Linear,

    /// Cubic (Catmull-Rom) interpolation.
    ///
    /// Uses four surrounding frames for smooth interpolation.
    Cubic,
}

/// Harmonicity (HNR) contour.
///
/// Values are in dB. Higher values indicate more periodic (harmonic) signal.
///
/// # Typical Values for Speech
///
/// - **Vowels**: 10-20 dB (highly periodic)
/// - **Voiced consonants**: 5-15 dB
/// - **Unvoiced/noise**: -200 dB (marker value for undefined)
#[derive(Debug, Clone)]
pub struct Harmonicity {
    /// Time points in seconds.
    ///
    /// Each time corresponds to the center of an analysis frame.
    times: Array1<f64>,

    /// HNR values in dB.
    ///
    /// -200 dB is used as a marker for unvoiced or undefined frames.
    values: Array1<f64>,

    /// Time step between frames.
    time_step: f64,

    /// Minimum pitch used for analysis.
    ///
    /// This determines the maximum pitch period to search, which affects
    /// the analysis window size.
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
    /// HNR in dB, or None if outside range.
    ///
    /// Note: Unvoiced frames return -200 dB (not None), which propagates
    /// through interpolation. This matches Praat's behavior.
    pub fn get_value_at_time(
        &self,
        time: f64,
        interpolation: HarmonicityInterpolation,
    ) -> Option<f64> {
        if self.n_frames() == 0 {
            return None;
        }

        // Compute floating-point index into frame array
        let t0 = self.times[0];
        let idx_float = (time - t0) / self.time_step;

        // Check bounds with half-frame tolerance at edges
        if idx_float < -0.5 || idx_float > self.n_frames() as f64 - 0.5 {
            return None;
        }

        match interpolation {
            HarmonicityInterpolation::Nearest => {
                // Round to nearest frame
                let idx = idx_float.round() as usize;
                let idx = idx.min(self.n_frames() - 1);
                Some(self.values[idx])
            }
            HarmonicityInterpolation::Linear => {
                // Linear interpolation between adjacent frames
                let idx = idx_float.floor() as isize;

                // Handle edge cases
                if idx < 0 {
                    return Some(self.values[0]);
                }
                let idx = idx as usize;
                if idx >= self.n_frames() - 1 {
                    return Some(self.values[self.n_frames() - 1]);
                }

                let frac = idx_float - idx as f64;

                // Linear interpolation: v = v0 × (1-t) + v1 × t
                Some(self.values[idx] * (1.0 - frac) + self.values[idx + 1] * frac)
            }
            HarmonicityInterpolation::Cubic => {
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

/// Convert pitch strength (correlation) to HNR in dB.
///
/// # Formula (from Praat manual, Harmonicity.html)
///
/// ```text
/// HNR = 10 × log₁₀(r / (1 - r))
/// ```
///
/// # Physical Interpretation
///
/// - r is the normalized autocorrelation at the pitch period
/// - r represents the fraction of energy in periodic components
/// - (1-r) represents the fraction of energy in noise
/// - The ratio r/(1-r) is the harmonic-to-noise power ratio
/// - Converting to dB: 10 × log₁₀ of the power ratio
///
/// # Arguments
///
/// * `r` - Pitch strength (normalized autocorrelation, typically 0-1)
///
/// # Returns
///
/// HNR in dB
///
/// # Examples
///
/// | r value | HNR (dB) | Meaning |
/// |---------|----------|---------|
/// | 0.5     | 0 dB     | Equal harmonic and noise energy |
/// | 0.9     | 9.5 dB   | 90% harmonic, 10% noise |
/// | 0.99    | 20 dB    | 99% harmonic, 1% noise |
/// | 0.999   | 30 dB    | 99.9% harmonic, 0.1% noise |
///
/// # Clamping
///
/// The autocorrelation normalization can produce values slightly > 1 or < 0
/// due to parabolic interpolation and numerical approximations. Values are
/// clamped to [1e-10, 1-1e-10] to avoid log(0) or log(infinity).
#[inline]
pub fn strength_to_hnr(r: f64) -> f64 {
    // Clamp r to valid range [epsilon, 1-epsilon]
    // This prevents:
    // - log(0) when r = 0
    // - log(infinity) when r = 1
    // - log(negative) when r > 1 (can happen with interpolation)
    let r = r.clamp(1e-10, 1.0 - 1e-10);

    // Apply the HNR formula: 10 × log₁₀(r / (1-r))
    10.0 * (r / (1.0 - r)).log10()
}

/// Compute harmonicity using autocorrelation method.
///
/// This function internally:
/// 1. Computes Pitch using the AC method with specific settings for HNR
/// 2. Extracts the correlation strength values from each frame
/// 3. Applies the HNR formula to convert strengths to dB
///
/// # Differences from Pitch AC
///
/// - **periods_per_window**: Uses caller-specified value (typically 4.5 vs 3.0)
/// - **frame_timing**: Uses Left alignment (vs Centered for Pitch)
/// - **octave_cost**: Disabled to get raw correlation strength
///
/// The longer window (4.5 periods vs 3) provides more stable HNR estimates
/// at the cost of reduced temporal resolution.
///
/// # Arguments
///
/// * `sound` - Sound object to analyze
/// * `time_step` - Time step in seconds
/// * `min_pitch` - Minimum pitch in Hz (determines analysis window size)
/// * `silence_threshold` - Silence threshold (0-1). Frames below this
///   amplitude ratio are more likely to be marked unvoiced.
/// * `periods_per_window` - Number of pitch periods per analysis window.
///   Praat default: 4.5 for AC method. Longer = smoother but less precise.
///
/// # Returns
///
/// Harmonicity object with HNR values in dB.
pub fn sound_to_harmonicity_ac(
    sound: &Sound,
    time_step: f64,
    min_pitch: f64,
    silence_threshold: f64,
    periods_per_window: f64,
) -> Harmonicity {
    // Step 1: Compute pitch using AC method with harmonicity-specific settings
    //
    // Key differences from standard Pitch AC:
    // - periods_per_window: user-specified (typically 4.5 for smoother HNR)
    // - frame_timing: Left (not Centered)
    // - apply_octave_cost: false (we want raw correlation for HNR formula)
    let pitch = sound_to_pitch_internal(
        sound,
        time_step,
        min_pitch,
        600.0,             // Standard pitch ceiling
        PitchMethod::Ac,
        0.45,              // voicing_threshold (Boersma default)
        silence_threshold, // User-specified silence threshold
        0.01,              // octave_cost
        0.35,              // octave_jump_cost
        0.14,              // voiced_unvoiced_cost
        periods_per_window,
        FrameTiming::Left, // Left-aligned for harmonicity
        false,             // apply_octave_cost = false for raw correlation
        false,             // apply_intensity_adjustment = false for raw correlation
    );

    // Step 2: Extract times and convert correlation strengths to HNR
    let times = pitch.times();
    let hnr_values: Vec<f64> = pitch
        .frames()
        .iter()
        .map(|frame| {
            if frame.voiced() {
                // Voiced frame: convert correlation strength to HNR
                strength_to_hnr(frame.strength())
            } else {
                // Unvoiced frame: use marker value
                // -200 dB is effectively "undefined" (silence or noise)
                -200.0
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
/// Uses the same approach as AC: compute Pitch CC internally with full
/// Viterbi voicing decisions, then extract the correlation strength from
/// voiced frames and apply the HNR formula.
///
/// Key differences from standard Pitch CC:
/// - `periods_per_window`: user-specified + 1 (for forward CC, DP29)
/// - `frame_timing`: Centered
/// - `apply_octave_cost`: false (raw correlation for HNR)
/// - `apply_intensity_adjustment`: false (raw correlation for HNR)
///
/// # Arguments
///
/// * `sound` - Sound object to analyze
/// * `time_step` - Time step in seconds
/// * `min_pitch` - Minimum pitch in Hz
/// * `silence_threshold` - Silence threshold (0-1)
/// * `periods_per_window` - Number of periods per window (Praat default: 1.0)
///
/// # Returns
///
/// Harmonicity object with HNR values in dB.
pub fn sound_to_harmonicity_cc(
    sound: &Sound,
    time_step: f64,
    min_pitch: f64,
    silence_threshold: f64,
    periods_per_window: f64,
) -> Harmonicity {
    // Step 1: Compute pitch using CC method with harmonicity-specific settings
    //
    // CC window = (ppw + 1) / min_pitch for forward cross-correlation (DP29)
    // Centered frame timing (matching Pitch CC behavior)
    // Octave cost and intensity adjustment disabled for raw correlation
    let pitch = sound_to_pitch_internal(
        sound,
        time_step,
        min_pitch,
        600.0,                     // Standard pitch ceiling
        PitchMethod::Cc,
        0.45,                      // voicing_threshold (Boersma default)
        silence_threshold,         // User-specified silence threshold
        0.01,                      // octave_cost
        0.35,                      // octave_jump_cost
        0.14,                      // voiced_unvoiced_cost
        periods_per_window + 1.0,  // +1 for forward CC (DP29)
        FrameTiming::Centered,     // Centered for CC harmonicity
        false,                     // apply_octave_cost = false for raw correlation
        false,                     // apply_intensity_adjustment = false for raw correlation
    );

    // Step 2: Extract times and convert correlation strengths to HNR
    let times = pitch.times();
    let hnr_values: Vec<f64> = pitch
        .frames()
        .iter()
        .map(|frame| {
            if frame.voiced() {
                // Voiced frame: convert correlation strength to HNR
                let r = frame.strength();
                if r > 0.0 && r < 1.0 {
                    strength_to_hnr(r)
                } else {
                    -200.0
                }
            } else {
                // Unvoiced frame: use marker value
                -200.0
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
