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
    /// # Undefined frames
    ///
    /// A contour carries two markers that are not measurements —
    /// [`UNDEFINED_HNR`] (-200 dB, no periodicity) and `NaN` (correlation not
    /// resolvable from 1). Both are excluded from the interpolation stencil
    /// rather than mixed into it: a single `NaN` frame would otherwise poison
    /// every query within the stencil, and a -200 frame would drag its
    /// neighbours 200 dB downward. If every frame in the stencil is a marker,
    /// that marker is returned verbatim; `Nearest` always returns the raw
    /// frame value.
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

                // Linear interpolation: v = v0 × (1-t) + v1 × t, unless a
                // frame carries no measurement.
                Some(self.linear_skipping_markers(idx, idx + 1, frac))
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

                if [y0, y1, y2, y3].iter().any(|&v| is_marker(v)) {
                    // Catmull-Rom weights are negative at the outer taps, so
                    // dropping frames and renormalising the survivors lets the
                    // remaining weights cancel to nearly zero and the quotient
                    // explode. Degrade to the two central frames instead.
                    return Some(self.linear_skipping_markers(i1, i2, frac));
                }

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

    /// Blend two frames linearly, unless either carries no measurement.
    ///
    /// Interpolating across a marker would let it leak into a real reading, so
    /// when one is present the nearest frame is returned verbatim instead.
    fn linear_skipping_markers(&self, i_lo: usize, i_hi: usize, frac: f64) -> f64 {
        let v_lo = self.values[i_lo];
        let v_hi = self.values[i_hi];
        if !is_marker(v_lo) && !is_marker(v_hi) {
            return v_lo * (1.0 - frac) + v_hi * frac;
        }
        // A marker is present, so there is nothing to interpolate along. Fall
        // back to the frame the query actually sits nearest: that returns the
        // real reading when the query leans towards it, and returns the marker
        // when it leans towards the marker — rather than silently borrowing a
        // neighbour's value for a frame that holds no measurement.
        let nearest = if frac < 0.5 { i_lo } else { i_hi };
        self.values[nearest]
    }
}

/// Value reported for frames in which no periodicity was found.
///
/// Determined by black-box observation: Praat reports exactly -200 dB for
/// pure silence and for a constant (DC) signal.
pub const UNDEFINED_HNR: f64 = -200.0;

/// Smallest `periods_per_window` for which the AC estimator is well posed.
///
/// The AC method normalises the signal autocorrelation by the analysis
/// window's own autocorrelation, `r_norm(tau) = (r_x(tau)/r_x(0)) /
/// (r_w(tau)/r_w(0))`, and searches lags up to `sample_rate/pitch_floor`. The
/// window is `periods_per_window/pitch_floor` long, so the longest lag
/// searched sits at `1/periods_per_window` of the window length. As that
/// ratio approaches 1 the denominator `r_w(tau)` decays to zero and the
/// quotient is dominated by division by a vanishing number.
///
/// Measured on a Hanning window at 16 kHz, floor 60 Hz — `r_w(tau_max) /
/// r_w(0)` is 0.472 at 3.0 periods per window, 0.167 at 2.0, and -4.2e-17
/// (zero to within rounding, and of arbitrary sign) at 1.0. The maximum
/// correlation observed on conversational speech follows directly: 1.13 at
/// 3.0 periods per window, 1.76 at 2.0, and 35.9 at 1.0 — the last two being
/// impossible for a normalised correlation.
///
/// Praat imposes the same lower bound, observable from outside as a
/// user-visible refusal ("Number of periods per window must be at least
/// 3.0"); note that parselmouth's `Sound.to_harmonicity_ac()` wrapper
/// defaults to 1.0 and bypasses that check.
pub const AC_MIN_PERIODS_PER_WINDOW: f64 = 3.0;

/// Whether a contour entry is a marker rather than a measurement.
#[inline]
fn is_marker(value: f64) -> bool {
    value.is_nan() || value == UNDEFINED_HNR
}

/// Whether `periods_per_window` puts the AC estimator outside its valid
/// range. Callers that can surface a diagnostic (the Python bindings, the
/// JSON pipe) warn on `true`; this will become a hard error in a future
/// release.
#[inline]
pub fn ac_periods_per_window_is_ill_conditioned(periods_per_window: f64) -> bool {
    periods_per_window < AC_MIN_PERIODS_PER_WINDOW
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
/// * `r` - Normalized autocorrelation at the pitch period. Values outside
///   `0 < r < 1` are not tolerated but represented — see the boundary policy.
///
/// # Returns
///
/// HNR in dB for `0 < r < 1`; [`UNDEFINED_HNR`] (-200 dB) when there is no
/// periodicity to measure; `f64::NAN` when the correlation is not resolvable
/// from 1.
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
/// # Boundary policy (DP10)
///
/// `r` is a normalized correlation, so a meaningful HNR exists only for
/// `0 < r < 1`. The two boundaries mean opposite things and are reported
/// differently:
///
/// * `r <= 0` — no periodic component was found. Reported as
///   [`UNDEFINED_HNR`] (-200 dB), which is also what Praat reports for
///   silence and for a DC signal (black-box observation).
/// * `1 - r <= f64::EPSILON` — the correlation is not resolvable from 1 in
///   double precision, so `10*log10(r/(1-r))` is determined entirely by
///   rounding. This includes `r > 1`, which no true correlation can have; it
///   arises from estimator noise on (near-)perfectly periodic material and
///   grows without bound once `periods_per_window` drops below
///   [`AC_MIN_PERIODS_PER_WINDOW`]. Reported as `f64::NAN`: no HNR can be
///   derived from a correlation that carries no information.
///
/// The test is `1 - r <= f64::EPSILON` rather than `r >= 1.0` so that this
/// and the pure-Python backend agree. Under the strict test the last
/// representable double below 1 still returns a finite 159.5459 dB, and
/// `1 - EPSILON` (two ulps below 1) returns 156.5356 dB, while a one-ULP-larger
/// `r` returns NaN — the two backends' correlations differ at exactly that
/// scale, so the strict test made them disagree on 59 frames of a 30 s excerpt.
///
/// Earlier releases clamped `r` to `[1e-10, 1-1e-10]` instead, which
/// collapsed every such frame onto the single value 99.99999964022885 dB and
/// made an arithmetic boundary look like a measurement of near-perfect
/// periodicity.
#[inline]
pub fn strength_to_hnr(r: f64) -> f64 {
    if !(r > 0.0) {
        // Also catches NaN.
        return UNDEFINED_HNR;
    }
    if 1.0 - r <= f64::EPSILON {
        return f64::NAN;
    }

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
    harmonicity_ac_internal(
        sound,
        time_step,
        min_pitch,
        silence_threshold,
        periods_per_window,
        None, // reference_peak: legacy whole-file statistic
    )
}

/// Compute HNR (AC method) against a speech-scoped amplitude reference.
///
/// Like [`sound_to_harmonicity_ac`], but the whole-file amplitude
/// statistic is replaced by `reference_peak` (`None` = estimate
/// internally via [`crate::speech_reference::estimate_speech_reference`]).
/// The original entry point is unchanged.
///
/// # Errors
///
/// `Error::InvalidParameter` if an explicit `reference_peak` is not
/// finite and > 0.
pub fn sound_to_harmonicity_ac_referenced(
    sound: &Sound,
    time_step: f64,
    min_pitch: f64,
    silence_threshold: f64,
    periods_per_window: f64,
    reference_peak: Option<f64>,
) -> crate::error::Result<Harmonicity> {
    let resolved = crate::speech_reference::resolve_reference_peak(
        sound.samples().as_slice().unwrap(),
        sound.sample_rate(),
        reference_peak,
    )?;
    Ok(harmonicity_ac_internal(
        sound,
        time_step,
        min_pitch,
        silence_threshold,
        periods_per_window,
        resolved,
    ))
}

fn harmonicity_ac_internal(
    sound: &Sound,
    time_step: f64,
    min_pitch: f64,
    silence_threshold: f64,
    periods_per_window: f64,
    reference_peak: Option<f64>,
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
        reference_peak,
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
                // Unvoiced frame: no periodicity found (DP11)
                UNDEFINED_HNR
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
    harmonicity_cc_internal(
        sound,
        time_step,
        min_pitch,
        silence_threshold,
        periods_per_window,
        None, // reference_peak: legacy whole-file statistic
    )
}

/// Compute HNR (CC method) against a speech-scoped amplitude reference.
///
/// See [`sound_to_harmonicity_ac_referenced`] for semantics.
pub fn sound_to_harmonicity_cc_referenced(
    sound: &Sound,
    time_step: f64,
    min_pitch: f64,
    silence_threshold: f64,
    periods_per_window: f64,
    reference_peak: Option<f64>,
) -> crate::error::Result<Harmonicity> {
    let resolved = crate::speech_reference::resolve_reference_peak(
        sound.samples().as_slice().unwrap(),
        sound.sample_rate(),
        reference_peak,
    )?;
    Ok(harmonicity_cc_internal(
        sound,
        time_step,
        min_pitch,
        silence_threshold,
        periods_per_window,
        resolved,
    ))
}

fn harmonicity_cc_internal(
    sound: &Sound,
    time_step: f64,
    min_pitch: f64,
    silence_threshold: f64,
    periods_per_window: f64,
    reference_peak: Option<f64>,
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
        reference_peak,
    );

    // Step 2: Extract times and convert correlation strengths to HNR
    let times = pitch.times();
    let hnr_values: Vec<f64> = pitch
        .frames()
        .iter()
        .map(|frame| {
            if frame.voiced() {
                // strength_to_hnr applies the shared boundary policy:
                // r <= 0 -> UNDEFINED_HNR, r >= 1 -> NaN.
                strength_to_hnr(frame.strength())
            } else {
                // Unvoiced frame: no periodicity found (DP11)
                UNDEFINED_HNR
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

#[cfg(test)]
mod boundary_tests {
    use super::*;

    /// The value the old `r.clamp(1e-10, 1.0 - 1e-10)` collapsed near-perfect
    /// correlations onto. See BUG-hnr-ac-saturation.md.
    const LEGACY_CLIP_CONSTANT: f64 = 99.99999964022885;

    #[test]
    fn valid_correlations_use_the_documented_formula() {
        assert!((strength_to_hnr(0.5) - 0.0).abs() < 1e-12);
        assert!((strength_to_hnr(0.9) - 9.542_425_094_393_249).abs() < 1e-12);
        assert!((strength_to_hnr(0.99) - 19.956_351_945_975_5).abs() < 1e-9);
    }

    #[test]
    fn no_periodicity_reports_undefined() {
        for r in [0.0, -0.0, -1e-12, -0.5, f64::NAN] {
            assert_eq!(strength_to_hnr(r), UNDEFINED_HNR, "r = {r}");
        }
    }

    #[test]
    fn invalid_correlations_report_nan() {
        for r in [1.0, 1.002, 1.13, 15.0, 35.9] {
            assert!(strength_to_hnr(r).is_nan(), "r = {r}");
        }
    }

    #[test]
    fn boundary_is_ulp_robust() {
        // The last representable double below 1 must not produce a finite
        // reading, or this backend and the pure-Python one disagree by
        // ~156 dB on frames whose correlations differ only in the final bit.
        assert!(strength_to_hnr(1.0 - f64::EPSILON).is_nan());
        assert!(strength_to_hnr(f64::from_bits(0x3FEF_FFFF_FFFF_FFFF)).is_nan());
    }

    #[test]
    fn the_legacy_clip_constant_is_no_longer_a_plateau() {
        // The old clamp mapped *every* r >= 1-1e-10 onto one value. The
        // number itself is still the correct reading for r == 1-1e-10
        // exactly; what must not survive is the plateau above it.
        assert!(
            (strength_to_hnr(1.0 - 1e-10) - LEGACY_CLIP_CONSTANT).abs() < 1e-6,
            "r = 1-1e-10 should still map to its own correct value"
        );
        for r in [1.0 - 1e-11, 1.0 - 1e-12, 1.0 - 1e-13, 1.0 - 1e-14] {
            let v = strength_to_hnr(r);
            assert!(
                v.is_nan() || v > LEGACY_CLIP_CONSTANT + 1.0,
                "r = {r} collapsed back onto the clamp: {v}"
            );
        }
    }

    #[test]
    fn finite_readings_are_strictly_increasing_in_r() {
        let rs = [0.5, 0.9, 0.99, 0.999, 1.0 - 1e-6, 1.0 - 1e-12];
        let mut prev = f64::NEG_INFINITY;
        for r in rs {
            let v = strength_to_hnr(r);
            assert!(v.is_finite(), "r = {r}");
            assert!(v > prev, "not increasing at r = {r}");
            prev = v;
        }
    }

    #[test]
    fn ill_conditioned_bound_matches_the_documented_value() {
        assert_eq!(AC_MIN_PERIODS_PER_WINDOW, 3.0);
        assert!(ac_periods_per_window_is_ill_conditioned(1.0));
        assert!(ac_periods_per_window_is_ill_conditioned(2.999));
        assert!(!ac_periods_per_window_is_ill_conditioned(3.0));
        assert!(!ac_periods_per_window_is_ill_conditioned(4.5));
    }
}

#[cfg(test)]
mod interpolation_tests {
    use super::*;
    use ndarray::Array1;

    fn contour(values: &[f64]) -> Harmonicity {
        let times: Vec<f64> = (0..values.len()).map(|i| i as f64 * 0.01).collect();
        Harmonicity::new(Array1::from_vec(times), Array1::from_vec(values.to_vec()), 0.01, 75.0)
    }

    /// A marker frame must not leak into its neighbours' readings. Before this
    /// was fixed a single NaN poisoned every query in the stencil, and the two
    /// backends disagreed on how far the damage spread.
    #[test]
    fn markers_do_not_poison_neighbours() {
        let h = contour(&[16.57, 16.53, 14.61, f64::NAN, f64::NAN, 10.97, 11.10, 9.45]);
        for interp in [
            HarmonicityInterpolation::Cubic,
            HarmonicityInterpolation::Linear,
        ] {
            assert!((h.get_value_at_time(0.02, interp).unwrap() - 14.61).abs() < 1e-9);
            assert!((h.get_value_at_time(0.05, interp).unwrap() - 10.97).abs() < 1e-9);
        }

        let h = contour(&[16.57, 16.53, 14.61, UNDEFINED_HNR, 10.97, 11.10]);
        for interp in [
            HarmonicityInterpolation::Cubic,
            HarmonicityInterpolation::Linear,
        ] {
            assert!((h.get_value_at_time(0.02, interp).unwrap() - 14.61).abs() < 1e-9);
            assert!((h.get_value_at_time(0.04, interp).unwrap() - 10.97).abs() < 1e-9);
        }
    }

    #[test]
    fn querying_a_marker_frame_returns_the_marker() {
        let h = contour(&[16.57, 16.53, f64::NAN, 10.97, 11.10]);
        assert!(h
            .get_value_at_time(0.02, HarmonicityInterpolation::Cubic)
            .unwrap()
            .is_nan());
        let h = contour(&[16.57, 16.53, UNDEFINED_HNR, 10.97, 11.10]);
        assert_eq!(
            h.get_value_at_time(0.02, HarmonicityInterpolation::Cubic)
                .unwrap(),
            UNDEFINED_HNR
        );
    }

    #[test]
    fn all_marker_contour_returns_the_marker() {
        let h = contour(&[f64::NAN, f64::NAN, f64::NAN, f64::NAN]);
        assert!(h
            .get_value_at_time(0.01, HarmonicityInterpolation::Cubic)
            .unwrap()
            .is_nan());
    }

    /// Catmull-Rom weights sum to 1, so a marker-free stencil must reproduce
    /// the direct form exactly.
    #[test]
    fn defined_contour_is_unchanged_by_the_exclusion() {
        let values = [3.0, 7.0, 11.0, 5.0, 9.0, 2.0];
        let h = contour(&values);
        let n = values.len() as isize;
        let mut t: f64 = 0.0;
        while t < 0.05 {
            let idx_float = t / 0.01;
            let idx = idx_float.floor() as isize;
            let frac = idx_float - idx as f64;
            let y0 = values[0.max(idx - 1) as usize];
            let y1 = values[0.max(idx).min(n - 1) as usize];
            let y2 = values[0.max(idx + 1).min(n - 1) as usize];
            let y3 = values[(idx + 2).min(n - 1) as usize];
            let (t2, t3) = (frac * frac, frac * frac * frac);
            let expected = 0.5
                * ((2.0 * y1)
                    + (-y0 + y2) * frac
                    + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t2
                    + (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t3);
            let got = h
                .get_value_at_time(t, HarmonicityInterpolation::Cubic)
                .unwrap();
            assert!((got - expected).abs() < 1e-9, "t={t}: {got} vs {expected}");
            t += 0.003;
        }
    }
}
