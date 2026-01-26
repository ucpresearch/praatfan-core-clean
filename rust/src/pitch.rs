//! Pitch - Fundamental frequency (F0) contour.
//!
//! This module computes the pitch (fundamental frequency) of voiced speech
//! using autocorrelation (AC) or cross-correlation (CC) methods.
//!
//! # Documentation Sources
//!
//! - **Primary**: Boersma (1993): "Accurate short-term analysis of the fundamental
//!   frequency and the harmonics-to-noise ratio of a sampled sound"
//!   (https://www.fon.hum.uva.nl/paul/papers/Proceedings_1993.pdf)
//! - Praat manual: Sound: To Pitch...
//!
//! # Key Documented Facts (from Boersma 1993)
//!
//! - **Autocorrelation normalization**: r_x(τ) ≈ r_a(τ) / r_w(τ) (Eq. 9)
//!   The signal autocorrelation is normalized by the window autocorrelation
//!   to correct for the window's effect on the correlation values.
//!
//! - **Sinc interpolation**: For sub-sample precision (Eq. 22)
//!
//! - **Candidate strength formulas**:
//!   - Unvoiced strength (Eq. 23): voicing_threshold + bonus for weak signals
//!   - Octave cost (Eq. 24): penalizes selection of subharmonics
//!
//! - **Viterbi transition costs** (Eq. 27):
//!   - 0 for unvoiced-to-unvoiced transitions
//!   - voiced_unvoiced_cost for voicing changes
//!   - octave_jump_cost × |log₂(F1/F2)| for pitch jumps
//!
//! - **Gaussian window formula**: Documented in the paper's postscript
//!
//! # Algorithm Overview
//!
//! 1. **Frame Analysis**: For each analysis frame:
//!    - Extract windowed samples
//!    - Compute autocorrelation or cross-correlation
//!    - Find peaks corresponding to pitch period candidates
//!    - Apply octave cost to penalize low-frequency candidates
//!
//! 2. **Viterbi Path Finding**: Find optimal path through candidates
//!    across all frames by minimizing transition costs + maximizing strength.

use ndarray::Array1;

use crate::sound::Sound;

/// A pitch candidate for a frame.
///
/// Each frame typically has multiple candidates:
/// - One unvoiced candidate (frequency = 0)
/// - Multiple voiced candidates at different pitch periods
///
/// The Viterbi algorithm selects the optimal candidate for each frame
/// by considering both local strength and inter-frame transition costs.
#[derive(Debug, Clone)]
pub struct PitchCandidate {
    /// Frequency in Hz.
    ///
    /// 0 indicates the unvoiced candidate.
    /// For voiced candidates, this is computed from the autocorrelation lag:
    /// frequency = sample_rate / lag
    pub frequency: f64,

    /// Correlation strength (typically 0-1 for voiced, higher for unvoiced).
    ///
    /// For AC method: normalized autocorrelation r(τ) / (r_w(τ) × r_w(0))
    /// For CC method: normalized cross-correlation
    ///
    /// The unvoiced candidate's strength includes the voicing threshold
    /// plus a bonus for weak signals (Boersma 1993, Eq. 23).
    pub strength: f64,
}

impl PitchCandidate {
    /// Create a new pitch candidate.
    pub fn new(frequency: f64, strength: f64) -> Self {
        Self { frequency, strength }
    }
}

/// Pitch analysis results for a single frame.
///
/// Each frame contains:
/// - A time stamp
/// - A list of candidates (first is selected after Viterbi)
/// - Local intensity for silence detection
#[derive(Debug, Clone)]
pub struct PitchFrame {
    /// Time in seconds (center of analysis window).
    pub time: f64,

    /// Pitch candidates for this frame.
    ///
    /// After Viterbi path finding, candidates are reordered so the
    /// selected candidate is first. The selected candidate may be
    /// unvoiced (frequency = 0).
    pub candidates: Vec<PitchCandidate>,

    /// Local intensity (0-1).
    ///
    /// This is the ratio of local peak amplitude to global peak amplitude.
    /// Used for silence detection: weak frames get a bonus for the unvoiced
    /// candidate (Boersma 1993, Eq. 23).
    pub intensity: f64,
}

impl PitchFrame {
    /// Create a new pitch frame.
    pub fn new(time: f64, candidates: Vec<PitchCandidate>, intensity: f64) -> Self {
        Self {
            time,
            candidates,
            intensity,
        }
    }

    /// Get the selected pitch frequency (0 if unvoiced).
    ///
    /// After Viterbi, the selected candidate is always first in the list.
    #[inline]
    pub fn frequency(&self) -> f64 {
        if !self.candidates.is_empty() {
            self.candidates[0].frequency
        } else {
            0.0
        }
    }

    /// Get the selected pitch strength.
    ///
    /// This is the correlation value used for computing HNR.
    #[inline]
    pub fn strength(&self) -> f64 {
        if !self.candidates.is_empty() {
            self.candidates[0].strength
        } else {
            0.0
        }
    }

    /// Check whether this frame is voiced.
    ///
    /// A frame is voiced if its selected candidate has non-zero frequency.
    #[inline]
    pub fn voiced(&self) -> bool {
        self.frequency() > 0.0
    }
}

/// Frame timing mode for pitch analysis.
///
/// Different timing modes are used for different analysis types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameTiming {
    /// Centered: Frames are centered in the signal.
    ///
    /// Used for Pitch analysis. The first frame time is chosen so that
    /// frames are symmetrically distributed around the signal center.
    Centered,

    /// Left: Left-aligned with centering constraint.
    ///
    /// Used for Harmonicity analysis. Frames start at the left edge
    /// of the valid analysis region but are centered within that region.
    Left,
}

/// Pitch (F0) contour.
///
/// Contains pitch frames with candidate information plus analysis parameters.
#[derive(Debug, Clone)]
pub struct Pitch {
    /// List of pitch frames.
    ///
    /// Each frame contains candidates; the first candidate in each frame
    /// is the selected one (after Viterbi path finding).
    frames: Vec<PitchFrame>,

    /// Time step between frames.
    ///
    /// Default: 0.75 / pitch_floor (about 10 ms for 75 Hz floor)
    time_step: f64,

    /// Minimum pitch in Hz.
    ///
    /// Determines the maximum lag to search in autocorrelation.
    pitch_floor: f64,

    /// Maximum pitch in Hz.
    ///
    /// Determines the minimum lag to search in autocorrelation.
    pitch_ceiling: f64,
}

impl Pitch {
    /// Create a new Pitch object.
    pub fn new(
        frames: Vec<PitchFrame>,
        time_step: f64,
        pitch_floor: f64,
        pitch_ceiling: f64,
    ) -> Self {
        Self {
            frames,
            time_step,
            pitch_floor,
            pitch_ceiling,
        }
    }

    /// Get the pitch frames.
    #[inline]
    pub fn frames(&self) -> &[PitchFrame] {
        &self.frames
    }

    /// Get mutable reference to pitch frames (for Viterbi reordering).
    #[inline]
    pub fn frames_mut(&mut self) -> &mut [PitchFrame] {
        &mut self.frames
    }

    /// Get the number of frames.
    #[inline]
    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }

    /// Get the time step between frames.
    #[inline]
    pub fn time_step(&self) -> f64 {
        self.time_step
    }

    /// Get the minimum pitch in Hz.
    #[inline]
    pub fn pitch_floor(&self) -> f64 {
        self.pitch_floor
    }

    /// Get the maximum pitch in Hz.
    #[inline]
    pub fn pitch_ceiling(&self) -> f64 {
        self.pitch_ceiling
    }

    /// Get array of frame times.
    pub fn times(&self) -> Array1<f64> {
        Array1::from_iter(self.frames.iter().map(|f| f.time))
    }

    /// Get array of pitch values (0 for unvoiced).
    pub fn values(&self) -> Array1<f64> {
        Array1::from_iter(self.frames.iter().map(|f| f.frequency()))
    }

    /// Get array of pitch strengths.
    pub fn strengths(&self) -> Array1<f64> {
        Array1::from_iter(self.frames.iter().map(|f| f.strength()))
    }

    /// Get pitch value at a specific time.
    ///
    /// # Arguments
    ///
    /// * `time` - Time in seconds
    /// * `interpolation` - Interpolation method ("linear" or "nearest")
    ///
    /// # Returns
    ///
    /// Pitch value in Hz, or None if unvoiced or outside range.
    pub fn get_value_at_time(&self, time: f64, interpolation: &str) -> Option<f64> {
        if self.n_frames() == 0 {
            return None;
        }

        // Compute floating-point index
        let t0 = self.frames[0].time;
        let idx_float = (time - t0) / self.time_step;

        // Check bounds
        if idx_float < -0.5 || idx_float > self.n_frames() as f64 - 0.5 {
            return None;
        }

        match interpolation {
            "nearest" => {
                // Round to nearest frame
                let idx = idx_float.round() as usize;
                let idx = idx.min(self.n_frames() - 1);
                let frame = &self.frames[idx];

                // Return None for unvoiced frames
                if !frame.voiced() {
                    return None;
                }
                Some(frame.frequency())
            }
            "linear" => {
                // Linear interpolation between adjacent frames
                let idx = idx_float.floor() as isize;
                let frac = idx_float - idx as f64;

                // Get adjacent frame indices with clamping
                let i1 = 0.max(idx).min(self.n_frames() as isize - 1) as usize;
                let i2 = 0.max(idx + 1).min(self.n_frames() as isize - 1) as usize;

                let f1 = &self.frames[i1];
                let f2 = &self.frames[i2];

                // Both frames must be voiced for interpolation
                if !f1.voiced() || !f2.voiced() {
                    // Fall back to nearest voiced frame
                    if frac < 0.5 && f1.voiced() {
                        Some(f1.frequency())
                    } else if f2.voiced() {
                        Some(f2.frequency())
                    } else if f1.voiced() {
                        Some(f1.frequency())
                    } else {
                        None
                    }
                } else {
                    // Linear interpolation: f = f1 × (1-t) + f2 × t
                    Some(f1.frequency() * (1.0 - frac) + f2.frequency() * frac)
                }
            }
            _ => None,
        }
    }

    /// Get pitch strength at a specific time.
    ///
    /// This is the correlation value used to compute HNR.
    /// Unlike pitch values, strengths are returned even for unvoiced frames.
    ///
    /// # Arguments
    ///
    /// * `time` - Time in seconds
    /// * `interpolation` - Interpolation method ("linear" or "nearest")
    ///
    /// # Returns
    ///
    /// Strength value (typically 0-1), or None if outside range.
    pub fn get_strength_at_time(&self, time: f64, interpolation: &str) -> Option<f64> {
        if self.n_frames() == 0 {
            return None;
        }

        let t0 = self.frames[0].time;
        let idx_float = (time - t0) / self.time_step;

        if idx_float < -0.5 || idx_float > self.n_frames() as f64 - 0.5 {
            return None;
        }

        match interpolation {
            "nearest" => {
                let idx = idx_float.round() as usize;
                let idx = idx.min(self.n_frames() - 1);
                Some(self.frames[idx].strength())
            }
            "linear" => {
                let idx = idx_float.floor() as isize;
                let frac = idx_float - idx as f64;

                let i1 = 0.max(idx).min(self.n_frames() as isize - 1) as usize;
                let i2 = 0.max(idx + 1).min(self.n_frames() as isize - 1) as usize;

                let s1 = self.frames[i1].strength();
                let s2 = self.frames[i2].strength();

                Some(s1 * (1.0 - frac) + s2 * frac)
            }
            _ => None,
        }
    }
}

/// Generate Hanning window.
///
/// The Hanning (or Hann) window is defined as:
/// ```text
/// w(n) = 0.5 - 0.5 × cos(2π × n / (N-1))
/// ```
///
/// This window has good frequency resolution and moderate sidelobe levels,
/// making it suitable for pitch analysis where we want to resolve the
/// fundamental frequency accurately.
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
            // Hanning formula: 0.5 - 0.5 × cos(2πi/(N-1))
            0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
        })
        .collect()
}

/// Compute autocorrelation of a window function numerically.
///
/// This is needed for the AC method's normalization (Boersma 1993, Eq. 9).
/// The window autocorrelation r_w(τ) corrects for the reduced overlap
/// between the window and itself at larger lags.
///
/// # Formula
///
/// ```text
/// r_w(τ) = Σᵢ w(i) × w(i + τ)
/// ```
///
/// # Arguments
///
/// * `window` - Window coefficients
/// * `max_lag` - Maximum lag to compute
///
/// # Returns
///
/// Vector of autocorrelation values r_w[0..max_lag]
fn compute_window_autocorrelation(window: &[f64], max_lag: usize) -> Vec<f64> {
    let n = window.len();
    let mut r = vec![0.0; max_lag + 1];

    // For each lag, sum product of overlapping window values
    for lag in 0..=max_lag.min(n - 1) {
        r[lag] = window[..n - lag]
            .iter()
            .zip(window[lag..].iter())
            .map(|(&a, &b)| a * b)
            .sum();
    }

    r
}

/// Compute autocorrelation for lags 0 to max_lag.
///
/// The autocorrelation measures self-similarity at different time shifts.
/// Peaks in the autocorrelation correspond to periodic components in the signal.
///
/// # Formula
///
/// ```text
/// r(τ) = Σᵢ x(i) × x(i + τ)
/// ```
///
/// For pitch detection, we look for the first significant peak after lag 0,
/// which corresponds to the pitch period.
///
/// # Arguments
///
/// * `samples` - Input signal samples
/// * `max_lag` - Maximum lag to compute
///
/// # Returns
///
/// Vector of autocorrelation values r[0..max_lag]
fn compute_autocorrelation(samples: &[f64], max_lag: usize) -> Vec<f64> {
    let n = samples.len();
    let mut r = vec![0.0; max_lag + 1];

    for lag in 0..=max_lag {
        if lag >= n {
            break;
        }
        // Sum product of samples at positions i and i+lag
        r[lag] = samples[..n - lag]
            .iter()
            .zip(samples[lag..].iter())
            .map(|(&a, &b)| a * b)
            .sum();
    }

    r
}

/// Compute full-frame cross-correlation for the CC pitch method.
///
/// The cross-correlation method computes the normalized correlation between
/// the signal and a time-shifted version of itself, normalized by the
/// geometric mean of the energies in both segments.
///
/// # Formula
///
/// ```text
/// r(τ) = Σ(x[i] × x[i+τ]) / sqrt(Σx[0:n-τ]² × Σx[τ:n]²)
/// ```
///
/// This normalization makes the CC method more robust to amplitude variations
/// within the analysis window compared to the AC method.
///
/// # Arguments
///
/// * `samples` - Input signal samples
/// * `min_lag` - Minimum lag (corresponding to pitch_ceiling)
/// * `max_lag` - Maximum lag (corresponding to pitch_floor)
///
/// # Returns
///
/// Vector of correlation values (indices 0..min_lag will be 0)
fn compute_cross_correlation(samples: &[f64], min_lag: usize, max_lag: usize) -> Vec<f64> {
    let n = samples.len();
    let mut r = vec![0.0; max_lag + 1];

    for lag in min_lag..=max_lag.min(n - 1) {
        // Split signal into two overlapping parts
        let x1 = &samples[..n - lag];  // First n-lag samples
        let x2 = &samples[lag..];      // Last n-lag samples

        // Compute correlation (unnormalized)
        let corr: f64 = x1.iter().zip(x2.iter()).map(|(&a, &b)| a * b).sum();

        // Compute energies of both segments
        let e1: f64 = x1.iter().map(|&x| x * x).sum();
        let e2: f64 = x2.iter().map(|&x| x * x).sum();

        // Normalize by geometric mean of energies
        if e1 > 0.0 && e2 > 0.0 {
            r[lag] = corr / (e1 * e2).sqrt();
        }
    }

    r
}

/// Find peaks in cross-correlation.
///
/// Unlike AC, CC values are already normalized (0-1) via energy normalization,
/// so no window autocorrelation normalization is needed.
///
/// # Peak Detection
///
/// A peak is a local maximum where r[lag] > r[lag-1] and r[lag] > r[lag+1].
/// Parabolic interpolation is used for sub-sample lag precision:
///
/// ```text
/// δ = 0.5 × (r[lag-1] - r[lag+1]) / (r[lag-1] - 2×r[lag] + r[lag+1])
/// refined_lag = lag + δ
/// ```
///
/// # Arguments
///
/// * `r` - Cross-correlation values
/// * `min_lag` - Minimum lag to search
/// * `max_lag` - Maximum lag to search
/// * `sample_rate` - Sample rate for converting lag to frequency
/// * `max_candidates` - Maximum number of candidates to return
///
/// # Returns
///
/// Vector of (frequency, strength) tuples, sorted by strength
fn find_cc_peaks(
    r: &[f64],
    min_lag: usize,
    max_lag: usize,
    sample_rate: f64,
    max_candidates: usize,
) -> Vec<(f64, f64)> {
    let mut candidates = Vec::new();

    // Find all local maxima
    for lag in min_lag..max_lag.min(r.len() - 1) {
        // Check for local maximum
        if r[lag] > r[lag - 1] && r[lag] > r[lag + 1] {
            let r_curr = r[lag];

            // Parabolic interpolation for sub-sample precision
            let r_prev = r[lag - 1];
            let r_next = r[lag + 1];

            let denom = r_prev - 2.0 * r_curr + r_next;
            if denom.abs() > 1e-10 {
                // Compute refined lag position
                let delta = 0.5 * (r_prev - r_next) / denom;
                if delta.abs() < 1.0 {
                    let refined_lag = lag as f64 + delta;
                    // Convert lag to frequency: f = sample_rate / lag
                    let freq = sample_rate / refined_lag;
                    // Use raw peak strength (not interpolated) to avoid overshoot
                    candidates.push((freq, r_curr));
                } else {
                    // Delta too large, use unrefined lag
                    candidates.push((sample_rate / lag as f64, r_curr));
                }
            } else {
                // Degenerate case, use unrefined lag
                candidates.push((sample_rate / lag as f64, r_curr));
            }
        }
    }

    // Sort by strength (highest first) and truncate
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(max_candidates);
    candidates
}

/// Find all autocorrelation peaks in the given lag range.
///
/// For the AC method, the autocorrelation is normalized by the window
/// autocorrelation (Boersma 1993, Eq. 9):
///
/// ```text
/// r_normalized(τ) = (r(τ) / r(0)) / (r_w(τ) / r_w(0))
/// ```
///
/// This normalization corrects for the reduced overlap at larger lags.
///
/// # Arguments
///
/// * `r` - Signal autocorrelation
/// * `r_w` - Window autocorrelation (for normalization)
/// * `min_lag` - Minimum lag to search
/// * `max_lag` - Maximum lag to search
/// * `sample_rate` - Sample rate for converting lag to frequency
/// * `max_candidates` - Maximum number of candidates to return
/// * `local_intensity` - Frame intensity relative to global peak (0-1)
///
/// # Note on strength adjustment
///
/// The returned strength is adjusted by local_intensity to handle an edge case
/// where low-intensity frames (silence, DC offset) can have spuriously high
/// r_norm values (~1.0) that cause false voicing detections.
///
/// The adjustment formula is: strength = 0.5 * r_norm + 0.5 * local_intensity
///
/// This is an empirically-derived approximation found via black-box testing.
/// The actual formula used by Praat is unknown and could be different (possibly
/// a nonlinear function like GLM/GAM). The coefficients (0.5, 0.5) were tuned
/// on limited test data and may not be optimal for all audio types.
fn find_autocorrelation_peaks(
    r: &[f64],
    r_w: &[f64],
    min_lag: usize,
    max_lag: usize,
    sample_rate: f64,
    max_candidates: usize,
    local_intensity: f64,
) -> Vec<(f64, f64)> {
    // Validate inputs
    if max_lag >= r.len() || max_lag >= r_w.len() {
        return Vec::new();
    }

    // r(0) is the total energy - needed for normalization
    let r_0 = r[0];
    if r_0 <= 0.0 {
        return Vec::new();
    }

    // Compute normalized autocorrelation array
    // r_norm[τ] = (r[τ]/r[0]) / (r_w[τ]/r_w[0])
    let mut r_norm = vec![0.0; max_lag + 1];
    for lag in 0..=max_lag {
        if r_w[lag] > 0.0 && r_w[0] > 0.0 {
            // Boersma 1993, Eq. 9
            r_norm[lag] = (r[lag] / r_0) / (r_w[lag] / r_w[0]);
        }
    }

    let mut candidates = Vec::new();

    // Find all peaks in normalized autocorrelation
    for lag in min_lag..max_lag.min(r_norm.len() - 1) {
        // Check for local maximum
        if r_norm[lag] > r_norm[lag - 1] && r_norm[lag] > r_norm[lag + 1] {
            let r_curr = r_norm[lag];

            // Apply intensity-periodicity interaction to adjust strength.
            // This handles the case where low-intensity frames have spuriously
            // high r_norm values (e.g., DC offset gives r_norm ≈ 1.0).
            let adjusted_strength = 0.5 * r_curr + 0.5 * local_intensity;

            // Parabolic interpolation for sub-sample precision on frequency only
            let r_prev = r_norm[lag - 1];
            let r_next = r_norm[lag + 1];

            let denom = r_prev - 2.0 * r_curr + r_next;
            if denom.abs() > 1e-10 {
                let delta = 0.5 * (r_prev - r_next) / denom;
                if delta.abs() < 1.0 {
                    let refined_lag = lag as f64 + delta;
                    let freq = sample_rate / refined_lag;
                    // Use adjusted strength (intensity-weighted)
                    candidates.push((freq, adjusted_strength));
                } else {
                    candidates.push((sample_rate / lag as f64, adjusted_strength));
                }
            } else {
                candidates.push((sample_rate / lag as f64, adjusted_strength));
            }
        }
    }

    // Sort by strength and return top candidates
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(max_candidates);
    candidates
}

/// Apply Viterbi algorithm to find optimal path through candidates.
///
/// The Viterbi algorithm finds the globally optimal sequence of pitch
/// candidates by minimizing a cost function that combines:
/// - Local costs (negative of candidate strength)
/// - Transition costs (penalize pitch jumps and voicing changes)
///
/// # Cost Function (from Boersma 1993, Eq. 27)
///
/// Transition cost between frames i-1 and i:
/// - **Both unvoiced**: 0
/// - **Voicing change**: voiced_unvoiced_cost
/// - **Both voiced**: octave_jump_cost × |log₂(F₁/F₂)|
///
/// Costs are scaled by time_step to make them independent of frame rate:
/// ```text
/// scaled_cost = cost × (0.01 / time_step)
/// ```
///
/// # Dynamic Programming
///
/// 1. **Forward pass**: For each frame, compute minimum total cost to
///    reach each candidate, tracking the best predecessor.
///
/// 2. **Backward pass**: Starting from the best final candidate, trace
///    back through best predecessors to find optimal path.
///
/// 3. **Reordering**: Reorder candidates in each frame so the selected
///    candidate is first.
///
/// # Arguments
///
/// * `frames` - Mutable slice of pitch frames (candidates will be reordered)
/// * `time_step` - Time step between frames (for cost scaling)
/// * `octave_jump_cost` - Cost per octave of frequency change
/// * `voiced_unvoiced_cost` - Cost for voicing transitions
fn viterbi_path(
    frames: &mut [PitchFrame],
    time_step: f64,
    octave_jump_cost: f64,
    voiced_unvoiced_cost: f64,
) {
    let n_frames = frames.len();
    if n_frames <= 1 {
        return;
    }

    // Time correction factor (Boersma 1993)
    // Costs are defined per 10 ms; scale to actual time step
    let time_correction = 0.01 / time_step;

    // Get candidate counts per frame
    let n_cands: Vec<usize> = frames.iter().map(|f| f.candidates.len()).collect();

    // Dynamic programming tables:
    // best_cost[i][j] = minimum total cost to reach candidate j at frame i
    // best_prev[i][j] = predecessor candidate index that achieved this cost
    let mut best_cost: Vec<Vec<f64>> = n_cands
        .iter()
        .map(|&n| vec![f64::INFINITY; n])
        .collect();
    let mut best_prev: Vec<Vec<usize>> = n_cands.iter().map(|&n| vec![0; n]).collect();

    // Initialize first frame: cost = negative strength (we minimize cost)
    for (j, cand) in frames[0].candidates.iter().enumerate() {
        best_cost[0][j] = -cand.strength;
    }

    // Forward pass: compute minimum cost to reach each candidate
    for i in 1..n_frames {
        for j in 0..n_cands[i] {
            let cand_j = &frames[i].candidates[j];

            // Consider all possible predecessors
            for k in 0..n_cands[i - 1] {
                let cand_k = &frames[i - 1].candidates[k];

                // Compute transition cost based on voicing states
                let f_k = cand_k.frequency;
                let f_j = cand_j.frequency;

                let trans_cost = if f_k == 0.0 && f_j == 0.0 {
                    // Both unvoiced: no cost
                    0.0
                } else if f_k == 0.0 || f_j == 0.0 {
                    // Voicing change: fixed penalty
                    voiced_unvoiced_cost
                } else {
                    // Both voiced: penalize octave jumps
                    // Cost proportional to |log₂(F_j / F_k)|
                    octave_jump_cost * (f_j / f_k).log2().abs()
                };

                // Scale transition cost by time correction
                let trans_cost = trans_cost * time_correction;

                // Total cost = previous cost + transition + local (negative strength)
                let total_cost = best_cost[i - 1][k] + trans_cost - cand_j.strength;

                // Update if this is the best path to candidate j
                if total_cost < best_cost[i][j] {
                    best_cost[i][j] = total_cost;
                    best_prev[i][j] = k;
                }
            }
        }
    }

    // Backward pass: find best path
    // Start from the candidate with minimum cost at the last frame
    let mut path = vec![0usize; n_frames];
    path[n_frames - 1] = best_cost[n_frames - 1]
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Trace back through best predecessors
    for i in (0..n_frames - 1).rev() {
        path[i] = best_prev[i + 1][path[i + 1]];
    }

    // Reorder candidates so best path candidate is first
    for i in 0..n_frames {
        let best_idx = path[i];
        if best_idx > 0 {
            // Swap selected candidate to position 0
            frames[i].candidates.swap(0, best_idx);
        }
    }
}

/// Compute pitch from sound using autocorrelation method.
///
/// The AC method is the primary pitch detection algorithm described in
/// Boersma (1993). It provides accurate pitch tracking by:
/// 1. Windowing the signal with a Hanning window
/// 2. Computing autocorrelation with window normalization
/// 3. Finding peaks corresponding to pitch period candidates
/// 4. Using Viterbi to find the optimal path through candidates
///
/// # Arguments
///
/// * `sound` - Sound object
/// * `time_step` - Time step in seconds (0 = auto: 0.75/floor)
/// * `pitch_floor` - Minimum pitch in Hz
/// * `pitch_ceiling` - Maximum pitch in Hz
///
/// # Returns
///
/// Pitch object with frames containing selected pitch values
pub fn sound_to_pitch_ac(
    sound: &Sound,
    time_step: f64,
    pitch_floor: f64,
    pitch_ceiling: f64,
) -> Pitch {
    sound_to_pitch_internal(
        sound,
        time_step,
        pitch_floor,
        pitch_ceiling,
        PitchMethod::Ac,
        0.45,  // voicing_threshold (Boersma default)
        0.03,  // silence_threshold
        0.01,  // octave_cost (favors higher frequencies slightly)
        0.35,  // octave_jump_cost
        0.14,  // voiced_unvoiced_cost
        3.0,   // periods_per_window (3 for AC)
        FrameTiming::Centered,
        true,  // apply_octave_cost
    )
}

/// Compute pitch from sound using cross-correlation method.
///
/// The CC method is an alternative to AC that normalizes by energy,
/// making it more robust to amplitude variations within the analysis window.
///
/// # Arguments
///
/// * `sound` - Sound object
/// * `time_step` - Time step in seconds (0 = auto: 0.75/floor)
/// * `pitch_floor` - Minimum pitch in Hz
/// * `pitch_ceiling` - Maximum pitch in Hz
pub fn sound_to_pitch_cc(
    sound: &Sound,
    time_step: f64,
    pitch_floor: f64,
    pitch_ceiling: f64,
) -> Pitch {
    sound_to_pitch_internal(
        sound,
        time_step,
        pitch_floor,
        pitch_ceiling,
        PitchMethod::Cc,
        0.45,  // voicing_threshold
        0.03,  // silence_threshold
        0.01,  // octave_cost
        0.35,  // octave_jump_cost
        0.14,  // voiced_unvoiced_cost
        2.0,   // periods_per_window (2 for CC, shorter than AC)
        FrameTiming::Centered,
        true,  // apply_octave_cost
    )
}

/// Internal pitch computation with full parameter control.
///
/// This function is also used by the harmonicity module to compute pitch
/// with specific settings (e.g., different periods_per_window, no octave cost).
///
/// # Algorithm
///
/// For each frame:
/// 1. Extract samples centered at frame time
/// 2. For AC: apply window and compute autocorrelation with normalization
///    For CC: compute full-frame cross-correlation
/// 3. Find peaks in correlation function
/// 4. Create candidate list:
///    - Unvoiced candidate (Boersma 1993, Eq. 23)
///    - Voiced candidates with optional octave cost (Eq. 24)
///
/// After all frames are processed:
/// 5. Apply Viterbi algorithm to find optimal candidate sequence
#[allow(clippy::too_many_arguments)]
pub fn sound_to_pitch_internal(
    sound: &Sound,
    time_step: f64,
    pitch_floor: f64,
    pitch_ceiling: f64,
    method: PitchMethod,
    voicing_threshold: f64,
    silence_threshold: f64,
    octave_cost: f64,
    octave_jump_cost: f64,
    voiced_unvoiced_cost: f64,
    periods_per_window: f64,
    frame_timing: FrameTiming,
    apply_octave_cost: bool,
) -> Pitch {
    let samples = sound.samples();
    let sample_rate = sound.sample_rate();
    let duration = sound.duration();

    // Default time step (documented: 0.75 / floor)
    let time_step = if time_step <= 0.0 {
        0.75 / pitch_floor
    } else {
        time_step
    };

    // Window duration: periods_per_window periods of minimum pitch
    // AC uses 3 periods, CC uses 2 periods
    let window_duration = periods_per_window / pitch_floor;

    // Lag range for pitch search
    // min_lag corresponds to pitch_ceiling (highest frequency = shortest period)
    // max_lag corresponds to pitch_floor (lowest frequency = longest period)
    let min_lag = (sample_rate / pitch_ceiling).ceil() as usize;
    let max_lag = (sample_rate / pitch_floor).floor() as usize;

    // Number of samples in window
    let mut window_samples = (window_duration * sample_rate).round() as usize;
    if window_samples % 2 == 0 {
        window_samples += 1;  // Ensure odd for symmetric window
    }
    let half_window_samples = window_samples / 2;

    // For AC method: generate window and compute its autocorrelation
    // The window autocorrelation is used for normalization (Eq. 9)
    let (window, r_w) = match method {
        PitchMethod::Ac => {
            let w = hanning_window(window_samples);
            let rw = compute_window_autocorrelation(&w, max_lag);
            (Some(w), Some(rw))
        }
        PitchMethod::Cc => (None, None),
    };

    // Frame timing calculation
    let (n_frames, t1) = match frame_timing {
        FrameTiming::Left => {
            // Left-aligned with centering: used for Harmonicity
            // Frames start window_duration into signal and end window_duration before end
            let n = ((duration - 2.0 * window_duration) / time_step + 1e-9).floor() as usize + 1;
            let n = n.max(1);
            // Center frames in available region
            let remaining = duration - 2.0 * window_duration - (n - 1) as f64 * time_step;
            let t1 = window_duration + remaining / 2.0;
            (n, t1)
        }
        FrameTiming::Centered => {
            // Centered: frames centered in signal, used for Pitch
            // Need half window on each side, so valid region is [window_duration/2, duration - window_duration/2]
            let n = ((duration - window_duration) / time_step + 1e-9).floor() as usize + 1;
            let n = n.max(1);
            // Center frames symmetrically
            let t1 = (duration - (n - 1) as f64 * time_step) / 2.0;
            (n, t1)
        }
    };

    // Compute global peak for silence detection
    // This is used to determine local intensity relative to overall signal
    let global_peak = samples.iter().map(|&s| s.abs()).fold(0.0f64, f64::max);

    // Process each frame
    let mut frames = Vec::with_capacity(n_frames);
    let n_samples = samples.len();

    for i in 0..n_frames {
        let t = t1 + i as f64 * time_step;

        // Extract frame samples centered at time t
        let center_sample = (t * sample_rate).round() as isize;
        let start_sample = center_sample - half_window_samples as isize;
        let end_sample = start_sample + window_samples as isize;

        // Handle boundary conditions (zero-pad outside signal)
        let mut frame_samples = vec![0.0; window_samples];
        if start_sample < 0 || end_sample > n_samples as isize {
            // Partial overlap with signal boundaries
            let src_start = 0.max(start_sample) as usize;
            let src_end = (n_samples as isize).min(end_sample) as usize;
            let dst_start = (src_start as isize - start_sample) as usize;
            let dst_end = dst_start + (src_end - src_start);
            frame_samples[dst_start..dst_end]
                .copy_from_slice(&samples.as_slice().unwrap()[src_start..src_end]);
        } else {
            // Full overlap - copy all samples
            let start = start_sample as usize;
            let end = end_sample as usize;
            frame_samples.copy_from_slice(&samples.as_slice().unwrap()[start..end]);
        }

        // Compute local peak for silence detection
        let local_peak = frame_samples.iter().map(|&s| s.abs()).fold(0.0f64, f64::max);
        let local_intensity = local_peak / (global_peak + 1e-30);

        // Compute correlation and find peaks based on method
        let peaks = match method {
            PitchMethod::Ac => {
                // AC method: apply window and compute autocorrelation
                let window = window.as_ref().unwrap();
                let r_w = r_w.as_ref().unwrap();

                // Apply window to samples
                let windowed: Vec<f64> = frame_samples
                    .iter()
                    .zip(window.iter())
                    .map(|(&s, &w)| s * w)
                    .collect();

                // Compute autocorrelation
                let r = compute_autocorrelation(&windowed, max_lag);

                // Find peaks with window normalization
                find_autocorrelation_peaks(&r, r_w, min_lag, max_lag, sample_rate, 15, local_intensity)
            }
            PitchMethod::Cc => {
                // CC method: full-frame cross-correlation on raw samples
                let r = compute_cross_correlation(&frame_samples, min_lag, max_lag);
                find_cc_peaks(&r, min_lag, max_lag, sample_rate, 15)
            }
        };

        // Create candidates list
        let mut candidates = Vec::new();

        // Unvoiced candidate (Boersma 1993, Eq. 23)
        // strength = voicing_threshold + bonus for weak signals
        // Weak signals (low intensity) get higher unvoiced strength, making them
        // more likely to be classified as unvoiced
        let unvoiced_strength = voicing_threshold
            + (2.0 - local_intensity / silence_threshold).max(0.0) * (1.0 + voicing_threshold);
        candidates.push(PitchCandidate::new(0.0, unvoiced_strength));

        // Voiced candidates from correlation peaks
        for (freq, strength) in peaks {
            if freq > 0.0 && strength > 0.0 {
                // Optionally apply octave cost (Boersma 1993, Eq. 24)
                // This penalizes lower frequencies to prevent selection of subharmonics
                let adjusted_strength = if apply_octave_cost {
                    // Cost increases logarithmically for lower frequencies
                    strength - octave_cost * (pitch_floor / freq + 1e-30).log2()
                } else {
                    // Raw strength for harmonicity computation
                    strength
                };
                candidates.push(PitchCandidate::new(freq, adjusted_strength));
            }
        }

        // Sort by strength (highest first) for initial ordering
        candidates.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        frames.push(PitchFrame::new(t, candidates, local_intensity));
    }

    // Apply Viterbi path finding to resolve octave errors and voicing decisions
    viterbi_path(&mut frames, time_step, octave_jump_cost, voiced_unvoiced_cost);

    Pitch::new(frames, time_step, pitch_floor, pitch_ceiling)
}

/// Pitch method (AC or CC).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PitchMethod {
    /// Autocorrelation method.
    ///
    /// The primary method described in Boersma (1993). Uses windowed
    /// autocorrelation with normalization for accurate pitch detection.
    Ac,

    /// Cross-correlation method.
    ///
    /// Alternative method that normalizes by energy, making it more
    /// robust to amplitude variations within the analysis window.
    Cc,
}
