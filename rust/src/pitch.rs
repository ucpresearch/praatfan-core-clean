//! Pitch - Fundamental frequency (F0) contour.
//!
//! Documentation sources:
//! - Boersma (1993): "Accurate short-term analysis of the fundamental frequency
//!   and the harmonics-to-noise ratio of a sampled sound"
//! - Praat manual: Sound: To Pitch...
//!
//! Key documented facts (from Boersma 1993):
//! - Autocorrelation normalization: r_x(τ) ≈ r_a(τ) / r_w(τ) (Eq. 9)
//! - Sinc interpolation formula (Eq. 22)
//! - Candidate strength formulas (Eq. 23, 24)
//! - Viterbi transition costs (Eq. 27)
//! - Gaussian window formula (postscript)

use ndarray::Array1;

use crate::sound::Sound;

/// A pitch candidate for a frame.
#[derive(Debug, Clone)]
pub struct PitchCandidate {
    /// Frequency in Hz (0 = unvoiced).
    pub frequency: f64,
    /// Correlation strength (0-1).
    pub strength: f64,
}

impl PitchCandidate {
    /// Create a new pitch candidate.
    pub fn new(frequency: f64, strength: f64) -> Self {
        Self { frequency, strength }
    }
}

/// Pitch analysis results for a single frame.
#[derive(Debug, Clone)]
pub struct PitchFrame {
    /// Time in seconds.
    pub time: f64,
    /// Candidates (first is selected).
    pub candidates: Vec<PitchCandidate>,
    /// Local intensity (0-1).
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

    /// Selected pitch frequency (0 if unvoiced).
    #[inline]
    pub fn frequency(&self) -> f64 {
        if !self.candidates.is_empty() {
            self.candidates[0].frequency
        } else {
            0.0
        }
    }

    /// Selected pitch strength.
    #[inline]
    pub fn strength(&self) -> f64 {
        if !self.candidates.is_empty() {
            self.candidates[0].strength
        } else {
            0.0
        }
    }

    /// Whether this frame is voiced.
    #[inline]
    pub fn voiced(&self) -> bool {
        self.frequency() > 0.0
    }
}

/// Frame timing mode for pitch analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameTiming {
    /// Centered: frames centered in signal (used for Pitch).
    Centered,
    /// Left: left-aligned with centering constraint (used for Harmonicity).
    Left,
}

/// Pitch (F0) contour.
#[derive(Debug, Clone)]
pub struct Pitch {
    /// List of pitch frames.
    frames: Vec<PitchFrame>,
    /// Time step between frames.
    time_step: f64,
    /// Minimum pitch in Hz.
    pitch_floor: f64,
    /// Maximum pitch in Hz.
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
    /// Pitch value in Hz, or None if unvoiced or outside range
    pub fn get_value_at_time(&self, time: f64, interpolation: &str) -> Option<f64> {
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
                let frame = &self.frames[idx];
                if !frame.voiced() {
                    return None;
                }
                Some(frame.frequency())
            }
            "linear" => {
                let idx = idx_float.floor() as isize;
                let frac = idx_float - idx as f64;

                let i1 = 0.max(idx).min(self.n_frames() as isize - 1) as usize;
                let i2 = 0.max(idx + 1).min(self.n_frames() as isize - 1) as usize;

                let f1 = &self.frames[i1];
                let f2 = &self.frames[i2];

                // Both frames must be voiced for interpolation
                if !f1.voiced() || !f2.voiced() {
                    // Fall back to nearest voiced
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
                    Some(f1.frequency() * (1.0 - frac) + f2.frequency() * frac)
                }
            }
            _ => None,
        }
    }

    /// Get pitch strength at a specific time.
    ///
    /// This is the correlation value used to compute HNR.
    ///
    /// # Arguments
    ///
    /// * `time` - Time in seconds
    /// * `interpolation` - Interpolation method ("linear" or "nearest")
    ///
    /// # Returns
    ///
    /// Strength value (0-1), or None if outside range
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

/// Compute autocorrelation of a window function numerically.
fn compute_window_autocorrelation(window: &[f64], max_lag: usize) -> Vec<f64> {
    let n = window.len();
    let mut r = vec![0.0; max_lag + 1];

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
fn compute_autocorrelation(samples: &[f64], max_lag: usize) -> Vec<f64> {
    let n = samples.len();
    let mut r = vec![0.0; max_lag + 1];

    for lag in 0..=max_lag {
        if lag >= n {
            break;
        }
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
/// For each lag τ, computes the normalized correlation between:
/// - samples[0 : n-τ]  (signal from start to n-τ)
/// - samples[τ : n]    (signal shifted by τ samples)
///
/// This is normalized by the geometric mean of the energies:
///     r(τ) = Σ(x[i] × x[i+τ]) / sqrt(Σx[0:n-τ]² × Σx[τ:n]²)
fn compute_cross_correlation(samples: &[f64], min_lag: usize, max_lag: usize) -> Vec<f64> {
    let n = samples.len();
    let mut r = vec![0.0; max_lag + 1];

    for lag in min_lag..=max_lag.min(n - 1) {
        let x1 = &samples[..n - lag];
        let x2 = &samples[lag..];

        let corr: f64 = x1.iter().zip(x2.iter()).map(|(&a, &b)| a * b).sum();
        let e1: f64 = x1.iter().map(|&x| x * x).sum();
        let e2: f64 = x2.iter().map(|&x| x * x).sum();

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
fn find_cc_peaks(
    r: &[f64],
    min_lag: usize,
    max_lag: usize,
    sample_rate: f64,
    max_candidates: usize,
) -> Vec<(f64, f64)> {
    let mut candidates = Vec::new();

    // Find all peaks
    for lag in min_lag..max_lag.min(r.len() - 1) {
        if r[lag] > r[lag - 1] && r[lag] > r[lag + 1] {
            let r_curr = r[lag];

            // Parabolic interpolation for sub-sample precision
            let r_prev = r[lag - 1];
            let r_next = r[lag + 1];

            let denom = r_prev - 2.0 * r_curr + r_next;
            if denom.abs() > 1e-10 {
                let delta = 0.5 * (r_prev - r_next) / denom;
                if delta.abs() < 1.0 {
                    let refined_lag = lag as f64 + delta;
                    // Use raw peak strength (not interpolated) to avoid overshoot
                    let freq = sample_rate / refined_lag;
                    candidates.push((freq, r_curr));
                } else {
                    candidates.push((sample_rate / lag as f64, r_curr));
                }
            } else {
                candidates.push((sample_rate / lag as f64, r_curr));
            }
        }
    }

    // Sort by strength and return top candidates
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(max_candidates);
    candidates
}

/// Find all autocorrelation peaks in the given lag range.
fn find_autocorrelation_peaks(
    r: &[f64],
    r_w: &[f64],
    min_lag: usize,
    max_lag: usize,
    sample_rate: f64,
    max_candidates: usize,
) -> Vec<(f64, f64)> {
    if max_lag >= r.len() || max_lag >= r_w.len() {
        return Vec::new();
    }

    let r_0 = r[0];
    if r_0 <= 0.0 {
        return Vec::new();
    }

    // Compute normalized autocorrelation array
    let mut r_norm = vec![0.0; max_lag + 1];
    for lag in 0..=max_lag {
        if r_w[lag] > 0.0 && r_w[0] > 0.0 {
            r_norm[lag] = (r[lag] / r_0) / (r_w[lag] / r_w[0]);
        }
    }

    let mut candidates = Vec::new();

    // Find all peaks in normalized autocorrelation
    for lag in min_lag..max_lag.min(r_norm.len() - 1) {
        if r_norm[lag] > r_norm[lag - 1] && r_norm[lag] > r_norm[lag + 1] {
            let r_curr = r_norm[lag];

            // Parabolic interpolation for sub-sample precision on frequency only
            let r_prev = r_norm[lag - 1];
            let r_next = r_norm[lag + 1];

            let denom = r_prev - 2.0 * r_curr + r_next;
            if denom.abs() > 1e-10 {
                let delta = 0.5 * (r_prev - r_next) / denom;
                if delta.abs() < 1.0 {
                    let refined_lag = lag as f64 + delta;
                    let freq = sample_rate / refined_lag;
                    // Use raw peak strength to avoid overshoot
                    candidates.push((freq, r_curr));
                } else {
                    candidates.push((sample_rate / lag as f64, r_curr));
                }
            } else {
                candidates.push((sample_rate / lag as f64, r_curr));
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
/// From Boersma (1993) Eq. 27, the transition cost is:
/// - 0 if both unvoiced
/// - voiced_unvoiced_cost if voicing changes
/// - octave_jump_cost × |log₂(F1/F2)| if both voiced
///
/// The costs are corrected for time step: multiply by 0.01 / time_step
///
/// This modifies the frames in place, reordering candidates.
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

    // Time correction factor
    let time_correction = 0.01 / time_step;

    // Get candidate counts
    let n_cands: Vec<usize> = frames.iter().map(|f| f.candidates.len()).collect();

    // Dynamic programming
    // best_cost[i][j] = best total cost to reach candidate j at frame i
    // best_prev[i][j] = previous candidate index that led to this best cost

    // Initialize arrays
    let mut best_cost: Vec<Vec<f64>> = n_cands
        .iter()
        .map(|&n| vec![f64::INFINITY; n])
        .collect();
    let mut best_prev: Vec<Vec<usize>> = n_cands.iter().map(|&n| vec![0; n]).collect();

    // First frame: cost is negative of strength (we minimize cost, maximize strength)
    for (j, cand) in frames[0].candidates.iter().enumerate() {
        best_cost[0][j] = -cand.strength;
    }

    // Forward pass
    for i in 1..n_frames {
        for j in 0..n_cands[i] {
            let cand_j = &frames[i].candidates[j];
            for k in 0..n_cands[i - 1] {
                let cand_k = &frames[i - 1].candidates[k];

                // Transition cost
                let f_k = cand_k.frequency;
                let f_j = cand_j.frequency;

                let trans_cost = if f_k == 0.0 && f_j == 0.0 {
                    // Both unvoiced
                    0.0
                } else if f_k == 0.0 || f_j == 0.0 {
                    // Voicing change
                    voiced_unvoiced_cost
                } else {
                    // Both voiced - octave jump cost
                    octave_jump_cost * (f_j / f_k).log2().abs()
                };

                let trans_cost = trans_cost * time_correction;

                // Total cost to reach candidate j at frame i via candidate k at frame i-1
                let total_cost = best_cost[i - 1][k] + trans_cost - cand_j.strength;

                if total_cost < best_cost[i][j] {
                    best_cost[i][j] = total_cost;
                    best_prev[i][j] = k;
                }
            }
        }
    }

    // Backward pass to find best path
    let mut path = vec![0usize; n_frames];
    path[n_frames - 1] = best_cost[n_frames - 1]
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    for i in (0..n_frames - 1).rev() {
        path[i] = best_prev[i + 1][path[i + 1]];
    }

    // Reorder candidates in each frame so best path candidate is first
    for i in 0..n_frames {
        let best_idx = path[i];
        if best_idx > 0 {
            frames[i].candidates.swap(0, best_idx);
        }
    }
}

/// Compute pitch from sound using autocorrelation method.
///
/// Based on Boersma (1993): "Accurate short-term analysis of the fundamental
/// frequency and the harmonics-to-noise ratio of a sampled sound."
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
/// Pitch object
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
        0.45,  // voicing_threshold
        0.03,  // silence_threshold
        0.01,  // octave_cost
        0.35,  // octave_jump_cost
        0.14,  // voiced_unvoiced_cost
        3.0,   // periods_per_window
        FrameTiming::Centered,
        true,  // apply_octave_cost
    )
}

/// Compute pitch from sound using cross-correlation method.
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
/// Pitch object
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
        2.0,   // periods_per_window (CC uses 2)
        FrameTiming::Centered,
        true,  // apply_octave_cost
    )
}

/// Internal pitch computation with full parameter control.
///
/// This is also used by harmonicity to compute pitch with specific settings.
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
    let window_duration = periods_per_window / pitch_floor;

    // Lag range for pitch search
    let min_lag = (sample_rate / pitch_ceiling).ceil() as usize;
    let max_lag = (sample_rate / pitch_floor).floor() as usize;

    // Number of samples in window
    let mut window_samples = (window_duration * sample_rate).round() as usize;
    if window_samples % 2 == 0 {
        window_samples += 1;
    }
    let half_window_samples = window_samples / 2;

    // For AC method: generate window and compute its autocorrelation
    // For CC method: no windowing needed
    let (window, r_w) = match method {
        PitchMethod::Ac => {
            let w = hanning_window(window_samples);
            let rw = compute_window_autocorrelation(&w, max_lag);
            (Some(w), Some(rw))
        }
        PitchMethod::Cc => (None, None),
    };

    // Frame timing
    let (n_frames, t1) = match frame_timing {
        FrameTiming::Left => {
            // Left-aligned with centering: used for Harmonicity
            let n = ((duration - 2.0 * window_duration) / time_step + 1e-9).floor() as usize + 1;
            let n = n.max(1);
            let remaining = duration - 2.0 * window_duration - (n - 1) as f64 * time_step;
            let t1 = window_duration + remaining / 2.0;
            (n, t1)
        }
        FrameTiming::Centered => {
            // Centered: frames centered in signal, used for Pitch
            let n = ((duration - window_duration) / time_step + 1e-9).floor() as usize + 1;
            let n = n.max(1);
            let t1 = (duration - (n - 1) as f64 * time_step) / 2.0;
            (n, t1)
        }
    };

    // Compute global peak for silence detection
    let global_peak = samples.iter().map(|&s| s.abs()).fold(0.0f64, f64::max);

    // Process each frame
    let mut frames = Vec::with_capacity(n_frames);
    let n_samples = samples.len();

    for i in 0..n_frames {
        let t = t1 + i as f64 * time_step;

        // Extract frame samples
        let center_sample = (t * sample_rate).round() as isize;
        let start_sample = center_sample - half_window_samples as isize;
        let end_sample = start_sample + window_samples as isize;

        // Handle boundaries
        let mut frame_samples = vec![0.0; window_samples];
        if start_sample < 0 || end_sample > n_samples as isize {
            let src_start = 0.max(start_sample) as usize;
            let src_end = (n_samples as isize).min(end_sample) as usize;
            let dst_start = (src_start as isize - start_sample) as usize;
            let dst_end = dst_start + (src_end - src_start);
            frame_samples[dst_start..dst_end]
                .copy_from_slice(&samples.as_slice().unwrap()[src_start..src_end]);
        } else {
            let start = start_sample as usize;
            let end = end_sample as usize;
            frame_samples.copy_from_slice(&samples.as_slice().unwrap()[start..end]);
        }

        // Compute local peak (for silence detection)
        let local_peak = frame_samples.iter().map(|&s| s.abs()).fold(0.0f64, f64::max);
        let local_intensity = local_peak / (global_peak + 1e-30);

        // Compute correlation and find peaks based on method
        let peaks = match method {
            PitchMethod::Ac => {
                // AC: Apply window and compute autocorrelation with normalization
                let window = window.as_ref().unwrap();
                let r_w = r_w.as_ref().unwrap();
                let windowed: Vec<f64> = frame_samples
                    .iter()
                    .zip(window.iter())
                    .map(|(&s, &w)| s * w)
                    .collect();
                let r = compute_autocorrelation(&windowed, max_lag);
                find_autocorrelation_peaks(&r, r_w, min_lag, max_lag, sample_rate, 15)
            }
            PitchMethod::Cc => {
                // CC: Full-frame cross-correlation on raw samples
                let r = compute_cross_correlation(&frame_samples, min_lag, max_lag);
                find_cc_peaks(&r, min_lag, max_lag, sample_rate, 15)
            }
        };

        // Create candidates list
        let mut candidates = Vec::new();

        // Unvoiced candidate
        // From Boersma (1993) Eq. 23
        let unvoiced_strength = voicing_threshold
            + (2.0 - local_intensity / silence_threshold).max(0.0) * (1.0 + voicing_threshold);
        candidates.push(PitchCandidate::new(0.0, unvoiced_strength));

        // Voiced candidates
        for (freq, strength) in peaks {
            if freq > 0.0 && strength > 0.0 {
                let adjusted_strength = if apply_octave_cost {
                    // Apply octave cost (Eq. 24) for pitch tracking
                    strength - octave_cost * (pitch_floor / freq + 1e-30).log2()
                } else {
                    // Use raw strength for harmonicity computation
                    strength
                };
                candidates.push(PitchCandidate::new(freq, adjusted_strength));
            }
        }

        // Sort by strength (highest first)
        candidates.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        frames.push(PitchFrame::new(t, candidates, local_intensity));
    }

    // Apply Viterbi path finding to resolve octave errors
    viterbi_path(&mut frames, time_step, octave_jump_cost, voiced_unvoiced_cost);

    Pitch::new(frames, time_step, pitch_floor, pitch_ceiling)
}

/// Pitch method (AC or CC).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PitchMethod {
    /// Autocorrelation method.
    Ac,
    /// Cross-correlation method.
    Cc,
}
