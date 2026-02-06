//! Formant analysis module - LPC-based formant frequency tracks.
//!
//! # Overview
//!
//! This module implements formant analysis using Burg's Linear Predictive Coding (LPC)
//! algorithm. Formants are resonant frequencies of the vocal tract, typically labeled
//! F1, F2, F3, etc. They are essential for vowel identification and speech analysis.
//!
//! # Algorithm
//!
//! The formant extraction algorithm consists of these steps:
//!
//! 1. **Resampling**: Downsample to 2× max_formant_hz (Nyquist criterion)
//! 2. **Pre-emphasis**: High-pass filter to compensate for lip radiation
//! 3. **Windowing**: Apply Gaussian window to each analysis frame
//! 4. **LPC analysis**: Compute LPC coefficients using Burg's algorithm
//! 5. **Root finding**: Find roots of LPC polynomial via companion matrix eigenvalues
//! 6. **Formant extraction**: Convert roots to frequencies and bandwidths
//!
//! # Documentation Sources
//!
//! - Praat manual: "Sound: To Formant (burg)..."
//! - Childers (1978): "Modern Spectrum Analysis", pp. 252-255 (Burg's algorithm)
//! - Numerical Recipes Ch. 9.5 (root finding via companion matrix)
//! - Markel & Gray (1976): root-to-formant conversion formulas
//!
//! # Key Parameters (from Praat documentation)
//!
//! - Window length parameter: "actual length is twice this value"
//! - Resample to 2 × max_formant_hz before analysis
//! - Pre-emphasis: x'[i] = x[i] - α × x[i-1], where α = exp(-2π × F × Δt)
//! - LPC order: 2 × max_formants (each formant needs 2 poles)
//! - Formant filtering: remove frequencies < 50 Hz and > (max_formant - 50) Hz

use ndarray::Array1;
use num_complex::Complex64;
use rustfft::{num_complex::Complex, FftPlanner};

use crate::sound::Sound;

// =============================================================================
// Data Structures
// =============================================================================

/// A single formant measurement at a point in time.
///
/// Each formant is characterized by:
/// - **Frequency**: The center frequency of the resonance (Hz)
/// - **Bandwidth**: The width of the resonance at -3dB (Hz)
///
/// Narrower bandwidths indicate sharper, more defined resonances.
#[derive(Debug, Clone)]
pub struct FormantPoint {
    /// Formant frequency in Hz.
    pub frequency: f64,
    /// Formant bandwidth in Hz (3dB bandwidth of the resonance peak).
    pub bandwidth: f64,
}

impl FormantPoint {
    /// Create a new FormantPoint with the given frequency and bandwidth.
    pub fn new(frequency: f64, bandwidth: f64) -> Self {
        Self {
            frequency,
            bandwidth,
        }
    }
}

/// Formant analysis results for a single time frame.
///
/// Contains all detected formants (F1, F2, F3, ...) at a specific time point.
/// The number of formants may vary between frames (some frames may have fewer
/// formants detected than requested).
#[derive(Debug, Clone)]
pub struct FormantFrame {
    /// Time at the center of this frame (seconds).
    pub time: f64,
    /// List of formants, ordered by frequency (F1, F2, F3, ...).
    pub formants: Vec<FormantPoint>,
}

impl FormantFrame {
    /// Create a new FormantFrame at the given time with the specified formants.
    pub fn new(time: f64, formants: Vec<FormantPoint>) -> Self {
        Self { time, formants }
    }

    /// Get the number of formants detected in this frame.
    #[inline]
    pub fn n_formants(&self) -> usize {
        self.formants.len()
    }

    /// Get a specific formant by number (1-based index: 1=F1, 2=F2, etc.).
    ///
    /// Returns `None` if the formant number is out of range.
    pub fn get_formant(&self, n: usize) -> Option<&FormantPoint> {
        if n >= 1 && n <= self.formants.len() {
            Some(&self.formants[n - 1])
        } else {
            None
        }
    }
}

/// Complete formant analysis results over time.
///
/// Contains formant tracks for an entire sound, with one frame per time step.
/// Provides methods to extract formant contours and query values at specific times.
#[derive(Debug, Clone)]
pub struct Formant {
    /// All analysis frames, ordered by time.
    frames: Vec<FormantFrame>,
    /// Time step between consecutive frames (seconds).
    time_step: f64,
    /// Maximum formant frequency used in analysis (Hz).
    max_formant_hz: f64,
    /// Maximum number of formants per frame.
    max_num_formants: usize,
}

impl Formant {
    /// Create a new Formant object from pre-computed frames.
    pub fn new(
        frames: Vec<FormantFrame>,
        time_step: f64,
        max_formant_hz: f64,
        max_num_formants: usize,
    ) -> Self {
        Self {
            frames,
            time_step,
            max_formant_hz,
            max_num_formants,
        }
    }

    /// Get a slice of all formant frames.
    #[inline]
    pub fn frames(&self) -> &[FormantFrame] {
        &self.frames
    }

    /// Get the total number of analysis frames.
    #[inline]
    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }

    /// Get the time step between consecutive frames (seconds).
    #[inline]
    pub fn time_step(&self) -> f64 {
        self.time_step
    }

    /// Get the maximum formant frequency used in analysis (Hz).
    #[inline]
    pub fn max_formant_hz(&self) -> f64 {
        self.max_formant_hz
    }

    /// Get the maximum number of formants per frame.
    #[inline]
    pub fn max_num_formants(&self) -> usize {
        self.max_num_formants
    }

    /// Get an array of all frame center times.
    pub fn times(&self) -> Array1<f64> {
        Array1::from_iter(self.frames.iter().map(|f| f.time))
    }

    /// Extract the frequency contour for a specific formant.
    ///
    /// # Arguments
    ///
    /// * `formant_number` - Which formant to extract (1 = F1, 2 = F2, etc.)
    ///
    /// # Returns
    ///
    /// An array of frequencies for each frame. Frames where the formant
    /// was not detected contain NaN.
    pub fn formant_values(&self, formant_number: usize) -> Array1<f64> {
        Array1::from_iter(self.frames.iter().map(|frame| {
            frame
                .get_formant(formant_number)
                .map(|fp| fp.frequency)
                .unwrap_or(f64::NAN)
        }))
    }

    /// Extract the bandwidth contour for a specific formant.
    ///
    /// # Arguments
    ///
    /// * `formant_number` - Which formant to extract (1 = B1, 2 = B2, etc.)
    ///
    /// # Returns
    ///
    /// An array of bandwidths for each frame. Frames where the formant
    /// was not detected contain NaN.
    pub fn bandwidth_values(&self, formant_number: usize) -> Array1<f64> {
        Array1::from_iter(self.frames.iter().map(|frame| {
            frame
                .get_formant(formant_number)
                .map(|fp| fp.bandwidth)
                .unwrap_or(f64::NAN)
        }))
    }

    /// Get the formant frequency at a specific time with interpolation.
    ///
    /// # Arguments
    ///
    /// * `formant_number` - Which formant (1-based)
    /// * `time` - Query time in seconds
    /// * `interpolation` - "linear" for linear interpolation, "nearest" for nearest frame
    ///
    /// # Returns
    ///
    /// The formant frequency in Hz, or `None` if not available at that time.
    pub fn get_value_at_time(
        &self,
        formant_number: usize,
        time: f64,
        interpolation: &str,
    ) -> Option<f64> {
        if self.n_frames() == 0 {
            return None;
        }

        // Calculate fractional frame index
        let t0 = self.frames[0].time;
        let idx_float = (time - t0) / self.time_step;

        // Check if time is within valid range (allowing half a frame of extrapolation)
        if idx_float < -0.5 || idx_float > self.n_frames() as f64 - 0.5 {
            return None;
        }

        match interpolation {
            "nearest" => {
                // Round to nearest frame
                let idx = idx_float.round() as usize;
                let idx = idx.min(self.n_frames() - 1);
                self.frames[idx]
                    .get_formant(formant_number)
                    .map(|fp| fp.frequency)
            }
            "linear" => {
                // Linear interpolation between adjacent frames
                let idx = idx_float.floor() as isize;
                let frac = idx_float - idx as f64;

                // Clamp indices to valid range
                let i1 = 0.max(idx).min(self.n_frames() as isize - 1) as usize;
                let i2 = 0.max(idx + 1).min(self.n_frames() as isize - 1) as usize;

                let fp1 = self.frames[i1].get_formant(formant_number);
                let fp2 = self.frames[i2].get_formant(formant_number);

                // Handle cases where one or both formants are missing
                match (fp1, fp2) {
                    (None, None) => None,
                    (None, Some(f)) => Some(f.frequency),
                    (Some(f), None) => Some(f.frequency),
                    (Some(f1), Some(f2)) => {
                        Some(f1.frequency * (1.0 - frac) + f2.frequency * frac)
                    }
                }
            }
            _ => None,
        }
    }

    /// Get the formant bandwidth at a specific time with interpolation.
    ///
    /// # Arguments
    ///
    /// * `formant_number` - Which formant (1-based)
    /// * `time` - Query time in seconds
    /// * `interpolation` - "linear" for linear interpolation, "nearest" for nearest frame
    ///
    /// # Returns
    ///
    /// The bandwidth in Hz, or `None` if not available at that time.
    pub fn get_bandwidth_at_time(
        &self,
        formant_number: usize,
        time: f64,
        interpolation: &str,
    ) -> Option<f64> {
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
                self.frames[idx]
                    .get_formant(formant_number)
                    .map(|fp| fp.bandwidth)
            }
            "linear" => {
                let idx = idx_float.floor() as isize;
                let frac = idx_float - idx as f64;

                let i1 = 0.max(idx).min(self.n_frames() as isize - 1) as usize;
                let i2 = 0.max(idx + 1).min(self.n_frames() as isize - 1) as usize;

                let fp1 = self.frames[i1].get_formant(formant_number);
                let fp2 = self.frames[i2].get_formant(formant_number);

                match (fp1, fp2) {
                    (None, None) => None,
                    (None, Some(f)) => Some(f.bandwidth),
                    (Some(f), None) => Some(f.bandwidth),
                    (Some(f1), Some(f2)) => {
                        Some(f1.bandwidth * (1.0 - frac) + f2.bandwidth * frac)
                    }
                }
            }
            _ => None,
        }
    }
}

// =============================================================================
// Window Functions
// =============================================================================

/// Generate a Gaussian window for formant analysis.
///
/// The Gaussian window is defined as:
///   w[i] = exp(-α × x²)
/// where x = (i - mid) / mid ranges from -1 to 1.
///
/// We use α = 12.0 which gives approximately -120 dB sidelobes,
/// as specified in the Praat documentation for formant analysis.
///
/// # Arguments
///
/// * `n` - Number of samples in the window
///
/// # Returns
///
/// Vector of window coefficients, symmetric around the center.
fn gaussian_window(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0];
    }

    // α = 12.0 gives Gaussian with sidelobes below -120 dB
    // This is the standard value for formant analysis
    let alpha = 12.0;

    // Center point of the window
    let mid = (n - 1) as f64 / 2.0;

    (0..n)
        .map(|i| {
            // Normalize position to range [-1, 1]
            let x = (i as f64 - mid) / mid;
            // Gaussian function
            (-alpha * x * x).exp()
        })
        .collect()
}

// =============================================================================
// LPC Analysis (Burg's Algorithm)
// =============================================================================

/// Compute LPC coefficients using Burg's algorithm.
///
/// Burg's algorithm is a method for estimating the parameters of an
/// autoregressive (AR) model. It directly computes the reflection coefficients
/// by minimizing the forward and backward prediction errors simultaneously.
///
/// The AR model is: x[n] = -Σ(k=1 to p) a[k] × x[n-k] + e[n]
///
/// # Algorithm (Childers 1978, pp. 252-255)
///
/// 1. Initialize forward error ef = backward error eb = input signal
/// 2. For each order k from 1 to p:
///    a. Compute reflection coefficient:
///       r[k] = -2 × Σ(ef[i] × eb[i-1]) / Σ(ef[i]² + eb[i-1]²)
///    b. Update prediction errors:
///       ef'[i] = ef[i] + r[k] × eb[i-1]
///       eb'[i] = eb[i-1] + r[k] × ef[i]
///    c. Update LPC coefficients via Levinson recursion
///
/// # Arguments
///
/// * `samples` - Windowed signal samples
/// * `order` - LPC order (typically 2 × number of formants)
///
/// # Returns
///
/// LPC coefficients a[0..order] where a[0] = 1.0 (the AR polynomial).
fn burg_lpc(samples: &[f64], order: usize) -> Vec<f64> {
    let n = samples.len();

    // Need more samples than coefficients for valid estimation
    if n <= order {
        return vec![0.0; order + 1];
    }

    // Initialize LPC coefficients: a[0] = 1.0, rest = 0.0
    let mut a = vec![0.0; order + 1];
    a[0] = 1.0;

    // Initialize forward and backward prediction errors to the input signal
    // These will be updated iteratively as we compute higher-order coefficients
    let mut ef: Vec<f64> = samples.to_vec();
    let mut eb: Vec<f64> = samples.to_vec();

    // Iterate through each order
    for k in 1..=order {
        // =====================================================================
        // Step 1: Compute reflection coefficient r[k]
        // =====================================================================
        // The reflection coefficient minimizes the sum of forward and backward
        // prediction error energies. Formula from Burg (1968):
        //   r[k] = -2 × Σ(ef[i] × eb[i-1]) / Σ(ef[i]² + eb[i-1]²)
        let mut num = 0.0; // Numerator: cross-correlation
        let mut den = 0.0; // Denominator: sum of energies
        for i in k..n {
            num += ef[i] * eb[i - 1];
            den += ef[i] * ef[i] + eb[i - 1] * eb[i - 1];
        }

        // Avoid division by zero (can happen with silent input)
        if den < 1e-30 {
            break;
        }

        let reflection = -2.0 * num / den;

        // =====================================================================
        // Step 2: Update prediction errors using lattice filter structure
        // =====================================================================
        let mut ef_new = vec![0.0; n];
        let mut eb_new = vec![0.0; n];
        for i in k..n {
            // Forward error: current ef plus reflection × previous backward error
            ef_new[i] = ef[i] + reflection * eb[i - 1];
            // Backward error: previous backward error plus reflection × current ef
            eb_new[i] = eb[i - 1] + reflection * ef[i];
        }
        ef = ef_new;
        eb = eb_new;

        // =====================================================================
        // Step 3: Update LPC coefficients using Levinson-Durbin recursion
        // =====================================================================
        // a[i] = a[i] + reflection × a[k-i] for i = 1..k-1
        // a[k] = reflection
        let mut a_new = vec![0.0; order + 1];
        a_new[0] = 1.0;
        for i in 1..k {
            a_new[i] = a[i] + reflection * a[k - i];
        }
        a_new[k] = reflection;
        a = a_new;
    }

    a
}

// =============================================================================
// Polynomial Root Finding
// =============================================================================

/// Evaluate the LPC polynomial and its derivative at a complex point z.
///
/// The LPC polynomial is: P(z) = z^p + a[1]×z^(p-1) + ... + a[p]
///
/// Uses Horner's method for efficient and numerically stable evaluation:
///   P(z) = (...((1×z + a[1])×z + a[2])×z + ...)×z + a[p]
///
/// The derivative P'(z) is computed simultaneously using the chain rule.
///
/// # Arguments
///
/// * `a` - LPC coefficients where a[0] = 1.0
/// * `z` - Complex point to evaluate at
///
/// # Returns
///
/// Tuple (P(z), P'(z)) - polynomial value and its derivative.
fn eval_polynomial(a: &[f64], z: Complex64) -> (Complex64, Complex64) {
    let order = a.len() - 1;
    if order < 1 {
        return (Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
    }

    // Horner's method: evaluate polynomial and derivative simultaneously
    // p_val accumulates the polynomial value
    // dp_val accumulates the derivative
    let mut p_val = Complex64::new(1.0, 0.0); // Start with leading coefficient (1)
    let mut dp_val = Complex64::new(0.0, 0.0); // Derivative starts at 0

    for coeff in a.iter().skip(1) {
        // Update derivative first: d/dz[P×z + c] = P + z×dP
        dp_val = p_val + z * dp_val;
        // Update polynomial: P×z + c
        p_val = p_val * z + Complex64::new(*coeff, 0.0);
    }

    (p_val, dp_val)
}

/// Polish a polynomial root using Newton-Raphson iteration.
///
/// Given an approximate root z₀, Newton-Raphson refines it:
///   z_{n+1} = z_n - P(z_n) / P'(z_n)
///
/// This converges quadratically for simple roots.
///
/// Reference: Numerical Recipes Ch. 9.5
///
/// # Arguments
///
/// * `a` - LPC coefficients
/// * `z` - Initial root estimate
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Relative tolerance for convergence
///
/// # Returns
///
/// Refined root estimate.
fn polish_root(a: &[f64], mut z: Complex64, max_iter: usize, tol: f64) -> Complex64 {
    for _ in 0..max_iter {
        let (p_val, dp_val) = eval_polynomial(a, z);

        // Avoid division by near-zero derivative
        if dp_val.norm() < 1e-30 {
            break;
        }

        // Newton-Raphson step
        let delta = p_val / dp_val;
        z = z - delta;

        // Check for convergence (relative change)
        if delta.norm() < tol * z.norm() {
            break;
        }
    }

    z
}

/// Reflect unstable roots to inside the unit circle.
///
/// For a stable AR filter, all poles must be inside the unit circle (|z| < 1).
/// If we have an unstable root z with |z| > 1, we reflect it to 1/conj(z),
/// which preserves the frequency (angle) but makes the filter stable.
///
/// This is standard practice in LPC analysis to ensure stable synthesis filters.
///
/// Note: With Burg's algorithm, unstable roots rarely occur because the method
/// guarantees minimum-phase solutions. This function is included for robustness.
///
/// # Arguments
///
/// * `roots` - Mutable slice of complex roots to process in-place.
fn reflect_unstable_roots(roots: &mut [Complex64]) {
    for root in roots.iter_mut() {
        let r = root.norm();
        if r > 1.0 {
            // Reflect to inside unit circle: z_new = 1 / conj(z) = conj(z) / |z|²
            // This preserves the angle (frequency) while reducing the magnitude
            *root = root.conj() / (r * r);
        }
    }
}

/// Find roots of the LPC polynomial using companion matrix eigenvalues.
///
/// The polynomial z^p + a[1]×z^(p-1) + ... + a[p] can be converted to an
/// eigenvalue problem by constructing the companion matrix:
///
/// ```text
///     [ -a[1]  -a[2]  -a[3]  ...  -a[p] ]
///     [   1      0      0    ...    0   ]
/// C = [   0      1      0    ...    0   ]
///     [   ⋮      ⋮      ⋮    ⋱     ⋮   ]
///     [   0      0      0    ...    0   ]
/// ```
///
/// The eigenvalues of C are exactly the roots of the polynomial.
///
/// Reference: Numerical Recipes Ch. 9.5
///
/// # Arguments
///
/// * `a` - LPC coefficients where a[0] = 1.0
/// * `polish` - Whether to refine roots with Newton-Raphson
/// * `reflect_unstable` - Whether to reflect roots with |z| > 1
///
/// # Returns
///
/// Vector of complex roots.
fn lpc_roots(a: &[f64], polish: bool, reflect_unstable: bool) -> Vec<Complex64> {
    let order = a.len() - 1;
    if order < 1 {
        return Vec::new();
    }

    // Check if coefficients are essentially zero (silent frame).
    // This avoids numerical issues in eigenvalue computation for degenerate matrices.
    let coeff_sum: f64 = a.iter().skip(1).map(|c| c.abs()).sum();
    if coeff_sum < 1e-10 {
        return Vec::new();
    }

    // =========================================================================
    // Build companion matrix
    // =========================================================================
    // For polynomial z^p + c1×z^(p-1) + ... + cp:
    // - First row contains negated coefficients: [-c1, -c2, ..., -cp]
    // - Subdiagonal contains ones
    // - All other entries are zero
    let mut companion = nalgebra::DMatrix::<f64>::zeros(order, order);

    // First row: -a[1], -a[2], ..., -a[p]
    for i in 0..order {
        companion[(0, i)] = -a[i + 1];
    }

    // Subdiagonal: ones
    for i in 1..order {
        companion[(i, i - 1)] = 1.0;
    }

    // =========================================================================
    // Compute eigenvalues using Schur decomposition
    // =========================================================================
    // Schur decomposition is more numerically stable than direct eigenvalue
    // computation, especially for non-symmetric matrices like the companion matrix.
    let schur = companion.schur();
    let eigenvalues = schur.complex_eigenvalues();

    // Convert from nalgebra Complex to num_complex Complex64
    let mut roots: Vec<Complex64> = eigenvalues
        .iter()
        .map(|e| Complex64::new(e.re, e.im))
        .collect();

    // =========================================================================
    // Post-processing: reflect unstable roots
    // =========================================================================
    if reflect_unstable {
        reflect_unstable_roots(&mut roots);
    }

    // =========================================================================
    // Post-processing: polish roots with Newton-Raphson
    // =========================================================================
    // This improves accuracy, especially for closely-spaced roots
    if polish {
        for root in roots.iter_mut() {
            *root = polish_root(a, *root, 10, 1e-10);
        }
    }

    roots
}

// =============================================================================
// Root-to-Formant Conversion
// =============================================================================

/// Convert complex polynomial roots to formant frequencies and bandwidths.
///
/// For an LPC polynomial root z = r × e^(iθ):
/// - **Frequency** = θ × fs / (2π)   (angle determines frequency)
/// - **Bandwidth** = -ln(r) × fs / π  (distance from unit circle determines bandwidth)
///
/// Roots closer to the unit circle (r → 1) give narrower bandwidths (sharper resonances).
/// Roots inside the unit circle (r < 1) give positive bandwidths (stable formants).
///
/// We only consider roots in the upper half-plane (positive imaginary part)
/// since the lower half-plane gives negative frequencies (redundant for real signals).
///
/// Reference: Markel & Gray (1976)
///
/// # Arguments
///
/// * `roots` - Complex roots of the LPC polynomial
/// * `sample_rate` - Sample rate after resampling (Hz)
/// * `min_freq` - Minimum valid formant frequency (Hz)
/// * `max_freq` - Maximum valid formant frequency (Hz)
///
/// # Returns
///
/// Vector of FormantPoint objects, sorted by frequency.
fn roots_to_formants(
    roots: &[Complex64],
    sample_rate: f64,
    min_freq: f64,
    max_freq: f64,
) -> Vec<FormantPoint> {
    let mut formants = Vec::new();

    for root in roots {
        // Only consider roots in upper half-plane (positive frequency)
        // Lower half-plane roots give negative frequencies
        if root.im <= 0.0 {
            continue;
        }

        // Magnitude and angle of the root
        let r = root.norm(); // Distance from origin
        let theta = root.arg(); // Angle in radians

        // Convert angle to frequency
        // θ radians corresponds to θ/(2π) cycles, at sample_rate samples/second
        let freq = theta * sample_rate / (2.0 * std::f64::consts::PI);

        // Convert magnitude to bandwidth
        // A root at z = r×e^(iθ) gives bandwidth B = -ln(r) × fs / π
        // Closer to unit circle (r→1) means narrower bandwidth (ln(1)=0)
        let bandwidth = if r > 0.0 {
            -r.ln() * sample_rate / std::f64::consts::PI
        } else {
            f64::INFINITY
        };

        // Filter by frequency range and validity
        // Formants too close to DC or Nyquist are typically artifacts
        if freq >= min_freq && freq <= max_freq && bandwidth > 0.0 {
            formants.push(FormantPoint::new(freq, bandwidth));
        }
    }

    // Sort formants by frequency (F1, F2, F3, ...)
    formants.sort_by(|a, b| {
        a.frequency
            .partial_cmp(&b.frequency)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    formants
}

// =============================================================================
// Resampling
// =============================================================================

/// Resample audio using FFT-based sinc interpolation.
///
/// This method performs ideal (sinc) interpolation via the frequency domain:
///
/// 1. Compute FFT of input signal
/// 2. Truncate (downsampling) or zero-pad (upsampling) the spectrum
/// 3. Compute inverse FFT
///
/// This matches the behavior of scipy.signal.resample and provides high-quality
/// resampling without aliasing artifacts.
///
/// # Algorithm Details
///
/// For **downsampling** (new_rate < old_rate):
/// - Keep only the low-frequency components that fit in the new Nyquist range
/// - Discard high frequencies that would alias
/// - Handle Nyquist frequency specially (split between positive/negative)
///
/// For **upsampling** (new_rate > old_rate):
/// - Copy all original frequency components
/// - Zero-pad the high frequencies (no new information added)
///
/// # Arguments
///
/// * `samples` - Input audio samples
/// * `old_rate` - Original sample rate (Hz)
/// * `new_rate` - Target sample rate (Hz)
///
/// # Returns
///
/// Resampled audio samples at the new sample rate.
fn resample(samples: &[f64], old_rate: f64, new_rate: f64) -> Vec<f64> {
    // No resampling needed if rates are equal
    if (old_rate - new_rate).abs() < 1e-6 {
        return samples.to_vec();
    }

    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }

    // Calculate target length based on rate ratio
    let new_length = (n as f64 * new_rate / old_rate).round() as usize;
    if new_length == 0 {
        return Vec::new();
    }

    // =========================================================================
    // Step 1: Forward FFT of input signal
    // =========================================================================
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);

    // Convert real samples to complex for FFT
    let mut spectrum: Vec<Complex<f64>> = samples.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut spectrum);

    // =========================================================================
    // Step 2: Resize spectrum to target length
    // =========================================================================
    let mut new_spectrum = vec![Complex::new(0.0, 0.0); new_length];

    // Calculate Nyquist indices
    let half_n = n / 2; // Nyquist index for original
    let half_new = new_length / 2; // Nyquist index for resampled

    if new_length <= n {
        // -----------------------------------------------------------------
        // Downsampling: truncate high frequencies
        // -----------------------------------------------------------------
        // Copy positive frequencies (indices 0 to half_new)
        for i in 0..=half_new.min(half_n) {
            new_spectrum[i] = spectrum[i];
        }

        // Copy negative frequencies (from end of array)
        // In FFT output, negative frequencies are at indices n-1, n-2, ...
        for i in 1..half_new.min(half_n) {
            new_spectrum[new_length - i] = spectrum[n - i];
        }

        // Handle output Nyquist frequency for even-length output.
        // When downsampling, the output Nyquist bin (half_new) was copied from
        // spectrum[half_new] in the positive frequency loop. But this bin's
        // negative mirror (spectrum[n - half_new]) was lost in the truncation.
        // We must combine both to preserve the full spectral energy at this
        // frequency: new[half_new] = spectrum[half_new] + spectrum[n - half_new].
        // For real signals: spectrum[n-k] = conj(spectrum[k]), so the sum
        // equals 2 * real(spectrum[half_new]).
        if new_length % 2 == 0 && half_new < half_n {
            new_spectrum[half_new] = spectrum[half_new] + spectrum[n - half_new];
        }
    } else {
        // -----------------------------------------------------------------
        // Upsampling: zero-pad high frequencies
        // -----------------------------------------------------------------
        // Copy all positive frequencies
        for i in 0..=half_n {
            new_spectrum[i] = spectrum[i];
        }

        // Copy all negative frequencies
        for i in 1..half_n {
            new_spectrum[new_length - i] = spectrum[n - i];
        }

        // High frequencies (between old and new Nyquist) remain zero
    }

    // =========================================================================
    // Step 3: Inverse FFT to get resampled signal
    // =========================================================================
    let ifft = planner.plan_fft_inverse(new_length);
    ifft.process(&mut new_spectrum);

    // Scale output: IFFT gives sum, need to divide by length
    // Also scale by length ratio to preserve signal amplitude
    let scale = new_length as f64 / n as f64;
    new_spectrum
        .iter()
        .map(|c| c.re / new_length as f64 * scale)
        .collect()
}

// =============================================================================
// Main Formant Analysis Function
// =============================================================================

/// Compute formant tracks using Burg's LPC method.
///
/// This is the main entry point for formant analysis. It processes a sound
/// file and returns formant frequency and bandwidth tracks over time.
///
/// # Algorithm Steps
///
/// 1. **Resample** to 2 × max_formant_hz (ensures adequate frequency resolution)
/// 2. **Pre-emphasize** to compensate for natural spectral tilt
/// 3. **For each frame:**
///    - Extract samples centered at frame time
///    - Apply Gaussian window
///    - Compute LPC coefficients using Burg's algorithm
///    - Find polynomial roots via companion matrix eigenvalues
///    - Convert roots to frequencies and bandwidths
///    - Filter and sort formants
///
/// # Arguments
///
/// * `sound` - Input Sound object (must be mono)
/// * `time_step` - Time step between frames in seconds (0 = auto: 25% of window)
/// * `max_num_formants` - Maximum number of formants to track (typically 5)
/// * `max_formant_hz` - Maximum formant frequency to consider (e.g., 5500 Hz for male, 5000 Hz for female)
/// * `window_length` - Effective window length in seconds (actual = 2× this value)
/// * `pre_emphasis_from` - Pre-emphasis starts at this frequency (typically 50 Hz)
///
/// # Returns
///
/// A `Formant` object containing all analysis frames.
///
/// # Example
///
/// ```ignore
/// let formant = sound_to_formant_burg(&sound, 0.0, 5, 5500.0, 0.025, 50.0);
/// let f1_values = formant.formant_values(1); // Get F1 contour
/// ```
pub fn sound_to_formant_burg(
    sound: &Sound,
    time_step: f64,
    max_num_formants: usize,
    max_formant_hz: f64,
    window_length: f64,
    pre_emphasis_from: f64,
) -> Formant {
    let original_samples = sound.samples();
    let original_rate = sound.sample_rate();
    let duration = sound.duration();

    // =========================================================================
    // Step 1: Resample to 2 × max_formant_hz
    // =========================================================================
    // The Nyquist frequency of the resampled signal will be max_formant_hz,
    // ensuring we can accurately represent formants up to that frequency.
    // Only downsample if the original rate is higher than needed.
    let target_rate = 2.0 * max_formant_hz;
    let (samples, sample_rate) = if target_rate < original_rate {
        let resampled = resample(original_samples.as_slice().unwrap(), original_rate, target_rate);
        (resampled, target_rate)
    } else {
        (original_samples.to_vec(), original_rate)
    };

    // =========================================================================
    // Step 2: Pre-emphasis filter
    // =========================================================================
    // Pre-emphasis boosts high frequencies to compensate for:
    // - Natural -6dB/octave spectral tilt from lip radiation
    // - Better numerical conditioning for LPC
    //
    // Filter: x'[i] = x[i] - α × x[i-1]
    // where α = exp(-2π × F × Δt), F = pre_emphasis_from frequency
    let dt = 1.0 / sample_rate;
    let alpha = (-2.0 * std::f64::consts::PI * pre_emphasis_from * dt).exp();

    let mut pre_emphasized = vec![0.0; samples.len()];
    if !samples.is_empty() {
        pre_emphasized[0] = samples[0];
        for i in 1..samples.len() {
            pre_emphasized[i] = samples[i] - alpha * samples[i - 1];
        }
    }

    // =========================================================================
    // Step 3: Window and frame parameters
    // =========================================================================
    // Per Praat documentation: "actual length is twice this value"
    // This means if window_length = 0.025s, the actual window is 0.050s
    let physical_window_duration = 2.0 * window_length;

    // Convert to samples, ensuring odd number for symmetric window
    let mut window_samples = (physical_window_duration * sample_rate).round() as usize;
    if window_samples % 2 == 0 {
        window_samples += 1;
    }
    let half_window = window_samples / 2;

    // Time step: default is 25% of window length (4× overlap)
    let time_step = if time_step <= 0.0 {
        window_length / 4.0
    } else {
        time_step
    };

    // LPC order: 2 coefficients per formant (each formant is a complex conjugate pair)
    let lpc_order = 2 * max_num_formants;

    // Generate the Gaussian analysis window (computed once, reused for all frames)
    let window = gaussian_window(window_samples);

    // =========================================================================
    // Step 4: Frame timing calculation
    // =========================================================================
    // Frames are centered within the signal, with equal margins at start and end.
    // n_frames = floor((duration - window_duration) / time_step) + 1
    // t1 = (duration - (n_frames - 1) × time_step) / 2
    let n_frames = ((duration - physical_window_duration) / time_step).floor() as usize + 1;
    let n_frames = n_frames.max(1);
    let t1 = (duration - (n_frames - 1) as f64 * time_step) / 2.0;

    // =========================================================================
    // Step 5: Process each frame
    // =========================================================================
    let mut frames = Vec::with_capacity(n_frames);

    for i in 0..n_frames {
        // Frame center time
        let t = t1 + i as f64 * time_step;

        // -----------------------------------------------------------------
        // Extract frame samples from pre-emphasized signal
        // -----------------------------------------------------------------
        let center_sample = (t * sample_rate).round() as isize;
        let start_sample = center_sample - half_window as isize;
        let end_sample = start_sample + window_samples as isize;

        // Handle boundary cases (frames near start/end of signal)
        let mut frame_samples = vec![0.0; window_samples];
        if start_sample < 0 || end_sample > pre_emphasized.len() as isize {
            // Partial overlap with signal - copy available samples, pad rest with zeros
            let src_start = 0.max(start_sample) as usize;
            let src_end = (pre_emphasized.len() as isize).min(end_sample) as usize;
            let dst_start = (src_start as isize - start_sample) as usize;
            let dst_end = dst_start + (src_end - src_start);
            frame_samples[dst_start..dst_end].copy_from_slice(&pre_emphasized[src_start..src_end]);
        } else {
            // Full frame within signal bounds
            let start = start_sample as usize;
            let end = end_sample as usize;
            frame_samples.copy_from_slice(&pre_emphasized[start..end]);
        }

        // -----------------------------------------------------------------
        // Apply Gaussian window
        // -----------------------------------------------------------------
        let windowed: Vec<f64> = frame_samples
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect();

        // -----------------------------------------------------------------
        // Compute LPC coefficients using Burg's algorithm
        // -----------------------------------------------------------------
        let lpc_coeffs = burg_lpc(&windowed, lpc_order);

        // -----------------------------------------------------------------
        // Find roots of LPC polynomial
        // -----------------------------------------------------------------
        // polish=true: refine roots with Newton-Raphson
        // reflect_unstable=true: ensure all roots are inside unit circle
        let roots = lpc_roots(&lpc_coeffs, true, true);

        // -----------------------------------------------------------------
        // Convert roots to formant frequencies and bandwidths
        // -----------------------------------------------------------------
        // Filter out formants outside valid range (50 Hz to max_formant - 50 Hz)
        let mut formant_points = roots_to_formants(&roots, sample_rate, 50.0, max_formant_hz - 50.0);

        // Keep only the requested number of formants
        formant_points.truncate(max_num_formants);

        frames.push(FormantFrame::new(t, formant_points));
    }

    Formant::new(frames, time_step, max_formant_hz, max_num_formants)
}
