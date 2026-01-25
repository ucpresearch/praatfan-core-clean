//! Formant - LPC-based formant frequency tracks.
//!
//! Documentation sources:
//! - Praat manual: Sound: To Formant (burg)...
//! - Childers (1978): "Modern Spectrum Analysis", pp. 252-255 (Burg's algorithm)
//! - Numerical Recipes Ch. 9.5 (root finding via companion matrix)
//! - Markel & Gray (1976): root-to-formant conversion
//!
//! Key documented facts:
//! - Window length parameter: "actual length is twice this value"
//! - Resample to 2 × max_formant_hz before analysis
//! - Pre-emphasis: x'[i] = x[i] - α × x[i-1], α = exp(-2π × F × Δt)
//! - LPC order: 2 × max_formants
//! - Formant filtering: remove < 50 Hz and > (max_formant - 50) Hz

use ndarray::Array1;
use num_complex::Complex64;
use rustfft::{FftPlanner, num_complex::Complex};

use crate::sound::Sound;

/// A single formant at a point in time.
#[derive(Debug, Clone)]
pub struct FormantPoint {
    /// Frequency in Hz.
    pub frequency: f64,
    /// Bandwidth in Hz.
    pub bandwidth: f64,
}

impl FormantPoint {
    /// Create a new FormantPoint.
    pub fn new(frequency: f64, bandwidth: f64) -> Self {
        Self {
            frequency,
            bandwidth,
        }
    }
}

/// Formant analysis results for a single frame.
#[derive(Debug, Clone)]
pub struct FormantFrame {
    /// Time in seconds.
    pub time: f64,
    /// List of formants (F1, F2, ...).
    pub formants: Vec<FormantPoint>,
}

impl FormantFrame {
    /// Create a new FormantFrame.
    pub fn new(time: f64, formants: Vec<FormantPoint>) -> Self {
        Self { time, formants }
    }

    /// Number of formants in this frame.
    #[inline]
    pub fn n_formants(&self) -> usize {
        self.formants.len()
    }

    /// Get formant n (1-based index).
    pub fn get_formant(&self, n: usize) -> Option<&FormantPoint> {
        if n >= 1 && n <= self.formants.len() {
            Some(&self.formants[n - 1])
        } else {
            None
        }
    }
}

/// Formant tracks over time.
#[derive(Debug, Clone)]
pub struct Formant {
    /// List of formant frames.
    frames: Vec<FormantFrame>,
    /// Time step between frames.
    time_step: f64,
    /// Maximum formant frequency.
    max_formant_hz: f64,
    /// Maximum number of formants per frame.
    max_num_formants: usize,
}

impl Formant {
    /// Create a new Formant object.
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

    /// Get the formant frames.
    #[inline]
    pub fn frames(&self) -> &[FormantFrame] {
        &self.frames
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

    /// Get the maximum formant frequency.
    #[inline]
    pub fn max_formant_hz(&self) -> f64 {
        self.max_formant_hz
    }

    /// Get the maximum number of formants per frame.
    #[inline]
    pub fn max_num_formants(&self) -> usize {
        self.max_num_formants
    }

    /// Get array of frame times.
    pub fn times(&self) -> Array1<f64> {
        Array1::from_iter(self.frames.iter().map(|f| f.time))
    }

    /// Get array of formant frequencies for a specific formant.
    ///
    /// # Arguments
    ///
    /// * `formant_number` - Formant number (1 = F1, 2 = F2, etc.)
    ///
    /// # Returns
    ///
    /// Array of frequencies (NaN where formant not present)
    pub fn formant_values(&self, formant_number: usize) -> Array1<f64> {
        Array1::from_iter(self.frames.iter().map(|frame| {
            frame
                .get_formant(formant_number)
                .map(|fp| fp.frequency)
                .unwrap_or(f64::NAN)
        }))
    }

    /// Get array of bandwidths for a specific formant.
    ///
    /// # Arguments
    ///
    /// * `formant_number` - Formant number (1 = B1, 2 = B2, etc.)
    ///
    /// # Returns
    ///
    /// Array of bandwidths (NaN where formant not present)
    pub fn bandwidth_values(&self, formant_number: usize) -> Array1<f64> {
        Array1::from_iter(self.frames.iter().map(|frame| {
            frame
                .get_formant(formant_number)
                .map(|fp| fp.bandwidth)
                .unwrap_or(f64::NAN)
        }))
    }

    /// Get formant frequency at a specific time.
    ///
    /// # Arguments
    ///
    /// * `formant_number` - Formant number (1-based)
    /// * `time` - Time in seconds
    /// * `interpolation` - Interpolation method ("linear" or "nearest")
    ///
    /// # Returns
    ///
    /// Formant frequency in Hz, or None if not present
    pub fn get_value_at_time(
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
                    .map(|fp| fp.frequency)
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

    /// Get bandwidth at a specific time.
    ///
    /// # Arguments
    ///
    /// * `formant_number` - Formant number (1-based)
    /// * `time` - Time in seconds
    /// * `interpolation` - Interpolation method ("linear" or "nearest")
    ///
    /// # Returns
    ///
    /// Bandwidth in Hz, or None if not present
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

/// Generate Gaussian window for formant analysis.
fn gaussian_window(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0];
    }

    let alpha = 12.0;
    let mid = (n - 1) as f64 / 2.0;

    (0..n)
        .map(|i| {
            let x = (i as f64 - mid) / mid;
            (-alpha * x * x).exp()
        })
        .collect()
}

/// Compute LPC coefficients using Burg's algorithm.
///
/// Reference: Childers (1978), "Modern Spectrum Analysis", pp. 252-255
///
/// # Arguments
///
/// * `samples` - Windowed signal samples
/// * `order` - LPC order (2 × number of formants)
///
/// # Returns
///
/// LPC coefficients a[0..order] where a[0] = 1.0
fn burg_lpc(samples: &[f64], order: usize) -> Vec<f64> {
    let n = samples.len();
    if n <= order {
        return vec![0.0; order + 1];
    }

    // Initialize
    let mut a = vec![0.0; order + 1];
    a[0] = 1.0;

    // Forward and backward prediction errors
    let mut ef: Vec<f64> = samples.to_vec();
    let mut eb: Vec<f64> = samples.to_vec();

    for k in 1..=order {
        // Compute reflection coefficient
        let mut num = 0.0;
        let mut den = 0.0;
        for i in k..n {
            num += ef[i] * eb[i - 1];
            den += ef[i] * ef[i] + eb[i - 1] * eb[i - 1];
        }

        if den < 1e-30 {
            break;
        }

        let reflection = -2.0 * num / den;

        // Update prediction errors
        let mut ef_new = vec![0.0; n];
        let mut eb_new = vec![0.0; n];
        for i in k..n {
            ef_new[i] = ef[i] + reflection * eb[i - 1];
            eb_new[i] = eb[i - 1] + reflection * ef[i];
        }
        ef = ef_new;
        eb = eb_new;

        // Update LPC coefficients (Levinson recursion)
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

/// Evaluate LPC polynomial and its derivative at z.
///
/// The polynomial is: P(z) = z^p + a[1]*z^{p-1} + ... + a[p]
///
/// # Arguments
///
/// * `a` - LPC coefficients (a[0] = 1.0)
/// * `z` - Point to evaluate at
///
/// # Returns
///
/// (P(z), P'(z)) - polynomial value and derivative
fn eval_polynomial(a: &[f64], z: Complex64) -> (Complex64, Complex64) {
    let order = a.len() - 1;
    if order < 1 {
        return (Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
    }

    // Horner's method for polynomial evaluation
    let mut p_val = Complex64::new(1.0, 0.0);
    let mut dp_val = Complex64::new(0.0, 0.0);

    for coeff in a.iter().skip(1) {
        dp_val = p_val + z * dp_val;
        p_val = p_val * z + Complex64::new(*coeff, 0.0);
    }

    (p_val, dp_val)
}

/// Polish a root using Newton-Raphson iteration.
///
/// Reference: Numerical Recipes Ch. 9.5
fn polish_root(a: &[f64], mut z: Complex64, max_iter: usize, tol: f64) -> Complex64 {
    for _ in 0..max_iter {
        let (p_val, dp_val) = eval_polynomial(a, z);

        if dp_val.norm() < 1e-30 {
            break;
        }

        let delta = p_val / dp_val;
        z = z - delta;

        if delta.norm() < tol * z.norm() {
            break;
        }
    }

    z
}

/// Reflect unstable roots (|z| > 1) to inside the unit circle.
///
/// For a root z with |z| > 1, the reflected root is 1/conj(z).
fn reflect_unstable_roots(roots: &mut [Complex64]) {
    for root in roots.iter_mut() {
        let r = root.norm();
        if r > 1.0 {
            // Reflect: z_new = 1 / conj(z) = conj(z) / |z|^2
            *root = root.conj() / (r * r);
        }
    }
}

/// Find roots of LPC polynomial using companion matrix eigenvalues.
///
/// The polynomial is: 1 + a[1]*z^{-1} + ... + a[p]*z^{-p}
/// We find roots of: z^p + a[1]*z^{p-1} + ... + a[p]
fn lpc_roots(a: &[f64], polish: bool, reflect_unstable: bool) -> Vec<Complex64> {
    let order = a.len() - 1;
    if order < 1 {
        return Vec::new();
    }

    // Check if coefficients are essentially zero (silent frame)
    // This avoids hanging in Schur decomposition for degenerate matrices
    let coeff_sum: f64 = a.iter().skip(1).map(|c| c.abs()).sum();
    if coeff_sum < 1e-10 {
        return Vec::new();
    }

    // Build companion matrix using nalgebra
    // For polynomial z^p + c1*z^{p-1} + ... + cp
    // Companion matrix has -coefficients in first row, 1s on subdiagonal
    let mut companion = nalgebra::DMatrix::<f64>::zeros(order, order);

    // First row: -a[1], -a[2], ..., -a[p]
    for i in 0..order {
        companion[(0, i)] = -a[i + 1];
    }

    // Subdiagonal: 1s
    for i in 1..order {
        companion[(i, i - 1)] = 1.0;
    }

    // Compute eigenvalues using nalgebra's Schur decomposition
    let schur = companion.schur();
    let eigenvalues = schur.complex_eigenvalues();

    let mut roots: Vec<Complex64> = eigenvalues
        .iter()
        .map(|e| Complex64::new(e.re, e.im))
        .collect();

    // Reflect unstable roots (|z| > 1) inside unit circle
    if reflect_unstable {
        reflect_unstable_roots(&mut roots);
    }

    // Polish roots with Newton-Raphson
    if polish {
        for root in roots.iter_mut() {
            *root = polish_root(a, *root, 10, 1e-10);
        }
    }

    roots
}

/// Convert complex roots to formant frequencies and bandwidths.
///
/// For a root z = r * exp(i*theta):
/// - Frequency = theta * sample_rate / (2*pi)
/// - Bandwidth = -ln(r) * sample_rate / pi
fn roots_to_formants(
    roots: &[Complex64],
    sample_rate: f64,
    min_freq: f64,
    max_freq: f64,
) -> Vec<FormantPoint> {
    let mut formants = Vec::new();

    for root in roots {
        // Only consider roots in upper half-plane (positive frequency)
        if root.im <= 0.0 {
            continue;
        }

        let r = root.norm();
        let theta = root.arg();

        // Frequency from angle
        let freq = theta * sample_rate / (2.0 * std::f64::consts::PI);

        // Bandwidth from radius
        let bandwidth = if r > 0.0 {
            -r.ln() * sample_rate / std::f64::consts::PI
        } else {
            f64::INFINITY
        };

        // Filter by frequency range
        if freq >= min_freq && freq <= max_freq && bandwidth > 0.0 {
            formants.push(FormantPoint::new(freq, bandwidth));
        }
    }

    // Sort by frequency
    formants.sort_by(|a, b| a.frequency.partial_cmp(&b.frequency).unwrap_or(std::cmp::Ordering::Equal));

    formants
}

/// Resample audio using FFT-based method (matches scipy.signal.resample).
///
/// This performs sinc interpolation via the frequency domain:
/// 1. Compute FFT of input
/// 2. Zero-pad or truncate in frequency domain to new length
/// 3. Inverse FFT
fn resample(samples: &[f64], old_rate: f64, new_rate: f64) -> Vec<f64> {
    if (old_rate - new_rate).abs() < 1e-6 {
        return samples.to_vec();
    }

    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }

    // Calculate new length
    let new_length = (n as f64 * new_rate / old_rate).round() as usize;
    if new_length == 0 {
        return Vec::new();
    }

    // Forward FFT
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);

    let mut spectrum: Vec<Complex<f64>> = samples
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    fft.process(&mut spectrum);

    // Create new spectrum with target length
    let mut new_spectrum = vec![Complex::new(0.0, 0.0); new_length];

    // Copy frequencies, handling the Nyquist properly
    // For downsampling (new_length < n): truncate high frequencies
    // For upsampling (new_length > n): zero-pad high frequencies
    let half_n = n / 2;
    let half_new = new_length / 2;

    if new_length <= n {
        // Downsampling: copy low frequencies
        // Positive frequencies (0 to half_new)
        for i in 0..=half_new.min(half_n) {
            new_spectrum[i] = spectrum[i];
        }
        // Negative frequencies (from end)
        for i in 1..half_new.min(half_n) {
            new_spectrum[new_length - i] = spectrum[n - i];
        }
        // Handle Nyquist if both have it
        if new_length % 2 == 0 && n % 2 == 0 && half_new <= half_n {
            // Split the Nyquist between positive and negative
            new_spectrum[half_new] = spectrum[half_n] * 0.5;
            new_spectrum[half_new] = new_spectrum[half_new] + spectrum[n - half_n] * 0.5;
        }
    } else {
        // Upsampling: copy all frequencies, zero-pad the rest
        // Positive frequencies
        for i in 0..=half_n {
            new_spectrum[i] = spectrum[i];
        }
        // Negative frequencies
        for i in 1..half_n {
            new_spectrum[new_length - i] = spectrum[n - i];
        }
    }

    // Inverse FFT
    let ifft = planner.plan_fft_inverse(new_length);
    ifft.process(&mut new_spectrum);

    // Scale and extract real part
    let scale = new_length as f64 / n as f64;
    new_spectrum
        .iter()
        .map(|c| c.re / new_length as f64 * scale)
        .collect()
}

/// Compute formants using Burg's LPC method.
///
/// Algorithm steps:
/// 1. Resample to 2 × max_formant_hz
/// 2. Pre-emphasize
/// 3. For each frame:
///    a. Extract windowed samples
///    b. Apply Gaussian window
///    c. Compute LPC coefficients using Burg's algorithm
///    d. Find polynomial roots via companion matrix eigenvalues
///    e. Convert roots to frequencies and bandwidths
///    f. Filter and sort formants
///
/// # Arguments
///
/// * `sound` - Sound object
/// * `time_step` - Time step in seconds (0 = auto: 25% of window)
/// * `max_num_formants` - Maximum number of formants to find
/// * `max_formant_hz` - Maximum formant frequency in Hz
/// * `window_length` - Window length in seconds (actual = 2× this value)
/// * `pre_emphasis_from` - Pre-emphasis from frequency in Hz
///
/// # Returns
///
/// Formant object
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

    // Step 1: Resample to 2 × max_formant_hz
    let target_rate = 2.0 * max_formant_hz;
    let (samples, sample_rate) = if target_rate < original_rate {
        let resampled = resample(original_samples.as_slice().unwrap(), original_rate, target_rate);
        (resampled, target_rate)
    } else {
        (original_samples.to_vec(), original_rate)
    };

    // Step 2: Pre-emphasis
    // x'[i] = x[i] - α × x[i-1]
    // α = exp(-2π × F × Δt)
    let dt = 1.0 / sample_rate;
    let alpha = (-2.0 * std::f64::consts::PI * pre_emphasis_from * dt).exp();
    let mut pre_emphasized = vec![0.0; samples.len()];
    if !samples.is_empty() {
        pre_emphasized[0] = samples[0];
        for i in 1..samples.len() {
            pre_emphasized[i] = samples[i] - alpha * samples[i - 1];
        }
    }

    // Window: actual length is 2× the parameter value
    let physical_window_duration = 2.0 * window_length;
    let mut window_samples = (physical_window_duration * sample_rate).round() as usize;
    if window_samples % 2 == 0 {
        window_samples += 1;
    }
    let half_window = window_samples / 2;

    // Time step: default is 25% of window length
    let time_step = if time_step <= 0.0 {
        window_length / 4.0
    } else {
        time_step
    };

    // LPC order: 2 × number of formants
    let lpc_order = 2 * max_num_formants;

    // Generate Gaussian window
    let window = gaussian_window(window_samples);

    // Frame timing - centered
    let n_frames = ((duration - physical_window_duration) / time_step).floor() as usize + 1;
    let n_frames = n_frames.max(1);
    let t1 = (duration - (n_frames - 1) as f64 * time_step) / 2.0;

    let mut frames = Vec::with_capacity(n_frames);

    for i in 0..n_frames {
        let t = t1 + i as f64 * time_step;

        // Extract frame in resampled signal
        let center_sample = (t * sample_rate).round() as isize;
        let start_sample = center_sample - half_window as isize;
        let end_sample = start_sample + window_samples as isize;

        // Handle boundaries
        let mut frame_samples = vec![0.0; window_samples];
        if start_sample < 0 || end_sample > pre_emphasized.len() as isize {
            let src_start = 0.max(start_sample) as usize;
            let src_end = (pre_emphasized.len() as isize).min(end_sample) as usize;
            let dst_start = (src_start as isize - start_sample) as usize;
            let dst_end = dst_start + (src_end - src_start);
            frame_samples[dst_start..dst_end].copy_from_slice(&pre_emphasized[src_start..src_end]);
        } else {
            let start = start_sample as usize;
            let end = end_sample as usize;
            frame_samples.copy_from_slice(&pre_emphasized[start..end]);
        }

        // Apply window
        let windowed: Vec<f64> = frame_samples
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect();

        // Compute LPC coefficients using Burg's algorithm
        let lpc_coeffs = burg_lpc(&windowed, lpc_order);

        // Find roots of LPC polynomial
        let roots = lpc_roots(&lpc_coeffs, true, true);

        // Convert roots to formant frequencies and bandwidths
        let mut formant_points = roots_to_formants(
            &roots,
            sample_rate,
            50.0,
            max_formant_hz - 50.0,
        );

        // Limit to max_num_formants
        formant_points.truncate(max_num_formants);

        frames.push(FormantFrame::new(t, formant_points));
    }

    Formant::new(frames, time_step, max_formant_hz, max_num_formants)
}
