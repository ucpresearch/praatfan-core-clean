//! Spectrum - Single-frame FFT magnitude spectrum.
//!
//! This module implements single-frame Fourier analysis of audio signals.
//! For time-varying spectral analysis, see the spectrogram module.
//!
//! # Documentation Sources
//!
//! - Praat manual: Spectrum, Spectrum: Get centre of gravity...
//! - Standard DFT definition from signal processing textbooks
//!
//! # Clean Room Implementation
//!
//! The Fourier transform is a standard mathematical operation. Our implementation
//! uses the rustfft library for efficient FFT computation, with the Praat-documented
//! scaling by dt (sample period) to convert from discrete to continuous transform.
//!
//! # Spectral Moments
//!
//! The spectral moment formulas are fully documented in the Praat manual and
//! are standard definitions used in acoustics:
//!
//! - **Center of Gravity**: First spectral moment (weighted mean frequency)
//! - **Standard Deviation**: Square root of second central moment
//! - **Skewness**: Third central moment normalized by std^1.5
//! - **Kurtosis**: Fourth central moment normalized by std^2, minus 3

use ndarray::Array1;
use rustfft::{num_complex::Complex, FftPlanner};

use crate::sound::Sound;

/// Single-frame FFT spectrum.
///
/// The spectrum stores complex values for frequencies from 0 (DC) to Nyquist.
/// Negative frequencies are not stored since they are conjugate symmetric
/// for real-valued input signals.
///
/// # Frequency Indexing
///
/// - Bin 0: DC component (0 Hz)
/// - Bin k: frequency = k × df Hz
/// - Bin N/2: Nyquist frequency (sample_rate / 2)
///
/// where df = sample_rate / FFT_size is the frequency resolution.
#[derive(Debug, Clone)]
pub struct Spectrum {
    /// Real parts of spectrum bins.
    ///
    /// For a real-valued input signal, these are symmetric around the Nyquist
    /// frequency in the full FFT output.
    real: Array1<f64>,

    /// Imaginary parts of spectrum bins.
    ///
    /// For a real-valued input signal, these are antisymmetric around the
    /// Nyquist frequency in the full FFT output.
    imag: Array1<f64>,

    /// Frequency resolution (bin width) in Hz.
    ///
    /// df = sample_rate / FFT_size
    ///
    /// Smaller df means better frequency resolution but requires longer
    /// analysis windows (more samples).
    df: f64,

    /// Maximum frequency (Nyquist) in Hz.
    ///
    /// f_max = sample_rate / 2
    ///
    /// This is the highest frequency that can be represented without aliasing.
    f_max: f64,
}

impl Spectrum {
    /// Create a new Spectrum.
    ///
    /// # Arguments
    ///
    /// * `real` - Real parts of spectrum bins
    /// * `imag` - Imaginary parts of spectrum bins
    /// * `df` - Frequency resolution in Hz
    /// * `f_max` - Maximum frequency in Hz
    pub fn new(real: Array1<f64>, imag: Array1<f64>, df: f64, f_max: f64) -> Self {
        Self { real, imag, df, f_max }
    }

    /// Get the real parts of spectrum bins.
    #[inline]
    pub fn real(&self) -> &Array1<f64> {
        &self.real
    }

    /// Get the imaginary parts of spectrum bins.
    #[inline]
    pub fn imag(&self) -> &Array1<f64> {
        &self.imag
    }

    /// Get the frequency resolution (bin width) in Hz.
    #[inline]
    pub fn df(&self) -> f64 {
        self.df
    }

    /// Get the maximum frequency (Nyquist) in Hz.
    #[inline]
    pub fn f_max(&self) -> f64 {
        self.f_max
    }

    /// Get the number of frequency bins.
    ///
    /// This is FFT_size / 2 + 1, which includes both DC and Nyquist bins.
    #[inline]
    pub fn n_bins(&self) -> usize {
        self.real.len()
    }

    /// Get frequency for a bin index.
    ///
    /// Frequency = bin_index × df
    ///
    /// # Arguments
    ///
    /// * `bin_index` - Index of the frequency bin (0 = DC)
    #[inline]
    pub fn get_frequency(&self, bin_index: usize) -> f64 {
        bin_index as f64 * self.df
    }

    /// Compute center of gravity (spectral centroid).
    ///
    /// The center of gravity is the first spectral moment - a weighted average
    /// of frequencies where the weights are the spectral magnitudes raised to
    /// a power.
    ///
    /// # Formula (from Praat manual)
    ///
    /// ```text
    /// f_c = ∫ f × |S(f)|^p df / ∫ |S(f)|^p df
    /// ```
    ///
    /// This is discretized as:
    /// ```text
    /// f_c = Σ f_k × |S_k|^p / Σ |S_k|^p
    /// ```
    ///
    /// # Arguments
    ///
    /// * `power` - Power to raise magnitude to (default 2.0 for power spectrum)
    ///
    /// # Returns
    ///
    /// Center of gravity in Hz. Higher values indicate a "brighter" sound
    /// with more high-frequency energy.
    pub fn get_center_of_gravity(&self, power: f64) -> f64 {
        let n = self.n_bins();
        let mut numerator = 0.0;    // Σ f × |S|^p
        let mut denominator = 0.0;  // Σ |S|^p

        // Iterate over all frequency bins
        for i in 0..n {
            // Compute magnitude: |S| = sqrt(Re² + Im²)
            let magnitude = (self.real[i].powi(2) + self.imag[i].powi(2)).sqrt();

            // Raise to power p: |S|^p
            let weighted = magnitude.powf(power);

            // Frequency for this bin
            let freq = i as f64 * self.df;

            // Accumulate weighted sums
            numerator += freq * weighted;
            denominator += weighted;
        }

        // Avoid division by zero for silent signals
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Compute nth central moment of the spectrum.
    ///
    /// Central moments measure the shape of the spectral distribution
    /// around its center of gravity.
    ///
    /// # Formula
    ///
    /// ```text
    /// μ_n = Σ (f_k - f_c)^n × |S_k|^p / Σ |S_k|^p
    /// ```
    ///
    /// where f_c is the center of gravity.
    ///
    /// # Arguments
    ///
    /// * `n` - Order of the moment (2 for variance, 3 for skewness, 4 for kurtosis)
    /// * `power` - Power to raise magnitude to
    fn get_central_moment(&self, n: i32, power: f64) -> f64 {
        let n_bins = self.n_bins();
        let mut weighted_sum = 0.0;      // Σ |S|^p (denominator)
        let mut freq_weighted_sum = 0.0; // Σ f × |S|^p (for computing CoG)

        // First pass: compute center of gravity and total weighted sum
        for i in 0..n_bins {
            let magnitude = (self.real[i].powi(2) + self.imag[i].powi(2)).sqrt();
            let weighted = magnitude.powf(power);
            let freq = i as f64 * self.df;

            freq_weighted_sum += freq * weighted;
            weighted_sum += weighted;
        }

        // Avoid division by zero
        if weighted_sum == 0.0 {
            return 0.0;
        }

        // Center of gravity (mean frequency)
        let cog = freq_weighted_sum / weighted_sum;

        // Second pass: compute central moment Σ (f - μ)^n × w / Σ w
        let mut numerator = 0.0;
        for i in 0..n_bins {
            let magnitude = (self.real[i].powi(2) + self.imag[i].powi(2)).sqrt();
            let weighted = magnitude.powf(power);
            let freq = i as f64 * self.df;

            // Deviation from center of gravity
            let deviation = freq - cog;

            // Accumulate: (f - CoG)^n × weight
            numerator += deviation.powi(n) * weighted;
        }

        numerator / weighted_sum
    }

    /// Compute standard deviation of the spectrum.
    ///
    /// The standard deviation is the square root of the second central moment
    /// (variance). It measures the spread of spectral energy around the center
    /// of gravity.
    ///
    /// # Formula
    ///
    /// ```text
    /// σ = sqrt(μ₂)
    /// ```
    ///
    /// where μ₂ is the second central moment.
    ///
    /// # Arguments
    ///
    /// * `power` - Power to raise magnitude to (default 2.0)
    ///
    /// # Returns
    ///
    /// Standard deviation in Hz. Higher values indicate more spread-out
    /// spectral energy.
    pub fn get_standard_deviation(&self, power: f64) -> f64 {
        let mu2 = self.get_central_moment(2, power);
        mu2.sqrt()
    }

    /// Compute skewness of the spectrum.
    ///
    /// Skewness measures the asymmetry of the spectral distribution around
    /// its center of gravity. Positive skewness indicates more energy on the
    /// high-frequency side; negative skewness indicates more on the low side.
    ///
    /// # Formula (from Praat manual)
    ///
    /// ```text
    /// skewness = μ₃ / μ₂^1.5
    /// ```
    ///
    /// where μ₂ and μ₃ are the second and third central moments.
    ///
    /// # Arguments
    ///
    /// * `power` - Power to raise magnitude to (default 2.0)
    ///
    /// # Returns
    ///
    /// Skewness (dimensionless). Zero for symmetric distributions.
    pub fn get_skewness(&self, power: f64) -> f64 {
        let mu2 = self.get_central_moment(2, power);
        let mu3 = self.get_central_moment(3, power);

        if mu2 == 0.0 {
            0.0
        } else {
            // Normalize by σ^3 = (μ₂^0.5)^3 = μ₂^1.5
            mu3 / mu2.powf(1.5)
        }
    }

    /// Compute kurtosis of the spectrum.
    ///
    /// Kurtosis measures the "tailedness" of the spectral distribution.
    /// Higher kurtosis indicates more of the variance comes from extreme
    /// deviations (heavy tails).
    ///
    /// # Formula (from Praat manual)
    ///
    /// This computes **excess kurtosis** (kurtosis - 3), which is zero for
    /// a Gaussian distribution:
    ///
    /// ```text
    /// kurtosis = μ₄ / μ₂² - 3
    /// ```
    ///
    /// where μ₂ and μ₄ are the second and fourth central moments.
    ///
    /// # Arguments
    ///
    /// * `power` - Power to raise magnitude to (default 2.0)
    ///
    /// # Returns
    ///
    /// Excess kurtosis (dimensionless). Zero for Gaussian, positive for
    /// heavy-tailed distributions.
    pub fn get_kurtosis(&self, power: f64) -> f64 {
        let mu2 = self.get_central_moment(2, power);
        let mu4 = self.get_central_moment(4, power);

        if mu2 == 0.0 {
            0.0
        } else {
            // Excess kurtosis: μ₄/σ⁴ - 3 = μ₄/μ₂² - 3
            mu4 / mu2.powi(2) - 3.0
        }
    }

    /// Compute energy in a frequency band.
    ///
    /// Band energy is the integral of squared magnitude over a frequency range,
    /// which corresponds to Parseval's theorem relating time-domain and
    /// frequency-domain energy.
    ///
    /// # Formula
    ///
    /// ```text
    /// E = ∫_{f_min}^{f_max} |S(f)|² df
    /// ```
    ///
    /// Discretized as sum over bins, with df as the bin width.
    ///
    /// # Note on Symmetric Frequencies
    ///
    /// For real-valued signals, positive and negative frequencies have
    /// equal energy. Since we only store positive frequencies, we double
    /// the energy for non-DC, non-Nyquist bins to account for the missing
    /// negative frequencies.
    ///
    /// # Arguments
    ///
    /// * `f_min` - Minimum frequency (0 = DC)
    /// * `f_max` - Maximum frequency (0 = use Nyquist)
    ///
    /// # Returns
    ///
    /// Band energy in Pa² s (pressure squared times time)
    pub fn get_band_energy(&self, f_min: f64, f_max: f64) -> f64 {
        // Default f_max to Nyquist if not specified
        let f_max = if f_max <= 0.0 { self.f_max } else { f_max };

        // Find bin indices for the frequency range
        let bin_min = (f_min / self.df).floor() as usize;
        let bin_max = (f_max / self.df).ceil() as usize;

        // Clamp to valid range
        let bin_min = bin_min.min(self.n_bins() - 1);
        let bin_max = bin_max.min(self.n_bins() - 1);

        let n_bins = self.n_bins();
        let mut energy = 0.0;

        // Sum energy across bins in range
        for i in bin_min..=bin_max {
            // Magnitude squared: |S|² = Re² + Im²
            let mag_squared = self.real[i].powi(2) + self.imag[i].powi(2);

            // Energy contribution: |S|² × df
            let bin_energy = mag_squared * self.df;

            // DC (bin 0) and Nyquist (bin N/2) are not doubled because they
            // don't have a symmetric negative frequency counterpart
            if i == 0 || i == n_bins - 1 {
                energy += bin_energy;
            } else {
                // All other bins: double to account for conjugate symmetric
                // negative frequencies that we don't store
                energy += 2.0 * bin_energy;
            }
        }

        energy
    }
}

/// Compute spectrum from sound.
///
/// This function computes the single-sided Fourier transform of the entire
/// sound, scaled to approximate the continuous Fourier transform.
///
/// # Algorithm
///
/// 1. Optionally zero-pad to power-of-2 length for efficient FFT
/// 2. Compute forward FFT
/// 3. Scale by dt (sample period) to approximate continuous transform
/// 4. Keep only positive frequencies (0 to Nyquist)
///
/// # Scaling Factor
///
/// The DFT computes:
/// ```text
/// X[k] = Σₙ x[n] × e^(-2πikn/N)
/// ```
///
/// To approximate the continuous Fourier transform:
/// ```text
/// X(f) = ∫ x(t) × e^(-2πift) dt
/// ```
///
/// We multiply by dt (the sample period), converting the discrete sum
/// to a Riemann sum approximation of the integral:
/// ```text
/// X[k] × dt ≈ X(f_k)
/// ```
///
/// This is documented in the Praat manual and is standard practice for
/// relating DFT to continuous transforms.
///
/// # Arguments
///
/// * `sound` - Sound object
/// * `fast` - If true, use power-of-2 FFT size for speed
///
/// # Returns
///
/// Spectrum object containing positive frequencies (DC to Nyquist)
pub fn sound_to_spectrum(sound: &Sound, fast: bool) -> Spectrum {
    let samples = sound.samples();
    let n_samples = samples.len();
    let sample_rate = sound.sample_rate();

    // dt = sample period = 1/sample_rate
    // This is the scaling factor for continuous transform approximation
    let dt = 1.0 / sample_rate;

    // Determine FFT size
    let fft_size = if fast {
        // Find smallest power of 2 >= n_samples for efficient FFT
        // Powers of 2 allow the Cooley-Tukey algorithm to be used optimally
        let mut size = 1;
        while size < n_samples {
            size *= 2;
        }
        size
    } else {
        // Use exact sample count (may result in slower FFT)
        n_samples
    };

    // Prepare input buffer with zero-padding if needed
    // Zero-padding provides interpolation in the frequency domain
    let mut buffer: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); fft_size];
    for (i, &sample) in samples.iter().enumerate() {
        buffer[i] = Complex::new(sample, 0.0);
    }

    // Compute forward FFT using rustfft
    // The FFT computes X[k] = Σ x[n] × e^(-2πikn/N)
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    fft.process(&mut buffer);

    // Multiply by dt (documented in Praat manual)
    // This converts DFT to continuous Fourier transform approximation
    for c in buffer.iter_mut() {
        *c *= dt;
    }

    // Keep only positive frequencies (0 to Nyquist inclusive)
    // For real input, negative frequencies are conjugate symmetric:
    // X[-k] = X[k]*, so we don't need to store them
    let n_positive = fft_size / 2 + 1;
    let real: Vec<f64> = buffer[..n_positive].iter().map(|c| c.re).collect();
    let imag: Vec<f64> = buffer[..n_positive].iter().map(|c| c.im).collect();

    // Frequency resolution: df = sample_rate / FFT_size
    // This is the frequency spacing between adjacent bins
    let df = sample_rate / fft_size as f64;

    // Maximum frequency: f_max = sample_rate / 2 (Nyquist)
    let f_max = sample_rate / 2.0;

    Spectrum::new(
        Array1::from_vec(real),
        Array1::from_vec(imag),
        df,
        f_max,
    )
}
