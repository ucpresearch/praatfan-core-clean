//! Spectrum - Single-frame FFT magnitude spectrum.
//!
//! Documentation sources:
//! - Praat manual: Spectrum, Spectrum: Get centre of gravity...
//! - Standard FFT definition

use ndarray::Array1;
use rustfft::{num_complex::Complex, FftPlanner};

use crate::sound::Sound;

/// Single-frame FFT spectrum.
///
/// The spectrum stores complex values for frequencies from 0 to Nyquist.
/// Negative frequencies are not stored (conjugate symmetric for real signals).
#[derive(Debug, Clone)]
pub struct Spectrum {
    /// Real parts of spectrum bins.
    real: Array1<f64>,
    /// Imaginary parts of spectrum bins.
    imag: Array1<f64>,
    /// Frequency resolution (bin width) in Hz.
    df: f64,
    /// Maximum frequency (Nyquist) in Hz.
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
    #[inline]
    pub fn n_bins(&self) -> usize {
        self.real.len()
    }

    /// Get frequency for a bin index.
    #[inline]
    pub fn get_frequency(&self, bin_index: usize) -> f64 {
        bin_index as f64 * self.df
    }

    /// Compute center of gravity (spectral centroid).
    ///
    /// Formula (documented in Praat manual):
    ///     f_c = ∫ f × |S(f)|^p df / ∫ |S(f)|^p df
    ///
    /// # Arguments
    ///
    /// * `power` - Power to raise magnitude to (default 2.0)
    ///
    /// # Returns
    ///
    /// Center of gravity in Hz
    pub fn get_center_of_gravity(&self, power: f64) -> f64 {
        let n = self.n_bins();
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            let magnitude = (self.real[i].powi(2) + self.imag[i].powi(2)).sqrt();
            let weighted = magnitude.powf(power);
            let freq = i as f64 * self.df;

            numerator += freq * weighted;
            denominator += weighted;
        }

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Compute nth central moment.
    fn get_central_moment(&self, n: i32, power: f64) -> f64 {
        let n_bins = self.n_bins();
        let mut weighted_sum = 0.0;
        let mut freq_weighted_sum = 0.0;

        // First pass: compute CoG and denominator
        for i in 0..n_bins {
            let magnitude = (self.real[i].powi(2) + self.imag[i].powi(2)).sqrt();
            let weighted = magnitude.powf(power);
            let freq = i as f64 * self.df;

            freq_weighted_sum += freq * weighted;
            weighted_sum += weighted;
        }

        if weighted_sum == 0.0 {
            return 0.0;
        }

        let cog = freq_weighted_sum / weighted_sum;

        // Second pass: compute central moment
        let mut numerator = 0.0;
        for i in 0..n_bins {
            let magnitude = (self.real[i].powi(2) + self.imag[i].powi(2)).sqrt();
            let weighted = magnitude.powf(power);
            let freq = i as f64 * self.df;
            let deviation = freq - cog;

            numerator += deviation.powi(n) * weighted;
        }

        numerator / weighted_sum
    }

    /// Compute standard deviation of the spectrum.
    ///
    /// Formula: sqrt(μ₂) where μ₂ is the second central moment.
    ///
    /// # Arguments
    ///
    /// * `power` - Power to raise magnitude to (default 2.0)
    ///
    /// # Returns
    ///
    /// Standard deviation in Hz
    pub fn get_standard_deviation(&self, power: f64) -> f64 {
        let mu2 = self.get_central_moment(2, power);
        mu2.sqrt()
    }

    /// Compute skewness of the spectrum.
    ///
    /// Formula (documented in Praat manual): μ₃ / μ₂^1.5
    ///
    /// # Arguments
    ///
    /// * `power` - Power to raise magnitude to (default 2.0)
    ///
    /// # Returns
    ///
    /// Skewness (dimensionless)
    pub fn get_skewness(&self, power: f64) -> f64 {
        let mu2 = self.get_central_moment(2, power);
        let mu3 = self.get_central_moment(3, power);

        if mu2 == 0.0 {
            0.0
        } else {
            mu3 / mu2.powf(1.5)
        }
    }

    /// Compute kurtosis of the spectrum.
    ///
    /// Formula (documented in Praat manual): μ₄ / μ₂² - 3 (excess kurtosis)
    ///
    /// # Arguments
    ///
    /// * `power` - Power to raise magnitude to (default 2.0)
    ///
    /// # Returns
    ///
    /// Kurtosis (dimensionless)
    pub fn get_kurtosis(&self, power: f64) -> f64 {
        let mu2 = self.get_central_moment(2, power);
        let mu4 = self.get_central_moment(4, power);

        if mu2 == 0.0 {
            0.0
        } else {
            mu4 / mu2.powi(2) - 3.0
        }
    }

    /// Compute energy in a frequency band.
    ///
    /// Formula: E = ∫_{f_min}^{f_max} |S(f)|² df
    ///
    /// # Arguments
    ///
    /// * `f_min` - Minimum frequency (0 = DC)
    /// * `f_max` - Maximum frequency (0 = Nyquist)
    ///
    /// # Returns
    ///
    /// Band energy (Pa² s)
    pub fn get_band_energy(&self, f_min: f64, f_max: f64) -> f64 {
        let f_max = if f_max <= 0.0 { self.f_max } else { f_max };

        // Find bin indices for the frequency range
        let bin_min = (f_min / self.df).floor() as usize;
        let bin_max = (f_max / self.df).ceil() as usize;

        // Clamp to valid range
        let bin_min = bin_min.min(self.n_bins() - 1);
        let bin_max = bin_max.min(self.n_bins() - 1);

        let n_bins = self.n_bins();
        let mut energy = 0.0;

        for i in bin_min..=bin_max {
            let mag_squared = self.real[i].powi(2) + self.imag[i].powi(2);
            let bin_energy = mag_squared * self.df;

            // DC and Nyquist bins are not doubled
            if i == 0 || i == n_bins - 1 {
                energy += bin_energy;
            } else {
                // Other bins account for conjugate symmetric negative frequencies
                energy += 2.0 * bin_energy;
            }
        }

        energy
    }
}

/// Compute spectrum from sound.
///
/// Documented formula (Praat manual, standard DFT):
///     X[k] = Σₙ x[n] × e^(-2πikn/N) × Δt
///
/// The multiplication by Δt (sample period) converts the discrete sum
/// to an approximation of the continuous Fourier transform integral.
///
/// # Arguments
///
/// * `sound` - Sound object
/// * `fast` - If true, use power-of-2 FFT size for speed
///
/// # Returns
///
/// Spectrum object
pub fn sound_to_spectrum(sound: &Sound, fast: bool) -> Spectrum {
    let samples = sound.samples();
    let n_samples = samples.len();
    let sample_rate = sound.sample_rate();
    let dt = 1.0 / sample_rate;

    // Determine FFT size
    let fft_size = if fast {
        let mut size = 1;
        while size < n_samples {
            size *= 2;
        }
        size
    } else {
        n_samples
    };

    // Prepare input buffer (zero-padded)
    let mut buffer: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); fft_size];
    for (i, &sample) in samples.iter().enumerate() {
        buffer[i] = Complex::new(sample, 0.0);
    }

    // Compute FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    fft.process(&mut buffer);

    // Multiply by dt (documented: integral approximation)
    for c in buffer.iter_mut() {
        *c *= dt;
    }

    // Keep only positive frequencies (0 to Nyquist inclusive)
    let n_positive = fft_size / 2 + 1;
    let real: Vec<f64> = buffer[..n_positive].iter().map(|c| c.re).collect();
    let imag: Vec<f64> = buffer[..n_positive].iter().map(|c| c.im).collect();

    // Frequency resolution and maximum frequency
    let df = sample_rate / fft_size as f64;
    let f_max = sample_rate / 2.0;

    Spectrum::new(
        Array1::from_vec(real),
        Array1::from_vec(imag),
        df,
        f_max,
    )
}
