//! Python bindings for praatfan_rust using PyO3.
//!
//! This module provides Python bindings that are compatible with parselmouth's API,
//! allowing praatfan_rust to be used as a drop-in replacement in many cases.
//!
//! # Usage from Python
//!
//! ```python
//! import praatfan_rust
//!
//! # Load from file
//! sound = praatfan_rust.Sound("audio.wav")
//!
//! # Or create from numpy array
//! import numpy as np
//! samples = np.array([...], dtype=np.float64)
//! sound = praatfan_rust.Sound(samples, sampling_frequency=44100)
//!
//! # Analyze
//! pitch = sound.to_pitch_ac(time_step=0, pitch_floor=75, pitch_ceiling=600)
//! formant = sound.to_formant_burg(time_step=0, max_number_of_formants=5,
//!                                  maximum_formant=5500, window_length=0.025,
//!                                  pre_emphasis_from=50)
//!
//! # Access values (parselmouth-compatible)
//! times = pitch.xs()
//! frequencies = pitch.selected_array['frequency']
//! f1_values = formant.to_array(1)  # F1 frequencies
//! ```
//!
//! # Building
//!
//! ```bash
//! # Install maturin
//! pip install maturin
//!
//! # Build and install
//! maturin develop --features python
//!
//! # Build wheel
//! maturin build --features python --release
//! ```

use std::path::PathBuf;

use numpy::{PyArray1, PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::formant::Formant as RustFormant;
use crate::harmonicity::Harmonicity as RustHarmonicity;
use crate::intensity::Intensity as RustIntensity;
use crate::pitch::Pitch as RustPitch;
use crate::sound::Sound as RustSound;
use crate::spectrogram::Spectrogram as RustSpectrogram;
use crate::spectrum::Spectrum as RustSpectrum;

// ============================================================================
// Sound - Main audio type
// ============================================================================

/// Audio samples with sample rate.
///
/// This is the main type for acoustic analysis. Can be created from a file
/// path or from a numpy array of samples.
///
/// Examples
/// --------
/// >>> sound = praatfan.Sound("audio.wav")
/// >>> sound = praatfan.Sound(samples, sampling_frequency=44100)
#[pyclass(name = "Sound")]
pub struct PySound {
    inner: RustSound,
}

#[pymethods]
impl PySound {
    /// Create a Sound from a file path or numpy array.
    ///
    /// Parameters
    /// ----------
    /// path_or_samples : str or numpy.ndarray
    ///     Either a path to a WAV file, or a numpy array of samples
    /// sampling_frequency : float, optional
    ///     Sample rate in Hz (required if providing samples array)
    #[new]
    #[pyo3(signature = (path_or_samples, sampling_frequency=None))]
    fn new(
        py: Python<'_>,
        path_or_samples: PyObject,
        sampling_frequency: Option<f64>,
    ) -> PyResult<Self> {
        // Try to interpret as a path string first
        if let Ok(path_str) = path_or_samples.extract::<String>(py) {
            let path = PathBuf::from(&path_str);
            let sound = RustSound::from_file(&path).map_err(|e| {
                PyValueError::new_err(format!("Failed to load audio file: {}", e))
            })?;
            return Ok(PySound { inner: sound });
        }

        // Try to interpret as a numpy array
        if let Ok(arr) = path_or_samples.extract::<Bound<'_, PyArray1<f64>>>(py) {
            let sample_rate = sampling_frequency.ok_or_else(|| {
                PyValueError::new_err("sampling_frequency is required when providing samples array")
            })?;
            let samples: Vec<f64> = arr.to_vec()?;
            let sound = RustSound::from_slice(&samples, sample_rate);
            return Ok(PySound { inner: sound });
        }

        Err(PyValueError::new_err(
            "path_or_samples must be a file path string or numpy array",
        ))
    }

    /// Load a specific channel from a multi-channel audio file.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path to the WAV file
    /// channel : int
    ///     Channel index (0 = left, 1 = right, etc.)
    ///
    /// Returns
    /// -------
    /// Sound
    #[staticmethod]
    fn from_file_channel(path: &str, channel: usize) -> PyResult<Self> {
        let sound = RustSound::from_file_channel(path, channel).map_err(|e| {
            PyValueError::new_err(format!("Failed to load audio file: {}", e))
        })?;
        Ok(PySound { inner: sound })
    }

    /// Get the number of samples.
    #[getter]
    fn n_samples(&self) -> usize {
        self.inner.n_samples()
    }

    /// Get the sample rate in Hz.
    #[getter]
    fn sampling_frequency(&self) -> f64 {
        self.inner.sample_rate()
    }

    /// Get the duration in seconds.
    #[getter]
    fn duration(&self) -> f64 {
        self.inner.duration()
    }

    /// Get the audio samples as a numpy array.
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.samples().to_vec().to_pyarray(py)
    }

    /// Get time values for each sample.
    fn xs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let dt = self.inner.dx();
        let times: Vec<f64> = (0..self.inner.n_samples())
            .map(|i| (i as f64 + 0.5) * dt)
            .collect();
        times.to_pyarray(py)
    }

    // ========================================================================
    // Analysis methods - parselmouth-compatible names
    // ========================================================================

    /// Compute pitch contour using autocorrelation method.
    ///
    /// Parameters
    /// ----------
    /// time_step : float, optional
    ///     Time step in seconds (0 = auto, default)
    /// pitch_floor : float, optional
    ///     Minimum pitch in Hz (default: 75)
    /// pitch_ceiling : float, optional
    ///     Maximum pitch in Hz (default: 600)
    ///
    /// Returns
    /// -------
    /// Pitch
    #[pyo3(signature = (time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0))]
    fn to_pitch_ac(&self, time_step: f64, pitch_floor: f64, pitch_ceiling: f64) -> PyPitch {
        PyPitch {
            inner: self.inner.to_pitch_ac(time_step, pitch_floor, pitch_ceiling),
        }
    }

    /// Compute pitch contour using cross-correlation method.
    ///
    /// Parameters
    /// ----------
    /// time_step : float, optional
    ///     Time step in seconds (0 = auto, default)
    /// pitch_floor : float, optional
    ///     Minimum pitch in Hz (default: 75)
    /// pitch_ceiling : float, optional
    ///     Maximum pitch in Hz (default: 600)
    ///
    /// Returns
    /// -------
    /// Pitch
    #[pyo3(signature = (time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0))]
    fn to_pitch_cc(&self, time_step: f64, pitch_floor: f64, pitch_ceiling: f64) -> PyPitch {
        PyPitch {
            inner: self.inner.to_pitch_cc(time_step, pitch_floor, pitch_ceiling),
        }
    }

    /// Alias for to_pitch_ac (parselmouth compatibility).
    #[pyo3(signature = (time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0))]
    fn to_pitch(&self, time_step: f64, pitch_floor: f64, pitch_ceiling: f64) -> PyPitch {
        self.to_pitch_ac(time_step, pitch_floor, pitch_ceiling)
    }

    /// Compute formants using Burg's LPC method.
    ///
    /// Parameters
    /// ----------
    /// time_step : float, optional
    ///     Time step in seconds (0 = auto, default)
    /// max_number_of_formants : int, optional
    ///     Maximum number of formants (default: 5)
    /// maximum_formant : float, optional
    ///     Maximum formant frequency in Hz (default: 5500)
    /// window_length : float, optional
    ///     Window length in seconds (default: 0.025)
    /// pre_emphasis_from : float, optional
    ///     Pre-emphasis frequency in Hz (default: 50)
    ///
    /// Returns
    /// -------
    /// Formant
    #[pyo3(signature = (time_step=0.0, max_number_of_formants=5, maximum_formant=5500.0, window_length=0.025, pre_emphasis_from=50.0))]
    fn to_formant_burg(
        &self,
        time_step: f64,
        max_number_of_formants: usize,
        maximum_formant: f64,
        window_length: f64,
        pre_emphasis_from: f64,
    ) -> PyFormant {
        PyFormant {
            inner: self.inner.to_formant_burg(
                time_step,
                max_number_of_formants,
                maximum_formant,
                window_length,
                pre_emphasis_from,
            ),
        }
    }

    /// Compute intensity contour.
    ///
    /// Parameters
    /// ----------
    /// minimum_pitch : float, optional
    ///     Minimum pitch in Hz (default: 100)
    /// time_step : float, optional
    ///     Time step in seconds (0 = auto, default)
    ///
    /// Returns
    /// -------
    /// Intensity
    #[pyo3(signature = (minimum_pitch=100.0, time_step=0.0))]
    fn to_intensity(&self, minimum_pitch: f64, time_step: f64) -> PyIntensity {
        PyIntensity {
            inner: self.inner.to_intensity(minimum_pitch, time_step),
        }
    }

    /// Compute harmonicity (HNR) using autocorrelation method.
    ///
    /// Parameters
    /// ----------
    /// time_step : float, optional
    ///     Time step in seconds (default: 0.01)
    /// minimum_pitch : float, optional
    ///     Minimum pitch in Hz (default: 75)
    /// silence_threshold : float, optional
    ///     Silence threshold (default: 0.1)
    /// periods_per_window : float, optional
    ///     Number of periods per window (default: 4.5)
    ///
    /// Returns
    /// -------
    /// Harmonicity
    #[pyo3(signature = (time_step=0.01, minimum_pitch=75.0, silence_threshold=0.1, periods_per_window=4.5))]
    fn to_harmonicity_ac(
        &self,
        time_step: f64,
        minimum_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> PyHarmonicity {
        PyHarmonicity {
            inner: self.inner.to_harmonicity_ac(
                time_step,
                minimum_pitch,
                silence_threshold,
                periods_per_window,
            ),
        }
    }

    /// Compute harmonicity (HNR) using cross-correlation method.
    #[pyo3(signature = (time_step=0.01, minimum_pitch=75.0, silence_threshold=0.1, periods_per_window=1.0))]
    fn to_harmonicity_cc(
        &self,
        time_step: f64,
        minimum_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> PyHarmonicity {
        PyHarmonicity {
            inner: self.inner.to_harmonicity_cc(
                time_step,
                minimum_pitch,
                silence_threshold,
                periods_per_window,
            ),
        }
    }

    /// Alias for to_harmonicity_ac (parselmouth compatibility).
    #[pyo3(signature = (time_step=0.01, minimum_pitch=75.0, silence_threshold=0.1, periods_per_window=4.5))]
    fn to_harmonicity(
        &self,
        time_step: f64,
        minimum_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> PyHarmonicity {
        self.to_harmonicity_ac(time_step, minimum_pitch, silence_threshold, periods_per_window)
    }

    /// Compute spectrum (single-frame FFT).
    ///
    /// Parameters
    /// ----------
    /// fast : bool, optional
    ///     Use power-of-2 FFT size (default: True)
    ///
    /// Returns
    /// -------
    /// Spectrum
    #[pyo3(signature = (fast=true))]
    fn to_spectrum(&self, fast: bool) -> PySpectrum {
        PySpectrum {
            inner: self.inner.to_spectrum(fast),
        }
    }

    /// Compute spectrogram.
    ///
    /// Parameters
    /// ----------
    /// window_length : float, optional
    ///     Window length in seconds (default: 0.005)
    /// maximum_frequency : float, optional
    ///     Maximum frequency in Hz (default: 5000)
    /// time_step : float, optional
    ///     Time step in seconds (default: 0.002)
    /// frequency_step : float, optional
    ///     Frequency step in Hz (default: 20)
    ///
    /// Returns
    /// -------
    /// Spectrogram
    #[pyo3(signature = (window_length=0.005, maximum_frequency=5000.0, time_step=0.002, frequency_step=20.0))]
    fn to_spectrogram(
        &self,
        window_length: f64,
        maximum_frequency: f64,
        time_step: f64,
        frequency_step: f64,
    ) -> PySpectrogram {
        PySpectrogram {
            inner: self.inner.to_spectrogram(
                window_length,
                maximum_frequency,
                time_step,
                frequency_step,
            ),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Sound({} samples, {} Hz, {:.3}s)",
            self.inner.n_samples(),
            self.inner.sample_rate(),
            self.inner.duration()
        )
    }
}

// ============================================================================
// Pitch - Pitch analysis result
// ============================================================================

/// Pitch (F0) analysis result.
///
/// Contains fundamental frequency estimates for each time frame.
#[pyclass(name = "Pitch")]
pub struct PyPitch {
    inner: RustPitch,
}

#[pymethods]
impl PyPitch {
    /// Get the number of frames.
    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.n_frames()
    }

    /// Get the time step between frames.
    #[getter]
    fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    /// Get time values for all frames.
    fn xs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.times().to_vec().to_pyarray(py)
    }

    /// Get F0 values for all frames.
    ///
    /// Returns NaN for unvoiced frames.
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values: Vec<f64> = self
            .inner
            .values()
            .iter()
            .map(|&v| if v > 0.0 { v } else { f64::NAN })
            .collect();
        values.to_pyarray(py)
    }

    /// Get selected array (parselmouth compatibility).
    ///
    /// Returns a dict with 'frequency' and 'strength' keys.
    #[getter]
    fn selected_array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let frequencies: Vec<f64> = self
            .inner
            .values()
            .iter()
            .map(|&v| if v > 0.0 { v } else { f64::NAN })
            .collect();
        let strengths: Vec<f64> = self.inner.strengths().to_vec();

        dict.set_item("frequency", frequencies.to_pyarray(py))?;
        dict.set_item("strength", strengths.to_pyarray(py))?;
        Ok(dict)
    }

    /// Get pitch value at a specific time.
    ///
    /// Parameters
    /// ----------
    /// time : float
    ///     Time in seconds
    ///
    /// Returns
    /// -------
    /// float or None
    fn get_value_at_time(&self, time: f64) -> Option<f64> {
        self.inner.get_value_at_time(time, "linear")
    }

    /// Get pitch strength at a specific time.
    fn get_strength_at_time(&self, time: f64) -> Option<f64> {
        self.inner.get_strength_at_time(time, "linear")
    }

    /// Get value at a specific frame (0-indexed).
    fn get_value_in_frame(&self, frame: usize) -> f64 {
        if frame < self.inner.n_frames() {
            let freq = self.inner.frames()[frame].frequency();
            if freq > 0.0 {
                freq
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        }
    }

    /// Convert to numpy array of F0 values.
    fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.values(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "Pitch({} frames, {:.3}s step, {:.0}-{:.0} Hz)",
            self.inner.n_frames(),
            self.inner.time_step(),
            self.inner.pitch_floor(),
            self.inner.pitch_ceiling()
        )
    }
}

// ============================================================================
// Formant - Formant analysis result
// ============================================================================

/// Formant analysis result.
///
/// Contains formant frequencies and bandwidths for each time frame.
#[pyclass(name = "Formant")]
pub struct PyFormant {
    inner: RustFormant,
}

#[pymethods]
impl PyFormant {
    /// Get the number of frames.
    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.n_frames()
    }

    /// Get the time step between frames.
    #[getter]
    fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    /// Get time values for all frames.
    fn xs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.times().to_vec().to_pyarray(py)
    }

    /// Get formant values for a specific formant number.
    ///
    /// Parameters
    /// ----------
    /// formant_number : int
    ///     Formant number (1 = F1, 2 = F2, etc.)
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    fn to_array<'py>(&self, py: Python<'py>, formant_number: usize) -> Bound<'py, PyArray1<f64>> {
        self.inner.formant_values(formant_number).to_vec().to_pyarray(py)
    }

    /// Get bandwidth values for a specific formant number.
    fn to_bandwidth_array<'py>(
        &self,
        py: Python<'py>,
        formant_number: usize,
    ) -> Bound<'py, PyArray1<f64>> {
        self.inner
            .bandwidth_values(formant_number)
            .to_vec()
            .to_pyarray(py)
    }

    /// Get formant value at a specific time.
    ///
    /// Parameters
    /// ----------
    /// formant_number : int
    ///     Formant number (1 = F1, 2 = F2, etc.)
    /// time : float
    ///     Time in seconds
    ///
    /// Returns
    /// -------
    /// float or None
    fn get_value_at_time(&self, formant_number: usize, time: f64) -> Option<f64> {
        self.inner.get_value_at_time(formant_number, time, "linear")
    }

    /// Get bandwidth at a specific time.
    fn get_bandwidth_at_time(&self, formant_number: usize, time: f64) -> Option<f64> {
        self.inner
            .get_bandwidth_at_time(formant_number, time, "linear")
    }

    /// Get formant value at a specific frame (0-indexed).
    fn get_value_in_frame(&self, formant_number: usize, frame: usize) -> f64 {
        if frame < self.inner.n_frames() {
            self.inner.frames()[frame]
                .get_formant(formant_number)
                .map(|f| f.frequency)
                .unwrap_or(f64::NAN)
        } else {
            f64::NAN
        }
    }

    /// Get bandwidth at a specific frame.
    fn get_bandwidth_in_frame(&self, formant_number: usize, frame: usize) -> f64 {
        if frame < self.inner.n_frames() {
            self.inner.frames()[frame]
                .get_formant(formant_number)
                .map(|f| f.bandwidth)
                .unwrap_or(f64::NAN)
        } else {
            f64::NAN
        }
    }

    /// Get the number of formants in a specific frame.
    fn get_number_of_formants(&self, frame: usize) -> usize {
        if frame < self.inner.n_frames() {
            self.inner.frames()[frame].n_formants()
        } else {
            0
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Formant({} frames, {:.3}s step, max {} formants)",
            self.inner.n_frames(),
            self.inner.time_step(),
            self.inner.max_num_formants()
        )
    }
}

// ============================================================================
// Intensity - Intensity analysis result
// ============================================================================

/// Intensity analysis result.
///
/// Contains RMS energy values in dB for each time frame.
#[pyclass(name = "Intensity")]
pub struct PyIntensity {
    inner: RustIntensity,
}

#[pymethods]
impl PyIntensity {
    /// Get the number of frames.
    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.n_frames()
    }

    /// Get the time step between frames.
    #[getter]
    fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    /// Get time values for all frames.
    fn xs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.times().to_vec().to_pyarray(py)
    }

    /// Get intensity values in dB.
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.values().to_vec().to_pyarray(py)
    }

    /// Convert to numpy array of intensity values.
    fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.values(py)
    }

    /// Get intensity value at a specific time.
    fn get_value_at_time(&self, time: f64) -> Option<f64> {
        self.inner
            .get_value_at_time(time, crate::intensity::Interpolation::Linear)
    }

    /// Get minimum intensity.
    fn get_minimum(&self) -> f64 {
        self.inner
            .values()
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Get maximum intensity.
    fn get_maximum(&self) -> f64 {
        self.inner
            .values()
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get mean intensity.
    fn get_mean(&self) -> f64 {
        let values = self.inner.values();
        if values.is_empty() {
            f64::NAN
        } else {
            values.sum() / values.len() as f64
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Intensity({} frames, {:.3}s step)",
            self.inner.n_frames(),
            self.inner.time_step()
        )
    }
}

// ============================================================================
// Harmonicity - Harmonicity analysis result
// ============================================================================

/// Harmonicity (HNR) analysis result.
///
/// Contains harmonics-to-noise ratio values in dB for each time frame.
#[pyclass(name = "Harmonicity")]
pub struct PyHarmonicity {
    inner: RustHarmonicity,
}

#[pymethods]
impl PyHarmonicity {
    /// Get the number of frames.
    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.n_frames()
    }

    /// Get the time step between frames.
    #[getter]
    fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    /// Get time values for all frames.
    fn xs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.times().to_vec().to_pyarray(py)
    }

    /// Get HNR values in dB.
    ///
    /// Returns -200 for unvoiced/silent frames.
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.values().to_vec().to_pyarray(py)
    }

    /// Convert to numpy array of HNR values.
    fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.values(py)
    }

    /// Get HNR value at a specific time.
    fn get_value_at_time(&self, time: f64) -> Option<f64> {
        self.inner
            .get_value_at_time(time, crate::harmonicity::HarmonicityInterpolation::Linear)
    }

    fn __repr__(&self) -> String {
        format!(
            "Harmonicity({} frames, {:.3}s step)",
            self.inner.n_frames(),
            self.inner.time_step()
        )
    }
}

// ============================================================================
// Spectrum - Spectrum analysis result
// ============================================================================

/// Spectrum (single-frame FFT) analysis result.
#[pyclass(name = "Spectrum")]
pub struct PySpectrum {
    inner: RustSpectrum,
}

#[pymethods]
impl PySpectrum {
    /// Get the number of frequency bins.
    #[getter]
    fn n_bins(&self) -> usize {
        self.inner.n_bins()
    }

    /// Get the frequency resolution in Hz.
    #[getter]
    fn df(&self) -> f64 {
        self.inner.df()
    }

    /// Get the maximum frequency in Hz.
    #[getter]
    fn maximum_frequency(&self) -> f64 {
        self.inner.f_max()
    }

    /// Get frequency values for all bins.
    fn xs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let freqs: Vec<f64> = (0..self.inner.n_bins())
            .map(|i| self.inner.get_frequency(i))
            .collect();
        freqs.to_pyarray(py)
    }

    /// Get real parts of the spectrum.
    fn real<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.real().to_vec().to_pyarray(py)
    }

    /// Get imaginary parts of the spectrum.
    fn imag<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.imag().to_vec().to_pyarray(py)
    }

    /// Get center of gravity (spectral centroid).
    #[pyo3(signature = (power=2.0))]
    fn get_centre_of_gravity(&self, power: f64) -> f64 {
        self.inner.get_center_of_gravity(power)
    }

    /// Alias for get_centre_of_gravity (American spelling).
    #[pyo3(signature = (power=2.0))]
    fn get_center_of_gravity(&self, power: f64) -> f64 {
        self.inner.get_center_of_gravity(power)
    }

    /// Get standard deviation.
    #[pyo3(signature = (power=2.0))]
    fn get_standard_deviation(&self, power: f64) -> f64 {
        self.inner.get_standard_deviation(power)
    }

    /// Get skewness.
    #[pyo3(signature = (power=2.0))]
    fn get_skewness(&self, power: f64) -> f64 {
        self.inner.get_skewness(power)
    }

    /// Get kurtosis.
    #[pyo3(signature = (power=2.0))]
    fn get_kurtosis(&self, power: f64) -> f64 {
        self.inner.get_kurtosis(power)
    }

    /// Get band energy.
    #[pyo3(signature = (band_floor=0.0, band_ceiling=0.0))]
    fn get_band_energy(&self, band_floor: f64, band_ceiling: f64) -> f64 {
        self.inner.get_band_energy(band_floor, band_ceiling)
    }

    fn __repr__(&self) -> String {
        format!(
            "Spectrum({} bins, {:.1} Hz resolution, 0-{:.0} Hz)",
            self.inner.n_bins(),
            self.inner.df(),
            self.inner.f_max()
        )
    }
}

// ============================================================================
// Spectrogram - Spectrogram analysis result
// ============================================================================

/// Spectrogram (time-frequency) analysis result.
#[pyclass(name = "Spectrogram")]
pub struct PySpectrogram {
    inner: RustSpectrogram,
}

#[pymethods]
impl PySpectrogram {
    /// Get the number of time frames.
    #[getter]
    fn n_times(&self) -> usize {
        self.inner.n_times()
    }

    /// Get the number of frequency bins.
    #[getter]
    fn n_freqs(&self) -> usize {
        self.inner.n_freqs()
    }

    /// Get the time step.
    #[getter]
    fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    /// Get the frequency step.
    #[getter]
    fn frequency_step(&self) -> f64 {
        self.inner.freq_step()
    }

    /// Get time values for all frames.
    fn xs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.times().to_pyarray(py)
    }

    /// Get frequency values for all bins.
    fn ys<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.frequencies().to_pyarray(py)
    }

    /// Get all power values as a 1D array (row-major: freq × time).
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values: Vec<f64> = self.inner.values().iter().copied().collect();
        values.to_pyarray(py)
    }

    /// Get power value at a specific time and frequency.
    fn get_power_at(&self, time: f64, frequency: f64) -> f64 {
        // Find nearest frame index using first frame time (t1), not time_min
        let first_frame_time = self.inner.get_time_from_frame(0);
        let t_idx = ((time - first_frame_time) / self.inner.time_step()).round() as isize;
        let f_idx = ((frequency - self.inner.freq_min()) / self.inner.freq_step()).round() as isize;

        if t_idx >= 0
            && (t_idx as usize) < self.inner.n_times()
            && f_idx >= 0
            && (f_idx as usize) < self.inner.n_freqs()
        {
            self.inner.values()[[f_idx as usize, t_idx as usize]]
        } else {
            f64::NAN
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Spectrogram({} times × {} freqs, {:.3}s step, 0-{:.0} Hz)",
            self.inner.n_times(),
            self.inner.n_freqs(),
            self.inner.time_step(),
            self.inner.freq_max()
        )
    }
}

// ============================================================================
// Module definition
// ============================================================================

/// praatfan_rust - Clean-room reimplementation of Praat's acoustic analysis algorithms.
///
/// This module provides Python bindings compatible with parselmouth's API.
#[pymodule]
fn praatfan_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySound>()?;
    m.add_class::<PyPitch>()?;
    m.add_class::<PyFormant>()?;
    m.add_class::<PyIntensity>()?;
    m.add_class::<PyHarmonicity>()?;
    m.add_class::<PySpectrum>()?;
    m.add_class::<PySpectrogram>()?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
