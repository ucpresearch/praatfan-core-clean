//! WASM bindings for praatfan.
//!
//! This module provides JavaScript bindings for all acoustic analysis functions
//! using wasm-bindgen. It enables praatfan to run in web browsers and Node.js.
//!
//! # Usage from JavaScript
//!
//! ```javascript
//! import init, { Sound } from './pkg/praatfan.js';
//!
//! await init();
//!
//! // Create from raw samples (Float64Array)
//! const samples = new Float64Array([...]);
//! const sound = new Sound(samples, 44100);
//!
//! // Analyze
//! const pitch = sound.to_pitch_ac(0, 75, 600);
//! const formant = sound.to_formant_burg(0, 5, 5500, 0.025, 50);
//!
//! // Get values as typed arrays
//! const f0Values = pitch.values();          // Float64Array
//! const f1Values = formant.formant_values(1); // Float64Array
//! ```
//!
//! # Building for WASM
//!
//! ```bash
//! # Install wasm-pack
//! cargo install wasm-pack
//!
//! # Build for web
//! wasm-pack build --target web --features wasm
//!
//! # Build for Node.js
//! wasm-pack build --target nodejs --features wasm
//! ```

use std::io::Cursor;
use wasm_bindgen::prelude::*;

use crate::sound::Sound as RustSound;
use crate::pitch::Pitch as RustPitch;
use crate::formant::Formant as RustFormant;
use crate::intensity::Intensity as RustIntensity;
use crate::harmonicity::Harmonicity as RustHarmonicity;
use crate::spectrum::Spectrum as RustSpectrum;
use crate::spectrogram::Spectrogram as RustSpectrogram;

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the WASM module.
///
/// This sets up panic hooks for better error messages in the browser console.
/// Call this once before using any other functions.
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "wasm")]
    console_error_panic_hook::set_once();
}

// ============================================================================
// Sound - Main audio type for WASM
// ============================================================================

/// Audio samples with sample rate.
///
/// This is the main type for acoustic analysis in WASM. Create a Sound
/// from raw samples, then call analysis methods.
#[wasm_bindgen]
pub struct Sound {
    inner: RustSound,
}

#[wasm_bindgen]
impl Sound {
    /// Create a Sound from raw audio samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples as Float64Array, typically in range [-1, 1]
    /// * `sample_rate` - Sample rate in Hz (e.g., 44100, 48000)
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const samples = new Float64Array(audioContext.length);
    /// const sound = new Sound(samples, 44100);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(samples: &[f64], sample_rate: f64) -> Sound {
        Sound {
            inner: RustSound::from_slice(samples, sample_rate),
        }
    }

    /// Create a Sound from WAV file bytes.
    ///
    /// Supports mono WAV files. For stereo files, use `from_wav_channel()`.
    ///
    /// # Arguments
    ///
    /// * `wav_bytes` - WAV file contents as Uint8Array
    ///
    /// # Errors
    ///
    /// Throws if the WAV file is invalid or has multiple channels.
    pub fn from_wav(wav_bytes: &[u8]) -> Result<Sound, JsError> {
        let cursor = Cursor::new(wav_bytes);
        let reader = hound::WavReader::new(cursor)
            .map_err(|e| JsError::new(&format!("Failed to read WAV: {}", e)))?;

        let spec = reader.spec();

        if spec.channels != 1 {
            return Err(JsError::new(&format!(
                "Audio must be mono. Got {} channels. Use from_wav_channel() for multi-channel files.",
                spec.channels
            )));
        }

        let sample_rate = spec.sample_rate as f64;

        let samples: Vec<f64> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.into_samples::<f32>()
                    .map(|s| s.map(|v| v as f64))
                    .collect::<Result<Vec<f64>, _>>()
                    .map_err(|e| JsError::new(&format!("Failed to read samples: {}", e)))?
            }
            hound::SampleFormat::Int => {
                let bits = spec.bits_per_sample;
                let max_val = (1i64 << (bits - 1)) as f64;
                reader.into_samples::<i32>()
                    .map(|s| s.map(|v| v as f64 / max_val))
                    .collect::<Result<Vec<f64>, _>>()
                    .map_err(|e| JsError::new(&format!("Failed to read samples: {}", e)))?
            }
        };

        Ok(Sound {
            inner: RustSound::from_slice(&samples, sample_rate),
        })
    }

    /// Create a Sound from a specific channel of a WAV file.
    ///
    /// # Arguments
    ///
    /// * `wav_bytes` - WAV file contents as Uint8Array
    /// * `channel` - Channel index (0 = left, 1 = right, etc.)
    pub fn from_wav_channel(wav_bytes: &[u8], channel: usize) -> Result<Sound, JsError> {
        let cursor = Cursor::new(wav_bytes);
        let reader = hound::WavReader::new(cursor)
            .map_err(|e| JsError::new(&format!("Failed to read WAV: {}", e)))?;

        let spec = reader.spec();
        let n_channels = spec.channels as usize;

        if channel >= n_channels {
            return Err(JsError::new(&format!(
                "Channel {} does not exist. File has {} channels.",
                channel, n_channels
            )));
        }

        let sample_rate = spec.sample_rate as f64;

        let all_samples: Vec<f64> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.into_samples::<f32>()
                    .map(|s| s.map(|v| v as f64))
                    .collect::<Result<Vec<f64>, _>>()
                    .map_err(|e| JsError::new(&format!("Failed to read samples: {}", e)))?
            }
            hound::SampleFormat::Int => {
                let bits = spec.bits_per_sample;
                let max_val = (1i64 << (bits - 1)) as f64;
                reader.into_samples::<i32>()
                    .map(|s| s.map(|v| v as f64 / max_val))
                    .collect::<Result<Vec<f64>, _>>()
                    .map_err(|e| JsError::new(&format!("Failed to read samples: {}", e)))?
            }
        };

        // Extract the specified channel
        let samples: Vec<f64> = all_samples
            .iter()
            .skip(channel)
            .step_by(n_channels)
            .copied()
            .collect();

        Ok(Sound {
            inner: RustSound::from_slice(&samples, sample_rate),
        })
    }

    /// Get the number of samples.
    pub fn n_samples(&self) -> usize {
        self.inner.n_samples()
    }

    /// Get the sample rate in Hz.
    pub fn sample_rate(&self) -> f64 {
        self.inner.sample_rate()
    }

    /// Get the duration in seconds.
    pub fn duration(&self) -> f64 {
        self.inner.duration()
    }

    /// Get the audio samples as a Float64Array.
    pub fn samples(&self) -> Vec<f64> {
        self.inner.samples().to_vec()
    }

    // ========================================================================
    // Analysis methods - snake_case names to match Praat conventions
    // ========================================================================

    /// Compute pitch (F0) contour using autocorrelation method.
    ///
    /// # Arguments
    ///
    /// * `time_step` - Time step in seconds (0 = auto)
    /// * `pitch_floor` - Minimum pitch in Hz (e.g., 75)
    /// * `pitch_ceiling` - Maximum pitch in Hz (e.g., 600)
    pub fn to_pitch_ac(
        &self,
        time_step: f64,
        pitch_floor: f64,
        pitch_ceiling: f64,
    ) -> Pitch {
        Pitch {
            inner: self.inner.to_pitch_ac(time_step, pitch_floor, pitch_ceiling),
        }
    }

    /// Compute pitch (F0) contour using cross-correlation method.
    pub fn to_pitch_cc(
        &self,
        time_step: f64,
        pitch_floor: f64,
        pitch_ceiling: f64,
    ) -> Pitch {
        Pitch {
            inner: self.inner.to_pitch_cc(time_step, pitch_floor, pitch_ceiling),
        }
    }

    /// Compute formants using Burg's LPC method.
    ///
    /// # Arguments
    ///
    /// * `time_step` - Time step in seconds (0 = auto)
    /// * `max_num_formants` - Maximum number of formants (typically 5)
    /// * `max_formant_hz` - Maximum formant frequency (5500 for male, 5000 for female)
    /// * `window_length` - Window length in seconds (typically 0.025)
    /// * `pre_emphasis_from` - Pre-emphasis frequency in Hz (typically 50)
    pub fn to_formant_burg(
        &self,
        time_step: f64,
        max_num_formants: usize,
        max_formant_hz: f64,
        window_length: f64,
        pre_emphasis_from: f64,
    ) -> Formant {
        Formant {
            inner: self.inner.to_formant_burg(
                time_step,
                max_num_formants,
                max_formant_hz,
                window_length,
                pre_emphasis_from,
            ),
        }
    }

    /// Compute intensity contour.
    ///
    /// # Arguments
    ///
    /// * `min_pitch` - Minimum pitch in Hz (determines window size)
    /// * `time_step` - Time step in seconds (0 = auto)
    pub fn to_intensity(&self, min_pitch: f64, time_step: f64) -> Intensity {
        Intensity {
            inner: self.inner.to_intensity(min_pitch, time_step),
        }
    }

    /// Compute harmonicity (HNR) using autocorrelation method.
    ///
    /// # Arguments
    ///
    /// * `time_step` - Time step in seconds
    /// * `min_pitch` - Minimum pitch in Hz
    /// * `silence_threshold` - Silence threshold (0-1)
    /// * `periods_per_window` - Number of periods per window (typically 4.5)
    pub fn to_harmonicity_ac(
        &self,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> Harmonicity {
        Harmonicity {
            inner: self.inner.to_harmonicity_ac(
                time_step,
                min_pitch,
                silence_threshold,
                periods_per_window,
            ),
        }
    }

    /// Compute harmonicity (HNR) using cross-correlation method.
    pub fn to_harmonicity_cc(
        &self,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> Harmonicity {
        Harmonicity {
            inner: self.inner.to_harmonicity_cc(
                time_step,
                min_pitch,
                silence_threshold,
                periods_per_window,
            ),
        }
    }

    /// Compute spectrum (single-frame FFT).
    ///
    /// # Arguments
    ///
    /// * `fast` - If true, use power-of-2 FFT size for speed
    pub fn to_spectrum(&self, fast: bool) -> Spectrum {
        Spectrum {
            inner: self.inner.to_spectrum(fast),
        }
    }

    /// Compute spectrogram (time-frequency representation).
    ///
    /// # Arguments
    ///
    /// * `window_length` - Window length in seconds
    /// * `max_frequency` - Maximum frequency in Hz
    /// * `time_step` - Time step in seconds
    /// * `frequency_step` - Frequency step in Hz
    pub fn to_spectrogram(
        &self,
        window_length: f64,
        max_frequency: f64,
        time_step: f64,
        frequency_step: f64,
    ) -> Spectrogram {
        Spectrogram {
            inner: self.inner.to_spectrogram(
                window_length,
                max_frequency,
                time_step,
                frequency_step,
            ),
        }
    }
}

// ============================================================================
// Pitch - Pitch analysis result
// ============================================================================

/// Pitch (F0) analysis result.
#[wasm_bindgen]
pub struct Pitch {
    inner: RustPitch,
}

#[wasm_bindgen]
impl Pitch {
    /// Get the number of frames.
    pub fn n_frames(&self) -> usize {
        self.inner.n_frames()
    }

    /// Get the time step between frames.
    pub fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    /// Get the pitch floor in Hz.
    pub fn pitch_floor(&self) -> f64 {
        self.inner.pitch_floor()
    }

    /// Get the pitch ceiling in Hz.
    pub fn pitch_ceiling(&self) -> f64 {
        self.inner.pitch_ceiling()
    }

    /// Get F0 values for all frames.
    ///
    /// Returns Float64Array with F0 in Hz for voiced frames, NaN for unvoiced.
    pub fn values(&self) -> Vec<f64> {
        self.inner.values().iter().map(|&v| if v > 0.0 { v } else { f64::NAN }).collect()
    }

    /// Get time points for all frames.
    pub fn times(&self) -> Vec<f64> {
        self.inner.times().to_vec()
    }

    /// Get strength (correlation) values for all frames.
    pub fn strengths(&self) -> Vec<f64> {
        self.inner.strengths().to_vec()
    }

    /// Get time at a specific frame index.
    pub fn get_time_from_frame(&self, frame: usize) -> f64 {
        if frame < self.inner.n_frames() {
            self.inner.frames()[frame].time
        } else {
            f64::NAN
        }
    }

    /// Get F0 at a specific frame index. Returns NaN for unvoiced frames.
    pub fn get_value_in_frame(&self, frame: usize) -> f64 {
        if frame < self.inner.n_frames() {
            let freq = self.inner.frames()[frame].frequency();
            if freq > 0.0 { freq } else { f64::NAN }
        } else {
            f64::NAN
        }
    }
}

// ============================================================================
// Formant - Formant analysis result
// ============================================================================

/// Formant analysis result.
#[wasm_bindgen]
pub struct Formant {
    inner: RustFormant,
}

#[wasm_bindgen]
impl Formant {
    /// Get the number of frames.
    pub fn n_frames(&self) -> usize {
        self.inner.n_frames()
    }

    /// Get the time step between frames.
    pub fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    /// Get the maximum number of formants per frame.
    pub fn max_num_formants(&self) -> usize {
        self.inner.max_num_formants()
    }

    /// Get formant frequency values for a specific formant number (1=F1, 2=F2, etc.)
    pub fn formant_values(&self, formant_num: usize) -> Vec<f64> {
        self.inner.formant_values(formant_num).to_vec()
    }

    /// Get bandwidth values for a specific formant number.
    pub fn bandwidth_values(&self, formant_num: usize) -> Vec<f64> {
        self.inner.bandwidth_values(formant_num).to_vec()
    }

    /// Get time points for all frames.
    pub fn times(&self) -> Vec<f64> {
        self.inner.times().to_vec()
    }

    /// Get time at a specific frame index.
    pub fn get_time_from_frame(&self, frame: usize) -> f64 {
        if frame < self.inner.n_frames() {
            self.inner.frames()[frame].time
        } else {
            f64::NAN
        }
    }

    /// Get formant frequency at a specific frame and formant number.
    pub fn get_value_at_frame(&self, frame: usize, formant_num: usize) -> f64 {
        if frame < self.inner.n_frames() {
            self.inner.frames()[frame]
                .get_formant(formant_num)
                .map(|f| f.frequency)
                .unwrap_or(f64::NAN)
        } else {
            f64::NAN
        }
    }

    /// Get bandwidth at a specific frame and formant number.
    pub fn get_bandwidth_at_frame(&self, frame: usize, formant_num: usize) -> f64 {
        if frame < self.inner.n_frames() {
            self.inner.frames()[frame]
                .get_formant(formant_num)
                .map(|f| f.bandwidth)
                .unwrap_or(f64::NAN)
        } else {
            f64::NAN
        }
    }
}

// ============================================================================
// Intensity - Intensity analysis result
// ============================================================================

/// Intensity analysis result.
#[wasm_bindgen]
pub struct Intensity {
    inner: RustIntensity,
}

#[wasm_bindgen]
impl Intensity {
    /// Get the number of frames.
    pub fn n_frames(&self) -> usize {
        self.inner.n_frames()
    }

    /// Get the time step between frames.
    pub fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    /// Get intensity values for all frames in dB.
    pub fn values(&self) -> Vec<f64> {
        self.inner.values().to_vec()
    }

    /// Get time points for all frames.
    pub fn times(&self) -> Vec<f64> {
        self.inner.times().to_vec()
    }

    /// Get time at a specific frame index.
    pub fn get_time_from_frame(&self, frame: usize) -> f64 {
        if frame < self.inner.n_frames() {
            self.inner.times()[frame]
        } else {
            f64::NAN
        }
    }

    /// Get intensity at a specific frame index in dB.
    pub fn get_value_in_frame(&self, frame: usize) -> f64 {
        if frame < self.inner.n_frames() {
            self.inner.values()[frame]
        } else {
            f64::NAN
        }
    }

    /// Get the minimum intensity value in dB.
    pub fn get_minimum(&self) -> f64 {
        self.inner.values().iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Get the maximum intensity value in dB.
    pub fn get_maximum(&self) -> f64 {
        self.inner.values().iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get the mean intensity value in dB.
    pub fn get_mean(&self) -> f64 {
        let values = self.inner.values();
        if values.is_empty() {
            f64::NAN
        } else {
            values.sum() / values.len() as f64
        }
    }
}

// ============================================================================
// Harmonicity - Harmonicity analysis result
// ============================================================================

/// Harmonicity (HNR) analysis result.
#[wasm_bindgen]
pub struct Harmonicity {
    inner: RustHarmonicity,
}

#[wasm_bindgen]
impl Harmonicity {
    /// Get the number of frames.
    pub fn n_frames(&self) -> usize {
        self.inner.n_frames()
    }

    /// Get the time step between frames.
    pub fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    /// Get HNR values for all frames in dB. Returns -200 for silent/unvoiced frames.
    pub fn values(&self) -> Vec<f64> {
        self.inner.values().to_vec()
    }

    /// Get time points for all frames.
    pub fn times(&self) -> Vec<f64> {
        self.inner.times().to_vec()
    }

    /// Get time at a specific frame index.
    pub fn get_time_from_frame(&self, frame: usize) -> f64 {
        if frame < self.inner.n_frames() {
            self.inner.times()[frame]
        } else {
            f64::NAN
        }
    }

    /// Get HNR at a specific frame index in dB.
    pub fn get_value_in_frame(&self, frame: usize) -> f64 {
        if frame < self.inner.n_frames() {
            self.inner.values()[frame]
        } else {
            f64::NAN
        }
    }
}

// ============================================================================
// Spectrum - Spectrum analysis result
// ============================================================================

/// Spectrum (single-frame FFT) analysis result.
#[wasm_bindgen]
pub struct Spectrum {
    inner: RustSpectrum,
}

#[wasm_bindgen]
impl Spectrum {
    /// Get the number of frequency bins.
    pub fn n_bins(&self) -> usize {
        self.inner.n_bins()
    }

    /// Get the frequency resolution (bin width) in Hz.
    pub fn df(&self) -> f64 {
        self.inner.df()
    }

    /// Get the maximum frequency in Hz.
    pub fn max_frequency(&self) -> f64 {
        self.inner.f_max()
    }

    /// Get the real parts of the spectrum.
    pub fn real(&self) -> Vec<f64> {
        self.inner.real().to_vec()
    }

    /// Get the imaginary parts of the spectrum.
    pub fn imag(&self) -> Vec<f64> {
        self.inner.imag().to_vec()
    }

    /// Get frequency for a bin index.
    pub fn get_freq_from_bin(&self, bin: usize) -> f64 {
        self.inner.get_frequency(bin)
    }

    /// Get the center of gravity (spectral centroid) in Hz.
    pub fn get_center_of_gravity(&self, power: f64) -> f64 {
        self.inner.get_center_of_gravity(power)
    }

    /// Get the standard deviation (spectral spread) in Hz.
    pub fn get_standard_deviation(&self, power: f64) -> f64 {
        self.inner.get_standard_deviation(power)
    }

    /// Get the skewness of the spectrum.
    pub fn get_skewness(&self, power: f64) -> f64 {
        self.inner.get_skewness(power)
    }

    /// Get the kurtosis of the spectrum.
    pub fn get_kurtosis(&self, power: f64) -> f64 {
        self.inner.get_kurtosis(power)
    }

    /// Get the energy in a frequency band.
    pub fn get_band_energy(&self, f_min: f64, f_max: f64) -> f64 {
        self.inner.get_band_energy(f_min, f_max)
    }
}

// ============================================================================
// Spectrogram - Spectrogram analysis result
// ============================================================================

/// Spectrogram (time-frequency) analysis result.
#[wasm_bindgen]
pub struct Spectrogram {
    inner: RustSpectrogram,
}

#[wasm_bindgen]
impl Spectrogram {
    /// Get the number of time frames.
    pub fn n_times(&self) -> usize {
        self.inner.n_times()
    }

    /// Get the number of frequency bins.
    pub fn n_freqs(&self) -> usize {
        self.inner.n_freqs()
    }

    /// Get the time step between frames.
    pub fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    /// Get the frequency step between bins.
    pub fn freq_step(&self) -> f64 {
        self.inner.freq_step()
    }

    /// Get the minimum time in seconds.
    pub fn time_min(&self) -> f64 {
        self.inner.time_min()
    }

    /// Get the maximum time in seconds.
    pub fn time_max(&self) -> f64 {
        self.inner.time_max()
    }

    /// Get the minimum frequency in Hz.
    pub fn freq_min(&self) -> f64 {
        self.inner.freq_min()
    }

    /// Get the maximum frequency in Hz.
    pub fn freq_max(&self) -> f64 {
        self.inner.freq_max()
    }

    /// Get time points for all frames.
    pub fn times(&self) -> Vec<f64> {
        self.inner.times()
    }

    /// Get frequency points for all bins.
    pub fn frequencies(&self) -> Vec<f64> {
        self.inner.frequencies()
    }

    /// Get time at a specific frame index.
    pub fn get_time_from_frame(&self, frame: usize) -> f64 {
        self.inner.get_time_from_frame(frame)
    }

    /// Get frequency at a specific bin index.
    pub fn get_freq_from_bin(&self, bin: usize) -> f64 {
        self.inner.get_freq_from_bin(bin)
    }

    /// Get all power values as a flat array (row-major: freq × time).
    pub fn values(&self) -> Vec<f64> {
        self.inner.values().iter().copied().collect()
    }

    /// Get power value at a specific time frame and frequency bin.
    /// Returns NaN if indices are out of bounds.
    pub fn get_value_at(&self, time_frame: usize, freq_bin: usize) -> f64 {
        if time_frame >= self.inner.n_times() || freq_bin >= self.inner.n_freqs() {
            return f64::NAN;
        }
        self.inner.values()[[freq_bin, time_frame]]
    }
}
