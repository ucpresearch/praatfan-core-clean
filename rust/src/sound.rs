//! Sound - Audio samples with sample rate.
//!
//! This is the foundation type for all acoustic analysis in praatfan.
//!
//! # Clean Room Implementation
//!
//! This module provides a standard audio container. The implementation follows
//! common audio processing conventions without any Praat-specific code.
//!
//! # Mono Audio Only
//!
//! This implementation supports **mono audio only** to simplify the codebase
//! and avoid undocumented multi-channel handling concerns. Multi-channel files
//! require explicit channel selection via `from_file_channel()`.
//!
//! # Sample Format
//!
//! Audio samples are stored as 64-bit floating point values, normalized to
//! the range [-1.0, 1.0] for integer formats. This provides maximum precision
//! for acoustic analysis algorithms.

use std::path::Path;

use ndarray::Array1;

use crate::error::{Error, Result};
use crate::spectrum::Spectrum;
use crate::intensity::Intensity;
use crate::pitch::Pitch;
use crate::harmonicity::Harmonicity;
use crate::formant::Formant;
use crate::spectrogram::Spectrogram;

/// Represents audio samples with sample rate.
///
/// This is the foundation type for all acoustic analysis in praatfan.
/// Only mono (single-channel) audio is supported.
///
/// # Example
///
/// ```no_run
/// use praatfan::Sound;
///
/// let sound = Sound::from_file("audio.wav").unwrap();
/// println!("Duration: {:.3}s", sound.duration());
/// ```
#[derive(Debug, Clone)]
pub struct Sound {
    /// Audio samples as a 1D array.
    ///
    /// Values are typically in the range [-1.0, 1.0] when loaded from
    /// integer WAV files, but may exceed this range for floating-point
    /// files or after processing.
    samples: Array1<f64>,

    /// Sample rate in Hz.
    ///
    /// Common values: 8000, 16000, 22050, 44100, 48000.
    /// The sample rate determines the Nyquist frequency (sample_rate / 2),
    /// which is the maximum frequency that can be represented.
    sample_rate: f64,
}

impl Sound {
    /// Create a Sound from samples and sample rate.
    ///
    /// This is the primary constructor for creating Sound objects from
    /// existing sample data (e.g., from synthesis or processing).
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples as a 1D array
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// A new Sound object.
    pub fn new(samples: Array1<f64>, sample_rate: f64) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    /// Create a Sound from a slice of samples.
    ///
    /// Convenience constructor that copies data from a slice.
    /// Use `new()` with an Array1 to avoid copying if you already have one.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples as a slice
    /// * `sample_rate` - Sample rate in Hz
    pub fn from_slice(samples: &[f64], sample_rate: f64) -> Self {
        Self {
            samples: Array1::from_vec(samples.to_vec()),
            sample_rate,
        }
    }

    /// Load audio from a WAV file.
    ///
    /// Only mono files are supported. Multi-channel files will return an error.
    /// For multi-channel files, use `from_file_channel()` to select a specific channel.
    ///
    /// # Sample Format Handling
    ///
    /// - **Integer formats** (8, 16, 24, 32 bit): Normalized to [-1.0, 1.0]
    ///   by dividing by the maximum value (2^(bits-1))
    /// - **Float formats**: Loaded as-is (typically already normalized)
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the WAV file
    ///
    /// # Returns
    ///
    /// A Result containing the Sound or an error.
    ///
    /// # Errors
    ///
    /// - `Error::NotMono` if the file has more than one channel
    /// - `Error::WavError` if the file cannot be read
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Use the hound crate for WAV file reading
        let reader = hound::WavReader::open(path)?;
        let spec = reader.spec();

        // Enforce mono audio requirement
        if spec.channels != 1 {
            return Err(Error::NotMono(spec.channels));
        }

        let sample_rate = spec.sample_rate as f64;

        // Convert samples to f64, normalizing integer formats
        let samples: Vec<f64> = match spec.sample_format {
            hound::SampleFormat::Float => {
                // Float samples: convert f32 to f64 directly
                reader.into_samples::<f32>()
                    .map(|s| s.map(|v| v as f64))
                    .collect::<std::result::Result<Vec<f64>, _>>()?
            }
            hound::SampleFormat::Int => {
                // Integer samples: normalize to [-1.0, 1.0]
                // max_val = 2^(bits-1), e.g., 32768 for 16-bit audio
                let bits = spec.bits_per_sample;
                let max_val = (1i64 << (bits - 1)) as f64;
                reader.into_samples::<i32>()
                    .map(|s| s.map(|v| v as f64 / max_val))
                    .collect::<std::result::Result<Vec<f64>, _>>()?
            }
        };

        Ok(Self {
            samples: Array1::from_vec(samples),
            sample_rate,
        })
    }

    /// Load a specific channel from a WAV file.
    ///
    /// Use this method for multi-channel files when you want to analyze
    /// a specific channel (e.g., left channel = 0, right channel = 1).
    ///
    /// # Channel Extraction
    ///
    /// WAV files store interleaved samples: [L0, R0, L1, R1, ...]
    /// This method extracts every Nth sample starting at index `channel`,
    /// where N is the number of channels.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the WAV file
    /// * `channel` - Channel index (0-based)
    ///
    /// # Returns
    ///
    /// A Result containing the Sound or an error.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` if the channel index is out of range
    /// - `Error::WavError` if the file cannot be read
    pub fn from_file_channel<P: AsRef<Path>>(path: P, channel: usize) -> Result<Self> {
        let reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        let n_channels = spec.channels as usize;

        // Validate channel index
        if channel >= n_channels {
            return Err(Error::InvalidParameter(format!(
                "Channel {} does not exist. File has {} channels.",
                channel, n_channels
            )));
        }

        let sample_rate = spec.sample_rate as f64;

        // Read all interleaved samples
        let all_samples: Vec<f64> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.into_samples::<f32>()
                    .map(|s| s.map(|v| v as f64))
                    .collect::<std::result::Result<Vec<f64>, _>>()?
            }
            hound::SampleFormat::Int => {
                let bits = spec.bits_per_sample;
                let max_val = (1i64 << (bits - 1)) as f64;
                reader.into_samples::<i32>()
                    .map(|s| s.map(|v| v as f64 / max_val))
                    .collect::<std::result::Result<Vec<f64>, _>>()?
            }
        };

        // Extract the specified channel by taking every Nth sample
        // starting at the channel index
        let samples: Vec<f64> = all_samples
            .iter()
            .skip(channel)        // Start at the channel offset
            .step_by(n_channels)  // Take every Nth sample
            .copied()
            .collect();

        Ok(Self {
            samples: Array1::from_vec(samples),
            sample_rate,
        })
    }

    /// Get the audio samples.
    ///
    /// Returns a reference to the internal sample array. Use this for
    /// direct access to sample values or for implementing custom analysis.
    #[inline]
    pub fn samples(&self) -> &Array1<f64> {
        &self.samples
    }

    /// Get the sample rate in Hz.
    ///
    /// The sample rate determines how many samples represent one second
    /// of audio. Higher sample rates can represent higher frequencies
    /// (up to sample_rate / 2, the Nyquist frequency).
    #[inline]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Get the number of samples.
    ///
    /// This is the total length of the audio in samples.
    /// Duration = n_samples / sample_rate.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    /// Get the total duration in seconds.
    ///
    /// Calculated as: n_samples / sample_rate
    #[inline]
    pub fn duration(&self) -> f64 {
        self.n_samples() as f64 / self.sample_rate
    }

    /// Get the sample period (1 / sample_rate).
    ///
    /// This is the time between consecutive samples in seconds.
    /// Also known as dt or Δt in signal processing literature.
    #[inline]
    pub fn dx(&self) -> f64 {
        1.0 / self.sample_rate
    }

    /// Get the time of the first sample (centered on sample).
    ///
    /// In the Praat time domain convention, samples are considered to be
    /// centered at their time points, so the first sample is at t = dt/2
    /// rather than t = 0. This ensures that the signal is symmetric around
    /// its center for even-length signals.
    #[inline]
    pub fn x1(&self) -> f64 {
        0.5 * self.dx()
    }

    // ========== Analysis Methods ==========
    //
    // These methods provide the primary API for acoustic analysis.
    // Each method delegates to a specialized module that implements
    // the algorithm. Parameters match Praat's defaults where applicable.

    /// Compute the spectrum (single-frame FFT).
    ///
    /// The spectrum represents the frequency content of the entire sound
    /// as a single FFT frame. For time-varying analysis, use `to_spectrogram()`.
    ///
    /// # Arguments
    ///
    /// * `fast` - If true, use power-of-2 FFT size for speed.
    ///   If false, use the exact number of samples (may be slower but
    ///   provides exact frequency resolution).
    pub fn to_spectrum(&self, fast: bool) -> Spectrum {
        crate::spectrum::sound_to_spectrum(self, fast)
    }

    /// Compute intensity contour.
    ///
    /// Intensity represents the loudness (energy) of the signal over time,
    /// measured in decibels (dB) relative to a reference pressure.
    ///
    /// # Arguments
    ///
    /// * `min_pitch` - Minimum pitch in Hz. This determines the window size
    ///   used for analysis. Lower values = longer windows = smoother contour.
    ///   Typical value: 75 Hz for male voice, 100 Hz for female voice.
    /// * `time_step` - Time step in seconds. Use 0 for auto (0.8/min_pitch).
    pub fn to_intensity(&self, min_pitch: f64, time_step: f64) -> Intensity {
        crate::intensity::sound_to_intensity(self, min_pitch, time_step)
    }

    /// Compute pitch (F0) contour using autocorrelation method.
    ///
    /// The AC method from Boersma (1993) provides robust pitch tracking
    /// by computing the normalized autocorrelation of windowed frames
    /// and finding peaks corresponding to periodic components.
    ///
    /// # Arguments
    ///
    /// * `time_step` - Time step in seconds. Use 0 for auto (0.75/floor).
    /// * `pitch_floor` - Minimum pitch in Hz. Typical: 75 Hz (male), 100 Hz (female).
    /// * `pitch_ceiling` - Maximum pitch in Hz. Typical: 300-600 Hz.
    ///
    /// # Algorithm
    ///
    /// 1. Window the signal with a Hanning window
    /// 2. Compute autocorrelation
    /// 3. Normalize by window autocorrelation (Boersma Eq. 9)
    /// 4. Find peaks in the normalized autocorrelation
    /// 5. Use Viterbi algorithm to find optimal path through candidates
    pub fn to_pitch_ac(&self, time_step: f64, pitch_floor: f64, pitch_ceiling: f64) -> Pitch {
        crate::pitch::sound_to_pitch_ac(self, time_step, pitch_floor, pitch_ceiling)
    }

    /// Compute pitch (F0) contour using cross-correlation method.
    ///
    /// The CC method computes normalized cross-correlation between
    /// the signal and itself at various lags. It's more robust to
    /// amplitude variations but may be less accurate for some signals.
    ///
    /// # Arguments
    ///
    /// * `time_step` - Time step in seconds. Use 0 for auto (0.75/floor).
    /// * `pitch_floor` - Minimum pitch in Hz.
    /// * `pitch_ceiling` - Maximum pitch in Hz.
    pub fn to_pitch_cc(&self, time_step: f64, pitch_floor: f64, pitch_ceiling: f64) -> Pitch {
        crate::pitch::sound_to_pitch_cc(self, time_step, pitch_floor, pitch_ceiling)
    }

    /// Compute harmonicity (HNR) using autocorrelation method.
    ///
    /// Harmonicity measures the ratio of harmonic (periodic) energy to
    /// noise energy, expressed in decibels. Higher values indicate more
    /// periodic (voiced) signals.
    ///
    /// # Arguments
    ///
    /// * `time_step` - Time step in seconds
    /// * `min_pitch` - Minimum pitch in Hz (determines analysis window)
    /// * `silence_threshold` - Frames below this amplitude ratio are marked silent (0-1)
    /// * `periods_per_window` - Number of pitch periods per analysis window.
    ///   Typical value: 4.5 for AC method.
    ///
    /// # Formula
    ///
    /// HNR = 10 × log₁₀(r / (1-r))
    ///
    /// where r is the normalized autocorrelation at the pitch period
    /// (from Praat manual, Harmonicity.html).
    pub fn to_harmonicity_ac(
        &self,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> Harmonicity {
        crate::harmonicity::sound_to_harmonicity_ac(
            self,
            time_step,
            min_pitch,
            silence_threshold,
            periods_per_window,
        )
    }

    /// Compute harmonicity (HNR) using cross-correlation method.
    ///
    /// Uses the CC pitch method internally, which may give different
    /// results than the AC method for some signals.
    ///
    /// # Arguments
    ///
    /// * `time_step` - Time step in seconds
    /// * `min_pitch` - Minimum pitch in Hz
    /// * `silence_threshold` - Silence threshold (0-1)
    /// * `periods_per_window` - Number of periods per window.
    ///   Typical value: 1.0 for CC method.
    pub fn to_harmonicity_cc(
        &self,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> Harmonicity {
        crate::harmonicity::sound_to_harmonicity_cc(
            self,
            time_step,
            min_pitch,
            silence_threshold,
            periods_per_window,
        )
    }

    /// Compute formants using Burg's LPC method.
    ///
    /// Formants are resonance frequencies of the vocal tract, typically
    /// denoted F1, F2, F3, etc. They are crucial for vowel identification
    /// and speech analysis.
    ///
    /// # Arguments
    ///
    /// * `time_step` - Time step in seconds. Use 0 for auto (25% of window).
    /// * `max_num_formants` - Maximum number of formants to find. Typical: 5.
    /// * `max_formant_hz` - Maximum formant frequency in Hz. This determines
    ///   the resampling rate: signal is resampled to 2× this frequency.
    ///   Typical: 5500 Hz for male, 5000 Hz for female.
    /// * `window_length` - Window length parameter in seconds. The actual
    ///   window used is 2× this value (Praat convention). Typical: 0.025s.
    /// * `pre_emphasis_from` - Apply pre-emphasis starting from this frequency.
    ///   This boosts high frequencies to compensate for the -6 dB/octave
    ///   roll-off of the glottal source. Typical: 50 Hz.
    ///
    /// # Algorithm
    ///
    /// 1. Resample to 2× max_formant_hz using FFT-based sinc interpolation
    /// 2. Apply pre-emphasis filter
    /// 3. For each frame:
    ///    a. Apply Gaussian window
    ///    b. Compute LPC coefficients using Burg's algorithm
    ///    c. Find roots of the LPC polynomial via companion matrix eigenvalues
    ///    d. Convert roots to formant frequencies and bandwidths
    pub fn to_formant_burg(
        &self,
        time_step: f64,
        max_num_formants: usize,
        max_formant_hz: f64,
        window_length: f64,
        pre_emphasis_from: f64,
    ) -> Formant {
        crate::formant::sound_to_formant_burg(
            self,
            time_step,
            max_num_formants,
            max_formant_hz,
            window_length,
            pre_emphasis_from,
        )
    }

    /// Compute spectrogram (time-frequency representation).
    ///
    /// The spectrogram shows how spectral energy varies over time,
    /// providing a visual representation of the sound's frequency content.
    ///
    /// # Arguments
    ///
    /// * `window_length` - Window length in seconds. Determines the
    ///   trade-off between time and frequency resolution. Shorter windows
    ///   give better time resolution, longer windows give better frequency
    ///   resolution.
    /// * `max_frequency` - Maximum frequency to include in Hz.
    /// * `time_step` - Time step between frames in seconds.
    /// * `frequency_step` - Frequency resolution in Hz.
    pub fn to_spectrogram(
        &self,
        window_length: f64,
        max_frequency: f64,
        time_step: f64,
        frequency_step: f64,
    ) -> Spectrogram {
        crate::spectrogram::sound_to_spectrogram(
            self,
            window_length,
            max_frequency,
            time_step,
            frequency_step,
        )
    }
}

impl std::fmt::Display for Sound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Sound({} samples, {} Hz, {:.3}s)",
            self.n_samples(),
            self.sample_rate,
            self.duration()
        )
    }
}
