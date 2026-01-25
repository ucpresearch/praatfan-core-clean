//! Sound - Audio samples with sample rate.
//!
//! This is the foundation type for all acoustic analysis.

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
    samples: Array1<f64>,
    /// Sample rate in Hz.
    sample_rate: f64,
}

impl Sound {
    /// Create a Sound from samples and sample rate.
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
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the WAV file
    ///
    /// # Returns
    ///
    /// A Result containing the Sound or an error.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let reader = hound::WavReader::open(path)?;
        let spec = reader.spec();

        if spec.channels != 1 {
            return Err(Error::NotMono(spec.channels));
        }

        let sample_rate = spec.sample_rate as f64;

        let samples: Vec<f64> = match spec.sample_format {
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

        Ok(Self {
            samples: Array1::from_vec(samples),
            sample_rate,
        })
    }

    /// Load a specific channel from a WAV file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the WAV file
    /// * `channel` - Channel index (0-based)
    ///
    /// # Returns
    ///
    /// A Result containing the Sound or an error.
    pub fn from_file_channel<P: AsRef<Path>>(path: P, channel: usize) -> Result<Self> {
        let reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        let n_channels = spec.channels as usize;

        if channel >= n_channels {
            return Err(Error::InvalidParameter(format!(
                "Channel {} does not exist. File has {} channels.",
                channel, n_channels
            )));
        }

        let sample_rate = spec.sample_rate as f64;

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

        // Extract the specified channel
        let samples: Vec<f64> = all_samples
            .iter()
            .skip(channel)
            .step_by(n_channels)
            .copied()
            .collect();

        Ok(Self {
            samples: Array1::from_vec(samples),
            sample_rate,
        })
    }

    /// Get the audio samples.
    #[inline]
    pub fn samples(&self) -> &Array1<f64> {
        &self.samples
    }

    /// Get the sample rate in Hz.
    #[inline]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Get the number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    /// Get the total duration in seconds.
    #[inline]
    pub fn duration(&self) -> f64 {
        self.n_samples() as f64 / self.sample_rate
    }

    /// Get the sample period (1 / sample_rate).
    #[inline]
    pub fn dx(&self) -> f64 {
        1.0 / self.sample_rate
    }

    /// Get the time of the first sample (centered on sample).
    #[inline]
    pub fn x1(&self) -> f64 {
        0.5 * self.dx()
    }

    // ========== Analysis Methods ==========

    /// Compute the spectrum (single-frame FFT).
    ///
    /// # Arguments
    ///
    /// * `fast` - If true, use power-of-2 FFT size for speed
    pub fn to_spectrum(&self, fast: bool) -> Spectrum {
        crate::spectrum::sound_to_spectrum(self, fast)
    }

    /// Compute intensity contour.
    ///
    /// # Arguments
    ///
    /// * `min_pitch` - Minimum pitch in Hz (determines window size)
    /// * `time_step` - Time step in seconds (0 = auto)
    pub fn to_intensity(&self, min_pitch: f64, time_step: f64) -> Intensity {
        crate::intensity::sound_to_intensity(self, min_pitch, time_step)
    }

    /// Compute pitch (F0) contour using autocorrelation method.
    ///
    /// # Arguments
    ///
    /// * `time_step` - Time step in seconds (0 = auto)
    /// * `pitch_floor` - Minimum pitch in Hz
    /// * `pitch_ceiling` - Maximum pitch in Hz
    pub fn to_pitch_ac(&self, time_step: f64, pitch_floor: f64, pitch_ceiling: f64) -> Pitch {
        crate::pitch::sound_to_pitch_ac(self, time_step, pitch_floor, pitch_ceiling)
    }

    /// Compute pitch (F0) contour using cross-correlation method.
    ///
    /// # Arguments
    ///
    /// * `time_step` - Time step in seconds (0 = auto)
    /// * `pitch_floor` - Minimum pitch in Hz
    /// * `pitch_ceiling` - Maximum pitch in Hz
    pub fn to_pitch_cc(&self, time_step: f64, pitch_floor: f64, pitch_ceiling: f64) -> Pitch {
        crate::pitch::sound_to_pitch_cc(self, time_step, pitch_floor, pitch_ceiling)
    }

    /// Compute harmonicity (HNR) using autocorrelation method.
    ///
    /// # Arguments
    ///
    /// * `time_step` - Time step in seconds
    /// * `min_pitch` - Minimum pitch in Hz
    /// * `silence_threshold` - Silence threshold (0-1)
    /// * `periods_per_window` - Number of periods per window
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
    /// # Arguments
    ///
    /// * `time_step` - Time step in seconds
    /// * `min_pitch` - Minimum pitch in Hz
    /// * `silence_threshold` - Silence threshold (0-1)
    /// * `periods_per_window` - Number of periods per window
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
    /// # Arguments
    ///
    /// * `time_step` - Time step in seconds (0 = auto: 25% of window)
    /// * `max_num_formants` - Maximum number of formants to find
    /// * `max_formant_hz` - Maximum formant frequency in Hz
    /// * `window_length` - Window length in seconds (actual = 2x this value)
    /// * `pre_emphasis_from` - Pre-emphasis from frequency in Hz
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
