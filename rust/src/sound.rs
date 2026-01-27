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

use std::fs::File;
use std::path::Path;

use ndarray::Array1;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

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

/// Decode an audio file using symphonia.
///
/// Returns (samples, sample_rate, n_channels).
/// If `channel` is Some, extracts only that channel; otherwise returns interleaved samples.
fn decode_audio_file(path: &Path, channel: Option<usize>) -> Result<(Vec<f64>, f64, usize)> {
    // Open the file
    let file = File::open(path).map_err(|e| {
        Error::AudioDecodeError(format!("Failed to open file: {}", e))
    })?;

    // Create a media source stream
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Use file extension as a hint for format detection
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    // Probe the file to detect format
    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| Error::AudioDecodeError(format!("Failed to probe file format: {}", e)))?;

    let mut format = probed.format;

    // Find the first audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| Error::AudioDecodeError("No audio track found".to_string()))?;

    let track_id = track.id;

    // Get audio parameters
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| Error::AudioDecodeError("Unknown sample rate".to_string()))? as f64;

    let n_channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);

    // Validate channel index if specified
    if let Some(ch) = channel {
        if ch >= n_channels {
            return Err(Error::InvalidParameter(format!(
                "Channel {} does not exist. File has {} channels.",
                ch, n_channels
            )));
        }
    }

    // Create a decoder
    let decoder_opts = DecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .map_err(|e| Error::AudioDecodeError(format!("Failed to create decoder: {}", e)))?;

    // Decode all packets
    let mut all_samples: Vec<f64> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break; // End of file
            }
            Err(e) => {
                return Err(Error::AudioDecodeError(format!("Failed to read packet: {}", e)));
            }
        };

        // Skip packets from other tracks
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(symphonia::core::errors::Error::DecodeError(_)) => {
                continue; // Skip decode errors
            }
            Err(e) => {
                return Err(Error::AudioDecodeError(format!("Decode error: {}", e)));
            }
        };

        // Convert samples to f64
        append_samples(&decoded, &mut all_samples, channel, n_channels);
    }

    Ok((all_samples, sample_rate, n_channels))
}

/// Append samples from an audio buffer to the output vector.
/// If `channel` is Some, extracts only that channel; otherwise extracts all channels interleaved.
fn append_samples(
    buffer: &AudioBufferRef,
    output: &mut Vec<f64>,
    channel: Option<usize>,
    _n_channels: usize,
) {
    macro_rules! process_buffer {
        ($buf:expr, $convert:expr) => {{
            let buf = $buf;
            let n_frames = buf.frames();
            let actual_channels = buf.spec().channels.count();
            let convert = $convert;

            match channel {
                Some(ch) if ch < actual_channels => {
                    // Extract single channel
                    for frame in 0..n_frames {
                        let sample = buf.chan(ch)[frame];
                        output.push(convert(sample));
                    }
                }
                Some(_) => {
                    // Channel out of range - should have been caught earlier
                }
                None => {
                    // For mono, just copy; otherwise interleave
                    if actual_channels == 1 {
                        for frame in 0..n_frames {
                            output.push(convert(buf.chan(0)[frame]));
                        }
                    } else {
                        for frame in 0..n_frames {
                            for ch in 0..actual_channels {
                                output.push(convert(buf.chan(ch)[frame]));
                            }
                        }
                    }
                }
            }
        }};
    }

    match buffer {
        AudioBufferRef::U8(buf) => {
            process_buffer!(&**buf, |s: u8| (s as f64 - 128.0) / 128.0);
        }
        AudioBufferRef::U16(buf) => {
            process_buffer!(&**buf, |s: u16| (s as f64 - 32768.0) / 32768.0);
        }
        AudioBufferRef::U24(buf) => {
            let buf = &**buf;
            let n_frames = buf.frames();
            let actual_channels = buf.spec().channels.count();

            match channel {
                Some(ch) if ch < actual_channels => {
                    for frame in 0..n_frames {
                        let sample = buf.chan(ch)[frame];
                        output.push((sample.inner() as f64 - 8388608.0) / 8388608.0);
                    }
                }
                Some(_) => {}
                None => {
                    if actual_channels == 1 {
                        for frame in 0..n_frames {
                            output.push((buf.chan(0)[frame].inner() as f64 - 8388608.0) / 8388608.0);
                        }
                    } else {
                        for frame in 0..n_frames {
                            for ch in 0..actual_channels {
                                output.push((buf.chan(ch)[frame].inner() as f64 - 8388608.0) / 8388608.0);
                            }
                        }
                    }
                }
            }
        }
        AudioBufferRef::U32(buf) => {
            process_buffer!(&**buf, |s: u32| (s as f64 - 2147483648.0) / 2147483648.0);
        }
        AudioBufferRef::S8(buf) => {
            process_buffer!(&**buf, |s: i8| s as f64 / 128.0);
        }
        AudioBufferRef::S16(buf) => {
            process_buffer!(&**buf, |s: i16| s as f64 / 32768.0);
        }
        AudioBufferRef::S24(buf) => {
            let buf = &**buf;
            let n_frames = buf.frames();
            let actual_channels = buf.spec().channels.count();

            match channel {
                Some(ch) if ch < actual_channels => {
                    for frame in 0..n_frames {
                        let sample = buf.chan(ch)[frame];
                        output.push(sample.inner() as f64 / 8388608.0);
                    }
                }
                Some(_) => {}
                None => {
                    if actual_channels == 1 {
                        for frame in 0..n_frames {
                            output.push(buf.chan(0)[frame].inner() as f64 / 8388608.0);
                        }
                    } else {
                        for frame in 0..n_frames {
                            for ch in 0..actual_channels {
                                output.push(buf.chan(ch)[frame].inner() as f64 / 8388608.0);
                            }
                        }
                    }
                }
            }
        }
        AudioBufferRef::S32(buf) => {
            process_buffer!(&**buf, |s: i32| s as f64 / 2147483648.0);
        }
        AudioBufferRef::F32(buf) => {
            process_buffer!(&**buf, |s: f32| s as f64);
        }
        AudioBufferRef::F64(buf) => {
            process_buffer!(&**buf, |s: f64| s);
        }
    }
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

    /// Load audio from a file.
    ///
    /// Supports multiple formats via symphonia: WAV, MP3, OGG Vorbis, FLAC, AAC.
    ///
    /// Only mono files are supported. Multi-channel files will return an error.
    /// For multi-channel files, use `from_file_channel()` to select a specific channel.
    ///
    /// # Sample Format Handling
    ///
    /// All samples are normalized to the range [-1.0, 1.0].
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the audio file
    ///
    /// # Returns
    ///
    /// A Result containing the Sound or an error.
    ///
    /// # Errors
    ///
    /// - `Error::NotMono` if the file has more than one channel
    /// - `Error::AudioDecodeError` if the file cannot be read or decoded
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let (samples, sample_rate, n_channels) = decode_audio_file(path.as_ref(), None)?;

        // Enforce mono audio requirement
        if n_channels != 1 {
            return Err(Error::NotMono(n_channels as u16));
        }

        Ok(Self {
            samples: Array1::from_vec(samples),
            sample_rate,
        })
    }

    /// Load a specific channel from an audio file.
    ///
    /// Supports multiple formats via symphonia: WAV, MP3, OGG Vorbis, FLAC, AAC.
    ///
    /// Use this method for multi-channel files when you want to analyze
    /// a specific channel (e.g., left channel = 0, right channel = 1).
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the audio file
    /// * `channel` - Channel index (0-based)
    ///
    /// # Returns
    ///
    /// A Result containing the Sound or an error.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` if the channel index is out of range
    /// - `Error::AudioDecodeError` if the file cannot be read or decoded
    pub fn from_file_channel<P: AsRef<Path>>(path: P, channel: usize) -> Result<Self> {
        let (samples, sample_rate, n_channels) = decode_audio_file(path.as_ref(), Some(channel))?;

        // Validate channel index (also checked in decode_audio_file, but be explicit)
        if channel >= n_channels {
            return Err(Error::InvalidParameter(format!(
                "Channel {} does not exist. File has {} channels.",
                channel, n_channels
            )));
        }

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
