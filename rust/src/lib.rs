//! # praatfan
//!
//! Clean-room reimplementation of Praat's acoustic analysis algorithms.
//!
//! This library provides acoustic analysis functions equivalent to Praat,
//! implemented without access to GPL source code. The implementation is
//! based on:
//!
//! - Published academic papers (especially Boersma 1993 for pitch/HNR)
//! - Praat's public documentation (manual pages)
//! - Black-box testing against parselmouth for validation
//!
//! # License
//!
//! This is a clean-room implementation designed to be non-GPL (MIT/Apache-2.0).
//! No Praat source code was used in the implementation.
//!
//! # Supported Analysis Types
//!
//! - **Sound**: Audio loading and basic operations (mono WAV only)
//! - **Spectrum**: Single-frame FFT and spectral moments
//! - **Intensity**: RMS energy contour in dB
//! - **Pitch**: F0 detection using AC and CC methods
//! - **Harmonicity**: Harmonics-to-noise ratio (HNR)
//! - **Formant**: LPC-based formant tracking (Burg's method)
//! - **Spectrogram**: Time-frequency representation (STFT)
//!
//! # Quick Start
//!
//! ```no_run
//! use praatfan::Sound;
//!
//! // Load a mono WAV file
//! let sound = Sound::from_file("audio.wav").unwrap();
//!
//! // Compute pitch contour using autocorrelation method
//! // Parameters: time_step (0=auto), pitch_floor (Hz), pitch_ceiling (Hz)
//! let pitch = sound.to_pitch_ac(0.0, 75.0, 600.0);
//!
//! // Compute formants using Burg's LPC method
//! // Parameters: time_step (0=auto), max_formants, max_formant_hz, window_length, pre_emphasis
//! let formant = sound.to_formant_burg(0.0, 5, 5500.0, 0.025, 50.0);
//!
//! // Get F1 values for all frames
//! let f1_values = formant.formant_values(1);
//! ```
//!
//! # Module Organization
//!
//! Each analysis type has its own module containing:
//! - A result struct (e.g., `Pitch`, `Formant`)
//! - A conversion function (e.g., `sound_to_pitch_ac`)
//! - Supporting types and helper functions
//!
//! The `Sound` struct provides convenience methods that delegate to these modules.
//!
//! # Validation
//!
//! All algorithms have been validated against parselmouth (Python bindings for Praat).
//! See the project's `scripts/` directory for comparison tools and the `PROGRESS.md`
//! file for detailed accuracy statistics.

// Module declarations
pub mod error;
pub mod formant;
pub mod harmonicity;
pub mod intensity;
pub mod pitch;
pub mod sound;
pub mod spectrogram;
pub mod spectrum;

// WASM bindings (enabled with "wasm" feature)
#[cfg(feature = "wasm")]
pub mod wasm;

// Python bindings (enabled with "python" feature)
#[cfg(feature = "python")]
pub mod python;

// Re-export main types at crate root for convenient access
//
// Users can import the most common types directly:
//   use praatfan::{Sound, Formant, Pitch};
//
// Or import from specific modules for less common types:
//   use praatfan::pitch::{PitchMethod, FrameTiming};

/// Error types for praatfan operations.
pub use error::{Error, Result};

/// Formant analysis types and functions.
///
/// - `Formant`: Result of formant analysis
/// - `FormantFrame`: Single frame of formant data
/// - `FormantPoint`: Individual formant (frequency + bandwidth)
/// - `sound_to_formant_burg`: Compute formants using Burg's LPC method
pub use formant::{sound_to_formant_burg, Formant, FormantFrame, FormantPoint};

/// Harmonicity (HNR) analysis types and functions.
///
/// - `Harmonicity`: Result of HNR analysis
/// - `sound_to_harmonicity_ac`: Compute HNR using autocorrelation method
/// - `sound_to_harmonicity_cc`: Compute HNR using cross-correlation method
pub use harmonicity::{sound_to_harmonicity_ac, sound_to_harmonicity_cc, Harmonicity};

/// Intensity analysis types and functions.
///
/// - `Intensity`: Result of intensity analysis
/// - `Interpolation`: Method for interpolating between frames
/// - `sound_to_intensity`: Compute intensity contour
pub use intensity::{sound_to_intensity, Intensity, Interpolation};

/// Pitch (F0) analysis types and functions.
///
/// - `Pitch`: Result of pitch analysis
/// - `PitchFrame`: Single frame with candidates
/// - `PitchCandidate`: Individual pitch candidate
/// - `sound_to_pitch_ac`: Compute pitch using autocorrelation method
/// - `sound_to_pitch_cc`: Compute pitch using cross-correlation method
pub use pitch::{sound_to_pitch_ac, sound_to_pitch_cc, Pitch, PitchCandidate, PitchFrame};

/// Sound loading and basic operations.
///
/// `Sound` is the foundation type for all acoustic analysis.
pub use sound::Sound;

/// Spectrogram (time-frequency) analysis types and functions.
///
/// - `Spectrogram`: Result of STFT analysis
/// - `WindowShape`: Window function selection (Gaussian, Hanning)
/// - `sound_to_spectrogram`: Compute spectrogram
pub use spectrogram::{sound_to_spectrogram, Spectrogram, WindowShape};

/// Spectrum (single-frame FFT) analysis types and functions.
///
/// - `Spectrum`: Result of FFT analysis
/// - `sound_to_spectrum`: Compute single-frame spectrum
pub use spectrum::{sound_to_spectrum, Spectrum};
