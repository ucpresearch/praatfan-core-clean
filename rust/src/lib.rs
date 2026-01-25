//! # praatfan
//!
//! Clean-room reimplementation of Praat's acoustic analysis algorithms.
//!
//! This library provides acoustic analysis functions equivalent to Praat,
//! implemented without access to GPL source code. The implementation is
//! based on published papers, Praat's public documentation, and black-box
//! testing against parselmouth.
//!
//! ## Supported Analysis Types
//!
//! - **Sound**: Audio loading and basic operations
//! - **Spectrum**: FFT and spectral moments
//! - **Intensity**: RMS energy contour in dB
//! - **Pitch**: F0 detection using AC and CC methods
//! - **Harmonicity**: Harmonics-to-noise ratio (HNR)
//! - **Formant**: LPC-based formant tracking
//! - **Spectrogram**: Time-frequency representation
//!
//! ## Example
//!
//! ```no_run
//! use praatfan::Sound;
//!
//! let sound = Sound::from_file("audio.wav").unwrap();
//! let pitch = sound.to_pitch_ac(0.0, 75.0, 600.0);
//! let formant = sound.to_formant_burg(0.0, 5, 5500.0, 0.025, 50.0);
//! ```

pub mod error;
pub mod formant;
pub mod harmonicity;
pub mod intensity;
pub mod pitch;
pub mod sound;
pub mod spectrogram;
pub mod spectrum;

// Re-export main types at crate root
pub use error::{Error, Result};
pub use formant::{sound_to_formant_burg, Formant, FormantFrame, FormantPoint};
pub use harmonicity::{sound_to_harmonicity_ac, sound_to_harmonicity_cc, Harmonicity};
pub use intensity::{sound_to_intensity, Intensity, Interpolation};
pub use pitch::{sound_to_pitch_ac, sound_to_pitch_cc, Pitch, PitchCandidate, PitchFrame};
pub use sound::Sound;
pub use spectrogram::{sound_to_spectrogram, Spectrogram, WindowShape};
pub use spectrum::{sound_to_spectrum, Spectrum};
