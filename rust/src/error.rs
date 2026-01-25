//! Error types for praatfan.
//!
//! This module defines the error types that can occur during acoustic analysis.
//! All errors implement `std::error::Error` via the `thiserror` crate for
//! convenient error handling and display.
//!
//! # Error Handling Philosophy
//!
//! - **Specific errors**: Each error type indicates a specific failure mode
//! - **Informative messages**: Errors include context about what went wrong
//! - **Recoverable where possible**: Use `Result<T>` to allow callers to handle errors

use thiserror::Error;

/// Result type alias using praatfan's Error type.
///
/// This is the standard return type for fallible operations in praatfan.
///
/// # Example
///
/// ```no_run
/// use praatfan::{Result, Sound};
///
/// fn load_and_analyze(path: &str) -> Result<()> {
///     let sound = Sound::from_file(path)?;
///     // ... analysis ...
///     Ok(())
/// }
/// ```
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during acoustic analysis.
///
/// This enum covers all error conditions that can arise when using praatfan:
/// - File I/O errors
/// - Format errors
/// - Parameter validation errors
/// - Analysis failures
#[derive(Error, Debug)]
pub enum Error {
    /// Error reading audio file.
    ///
    /// This wraps errors from the `hound` WAV library.
    /// Common causes:
    /// - File not found
    /// - File is not a valid WAV file
    /// - Corrupted audio data
    #[error("Failed to read audio file: {0}")]
    AudioRead(#[from] hound::Error),

    /// Error with I/O operations.
    ///
    /// General file system errors not specific to WAV reading.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Audio file has unsupported format.
    ///
    /// Currently, praatfan only supports WAV files.
    /// This error is returned for other formats.
    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),

    /// Audio file must be mono.
    ///
    /// Praatfan only supports single-channel (mono) audio.
    /// For multi-channel files, use `Sound::from_file_channel()` to
    /// explicitly select which channel to analyze.
    ///
    /// The u16 parameter contains the actual number of channels.
    #[error("Audio must be mono (single channel), got {0} channels")]
    NotMono(u16),

    /// Invalid parameter value.
    ///
    /// Returned when a function receives a parameter outside the valid range.
    /// The string contains a description of what was invalid.
    ///
    /// # Examples of invalid parameters
    ///
    /// - Negative time step
    /// - Zero or negative pitch floor
    /// - Max formant frequency above Nyquist
    /// - Channel index out of range
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Analysis failed.
    ///
    /// Returned when an analysis algorithm fails to produce results.
    /// This can happen due to:
    /// - Numerical instability
    /// - Degenerate input (e.g., silent audio)
    /// - Algorithm-specific edge cases
    #[error("Analysis failed: {0}")]
    AnalysisFailed(String),

    /// Resampling error.
    ///
    /// Returned when audio resampling fails.
    /// This is used by formant analysis which resamples to 2× max_formant_hz.
    #[error("Resampling failed: {0}")]
    ResampleError(String),
}
