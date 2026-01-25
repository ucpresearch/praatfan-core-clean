//! Error types for praatfan.

use thiserror::Error;

/// Result type alias using praatfan's Error type.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during acoustic analysis.
#[derive(Error, Debug)]
pub enum Error {
    /// Error reading audio file.
    #[error("Failed to read audio file: {0}")]
    AudioRead(#[from] hound::Error),

    /// Error with I/O operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Audio file has unsupported format.
    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),

    /// Audio file must be mono.
    #[error("Audio must be mono (single channel), got {0} channels")]
    NotMono(u16),

    /// Invalid parameter value.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Analysis failed.
    #[error("Analysis failed: {0}")]
    AnalysisFailed(String),

    /// Resampling error.
    #[error("Resampling failed: {0}")]
    ResampleError(String),
}
