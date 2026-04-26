//! Vendored smallft (FFTPACK port) — public-domain real-valued FFT.
//!
//! Originally from Xiph/Vorbis `smallft.c`, c2rust-translated by speexdsp-rs:
//! <https://github.com/rust-av/speexdsp-rs/tree/master/fft/src>.
//!
//! Vendored here at f64 precision with two operator-precedence / truncated-
//! constant fixes documented in `docs/TRANSFERABLE_FINDINGS.md`.
//!
//! Public-domain origin (FFTPACK, Paul N. Swarztrauber 1985); the speexdsp
//! wrapper is BSD-licensed.

pub(crate) mod dradb;
pub(crate) mod dradf;
pub(crate) mod fftwrap;
pub(crate) mod smallft;

pub use fftwrap::DrftLookup;
