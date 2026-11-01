//! Vendored smallft (FFTPACK port): real-valued FFT used by the resampler.
//!
//! Lineage: FFTPACK (Paul N. Swarztrauber, NCAR, 1985) -> `smallft.c` from
//! OggSquish/Ogg Vorbis, as carried in Xiph.Org's speexdsp (with `fftwrap.c`
//! by Jean-Marc Valin) -> c2rust translation in speexdsp-rs (Luca Barbato):
//! <https://github.com/rust-av/speexdsp-rs/tree/master/fft/src>.
//!
//! Vendored here at f64 precision with two operator-precedence / truncated-
//! constant fixes documented in `docs/TRANSFERABLE_FINDINGS.md`.
//!
//! Licenses (all permissive, MIT-compatible; NOT public domain):
//! - FFTPACK: UCAR BSD-style license (NCAR's FFTPACK curators state that the
//!   FFTPACK5 license covers the version 4 sources);
//! - smallft.c / fftwrap.c: Xiph.Org BSD license;
//! - speexdsp-rs: MIT.
//! Full texts: `rust/third-party/texts/` (fftpack5-UCAR.txt, speexdsp-COPYING,
//! vorbis-COPYING, speexdsp-rs-LICENSE), reproduced for binary distributions
//! in `rust/LICENSE-THIRD-PARTY`.

pub(crate) mod dradb;
pub(crate) mod dradf;
pub(crate) mod fftwrap;
pub(crate) mod smallft;

pub use fftwrap::DrftLookup;
pub(crate) use smallft::{drftb1, drftf1, fdrffti};
