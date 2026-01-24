"""
praatfan - Clean room reimplementation of Praat's acoustic analysis algorithms.

This is a Python-first implementation designed to match Praat/parselmouth output
exactly, then be ported to Rust for WASM compatibility.

Modules:
    sound       - Audio loading and basic operations
    spectrum    - FFT and spectral moments
    intensity   - Intensity (loudness) analysis
    pitch       - Fundamental frequency (F0) detection
    harmonicity - Harmonics-to-noise ratio (wraps pitch)
    formant     - Formant frequency analysis (Burg LPC)
    spectrogram - Time-frequency representation
"""

from .sound import Sound

__version__ = "0.1.0"
__all__ = ["Sound"]
