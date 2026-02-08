"""
praatfan - Clean room reimplementation of Praat's acoustic analysis algorithms.

This package provides a unified API for acoustic analysis that can use
different backends (parselmouth, praatfan_rust, praatfan_gpl, or the built-in
Python implementation).

Usage:
    from praatfan import Sound

    sound = Sound("audio.wav")
    pitch = sound.to_pitch_ac()
    formant = sound.to_formant_burg()

    # Check/change backend
    from praatfan import get_backend, set_backend
    print(get_backend())  # e.g., "praatfan"
    set_backend("parselmouth")  # Switch to parselmouth

Backend selection (in order of preference):
    1. PRAATFAN_BACKEND environment variable
    2. Config file (~/.praatfan/config.toml or ./praatfan.toml)
    3. First available: praatfan_gpl, praatfan_rust, praatfan, parselmouth

Parselmouth compatibility:
    from praatfan import Sound, call

    sound = Sound("audio.wav")
    pitch = call(sound, "To Pitch (ac)", 0, 75, 600)
    f0 = call(pitch, "Get value in frame", 10, "Hertz")

For direct access to the Python implementation (bypassing selector):
    from praatfan.sound import Sound as PythonSound
"""

# Import from local selector module
from .selector import (
    Sound,
    get_backend,
    set_backend,
    get_available_backends,
    BackendNotAvailableError,
    # Unified result types
    UnifiedPitch,
    UnifiedFormant,
    UnifiedIntensity,
    UnifiedHarmonicity,
    UnifiedSpectrum,
    UnifiedSpectrogram,
)

# Import parselmouth compatibility
from .compatibility import call, PraatCallError

# Convenient aliases for result types
Pitch = UnifiedPitch
Formant = UnifiedFormant
Intensity = UnifiedIntensity
Harmonicity = UnifiedHarmonicity
Spectrum = UnifiedSpectrum
Spectrogram = UnifiedSpectrogram

# Import praat compatibility module for use as praatfan.praat
from . import praat

__version__ = "0.1.2"
__all__ = [
    # Core types
    "Sound",
    # Backend management
    "get_backend",
    "set_backend",
    "get_available_backends",
    "BackendNotAvailableError",
    # Result types
    "Pitch",
    "Formant",
    "Intensity",
    "Harmonicity",
    "Spectrum",
    "Spectrogram",
    # Parselmouth compatibility
    "call",
    "PraatCallError",
    "praat",
]
