"""
praatfan-selector - Backend selector for acoustic analysis.

Provides a unified API that can use different backends:
- parselmouth (Praat Python bindings, GPL)
- praatfan-core-rs (original Rust implementation)
- praatfan (clean-room Python implementation)
- praatfan-rust (clean-room Rust implementation with PyO3)

Backend selection priority:
1. PRAATFAN_BACKEND environment variable
2. Configuration file (~/.praatfan/config.toml or ./praatfan.toml)
3. First available backend in order: praatfan-rust, praatfan, parselmouth, praatfan-core-rs

Usage:
    from praatfan_selector import Sound

    sound = Sound.from_file("audio.wav")
    pitch = sound.to_pitch_ac()
    formant = sound.to_formant_burg()

    # Or explicitly select backend:
    from praatfan_selector import set_backend, get_backend
    set_backend("parselmouth")
    print(get_backend())  # "parselmouth"

Parselmouth compatibility:
    from praatfan_selector import Sound, call

    sound = Sound("audio.wav")
    pitch = call(sound, "To Pitch (ac)", 0, 75, 600)
    f0 = call(pitch, "Get value in frame", 10, "Hertz")
"""

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
from .compatibility import call, PraatCallError

# Convenient aliases
Pitch = UnifiedPitch
Formant = UnifiedFormant
Intensity = UnifiedIntensity
Harmonicity = UnifiedHarmonicity
Spectrum = UnifiedSpectrum
Spectrogram = UnifiedSpectrogram

__version__ = "0.1.0"
__all__ = [
    "Sound",
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
]
