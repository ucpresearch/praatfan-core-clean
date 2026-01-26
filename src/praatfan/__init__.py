"""
praatfan - Clean room reimplementation of Praat's acoustic analysis algorithms.

This package provides a unified API for acoustic analysis that can use
different backends (parselmouth, praatfan-core, or the built-in Python/Rust
implementations).

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
    3. First available: praatfan-rust, praatfan, parselmouth, praatfan-core

For direct access to the Python implementation (bypassing selector):
    from praatfan.sound import Sound as PythonSound
"""

# Try to use the selector for unified backend support
try:
    from praatfan_selector import (
        Sound,
        get_backend,
        set_backend,
        get_available_backends,
        BackendNotAvailableError,
        Pitch,
        Formant,
        Intensity,
        Harmonicity,
        Spectrum,
        Spectrogram,
    )
    _HAS_SELECTOR = True
except ImportError:
    # Selector not available, use native Python implementation
    from .sound import Sound
    _HAS_SELECTOR = False

    def get_backend():
        """Get the current backend name."""
        return "praatfan"

    def set_backend(name):
        """Set the backend (no-op when selector not installed)."""
        if name != "praatfan":
            raise ImportError(
                f"Backend '{name}' not available. Install praatfan_selector for "
                "multi-backend support, or install the requested backend directly."
            )

    def get_available_backends():
        """Return list of available backends."""
        return ["praatfan"]

    class BackendNotAvailableError(Exception):
        """Raised when no suitable backend is available."""
        pass

    # Re-export result types from Python implementation
    from .pitch import Pitch
    from .formant import Formant
    from .intensity import Intensity
    from .harmonicity import Harmonicity
    from .spectrum import Spectrum
    from .spectrogram import Spectrogram

# Import praat compatibility module for use as praatfan.praat
from . import praat

__version__ = "0.1.0"
__all__ = [
    "Sound",
    "get_backend",
    "set_backend",
    "get_available_backends",
    "BackendNotAvailableError",
    "Pitch",
    "Formant",
    "Intensity",
    "Harmonicity",
    "Spectrum",
    "Spectrogram",
    "praat",
]
