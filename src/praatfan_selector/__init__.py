"""
praatfan_selector - DEPRECATED: Use praatfan instead.

This package has been merged into praatfan. All functionality is now available
directly from the praatfan package. This module re-exports from praatfan for
backwards compatibility, but will be removed in a future version.

Migration:
    # Before
    from praatfan_selector import Sound, call, set_backend

    # After
    from praatfan import Sound, call, set_backend

The API is identical - no code changes required except for the import statement.
"""

import warnings

# Issue deprecation warning on import
warnings.warn(
    "praatfan_selector is deprecated and will be removed in a future version. "
    "Use 'from praatfan import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from praatfan
from praatfan import (
    Sound,
    get_backend,
    set_backend,
    get_available_backends,
    BackendNotAvailableError,
    # Result types
    Pitch,
    Formant,
    Intensity,
    Harmonicity,
    Spectrum,
    Spectrogram,
    # Parselmouth compatibility
    call,
    PraatCallError,
)

# Also export as UnifiedXxx for backwards compatibility
from praatfan.selector import (
    UnifiedPitch,
    UnifiedFormant,
    UnifiedIntensity,
    UnifiedHarmonicity,
    UnifiedSpectrum,
    UnifiedSpectrogram,
)

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
    # Unified result types (backwards compatibility)
    "UnifiedPitch",
    "UnifiedFormant",
    "UnifiedIntensity",
    "UnifiedHarmonicity",
    "UnifiedSpectrum",
    "UnifiedSpectrogram",
    # Parselmouth compatibility
    "call",
    "PraatCallError",
]
