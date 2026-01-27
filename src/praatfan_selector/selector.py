"""
DEPRECATED: Use praatfan.selector instead.

This module re-exports from praatfan.selector for backwards compatibility.
"""

import warnings

warnings.warn(
    "praatfan_selector.selector is deprecated. Use praatfan.selector instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from praatfan.selector
from praatfan.selector import *
from praatfan.selector import (
    BackendNotAvailableError,
    UnifiedPitch,
    UnifiedFormant,
    UnifiedIntensity,
    UnifiedHarmonicity,
    UnifiedSpectrum,
    UnifiedSpectrogram,
    get_backend,
    set_backend,
    get_available_backends,
    Sound,
)
