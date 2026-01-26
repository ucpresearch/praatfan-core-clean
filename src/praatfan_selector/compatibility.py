"""Parselmouth compatibility layer for praatfan_selector.

This module provides a `call()` function that emulates parselmouth's
functional API, working with praatfan_selector's unified types. This
allows existing parselmouth scripts to work with any backend (Python,
Rust, parselmouth, praatfan-core) with minimal changes.

Usage:
    from praatfan_selector import Sound
    from praatfan_selector.compatibility import call

    # Instead of parselmouth:
    # import parselmouth
    # from parselmouth.praat import call

    sound = Sound("audio.wav")
    pitch = call(sound, "To Pitch (ac)", 0, 75, 600)
    f0 = call(pitch, "Get value in frame", 10, "Hertz")

Note:
    Parselmouth uses 1-based frame indices while praatfan uses 0-based.
    This compatibility layer handles the conversion automatically.
"""

# Re-export from praatfan.praat - it's designed to be flexible and works
# with any objects that have the expected methods (xs, values, n_frames, etc.)
# The unified types from praatfan_selector have these methods.
from praatfan.praat import call, PraatCallError

__all__ = ["call", "PraatCallError"]
