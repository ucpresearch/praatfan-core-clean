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

# Try to import from praatfan.praat (Python version)
# If Rust version is installed, praatfan.praat won't exist
try:
    from praatfan.praat import call, PraatCallError
except ImportError:
    # Rust praatfan doesn't have praat module - define PraatCallError here
    # and import call from a local implementation or raise helpful error
    class PraatCallError(Exception):
        """Error from Praat call() function."""
        pass

    def call(*args, **kwargs):
        raise NotImplementedError(
            "call() requires the Python praatfan package. "
            "The Rust praatfan package is installed but doesn't include the praat compatibility layer. "
            "Install the Python version: pip install praatfan (pure Python wheel)"
        )

__all__ = ["call", "PraatCallError"]
