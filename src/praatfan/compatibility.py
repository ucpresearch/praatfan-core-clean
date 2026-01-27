"""
Parselmouth compatibility layer for praatfan.

This module re-exports the `call()` function from praatfan.praat, providing
parselmouth-style command syntax for acoustic analysis. This allows existing
parselmouth scripts to work with praatfan's unified types with minimal changes -
typically just changing the import statements.

Why This Exists:
----------------
Parselmouth uses a functional API based on Praat's scripting language:

    pitch = call(sound, "To Pitch (ac)", 0, 75, 600)
    f0 = call(pitch, "Get value in frame", 10, "Hertz")

This is different from the object-oriented API:

    pitch = sound.to_pitch_ac(0, 75, 600)
    f0 = pitch.get_value_in_frame(9)  # Note: 0-based index

The call() function translates Praat command strings to method calls and
handles the 1-based to 0-based index conversion automatically.

Usage:
------
    from praatfan import Sound, call

    sound = Sound("audio.wav")
    pitch = call(sound, "To Pitch (ac)", 0, 75, 600)
    f0 = call(pitch, "Get value in frame", 10, "Hertz")

Supported Commands:
-------------------
See praatfan.praat module for the full list of supported Praat commands.
The main categories are:
  - Sound creation commands: "To Pitch", "To Formant (burg)", etc.
  - Query commands: "Get value at time", "Get number of frames", etc.
  - Sound manipulation: "Extract part", "Filter (pre-emphasis)", etc.

Index Conversion:
-----------------
Parselmouth uses 1-based frame indices (like Praat) while praatfan uses
0-based indices (like Python/numpy). The call() function automatically
converts 1-based indices to 0-based when calling praatfan methods.

For example:
    call(pitch, "Get value in frame", 10, "Hertz")

Internally calls:
    pitch.get_value_in_frame(9)  # frame 10 -> index 9
"""

# =============================================================================
# Import call() from praatfan.praat
# =============================================================================
#
# The call() implementation lives in praatfan.praat (this same package).
# =============================================================================

from .praat import call, PraatCallError

# Public API
__all__ = ["call", "PraatCallError"]
