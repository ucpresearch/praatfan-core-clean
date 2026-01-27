"""
Parselmouth compatibility layer for praatfan_selector.

This module re-exports the `call()` function from praatfan.praat, providing
parselmouth-style command syntax for acoustic analysis. This allows existing
parselmouth scripts to work with praatfan_selector's unified types with
minimal changes - typically just changing the import statements.

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
    from praatfan_selector import Sound
    from praatfan_selector.compatibility import call

    # Instead of:
    # import parselmouth
    # from parselmouth.praat import call

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

Rust Backend Limitation:
------------------------
The call() function requires the Python praatfan package because it contains
the command parsing and dispatch logic. If only the Rust praatfan package is
installed (which provides the praatfan module but not praatfan.praat), using
call() will raise NotImplementedError with instructions to install the Python
version.

To use call() with praatfan_selector:
  1. Install the pure Python wheel: pip install praatfan (py3-none-any.whl)
  2. Or install alongside Rust version using PYTHONPATH

Alternatively, use the object-oriented API which works with all backends:
    pitch = sound.to_pitch_ac(time_step=0, pitch_floor=75, pitch_ceiling=600)
    f0 = pitch.values()[9]  # 0-based index
"""

# =============================================================================
# Import call() from praatfan.praat
# =============================================================================
#
# The call() implementation lives in praatfan.praat (Python version).
# If the Rust version of praatfan is installed, it won't have the praat
# submodule, so we provide a helpful error message.
# =============================================================================

try:
    # Try to import from praatfan.praat (Python version)
    from praatfan.praat import call, PraatCallError
except ImportError:
    # Rust praatfan doesn't have the praat module.
    # Define fallbacks that give helpful error messages.

    class PraatCallError(Exception):
        """
        Error raised when a Praat call() command fails.

        This is a placeholder for compatibility when the Python praatfan
        package is not installed. The actual implementation is in
        praatfan.praat.PraatCallError.
        """
        pass

    def call(*args, **kwargs):
        """
        Placeholder for call() when Python praatfan is not installed.

        Raises:
            NotImplementedError: Always, with instructions for installing
                the Python praatfan package.
        """
        raise NotImplementedError(
            "call() requires the Python praatfan package. "
            "The Rust praatfan package is installed but doesn't include "
            "the praat compatibility layer. "
            "Install the Python version: pip install praatfan (pure Python wheel)"
        )


# Public API
__all__ = ["call", "PraatCallError"]
