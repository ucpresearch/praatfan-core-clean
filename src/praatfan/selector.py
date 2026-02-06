"""
Backend selector and unified API for acoustic analysis.

This module provides a unified interface for acoustic analysis that works with
multiple backends (parselmouth, praatfan, praatfan_rust, praatfan_gpl). The
key design principle is that user code should work identically regardless of
which backend is installed or selected.

Architecture Overview:
----------------------
1. Unified Result Types (UnifiedPitch, UnifiedFormant, etc.)
   - Wrap backend-specific result objects
   - Provide consistent API for accessing analysis results
   - Handle differences in method names, return types, and data structures

2. Backend Detection Functions (_try_import_*)
   - Check availability of each backend
   - Distinguish between Python and Rust versions of praatfan

3. Backend Adapters (ParselmouthSound, PraatfanPythonSound, etc.)
   - Wrap backend-specific Sound objects
   - Translate unified API calls to backend-specific method calls
   - Handle parameter differences (e.g., 0.0 vs None for auto time_step)

4. Unified Sound Class
   - Public API that users interact with
   - Delegates to appropriate backend adapter
   - Provides fallback implementations for methods not in all backends

Backend Selection Priority:
---------------------------
1. PRAATFAN_BACKEND environment variable (if set)
2. Config file (~/.praatfan/config.toml or ./praatfan.toml)
3. Auto-detect in order: praatfan_gpl, praatfan_rust, praatfan, parselmouth

Usage:
------
    from praatfan import Sound, set_backend, get_available_backends

    # Check available backends
    print(get_available_backends())  # ['praatfan', 'parselmouth', ...]

    # Load sound (uses auto-selected backend)
    sound = Sound("audio.wav")

    # Analyze
    pitch = sound.to_pitch_ac()
    formant = sound.to_formant_burg()

    # Access results (same API regardless of backend)
    f0_values = pitch.values()  # numpy array
    times = pitch.xs()          # numpy array

    # Switch backends at runtime
    set_backend("parselmouth")
"""

import os
from pathlib import Path
from typing import Optional, List, Union
from abc import ABC, abstractmethod

import numpy as np


class BackendNotAvailableError(Exception):
    """Raised when the requested backend is not installed or available."""
    pass


# =============================================================================
# Unified Result Types
# =============================================================================
#
# These classes wrap backend-specific result objects and provide a consistent
# API. Each backend may have different:
#   - Method names (xs() vs times(), values() vs selected_array)
#   - Return types (numpy array vs list, 1D vs 2D)
#   - Property names (n_frames vs num_frames)
#
# The Unified* classes normalize these differences so user code works the same
# regardless of which backend produced the result.
# =============================================================================

class UnifiedPitch:
    """Unified Pitch result - same API regardless of backend."""

    def __init__(self, inner, backend: str):
        self._inner = inner
        self._backend = backend

    def xs(self) -> np.ndarray:
        """Time values for each frame."""
        if self._backend == "parselmouth":
            return np.array(self._inner.xs())
        elif self._backend == "praatfan":
            return self._inner.times()
        elif self._backend == "praatfan_rust":
            return np.array(self._inner.xs())
        elif self._backend == "praatfan_gpl":
            return np.array(self._inner.times())
        raise ValueError(f"Unknown backend: {self._backend}")

    def values(self) -> np.ndarray:
        """F0 values (Hz) for each frame. NaN for unvoiced."""
        if self._backend == "parselmouth":
            return self._inner.selected_array['frequency']
        elif self._backend == "praatfan":
            return self._inner.values()
        elif self._backend == "praatfan_rust":
            return np.array(self._inner.values())
        elif self._backend == "praatfan_gpl":
            return np.array(self._inner.values())
        raise ValueError(f"Unknown backend: {self._backend}")

    def strengths(self) -> np.ndarray:
        """Voicing strength for each frame."""
        if self._backend == "parselmouth":
            return self._inner.selected_array['strength']
        elif self._backend == "praatfan":
            return self._inner.strengths()
        elif self._backend == "praatfan_rust":
            return np.array(self._inner.selected_array['strength'])
        elif self._backend == "praatfan_gpl":
            # praatfan_gpl doesn't have strengths, return ones for voiced frames
            vals = np.array(self._inner.values())
            return np.where(np.isnan(vals), 0.0, 1.0)
        raise ValueError(f"Unknown backend: {self._backend}")

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        if self._backend == "parselmouth":
            return self._inner.n_frames
        elif self._backend == "praatfan":
            return self._inner.n_frames
        elif self._backend == "praatfan_rust":
            return self._inner.n_frames
        elif self._backend == "praatfan_gpl":
            return self._inner.num_frames
        raise ValueError(f"Unknown backend: {self._backend}")

    @property
    def time_step(self) -> float:
        """Time step between frames."""
        if self._backend == "parselmouth":
            return self._inner.time_step
        elif self._backend == "praatfan":
            return self._inner.time_step
        elif self._backend == "praatfan_rust":
            return self._inner.time_step
        elif self._backend == "praatfan_gpl":
            return self._inner.time_step
        raise ValueError(f"Unknown backend: {self._backend}")

    @property
    def backend(self) -> str:
        return self._backend

    def __repr__(self):
        return f"Pitch<{self._backend}>({self.n_frames} frames)"


class UnifiedFormant:
    """Unified Formant result - same API regardless of backend."""

    def __init__(self, inner, backend: str):
        self._inner = inner
        self._backend = backend

    def xs(self) -> np.ndarray:
        """Time values for each frame."""
        if self._backend == "parselmouth":
            return np.array(self._inner.xs())
        elif self._backend == "praatfan":
            return self._inner.times()
        elif self._backend == "praatfan_rust":
            return np.array(self._inner.xs())
        elif self._backend == "praatfan_gpl":
            return np.array(self._inner.times())
        raise ValueError(f"Unknown backend: {self._backend}")

    def formant_values(self, formant_number: int) -> np.ndarray:
        """Get F1, F2, F3, etc. values (formant_number is 1-indexed)."""
        if self._backend == "parselmouth":
            # parselmouth uses to_array() but it's different
            from parselmouth.praat import call
            n = self._inner.n_frames
            result = np.zeros(n)
            for i in range(n):
                t = call(self._inner, "Get time from frame number", i + 1)
                val = call(self._inner, "Get value at time", formant_number, t, "Hertz", "Linear")
                result[i] = val if val is not None else np.nan
            return result
        elif self._backend == "praatfan":
            return self._inner.formant_values(formant_number)
        elif self._backend == "praatfan_rust":
            return np.array(self._inner.to_array(formant_number))
        elif self._backend == "praatfan_gpl":
            return np.array(self._inner.formant_values(formant_number))
        raise ValueError(f"Unknown backend: {self._backend}")

    def bandwidth_values(self, formant_number: int) -> np.ndarray:
        """Get bandwidth values for a formant (formant_number is 1-indexed)."""
        if self._backend == "parselmouth":
            from parselmouth.praat import call
            n = self._inner.n_frames
            result = np.zeros(n)
            for i in range(n):
                t = call(self._inner, "Get time from frame number", i + 1)
                val = call(self._inner, "Get bandwidth at time", formant_number, t, "Hertz", "Linear")
                result[i] = val if val is not None else np.nan
            return result
        elif self._backend == "praatfan":
            return self._inner.bandwidth_values(formant_number)
        elif self._backend == "praatfan_rust":
            return np.array(self._inner.to_bandwidth_array(formant_number))
        elif self._backend == "praatfan_gpl":
            return np.array(self._inner.bandwidth_values(formant_number))
        raise ValueError(f"Unknown backend: {self._backend}")

    @property
    def n_frames(self) -> int:
        if self._backend == "praatfan_gpl":
            return self._inner.num_frames
        return self._inner.n_frames

    @property
    def time_step(self) -> float:
        if self._backend == "parselmouth":
            return self._inner.time_step
        return self._inner.time_step

    @property
    def backend(self) -> str:
        return self._backend

    def __repr__(self):
        try:
            return f"Formant<{self._backend}>({self.n_frames} frames)"
        except AttributeError:
            return f"Formant<{self._backend}>"


class UnifiedIntensity:
    """Unified Intensity result - same API regardless of backend."""

    def __init__(self, inner, backend: str):
        self._inner = inner
        self._backend = backend

    def xs(self) -> np.ndarray:
        """Time values for each frame."""
        if self._backend == "parselmouth":
            return np.array(self._inner.xs())
        elif self._backend == "praatfan":
            return self._inner.times
        elif self._backend == "praatfan_rust":
            return np.array(self._inner.xs())
        elif self._backend == "praatfan_gpl":
            return np.array(self._inner.times())
        raise ValueError(f"Unknown backend: {self._backend}")

    def values(self) -> np.ndarray:
        """Intensity values in dB."""
        if self._backend == "parselmouth":
            return self._inner.values[0]  # parselmouth returns 2D
        elif self._backend == "praatfan":
            return self._inner.values
        elif self._backend == "praatfan_rust":
            return np.array(self._inner.values())
        elif self._backend == "praatfan_gpl":
            return np.array(self._inner.values())
        raise ValueError(f"Unknown backend: {self._backend}")

    @property
    def n_frames(self) -> int:
        if self._backend == "praatfan_gpl":
            return self._inner.num_frames
        return self._inner.n_frames

    @property
    def time_step(self) -> float:
        if self._backend == "parselmouth":
            return self._inner.time_step
        return self._inner.time_step

    @property
    def backend(self) -> str:
        return self._backend

    def __repr__(self):
        return f"Intensity<{self._backend}>({self.n_frames} frames)"


class UnifiedHarmonicity:
    """Unified Harmonicity (HNR) result - same API regardless of backend."""

    def __init__(self, inner, backend: str):
        self._inner = inner
        self._backend = backend

    def xs(self) -> np.ndarray:
        """Time values for each frame."""
        if self._backend == "parselmouth":
            return np.array(self._inner.xs())
        elif self._backend == "praatfan":
            return self._inner.times
        elif self._backend == "praatfan_rust":
            return np.array(self._inner.xs())
        elif self._backend == "praatfan_gpl":
            return np.array(self._inner.times())
        raise ValueError(f"Unknown backend: {self._backend}")

    def values(self) -> np.ndarray:
        """HNR values in dB."""
        if self._backend == "parselmouth":
            return self._inner.values[0]  # parselmouth returns 2D
        elif self._backend == "praatfan":
            return self._inner.values
        elif self._backend == "praatfan_rust":
            return np.array(self._inner.values())
        elif self._backend == "praatfan_gpl":
            return np.array(self._inner.values())
        raise ValueError(f"Unknown backend: {self._backend}")

    @property
    def n_frames(self) -> int:
        if self._backend == "praatfan_gpl":
            return self._inner.num_frames
        return self._inner.n_frames

    @property
    def time_step(self) -> float:
        if self._backend == "parselmouth":
            return self._inner.time_step
        return self._inner.time_step

    @property
    def backend(self) -> str:
        return self._backend

    def __repr__(self):
        return f"Harmonicity<{self._backend}>({self.n_frames} frames)"


class UnifiedSpectrum:
    """Unified Spectrum result - same API regardless of backend."""

    def __init__(self, inner, backend: str):
        self._inner = inner
        self._backend = backend

    def xs(self) -> np.ndarray:
        """Frequency values for each bin."""
        if self._backend == "parselmouth":
            return np.array(self._inner.xs())
        elif self._backend == "praatfan":
            # Compute frequencies from df and n_bins
            return np.arange(self._inner.n_bins) * self._inner.df
        elif self._backend == "praatfan_rust":
            return np.array(self._inner.xs())
        elif self._backend == "praatfan_gpl":
            # praatfan_gpl has num_bins not n_bins, and no xs() method
            return np.arange(self._inner.num_bins) * self._inner.df
        raise ValueError(f"Unknown backend: {self._backend}")

    def values(self) -> np.ndarray:
        """Complex spectrum values."""
        if self._backend == "parselmouth":
            return self._inner.values[0]  # parselmouth returns 2D
        elif self._backend == "praatfan":
            return self._inner.real + 1j * self._inner.imag
        elif self._backend == "praatfan_rust":
            return np.array(self._inner.real()) + 1j * np.array(self._inner.imag())
        elif self._backend == "praatfan_gpl":
            return np.array(self._inner.real()) + 1j * np.array(self._inner.imag())
        raise ValueError(f"Unknown backend: {self._backend}")

    def get_center_of_gravity(self, power: float = 2.0) -> float:
        """Spectral centroid."""
        if self._backend == "parselmouth":
            return self._inner.get_centre_of_gravity(power)
        elif self._backend == "praatfan":
            return self._inner.get_center_of_gravity(power)
        elif self._backend == "praatfan_rust":
            return self._inner.get_center_of_gravity(power)
        elif self._backend == "praatfan_gpl":
            return self._inner.get_center_of_gravity(power)
        raise ValueError(f"Unknown backend: {self._backend}")

    def get_standard_deviation(self, power: float = 2.0) -> float:
        """Spectral standard deviation."""
        if self._backend == "parselmouth":
            return self._inner.get_standard_deviation(power)
        elif self._backend == "praatfan":
            return self._inner.get_standard_deviation(power)
        elif self._backend == "praatfan_rust":
            return self._inner.get_standard_deviation(power)
        elif self._backend == "praatfan_gpl":
            return self._inner.get_standard_deviation(power)
        raise ValueError(f"Unknown backend: {self._backend}")

    def get_skewness(self, power: float = 2.0) -> float:
        """Spectral skewness."""
        if self._backend == "parselmouth":
            return self._inner.get_skewness(power)
        elif self._backend == "praatfan":
            return self._inner.get_skewness(power)
        elif self._backend == "praatfan_rust":
            return self._inner.get_skewness(power)
        elif self._backend == "praatfan_gpl":
            return self._inner.get_skewness(power)
        raise ValueError(f"Unknown backend: {self._backend}")

    def get_kurtosis(self, power: float = 2.0) -> float:
        """Spectral kurtosis."""
        if self._backend == "parselmouth":
            return self._inner.get_kurtosis(power)
        elif self._backend == "praatfan":
            return self._inner.get_kurtosis(power)
        elif self._backend == "praatfan_rust":
            return self._inner.get_kurtosis(power)
        elif self._backend == "praatfan_gpl":
            return self._inner.get_kurtosis(power)
        raise ValueError(f"Unknown backend: {self._backend}")

    def get_band_energy(self, f_min: float = 0.0, f_max: float = 0.0) -> float:
        """Band energy between frequencies."""
        if self._backend == "parselmouth":
            return self._inner.get_band_energy(f_min, f_max)
        elif self._backend == "praatfan":
            return self._inner.get_band_energy(f_min, f_max)
        elif self._backend == "praatfan_rust":
            return self._inner.get_band_energy(f_min, f_max)
        elif self._backend == "praatfan_gpl":
            return self._inner.get_band_energy(f_min, f_max)
        raise ValueError(f"Unknown backend: {self._backend}")

    @property
    def n_bins(self) -> int:
        if self._backend == "praatfan_gpl":
            return self._inner.num_bins
        return self._inner.n_bins

    @property
    def df(self) -> float:
        """Frequency resolution."""
        return self._inner.df

    @property
    def backend(self) -> str:
        return self._backend

    def __repr__(self):
        return f"Spectrum<{self._backend}>({self.n_bins} bins)"


class UnifiedSpectrogram:
    """Unified Spectrogram result - same API regardless of backend."""

    def __init__(self, inner, backend: str):
        self._inner = inner
        self._backend = backend

    def xs(self) -> np.ndarray:
        """Time values."""
        if self._backend == "parselmouth":
            return np.array(self._inner.xs())
        elif self._backend == "praatfan":
            return self._inner.times()
        elif self._backend == "praatfan_rust":
            return np.array(self._inner.xs())
        elif self._backend == "praatfan_gpl":
            # praatfan_gpl uses get_time_from_frame (0-indexed)
            n = self._inner.num_frames
            return np.array([self._inner.get_time_from_frame(i) for i in range(n)])
        raise ValueError(f"Unknown backend: {self._backend}")

    def ys(self) -> np.ndarray:
        """Frequency values."""
        if self._backend == "parselmouth":
            return np.array(self._inner.ys())
        elif self._backend == "praatfan":
            return self._inner.frequencies()
        elif self._backend == "praatfan_rust":
            return np.array(self._inner.ys())
        elif self._backend == "praatfan_gpl":
            # praatfan_gpl uses get_frequency_from_bin (0-indexed)
            n = self._inner.num_freq_bins
            return np.array([self._inner.get_frequency_from_bin(i) for i in range(n)])
        raise ValueError(f"Unknown backend: {self._backend}")

    def values(self) -> np.ndarray:
        """Power values as 2D array (frequency × time)."""
        if self._backend == "parselmouth":
            return np.array(self._inner.values)
        elif self._backend == "praatfan":
            return self._inner.values
        elif self._backend == "praatfan_rust":
            # Reshape from 1D to 2D
            vals = np.array(self._inner.values())
            return vals.reshape(self._inner.n_freqs, self._inner.n_times)
        elif self._backend == "praatfan_gpl":
            vals = np.array(self._inner.values())
            return vals.reshape(self._inner.num_freq_bins, self._inner.num_frames)
        raise ValueError(f"Unknown backend: {self._backend}")

    @property
    def n_times(self) -> int:
        if self._backend == "praatfan_gpl":
            return self._inner.num_frames
        return self._inner.n_times

    @property
    def n_freqs(self) -> int:
        if self._backend == "parselmouth":
            return self._inner.n_frequencies
        elif self._backend == "praatfan_gpl":
            return self._inner.num_freq_bins
        return self._inner.n_freqs

    @property
    def time_step(self) -> float:
        return self._inner.time_step

    @property
    def backend(self) -> str:
        return self._backend

    def __repr__(self):
        return f"Spectrogram<{self._backend}>({self.n_times} times × {self.n_freqs} freqs)"


# =============================================================================
# Backend Detection
# =============================================================================
#
# These functions detect which acoustic analysis backends are installed.
# The challenge is that both Python and Rust versions of praatfan use the
# same package name 'praatfan', so we need to distinguish them by checking
# for native extension files (.so/.pyd/.dylib).
#
# Available backends:
#   - parselmouth: Python bindings to Praat (GPL licensed)
#   - praatfan: Pure Python clean-room implementation (MIT licensed)
#   - praatfan_rust: Rust implementation with PyO3 bindings (MIT licensed)
#   - praatfan_gpl: Separate Rust implementation (MIT licensed)
# =============================================================================

def _try_import_parselmouth() -> bool:
    """
    Check if parselmouth (Praat Python bindings) is installed.

    Returns:
        True if parselmouth can be imported, False otherwise.
    """
    try:
        import parselmouth
        return True
    except ImportError:
        return False


def _try_import_praatfan_gpl() -> bool:
    """
    Check if praatfan_gpl (original Rust implementation) is installed.

    praatfan_gpl is a separate package from praatfan, with its own
    namespace (praatfan_gpl).

    Returns:
        True if praatfan_gpl can be imported, False otherwise.
    """
    try:
        import praatfan_gpl
        return True
    except ImportError:
        return False


def _try_import_praatfan() -> bool:
    """
    Check if praatfan (Python) is installed.

    This checks specifically for the Python version by looking for
    the sound submodule which only exists in the Python implementation.

    Returns:
        True if praatfan Python can be imported, False otherwise.
    """
    try:
        from praatfan import sound
        return True
    except ImportError:
        return False


def _try_import_praatfan_rust() -> bool:
    """
    Check if praatfan_rust (Rust/PyO3 bindings) is installed.

    Returns:
        True if praatfan_rust can be imported, False otherwise.
    """
    try:
        import praatfan_rust
        return True
    except ImportError:
        return False


def get_available_backends() -> List[str]:
    """
    Return list of available backend names.

    Checks which acoustic analysis backends are installed and returns
    their names. The order in the returned list does not indicate
    preference - see _select_backend() for preference order.

    Returns:
        List of backend names, e.g., ['praatfan', 'parselmouth']
    """
    available = []

    # Check for praatfan_rust (separate package)
    if _try_import_praatfan_rust():
        available.append("praatfan_rust")

    # Check for praatfan (Python)
    if _try_import_praatfan():
        available.append("praatfan")

    if _try_import_parselmouth():
        available.append("parselmouth")

    if _try_import_praatfan_gpl():
        available.append("praatfan_gpl")

    return available


# =============================================================================
# Configuration
# =============================================================================
#
# Backend selection follows this priority:
#   1. Programmatic: set_backend() called by user code
#   2. Environment: PRAATFAN_BACKEND environment variable
#   3. Config file: ./praatfan.toml or ~/.praatfan/config.toml
#   4. Auto-detect: First available in preference order
#
# The preference order for auto-detection is:
#   praatfan_gpl > praatfan_rust > praatfan > parselmouth
#
# This prioritizes MIT-licensed backends and Rust implementations for
# performance, while still supporting parselmouth as a fallback.
# =============================================================================

# Global state: currently selected backend (None = not yet selected)
_current_backend: Optional[str] = None


def _read_config_file() -> Optional[str]:
    """
    Read backend preference from TOML config file.

    Checks for config files in this order:
      1. ./praatfan.toml (project-local config)
      2. ~/.praatfan/config.toml (user config)

    Config file format:
        backend = "parselmouth"

    Returns:
        Backend name from config, or None if no config file found.
    """
    # Try local config first (project-specific settings)
    local_config = Path("praatfan.toml")
    if local_config.exists():
        return _parse_toml_backend(local_config)

    # Try user config (global user preferences)
    user_config = Path.home() / ".praatfan" / "config.toml"
    if user_config.exists():
        return _parse_toml_backend(user_config)

    return None


def _parse_toml_backend(path: Path) -> Optional[str]:
    """
    Parse backend name from a TOML config file.

    Tries multiple TOML parsers in order of preference:
      1. tomllib (Python 3.11+ built-in)
      2. tomli (popular third-party library)
      3. Simple regex-based fallback parser

    Args:
        path: Path to the TOML config file.

    Returns:
        Backend name if found in config, None otherwise.
    """
    try:
        # Try tomllib (Python 3.11+) or tomli
        try:
            import tomllib
            with open(path, "rb") as f:
                config = tomllib.load(f)
        except ImportError:
            try:
                import tomli
                with open(path, "rb") as f:
                    config = tomli.load(f)
            except ImportError:
                # Fall back to simple parsing
                return _simple_toml_parse(path)

        return config.get("backend")
    except Exception:
        return None


def _simple_toml_parse(path: Path) -> Optional[str]:
    """Simple TOML parser for just the backend key."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("backend"):
                    # backend = "value" or backend = 'value'
                    if "=" in line:
                        value = line.split("=", 1)[1].strip()
                        # Remove quotes
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            return value[1:-1]
                        return value
    except Exception:
        pass
    return None


def _select_backend() -> str:
    """Select backend based on configuration and availability."""
    global _current_backend

    if _current_backend is not None:
        return _current_backend

    # 1. Check environment variable
    env_backend = os.environ.get("PRAATFAN_BACKEND")
    if env_backend:
        available = get_available_backends()
        if env_backend in available:
            _current_backend = env_backend
            return _current_backend
        raise BackendNotAvailableError(
            f"Requested backend '{env_backend}' is not available. "
            f"Available: {available}"
        )

    # 2. Check config file
    config_backend = _read_config_file()
    if config_backend:
        available = get_available_backends()
        if config_backend in available:
            _current_backend = config_backend
            return _current_backend
        # Config specifies unavailable backend - warn but continue
        import warnings
        warnings.warn(
            f"Configured backend '{config_backend}' is not available. "
            f"Falling back to auto-detection."
        )

    # 3. Auto-detect: prefer in order
    available = get_available_backends()
    if not available:
        raise BackendNotAvailableError(
            "No acoustic analysis backend available. Install one of: "
            "praatfan, praat-parselmouth, praatfan_gpl"
        )

    # Preference order
    preference = ["praatfan_gpl", "praatfan_rust", "praatfan", "parselmouth"]
    for backend in preference:
        if backend in available:
            _current_backend = backend
            return _current_backend

    # Fallback to first available
    _current_backend = available[0]
    return _current_backend


def get_backend() -> str:
    """Get the current backend name."""
    return _select_backend()


def set_backend(name: str) -> None:
    """
    Set the backend to use.

    Args:
        name: Backend name ("parselmouth", "praatfan", "praatfan_rust", "praatfan_gpl")

    Raises:
        BackendNotAvailableError: If the backend is not installed
    """
    global _current_backend

    available = get_available_backends()
    if name not in available:
        raise BackendNotAvailableError(
            f"Backend '{name}' is not available. Available: {available}"
        )

    _current_backend = name


# =============================================================================
# Backend Adapters
# =============================================================================
#
# Each backend has its own Sound class with slightly different APIs:
#   - Different method signatures (time_step=0.0 vs time_step=None)
#   - Different property names (sample_rate vs sampling_frequency)
#   - Different return types (numpy array vs list)
#
# The adapter classes normalize these differences by:
#   1. Inheriting from BaseSound (defines the expected interface)
#   2. Translating unified API calls to backend-specific calls
#   3. Wrapping results in Unified* classes for consistent access
#
# Each adapter has two factory methods:
#   - from_file(path): Load audio from a file
#   - from_samples(samples, sr): Create from numpy array
# =============================================================================

class BaseSound(ABC):
    """
    Abstract base class defining the Sound interface.

    All backend adapters must implement these methods. This ensures
    that the unified Sound class can delegate to any backend adapter
    without knowing which specific backend is being used.
    """

    @abstractmethod
    def to_pitch_ac(self, time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0) -> UnifiedPitch:
        pass

    @abstractmethod
    def to_pitch_cc(self, time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0) -> UnifiedPitch:
        pass

    @abstractmethod
    def to_formant_burg(self, time_step=0.0, max_number_of_formants=5,
                        maximum_formant=5500.0, window_length=0.025,
                        pre_emphasis_from=50.0) -> UnifiedFormant:
        pass

    @abstractmethod
    def to_intensity(self, minimum_pitch=100.0, time_step=0.0) -> UnifiedIntensity:
        pass

    @abstractmethod
    def to_harmonicity_ac(self, time_step=0.01, minimum_pitch=75.0,
                          silence_threshold=0.1, periods_per_window=4.5) -> UnifiedHarmonicity:
        pass

    @abstractmethod
    def to_harmonicity_cc(self, time_step=0.01, minimum_pitch=75.0,
                          silence_threshold=0.1, periods_per_window=1.0) -> UnifiedHarmonicity:
        pass

    @abstractmethod
    def to_spectrum(self, fast=True) -> UnifiedSpectrum:
        pass

    @abstractmethod
    def to_spectrogram(self, window_length=0.005, maximum_frequency=5000.0,
                       time_step=0.002, frequency_step=20.0) -> UnifiedSpectrogram:
        pass

    @property
    @abstractmethod
    def n_samples(self) -> int:
        pass

    @property
    @abstractmethod
    def sampling_frequency(self) -> float:
        pass

    @property
    @abstractmethod
    def duration(self) -> float:
        pass


class ParselmouthSound(BaseSound):
    """Adapter for parselmouth backend."""

    BACKEND = "parselmouth"

    def __init__(self, inner):
        self._inner = inner

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ParselmouthSound":
        import parselmouth
        return cls(parselmouth.Sound(str(path)))

    @classmethod
    def from_samples(cls, samples: np.ndarray, sampling_frequency: float) -> "ParselmouthSound":
        import parselmouth
        return cls(parselmouth.Sound(samples, sampling_frequency))

    def to_pitch_ac(self, time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0) -> UnifiedPitch:
        # parselmouth uses None for auto time_step, not 0.0
        ts = None if time_step == 0.0 else time_step
        result = self._inner.to_pitch_ac(time_step=ts, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
        return UnifiedPitch(result, self.BACKEND)

    def to_pitch_cc(self, time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0) -> UnifiedPitch:
        ts = None if time_step == 0.0 else time_step
        result = self._inner.to_pitch_cc(time_step=ts, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
        return UnifiedPitch(result, self.BACKEND)

    def to_formant_burg(self, time_step=0.0, max_number_of_formants=5,
                        maximum_formant=5500.0, window_length=0.025,
                        pre_emphasis_from=50.0) -> UnifiedFormant:
        ts = None if time_step == 0.0 else time_step
        result = self._inner.to_formant_burg(time_step=ts, max_number_of_formants=max_number_of_formants,
                                             maximum_formant=maximum_formant, window_length=window_length,
                                             pre_emphasis_from=pre_emphasis_from)
        return UnifiedFormant(result, self.BACKEND)

    def to_intensity(self, minimum_pitch=100.0, time_step=0.0) -> UnifiedIntensity:
        ts = None if time_step == 0.0 else time_step
        result = self._inner.to_intensity(minimum_pitch=minimum_pitch, time_step=ts)
        return UnifiedIntensity(result, self.BACKEND)

    def to_harmonicity_ac(self, time_step=0.01, minimum_pitch=75.0,
                          silence_threshold=0.1, periods_per_window=4.5) -> UnifiedHarmonicity:
        result = self._inner.to_harmonicity(time_step=time_step, minimum_pitch=minimum_pitch,
                                            silence_threshold=silence_threshold, periods_per_window=periods_per_window)
        return UnifiedHarmonicity(result, self.BACKEND)

    def to_harmonicity_cc(self, time_step=0.01, minimum_pitch=75.0,
                          silence_threshold=0.1, periods_per_window=1.0) -> UnifiedHarmonicity:
        result = self._inner.to_harmonicity_cc(time_step=time_step, minimum_pitch=minimum_pitch,
                                               silence_threshold=silence_threshold, periods_per_window=periods_per_window)
        return UnifiedHarmonicity(result, self.BACKEND)

    def to_spectrum(self, fast=True) -> UnifiedSpectrum:
        result = self._inner.to_spectrum(fast=fast)
        return UnifiedSpectrum(result, self.BACKEND)

    def to_spectrogram(self, window_length=0.005, maximum_frequency=5000.0,
                       time_step=0.002, frequency_step=20.0) -> UnifiedSpectrogram:
        result = self._inner.to_spectrogram(window_length=window_length, maximum_frequency=maximum_frequency,
                                            time_step=time_step, frequency_step=frequency_step)
        return UnifiedSpectrogram(result, self.BACKEND)

    @property
    def n_samples(self) -> int:
        return self._inner.n_samples

    @property
    def sampling_frequency(self) -> float:
        return self._inner.sampling_frequency

    @property
    def duration(self) -> float:
        return self._inner.duration

    @property
    def values(self) -> np.ndarray:
        return self._inner.values[0]  # parselmouth returns 2D

    def __repr__(self):
        return f"Sound<parselmouth>({self.n_samples} samples, {self.sampling_frequency} Hz)"


class PraatfanPythonSound(BaseSound):
    """Adapter for praatfan Python backend."""

    BACKEND = "praatfan"

    def __init__(self, inner):
        self._inner = inner

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "PraatfanPythonSound":
        from praatfan.sound import Sound as PfSound
        return cls(PfSound.from_file(path))

    @classmethod
    def from_samples(cls, samples: np.ndarray, sampling_frequency: float) -> "PraatfanPythonSound":
        from praatfan.sound import Sound as PfSound
        return cls(PfSound(samples, sampling_frequency))

    def to_pitch_ac(self, time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0) -> UnifiedPitch:
        result = self._inner.to_pitch(time_step, pitch_floor, pitch_ceiling, method="ac")
        return UnifiedPitch(result, self.BACKEND)

    def to_pitch_cc(self, time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0) -> UnifiedPitch:
        result = self._inner.to_pitch(time_step, pitch_floor, pitch_ceiling, method="cc")
        return UnifiedPitch(result, self.BACKEND)

    def to_formant_burg(self, time_step=0.0, max_number_of_formants=5,
                        maximum_formant=5500.0, window_length=0.025,
                        pre_emphasis_from=50.0) -> UnifiedFormant:
        result = self._inner.to_formant_burg(time_step, max_number_of_formants,
                                             maximum_formant, window_length,
                                             pre_emphasis_from)
        return UnifiedFormant(result, self.BACKEND)

    def to_intensity(self, minimum_pitch=100.0, time_step=0.0) -> UnifiedIntensity:
        result = self._inner.to_intensity(minimum_pitch, time_step)
        return UnifiedIntensity(result, self.BACKEND)

    def to_harmonicity_ac(self, time_step=0.01, minimum_pitch=75.0,
                          silence_threshold=0.1, periods_per_window=4.5) -> UnifiedHarmonicity:
        result = self._inner.to_harmonicity_ac(time_step, minimum_pitch,
                                               silence_threshold, periods_per_window)
        return UnifiedHarmonicity(result, self.BACKEND)

    def to_harmonicity_cc(self, time_step=0.01, minimum_pitch=75.0,
                          silence_threshold=0.1, periods_per_window=1.0) -> UnifiedHarmonicity:
        result = self._inner.to_harmonicity_cc(time_step, minimum_pitch,
                                               silence_threshold, periods_per_window)
        return UnifiedHarmonicity(result, self.BACKEND)

    def to_spectrum(self, fast=True) -> UnifiedSpectrum:
        result = self._inner.to_spectrum(fast)
        return UnifiedSpectrum(result, self.BACKEND)

    def to_spectrogram(self, window_length=0.005, maximum_frequency=5000.0,
                       time_step=0.002, frequency_step=20.0) -> UnifiedSpectrogram:
        result = self._inner.to_spectrogram(window_length, maximum_frequency,
                                            time_step, frequency_step)
        return UnifiedSpectrogram(result, self.BACKEND)

    @property
    def n_samples(self) -> int:
        return self._inner.n_samples

    @property
    def sampling_frequency(self) -> float:
        return self._inner.sample_rate

    @property
    def duration(self) -> float:
        return self._inner.duration

    @property
    def values(self) -> np.ndarray:
        return self._inner.samples

    def extract_part(self, start_time: float, end_time: float) -> "PraatfanPythonSound":
        result = self._inner.extract_part(start_time, end_time)
        return PraatfanPythonSound(result)

    def get_spectrum_at_time(self, time: float, window_length: float = 0.025) -> UnifiedSpectrum:
        result = self._inner.get_spectrum_at_time(time, window_length)
        return UnifiedSpectrum(result, self.BACKEND)

    def get_spectral_moments_at_times(self, times: np.ndarray, window_length: float = 0.025,
                                       power: float = 2.0) -> dict:
        return self._inner.get_spectral_moments_at_times(times, window_length, power)

    def get_band_energy_at_times(self, times: np.ndarray, f_min: float, f_max: float,
                                  window_length: float = 0.025) -> np.ndarray:
        return self._inner.get_band_energy_at_times(times, f_min, f_max, window_length)

    def __repr__(self):
        return f"Sound<praatfan>({self.n_samples} samples, {self.sampling_frequency} Hz)"


class PraatfanRustSound(BaseSound):
    """Adapter for praatfan Rust/PyO3 backend."""

    BACKEND = "praatfan_rust"

    def __init__(self, inner):
        self._inner = inner

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "PraatfanRustSound":
        import praatfan_rust
        return cls(praatfan_rust.Sound(str(path)))

    @classmethod
    def from_samples(cls, samples: np.ndarray, sampling_frequency: float) -> "PraatfanRustSound":
        import praatfan_rust
        return cls(praatfan_rust.Sound(samples, sampling_frequency=sampling_frequency))

    def to_pitch_ac(self, time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0) -> UnifiedPitch:
        result = self._inner.to_pitch_ac(time_step, pitch_floor, pitch_ceiling)
        return UnifiedPitch(result, self.BACKEND)

    def to_pitch_cc(self, time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0) -> UnifiedPitch:
        result = self._inner.to_pitch_cc(time_step, pitch_floor, pitch_ceiling)
        return UnifiedPitch(result, self.BACKEND)

    def to_formant_burg(self, time_step=0.0, max_number_of_formants=5,
                        maximum_formant=5500.0, window_length=0.025,
                        pre_emphasis_from=50.0) -> UnifiedFormant:
        result = self._inner.to_formant_burg(time_step, max_number_of_formants,
                                             maximum_formant, window_length,
                                             pre_emphasis_from)
        return UnifiedFormant(result, self.BACKEND)

    def to_intensity(self, minimum_pitch=100.0, time_step=0.0) -> UnifiedIntensity:
        result = self._inner.to_intensity(minimum_pitch, time_step)
        return UnifiedIntensity(result, self.BACKEND)

    def to_harmonicity_ac(self, time_step=0.01, minimum_pitch=75.0,
                          silence_threshold=0.1, periods_per_window=4.5) -> UnifiedHarmonicity:
        result = self._inner.to_harmonicity_ac(time_step, minimum_pitch,
                                               silence_threshold, periods_per_window)
        return UnifiedHarmonicity(result, self.BACKEND)

    def to_harmonicity_cc(self, time_step=0.01, minimum_pitch=75.0,
                          silence_threshold=0.1, periods_per_window=1.0) -> UnifiedHarmonicity:
        result = self._inner.to_harmonicity_cc(time_step, minimum_pitch,
                                               silence_threshold, periods_per_window)
        return UnifiedHarmonicity(result, self.BACKEND)

    def to_spectrum(self, fast=True) -> UnifiedSpectrum:
        result = self._inner.to_spectrum(fast)
        return UnifiedSpectrum(result, self.BACKEND)

    def to_spectrogram(self, window_length=0.005, maximum_frequency=5000.0,
                       time_step=0.002, frequency_step=20.0) -> UnifiedSpectrogram:
        result = self._inner.to_spectrogram(window_length, maximum_frequency,
                                            time_step, frequency_step)
        return UnifiedSpectrogram(result, self.BACKEND)

    @property
    def n_samples(self) -> int:
        return self._inner.n_samples

    @property
    def sampling_frequency(self) -> float:
        return self._inner.sampling_frequency

    @property
    def duration(self) -> float:
        return self._inner.duration

    @property
    def values(self) -> np.ndarray:
        return self._inner.values()

    def get_band_energy_at_times(self, times: np.ndarray, f_min: float, f_max: float,
                                  window_length: float = 0.025) -> np.ndarray:
        """
        Compute band energy at multiple time points efficiently.

        Uses a single spectrogram computation instead of per-frame spectrum calls,
        avoiding PyO3 boundary crossing overhead for each time point.
        """
        times = np.asarray(times)

        # Compute spectrogram with parameters matching the requested window
        # Use time_step close to minimum time spacing for good resolution
        time_step = min(0.01, np.min(np.diff(times)) if len(times) > 1 else 0.01)
        spec = self._inner.to_spectrogram(
            window_length=window_length,
            maximum_frequency=max(f_max * 1.1, 5000),  # Ensure we cover the band
            time_step=time_step,
            frequency_step=20.0
        )

        # Extract spectrogram data
        spec_times = np.array(spec.xs())
        spec_freqs = np.array(spec.ys())
        n_times = len(spec_times)
        n_freqs = len(spec_freqs)
        # Rust stores values as (n_freqs × n_times) row-major; transpose to (n_times × n_freqs)
        spec_values = np.array(spec.values()).reshape(n_freqs, n_times).T

        # Create frequency mask for the band
        mask = (spec_freqs >= f_min) & (spec_freqs <= f_max)

        # Compute band energy at each spectrogram time
        band_energy_spec = np.sum(spec_values[:, mask], axis=1)

        # Interpolate to requested times
        return np.interp(times, spec_times, band_energy_spec, left=np.nan, right=np.nan)

    def get_spectral_moments_at_times(self, times: np.ndarray, window_length: float = 0.025,
                                       power: float = 2.0) -> dict:
        """
        Compute spectral moments at multiple time points efficiently.

        Uses a single spectrogram computation instead of per-frame spectrum calls,
        avoiding PyO3 boundary crossing overhead for each time point.
        """
        times = np.asarray(times)

        # Compute spectrogram
        time_step = min(0.01, np.min(np.diff(times)) if len(times) > 1 else 0.01)
        spec = self._inner.to_spectrogram(
            window_length=window_length,
            maximum_frequency=8000,  # Cover full speech range for moments
            time_step=time_step,
            frequency_step=20.0
        )

        # Extract spectrogram data
        spec_times = np.array(spec.xs())
        spec_freqs = np.array(spec.ys())
        n_times = len(spec_times)
        n_freqs = len(spec_freqs)
        # Rust stores values as (n_freqs × n_times) row-major; transpose to (n_times × n_freqs)
        spec_values = np.array(spec.values()).reshape(n_freqs, n_times).T

        # Apply power weighting
        weighted = spec_values ** (power / 2)

        # Normalize each frame to get probability distribution
        total = np.sum(weighted, axis=1, keepdims=True)
        p = np.where(total > 0, weighted / total, 0)

        # Center of gravity (first moment)
        cog_spec = np.sum(spec_freqs * p, axis=1)

        # Standard deviation (second central moment)
        variance = np.sum(((spec_freqs - cog_spec[:, np.newaxis]) ** 2) * p, axis=1)
        std_spec = np.sqrt(np.maximum(variance, 0))

        # Skewness (third standardized moment)
        with np.errstate(divide='ignore', invalid='ignore'):
            skew_spec = np.where(
                std_spec > 0,
                np.sum(((spec_freqs - cog_spec[:, np.newaxis]) ** 3) * p, axis=1) / (std_spec ** 3),
                0
            )

        # Kurtosis (fourth standardized moment, excess kurtosis: subtract 3)
        with np.errstate(divide='ignore', invalid='ignore'):
            kurt_spec = np.where(
                std_spec > 0,
                np.sum(((spec_freqs - cog_spec[:, np.newaxis]) ** 4) * p, axis=1) / (std_spec ** 4) - 3.0,
                0
            )

        # Interpolate to requested times
        cog = np.interp(times, spec_times, cog_spec, left=np.nan, right=np.nan)
        std = np.interp(times, spec_times, std_spec, left=np.nan, right=np.nan)
        skew = np.interp(times, spec_times, skew_spec, left=np.nan, right=np.nan)
        kurt = np.interp(times, spec_times, kurt_spec, left=np.nan, right=np.nan)

        return {
            'times': times,
            'center_of_gravity': cog,
            'standard_deviation': std,
            'skewness': skew,
            'kurtosis': kurt
        }

    def get_variable_band_energy_at_times(self, times: np.ndarray, f_mins: np.ndarray,
                                           f_maxs: np.ndarray, window_length: float = 0.025) -> np.ndarray:
        """
        Compute band energy at multiple time points with variable frequency bands.

        This is useful for A1-P0 nasal ratio where the band depends on F0 at each time.
        Uses a single spectrogram computation for efficiency.

        Args:
            times: Array of time points in seconds.
            f_mins: Array of minimum frequencies (one per time point).
            f_maxs: Array of maximum frequencies (one per time point).
            window_length: Window length in seconds (default 25ms).

        Returns:
            Numpy array of band energy values (one per time point).
            NaN for time points where f_min or f_max is NaN.
        """
        times = np.asarray(times)
        f_mins = np.asarray(f_mins)
        f_maxs = np.asarray(f_maxs)

        # Compute spectrogram once
        max_freq = np.nanmax(f_maxs) * 1.1 if not np.all(np.isnan(f_maxs)) else 5000
        time_step = min(0.01, np.min(np.diff(times)) if len(times) > 1 else 0.01)
        spec = self._inner.to_spectrogram(
            window_length=window_length,
            maximum_frequency=max(max_freq, 1000),
            time_step=time_step,
            frequency_step=20.0
        )

        # Extract spectrogram data
        spec_times = np.array(spec.xs())
        spec_freqs = np.array(spec.ys())
        n_times_spec = len(spec_times)
        n_freqs = len(spec_freqs)
        # Rust stores values as (n_freqs × n_times) row-major; transpose to (n_times × n_freqs)
        spec_values = np.array(spec.values()).reshape(n_freqs, n_times_spec).T

        # Initialize result
        result = np.full(len(times), np.nan)

        # For each requested time, find nearest spectrogram frame and sum appropriate band
        for i, (t, f_min, f_max) in enumerate(zip(times, f_mins, f_maxs)):
            if np.isnan(f_min) or np.isnan(f_max):
                continue

            # Find nearest spectrogram time index
            idx = np.searchsorted(spec_times, t)
            if idx >= n_times_spec:
                idx = n_times_spec - 1
            elif idx > 0 and abs(spec_times[idx-1] - t) < abs(spec_times[idx] - t):
                idx = idx - 1

            # Sum energy in the frequency band
            mask = (spec_freqs >= f_min) & (spec_freqs <= f_max)
            if np.any(mask):
                result[i] = np.sum(spec_values[idx, mask])

        return result

    def __repr__(self):
        return f"Sound<praatfan_rust>({self.n_samples} samples, {self.sampling_frequency} Hz)"


class PraatfanCoreSound(BaseSound):
    """Adapter for praatfan_gpl backend."""

    BACKEND = "praatfan_gpl"

    def __init__(self, inner):
        self._inner = inner

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "PraatfanCoreSound":
        import praatfan_gpl
        return cls(praatfan_gpl.Sound.from_file(str(path)))

    @classmethod
    def from_samples(cls, samples: np.ndarray, sampling_frequency: float) -> "PraatfanCoreSound":
        import praatfan_gpl
        return cls(praatfan_gpl.Sound(samples, sampling_frequency))

    def to_pitch_ac(self, time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0) -> UnifiedPitch:
        # praatfan_gpl only has to_pitch (AC method)
        result = self._inner.to_pitch(time_step, pitch_floor, pitch_ceiling)
        return UnifiedPitch(result, self.BACKEND)

    def to_pitch_cc(self, time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0) -> UnifiedPitch:
        # praatfan_gpl only has to_pitch (AC method)
        import warnings
        warnings.warn(
            "praatfan_gpl backend does not support cross-correlation pitch. "
            "Falling back to autocorrelation method.",
            UserWarning,
            stacklevel=2
        )
        result = self._inner.to_pitch(time_step, pitch_floor, pitch_ceiling)
        return UnifiedPitch(result, self.BACKEND)

    def to_formant_burg(self, time_step=0.0, max_number_of_formants=5,
                        maximum_formant=5500.0, window_length=0.025,
                        pre_emphasis_from=50.0) -> UnifiedFormant:
        result = self._inner.to_formant_burg(time_step, max_number_of_formants,
                                             maximum_formant, window_length,
                                             pre_emphasis_from)
        return UnifiedFormant(result, self.BACKEND)

    def to_intensity(self, minimum_pitch=100.0, time_step=0.0) -> UnifiedIntensity:
        result = self._inner.to_intensity(minimum_pitch, time_step)
        return UnifiedIntensity(result, self.BACKEND)

    def to_harmonicity_ac(self, time_step=0.01, minimum_pitch=75.0,
                          silence_threshold=0.1, periods_per_window=4.5) -> UnifiedHarmonicity:
        result = self._inner.to_harmonicity_ac(time_step, minimum_pitch,
                                               silence_threshold, periods_per_window)
        return UnifiedHarmonicity(result, self.BACKEND)

    def to_harmonicity_cc(self, time_step=0.01, minimum_pitch=75.0,
                          silence_threshold=0.1, periods_per_window=1.0) -> UnifiedHarmonicity:
        result = self._inner.to_harmonicity_cc(time_step, minimum_pitch,
                                               silence_threshold, periods_per_window)
        return UnifiedHarmonicity(result, self.BACKEND)

    def to_spectrum(self, fast=True) -> UnifiedSpectrum:
        result = self._inner.to_spectrum(fast)
        return UnifiedSpectrum(result, self.BACKEND)

    def to_spectrogram(self, window_length=0.005, maximum_frequency=5000.0,
                       time_step=0.002, frequency_step=20.0) -> UnifiedSpectrogram:
        # praatfan_gpl requires window_shape argument
        result = self._inner.to_spectrogram(window_length, maximum_frequency,
                                            time_step, frequency_step, "Gaussian")
        return UnifiedSpectrogram(result, self.BACKEND)

    @property
    def n_samples(self) -> int:
        return self._inner.num_samples

    @property
    def sampling_frequency(self) -> float:
        return self._inner.sample_rate

    @property
    def duration(self) -> float:
        return self._inner.duration

    @property
    def values(self) -> np.ndarray:
        # praatfan_gpl samples is a method, not a property
        return np.array(self._inner.samples())

    def __repr__(self):
        return f"Sound<praatfan_gpl>({self.n_samples} samples, {self.sampling_frequency} Hz)"


# =============================================================================
# Unified Sound Class
# =============================================================================
#
# This is the main public API. Users create Sound objects and call analysis
# methods on them. The Sound class:
#   1. Selects the appropriate backend (or uses the user-specified one)
#   2. Creates a backend-specific adapter
#   3. Delegates method calls to the adapter
#   4. Returns Unified* result objects for consistent access
#
# The Sound class also provides fallback implementations for methods that
# aren't available in all backends (e.g., per-window spectral extraction).
# =============================================================================

class Sound:
    """
    Unified Sound class that delegates to the selected backend.

    This class provides a consistent API regardless of which backend is used.
    Users should interact with this class rather than backend-specific classes.

    The backend is selected based on (in order of priority):
        1. set_backend() if called programmatically
        2. PRAATFAN_BACKEND environment variable
        3. Config file (~/.praatfan/config.toml or ./praatfan.toml)
        4. Auto-detect: praatfan_gpl > praatfan_rust > praatfan > parselmouth

    Example:
        # Load from file
        sound = Sound("audio.wav")

        # Or from numpy array
        sound = Sound(samples, sampling_frequency=16000)

        # Analyze
        pitch = sound.to_pitch_ac()
        formant = sound.to_formant_burg()

        # Results have consistent API
        f0 = pitch.values()  # numpy array of F0 values
    """

    def __init__(self, samples_or_path: Union[str, Path, np.ndarray],
                 sampling_frequency: Optional[float] = None):
        """
        Create a Sound from a file path or numpy array.

        Args:
            samples_or_path: Either a path to an audio file, or a numpy array of samples
            sampling_frequency: Sample rate in Hz (required if providing samples)
        """
        backend = _select_backend()

        if isinstance(samples_or_path, (str, Path)):
            self._inner = self._load_from_file(backend, samples_or_path)
        else:
            if sampling_frequency is None:
                raise ValueError("sampling_frequency is required when providing samples")
            self._inner = self._load_from_samples(backend, samples_or_path, sampling_frequency)

        self._backend = backend

    @staticmethod
    def _load_from_file(backend: str, path: Union[str, Path]) -> BaseSound:
        if backend == "parselmouth":
            return ParselmouthSound.from_file(path)
        elif backend == "praatfan":
            return PraatfanPythonSound.from_file(path)
        elif backend == "praatfan_rust":
            return PraatfanRustSound.from_file(path)
        elif backend == "praatfan_gpl":
            return PraatfanCoreSound.from_file(path)
        else:
            raise BackendNotAvailableError(f"Unknown backend: {backend}")

    @staticmethod
    def _load_from_samples(backend: str, samples: np.ndarray, sampling_frequency: float) -> BaseSound:
        if backend == "parselmouth":
            return ParselmouthSound.from_samples(samples, sampling_frequency)
        elif backend == "praatfan":
            return PraatfanPythonSound.from_samples(samples, sampling_frequency)
        elif backend == "praatfan_rust":
            return PraatfanRustSound.from_samples(samples, sampling_frequency)
        elif backend == "praatfan_gpl":
            return PraatfanCoreSound.from_samples(samples, sampling_frequency)
        else:
            raise BackendNotAvailableError(f"Unknown backend: {backend}")

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Sound":
        """Load audio from a file."""
        return cls(path)

    # Delegate all methods to the inner backend

    def to_pitch_ac(self, time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0):
        """Compute pitch using autocorrelation method."""
        return self._inner.to_pitch_ac(time_step, pitch_floor, pitch_ceiling)

    def to_pitch_cc(self, time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0):
        """Compute pitch using cross-correlation method."""
        return self._inner.to_pitch_cc(time_step, pitch_floor, pitch_ceiling)

    def to_pitch(self, time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0):
        """Compute pitch (alias for to_pitch_ac)."""
        return self.to_pitch_ac(time_step, pitch_floor, pitch_ceiling)

    def to_formant_burg(self, time_step=0.0, max_number_of_formants=5,
                        maximum_formant=5500.0, window_length=0.025,
                        pre_emphasis_from=50.0):
        """Compute formants using Burg's LPC method."""
        return self._inner.to_formant_burg(time_step, max_number_of_formants,
                                           maximum_formant, window_length,
                                           pre_emphasis_from)

    def to_intensity(self, minimum_pitch=100.0, time_step=0.0):
        """Compute intensity contour."""
        return self._inner.to_intensity(minimum_pitch, time_step)

    def to_harmonicity_ac(self, time_step=0.01, minimum_pitch=75.0,
                          silence_threshold=0.1, periods_per_window=4.5):
        """Compute harmonicity using autocorrelation method."""
        return self._inner.to_harmonicity_ac(time_step, minimum_pitch,
                                             silence_threshold, periods_per_window)

    def to_harmonicity_cc(self, time_step=0.01, minimum_pitch=75.0,
                          silence_threshold=0.1, periods_per_window=1.0):
        """Compute harmonicity using cross-correlation method."""
        return self._inner.to_harmonicity_cc(time_step, minimum_pitch,
                                             silence_threshold, periods_per_window)

    def to_harmonicity(self, time_step=0.01, minimum_pitch=75.0,
                       silence_threshold=0.1, periods_per_window=4.5):
        """Compute harmonicity (alias for to_harmonicity_ac)."""
        return self.to_harmonicity_ac(time_step, minimum_pitch,
                                      silence_threshold, periods_per_window)

    def to_spectrum(self, fast=True):
        """Compute spectrum (single-frame FFT)."""
        return self._inner.to_spectrum(fast)

    def to_spectrogram(self, window_length=0.005, maximum_frequency=5000.0,
                       time_step=0.002, frequency_step=20.0):
        """Compute spectrogram."""
        return self._inner.to_spectrogram(window_length, maximum_frequency,
                                          time_step, frequency_step)

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self._inner.n_samples

    @property
    def sampling_frequency(self) -> float:
        """Sample rate in Hz."""
        return self._inner.sampling_frequency

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self._inner.duration

    def get_total_duration(self) -> float:
        """Get total duration in seconds (parselmouth compatibility)."""
        return self.duration

    @property
    def values(self) -> np.ndarray:
        """Audio samples as numpy array."""
        return self._inner.values

    @property
    def backend(self) -> str:
        """The backend being used."""
        return self._backend

    # =========================================================================
    # Per-Window Spectral Feature Extraction
    # =========================================================================
    #
    # These methods extract spectral features at specific time points. They are
    # useful for analyzing spectral characteristics aligned with other measurements
    # like formants or pitch.
    #
    # Design principle: If the backend has a native implementation, use it.
    # Otherwise, compose from simpler methods that all backends support:
    #   extract_part() -> to_spectrum() -> spectral moment methods
    #
    # This ensures all backends work, even if some are more efficient than others.
    # =========================================================================

    def extract_part(self, start_time: float, end_time: float) -> "Sound":
        """
        Extract a portion of the sound between two time points.

        Args:
            start_time: Start time in seconds.
            end_time: End time in seconds.

        Returns:
            New Sound object containing the extracted portion.

        Note:
            If the backend has a native extract_part method, it is used.
            Otherwise, falls back to slicing the samples array directly.
        """
        if hasattr(self._inner, 'extract_part'):
            # Backend has native implementation - use it
            result = self._inner.extract_part(start_time, end_time)
            # Wrap result in a new Sound with the same backend
            new_sound = Sound.__new__(Sound)
            # Handle different property names across backends
            new_sound._inner = self._load_from_samples(
                self._backend,
                np.asarray(result.values if hasattr(result, 'values') else result.samples),
                result.sampling_frequency if hasattr(result, 'sampling_frequency') else result.sample_rate
            )
            new_sound._backend = self._backend
            return new_sound

        # Fallback: slice samples array directly
        samples = np.asarray(self.values)
        sr = self.sampling_frequency
        start_sample = max(0, round(start_time * sr))
        end_sample = min(len(samples), round(end_time * sr))
        return Sound(samples[start_sample:end_sample].copy(), sr)

    def get_spectrum_at_time(self, time: float, window_length: float = 0.025) -> UnifiedSpectrum:
        """
        Extract spectrum for a window centered at a specific time.

        This extracts a short segment of audio around the specified time,
        applies a window, and computes the FFT.

        Args:
            time: Center time in seconds.
            window_length: Window length in seconds (default 25ms).

        Returns:
            UnifiedSpectrum object for the extracted window.

        Note:
            If time is near the start or end of the sound, the window
            will be truncated to fit within the sound duration.
        """
        if hasattr(self._inner, 'get_spectrum_at_time'):
            # Backend has native implementation
            return self._inner.get_spectrum_at_time(time, window_length)

        # Fallback: extract_part + to_spectrum
        half_window = window_length / 2.0
        start = max(0.0, time - half_window)
        end = min(self.duration, time + half_window)
        return self.extract_part(start, end).to_spectrum()

    def get_spectral_moments_at_times(self, times: np.ndarray, window_length: float = 0.025,
                                       power: float = 2.0) -> dict:
        """
        Compute spectral moments at multiple time points.

        This is useful for extracting spectral features (center of gravity,
        spread, skewness, kurtosis) at time points aligned with other
        measurements like formant tracks.

        Args:
            times: Array of time points in seconds.
            window_length: Window length in seconds (default 25ms).
            power: Power parameter for moment calculation (default 2.0).

        Returns:
            Dictionary with keys:
                - 'times': Input time array
                - 'center_of_gravity': Spectral centroid in Hz
                - 'standard_deviation': Spectral spread in Hz
                - 'skewness': Spectral asymmetry (dimensionless)
                - 'kurtosis': Spectral peakedness (dimensionless)

        Note:
            If the backend has a native batch implementation, it is used.
            Otherwise, falls back to calling get_spectrum_at_time() in a loop.
        """
        if hasattr(self._inner, 'get_spectral_moments_at_times'):
            # Backend has native batch implementation
            return self._inner.get_spectral_moments_at_times(times, window_length, power)

        # Fallback: loop and compute moments for each time point
        times = np.asarray(times)
        n = len(times)
        cog = np.zeros(n)
        std = np.zeros(n)
        skew = np.zeros(n)
        kurt = np.zeros(n)

        for i, t in enumerate(times):
            spectrum = self.get_spectrum_at_time(t, window_length)
            cog[i] = spectrum.get_center_of_gravity(power)
            std[i] = spectrum.get_standard_deviation(power)
            skew[i] = spectrum.get_skewness(power)
            kurt[i] = spectrum.get_kurtosis(power)

        return {
            'times': times,
            'center_of_gravity': cog,
            'standard_deviation': std,
            'skewness': skew,
            'kurtosis': kurt
        }

    def get_band_energy_at_times(self, times: np.ndarray, f_min: float, f_max: float,
                                  window_length: float = 0.025) -> np.ndarray:
        """
        Compute band energy at multiple time points.

        This computes the energy in a frequency band at each time point,
        useful for features like spectral tilt or band-limited energy ratios.

        Args:
            times: Array of time points in seconds.
            f_min: Minimum frequency in Hz.
            f_max: Maximum frequency in Hz.
            window_length: Window length in seconds (default 25ms).

        Returns:
            Numpy array of band energy values (one per time point).

        Note:
            If the backend has a native batch implementation, it is used.
            Otherwise, falls back to calling get_spectrum_at_time() in a loop.
        """
        if hasattr(self._inner, 'get_band_energy_at_times'):
            # Backend has native batch implementation
            return self._inner.get_band_energy_at_times(times, f_min, f_max, window_length)

        # Fallback: loop and compute band energy for each time point
        times = np.asarray(times)
        n = len(times)
        energy = np.zeros(n)

        for i, t in enumerate(times):
            spectrum = self.get_spectrum_at_time(t, window_length)
            energy[i] = spectrum.get_band_energy(f_min, f_max)

        return energy

    def get_variable_band_energy_at_times(self, times: np.ndarray, f_mins: np.ndarray,
                                           f_maxs: np.ndarray, window_length: float = 0.025) -> np.ndarray:
        """
        Compute band energy at multiple time points with variable frequency bands.

        This is useful for A1-P0 nasal ratio where the band depends on F0 at each time.

        Args:
            times: Array of time points in seconds.
            f_mins: Array of minimum frequencies (one per time point).
            f_maxs: Array of maximum frequencies (one per time point).
            window_length: Window length in seconds (default 25ms).

        Returns:
            Numpy array of band energy values (one per time point).
            NaN for time points where f_min or f_max is NaN.

        Note:
            If the backend has a native batch implementation, it is used.
            Otherwise, falls back to calling get_spectrum_at_time() in a loop.
        """
        if hasattr(self._inner, 'get_variable_band_energy_at_times'):
            return self._inner.get_variable_band_energy_at_times(times, f_mins, f_maxs, window_length)

        # Fallback: loop (slow but correct)
        times = np.asarray(times)
        f_mins = np.asarray(f_mins)
        f_maxs = np.asarray(f_maxs)
        n = len(times)
        energy = np.full(n, np.nan)

        for i, (t, f_min, f_max) in enumerate(zip(times, f_mins, f_maxs)):
            if np.isnan(f_min) or np.isnan(f_max):
                continue
            spectrum = self.get_spectrum_at_time(t, window_length)
            energy[i] = spectrum.get_band_energy(f_min, f_max)

        return energy

    def __repr__(self):
        return f"Sound<{self._backend}>({self.n_samples} samples, {self.sampling_frequency} Hz, {self.duration:.3f}s)"
