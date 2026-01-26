"""
Backend selector and unified API for acoustic analysis.
"""

import os
from pathlib import Path
from typing import Optional, List, Union
from abc import ABC, abstractmethod

import numpy as np


class BackendNotAvailableError(Exception):
    """Raised when no suitable backend is available."""
    pass


# =============================================================================
# Unified result types - same interface regardless of backend
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
        elif self._backend == "praatfan-rust":
            return np.array(self._inner.xs())
        elif self._backend == "praatfan-core":
            return np.array(self._inner.times())
        raise ValueError(f"Unknown backend: {self._backend}")

    def values(self) -> np.ndarray:
        """F0 values (Hz) for each frame. NaN for unvoiced."""
        if self._backend == "parselmouth":
            return self._inner.selected_array['frequency']
        elif self._backend == "praatfan":
            return self._inner.values()
        elif self._backend == "praatfan-rust":
            return np.array(self._inner.values())
        elif self._backend == "praatfan-core":
            return np.array(self._inner.values())
        raise ValueError(f"Unknown backend: {self._backend}")

    def strengths(self) -> np.ndarray:
        """Voicing strength for each frame."""
        if self._backend == "parselmouth":
            return self._inner.selected_array['strength']
        elif self._backend == "praatfan":
            return self._inner.strengths()
        elif self._backend == "praatfan-rust":
            return np.array(self._inner.selected_array['strength'])
        elif self._backend == "praatfan-core":
            # praatfan-core doesn't have strengths, return ones for voiced frames
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
        elif self._backend == "praatfan-rust":
            return self._inner.n_frames
        elif self._backend == "praatfan-core":
            return self._inner.num_frames
        raise ValueError(f"Unknown backend: {self._backend}")

    @property
    def time_step(self) -> float:
        """Time step between frames."""
        if self._backend == "parselmouth":
            return self._inner.time_step
        elif self._backend == "praatfan":
            return self._inner.time_step
        elif self._backend == "praatfan-rust":
            return self._inner.time_step
        elif self._backend == "praatfan-core":
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
        elif self._backend == "praatfan-rust":
            return np.array(self._inner.xs())
        elif self._backend == "praatfan-core":
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
        elif self._backend == "praatfan-rust":
            return np.array(self._inner.to_array(formant_number))
        elif self._backend == "praatfan-core":
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
        elif self._backend == "praatfan-rust":
            return np.array(self._inner.to_bandwidth_array(formant_number))
        elif self._backend == "praatfan-core":
            return np.array(self._inner.bandwidth_values(formant_number))
        raise ValueError(f"Unknown backend: {self._backend}")

    @property
    def n_frames(self) -> int:
        if self._backend == "praatfan-core":
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
        elif self._backend == "praatfan-rust":
            return np.array(self._inner.xs())
        elif self._backend == "praatfan-core":
            return np.array(self._inner.xs())
        raise ValueError(f"Unknown backend: {self._backend}")

    def values(self) -> np.ndarray:
        """Intensity values in dB."""
        if self._backend == "parselmouth":
            return self._inner.values[0]  # parselmouth returns 2D
        elif self._backend == "praatfan":
            return self._inner.values
        elif self._backend == "praatfan-rust":
            return np.array(self._inner.values())
        elif self._backend == "praatfan-core":
            return np.array(self._inner.values())
        raise ValueError(f"Unknown backend: {self._backend}")

    @property
    def n_frames(self) -> int:
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
        elif self._backend == "praatfan-rust":
            return np.array(self._inner.xs())
        elif self._backend == "praatfan-core":
            return np.array(self._inner.xs())
        raise ValueError(f"Unknown backend: {self._backend}")

    def values(self) -> np.ndarray:
        """HNR values in dB."""
        if self._backend == "parselmouth":
            return self._inner.values[0]  # parselmouth returns 2D
        elif self._backend == "praatfan":
            return self._inner.values
        elif self._backend == "praatfan-rust":
            return np.array(self._inner.values())
        elif self._backend == "praatfan-core":
            return np.array(self._inner.values())
        raise ValueError(f"Unknown backend: {self._backend}")

    @property
    def n_frames(self) -> int:
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
        elif self._backend == "praatfan-rust":
            return np.array(self._inner.xs())
        elif self._backend == "praatfan-core":
            # praatfan-core has num_bins not n_bins, and no xs() method
            return np.arange(self._inner.num_bins) * self._inner.df
        raise ValueError(f"Unknown backend: {self._backend}")

    def values(self) -> np.ndarray:
        """Complex spectrum values."""
        if self._backend == "parselmouth":
            return self._inner.values[0]  # parselmouth returns 2D
        elif self._backend == "praatfan":
            return self._inner.real + 1j * self._inner.imag
        elif self._backend == "praatfan-rust":
            return np.array(self._inner.real()) + 1j * np.array(self._inner.imag())
        elif self._backend == "praatfan-core":
            return np.array(self._inner.real()) + 1j * np.array(self._inner.imag())
        raise ValueError(f"Unknown backend: {self._backend}")

    def get_center_of_gravity(self, power: float = 2.0) -> float:
        """Spectral centroid."""
        if self._backend == "parselmouth":
            return self._inner.get_centre_of_gravity(power)
        elif self._backend == "praatfan":
            return self._inner.get_center_of_gravity(power)
        elif self._backend == "praatfan-rust":
            return self._inner.get_center_of_gravity(power)
        elif self._backend == "praatfan-core":
            return self._inner.get_center_of_gravity(power)
        raise ValueError(f"Unknown backend: {self._backend}")

    def get_standard_deviation(self, power: float = 2.0) -> float:
        """Spectral standard deviation."""
        if self._backend == "parselmouth":
            return self._inner.get_standard_deviation(power)
        elif self._backend == "praatfan":
            return self._inner.get_standard_deviation(power)
        elif self._backend == "praatfan-rust":
            return self._inner.get_standard_deviation(power)
        elif self._backend == "praatfan-core":
            return self._inner.get_standard_deviation(power)
        raise ValueError(f"Unknown backend: {self._backend}")

    def get_skewness(self, power: float = 2.0) -> float:
        """Spectral skewness."""
        if self._backend == "parselmouth":
            return self._inner.get_skewness(power)
        elif self._backend == "praatfan":
            return self._inner.get_skewness(power)
        elif self._backend == "praatfan-rust":
            return self._inner.get_skewness(power)
        elif self._backend == "praatfan-core":
            return self._inner.get_skewness(power)
        raise ValueError(f"Unknown backend: {self._backend}")

    def get_kurtosis(self, power: float = 2.0) -> float:
        """Spectral kurtosis."""
        if self._backend == "parselmouth":
            return self._inner.get_kurtosis(power)
        elif self._backend == "praatfan":
            return self._inner.get_kurtosis(power)
        elif self._backend == "praatfan-rust":
            return self._inner.get_kurtosis(power)
        elif self._backend == "praatfan-core":
            return self._inner.get_kurtosis(power)
        raise ValueError(f"Unknown backend: {self._backend}")

    def get_band_energy(self, f_min: float = 0.0, f_max: float = 0.0) -> float:
        """Band energy between frequencies."""
        if self._backend == "parselmouth":
            return self._inner.get_band_energy(f_min, f_max)
        elif self._backend == "praatfan":
            return self._inner.get_band_energy(f_min, f_max)
        elif self._backend == "praatfan-rust":
            return self._inner.get_band_energy(f_min, f_max)
        elif self._backend == "praatfan-core":
            return self._inner.get_band_energy(f_min, f_max)
        raise ValueError(f"Unknown backend: {self._backend}")

    @property
    def n_bins(self) -> int:
        if self._backend == "praatfan-core":
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
        elif self._backend == "praatfan-rust":
            return np.array(self._inner.xs())
        elif self._backend == "praatfan-core":
            # praatfan-core uses get_time_from_frame (0-indexed)
            n = self._inner.num_frames
            return np.array([self._inner.get_time_from_frame(i) for i in range(n)])
        raise ValueError(f"Unknown backend: {self._backend}")

    def ys(self) -> np.ndarray:
        """Frequency values."""
        if self._backend == "parselmouth":
            return np.array(self._inner.ys())
        elif self._backend == "praatfan":
            return self._inner.frequencies()
        elif self._backend == "praatfan-rust":
            return np.array(self._inner.ys())
        elif self._backend == "praatfan-core":
            # praatfan-core uses get_frequency_from_bin (0-indexed)
            n = self._inner.num_freq_bins
            return np.array([self._inner.get_frequency_from_bin(i) for i in range(n)])
        raise ValueError(f"Unknown backend: {self._backend}")

    def values(self) -> np.ndarray:
        """Power values as 2D array (frequency × time)."""
        if self._backend == "parselmouth":
            return np.array(self._inner.values)
        elif self._backend == "praatfan":
            return self._inner.values
        elif self._backend == "praatfan-rust":
            # Reshape from 1D to 2D
            vals = np.array(self._inner.values())
            return vals.reshape(self._inner.n_freqs, self._inner.n_times)
        elif self._backend == "praatfan-core":
            vals = np.array(self._inner.values())
            return vals.reshape(self._inner.num_freq_bins, self._inner.num_frames)
        raise ValueError(f"Unknown backend: {self._backend}")

    @property
    def n_times(self) -> int:
        if self._backend == "praatfan-core":
            return self._inner.num_frames
        return self._inner.n_times

    @property
    def n_freqs(self) -> int:
        if self._backend == "parselmouth":
            return self._inner.n_frequencies
        elif self._backend == "praatfan-core":
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
# Backend detection
# =============================================================================

def _try_import_parselmouth():
    """Try to import parselmouth."""
    try:
        import parselmouth
        return True
    except ImportError:
        return False


def _try_import_praatfan_core():
    """Try to import praatfan-core (original Rust implementation)."""
    try:
        import praatfan_core
        return True
    except ImportError:
        return False


def _try_import_praatfan():
    """Try to import praatfan (clean-room Python implementation)."""
    try:
        import praatfan
        return True
    except ImportError:
        return False


def _try_import_praatfan_rust():
    """Try to import praatfan Rust bindings (PyO3).

    Note: The Rust bindings are also named 'praatfan' when built with maturin.
    We distinguish by checking for Rust-specific attributes.
    """
    try:
        import praatfan
        # Check if this is the Rust version by looking for Rust-specific behavior
        # The Rust version has __file__ pointing to a .so/.pyd file
        if hasattr(praatfan, '__file__') and praatfan.__file__:
            if praatfan.__file__.endswith(('.so', '.pyd', '.dylib')):
                return True
        return False
    except ImportError:
        return False


def get_available_backends() -> List[str]:
    """Return list of available backend names."""
    available = []

    # Check each backend
    if _try_import_praatfan_rust():
        available.append("praatfan-rust")
    elif _try_import_praatfan():
        # Only add Python version if Rust version isn't masking it
        available.append("praatfan")

    if _try_import_parselmouth():
        available.append("parselmouth")

    if _try_import_praatfan_core():
        available.append("praatfan-core")

    return available


# =============================================================================
# Configuration
# =============================================================================

_current_backend: Optional[str] = None


def _read_config_file() -> Optional[str]:
    """Read backend preference from config file."""
    # Try local config first
    local_config = Path("praatfan.toml")
    if local_config.exists():
        return _parse_toml_backend(local_config)

    # Try user config
    user_config = Path.home() / ".praatfan" / "config.toml"
    if user_config.exists():
        return _parse_toml_backend(user_config)

    return None


def _parse_toml_backend(path: Path) -> Optional[str]:
    """Parse backend from TOML config file."""
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
            "praatfan, praat-parselmouth, praatfan-core"
        )

    # Preference order
    preference = ["praatfan-core", "praatfan-rust", "praatfan", "parselmouth"]
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
        name: Backend name ("parselmouth", "praatfan", "praatfan-rust", "praatfan-core")

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
# Backend adapters
# =============================================================================

class BaseSound(ABC):
    """Abstract base class for Sound across backends."""

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

    def __repr__(self):
        return f"Sound<praatfan>({self.n_samples} samples, {self.sampling_frequency} Hz)"


class PraatfanRustSound(BaseSound):
    """Adapter for praatfan Rust/PyO3 backend."""

    BACKEND = "praatfan-rust"

    def __init__(self, inner):
        self._inner = inner

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "PraatfanRustSound":
        import praatfan
        return cls(praatfan.Sound(str(path)))

    @classmethod
    def from_samples(cls, samples: np.ndarray, sampling_frequency: float) -> "PraatfanRustSound":
        import praatfan
        return cls(praatfan.Sound(samples, sampling_frequency=sampling_frequency))

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

    def __repr__(self):
        return f"Sound<praatfan-rust>({self.n_samples} samples, {self.sampling_frequency} Hz)"


class PraatfanCoreSound(BaseSound):
    """Adapter for praatfan-core backend."""

    BACKEND = "praatfan-core"

    def __init__(self, inner):
        self._inner = inner

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "PraatfanCoreSound":
        import praatfan_core
        return cls(praatfan_core.Sound.from_file(str(path)))

    @classmethod
    def from_samples(cls, samples: np.ndarray, sampling_frequency: float) -> "PraatfanCoreSound":
        import praatfan_core
        return cls(praatfan_core.Sound(samples, sampling_frequency))

    def to_pitch_ac(self, time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0) -> UnifiedPitch:
        # praatfan-core only has to_pitch (AC method)
        result = self._inner.to_pitch(time_step, pitch_floor, pitch_ceiling)
        return UnifiedPitch(result, self.BACKEND)

    def to_pitch_cc(self, time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0) -> UnifiedPitch:
        # praatfan-core only has to_pitch (AC method), use it for CC as well
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
        # praatfan-core requires window_shape argument
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
        return self._inner.samples

    def __repr__(self):
        return f"Sound<praatfan-core>({self.n_samples} samples, {self.sampling_frequency} Hz)"


# =============================================================================
# Unified Sound class
# =============================================================================

class Sound:
    """
    Unified Sound class that delegates to the selected backend.

    This class provides a consistent API regardless of which backend is used.
    The backend is selected based on:
    1. PRAATFAN_BACKEND environment variable
    2. Config file (~/.praatfan/config.toml or ./praatfan.toml)
    3. First available in order: praatfan-rust, praatfan, parselmouth, praatfan-core
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
        elif backend == "praatfan-rust":
            return PraatfanRustSound.from_file(path)
        elif backend == "praatfan-core":
            return PraatfanCoreSound.from_file(path)
        else:
            raise BackendNotAvailableError(f"Unknown backend: {backend}")

    @staticmethod
    def _load_from_samples(backend: str, samples: np.ndarray, sampling_frequency: float) -> BaseSound:
        if backend == "parselmouth":
            return ParselmouthSound.from_samples(samples, sampling_frequency)
        elif backend == "praatfan":
            return PraatfanPythonSound.from_samples(samples, sampling_frequency)
        elif backend == "praatfan-rust":
            return PraatfanRustSound.from_samples(samples, sampling_frequency)
        elif backend == "praatfan-core":
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

    def __repr__(self):
        return f"Sound<{self._backend}>({self.n_samples} samples, {self.sampling_frequency} Hz, {self.duration:.3f}s)"
