"""
Spectrogram - Time-frequency representation.

Implementation order: Phase 7 (after Spectrum)
Dependencies: Spectrum (windowed FFT per frame)

Documentation sources:
- Praat manual: Sound: To Spectrogram...
- Standard STFT definition

Key documented facts:
- Gaussian window: "analyzes a factor of 2 slower... twice as many samples"
- Gaussian -3dB bandwidth: 1.2982804 / window_length
- Time step minimum: never less than 1/(8√π) × window_length
- Frequency step minimum: never less than (√π)/8 / window_length
"""

import numpy as np
from typing import Optional


class Spectrogram:
    """
    Time-frequency representation (power spectral density over time).

    Attributes:
        values: 2D array of power values (n_freqs × n_times)
        times: Time points in seconds
        frequencies: Frequency points in Hz
        time_step: Time step between frames
        freq_step: Frequency step between bins
    """

    def __init__(
        self,
        values: np.ndarray,
        time_min: float,
        time_max: float,
        freq_min: float,
        freq_max: float,
        time_step: float,
        freq_step: float
    ):
        """
        Create a Spectrogram object.

        Args:
            values: 2D array of power values (n_freqs × n_times)
            time_min: Start time in seconds
            time_max: End time in seconds
            freq_min: Minimum frequency in Hz
            freq_max: Maximum frequency in Hz
            time_step: Time step between frames
            freq_step: Frequency step between bins
        """
        self._values = np.asarray(values, dtype=np.float64)
        self._time_min = time_min
        self._time_max = time_max
        self._freq_min = freq_min
        self._freq_max = freq_max
        self._time_step = time_step
        self._freq_step = freq_step

    @property
    def values(self) -> np.ndarray:
        """Power values (n_freqs × n_times)."""
        return self._values

    @property
    def n_times(self) -> int:
        """Number of time frames."""
        return self._values.shape[1]

    @property
    def n_freqs(self) -> int:
        """Number of frequency bins."""
        return self._values.shape[0]

    @property
    def time_min(self) -> float:
        """Start time in seconds."""
        return self._time_min

    @property
    def time_max(self) -> float:
        """End time in seconds."""
        return self._time_max

    @property
    def freq_min(self) -> float:
        """Minimum frequency in Hz."""
        return self._freq_min

    @property
    def freq_max(self) -> float:
        """Maximum frequency in Hz."""
        return self._freq_max

    @property
    def time_step(self) -> float:
        """Time step between frames."""
        return self._time_step

    @property
    def freq_step(self) -> float:
        """Frequency step between bins."""
        return self._freq_step

    def get_time_from_frame(self, frame: int) -> float:
        """Get time for a frame index (0-based)."""
        return self._time_min + frame * self._time_step

    def get_freq_from_bin(self, bin_index: int) -> float:
        """Get frequency for a bin index (0-based)."""
        return self._freq_min + bin_index * self._freq_step

    def times(self) -> np.ndarray:
        """Get array of time points."""
        return np.array([self.get_time_from_frame(i) for i in range(self.n_times)])

    def frequencies(self) -> np.ndarray:
        """Get array of frequency points."""
        return np.array([self.get_freq_from_bin(i) for i in range(self.n_freqs)])


def sound_to_spectrogram(
    sound: "Sound",
    window_length: float = 0.005,
    max_frequency: float = 5000.0,
    time_step: float = 0.002,
    frequency_step: float = 20.0,
    window_shape: str = "Gaussian"
) -> Spectrogram:
    """
    Compute spectrogram from sound.

    Args:
        sound: Sound object
        window_length: Window length in seconds
        max_frequency: Maximum frequency in Hz
        time_step: Time step in seconds
        frequency_step: Frequency step in Hz
        window_shape: Window shape ("Gaussian", "Hanning", etc.)

    Returns:
        Spectrogram object
    """
    raise NotImplementedError()
