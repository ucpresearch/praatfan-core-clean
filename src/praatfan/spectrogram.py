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
        freq_step: float,
        t1: float = 0.0
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
            t1: Time of first frame center
        """
        self._values = np.asarray(values, dtype=np.float64)
        self._time_min = time_min
        self._time_max = time_max
        self._freq_min = freq_min
        self._freq_max = freq_max
        self._time_step = time_step
        self._freq_step = freq_step
        self._t1 = t1

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
        return self._t1 + frame * self._time_step

    def get_freq_from_bin(self, bin_index: int) -> float:
        """Get frequency for a bin index (0-based)."""
        return self._freq_min + bin_index * self._freq_step

    def times(self) -> np.ndarray:
        """Get array of time points."""
        return np.array([self.get_time_from_frame(i) for i in range(self.n_times)])

    def frequencies(self) -> np.ndarray:
        """Get array of frequency points."""
        return np.array([self.get_freq_from_bin(i) for i in range(self.n_freqs)])


def _gaussian_window(n: int, alpha: float = 12.0) -> np.ndarray:
    """
    Generate Gaussian window for spectrogram.

    Uses the same Gaussian formula as Intensity analysis.

    Args:
        n: Window length in samples
        alpha: Gaussian width parameter (default 12.0)

    Returns:
        Energy-normalized Gaussian window
    """
    if n <= 1:
        return np.array([1.0])

    mid = (n - 1) / 2.0
    i = np.arange(n)
    x = (i - mid) / mid  # Range [-1, 1]

    window = np.exp(-alpha * x * x)

    # Energy normalization: divide by sqrt(sum of squares)
    window = window / np.sqrt(np.sum(window ** 2))

    return window


def _hanning_window(n: int) -> np.ndarray:
    """Generate Hanning window."""
    if n <= 1:
        return np.array([1.0])
    i = np.arange(n)
    return 0.5 - 0.5 * np.cos(2 * np.pi * i / (n - 1))


def sound_to_spectrogram(
    sound: "Sound",
    window_length: float = 0.005,
    max_frequency: float = 5000.0,
    time_step: float = 0.002,
    frequency_step: float = 20.0,
    window_shape: str = "Gaussian"
) -> Spectrogram:
    """
    Compute spectrogram from sound using Short-Time Fourier Transform.

    Args:
        sound: Sound object
        window_length: Effective window length in seconds (default 0.005)
        max_frequency: Maximum frequency in Hz (default 5000)
        time_step: Time step in seconds (default 0.002)
        frequency_step: Frequency step in Hz (default 20)
        window_shape: Window shape ("Gaussian" or "Hanning")

    Returns:
        Spectrogram object

    Note:
        For Gaussian windows, the physical window is twice the effective
        window length (documented: "analyzes a factor of 2 slower").
    """
    samples = sound.samples
    sample_rate = sound.sample_rate
    duration = sound.duration

    # For Gaussian window, physical window is 2x effective window
    if window_shape.lower() == "gaussian":
        physical_window_duration = 2.0 * window_length
    else:
        physical_window_duration = window_length

    # Number of samples in physical window
    window_samples = int(round(physical_window_duration * sample_rate))
    if window_samples % 2 == 0:
        window_samples += 1
    half_window = window_samples // 2

    # Generate window function
    if window_shape.lower() == "gaussian":
        window = _gaussian_window(window_samples)
    else:
        window = _hanning_window(window_samples)

    # Frame timing - centered (without 1e-9 to match Praat)
    n_frames = int(np.floor((duration - physical_window_duration) / time_step)) + 1
    if n_frames < 1:
        n_frames = 1
    t1 = (duration - (n_frames - 1) * time_step) / 2.0

    # Frequency bins from 0 to max_frequency (exclusive at max)
    n_freq_bins = int(round(max_frequency / frequency_step))
    freq_step_actual = frequency_step

    # FFT size - must be large enough for desired frequency resolution
    # The frequency resolution of FFT is sample_rate / fft_size
    # We need fft_size >= sample_rate / frequency_step
    min_fft_size = int(np.ceil(sample_rate / frequency_step))
    fft_size = 1
    while fft_size < max(window_samples, min_fft_size):
        fft_size *= 2

    # Frequency resolution from FFT
    df_fft = sample_rate / fft_size

    # Compute spectrogram
    # Values array is (n_freq_bins, n_frames)
    values = np.zeros((n_freq_bins, n_frames))

    for i in range(n_frames):
        t = t1 + i * time_step

        # Extract frame centered at t
        center = int(round(t * sample_rate))
        start = center - half_window
        end = start + window_samples

        # Handle boundaries
        if start < 0 or end > len(samples):
            frame = np.zeros(window_samples)
            src_start = max(0, start)
            src_end = min(len(samples), end)
            dst_start = src_start - start
            dst_end = dst_start + (src_end - src_start)
            frame[dst_start:dst_end] = samples[src_start:src_end]
        else:
            frame = samples[start:end].copy()

        # Apply window
        windowed = frame * window

        # Zero-pad to FFT size
        padded = np.zeros(fft_size)
        padded[:window_samples] = windowed

        # Compute FFT
        fft_result = np.fft.fft(padded)

        # Compute power: |X(f)|^2
        # Window is already energy-normalized, so no additional scaling needed
        power = np.abs(fft_result[:fft_size // 2 + 1]) ** 2

        # Extract power at desired frequency bins
        for j in range(n_freq_bins):
            freq = j * frequency_step
            # Find nearest FFT bin
            fft_bin = int(round(freq / df_fft))
            if fft_bin < len(power):
                values[j, i] = power[fft_bin]

    return Spectrogram(
        values=values,
        time_min=0.0,
        time_max=duration,
        freq_min=0.0,
        freq_max=max_frequency,
        time_step=time_step,
        freq_step=frequency_step,
        t1=t1
    )
