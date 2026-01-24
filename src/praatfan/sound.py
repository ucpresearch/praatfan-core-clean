"""
Sound - Audio samples with sample rate.

This is the foundation type for all acoustic analysis.
"""

import numpy as np
from pathlib import Path
from typing import Union


class Sound:
    """
    Represents audio samples with sample rate.

    Attributes:
        samples: 1D numpy array of audio samples (mono only)
        sample_rate: Sample rate in Hz

    Properties:
        duration: Total duration in seconds
        n_samples: Number of samples
        dx: Sample period (1 / sample_rate)
        x1: Time of first sample (0.5 * dx, centered on first sample)
    """

    def __init__(self, samples: np.ndarray, sample_rate: float):
        """
        Create a Sound from samples and sample rate.

        Args:
            samples: 1D numpy array of audio samples
            sample_rate: Sample rate in Hz

        Raises:
            ValueError: If samples is not 1D (mono only supported)
        """
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 1:
            raise ValueError("Only mono audio supported. Got shape: {}".format(samples.shape))

        self._samples = samples
        self._sample_rate = float(sample_rate)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Sound":
        """
        Load audio from a file.

        Supports WAV, FLAC, and other formats via soundfile.
        Multi-channel files will raise an error.

        Args:
            path: Path to audio file

        Returns:
            Sound object

        Raises:
            ValueError: If file has multiple channels
        """
        import soundfile as sf

        data, sample_rate = sf.read(path, dtype='float64')

        if data.ndim > 1:
            raise ValueError(
                f"Only mono audio supported. File has {data.shape[1]} channels. "
                "Use Sound.from_file_channel() to select a specific channel."
            )

        return cls(data, sample_rate)

    @classmethod
    def from_file_channel(cls, path: Union[str, Path], channel: int = 0) -> "Sound":
        """
        Load a specific channel from an audio file.

        Args:
            path: Path to audio file
            channel: Channel index (0-based)

        Returns:
            Sound object with the specified channel
        """
        import soundfile as sf

        data, sample_rate = sf.read(path, dtype='float64')

        if data.ndim == 1:
            if channel != 0:
                raise ValueError(f"File is mono, channel {channel} does not exist")
            return cls(data, sample_rate)

        if channel >= data.shape[1]:
            raise ValueError(f"Channel {channel} does not exist. File has {data.shape[1]} channels.")

        return cls(data[:, channel], sample_rate)

    @property
    def samples(self) -> np.ndarray:
        """Audio samples as 1D numpy array."""
        return self._samples

    @property
    def sample_rate(self) -> float:
        """Sample rate in Hz."""
        return self._sample_rate

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return len(self._samples)

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return self.n_samples / self._sample_rate

    @property
    def dx(self) -> float:
        """Sample period (1 / sample_rate)."""
        return 1.0 / self._sample_rate

    @property
    def x1(self) -> float:
        """Time of first sample (centered on sample)."""
        return 0.5 * self.dx

    def __repr__(self) -> str:
        return f"Sound({self.n_samples} samples, {self.sample_rate} Hz, {self.duration:.3f}s)"

    # Analysis methods - to be implemented

    def to_spectrum(self, fast: bool = True) -> "Spectrum":
        """
        Compute the spectrum (single-frame FFT).

        Args:
            fast: If True, use power-of-2 FFT size for speed

        Returns:
            Spectrum object
        """
        from .spectrum import Spectrum
        raise NotImplementedError("Spectrum not yet implemented")

    def to_intensity(self, min_pitch: float = 100.0, time_step: float = 0.0) -> "Intensity":
        """
        Compute intensity contour.

        Args:
            min_pitch: Minimum pitch in Hz (determines window size)
            time_step: Time step in seconds (0 = auto)

        Returns:
            Intensity object
        """
        from .intensity import Intensity
        raise NotImplementedError("Intensity not yet implemented")

    def to_pitch(
        self,
        time_step: float = 0.0,
        pitch_floor: float = 75.0,
        pitch_ceiling: float = 600.0
    ) -> "Pitch":
        """
        Compute pitch (F0) contour using autocorrelation method.

        Args:
            time_step: Time step in seconds (0 = auto)
            pitch_floor: Minimum pitch in Hz
            pitch_ceiling: Maximum pitch in Hz

        Returns:
            Pitch object
        """
        from .pitch import Pitch
        raise NotImplementedError("Pitch not yet implemented")

    def to_harmonicity_ac(
        self,
        time_step: float = 0.01,
        min_pitch: float = 75.0,
        silence_threshold: float = 0.1,
        periods_per_window: float = 4.5
    ) -> "Harmonicity":
        """
        Compute harmonicity (HNR) using autocorrelation method.

        Args:
            time_step: Time step in seconds
            min_pitch: Minimum pitch in Hz
            silence_threshold: Silence threshold (0-1)
            periods_per_window: Number of periods per window

        Returns:
            Harmonicity object
        """
        from .harmonicity import Harmonicity
        raise NotImplementedError("Harmonicity not yet implemented")

    def to_harmonicity_cc(
        self,
        time_step: float = 0.01,
        min_pitch: float = 75.0,
        silence_threshold: float = 0.1,
        periods_per_window: float = 1.0
    ) -> "Harmonicity":
        """
        Compute harmonicity (HNR) using cross-correlation method.

        Args:
            time_step: Time step in seconds
            min_pitch: Minimum pitch in Hz
            silence_threshold: Silence threshold (0-1)
            periods_per_window: Number of periods per window

        Returns:
            Harmonicity object
        """
        from .harmonicity import Harmonicity
        raise NotImplementedError("Harmonicity not yet implemented")

    def to_formant_burg(
        self,
        time_step: float = 0.0,
        max_num_formants: int = 5,
        max_formant_hz: float = 5500.0,
        window_length: float = 0.025,
        pre_emphasis_from: float = 50.0
    ) -> "Formant":
        """
        Compute formants using Burg's LPC method.

        Args:
            time_step: Time step in seconds (0 = auto: 25% of window)
            max_num_formants: Maximum number of formants to find
            max_formant_hz: Maximum formant frequency in Hz
            window_length: Window length in seconds (actual = 2x this value)
            pre_emphasis_from: Pre-emphasis from frequency in Hz

        Returns:
            Formant object
        """
        from .formant import Formant
        raise NotImplementedError("Formant not yet implemented")

    def to_spectrogram(
        self,
        window_length: float = 0.005,
        max_frequency: float = 5000.0,
        time_step: float = 0.002,
        frequency_step: float = 20.0
    ) -> "Spectrogram":
        """
        Compute spectrogram (time-frequency representation).

        Args:
            window_length: Window length in seconds
            max_frequency: Maximum frequency in Hz
            time_step: Time step in seconds
            frequency_step: Frequency step in Hz

        Returns:
            Spectrogram object
        """
        from .spectrogram import Spectrogram
        raise NotImplementedError("Spectrogram not yet implemented")
