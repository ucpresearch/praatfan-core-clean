"""
Sound - Audio samples with sample rate.

This module provides the Sound class, which is the foundation type for all
acoustic analysis in praatfan. A Sound object holds audio samples and sample
rate, and provides methods to compute various acoustic analyses.

Design Principles:
------------------
1. Mono only: Multi-channel audio is not supported. Use from_file_channel()
   to select a specific channel from multi-channel files.

2. Float64 samples: Audio is stored as 64-bit floating point, normalized
   to the range [-1, 1] for PCM formats.

3. Lazy imports: Analysis modules (pitch, formant, etc.) are imported only
   when the corresponding method is called. This keeps the base Sound class
   lightweight.

4. Praat-compatible timing: The x1 property gives the time of the first
   sample center (0.5 * dx), matching Praat's convention.

Supported Audio Formats:
------------------------
Via soundfile/libsndfile:
  - WAV (PCM 8/16/24/32-bit, 32/64-bit float, u-law, a-law)
  - FLAC (Free Lossless Audio Codec)
  - MP3 (requires libsndfile 1.1.0+)
  - OGG Vorbis
  - AIFF, AU, CAF, and many others

Usage:
------
    from praatfan import Sound

    # Load from file
    sound = Sound.from_file("audio.wav")

    # Or from numpy array
    import numpy as np
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
    sound = Sound(samples, sample_rate=16000)

    # Analyze
    pitch = sound.to_pitch()
    formant = sound.to_formant_burg()
    intensity = sound.to_intensity()

    # Per-window spectral features
    spectrum = sound.get_spectrum_at_time(0.5)
    moments = sound.get_spectral_moments_at_times(times)
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

        Supported formats (via soundfile/libsndfile):
            - WAV (PCM 8/16/24/32-bit, 32/64-bit float, u-law, a-law)
            - FLAC (Free Lossless Audio Codec)
            - MP3 (MPEG Audio Layer III) - requires libsndfile 1.1.0+
            - OGG Vorbis
            - AIFF, AU, CAF, and many others

        Multi-channel files will raise an error - use from_file_channel() instead.

        Args:
            path: Path to audio file

        Returns:
            Sound object

        Raises:
            ValueError: If file has multiple channels
            RuntimeError: If file format is not supported
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
        from .spectrum import sound_to_spectrum
        return sound_to_spectrum(self, fast=fast)

    def to_intensity(self, min_pitch: float = 100.0, time_step: float = 0.0) -> "Intensity":
        """
        Compute intensity contour.

        Args:
            min_pitch: Minimum pitch in Hz (determines window size)
            time_step: Time step in seconds (0 = auto)

        Returns:
            Intensity object
        """
        from .intensity import sound_to_intensity
        return sound_to_intensity(self, min_pitch=min_pitch, time_step=time_step)

    def to_pitch(
        self,
        time_step: float = 0.0,
        pitch_floor: float = 75.0,
        pitch_ceiling: float = 600.0,
        method: str = "ac"
    ) -> "Pitch":
        """
        Compute pitch (F0) contour.

        Args:
            time_step: Time step in seconds (0 = auto)
            pitch_floor: Minimum pitch in Hz
            pitch_ceiling: Maximum pitch in Hz
            method: Analysis method ("ac" for autocorrelation, "cc" for cross-correlation)

        Returns:
            Pitch object
        """
        from .pitch import sound_to_pitch
        return sound_to_pitch(self, time_step=time_step,
                             pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling,
                             method=method)

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
        from .harmonicity import sound_to_harmonicity_ac
        return sound_to_harmonicity_ac(
            self, time_step=time_step, min_pitch=min_pitch,
            silence_threshold=silence_threshold, periods_per_window=periods_per_window
        )

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
        from .harmonicity import sound_to_harmonicity_cc
        return sound_to_harmonicity_cc(
            self, time_step=time_step, min_pitch=min_pitch,
            silence_threshold=silence_threshold, periods_per_window=periods_per_window
        )

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
        from .formant import sound_to_formant_burg
        return sound_to_formant_burg(
            self, time_step=time_step, max_num_formants=max_num_formants,
            max_formant_hz=max_formant_hz, window_length=window_length,
            pre_emphasis_from=pre_emphasis_from
        )

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
        from .spectrogram import sound_to_spectrogram
        return sound_to_spectrogram(
            self, window_length=window_length, max_frequency=max_frequency,
            time_step=time_step, frequency_step=frequency_step
        )

    # =========================================================================
    # Per-Window Spectral Feature Extraction
    # =========================================================================
    #
    # These methods support extracting spectral features at specific time points.
    # This is useful for:
    #   - Analyzing spectral characteristics at formant measurement points
    #   - Computing spectral tilt or balance features aligned with F0 tracks
    #   - Extracting spectral moments at regular intervals for feature vectors
    #
    # The approach is:
    #   1. extract_part() - slice out a time window
    #   2. to_spectrum() - compute single-frame FFT
    #   3. Spectral moment methods - compute CoG, std, skewness, kurtosis
    #
    # This modular design reuses existing spectrum code rather than duplicating
    # the FFT and moment calculations.
    # =========================================================================

    def extract_part(self, start_time: float, end_time: float) -> "Sound":
        """
        Extract a portion of the sound.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            New Sound object containing the extracted portion
        """
        start_sample = int(start_time * self._sample_rate)
        end_sample = int(end_time * self._sample_rate)

        # Clamp to valid range
        start_sample = max(0, start_sample)
        end_sample = min(len(self._samples), end_sample)

        extracted = self._samples[start_sample:end_sample].copy()
        return Sound(extracted, self._sample_rate)

    def get_spectrum_at_time(
        self,
        time: float,
        window_length: float = 0.025
    ) -> "Spectrum":
        """
        Extract spectrum for a window centered at a specific time.

        Args:
            time: Center time in seconds
            window_length: Window length in seconds

        Returns:
            Spectrum object for that window
        """
        half_window = window_length / 2.0
        start_time = max(0.0, time - half_window)
        end_time = min(self.duration, time + half_window)

        window_sound = self.extract_part(start_time, end_time)
        return window_sound.to_spectrum()

    def get_spectral_moments_at_times(
        self,
        times: "np.ndarray",
        window_length: float = 0.025,
        power: float = 2.0
    ) -> dict:
        """
        Compute spectral moments at specified time points.

        Uses existing Spectrum methods for each window.

        Args:
            times: Array of time points in seconds
            window_length: Window length in seconds
            power: Power parameter for moment calculation

        Returns:
            Dictionary with keys:
                - 'times': Input time array
                - 'center_of_gravity': Array of CoG values (Hz)
                - 'standard_deviation': Array of std dev values (Hz)
                - 'skewness': Array of skewness values
                - 'kurtosis': Array of kurtosis values
        """
        import numpy as np

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
            'times': np.asarray(times),
            'center_of_gravity': cog,
            'standard_deviation': std,
            'skewness': skew,
            'kurtosis': kurt
        }

    def get_band_energy_at_times(
        self,
        times: "np.ndarray",
        f_min: float,
        f_max: float,
        window_length: float = 0.025
    ) -> "np.ndarray":
        """
        Compute band energy at specified time points.

        Args:
            times: Array of time points in seconds
            f_min: Minimum frequency in Hz
            f_max: Maximum frequency in Hz
            window_length: Window length in seconds

        Returns:
            Array of band energy values
        """
        import numpy as np

        n = len(times)
        energy = np.zeros(n)

        for i, t in enumerate(times):
            spectrum = self.get_spectrum_at_time(t, window_length)
            energy[i] = spectrum.get_band_energy(f_min, f_max)

        return energy
