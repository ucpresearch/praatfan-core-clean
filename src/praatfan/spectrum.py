"""
Spectrum - Single-frame FFT magnitude spectrum.

Implementation order: Phase 2 (after foundation)
Dependencies: None (foundation)

Documentation sources:
- Praat manual: Spectrum, Spectrum: Get centre of gravity...
- Standard FFT definition
"""

import numpy as np
from typing import Optional


class Spectrum:
    """
    Single-frame FFT spectrum.

    The spectrum stores complex values for frequencies from 0 to Nyquist.
    Negative frequencies are not stored (conjugate symmetric for real signals).

    Attributes:
        real: Real parts of spectrum bins
        imag: Imaginary parts of spectrum bins
        df: Frequency resolution (bin width) in Hz
        f_max: Maximum frequency (Nyquist) in Hz
    """

    def __init__(
        self,
        real: np.ndarray,
        imag: np.ndarray,
        df: float,
        f_max: float
    ):
        """
        Create a Spectrum.

        Args:
            real: Real parts of spectrum bins
            imag: Imaginary parts of spectrum bins
            df: Frequency resolution in Hz
            f_max: Maximum frequency in Hz
        """
        self._real = np.asarray(real, dtype=np.float64)
        self._imag = np.asarray(imag, dtype=np.float64)
        self._df = df
        self._f_max = f_max

    @property
    def real(self) -> np.ndarray:
        """Real parts of spectrum bins."""
        return self._real

    @property
    def imag(self) -> np.ndarray:
        """Imaginary parts of spectrum bins."""
        return self._imag

    @property
    def df(self) -> float:
        """Frequency resolution (bin width) in Hz."""
        return self._df

    @property
    def f_max(self) -> float:
        """Maximum frequency (Nyquist) in Hz."""
        return self._f_max

    @property
    def n_bins(self) -> int:
        """Number of frequency bins."""
        return len(self._real)

    def get_frequency(self, bin_index: int) -> float:
        """Get frequency for a bin index."""
        return bin_index * self._df

    # Spectral moments - to be implemented

    def get_center_of_gravity(self, power: float = 2.0) -> float:
        """
        Compute center of gravity (spectral centroid).

        Formula (documented in Praat manual):
            f_c = ∫ f × |S(f)|^p df / ∫ |S(f)|^p df

        Args:
            power: Power to raise magnitude to (default 2.0)

        Returns:
            Center of gravity in Hz
        """
        raise NotImplementedError()

    def get_standard_deviation(self, power: float = 2.0) -> float:
        """
        Compute standard deviation of the spectrum.

        Formula: sqrt(μ₂) where μ₂ is the second central moment.

        Args:
            power: Power to raise magnitude to (default 2.0)

        Returns:
            Standard deviation in Hz
        """
        raise NotImplementedError()

    def get_skewness(self, power: float = 2.0) -> float:
        """
        Compute skewness of the spectrum.

        Formula: μ₃ / μ₂^1.5

        Args:
            power: Power to raise magnitude to (default 2.0)

        Returns:
            Skewness (dimensionless)
        """
        raise NotImplementedError()

    def get_kurtosis(self, power: float = 2.0) -> float:
        """
        Compute kurtosis of the spectrum.

        Formula: μ₄ / μ₂² - 3 (excess kurtosis)

        Args:
            power: Power to raise magnitude to (default 2.0)

        Returns:
            Kurtosis (dimensionless)
        """
        raise NotImplementedError()

    def get_band_energy(self, f_min: float = 0.0, f_max: float = 0.0) -> float:
        """
        Compute energy in a frequency band.

        Formula: E = ∫_{f_min}^{f_max} |S(f)|² df

        Note: May need factor of 2 for one-sided spectrum (DP9).

        Args:
            f_min: Minimum frequency (0 = DC)
            f_max: Maximum frequency (0 = Nyquist)

        Returns:
            Band energy
        """
        raise NotImplementedError()


def sound_to_spectrum(sound: "Sound", fast: bool = True) -> Spectrum:
    """
    Compute spectrum from sound.

    Args:
        sound: Sound object
        fast: If True, use power-of-2 FFT size

    Returns:
        Spectrum object
    """
    raise NotImplementedError()
