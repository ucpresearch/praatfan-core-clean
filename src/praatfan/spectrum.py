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

        In discrete form (df cancels in ratio):
            f_c = Σ f_k × |S_k|^p / Σ |S_k|^p

        Args:
            power: Power to raise magnitude to (default 2.0)

        Returns:
            Center of gravity in Hz
        """
        # Compute magnitude
        magnitude = np.sqrt(self._real**2 + self._imag**2)

        # Raise to power
        weighted = magnitude ** power

        # Frequency array
        freqs = np.arange(self.n_bins) * self._df

        # Compute weighted sum
        numerator = np.sum(freqs * weighted)
        denominator = np.sum(weighted)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _get_central_moment(self, n: int, power: float) -> float:
        """
        Compute nth central moment.

        Formula (documented in Praat manual):
            μ_n = ∫ (f - f_c)^n × |S(f)|^p df / ∫ |S(f)|^p df

        Args:
            n: Moment order
            power: Power to raise magnitude to

        Returns:
            nth central moment
        """
        # Compute magnitude
        magnitude = np.sqrt(self._real**2 + self._imag**2)

        # Raise to power
        weighted = magnitude ** power

        # Frequency array
        freqs = np.arange(self.n_bins) * self._df

        # Get center of gravity
        denominator = np.sum(weighted)
        if denominator == 0:
            return 0.0

        cog = np.sum(freqs * weighted) / denominator

        # Compute central moment
        deviation = freqs - cog
        numerator = np.sum((deviation ** n) * weighted)

        return numerator / denominator

    def get_standard_deviation(self, power: float = 2.0) -> float:
        """
        Compute standard deviation of the spectrum.

        Formula: sqrt(μ₂) where μ₂ is the second central moment.

        Args:
            power: Power to raise magnitude to (default 2.0)

        Returns:
            Standard deviation in Hz
        """
        mu2 = self._get_central_moment(2, power)
        return np.sqrt(mu2)

    def get_skewness(self, power: float = 2.0) -> float:
        """
        Compute skewness of the spectrum.

        Formula (documented in Praat manual): μ₃ / μ₂^1.5

        Args:
            power: Power to raise magnitude to (default 2.0)

        Returns:
            Skewness (dimensionless)
        """
        mu2 = self._get_central_moment(2, power)
        mu3 = self._get_central_moment(3, power)

        if mu2 == 0:
            return 0.0

        return mu3 / (mu2 ** 1.5)

    def get_kurtosis(self, power: float = 2.0) -> float:
        """
        Compute kurtosis of the spectrum.

        Formula (documented in Praat manual): μ₄ / μ₂² - 3 (excess kurtosis)

        Args:
            power: Power to raise magnitude to (default 2.0)

        Returns:
            Kurtosis (dimensionless)
        """
        mu2 = self._get_central_moment(2, power)
        mu4 = self._get_central_moment(4, power)

        if mu2 == 0:
            return 0.0

        return mu4 / (mu2 ** 2) - 3.0

    def get_band_energy(self, f_min: float = 0.0, f_max: float = 0.0) -> float:
        """
        Compute energy in a frequency band.

        Formula: E = ∫_{f_min}^{f_max} |S(f)|² df

        For one-sided spectrum, we need to account for negative frequencies.
        DP9 determines the exact factor (testing needed).

        Args:
            f_min: Minimum frequency (0 = DC)
            f_max: Maximum frequency (0 = Nyquist)

        Returns:
            Band energy (Pa² s)
        """
        if f_max <= 0:
            f_max = self._f_max

        # Find bin indices for the frequency range
        bin_min = int(np.floor(f_min / self._df))
        bin_max = int(np.ceil(f_max / self._df))

        # Clamp to valid range
        bin_min = max(0, bin_min)
        bin_max = min(self.n_bins - 1, bin_max)

        # Compute magnitude squared
        mag_squared = self._real**2 + self._imag**2

        # Sum energy in band, multiplied by df for integration
        # Factor of 2 accounts for negative frequencies (one-sided spectrum)
        # Exception: DC (bin 0) and Nyquist (last bin) are not doubled
        energy = 0.0
        for i in range(bin_min, bin_max + 1):
            bin_energy = mag_squared[i] * self._df
            if i == 0 or i == self.n_bins - 1:
                # DC and Nyquist bins are not doubled
                energy += bin_energy
            else:
                # Other bins account for conjugate symmetric negative frequencies
                energy += 2.0 * bin_energy

        return energy


def sound_to_spectrum(sound: "Sound", fast: bool = True) -> Spectrum:
    """
    Compute spectrum from sound.

    Documented formula (Praat manual, standard DFT):
        X[k] = Σₙ x[n] × e^(-2πikn/N) × Δt

    The multiplication by Δt (sample period) converts the discrete sum
    to an approximation of the continuous Fourier transform integral.

    Args:
        sound: Sound object
        fast: If True, use power-of-2 FFT size for speed

    Returns:
        Spectrum object
    """
    samples = sound.samples
    n_samples = len(samples)
    sample_rate = sound.sample_rate
    dt = 1.0 / sample_rate

    if fast:
        # Zero-pad to next power of 2
        fft_size = 1
        while fft_size < n_samples:
            fft_size *= 2
    else:
        fft_size = n_samples

    # Zero-pad samples
    padded = np.zeros(fft_size)
    padded[:n_samples] = samples

    # Compute FFT
    fft_result = np.fft.fft(padded)

    # Multiply by dt (documented: integral approximation)
    fft_result *= dt

    # Keep only positive frequencies (0 to Nyquist inclusive)
    # For real signals, negative frequencies are conjugate symmetric
    n_positive = fft_size // 2 + 1
    real_part = fft_result[:n_positive].real.copy()
    imag_part = fft_result[:n_positive].imag.copy()

    # Frequency resolution and maximum frequency
    df = sample_rate / fft_size
    f_max = sample_rate / 2.0

    return Spectrum(real_part, imag_part, df, f_max)
