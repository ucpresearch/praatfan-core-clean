"""
Intensity - RMS energy contour in dB.

Implementation order: Phase 3
Dependencies: None (independent)

Documentation sources:
- Praat manual: Sound: To Intensity...
- Praat manual: Intro 6.2. Configuring the intensity contour

Key documented facts:
- Window: "Gaussian analysis window (Kaiser-20; sidelobes below -190 dB)"
- Effective duration: 3.2 / min_pitch
- Default time step: 0.8 / min_pitch (1/4 of effective duration)
- DC removal: "subtracting the mean... then applying the window"

Decision points determined via testing:
- DP3: Window type - Gaussian α=13.2 gave best results in testing. Documentation says
  "Gaussian (Kaiser-20)" which is under-specified. Kaiser might work with different
  parameterization; current implementation uses Gaussian for practical accuracy.
- DP4: Physical vs effective window ratio = 2.25× (empirically determined)
- DP5: DC removal = unweighted mean subtraction before windowing
"""

import numpy as np
from typing import List, Optional


class Intensity:
    """
    Intensity contour (loudness over time).

    Values are in dB relative to a reference pressure.

    Attributes:
        times: Time points in seconds
        values: Intensity values in dB
        time_step: Time step between frames
    """

    def __init__(
        self,
        times: np.ndarray,
        values: np.ndarray,
        time_step: float,
        min_pitch: float
    ):
        """
        Create an Intensity object.

        Args:
            times: Time points in seconds
            values: Intensity values in dB
            time_step: Time step between frames
            min_pitch: Minimum pitch used for analysis
        """
        self._times = np.asarray(times, dtype=np.float64)
        self._values = np.asarray(values, dtype=np.float64)
        self._time_step = time_step
        self._min_pitch = min_pitch

    @property
    def times(self) -> np.ndarray:
        """Time points in seconds."""
        return self._times

    @property
    def values(self) -> np.ndarray:
        """Intensity values in dB."""
        return self._values

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return len(self._times)

    @property
    def time_step(self) -> float:
        """Time step between frames."""
        return self._time_step

    def get_value_at_time(self, time: float, interpolation: str = "cubic") -> Optional[float]:
        """
        Get intensity value at a specific time.

        Args:
            time: Time in seconds
            interpolation: Interpolation method ("cubic", "linear", "nearest")

        Returns:
            Intensity in dB, or None if outside range
        """
        raise NotImplementedError()


def _kaiser_window(n: int, beta: float = 20.0) -> np.ndarray:
    """
    Generate Kaiser window.

    The Kaiser window achieves a specified sidelobe attenuation.
    Kaiser-20 means beta ≈ 20, giving approximately -155 dB sidelobes.

    Args:
        n: Window length
        beta: Kaiser beta parameter (higher = narrower main lobe, lower sidelobes)

    Returns:
        Window array normalized to [0, 1]
    """
    if n <= 1:
        return np.array([1.0])

    from scipy import special

    m = n - 1
    alpha = m / 2.0
    i0_beta = special.i0(beta)

    window = np.zeros(n)
    for i in range(n):
        x = beta * np.sqrt(1.0 - ((i - alpha) / alpha) ** 2)
        window[i] = special.i0(x) / i0_beta

    return window


def _gauss_window(n: int, alpha: float = 12.0) -> np.ndarray:
    """
    Generate Gaussian window matching Praat's intensity analysis.

    The Praat manual mentions "Gaussian analysis window (Kaiser-20)".
    Through black-box testing, we determine the exact parameters.

    Based on Boersma (1993) postscript, the Gaussian window is:
        w(t) = (exp(-α × (t/T - 0.5)²) - exp(-α/4)) / (1 - exp(-α/4))

    with α chosen to achieve desired sidelobe attenuation.

    Args:
        n: Window length
        alpha: Gaussian width parameter (default 12 from Boersma 1993)
    """
    if n <= 1:
        return np.array([1.0])

    mid = (n - 1) / 2.0
    t = np.arange(n)

    # Normalized position: (t/T - 0.5) where T = n-1
    x = (t - mid) / mid  # Range [-1, 1]

    # Gaussian with edge correction (ensures window goes to 0 at edges)
    exp_term = np.exp(-alpha * x * x)
    exp_edge = np.exp(-alpha)

    window = (exp_term - exp_edge) / (1.0 - exp_edge)

    return window


def sound_to_intensity(
    sound: "Sound",
    min_pitch: float = 100.0,
    time_step: float = 0.0,
    subtract_mean: bool = True
) -> Intensity:
    """
    Compute intensity from sound.

    Algorithm (from Praat manual):
    1. Square the signal values
    2. Convolve with Gaussian window (after DC removal if enabled)
    3. Convert to dB

    Documented parameters:
    - Effective duration: 3.2 / min_pitch
    - Physical duration: 7.2 / min_pitch (2.25× effective, determined via testing)
    - Time step: 0.8 / min_pitch (default)

    Args:
        sound: Sound object
        min_pitch: Minimum pitch in Hz (determines window size)
        time_step: Time step in seconds (0 = auto: 0.8 / min_pitch)
        subtract_mean: Whether to subtract DC before analysis

    Returns:
        Intensity object
    """
    samples = sound.samples
    sample_rate = sound.sample_rate
    duration = sound.duration
    dt = 1.0 / sample_rate

    # Default time step (documented)
    if time_step <= 0:
        time_step = 0.8 / min_pitch

    # Window duration (determined via black-box testing: ratio = 2.25)
    # physical_duration = 7.2 / min_pitch
    physical_window_duration = 7.2 / min_pitch
    half_window_duration = physical_window_duration / 2.0

    # Number of samples in window
    window_samples = int(round(physical_window_duration * sample_rate))
    if window_samples % 2 == 0:
        window_samples += 1  # Ensure odd for symmetric window

    half_window_samples = window_samples // 2

    # Generate window
    # Alpha = 13.2 determined via black-box testing (DP3)
    window = _gauss_window(window_samples, alpha=13.2)

    # Frame timing (left-aligned, determined via testing DP1)
    # t1 = half_window_duration
    # Last frame at t_max = duration - half_window_duration
    # n_frames = floor((t_max - t1) / time_step) + 1
    t1 = half_window_duration
    t_max = duration - half_window_duration

    # Use small epsilon for floating point precision
    n_frames = int(np.floor((t_max - t1) / time_step + 1e-9)) + 1
    if n_frames < 1:
        n_frames = 1

    # Compute intensity for each frame
    times = np.zeros(n_frames)
    values = np.zeros(n_frames)

    # Reference pressure (standard: 2e-5 Pa)
    # But Praat may use 1.0 for normalized audio - need to verify
    # For now, use 1.0 (relative intensity in arbitrary units)
    p_ref = 4e-10  # (2e-5)² for pressure squared reference

    for i in range(n_frames):
        t = t1 + i * time_step
        times[i] = t

        # Find center sample index
        center_sample = int(round(t * sample_rate))

        # Extract window region
        start_sample = center_sample - half_window_samples
        end_sample = start_sample + window_samples  # Use window_samples for consistent length

        # Handle boundary conditions (zero-padding)
        if start_sample < 0 or end_sample > len(samples):
            # Pad with zeros
            frame_samples = np.zeros(window_samples)
            src_start = max(0, start_sample)
            src_end = min(len(samples), end_sample)
            dst_start = src_start - start_sample
            dst_end = dst_start + (src_end - src_start)
            frame_samples[dst_start:dst_end] = samples[src_start:src_end]
        else:
            frame_samples = samples[start_sample:end_sample].copy()

        # Subtract mean (DC removal) - documented: "first subtracting the mean"
        if subtract_mean:
            frame_samples = frame_samples - np.mean(frame_samples)

        # Compute weighted mean square
        # Formula: mean_sq = sum(x² × w) / sum(w)
        window_sum = np.sum(window)
        mean_square = np.sum(frame_samples * frame_samples * window) / window_sum

        # Convert to dB
        if mean_square <= 0:
            intensity_db = -np.inf
        else:
            intensity_db = 10.0 * np.log10(mean_square / p_ref)

        values[i] = intensity_db

    return Intensity(times, values, time_step, min_pitch)
