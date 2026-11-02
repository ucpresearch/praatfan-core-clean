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
        if self.n_frames == 0:
            return None

        # Find position in frame array
        t0 = self._times[0]
        idx_float = (time - t0) / self._time_step

        if idx_float < -0.5 or idx_float > self.n_frames - 0.5:
            return None

        if interpolation == "nearest":
            idx = int(round(idx_float))
            idx = max(0, min(self.n_frames - 1, idx))
            return float(self._values[idx])

        elif interpolation == "linear":
            idx = int(np.floor(idx_float))
            if idx < 0:
                return float(self._values[0])
            if idx >= self.n_frames - 1:
                return float(self._values[-1])
            frac = idx_float - idx
            return float(self._values[idx] * (1 - frac) + self._values[idx + 1] * frac)

        elif interpolation == "cubic":
            idx = int(np.floor(idx_float))
            frac = idx_float - idx

            # Get 4 surrounding points for cubic interpolation
            i0 = max(0, idx - 1)
            i1 = max(0, min(self.n_frames - 1, idx))
            i2 = max(0, min(self.n_frames - 1, idx + 1))
            i3 = min(self.n_frames - 1, idx + 2)

            y0, y1, y2, y3 = self._values[i0], self._values[i1], self._values[i2], self._values[i3]

            # Cubic interpolation (Catmull-Rom spline)
            t = frac
            t2 = t * t
            t3 = t2 * t

            result = 0.5 * (
                (2 * y1) +
                (-y0 + y2) * t +
                (2*y0 - 5*y1 + 4*y2 - y3) * t2 +
                (-y0 + 3*y1 - 3*y2 + y3) * t3
            )
            return float(result)

        else:
            raise ValueError(f"Unknown interpolation method: {interpolation}")


def _bessel_i0(x: float) -> float:
    """Modified Bessel function I0 by its power series sum ((x/2)^k / k!)^2.

    Standard numerical analysis; mirrors ``bessel_i0`` in
    ``rust/src/intensity.rs`` term for term.
    """
    half = 0.5 * x
    term = 1.0
    total = 1.0
    k = 1.0
    while True:
        r = half / k
        term *= r * r
        total += term
        if term < total * 1e-17:
            return total
        k += 1.0


def _kaiser_window(n: int, beta: float = 20.0) -> np.ndarray:
    """
    Kaiser window: w[i] = I0(beta * sqrt(1 - x^2)) / I0(beta), x in [-1, 1].

    Praat's manual names the intensity window "Kaiser-20" (beta = 20).
    """
    if n <= 1:
        return np.array([1.0])

    mid = (n - 1) / 2.0
    norm = _bessel_i0(beta)
    window = np.zeros(n)
    for i in range(n):
        x = (i - mid) / mid
        arg = np.sqrt(max(0.0, 1.0 - x * x))
        window[i] = _bessel_i0(beta * arg) / norm
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
    2. Weight with a Kaiser-20 window (after DC removal if enabled)
    3. Convert to dB

    Documented parameters:
    - Effective duration: 3.2 / min_pitch
    - Physical duration: 6.4 / min_pitch (2x effective), Kaiser-20 window
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
    dt = 1.0 / sample_rate

    # Default time step (documented)
    if time_step <= 0:
        time_step = 0.8 / min_pitch

    # Window (2026-11-02): the documented Kaiser-20 window over a physical
    # duration of 2x the documented effective duration (3.2 / min_pitch).
    # Earlier versions used a Gaussian (alpha 13.2) over 7.2 / min_pitch,
    # fitted against a grid that was 4 ms off; on the correct grid the
    # documented window is ~10x closer to Praat (median 0.002 dB, max
    # 0.2-0.3 dB on the fixtures resampled to 8/16/24 kHz, vs 2-7 dB).
    physical_window_duration = 6.4 / min_pitch

    # Number of samples in window
    window_samples = int(round(physical_window_duration * sample_rate))
    if window_samples % 2 == 0:
        window_samples += 1  # Ensure odd for symmetric window

    half_window_samples = window_samples // 2

    window = _kaiser_window(window_samples, beta=20.0)

    # Frame timing (black-box, 2026-11-02): frames are centred in the sound,
    # and their count is set by a span of 6.4 / min_pitch (not the 7.2 /
    # min_pitch physical window):
    #   n_frames = floor((duration - 6.4 / min_pitch) / time_step) + 1
    #   t1 = (duration - (n_frames - 1) * time_step) / 2
    # Matched Praat's frame count and first time on 300/300 random
    # durations, sample rates, minimum pitches and time steps.
    # The duration is taken as n_samples * (1 / sample_rate), not
    # n_samples / sample_rate: at exact frame boundaries the two round
    # differently, and only the former reproduced Praat's count (130/130
    # boundary cases).
    frame_span = 6.4 / min_pitch
    duration = len(samples) * dt
    n_frames = int(np.floor((duration - frame_span) / time_step)) + 1
    if n_frames < 1:
        n_frames = 1
    t1 = (duration - (n_frames - 1) * time_step) / 2.0

    # Compute intensity for each frame
    times = np.zeros(n_frames)
    values = np.zeros(n_frames)

    # Reference pressure squared: (2e-5 Pa)² = 4e-10
    p_ref = 4e-10

    for i in range(n_frames):
        t = t1 + i * time_step
        times[i] = t

        # Centre on the sample nearest t; sample i sits at (i + 0.5) / sr.
        center_sample = int(round(t * sample_rate - 0.5))

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
        # Digital silence is -inf. (Praat floors at -300 dB; we keep -inf,
        # which existing callers rely on.)
        if mean_square <= 0:
            intensity_db = -np.inf
        else:
            intensity_db = 10.0 * np.log10(mean_square / p_ref)

        values[i] = intensity_db

    return Intensity(times, values, time_step, min_pitch)
