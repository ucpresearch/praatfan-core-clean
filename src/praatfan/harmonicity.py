"""
Harmonicity - Harmonics-to-noise ratio (HNR) contour.

Implementation order: Phase 5 (AFTER Pitch is validated)
Dependencies: Pitch (Harmonicity = trivial formula applied to Pitch strength)

CRITICAL: Harmonicity is NOT a standalone algorithm.
It uses Pitch internally and extracts HNR from the correlation strength.

Documentation sources:
- Praat manual: Harmonicity
  "if 99% of the energy of the signal is in the periodic part, and 1% is noise,
   the HNR is 10*log10(99/1) = 20 dB"
- Praat manual: Sound: To Harmonicity (ac)...
  "The algorithm performs an acoustic periodicity detection on the basis of
   an accurate autocorrelation method, as described in Boersma (1993)."

HNR Formula (from Praat manual):
    HNR (dB) = 10 × log₁₀(r / (1 - r))

where r is the normalized autocorrelation (pitch strength) at the pitch period.

Decision points:
- DP10: HNR clamping bounds (r → 0 or r → 1)
- DP11: Value for unvoiced frames
"""

import numpy as np
from typing import Optional
from .pitch import Pitch, sound_to_pitch


class Harmonicity:
    """
    Harmonicity (HNR) contour.

    Values are in dB. Higher values indicate more periodic (harmonic) signal.
    Typical values for speech: 10-20 dB for vowels.

    Attributes:
        times: Time points in seconds
        values: HNR values in dB (None or very negative for unvoiced)
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
        Create a Harmonicity object.

        Args:
            times: Time points in seconds
            values: HNR values in dB
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
        """HNR values in dB."""
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
        Get HNR value at a specific time.

        Args:
            time: Time in seconds
            interpolation: Interpolation method ("cubic", "linear", "nearest")

        Returns:
            HNR in dB, or None if outside range
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


def strength_to_hnr(r: float) -> float:
    """
    Convert pitch strength (correlation) to HNR in dB.

    Formula (from Praat manual):
        HNR = 10 × log₁₀(r / (1 - r))

    Args:
        r: Pitch strength (normalized autocorrelation, 0-1)

    Returns:
        HNR in dB

    Note:
        - r = 0.5 → HNR = 0 dB (equal harmonic and noise energy)
        - r = 0.99 → HNR = 20 dB
        - r = 0.999 → HNR = 30 dB
        Clamping is needed for r near 0 or 1.
    """
    # Clamp r to valid range [epsilon, 1-epsilon]
    # The autocorrelation normalization can produce values slightly > 1
    # due to parabolic interpolation and approximations
    r = max(1e-10, min(r, 1.0 - 1e-10))
    return 10.0 * np.log10(r / (1.0 - r))


def sound_to_harmonicity_ac(
    sound: "Sound",
    time_step: float = 0.01,
    min_pitch: float = 75.0,
    silence_threshold: float = 0.1,
    periods_per_window: float = 4.5
) -> Harmonicity:
    """
    Compute harmonicity using autocorrelation method.

    This function:
    1. Computes Pitch using AC method
    2. Extracts strength values from Pitch
    3. Applies HNR formula to get dB values

    Args:
        sound: Sound object
        time_step: Time step in seconds
        min_pitch: Minimum pitch in Hz
        silence_threshold: Silence threshold (0-1)
        periods_per_window: Number of periods per window

    Returns:
        Harmonicity object
    """
    # Step 1: Compute pitch using AC method with harmonicity windowing
    # Harmonicity uses periods_per_window (default 4.5) for longer windows
    # and left-aligned frame timing
    # Disable octave cost to get raw correlation strength for HNR
    pitch = sound_to_pitch(
        sound,
        time_step=time_step,
        pitch_floor=min_pitch,
        pitch_ceiling=600.0,  # Standard ceiling
        method="ac",
        periods_per_window=periods_per_window,
        frame_timing="left",
        apply_octave_cost=False
    )

    # Step 2: Extract times and convert strengths to HNR
    times = pitch.times()
    hnr_values = []

    for frame in pitch.frames:
        if frame.voiced:
            hnr = strength_to_hnr(frame.strength)
        else:
            hnr = -200.0  # Unvoiced value - determine via testing (DP11)
        hnr_values.append(hnr)

    return Harmonicity(
        times=times,
        values=np.array(hnr_values),
        time_step=time_step,
        min_pitch=min_pitch
    )


def sound_to_harmonicity_cc(
    sound: "Sound",
    time_step: float = 0.01,
    min_pitch: float = 75.0,
    silence_threshold: float = 0.1,
    periods_per_window: float = 1.0
) -> Harmonicity:
    """
    Compute harmonicity using cross-correlation method.

    This function computes HNR directly from cross-correlation for all frames,
    not just pitch-voiced frames. Uses the same frame timing as Pitch CC
    (2-period window).

    Args:
        sound: Sound object
        time_step: Time step in seconds
        min_pitch: Minimum pitch in Hz
        silence_threshold: Silence threshold (0-1)
        periods_per_window: Number of periods per window (Praat default: 1.0)

    Returns:
        Harmonicity object
    """
    samples = sound.samples
    sample_rate = sound.sample_rate
    duration = sound.duration

    max_pitch = 600.0

    # Frame timing matches Pitch CC (2-period window)
    window_duration = 2.0 / min_pitch
    window_samples = int(round(window_duration * sample_rate))
    if window_samples % 2 == 0:
        window_samples += 1
    half_window = window_samples // 2

    min_lag = int(np.ceil(sample_rate / max_pitch))
    max_lag = int(np.floor(sample_rate / min_pitch))

    # Centered frame timing (same as Pitch CC)
    n_frames = int(np.floor((duration - window_duration) / time_step + 1e-9)) + 1
    if n_frames < 1:
        n_frames = 1
    t1 = (duration - (n_frames - 1) * time_step) / 2.0

    global_peak = np.max(np.abs(samples))

    times = []
    hnr_values = []

    for i in range(n_frames):
        t = t1 + i * time_step
        times.append(t)

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

        # Check for silence
        local_peak = np.max(np.abs(frame))
        local_intensity = local_peak / (global_peak + 1e-30)

        if local_intensity < silence_threshold * 0.01:
            hnr_values.append(-200.0)
            continue

        # Compute full-frame cross-correlation
        n = len(frame)
        best_r = 0.0
        best_lag = 0

        for lag in range(min_lag, min(max_lag + 1, n - 1)):
            x1 = frame[:n-lag]
            x2 = frame[lag:]
            corr = np.sum(x1 * x2)
            e1 = np.sum(x1 * x1)
            e2 = np.sum(x2 * x2)

            if e1 > 0 and e2 > 0:
                r = corr / np.sqrt(e1 * e2)
                # Check if this is a peak
                if lag > min_lag:
                    # We need to check if it's a local max
                    # For simplicity, track the best correlation
                    if r > best_r:
                        best_r = r
                        best_lag = lag

        # Find actual peaks and apply parabolic interpolation
        r_array = np.zeros(max_lag + 1)
        for lag in range(min_lag, min(max_lag + 1, n)):
            x1 = frame[:n-lag]
            x2 = frame[lag:]
            corr = np.sum(x1 * x2)
            e1 = np.sum(x1 * x1)
            e2 = np.sum(x2 * x2)
            if e1 > 0 and e2 > 0:
                r_array[lag] = corr / np.sqrt(e1 * e2)

        best_r = 0.0
        for lag in range(min_lag + 1, min(max_lag, len(r_array) - 1)):
            if r_array[lag] > r_array[lag-1] and r_array[lag] > r_array[lag+1]:
                # Parabolic interpolation for refined strength
                r_prev = r_array[lag-1]
                r_curr = r_array[lag]
                r_next = r_array[lag+1]

                denom = r_prev - 2*r_curr + r_next
                if abs(denom) > 1e-10:
                    delta = 0.5 * (r_prev - r_next) / denom
                    if abs(delta) < 1:
                        refined_r = r_curr - 0.25 * (r_prev - r_next) * delta
                        if refined_r > best_r:
                            best_r = refined_r
                    elif r_curr > best_r:
                        best_r = r_curr
                elif r_curr > best_r:
                    best_r = r_curr

        if best_r > 0:
            hnr = strength_to_hnr(best_r)
        else:
            hnr = -200.0

        hnr_values.append(hnr)

    return Harmonicity(
        times=np.array(times),
        values=np.array(hnr_values),
        time_step=time_step,
        min_pitch=min_pitch
    )
