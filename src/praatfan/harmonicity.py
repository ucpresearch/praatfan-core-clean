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
        apply_octave_cost=False,
        apply_intensity_adjustment=False
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

    This function:
    1. Computes Pitch using CC method
    2. Extracts strength values from Pitch
    3. Applies HNR formula to get dB values

    The window duration is (periods_per_window + 1) / min_pitch to account
    for the "forward" cross-correlation needing an extra period.

    Args:
        sound: Sound object
        time_step: Time step in seconds
        min_pitch: Minimum pitch in Hz
        silence_threshold: Silence threshold (0-1)
        periods_per_window: Number of periods per window (Praat default: 1.0)

    Returns:
        Harmonicity object
    """
    # Step 1: Compute pitch using CC method
    # HNR CC window = (ppw + 1) / min_pitch for "forward" cross-correlation
    # Disable octave cost to get raw correlation strength for HNR
    pitch = sound_to_pitch(
        sound,
        time_step=time_step,
        pitch_floor=min_pitch,
        pitch_ceiling=600.0,  # Standard ceiling
        method="cc",
        periods_per_window=periods_per_window + 1.0,  # +1 for forward CC
        frame_timing="centered",
        apply_octave_cost=False,
        apply_intensity_adjustment=False
    )

    # Step 2: Extract times and convert strengths to HNR
    times = pitch.times()
    hnr_values = []

    for frame in pitch.frames:
        if frame.voiced:
            # Only use strength if it's a valid correlation (0 < r < 1)
            strength = frame.strength
            if 0 < strength < 1:
                hnr = strength_to_hnr(strength)
            else:
                hnr = -200.0
        else:
            hnr = -200.0
        hnr_values.append(hnr)

    return Harmonicity(
        times=times,
        values=np.array(hnr_values),
        time_step=time_step,
        min_pitch=min_pitch
    )
