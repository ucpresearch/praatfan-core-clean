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
            interpolation: Interpolation method

        Returns:
            HNR in dB, or None if outside range or unvoiced
        """
        raise NotImplementedError()


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
        Clamping may be needed for r near 0 or 1 (DP10).
    """
    # TODO: Determine clamping bounds via black-box testing (DP10)
    if r <= 1e-15:
        return -200.0  # Or some minimum value - determine via testing
    if r >= 1.0 - 1e-15:
        return 200.0   # Or some maximum value - determine via testing
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
    # Step 1: Compute pitch using AC method
    # Note: Pitch AC uses periods_per_window internally
    pitch = sound_to_pitch(
        sound,
        time_step=time_step,
        pitch_floor=min_pitch,
        pitch_ceiling=600.0,  # Standard ceiling
        method="ac"
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

    Args:
        sound: Sound object
        time_step: Time step in seconds
        min_pitch: Minimum pitch in Hz
        silence_threshold: Silence threshold (0-1)
        periods_per_window: Number of periods per window

    Returns:
        Harmonicity object
    """
    # Step 1: Compute pitch using CC method
    pitch = sound_to_pitch(
        sound,
        time_step=time_step,
        pitch_floor=min_pitch,
        pitch_ceiling=600.0,
        method="cc"
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
