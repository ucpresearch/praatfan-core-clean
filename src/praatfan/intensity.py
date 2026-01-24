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

Decision points to determine via testing:
- DP3: Kaiser β value (approximately 20)
- DP4: Physical vs effective window ratio (likely 2×)
- DP5: DC removal method (weighted vs unweighted mean)
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


def sound_to_intensity(
    sound: "Sound",
    min_pitch: float = 100.0,
    time_step: float = 0.0,
    subtract_mean: bool = True
) -> Intensity:
    """
    Compute intensity from sound.

    Args:
        sound: Sound object
        min_pitch: Minimum pitch in Hz (determines window size)
        time_step: Time step in seconds (0 = auto: 0.8 / min_pitch)
        subtract_mean: Whether to subtract DC before analysis

    Returns:
        Intensity object
    """
    raise NotImplementedError()
