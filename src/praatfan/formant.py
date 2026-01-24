"""
Formant - LPC-based formant frequency tracks.

Implementation order: Phase 6 (complex, can be parallel with Pitch)
Dependencies: None (independent)

Documentation sources:
- Praat manual: Sound: To Formant (burg)...
- Childers (1978): "Modern Spectrum Analysis", pp. 252-255 (Burg's algorithm)
- Numerical Recipes Ch. 9.5 (root finding via companion matrix)
- Markel & Gray (1976): root-to-formant conversion

Key documented facts:
- Window length parameter: "actual length is twice this value"
- Resample to 2 × max_formant_hz before analysis
- Pre-emphasis: x'[i] = x[i] - α × x[i-1], α = exp(-2π × F × Δt)
- LPC order: 2 × max_formants
- Formant filtering: remove < 50 Hz and > (max_formant - 50) Hz

Decision points:
- DP1: Frame timing (t1)
- DP2: Gaussian window coefficient
- DP6: LPC polynomial sign convention
- DP7: Unstable root reflection formula
- DP8: Root polishing iterations
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FormantPoint:
    """A single formant at a point in time."""
    frequency: float   # Hz
    bandwidth: float   # Hz


@dataclass
class FormantFrame:
    """Formant analysis results for a single frame."""
    time: float                    # Time in seconds
    formants: List[FormantPoint]   # List of formants (F1, F2, ...)

    @property
    def n_formants(self) -> int:
        """Number of formants in this frame."""
        return len(self.formants)

    def get_formant(self, n: int) -> Optional[FormantPoint]:
        """Get formant n (1-based index)."""
        if 1 <= n <= len(self.formants):
            return self.formants[n - 1]
        return None


class Formant:
    """
    Formant tracks over time.

    Attributes:
        frames: List of FormantFrame objects
        time_step: Time step between frames
        max_formant_hz: Maximum formant frequency
    """

    def __init__(
        self,
        frames: List[FormantFrame],
        time_step: float,
        max_formant_hz: float,
        max_num_formants: int
    ):
        """
        Create a Formant object.

        Args:
            frames: List of FormantFrame objects
            time_step: Time step between frames
            max_formant_hz: Maximum formant frequency
            max_num_formants: Maximum number of formants per frame
        """
        self._frames = frames
        self._time_step = time_step
        self._max_formant_hz = max_formant_hz
        self._max_num_formants = max_num_formants

    @property
    def frames(self) -> List[FormantFrame]:
        """List of formant frames."""
        return self._frames

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return len(self._frames)

    @property
    def time_step(self) -> float:
        """Time step between frames."""
        return self._time_step

    def times(self) -> np.ndarray:
        """Get array of frame times."""
        return np.array([f.time for f in self._frames])

    def formant_values(self, formant_number: int) -> np.ndarray:
        """
        Get array of formant frequencies for a specific formant.

        Args:
            formant_number: Formant number (1 = F1, 2 = F2, etc.)

        Returns:
            Array of frequencies (NaN where formant not present)
        """
        values = []
        for frame in self._frames:
            fp = frame.get_formant(formant_number)
            values.append(fp.frequency if fp else np.nan)
        return np.array(values)

    def bandwidth_values(self, formant_number: int) -> np.ndarray:
        """
        Get array of bandwidths for a specific formant.

        Args:
            formant_number: Formant number (1 = B1, 2 = B2, etc.)

        Returns:
            Array of bandwidths (NaN where formant not present)
        """
        values = []
        for frame in self._frames:
            fp = frame.get_formant(formant_number)
            values.append(fp.bandwidth if fp else np.nan)
        return np.array(values)

    def get_value_at_time(
        self,
        formant_number: int,
        time: float,
        unit: str = "Hertz",
        interpolation: str = "linear"
    ) -> Optional[float]:
        """
        Get formant frequency at a specific time.

        Args:
            formant_number: Formant number (1-based)
            time: Time in seconds
            unit: Unit for result ("Hertz", "Bark")
            interpolation: Interpolation method

        Returns:
            Formant frequency, or None if not present
        """
        raise NotImplementedError()

    def get_bandwidth_at_time(
        self,
        formant_number: int,
        time: float,
        unit: str = "Hertz",
        interpolation: str = "linear"
    ) -> Optional[float]:
        """
        Get bandwidth at a specific time.

        Args:
            formant_number: Formant number (1-based)
            time: Time in seconds
            unit: Unit for result
            interpolation: Interpolation method

        Returns:
            Bandwidth, or None if not present
        """
        raise NotImplementedError()


def sound_to_formant_burg(
    sound: "Sound",
    time_step: float = 0.0,
    max_num_formants: int = 5,
    max_formant_hz: float = 5500.0,
    window_length: float = 0.025,
    pre_emphasis_from: float = 50.0
) -> Formant:
    """
    Compute formants using Burg's LPC method.

    Algorithm steps:
    1. Resample to 2 × max_formant_hz
    2. Pre-emphasize
    3. For each frame:
       a. Extract windowed samples
       b. Apply Gaussian window
       c. Compute LPC coefficients using Burg's algorithm
       d. Find polynomial roots via companion matrix eigenvalues
       e. Polish roots with Newton-Raphson (optional)
       f. Convert roots to frequencies and bandwidths
       g. Filter and sort formants

    Args:
        sound: Sound object
        time_step: Time step in seconds (0 = auto: 25% of window)
        max_num_formants: Maximum number of formants to find
        max_formant_hz: Maximum formant frequency in Hz
        window_length: Window length in seconds (actual = 2× this value)
        pre_emphasis_from: Pre-emphasis from frequency in Hz

    Returns:
        Formant object
    """
    raise NotImplementedError()
