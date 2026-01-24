"""
Pitch - Fundamental frequency (F0) contour.

Implementation order: Phase 4
Dependencies: None (foundation for Harmonicity)

Documentation sources:
- Boersma (1993): "Accurate short-term analysis of the fundamental frequency
  and the harmonics-to-noise ratio of a sampled sound"
- Praat manual: Sound: To Pitch...

Key documented facts (from Boersma 1993):
- Autocorrelation normalization: r_x(τ) ≈ r_a(τ) / r_w(τ) (Eq. 9)
- Sinc interpolation formula (Eq. 22)
- Candidate strength formulas (Eq. 23, 24)
- Viterbi transition costs (Eq. 27)
- Gaussian window formula (postscript)

Decision points:
- DP1: Frame timing (t1)
- DP2: Gaussian window coefficient
- DP4: Physical vs effective window ratio
- DP15: Sinc interpolation depth
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PitchCandidate:
    """A pitch candidate for a frame."""
    frequency: float  # Hz (0 = unvoiced)
    strength: float   # Correlation strength (0-1)


@dataclass
class PitchFrame:
    """Pitch analysis results for a single frame."""
    time: float                      # Time in seconds
    candidates: List[PitchCandidate] # Candidates (first is selected)
    intensity: float                 # Local intensity (0-1)

    @property
    def frequency(self) -> float:
        """Selected pitch frequency (0 if unvoiced)."""
        if self.candidates:
            return self.candidates[0].frequency
        return 0.0

    @property
    def strength(self) -> float:
        """Selected pitch strength."""
        if self.candidates:
            return self.candidates[0].strength
        return 0.0

    @property
    def voiced(self) -> bool:
        """Whether this frame is voiced."""
        return self.frequency > 0.0


class Pitch:
    """
    Pitch (F0) contour.

    Attributes:
        frames: List of PitchFrame objects
        time_step: Time step between frames
        pitch_floor: Minimum pitch in Hz
        pitch_ceiling: Maximum pitch in Hz
    """

    def __init__(
        self,
        frames: List[PitchFrame],
        time_step: float,
        pitch_floor: float,
        pitch_ceiling: float
    ):
        """
        Create a Pitch object.

        Args:
            frames: List of PitchFrame objects
            time_step: Time step between frames
            pitch_floor: Minimum pitch in Hz
            pitch_ceiling: Maximum pitch in Hz
        """
        self._frames = frames
        self._time_step = time_step
        self._pitch_floor = pitch_floor
        self._pitch_ceiling = pitch_ceiling

    @property
    def frames(self) -> List[PitchFrame]:
        """List of pitch frames."""
        return self._frames

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return len(self._frames)

    @property
    def time_step(self) -> float:
        """Time step between frames."""
        return self._time_step

    @property
    def pitch_floor(self) -> float:
        """Minimum pitch in Hz."""
        return self._pitch_floor

    @property
    def pitch_ceiling(self) -> float:
        """Maximum pitch in Hz."""
        return self._pitch_ceiling

    def times(self) -> np.ndarray:
        """Get array of frame times."""
        return np.array([f.time for f in self._frames])

    def values(self) -> np.ndarray:
        """Get array of pitch values (0 for unvoiced)."""
        return np.array([f.frequency for f in self._frames])

    def strengths(self) -> np.ndarray:
        """Get array of pitch strengths."""
        return np.array([f.strength for f in self._frames])

    def get_value_at_time(
        self,
        time: float,
        unit: str = "Hertz",
        interpolation: str = "linear"
    ) -> Optional[float]:
        """
        Get pitch value at a specific time.

        Args:
            time: Time in seconds
            unit: Unit for result ("Hertz", "semitones", etc.)
            interpolation: Interpolation method

        Returns:
            Pitch value, or None if unvoiced
        """
        raise NotImplementedError()

    def get_strength_at_time(self, time: float) -> Optional[float]:
        """
        Get pitch strength at a specific time.

        This is the correlation value used to compute HNR.

        Args:
            time: Time in seconds

        Returns:
            Strength value (0-1), or None if outside range
        """
        raise NotImplementedError()


def sound_to_pitch(
    sound: "Sound",
    time_step: float = 0.0,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
    method: str = "ac"
) -> Pitch:
    """
    Compute pitch from sound.

    Args:
        sound: Sound object
        time_step: Time step in seconds (0 = auto)
        pitch_floor: Minimum pitch in Hz
        pitch_ceiling: Maximum pitch in Hz
        method: "ac" for autocorrelation, "cc" for cross-correlation

    Returns:
        Pitch object
    """
    raise NotImplementedError()
