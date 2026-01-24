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
            interpolation: Interpolation method ("linear", "nearest")

        Returns:
            Pitch value, or None if unvoiced or outside range
        """
        if self.n_frames == 0:
            return None

        # Find position in frame array
        t0 = self._frames[0].time
        idx_float = (time - t0) / self._time_step

        if idx_float < -0.5 or idx_float > self.n_frames - 0.5:
            return None

        if interpolation == "nearest":
            idx = int(round(idx_float))
            idx = max(0, min(self.n_frames - 1, idx))
            frame = self._frames[idx]
            if not frame.voiced:
                return None
            value = frame.frequency

        elif interpolation == "linear":
            idx = int(np.floor(idx_float))
            frac = idx_float - idx

            # Get surrounding frames
            i1 = max(0, min(self.n_frames - 1, idx))
            i2 = max(0, min(self.n_frames - 1, idx + 1))

            f1, f2 = self._frames[i1], self._frames[i2]

            # Both frames must be voiced for interpolation
            if not f1.voiced or not f2.voiced:
                # Fall back to nearest voiced
                if frac < 0.5 and f1.voiced:
                    value = f1.frequency
                elif f2.voiced:
                    value = f2.frequency
                elif f1.voiced:
                    value = f1.frequency
                else:
                    return None
            else:
                value = f1.frequency * (1 - frac) + f2.frequency * frac
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation}")

        # Convert units if needed
        if unit.lower() == "hertz":
            return float(value)
        elif unit.lower() == "semitones":
            # Semitones relative to 100 Hz
            return float(12.0 * np.log2(value / 100.0))
        elif unit.lower() == "mel":
            return float(1127.0 * np.log(1.0 + value / 700.0))
        elif unit.lower() == "erb":
            return float(21.4 * np.log10(0.00437 * value + 1.0))
        else:
            return float(value)

    def get_strength_at_time(self, time: float, interpolation: str = "linear") -> Optional[float]:
        """
        Get pitch strength at a specific time.

        This is the correlation value used to compute HNR.

        Args:
            time: Time in seconds
            interpolation: Interpolation method ("linear", "nearest")

        Returns:
            Strength value (0-1), or None if outside range
        """
        if self.n_frames == 0:
            return None

        # Find position in frame array
        t0 = self._frames[0].time
        idx_float = (time - t0) / self._time_step

        if idx_float < -0.5 or idx_float > self.n_frames - 0.5:
            return None

        if interpolation == "nearest":
            idx = int(round(idx_float))
            idx = max(0, min(self.n_frames - 1, idx))
            return float(self._frames[idx].strength)

        elif interpolation == "linear":
            idx = int(np.floor(idx_float))
            frac = idx_float - idx

            i1 = max(0, min(self.n_frames - 1, idx))
            i2 = max(0, min(self.n_frames - 1, idx + 1))

            s1 = self._frames[i1].strength
            s2 = self._frames[i2].strength

            return float(s1 * (1 - frac) + s2 * frac)
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation}")


def _hanning_window(n: int) -> np.ndarray:
    """Generate Hanning window."""
    if n <= 1:
        return np.array([1.0])
    i = np.arange(n)
    return 0.5 - 0.5 * np.cos(2 * np.pi * i / (n - 1))


def _gaussian_window(n: int, alpha: float = 12.0) -> np.ndarray:
    """
    Generate Gaussian window.

    From Boersma (1993) postscript:
        w(t) = (exp(-α × (t/T - 0.5)²) - exp(-α/4)) / (1 - exp(-α/4))
    """
    if n <= 1:
        return np.array([1.0])

    mid = (n - 1) / 2.0
    t = np.arange(n)
    x = (t - mid) / mid  # Range [-1, 1]

    exp_term = np.exp(-alpha * x * x)
    exp_edge = np.exp(-alpha)

    return (exp_term - exp_edge) / (1.0 - exp_edge)


def _hanning_window_autocorrelation(n: int, lag: int) -> float:
    """
    Compute autocorrelation of Hanning window at given lag.

    From Boersma (1993) Eq. 8:
    r_w(τ) = (1 - |τ|/T) × (2/3 + 1/3×cos(2πτ/T)) + (1/2π) × sin(2π|τ|/T)

    Args:
        n: Window length in samples
        lag: Lag in samples

    Returns:
        Autocorrelation value at the lag
    """
    if lag < 0:
        lag = -lag

    T = n - 1  # Window duration in samples
    if T <= 0 or lag >= n:
        return 0.0

    tau_norm = lag / T  # Normalized lag

    if tau_norm >= 1.0:
        return 0.0

    # Boersma (1993) Eq. 8
    r_w = ((1.0 - tau_norm) * (2.0/3.0 + (1.0/3.0) * np.cos(2 * np.pi * tau_norm)) +
           (1.0 / (2 * np.pi)) * np.sin(2 * np.pi * tau_norm))

    return r_w


def _compute_window_autocorrelation(window: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute autocorrelation of a window function numerically.

    Args:
        window: Window function array
        max_lag: Maximum lag to compute

    Returns:
        Array of autocorrelation values from lag 0 to max_lag
    """
    n = len(window)
    r = np.zeros(max_lag + 1)

    for lag in range(min(max_lag + 1, n)):
        r[lag] = np.sum(window[:n-lag] * window[lag:])

    return r


def _compute_autocorrelation(samples: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute autocorrelation for lags 0 to max_lag.

    Args:
        samples: Windowed samples
        max_lag: Maximum lag to compute

    Returns:
        Array of autocorrelation values
    """
    n = len(samples)
    r = np.zeros(max_lag + 1)

    for lag in range(max_lag + 1):
        if lag >= n:
            break
        r[lag] = np.sum(samples[:n-lag] * samples[lag:])

    return r


def _compute_cross_correlation(samples: np.ndarray, min_lag: int, max_lag: int) -> np.ndarray:
    """
    Compute full-frame cross-correlation for the CC pitch method.

    For each lag τ, computes the normalized correlation between:
    - samples[0 : n-τ]  (signal from start to n-τ)
    - samples[τ : n]    (signal shifted by τ samples)

    This is normalized by the geometric mean of the energies:
        r(τ) = Σ(x[i] × x[i+τ]) / sqrt(Σx[0:n-τ]² × Σx[τ:n]²)

    Args:
        samples: Raw (unwindowed) frame samples
        min_lag: Minimum lag to compute
        max_lag: Maximum lag to compute

    Returns:
        Array of normalized correlation values (0 to 1)
    """
    n = len(samples)
    r = np.zeros(max_lag + 1)

    for lag in range(min_lag, min(max_lag + 1, n)):
        x1 = samples[:n-lag]
        x2 = samples[lag:]

        corr = np.sum(x1 * x2)
        e1 = np.sum(x1 * x1)
        e2 = np.sum(x2 * x2)

        if e1 > 0 and e2 > 0:
            r[lag] = corr / np.sqrt(e1 * e2)

    return r


def _sinc_interpolate(r: np.ndarray, lag: float, depth: int = 70) -> float:
    """
    Windowed sinc interpolation for sub-sample precision.

    From Boersma (1993) Eq. 22:
    sinc_window(φ, N) = [sin(πφ) / (πφ)] × [0.5 + 0.5×cos(πφ/N)]

    Args:
        r: Autocorrelation array
        lag: Fractional lag to interpolate at
        depth: Number of samples to use on each side

    Returns:
        Interpolated value
    """
    n = len(r)
    result = 0.0

    lag_int = int(np.floor(lag))

    for i in range(max(0, lag_int - depth), min(n, lag_int + depth + 1)):
        phi = lag - i

        if abs(phi) < 1e-10:
            sinc = 1.0
        else:
            sinc = np.sin(np.pi * phi) / (np.pi * phi)

        # Cosine window
        if depth > 0:
            window = 0.5 + 0.5 * np.cos(np.pi * phi / depth)
        else:
            window = 1.0

        result += r[i] * sinc * window

    return result


def _find_cc_peaks(r: np.ndarray, min_lag: int, max_lag: int,
                   sample_rate: float, max_candidates: int = 15) -> List[tuple]:
    """
    Find peaks in cross-correlation.

    Unlike AC, CC values are already normalized (0-1) via energy normalization,
    so no window autocorrelation normalization is needed.

    Args:
        r: Cross-correlation values from _compute_cross_correlation
        min_lag: Minimum lag (1/ceiling in samples)
        max_lag: Maximum lag (1/floor in samples)
        sample_rate: Sample rate
        max_candidates: Maximum number of candidates

    Returns:
        List of (frequency, strength) tuples, sorted by strength.
    """
    candidates = []

    # Find all peaks
    for lag in range(min_lag, min(max_lag + 1, len(r) - 1)):
        if r[lag] > r[lag-1] and r[lag] > r[lag+1]:
            r_curr = r[lag]

            # Parabolic interpolation for sub-sample precision
            r_prev = r[lag-1]
            r_next = r[lag+1]

            denom = r_prev - 2*r_curr + r_next
            if abs(denom) > 1e-10:
                delta = 0.5 * (r_prev - r_next) / denom
                if abs(delta) < 1:
                    refined_lag = lag + delta
                    # Use raw peak strength (not interpolated) to avoid overshoot
                    freq = sample_rate / refined_lag
                    candidates.append((freq, r_curr))
                else:
                    candidates.append((sample_rate / lag, r_curr))
            else:
                candidates.append((sample_rate / lag, r_curr))

    # Sort by strength and return top candidates
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:max_candidates]


def _find_autocorrelation_peaks(r: np.ndarray, r_w: np.ndarray,
                                 min_lag: int, max_lag: int,
                                 sample_rate: float,
                                 max_candidates: int = 15,
                                 use_interpolation: bool = True) -> List[tuple]:
    """
    Find all autocorrelation peaks in the given lag range.

    Args:
        r: Raw autocorrelation of windowed signal
        r_w: Autocorrelation of window function
        min_lag: Minimum lag (1/ceiling in samples)
        max_lag: Maximum lag (1/floor in samples)
        sample_rate: Sample rate
        max_candidates: Maximum number of candidates
        use_interpolation: Use parabolic interpolation for frequency (default True)
            Note: For strength, we always use raw peak to avoid overshoot

    Returns list of (frequency, strength) tuples, sorted by strength.
    """
    if max_lag >= len(r) or max_lag >= len(r_w):
        return []

    r_0 = r[0]
    if r_0 <= 0:
        return []

    # Compute normalized autocorrelation array
    r_norm = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if r_w[lag] > 0 and r_w[0] > 0:
            r_norm[lag] = (r[lag] / r_0) / (r_w[lag] / r_w[0])
        else:
            r_norm[lag] = 0

    candidates = []

    # Find all peaks in normalized autocorrelation
    for lag in range(min_lag, min(max_lag + 1, len(r_norm) - 1)):
        if r_norm[lag] > r_norm[lag-1] and r_norm[lag] > r_norm[lag+1]:
            r_curr = r_norm[lag]

            if use_interpolation:
                # Parabolic interpolation for sub-sample precision on frequency only
                r_prev = r_norm[lag-1]
                r_next = r_norm[lag+1]

                denom = r_prev - 2*r_curr + r_next
                if abs(denom) > 1e-10:
                    delta = 0.5 * (r_prev - r_next) / denom
                    if abs(delta) < 1:
                        refined_lag = lag + delta
                        freq = sample_rate / refined_lag
                        # Use raw peak strength to avoid overshoot
                        candidates.append((freq, r_curr))
                    else:
                        candidates.append((sample_rate / lag, r_curr))
                else:
                    candidates.append((sample_rate / lag, r_curr))
            else:
                candidates.append((sample_rate / lag, r_curr))

    # Sort by strength and return top candidates
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:max_candidates]


def _find_autocorrelation_peak(r: np.ndarray, r_w: np.ndarray,
                                min_lag: int, max_lag: int,
                                sample_rate: float) -> tuple:
    """
    Find the best autocorrelation peak in the given lag range.

    From Boersma (1993) Eq. 9:
        r_x(τ) ≈ r_a(τ) / r_w(τ)

    This normalized autocorrelation should be between -1 and 1 for
    properly normalized signals.

    Args:
        r: Raw autocorrelation of windowed signal (r_a)
        r_w: Autocorrelation of window function
        min_lag: Minimum lag (1/ceiling in samples)
        max_lag: Maximum lag (1/floor in samples)
        sample_rate: Sample rate

    Returns:
        (frequency, strength) tuple, or (0, 0) if no valid peak
    """
    if max_lag >= len(r) or max_lag >= len(r_w):
        return (0.0, 0.0)

    # Normalize r[0] (energy normalization)
    r_0 = r[0]
    if r_0 <= 0:
        return (0.0, 0.0)

    # Compute normalized autocorrelation array
    r_norm = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if r_w[lag] > 0 and r_w[0] > 0:
            # Boersma (1993) Eq. 9: r_x(τ) = r_a(τ) / r_w(τ)
            # Normalized so r_x(0) = 1
            r_norm[lag] = (r[lag] / r_0) / (r_w[lag] / r_w[0])
        else:
            r_norm[lag] = 0

    best_freq = 0.0
    best_strength = 0.0

    # Find peaks in normalized autocorrelation
    for lag in range(min_lag, min(max_lag + 1, len(r_norm) - 1)):
        # Check if this is a local maximum
        if r_norm[lag] > r_norm[lag-1] and r_norm[lag] > r_norm[lag+1]:
            if r_norm[lag] > best_strength:
                # Parabolic interpolation for sub-sample precision
                r_prev = r_norm[lag-1]
                r_curr = r_norm[lag]
                r_next = r_norm[lag+1]

                denom = r_prev - 2*r_curr + r_next
                if abs(denom) > 1e-10:
                    delta = 0.5 * (r_prev - r_next) / denom
                    if abs(delta) < 1:
                        refined_lag = lag + delta
                        # Interpolate strength at refined position
                        refined_strength = r_curr - 0.25 * (r_prev - r_next) * delta
                        best_freq = sample_rate / refined_lag
                        best_strength = refined_strength
                    else:
                        best_freq = sample_rate / lag
                        best_strength = r_curr
                else:
                    best_freq = sample_rate / lag
                    best_strength = r_curr

    return (best_freq, best_strength)


def _viterbi_path(frames: List[PitchFrame], time_step: float,
                   octave_jump_cost: float = 0.35,
                   voiced_unvoiced_cost: float = 0.14) -> None:
    """
    Apply Viterbi algorithm to find optimal path through candidates.

    From Boersma (1993) Eq. 27, the transition cost is:
    - 0 if both unvoiced
    - voiced_unvoiced_cost if voicing changes
    - octave_jump_cost × |log₂(F1/F2)| if both voiced

    The costs are corrected for time step: multiply by 0.01 / time_step

    This modifies the frames in place, reordering candidates.

    Args:
        frames: List of PitchFrame objects
        time_step: Time step between frames
        octave_jump_cost: Cost per octave jump
        voiced_unvoiced_cost: Cost for voicing change
    """
    n_frames = len(frames)
    if n_frames <= 1:
        return

    # Time correction factor
    time_correction = 0.01 / time_step

    # Dynamic programming
    # best_cost[i][j] = best total cost to reach candidate j at frame i
    # best_prev[i][j] = previous candidate index that led to this best cost

    # Initialize first frame
    n_cands = [len(f.candidates) for f in frames]

    best_cost = [np.full(n_cands[i], np.inf) for i in range(n_frames)]
    best_prev = [np.zeros(n_cands[i], dtype=int) for i in range(n_frames)]

    # First frame: cost is negative of strength (we minimize cost, maximize strength)
    for j, cand in enumerate(frames[0].candidates):
        best_cost[0][j] = -cand.strength

    # Forward pass
    for i in range(1, n_frames):
        for j, cand_j in enumerate(frames[i].candidates):
            for k, cand_k in enumerate(frames[i-1].candidates):
                # Transition cost
                f_k = cand_k.frequency
                f_j = cand_j.frequency

                if f_k == 0 and f_j == 0:
                    # Both unvoiced
                    trans_cost = 0
                elif f_k == 0 or f_j == 0:
                    # Voicing change
                    trans_cost = voiced_unvoiced_cost
                else:
                    # Both voiced - octave jump cost
                    trans_cost = octave_jump_cost * abs(np.log2(f_j / f_k))

                trans_cost *= time_correction

                # Total cost to reach candidate j at frame i via candidate k at frame i-1
                total_cost = best_cost[i-1][k] + trans_cost - cand_j.strength

                if total_cost < best_cost[i][j]:
                    best_cost[i][j] = total_cost
                    best_prev[i][j] = k

    # Backward pass to find best path
    path = np.zeros(n_frames, dtype=int)
    path[-1] = np.argmin(best_cost[-1])

    for i in range(n_frames - 2, -1, -1):
        path[i] = best_prev[i+1][path[i+1]]

    # Reorder candidates in each frame so best path candidate is first
    for i in range(n_frames):
        best_idx = path[i]
        if best_idx > 0:
            cands = frames[i].candidates
            cands[0], cands[best_idx] = cands[best_idx], cands[0]


def sound_to_pitch(
    sound: "Sound",
    time_step: float = 0.0,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
    method: str = "ac",
    voicing_threshold: float = 0.45,
    silence_threshold: float = 0.03,
    octave_cost: float = 0.01,
    octave_jump_cost: float = 0.35,
    voiced_unvoiced_cost: float = 0.14,
    periods_per_window: float = 3.0,
    frame_timing: str = "centered",
    apply_octave_cost: bool = True
) -> Pitch:
    """
    Compute pitch from sound using autocorrelation or cross-correlation method.

    Based on Boersma (1993): "Accurate short-term analysis of the fundamental
    frequency and the harmonics-to-noise ratio of a sampled sound."

    Args:
        sound: Sound object
        time_step: Time step in seconds (0 = auto: 0.75/floor)
        pitch_floor: Minimum pitch in Hz
        pitch_ceiling: Maximum pitch in Hz
        method: "ac" for autocorrelation, "cc" for cross-correlation
        voicing_threshold: Threshold for voicing decision
        silence_threshold: Threshold for silence detection
        octave_cost: Cost for octave jumps
        octave_jump_cost: Cost for octave jumps between frames
        voiced_unvoiced_cost: Cost for voicing transitions
        periods_per_window: Number of pitch periods per analysis window
            (default 3.0 for AC, 2.0 for CC)
        frame_timing: Frame timing mode ("centered" for Pitch, "left" for Harmonicity)
        apply_octave_cost: Whether to apply octave cost to strength (default True).
            Set to False for Harmonicity to get raw correlation strength.

    Returns:
        Pitch object
    """
    if method not in ("ac", "cc"):
        raise NotImplementedError(f"Method '{method}' not implemented, only 'ac' and 'cc' supported")

    # CC uses 2 periods by default, AC uses 3
    if method == "cc" and periods_per_window == 3.0:
        periods_per_window = 2.0

    samples = sound.samples
    sample_rate = sound.sample_rate
    duration = sound.duration

    # Default time step (documented: 0.75 / floor)
    if time_step <= 0:
        time_step = 0.75 / pitch_floor

    # Window duration: periods_per_window periods of minimum pitch
    window_duration = periods_per_window / pitch_floor

    # Lag range for pitch search
    min_lag = int(np.ceil(sample_rate / pitch_ceiling))
    max_lag = int(np.floor(sample_rate / pitch_floor))

    # Number of samples in window
    window_samples = int(round(window_duration * sample_rate))
    if window_samples % 2 == 0:
        window_samples += 1
    half_window_samples = window_samples // 2

    # For AC method: generate window and compute its autocorrelation
    # For CC method: no windowing needed
    if method == "ac":
        window = _hanning_window(window_samples)
        r_w = _compute_window_autocorrelation(window, max_lag)
    else:
        window = None
        r_w = None

    # Frame timing
    if frame_timing == "left":
        # Left-aligned with centering: used for Harmonicity
        # Frames are constrained to [window_duration, duration - window_duration]
        # and centered within that valid range
        n_frames = int(np.floor((duration - 2 * window_duration) / time_step + 1e-9)) + 1
        if n_frames < 1:
            n_frames = 1
        remaining = duration - 2 * window_duration - (n_frames - 1) * time_step
        t1 = window_duration + remaining / 2.0
    else:
        # Centered: frames centered in signal, used for Pitch
        n_frames = int(np.floor((duration - window_duration) / time_step + 1e-9)) + 1
        if n_frames < 1:
            n_frames = 1
        t1 = (duration - (n_frames - 1) * time_step) / 2.0

    # Compute global peak for silence detection
    global_peak = np.max(np.abs(samples))

    # Process each frame
    frames = []

    for i in range(n_frames):
        t = t1 + i * time_step

        # Extract frame samples
        center_sample = int(round(t * sample_rate))
        start_sample = center_sample - half_window_samples
        end_sample = start_sample + window_samples

        # Handle boundaries
        if start_sample < 0 or end_sample > len(samples):
            frame_samples = np.zeros(window_samples)
            src_start = max(0, start_sample)
            src_end = min(len(samples), end_sample)
            dst_start = src_start - start_sample
            dst_end = dst_start + (src_end - src_start)
            frame_samples[dst_start:dst_end] = samples[src_start:src_end]
        else:
            frame_samples = samples[start_sample:end_sample].copy()

        # Compute local peak (for silence detection)
        local_peak = np.max(np.abs(frame_samples))
        local_intensity = local_peak / (global_peak + 1e-30)

        # Compute correlation and find peaks based on method
        if method == "ac":
            # AC: Apply window and compute autocorrelation with normalization
            windowed = frame_samples * window
            r = _compute_autocorrelation(windowed, max_lag)
            peaks = _find_autocorrelation_peaks(r, r_w, min_lag, max_lag, sample_rate)
        else:
            # CC: Full-frame cross-correlation on raw samples
            r = _compute_cross_correlation(frame_samples, min_lag, max_lag)
            peaks = _find_cc_peaks(r, min_lag, max_lag, sample_rate)

        # Create candidates list
        candidates = []

        # Unvoiced candidate
        # From Boersma (1993) Eq. 23
        unvoiced_strength = (voicing_threshold +
                           max(0, 2 - local_intensity/silence_threshold) *
                           (1 + voicing_threshold))
        candidates.append(PitchCandidate(0.0, unvoiced_strength))

        # Voiced candidates
        for freq, strength in peaks:
            if freq > 0 and strength > 0:
                if apply_octave_cost:
                    # Apply octave cost (Eq. 24) for pitch tracking
                    adjusted_strength = strength - octave_cost * np.log2(pitch_floor / freq + 1e-30)
                else:
                    # Use raw strength for harmonicity computation
                    adjusted_strength = strength
                candidates.append(PitchCandidate(freq, adjusted_strength))

        # Sort by strength (highest first)
        candidates.sort(key=lambda c: c.strength, reverse=True)

        frames.append(PitchFrame(t, candidates, local_intensity))

    # Apply Viterbi path finding to resolve octave errors
    _viterbi_path(frames, time_step, octave_jump_cost, voiced_unvoiced_cost)

    return Pitch(frames, time_step, pitch_floor, pitch_ceiling)
