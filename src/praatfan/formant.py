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
            interpolation: Interpolation method ("linear", "nearest")

        Returns:
            Formant frequency, or None if not present
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
            fp = self._frames[idx].get_formant(formant_number)
            if fp is None:
                return None
            value = fp.frequency

        elif interpolation == "linear":
            idx = int(np.floor(idx_float))
            frac = idx_float - idx

            i1 = max(0, min(self.n_frames - 1, idx))
            i2 = max(0, min(self.n_frames - 1, idx + 1))

            fp1 = self._frames[i1].get_formant(formant_number)
            fp2 = self._frames[i2].get_formant(formant_number)

            # Handle missing formants
            if fp1 is None and fp2 is None:
                return None
            elif fp1 is None:
                value = fp2.frequency
            elif fp2 is None:
                value = fp1.frequency
            else:
                value = fp1.frequency * (1 - frac) + fp2.frequency * frac
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation}")

        # Convert units if needed
        if unit.lower() == "hertz":
            return float(value)
        elif unit.lower() == "bark":
            # Bark scale conversion
            return float(7.0 * np.log(value / 650.0 + np.sqrt(1 + (value / 650.0) ** 2)))
        else:
            return float(value)

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
            interpolation: Interpolation method ("linear", "nearest")

        Returns:
            Bandwidth, or None if not present
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
            fp = self._frames[idx].get_formant(formant_number)
            if fp is None:
                return None
            return float(fp.bandwidth)

        elif interpolation == "linear":
            idx = int(np.floor(idx_float))
            frac = idx_float - idx

            i1 = max(0, min(self.n_frames - 1, idx))
            i2 = max(0, min(self.n_frames - 1, idx + 1))

            fp1 = self._frames[i1].get_formant(formant_number)
            fp2 = self._frames[i2].get_formant(formant_number)

            # Handle missing formants
            if fp1 is None and fp2 is None:
                return None
            elif fp1 is None:
                return float(fp2.bandwidth)
            elif fp2 is None:
                return float(fp1.bandwidth)
            else:
                return float(fp1.bandwidth * (1 - frac) + fp2.bandwidth * frac)
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation}")


def _gaussian_window(n: int) -> np.ndarray:
    """Generate Gaussian window for formant analysis.

    Subtracts edge value so window goes to zero at boundaries,
    matching standard Gaussian window practice.
    """
    if n <= 1:
        return np.array([1.0])

    alpha = 12.0
    mid = (n - 1) / 2.0
    i = np.arange(n)
    x = (i - mid) / mid
    w = np.exp(-alpha * x * x)
    edge = np.exp(-alpha)
    w = (w - edge) / (1.0 - edge)
    w = np.maximum(w, 0.0)
    return w


def _burg_lpc(samples: np.ndarray, order: int) -> np.ndarray:
    """
    Compute LPC coefficients using Burg's algorithm.

    Reference: Childers (1978), "Modern Spectrum Analysis", pp. 252-255

    Args:
        samples: Windowed signal samples
        order: LPC order (2 × number of formants)

    Returns:
        LPC coefficients a[0..order] where a[0] = 1.0
    """
    n = len(samples)
    if n <= order:
        a = np.zeros(order + 1)
        a[0] = 1.0  # Identity filter (pass-through)
        return a

    # Initialize
    a = np.zeros(order + 1)
    a[0] = 1.0

    # Forward and backward prediction errors
    ef = samples.copy()
    eb = samples.copy()

    for k in range(1, order + 1):
        # Compute reflection coefficient
        num = 0.0
        den = 0.0
        for i in range(k, n):
            num += ef[i] * eb[i - 1]
            den += ef[i] ** 2 + eb[i - 1] ** 2

        if den < 1e-30:
            break

        reflection = -2.0 * num / den

        # Update prediction errors
        ef_new = np.zeros(n)
        eb_new = np.zeros(n)
        for i in range(k, n):
            ef_new[i] = ef[i] + reflection * eb[i - 1]
            eb_new[i] = eb[i - 1] + reflection * ef[i]
        ef = ef_new
        eb = eb_new

        # Update LPC coefficients (Levinson recursion)
        a_new = np.zeros(order + 1)
        a_new[0] = 1.0
        for i in range(1, k):
            a_new[i] = a[i] + reflection * a[k - i]
        a_new[k] = reflection
        a = a_new

    return a


def _eval_polynomial(a: np.ndarray, z: complex) -> tuple:
    """
    Evaluate LPC polynomial and its derivative at z.

    The polynomial is: P(z) = z^p + a[1]*z^{p-1} + ... + a[p]

    Args:
        a: LPC coefficients (a[0] = 1.0)
        z: Point to evaluate at

    Returns:
        (P(z), P'(z)) - polynomial value and derivative
    """
    order = len(a) - 1
    if order < 1:
        return (1.0, 0.0)

    # Horner's method for polynomial evaluation
    # P(z) = z^p + a[1]*z^{p-1} + ... + a[p]
    # Rewrite as: P(z) = (...((z + a[1])*z + a[2])*z + ...)*z + a[p]
    p_val = 1.0  # coefficient of z^p
    dp_val = 0.0  # derivative

    for i in range(1, order + 1):
        # Update derivative: d/dz[P*z + c] = P + z*dP
        dp_val = p_val + z * dp_val
        # Update polynomial: P*z + c
        p_val = p_val * z + a[i]

    return (p_val, dp_val)


def _polish_root(a: np.ndarray, z: complex, max_iter: int = 10, tol: float = 1e-10) -> complex:
    """
    Polish a root using Newton-Raphson iteration.

    Reference: Numerical Recipes Ch. 9.5

    Args:
        a: LPC coefficients (a[0] = 1.0)
        z: Initial root estimate
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Refined root
    """
    for _ in range(max_iter):
        p_val, dp_val = _eval_polynomial(a, z)

        if abs(dp_val) < 1e-30:
            break

        delta = p_val / dp_val
        z = z - delta

        if abs(delta) < tol * abs(z):
            break

    return z


def _reflect_unstable_roots(roots: np.ndarray) -> np.ndarray:
    """
    Reflect unstable roots (|z| > 1) to inside the unit circle.

    For a root z with |z| > 1, the reflected root is 1/conj(z).
    This preserves the frequency (angle) while making the filter stable.

    Reference: Markel & Gray (1976)

    Args:
        roots: Complex roots of LPC polynomial

    Returns:
        Roots with unstable ones reflected inside unit circle
    """
    reflected = np.zeros_like(roots)

    for i, z in enumerate(roots):
        r = np.abs(z)
        if r > 1.0:
            # Reflect: z_new = 1 / conj(z) = conj(z) / |z|^2
            reflected[i] = np.conj(z) / (r * r)
        else:
            reflected[i] = z

    return reflected


def _lpc_roots(a: np.ndarray, polish: bool = True, reflect_unstable: bool = True) -> np.ndarray:
    """
    Find roots of LPC polynomial using companion matrix eigenvalues.

    Reference: Numerical Recipes Ch. 9.5

    The polynomial is: 1 + a[1]*z^{-1} + ... + a[p]*z^{-p}
    We find roots of: z^p + a[1]*z^{p-1} + ... + a[p]

    Args:
        a: LPC coefficients (a[0] = 1.0)
        polish: Whether to polish roots with Newton-Raphson
        reflect_unstable: Whether to reflect unstable roots inside unit circle

    Returns:
        Complex roots of the polynomial
    """
    order = len(a) - 1
    if order < 1:
        return np.array([])

    # Build companion matrix
    # For polynomial z^p + c1*z^{p-1} + ... + cp
    # Companion matrix has -coefficients in first row, 1s on subdiagonal
    companion = np.zeros((order, order))

    # First row: -a[1], -a[2], ..., -a[p]
    for i in range(order):
        companion[0, i] = -a[i + 1]

    # Subdiagonal: 1s
    for i in range(1, order):
        companion[i, i - 1] = 1.0

    # Compute eigenvalues
    roots = np.linalg.eigvals(companion)

    # Reflect unstable roots (|z| > 1) inside unit circle
    if reflect_unstable:
        roots = _reflect_unstable_roots(roots)

    # Polish roots with Newton-Raphson
    if polish:
        for i in range(len(roots)):
            roots[i] = _polish_root(a, roots[i])

    return roots


def _roots_to_formants(
    roots: np.ndarray,
    sample_rate: float,
    min_freq: float = 50.0,
    max_freq: float = 5450.0
) -> List[FormantPoint]:
    """
    Convert complex roots to formant frequencies and bandwidths.

    For a root z = r * exp(i*theta):
    - Frequency = theta * sample_rate / (2*pi)
    - Bandwidth = -ln(r) * sample_rate / pi

    Args:
        roots: Complex roots of LPC polynomial
        sample_rate: Sample rate in Hz (after resampling)
        min_freq: Minimum formant frequency
        max_freq: Maximum formant frequency

    Returns:
        List of FormantPoint objects, sorted by frequency
    """
    formants = []

    for root in roots:
        # Only consider roots in upper half-plane (positive frequency)
        if root.imag <= 0:
            continue

        r = np.abs(root)
        theta = np.angle(root)

        # Frequency from angle
        freq = theta * sample_rate / (2 * np.pi)

        # Bandwidth from radius
        if r > 0:
            bandwidth = -np.log(r) * sample_rate / np.pi
        else:
            bandwidth = float('inf')

        # Filter by frequency range
        if min_freq <= freq <= max_freq and bandwidth > 0:
            formants.append(FormantPoint(freq, bandwidth))

    # Sort by frequency
    formants.sort(key=lambda f: f.frequency)

    return formants


def _resample(samples: np.ndarray, old_rate: float, new_rate: float) -> np.ndarray:
    """
    Resample using FFT-based sinc interpolation (via scipy).

    Zero-pads the signal before FFT resampling to reduce edge artifacts
    from the FFT's circular convolution assumption. Without padding, the
    periodicity assumption creates ringing that degrades LPC analysis at
    low-energy frames. 4x padding reduces formant P95 error by ~40%.

    Args:
        samples: Input samples
        old_rate: Original sample rate
        new_rate: Target sample rate

    Returns:
        Resampled samples
    """
    if abs(old_rate - new_rate) < 1e-6:
        return samples.copy()

    from scipy import signal

    new_length = int(len(samples) * new_rate / old_rate)

    # Zero-pad to 5x length before FFT resample, then truncate.
    # This pushes the circular wrap-around point far from the signal,
    # reducing Gibbs-like ringing that corrupts LPC at low-energy frames.
    pad_factor = 5
    padded = np.zeros(len(samples) * pad_factor)
    padded[:len(samples)] = samples
    padded_new_length = new_length * pad_factor
    resampled = signal.resample(padded, padded_new_length)
    return resampled[:new_length]


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
       e. Convert roots to frequencies and bandwidths
       f. Filter and sort formants

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
    original_samples = sound.samples
    original_rate = sound.sample_rate
    duration = sound.duration

    # Step 1: Resample to 2 × max_formant_hz
    target_rate = 2.0 * max_formant_hz
    if target_rate < original_rate:
        samples = _resample(original_samples, original_rate, target_rate)
        sample_rate = target_rate
    else:
        samples = original_samples.copy()
        sample_rate = original_rate

    # Step 2: Pre-emphasis
    # x'[i] = x[i] - α × x[i-1]
    # α = exp(-2π × F × Δt)
    dt = 1.0 / sample_rate
    alpha = np.exp(-2 * np.pi * pre_emphasis_from * dt)
    pre_emphasized = np.zeros(len(samples))
    pre_emphasized[0] = samples[0]
    for i in range(1, len(samples)):
        pre_emphasized[i] = samples[i] - alpha * samples[i - 1]

    # Window: actual length is 2× the parameter value
    physical_window_duration = 2.0 * window_length
    window_samples = int(round(physical_window_duration * sample_rate))
    if window_samples % 2 == 0:
        window_samples += 1
    half_window = window_samples // 2

    # Time step: default is 25% of window length
    if time_step <= 0:
        time_step = window_length / 4.0

    # LPC order: 2 × number of formants
    lpc_order = 2 * max_num_formants

    # Generate Gaussian window
    window = _gaussian_window(window_samples)

    # Frame timing - centered
    n_frames = int(np.floor((duration - physical_window_duration) / time_step)) + 1
    if n_frames < 1:
        n_frames = 1
    t1 = (duration - (n_frames - 1) * time_step) / 2.0

    frames = []

    for i in range(n_frames):
        t = t1 + i * time_step

        # Extract frame in resampled signal
        center_sample = int(round(t * sample_rate))
        start_sample = center_sample - half_window
        end_sample = start_sample + window_samples

        # Handle boundaries
        if start_sample < 0 or end_sample > len(pre_emphasized):
            frame_samples = np.zeros(window_samples)
            src_start = max(0, start_sample)
            src_end = min(len(pre_emphasized), end_sample)
            dst_start = src_start - start_sample
            dst_end = dst_start + (src_end - src_start)
            frame_samples[dst_start:dst_end] = pre_emphasized[src_start:src_end]
        else:
            frame_samples = pre_emphasized[start_sample:end_sample].copy()

        # Apply window
        windowed = frame_samples * window

        # Compute LPC coefficients using Burg's algorithm
        lpc_coeffs = _burg_lpc(windowed, lpc_order)

        # Find roots of LPC polynomial
        roots = _lpc_roots(lpc_coeffs)

        # Convert roots to formant frequencies and bandwidths
        formant_points = _roots_to_formants(
            roots,
            sample_rate,
            min_freq=50.0,
            max_freq=max_formant_hz - 50.0
        )

        # Limit to max_num_formants
        formant_points = formant_points[:max_num_formants]

        frames.append(FormantFrame(t, formant_points))

    return Formant(frames, time_step, max_formant_hz, max_num_formants)
