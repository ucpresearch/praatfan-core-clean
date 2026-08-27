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
- DP10: HNR boundary representation (r <= 0 and r >= 1) — RESOLVED, see
  strength_to_hnr: r <= 0 -> UNDEFINED_HNR (-200 dB), r >= 1 -> nan.
- DP11: Value for unvoiced frames — RESOLVED: UNDEFINED_HNR (-200 dB),
  confirmed black-box (Praat reports exactly -200 for silence and for DC).
- DP34: Minimum periods_per_window for the AC method — RESOLVED, see
  AC_MIN_PERIODS_PER_WINDOW (3.0).
"""

import inspect
import os
import warnings

import numpy as np
from typing import Optional
from .pitch import Pitch, sound_to_pitch


#: Value reported for frames in which no periodicity was found.
#:
#: Determined by black-box observation: Praat reports exactly -200 dB for
#: pure silence and for a constant (DC) signal.
UNDEFINED_HNR = -200.0

#: Resolution limit on the correlation, used as the upper boundary test.
#:
#: A correlation within one ulp of 1 carries no information about how close to
#: 1 it is, so the HNR derived from it would be pure rounding. See
#: ``strength_to_hnr``.
EPS = float(np.finfo(np.float64).eps)

#: Smallest ``periods_per_window`` for which the AC estimator is well posed.
#:
#: The AC method normalises the signal autocorrelation by the analysis
#: window's own autocorrelation, ``r_norm(tau) = (r_x(tau)/r_x(0)) /
#: (r_w(tau)/r_w(0))``, and searches lags up to ``sample_rate/pitch_floor``.
#: The window is ``periods_per_window/pitch_floor`` long, so the longest lag
#: searched sits at ``1/periods_per_window`` of the window length. As that
#: ratio approaches 1 the denominator ``r_w(tau)`` decays to zero and the
#: quotient is dominated by division by a vanishing number.
#:
#: Measured on a Hanning window at 16 kHz, floor 60 Hz -- ``r_w(tau_max) /
#: r_w(0)`` is 0.472 at 3.0 periods per window, 0.167 at 2.0, and -4.2e-17
#: (i.e. zero to within rounding, and of arbitrary sign) at 1.0. The maximum
#: correlation observed on conversational speech follows directly: 1.13 at
#: 3.0 periods per window, 1.76 at 2.0, and 35.9 at 1.0 -- the last two being
#: impossible for a normalised correlation.
#:
#: Praat imposes the same lower bound, which is observable from outside as a
#: user-visible refusal ("Number of periods per window must be at least
#: 3.0"); note that parselmouth's ``Sound.to_harmonicity_ac()`` wrapper
#: defaults to 1.0 and bypasses that check.
AC_MIN_PERIODS_PER_WINDOW = 3.0


def _is_marker(value: float) -> bool:
    """Whether a contour entry is a marker rather than a measurement."""
    return bool(np.isnan(value)) or value == UNDEFINED_HNR


class Harmonicity:
    """
    Harmonicity (HNR) contour.

    Values are in dB. Higher values indicate more periodic (harmonic) signal.
    Typical values for speech: 10-20 dB for vowels.

    Attributes:
        times: Time points in seconds
        values: HNR values in dB; -200 where no periodicity was found,
            nan where the correlation is unresolvable from 1
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
            HNR in dB, or None if outside range.

        Undefined frames:
            A contour carries two markers that are not measurements --
            ``UNDEFINED_HNR`` (-200 dB, no periodicity) and ``nan``
            (correlation not resolvable from 1). Both are excluded from the
            interpolation stencil rather than mixed into it: a single nan
            frame would otherwise poison every query within +/-2 frames, and a
            -200 frame would drag its neighbours 200 dB downward. If every
            frame in the stencil is a marker, that marker is returned
            verbatim; ``interpolation="nearest"`` always returns the raw frame
            value.
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
            return self._linear_skipping_markers(idx, idx + 1, frac)

        elif interpolation == "cubic":
            idx = int(np.floor(idx_float))
            frac = idx_float - idx

            # Get 4 surrounding points for cubic interpolation
            i0 = max(0, idx - 1)
            i1 = max(0, min(self.n_frames - 1, idx))
            i2 = max(0, min(self.n_frames - 1, idx + 1))
            i3 = min(self.n_frames - 1, idx + 2)

            y0, y1, y2, y3 = (self._values[i0], self._values[i1],
                              self._values[i2], self._values[i3])

            if any(_is_marker(y) for y in (y0, y1, y2, y3)):
                # Catmull-Rom weights are negative at the outer taps, so
                # dropping frames and renormalising the survivors lets the
                # remaining weights cancel to nearly zero and the quotient
                # explode. Degrade to the two central frames instead: their
                # weights are non-negative and sum to 1, so the result stays
                # bounded and predictable.
                return self._linear_skipping_markers(i1, i2, frac)

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

    def _linear_skipping_markers(self, i_lo: int, i_hi: int,
                                 frac: float) -> float:
        """
        Blend two frames linearly, unless either carries no measurement.

        Interpolating across a marker would let it leak into a real reading,
        so when one is present the nearest frame is returned verbatim instead.
        """
        v_lo = self._values[i_lo]
        v_hi = self._values[i_hi]
        if not _is_marker(v_lo) and not _is_marker(v_hi):
            return float(v_lo * (1.0 - frac) + v_hi * frac)
        # A marker is present, so there is nothing to interpolate along. Fall
        # back to the frame the query actually sits nearest: that returns the
        # real reading when the query leans towards it, and returns the marker
        # when it leans towards the marker -- rather than silently borrowing a
        # neighbour's value for a frame that holds no measurement.
        nearest = i_lo if frac < 0.5 else i_hi
        return float(self._values[nearest])


def strength_to_hnr(r: float) -> float:
    """
    Convert pitch strength (correlation) to HNR in dB.

    Formula (Praat manual, Harmonicity):
        HNR = 10 * log10(r / (1 - r))

    Args:
        r: Normalized autocorrelation at the pitch period.

    Returns:
        HNR in dB; ``UNDEFINED_HNR`` (-200) when there is no periodicity to
        measure; ``nan`` when the correlation is not a valid one.

    Boundary policy (see DP10):
        ``r`` is a normalized correlation, so a meaningful HNR exists only
        for ``0 < r < 1``. The two boundaries are reported differently
        because they mean opposite things:

        * ``r <= 0`` -- no periodic component was found. Reported as
          ``UNDEFINED_HNR`` (-200 dB), which is also what Praat reports for
          silence and for a DC signal (black-box observation).
        * ``1 - r <= EPS`` -- the correlation is not resolvable from 1 in
          double precision, so ``10*log10(r/(1-r))`` is determined entirely by
          rounding. This includes ``r > 1``, which no true correlation can
          have; it arises from estimator noise on (near-)perfectly periodic
          material and grows without bound once ``periods_per_window`` drops
          below ``AC_MIN_PERIODS_PER_WINDOW``. Reported as ``nan``: no HNR can
          be derived from a correlation that carries no information, so none
          is invented.

        The test is ``1 - r <= EPS`` rather than ``r >= 1`` so that the two
        backends agree. Under the strict test the last representable double
        below 1 still returns a finite 159.5459 dB, and ``1 - EPS`` (two ulps
        below 1) returns 156.5356 dB, while a one-ULP-larger ``r`` returns nan
        -- the pure-Python and Rust correlations differ at exactly that scale,
        so the strict test made them disagree on 59 frames of a 30 s excerpt.

        Earlier releases clamped ``r`` to ``[1e-10, 1-1e-10]`` instead, which
        collapsed every such frame onto the single value 99.99999964022885 dB
        and made an arithmetic boundary look like a measurement of
        near-perfect periodicity. Both boundaries are now distinguishable
        from any real reading.

    Note:
        - r = 0.5 -> HNR = 0 dB (equal harmonic and noise energy)
        - r = 0.99 -> HNR = 20 dB
        - r = 0.999 -> HNR = 30 dB
    """
    if not (r > 0.0):          # also catches nan
        return UNDEFINED_HNR
    if 1.0 - r <= EPS:
        return float('nan')
    return 10.0 * np.log10(r / (1.0 - r))


def _caller_stacklevel() -> int:
    """
    Stack depth of the first frame outside this package.

    A fixed ``stacklevel`` cannot be right for every entry point: the same
    warning is reachable from the module function, ``Sound.to_harmonicity_ac``,
    the ``call()`` compatibility layer and the backend selector, each at a
    different depth. Attributing the warning to a praatfan file rather than the
    user's has a functional cost, not just a cosmetic one — the default warning
    filter dedupes on the attributed (module, lineno), so every ill-conditioned
    call in a process would collapse into one visible warning however many
    distinct call sites triggered it.
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))
    frame = inspect.currentframe()
    level = 0
    while frame is not None:
        filename = os.path.abspath(frame.f_code.co_filename)
        if level > 0 and os.path.dirname(filename) != package_dir:
            return level
        frame = frame.f_back
        level += 1
    return 2


def _warn_if_ill_conditioned(periods_per_window: float) -> None:
    """
    Warn when the AC estimator is asked to run outside its valid range.

    See ``AC_MIN_PERIODS_PER_WINDOW`` for why values below 3.0 cannot yield a
    valid normalized correlation. This is a warning rather than an error only
    to give existing callers a release to migrate; it will become an error.
    """
    if periods_per_window < AC_MIN_PERIODS_PER_WINDOW:
        warnings.warn(
            f"periods_per_window={periods_per_window!r} is below "
            f"{AC_MIN_PERIODS_PER_WINDOW} and the autocorrelation method is "
            f"ill-conditioned there: the longest lag searched falls where the "
            f"window's own autocorrelation has decayed to (near) zero, so the "
            f"normalized correlation is dominated by division by a vanishing "
            f"denominator and routinely exceeds 1. Frames whose correlation "
            f"exceeds 1 are reported as nan. Use "
            f"periods_per_window >= {AC_MIN_PERIODS_PER_WINDOW} (Praat's "
            f"standard value for this method is 4.5), or the cross-correlation "
            f"method, whose standard value is 1.0. This will raise an error in "
            f"a future release.",
            FutureWarning,
            stacklevel=_caller_stacklevel(),
        )


def sound_to_harmonicity_ac(
    sound: "Sound",
    time_step: float = 0.01,
    min_pitch: float = 75.0,
    silence_threshold: float = 0.1,
    periods_per_window: float = 4.5,
    reference_peak: float = None
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
        reference_peak: Amplitude reference substituted for the internal
            whole-file statistic; None keeps the legacy behavior (see
            sound_to_pitch). Use Sound.to_harmonicity_ac_referenced for
            the robust speech-referenced default.

    Returns:
        Harmonicity object. Values are dB; ``-200`` marks frames with no
        periodicity and ``nan`` marks frames whose correlation is not
        resolvable from 1 (see ``strength_to_hnr``).. Values are dB; ``-200`` marks frames with no
        periodicity and ``nan`` marks frames whose correlation is not
        resolvable from 1 (see ``strength_to_hnr``).

    Warns:
        FutureWarning: if ``periods_per_window`` is below
            ``AC_MIN_PERIODS_PER_WINDOW`` (3.0), where the AC estimator is
            ill-conditioned. This will become an error in a future release.
    """
    _warn_if_ill_conditioned(periods_per_window)

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
        silence_threshold=silence_threshold,
        periods_per_window=periods_per_window,
        frame_timing="left",
        apply_octave_cost=False,
        apply_intensity_adjustment=False,
        reference_peak=reference_peak
    )

    # Step 2: Extract times and convert strengths to HNR
    times = pitch.times()
    hnr_values = []

    for frame in pitch.frames:
        if frame.voiced:
            hnr = strength_to_hnr(frame.strength)
        else:
            hnr = UNDEFINED_HNR  # No periodicity found (DP11)
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
    periods_per_window: float = 1.0,
    reference_peak: float = None
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
        silence_threshold=silence_threshold,
        periods_per_window=periods_per_window + 1.0,  # +1 for forward CC
        frame_timing="centered",
        apply_octave_cost=False,
        apply_intensity_adjustment=False,
        reference_peak=reference_peak
    )

    # Step 2: Extract times and convert strengths to HNR
    times = pitch.times()
    hnr_values = []

    for frame in pitch.frames:
        if frame.voiced:
            # strength_to_hnr applies the shared boundary policy: r <= 0 ->
            # UNDEFINED_HNR, r >= 1 -> nan.
            hnr = strength_to_hnr(frame.strength)
        else:
            hnr = UNDEFINED_HNR  # No periodicity found (DP11)
        hnr_values.append(hnr)

    return Harmonicity(
        times=times,
        values=np.array(hnr_values),
        time_step=time_step,
        min_pitch=min_pitch
    )
