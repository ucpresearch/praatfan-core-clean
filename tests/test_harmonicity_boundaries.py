"""
Regression tests for the AC harmonicity boundary policy.

Fixed in v0.1.10 (see the release notes). Before that,
``to_harmonicity_ac`` emitted the constant 99.99999964022885 dB (the old
``r <= 1-1e-10`` clamp) on large fractions of frames, and at
``periods_per_window`` below 3.0 the underlying correlation reached values as
high as ~36, which no normalized correlation can take.
"""

import warnings

import numpy as np
import pytest

from praatfan.harmonicity import (
    AC_MIN_PERIODS_PER_WINDOW,
    EPS,
    UNDEFINED_HNR,
    Harmonicity,
    sound_to_harmonicity_ac,
    sound_to_harmonicity_cc,
    strength_to_hnr,
)
from praatfan.sound import Sound

#: The value the old clamp collapsed near-perfect correlations onto.
LEGACY_CLIP_CONSTANT = 10.0 * np.log10((1.0 - 1e-10) / (1.0 - (1.0 - 1e-10)))

FIXTURES = [
    "tests/fixtures/one_two_three_four_five.wav",
    "tests/fixtures/tam-haʃaʁav-haɡadol-mono.wav",
]


def test_legacy_clip_constant_is_the_value_we_are_guarding_against():
    """Pin the constant, so the guard below cannot silently stop matching."""
    assert LEGACY_CLIP_CONSTANT == pytest.approx(99.99999964022885, abs=1e-9)


class TestStrengthToHnr:
    """The single boundary-policy point (DP10)."""

    @pytest.mark.parametrize(
        "r, expected",
        # Exact 10*log10(r/(1-r)); the manual's "0.99 -> 20 dB" is rounded.
        [
            (0.5, 0.0),
            (0.9, 9.542425094393249),
            (0.99, 19.956351945975499),
            (0.999, 29.995654882259823),
        ],
    )
    def test_valid_correlations_use_the_documented_formula(self, r, expected):
        assert strength_to_hnr(r) == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize("r", [0.0, -0.0, -1e-12, -0.5, float("nan")])
    def test_no_periodicity_reports_undefined(self, r):
        """r <= 0 means no periodic component; Praat reports -200 for
        silence and for DC (black-box observation)."""
        assert strength_to_hnr(r) == UNDEFINED_HNR

    @pytest.mark.parametrize("r", [1.0, 1.0 + 2 ** -50, 1.002, 1.13, 15.0, 35.9])
    def test_invalid_correlations_report_nan_not_a_number(self, r):
        """r >= 1 is not a correlation, so no HNR is invented for it."""
        assert np.isnan(strength_to_hnr(r))

    def test_boundary_is_ulp_robust(self):
        """The last representable double below 1 must not produce a finite
        reading, or the two backends disagree by ~156 dB on frames whose
        correlations differ only in the final bit."""
        assert np.isnan(strength_to_hnr(1.0 - EPS))
        assert np.isnan(strength_to_hnr(np.nextafter(1.0, 0.0)))

    def test_legacy_clip_constant_is_no_longer_a_plateau(self):
        """The old clamp mapped *every* r >= 1-1e-10 onto one value. The
        number itself is still the correct reading for r == 1-1e-10 exactly;
        what must not survive is the plateau above it."""
        assert strength_to_hnr(1.0 - 1e-10) == pytest.approx(
            LEGACY_CLIP_CONSTANT, abs=1e-6)
        for r in [1.0 - 1e-11, 1.0 - 1e-12, 1.0 - 1e-13, 1.0 - 1e-14]:
            value = strength_to_hnr(r)
            assert np.isnan(value) or value > LEGACY_CLIP_CONSTANT + 1.0, (
                f"r = {r} collapsed back onto the clamp: {value}")

    def test_finite_readings_are_strictly_increasing_in_r(self):
        """No plateau: a saturating region would be a clip by another name."""
        rs = np.array([0.5, 0.9, 0.99, 0.999, 1.0 - 1e-6, 1.0 - 1e-12])
        values = np.array([strength_to_hnr(float(r)) for r in rs])
        assert np.all(np.isfinite(values))
        assert np.all(np.diff(values) > 0)


class TestIllConditionedPeriodsPerWindow:
    """``periods_per_window`` below 3.0 (DP34)."""

    @pytest.mark.parametrize("ppw", [0.5, 1.0, 2.0, 2.999])
    def test_ac_warns_below_the_minimum(self, ppw):
        sound = Sound.from_file(FIXTURES[0])
        with pytest.warns(FutureWarning, match="ill-conditioned"):
            sound_to_harmonicity_ac(sound, time_step=0.01, min_pitch=75.0,
                                    periods_per_window=ppw)

    @pytest.mark.parametrize("ppw", [3.0, 4.5, 6.0])
    def test_ac_is_silent_at_or_above_the_minimum(self, ppw):
        sound = Sound.from_file(FIXTURES[0])
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            sound_to_harmonicity_ac(sound, time_step=0.01, min_pitch=75.0,
                                    periods_per_window=ppw)

    def test_cc_does_not_warn_at_its_own_standard_value(self):
        """The CC method does not divide by the window autocorrelation, so
        the AC bound does not apply to it; its standard value is 1.0."""
        sound = Sound.from_file(FIXTURES[0])
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            sound_to_harmonicity_cc(sound, time_step=0.01, min_pitch=75.0,
                                    periods_per_window=1.0)

    def test_minimum_matches_the_documented_bound(self):
        assert AC_MIN_PERIODS_PER_WINDOW == 3.0


class TestContoursOnRealAudio:
    @pytest.mark.parametrize("path", FIXTURES)
    @pytest.mark.parametrize("ppw", [3.0, 4.5, 6.0])
    def test_no_frame_carries_the_clip_constant(self, path, ppw):
        values = sound_to_harmonicity_ac(
            Sound.from_file(path), time_step=0.005, min_pitch=60.0,
            periods_per_window=ppw).values
        assert not np.any(values == LEGACY_CLIP_CONSTANT)

    @pytest.mark.parametrize("path", FIXTURES)
    @pytest.mark.parametrize("ppw", [3.0, 4.5])
    def test_every_frame_is_undefined_nan_or_a_plausible_reading(self, path, ppw):
        values = sound_to_harmonicity_ac(
            Sound.from_file(path), time_step=0.005, min_pitch=60.0,
            periods_per_window=ppw).values
        finite = values[np.isfinite(values) & (values != UNDEFINED_HNR)]
        assert np.all(finite > UNDEFINED_HNR)
        # Nothing may sit at the old ~100 dB ceiling.
        assert not np.any(np.isclose(finite, LEGACY_CLIP_CONSTANT, atol=1e-6))

    @pytest.mark.parametrize("path", FIXTURES)
    def test_ill_conditioned_call_yields_no_invalid_finite_readings(self, path):
        """At ppw=1.0 the correlations are invalid; they must surface as nan
        rather than as a saturated dB value."""
        with pytest.warns(FutureWarning):
            values = sound_to_harmonicity_ac(
                Sound.from_file(path), time_step=0.005, min_pitch=60.0,
                periods_per_window=1.0).values
        assert not np.any(values == LEGACY_CLIP_CONSTANT)
        assert np.any(np.isnan(values)), "expected invalid frames to be nan"


class TestKnownHnrRecovery:
    """The estimator must track a known HNR, so the boundary policy cannot be
    satisfied by simply reporting nan everywhere."""

    @pytest.mark.parametrize("true_hnr", [0.0, 10.0, 20.0, 30.0])
    def test_recovers_synthetic_sine_plus_noise(self, true_hnr):
        sr = 16000
        t = np.arange(sr) / sr
        amp = 0.5
        noise_sd = amp / np.sqrt(2.0 * 10 ** (true_hnr / 10.0))
        x = amp * np.sin(2 * np.pi * 200 * t)
        x = x + np.random.RandomState(0).randn(sr) * noise_sd
        values = sound_to_harmonicity_ac(
            Sound(x, sr), time_step=0.01, min_pitch=60.0,
            periods_per_window=4.5).values
        voiced = values[np.isfinite(values) & (values > UNDEFINED_HNR)]
        assert len(voiced) > 20
        assert np.median(voiced) == pytest.approx(true_hnr, abs=1.0)


class TestUndefinedFramesDoNotPropagate:
    """
    Regression: a marker frame must not leak into interpolated queries.

    Both markers are excluded from the interpolation stencil. Before this was
    fixed, a single nan frame poisoned every query within +/-2 frames (cubic)
    or +/-1 frame (linear), so the two backends disagreed on the damage radius,
    and -200 frames dragged their neighbours 200 dB downward.
    """

    @staticmethod
    def _contour(values):
        values = np.asarray(values, dtype=np.float64)
        return Harmonicity(np.arange(len(values)) * 0.01, values, 0.01, 75.0)

    @pytest.mark.parametrize("interpolation", ["cubic", "linear"])
    def test_nan_frame_does_not_poison_its_neighbours(self, interpolation):
        h = self._contour([16.57, 16.53, 14.61, np.nan, np.nan, 10.97, 11.10, 9.45])
        # Frames either side of the nan gap must read back as themselves.
        assert h.get_value_at_time(0.02, interpolation) == pytest.approx(14.61)
        assert h.get_value_at_time(0.05, interpolation) == pytest.approx(10.97)

    @pytest.mark.parametrize("interpolation", ["cubic", "linear"])
    def test_undefined_marker_does_not_drag_neighbours(self, interpolation):
        h = self._contour([16.57, 16.53, 14.61, UNDEFINED_HNR, 10.97, 11.10])
        # Without exclusion the -200 neighbour pulls this hundreds of dB down.
        assert h.get_value_at_time(0.02, interpolation) == pytest.approx(14.61)
        assert h.get_value_at_time(0.04, interpolation) == pytest.approx(10.97)

    @pytest.mark.parametrize("interpolation", ["cubic", "linear"])
    def test_querying_a_marker_frame_returns_the_marker(self, interpolation):
        h = self._contour([16.57, 16.53, np.nan, 10.97, 11.10])
        assert np.isnan(h.get_value_at_time(0.02, interpolation))
        h2 = self._contour([16.57, 16.53, UNDEFINED_HNR, 10.97, 11.10])
        assert h2.get_value_at_time(0.02, interpolation) == UNDEFINED_HNR

    def test_all_marker_contour_returns_the_marker(self):
        h = self._contour([np.nan] * 4)
        assert np.isnan(h.get_value_at_time(0.01))

    def test_nearest_reads_frames_verbatim(self):
        h = self._contour([16.57, np.nan, 10.97])
        assert np.isnan(h.get_value_at_time(0.01, "nearest"))

    def test_defined_contour_is_unchanged_by_the_exclusion(self):
        """Catmull-Rom weights sum to 1, so a marker-free stencil is identical."""
        values = np.array([3.0, 7.0, 11.0, 5.0, 9.0, 2.0])
        h = self._contour(values)
        for t in np.arange(0.0, 0.05, 0.003):
            idx = t / 0.01
            i = int(np.floor(idx))
            frac = idx - i
            i0, i1 = max(0, i - 1), min(5, i)
            i2, i3 = min(5, i + 1), min(5, i + 2)
            y0, y1, y2, y3 = values[i0], values[i1], values[i2], values[i3]
            t2, t3 = frac * frac, frac ** 3
            expected = 0.5 * ((2 * y1) + (-y0 + y2) * frac
                              + (2 * y0 - 5 * y1 + 4 * y2 - y3) * t2
                              + (-y0 + 3 * y1 - 3 * y2 + y3) * t3)
            assert h.get_value_at_time(float(t), "cubic") == pytest.approx(expected)

    def test_call_layer_treats_nan_as_undefined(self):
        """`_is_undefined` gates the call() interpolation path."""
        from praatfan.praat import _is_undefined
        assert _is_undefined(float("nan"), UNDEFINED_HNR) is True
        assert _is_undefined(UNDEFINED_HNR, UNDEFINED_HNR) is True
        assert _is_undefined(12.3, UNDEFINED_HNR) is False


class TestWarningReachesTheCaller:
    """
    The ill-conditioning warning must be attributed to the user's frame.

    Not cosmetic: the default warning filter dedupes on the attributed
    (module, lineno), so a warning blamed on a praatfan file collapses every
    ill-conditioned call in a process into one visible warning — useless for a
    deprecation whose job is to drive callers off ppw < 3.
    """

    def test_module_function_attributes_to_caller(self):
        sound = Sound.from_file(FIXTURES[0])
        with pytest.warns(FutureWarning) as rec:
            sound_to_harmonicity_ac(sound, time_step=0.01, min_pitch=75.0,
                                    periods_per_window=1.0)
        assert rec[0].filename == __file__

    def test_sound_method_attributes_to_caller(self):
        sound = Sound.from_file(FIXTURES[0])
        with pytest.warns(FutureWarning) as rec:
            sound.to_harmonicity_ac(time_step=0.01, min_pitch=75.0,
                                    periods_per_window=1.0)
        assert rec[0].filename == __file__

    def test_call_layer_attributes_to_caller(self):
        from praatfan.praat import call
        sound = Sound.from_file(FIXTURES[0])
        with pytest.warns(FutureWarning) as rec:
            call(sound, "To Harmonicity (ac)", 0.01, 75, 0.1, 1.0)
        assert rec[0].filename == __file__


class TestDegenerateSignalsAreUndefined:
    """
    Deliberate divergence from Praat, recorded so it is not mistaken for a bug.

    A noiseless periodic signal has a normalized correlation of exactly 1, so
    HNR is 10*log10(1/0) — genuinely undefined. praatfan reports nan for both
    DC and a pure tone. Praat reports -200 for DC and a large finite value for
    a pure tone (its own correlation lands just short of 1). Any noise at all
    puts the correlation back in range: 1% noise gives r = 0.9998.
    """

    @staticmethod
    def _sound(x, sr=16000):
        return Sound(np.asarray(x, dtype=np.float64), sr)

    def test_dc_is_undefined(self):
        v = sound_to_harmonicity_cc(self._sound(np.ones(16000) * 0.5),
                                    time_step=0.01, min_pitch=75.0,
                                    periods_per_window=1.0).values
        assert np.all(np.isnan(v))

    def test_noiseless_tone_carries_no_trustworthy_reading(self):
        """
        Every frame is either nan or a value far above what the method can
        resolve. Which of the two depends on rounding: a float64 tone leaves
        the correlation a few ulps short of 1, giving readings in a narrow
        ~150-155 dB band that is f64 quantization rather than measurement,
        while the same tone written through 16-bit PCM lands on exactly 1 and
        reads nan. Praat's manual puts the reliable ceiling at 37 dB for 4.5
        periods per window, so nothing in that band is a measurement either
        way -- this test pins that no frame masquerades as an ordinary HNR.
        """
        t = np.arange(16000) / 16000
        v = sound_to_harmonicity_cc(self._sound(np.sin(2 * np.pi * 200 * t) * 0.5),
                                    time_step=0.01, min_pitch=75.0,
                                    periods_per_window=1.0).values
        assert len(v) > 50
        assert np.all(np.isnan(v) | (v > 100.0))
        # And none of them is the constant the old clamp manufactured.
        assert not np.any(v == LEGACY_CLIP_CONSTANT)

    def test_a_trace_of_noise_makes_it_measurable_again(self):
        t = np.arange(16000) / 16000
        x = np.sin(2 * np.pi * 200 * t) * 0.5
        x = x + np.random.RandomState(0).randn(16000) * 0.005
        v = sound_to_harmonicity_cc(self._sound(x), time_step=0.01,
                                    min_pitch=75.0, periods_per_window=1.0).values
        finite = v[np.isfinite(v)]
        assert len(finite) > 50
        assert np.all(finite > 20.0)
