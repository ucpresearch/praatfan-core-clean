"""Tests for speech-referenced amplitude normalization.

Covers the estimator (estimate_speech_reference) and the new
``*_referenced`` analysis variants, per
SPEC-/DECISIONS-speech-reference-normalization.md:

- estimator algorithm + degenerate cases (DECISIONS §1)
- numpy-type7 percentile agreement (§2)
- new variants substitute reference_peak; legacy entry points untouched (§3)
- explicit reference is deterministic + scale-invariant (§4)
- the synthetic 10-min-quiet-speech-plus-burst regression (§5)
- cross-implementation agreement (pure-Python vs praatfan_rust) when both
  are importable
- parselmouth / unsupported wheels warn-and-ignore loudly (§3.3)
"""

import warnings

import numpy as np
import pytest

from praatfan import (
    Sound,
    SpeechReference,
    estimate_speech_reference,
    get_available_backends,
    set_backend,
    ReferencePeakIgnoredWarning,
)
from praatfan.speech_reference import (
    estimate_speech_reference as py_estimate,
    resolve_reference_peak,
)
from praatfan.sound import Sound as PySound

SR = 16000


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _quiet_speech(duration_s=60.0, amp=0.02, f0=120.0, sr=SR):
    t = np.arange(int(sr * duration_s)) / sr
    return amp * np.sin(2 * np.pi * f0 * t)


def _with_burst(speech, at_s=30.0, dur_s=0.5, level=0.9, sr=SR):
    sig = speech.copy()
    start = int(sr * at_s)
    sig[start:start + int(sr * dur_s)] = level
    return sig


def _voiced_mask(pitch):
    v = pitch.values() if callable(pitch.values) else pitch.values
    return np.asarray(v).ravel() > 0


# ---------------------------------------------------------------------------
# Estimator: algorithm and degenerate cases (DECISIONS §1)
# ---------------------------------------------------------------------------

def test_estimator_returns_dataclass():
    ref = py_estimate(_quiet_speech(5.0), SR)
    assert isinstance(ref, SpeechReference)
    assert ref.speech_mask.dtype == bool
    assert ref.frame_times.shape == ref.speech_mask.shape
    assert ref.reference_peak > 0
    # Frame-level z-norm standards (the 75-version contract, SPEC §3).
    assert ref.std > 0
    assert np.isfinite(ref.mean)
    assert 0.0 <= ref.speech_fraction <= 1.0


def test_estimator_frame_level_standards():
    # std = median per-frame RMS; mean = median per-frame mean;
    # reference_peak = 75th-pct per-frame peak |x| over speech frames.
    sig = _quiet_speech(3.0, amp=0.2)
    ref = py_estimate(sig, SR)
    n_frame = round(0.05 * SR)
    n_hop = round(0.01 * SR)
    w = np.lib.stride_tricks.sliding_window_view(sig, n_frame)[::n_hop]
    rms = np.sqrt(np.einsum("ij,ij->i", w, w) / n_frame + 1e-20)
    sp = ref.speech_mask
    assert ref.std == pytest.approx(np.median(rms[sp]), abs=1e-12)
    assert ref.mean == pytest.approx(np.median(w.mean(axis=1)[sp]), abs=1e-12)
    # A pure sine is all-speech: scale should sit near the sine RMS.
    assert ref.std == pytest.approx(0.2 / np.sqrt(2), rel=0.02)


def test_estimator_frame_grid():
    # Full frames from sample 0: n_frames = 1 + (N - n_frame) // n_hop
    sr = SR
    n = sr  # 1 s
    ref = py_estimate(np.ones(n) * 0.1, sr)
    n_frame = round(0.05 * sr)
    n_hop = round(0.01 * sr)
    expected = 1 + (n - n_frame) // n_hop
    assert len(ref.speech_mask) == expected
    # First center at n_frame/2 / sr
    assert ref.frame_times[0] == pytest.approx(n_frame / 2.0 / sr)


def test_estimator_all_zero_signal():
    # Silent frames are finite in dB (+1e-20) and all equal, so the mask
    # is all-True, but every per-frame peak |x| is 0 → reference_peak == 0.
    ref = py_estimate(np.zeros(SR), SR)
    assert ref.reference_peak == 0.0


def test_estimator_empty_signal():
    ref = py_estimate(np.zeros(0), SR)
    assert ref.reference_peak == 0.0
    assert len(ref.speech_mask) == 0
    assert len(ref.frame_times) == 0


def test_estimator_short_signal_single_frame():
    s = np.ones(100) * 0.5
    ref = py_estimate(s, SR)
    assert len(ref.speech_mask) == 1
    assert ref.speech_mask[0]
    assert ref.reference_peak == pytest.approx(0.5)
    assert ref.frame_times[0] == pytest.approx(100 / (2.0 * SR))


def test_estimator_burst_has_bounded_leverage():
    # A 0.5 s burst inside 60 s of speech is ~50 frames out of ~6000 and
    # cannot reach the 75th percentile of per-frame peaks: the reference
    # stays near the sine level.
    speech = _quiet_speech(60.0)
    ref_clean = py_estimate(speech, SR).reference_peak
    ref_burst = py_estimate(_with_burst(speech), SR).reference_peak
    assert ref_burst == pytest.approx(ref_clean, rel=0.10)
    # Legacy whole-file statistic, by contrast, is fully saturated by it.
    legacy = np.percentile(np.abs(_with_burst(speech)), 99.99)
    assert legacy > 0.5


def test_estimator_cumulative_bursts_bounded():
    # The v1 killer (DECISIONS §5.6): cumulative loud non-speech at ~2.5%
    # of speech-masked time — well past v1's 2% sample-domain cliff, where
    # the reference moved +4400%. v2's frame-domain p=75 keeps it within
    # 5%. (Staying under the 95th-pct mask ceiling's own ~5% cliff, the
    # tighter of the two limits.)
    speech = _quiet_speech(60.0)
    contaminated = speech.copy()
    # 3 separate 0.5 s bursts = 1.5 s total ≈ 2.5% of the 60 s.
    for k in range(3):
        start = int(SR * (10.0 + k * 18.0))
        contaminated[start:start + int(SR * 0.5)] = 0.9
    ref_clean = py_estimate(speech, SR).reference_peak
    ref_dirty = py_estimate(contaminated, SR).reference_peak
    assert ref_dirty == pytest.approx(ref_clean, rel=0.05)


def test_estimator_silence_does_not_dilute():
    # Speech followed by long silence: reference is set by the speech, not
    # diluted toward zero by the silence (unlike a whole-file mean).
    speech = _quiet_speech(5.0)
    padded = np.concatenate([speech, np.zeros(SR * 30)])
    r_speech = py_estimate(speech, SR).reference_peak
    r_padded = py_estimate(padded, SR).reference_peak
    assert r_padded == pytest.approx(r_speech, rel=0.05)


# ---------------------------------------------------------------------------
# Percentile definition (DECISIONS §2)
# ---------------------------------------------------------------------------

def test_reference_peak_uses_numpy_type7():
    # reference_peak is the 75th percentile (numpy type-7) of the per-frame
    # peak |x| over speech-masked frames. Recompute it independently.
    sig = _quiet_speech(2.0, amp=0.3)
    ref = py_estimate(sig, SR)
    assert ref.speech_mask.all()
    n_frame = round(0.05 * SR)
    n_hop = round(0.01 * SR)
    windows = np.lib.stride_tricks.sliding_window_view(sig, n_frame)[::n_hop]
    frame_peak = np.maximum(windows.max(axis=1), -windows.min(axis=1))
    expected = np.percentile(frame_peak[ref.speech_mask], 75.0)
    assert ref.reference_peak == pytest.approx(expected, rel=0, abs=1e-12)


# ---------------------------------------------------------------------------
# resolve_reference_peak
# ---------------------------------------------------------------------------

def test_resolve_explicit_validates():
    s = _quiet_speech(2.0)
    assert resolve_reference_peak(s, SR, 0.5) == 0.5
    with pytest.raises(ValueError):
        resolve_reference_peak(s, SR, 0.0)
    with pytest.raises(ValueError):
        resolve_reference_peak(s, SR, -1.0)
    with pytest.raises(ValueError):
        resolve_reference_peak(s, SR, np.inf)
    with pytest.raises(ValueError):
        resolve_reference_peak(s, SR, np.nan)


def test_resolve_none_estimates_and_falls_back():
    s = _quiet_speech(2.0)
    assert resolve_reference_peak(s, SR, None) == pytest.approx(
        py_estimate(s, SR).reference_peak
    )
    # all-zero → estimator returns 0 → legacy fallback (None)
    assert resolve_reference_peak(np.zeros(SR), SR, None) is None


# ---------------------------------------------------------------------------
# New variants on the pure-Python Sound: legacy entry points untouched
# ---------------------------------------------------------------------------

def test_legacy_pitch_byte_identical():
    # The original entry point must not change when the new code exists.
    s = _quiet_speech(3.0)
    snd = PySound(s, SR)
    a = snd.to_pitch(time_step=0.05).values()
    b = snd.to_pitch(time_step=0.05).values()
    assert np.array_equal(a, b)


def test_referenced_explicit_equals_manual_global_peak():
    # to_pitch_ac_referenced(reference_peak=x) must match feeding x straight
    # into sound_to_pitch(reference_peak=x).
    from praatfan.pitch import sound_to_pitch
    s = _quiet_speech(3.0)
    snd = PySound(s, SR)
    ref = 0.02
    a = snd.to_pitch_ac_referenced(time_step=0.05, reference_peak=ref).values()
    b = sound_to_pitch(snd, time_step=0.05, method="ac", reference_peak=ref).values()
    assert np.array_equal(a, b)


def test_referenced_scale_invariance():
    # Scaling samples and reference together by a power of two is a no-op.
    s = _quiet_speech(5.0)
    snd1 = PySound(s, SR)
    snd8 = PySound(8.0 * s, SR)
    v1 = snd1.to_pitch_ac_referenced(time_step=0.05, reference_peak=0.02).values()
    v8 = snd8.to_pitch_ac_referenced(time_step=0.05, reference_peak=0.16).values()
    assert np.array_equal(np.asarray(v1), np.asarray(v8))


def test_harmonicity_referenced_runs():
    s = _quiet_speech(3.0)
    snd = PySound(s, SR)
    h_ac = snd.to_harmonicity_ac_referenced()
    h_cc = snd.to_harmonicity_cc_referenced()
    assert h_ac.n_frames > 0
    assert h_cc.n_frames > 0


# ---------------------------------------------------------------------------
# Synthetic regression (DECISIONS §5) — the core acceptance test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["ac", "cc"])
def test_burst_confined_to_neighbourhood(method):
    # Use 60 s (not 600) to keep the test fast; the mechanism is identical.
    speech = _quiet_speech(60.0)
    burst = _with_burst(speech, at_s=30.0, dur_s=0.5, level=0.9)
    snd_c = PySound(speech, SR)
    snd_b = PySound(burst, SR)
    ts = 0.05

    if method == "ac":
        legacy_c = snd_c.to_pitch(time_step=ts, method="ac")
        legacy_b = snd_b.to_pitch(time_step=ts, method="ac")
        ref_c = snd_c.to_pitch_ac_referenced(time_step=ts)
        ref_b = snd_b.to_pitch_ac_referenced(time_step=ts)
    else:
        legacy_c = snd_c.to_pitch(time_step=ts, method="cc")
        legacy_b = snd_b.to_pitch(time_step=ts, method="cc")
        ref_c = snd_c.to_pitch_cc_referenced(time_step=ts)
        ref_b = snd_b.to_pitch_cc_referenced(time_step=ts)

    times = np.asarray(legacy_c.times())
    far = np.abs(times - 30.25) > 2.0  # >2 s from the burst

    lc, lb = _voiced_mask(legacy_c), _voiced_mask(legacy_b)
    rc, rb = _voiced_mask(ref_c), _voiced_mask(ref_b)

    # Legacy: the burst changes voicing far from itself (the bug).
    assert int((lc != lb)[far].sum()) > 0
    # New variant: differences confined to the burst neighbourhood.
    assert int((rc != rb)[far].sum()) == 0


# ---------------------------------------------------------------------------
# Cross-implementation agreement: pure-Python vs praatfan_rust
# ---------------------------------------------------------------------------

def _has_rust():
    try:
        import praatfan_rust  # noqa: F401
        return hasattr(__import__("praatfan_rust"), "estimate_speech_reference")
    except ImportError:
        return False


@pytest.mark.skipif(not _has_rust(), reason="praatfan_rust not installed")
def test_estimator_py_rs_agree():
    import praatfan_rust
    for sig in (_quiet_speech(30.0), _with_burst(_quiet_speech(30.0), at_s=15.0)):
        rp = py_estimate(sig, SR)
        rr = praatfan_rust.estimate_speech_reference(sig, float(SR))
        # reference_peak: relative 1e-9 (DECISIONS §4)
        assert rr.reference_peak == pytest.approx(rp.reference_peak, rel=1e-9)
        # speech_mask identical (constants and grid match exactly)
        assert np.array_equal(np.asarray(rr.speech_mask), rp.speech_mask)
        assert np.allclose(np.asarray(rr.frame_times), rp.frame_times,
                           rtol=0, atol=1e-12)


@pytest.mark.skipif(not _has_rust(), reason="praatfan_rust not installed")
def test_referenced_pitch_py_rs_agree():
    import praatfan_rust
    sig = _quiet_speech(10.0)
    ref = 0.02
    pv = np.asarray(
        PySound(sig, SR).to_pitch_ac_referenced(time_step=0.05, reference_peak=ref).values()
    )
    rsnd = praatfan_rust.Sound(sig, sampling_frequency=SR)
    rv = np.asarray(
        rsnd.to_pitch_ac_referenced(time_step=0.05, reference_peak=ref).values()
    )
    assert pv.shape == rv.shape
    # Same tolerance the legacy py↔rs pitch parity holds to.
    assert np.nanmax(np.abs(pv - rv)) < 1e-9


# ---------------------------------------------------------------------------
# Unified layer: per-backend behavior (DECISIONS §3.3)
# ---------------------------------------------------------------------------

@pytest.fixture(params=get_available_backends())
def backend(request):
    set_backend(request.param)
    yield request.param


_NATIVE = {"praatfan", "praatfan_rust"}


def test_unified_estimator_backend_independent(backend):
    # The estimate must not depend on which backend is active.
    sig = _with_burst(_quiet_speech(20.0), at_s=10.0)
    ref = estimate_speech_reference(sig, SR)
    pure = py_estimate(sig, SR)
    assert ref.reference_peak == pytest.approx(pure.reference_peak, rel=1e-9)


def test_unified_referenced_warns_iff_unsupported(backend):
    sig = _with_burst(_quiet_speech(30.0), at_s=15.0)
    snd = Sound(sig, sampling_frequency=SR)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        snd.to_pitch_ac_referenced(time_step=0.05)
        warned = [x for x in w if issubclass(x.category, ReferencePeakIgnoredWarning)]
    native_supported = backend in _NATIVE
    # praatfan_gpl may or may not have native support depending on wheel;
    # only assert the unambiguous cases.
    if backend == "parselmouth":
        assert len(warned) == 1
    elif backend in _NATIVE:
        assert len(warned) == 0


def test_unified_referenced_warns_every_call(backend):
    # On unsupported backends the warning must fire per call, not once.
    if backend in _NATIVE:
        pytest.skip("backend natively supports reference_peak")
    sig = _quiet_speech(5.0)
    snd = Sound(sig, sampling_frequency=SR)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        snd.to_pitch_ac_referenced(time_step=0.05)
        snd.to_pitch_ac_referenced(time_step=0.05)
        warned = [x for x in w if issubclass(x.category, ReferencePeakIgnoredWarning)]
    if backend == "parselmouth":
        assert len(warned) == 2


def test_unified_referenced_fixes_voicing_on_native(backend):
    if backend not in _NATIVE:
        pytest.skip("non-native backends fall back to legacy (warned)")
    speech = _quiet_speech(60.0)
    burst = _with_burst(speech, at_s=30.0, dur_s=0.5, level=0.9)
    snd_c = Sound(speech, sampling_frequency=SR)
    snd_b = Sound(burst, sampling_frequency=SR)
    ts = 0.05
    rc = _voiced_mask(snd_c.to_pitch_ac_referenced(time_step=ts))
    rb = _voiced_mask(snd_b.to_pitch_ac_referenced(time_step=ts))
    times = np.asarray(snd_c.to_pitch_ac_referenced(time_step=ts).xs())
    far = np.abs(times - 30.25) > 2.0
    assert int((rc != rb)[far].sum()) == 0
