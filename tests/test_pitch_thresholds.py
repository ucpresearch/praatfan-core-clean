"""Tests that Sound.to_pitch_ac / to_pitch_cc accept and honor the
Boersma (1993) tuning parameters on every backend.

The unified interface historically only exposed time_step / pitch_floor /
pitch_ceiling. This suite guards the plumbing of voicing_threshold,
silence_threshold, octave_cost, octave_jump_cost, and voiced_unvoiced_cost
through all four adapters.
"""

import numpy as np
import pytest

from praatfan import Sound, get_available_backends, set_backend


def _voiced_count(pitch) -> int:
    v = pitch.values
    if callable(v):
        v = v()
    return int(np.sum(np.asarray(v).ravel() > 0))


def _make_sine_plus_click(sr: int = 16000, dur: float = 2.0, f0: float = 150.0,
                          sine_amp: float = 0.05) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    sig = sine_amp * np.sin(2 * np.pi * f0 * t).astype(np.float64)
    sig[sr] = 1.0  # one loud sample at t=1.0s; ~20x the sine
    return sig


@pytest.fixture(params=get_available_backends())
def backend(request):
    set_backend(request.param)
    return request.param


def test_pitch_ac_accepts_threshold_kwargs(backend):
    sig = _make_sine_plus_click()
    snd = Sound(sig, sampling_frequency=16000)
    # Each kwarg should be accepted without raising.
    snd.to_pitch_ac(
        voicing_threshold=0.45,
        silence_threshold=0.03,
        octave_cost=0.01,
        octave_jump_cost=0.35,
        voiced_unvoiced_cost=0.14,
    )


def test_pitch_cc_accepts_threshold_kwargs(backend):
    sig = _make_sine_plus_click()
    snd = Sound(sig, sampling_frequency=16000)
    snd.to_pitch_cc(
        voicing_threshold=0.45,
        silence_threshold=0.03,
        octave_cost=0.01,
        octave_jump_cost=0.35,
        voiced_unvoiced_cost=0.14,
    )


def test_silence_threshold_monotonic_under_transient(backend):
    """Lowering silence_threshold must never produce fewer voiced frames.

    On a sine with a single loud transient, all four backends should
    voice most frames at the default threshold — our peak denominator is
    a trimmed 99.99th percentile, which ignores the single-sample click.
    Lowering the threshold can only keep voicing or increase it.
    """
    sig = _make_sine_plus_click()
    snd = Sound(sig, sampling_frequency=16000)
    default = _voiced_count(snd.to_pitch_ac())
    permissive = _voiced_count(snd.to_pitch_ac(silence_threshold=1e-6))
    assert permissive >= default
    # All four backends should voice the sine despite the click.
    assert default > 150, (
        f"{backend} voiced only {default}/197 frames with default silence_threshold "
        f"— a single-sample transient is over-silencing voiced speech."
    )


def test_voicing_threshold_monotonic(backend):
    """A very high voicing_threshold should reject more voiced candidates."""
    sig = _make_sine_plus_click(sine_amp=0.2)  # clean-ish sine
    snd = Sound(sig, sampling_frequency=16000)
    low = _voiced_count(snd.to_pitch_ac(voicing_threshold=0.01))
    high = _voiced_count(snd.to_pitch_ac(voicing_threshold=0.99))
    assert low >= high
