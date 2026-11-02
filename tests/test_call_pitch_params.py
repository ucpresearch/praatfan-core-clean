"""call(..., "To Pitch (ac|cc)", <10 args>) must pass every threshold and cost
through (it used to keep only time step, floor and ceiling)."""
import numpy as np
import pytest

from praatfan import Sound, call
from praatfan.praat import PraatCallError


def _tone_then_fade(sr=16000):
    t0 = np.arange(int(0.5 * sr)) / sr
    n = int(2.0 * sr)
    t = np.arange(n) / sr
    amp = 10 ** np.linspace(np.log10(0.3), np.log10(0.003), n)
    x = np.concatenate([np.sin(2 * np.pi * 200 * t0), amp * np.sin(2 * np.pi * 200 * (t + 0.5))])
    return Sound(x, sampling_frequency=sr)


@pytest.mark.parametrize("method", ["ac", "cc"])
def test_full_form_passes_thresholds(method):
    snd = _tone_then_fade()
    args = (0.01, 75, 15, "no", 0.1, 0.6, 0.02, 0.5, 0.2, 500)
    via_call = call(snd, f"To Pitch ({method})", *args)
    direct = getattr(snd, f"to_pitch_{method}")(
        time_step=0.01, pitch_floor=75, pitch_ceiling=500,
        silence_threshold=0.1, voicing_threshold=0.6, octave_cost=0.02,
        octave_jump_cost=0.5, voiced_unvoiced_cost=0.2)
    a = np.asarray(via_call.values())
    b = np.asarray(direct.values())
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.allclose(a[~np.isnan(a)], b[~np.isnan(b)])
    # and the silence threshold really matters on this signal
    default = getattr(snd, f"to_pitch_{method}")(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    assert np.isnan(a).sum() != np.isnan(np.asarray(default.values())).sum()


def test_partial_form_rejected():
    snd = _tone_then_fade()
    with pytest.raises(PraatCallError):
        call(snd, "To Pitch (ac)", 0.01, 75, 15, "no", 0.03)


def test_very_accurate_warns():
    snd = _tone_then_fade()
    with pytest.warns(UserWarning, match="very_accurate"):
        call(snd, "To Pitch (ac)", 0.01, 75, 15, "yes", 0.03, 0.45, 0.01, 0.35, 0.14, 600)
