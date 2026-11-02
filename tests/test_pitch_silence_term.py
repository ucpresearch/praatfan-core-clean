"""Pitch silence term (Boersma 1993, Eq. 23) and the local peak it uses.

Expected values come from black-box measurements of Praat 6.1.38 via
parselmouth 0.4.7 (2026-11-02), not from Praat's source:
- with the path finder off, Praat stops voicing a slowly decaying 200 Hz tone
  where local/global amplitude = silence threshold, for every voicing
  threshold (ratio/ST measured 1.01-1.02);
- the local peak covers only the central +-half period of the pitch floor:
  a single impulse 150 samples from the frame centre (75 Hz floor, 16 kHz)
  does not count, one 50 samples away does.
"""
import numpy as np
import pytest

from praatfan.pitch import sound_to_pitch
from praatfan.sound import Sound

SR = 16000


def _decaying_tone():
    t0 = np.arange(int(0.5 * SR)) / SR
    n = int(4.0 * SR)
    t = np.arange(n) / SR
    amp = 10 ** np.linspace(np.log10(0.3), np.log10(0.003), n)
    x = np.concatenate([np.sin(2 * np.pi * 200 * t0), amp * np.sin(2 * np.pi * 200 * (t + 0.5))])
    return x, np.concatenate([np.ones(len(t0)), amp])


@pytest.mark.parametrize("method", ["ac", "cc"])
@pytest.mark.parametrize("st", [0.03, 0.1])
@pytest.mark.parametrize("vt", [0.2, 0.45, 0.7])
def test_voicing_stops_at_silence_threshold(method, st, vt):
    x, amp = _decaying_tone()
    p = sound_to_pitch(Sound(x, SR), time_step=0.01, pitch_floor=75, pitch_ceiling=600,
                       method=method, silence_threshold=st, voicing_threshold=vt,
                       octave_cost=0.0, octave_jump_cost=0.0, voiced_unvoiced_cost=0.0)
    last = max(f.time for f in p.frames if f.voiced and f.time > 0.6)
    ratio = amp[int(last * SR)] / st
    assert 0.95 < ratio < 1.1, ratio


@pytest.mark.parametrize("method", ["ac", "cc"])
def test_local_peak_is_central(method):
    x = np.zeros(int(1.0 * SR))
    p0 = sound_to_pitch(Sound(x + 0.0, SR), time_step=0.01, pitch_floor=75, method=method)
    times = np.array([f.time for f in p0.frames])
    k = len(times) // 2
    c = int(round(times[k] * SR))
    for offset, counts in ((50, True), (150, False)):
        y = x.copy()
        y[c + offset] = 1.0
        # reference_peak=1: a lone impulse sits above the p99.99 global peak
        p = sound_to_pitch(Sound(y, SR), time_step=0.01, pitch_floor=75, method=method,
                           reference_peak=1.0)
        li = p.frames[k].intensity
        assert (li > 0.5) == counts, (offset, li)
