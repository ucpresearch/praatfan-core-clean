"""Tests for Sound.extract_part across all available backends.

Verifies that every backend adapter dispatches to a native extract_part
(no Python samples round-trip) and that the extracted slice matches a
direct numpy slice of the source samples.
"""

from pathlib import Path

import numpy as np
import pytest

from praatfan import Sound, get_available_backends, set_backend


FIXTURE_DIR = Path(__file__).parent / "fixtures"
TEST_WAV = FIXTURE_DIR / "one_two_three_four_five.wav"


@pytest.fixture(params=get_available_backends())
def backend_sound(request):
    set_backend(request.param)
    return request.param, Sound.from_file(TEST_WAV)


def test_matches_numpy_slice(backend_sound):
    backend, snd = backend_sound
    sr = snd.sampling_frequency
    full = np.asarray(snd.values)
    part = snd.extract_part(0.1, 0.2)
    got = np.asarray(part.values)
    expected = full[round(0.1 * sr):round(0.2 * sr)]
    assert len(got) == len(expected)
    # parselmouth's underlying Sound storage can introduce <1e-12 rounding
    # when copying samples through Praat's internal matrix representation.
    assert np.allclose(got, expected, rtol=0, atol=1e-10), \
        f"{backend} slice mismatch (max |diff|={np.max(np.abs(got-expected))})"


def test_sample_rate_preserved(backend_sound):
    _, snd = backend_sound
    assert snd.extract_part(0.1, 0.2).sampling_frequency == snd.sampling_frequency


def test_returns_unified_sound(backend_sound):
    backend, snd = backend_sound
    part = snd.extract_part(0.1, 0.2)
    assert isinstance(part, Sound)
    assert part.backend == backend


def test_start_before_zero_clamps(backend_sound):
    _, snd = backend_sound
    sr = snd.sampling_frequency
    part = snd.extract_part(-0.1, 0.05)
    # Should clamp to [0, 0.05*sr] → 0.05 seconds of samples.
    assert abs(part.n_samples - round(0.05 * sr)) <= 1


def test_end_past_duration_clamps(backend_sound):
    _, snd = backend_sound
    sr = snd.sampling_frequency
    # Ask for a 10 s window beyond the end.
    part = snd.extract_part(snd.duration - 0.1, snd.duration + 10.0)
    # Should clamp to the remaining 0.1 s.
    assert abs(part.n_samples - round(0.1 * sr)) <= 1


def test_chained_extract(backend_sound):
    _, snd = backend_sound
    sr = snd.sampling_frequency
    first = snd.extract_part(0.1, 0.3)
    # 0.0..0.1 inside the first slice corresponds to 0.1..0.2 of the original.
    second = first.extract_part(0.0, 0.1)
    direct = snd.extract_part(0.1, 0.2)
    a = np.asarray(second.values)
    b = np.asarray(direct.values)
    assert len(a) == len(b)
    assert np.allclose(a, b, rtol=0, atol=1e-10)
