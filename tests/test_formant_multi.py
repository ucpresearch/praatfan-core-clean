"""Tests for Sound.to_formant_burg_multi across all available backends.

Verifies that the batched entry point is bit-parity equivalent to calling
to_formant_burg once per ceiling, and that edge cases (empty input, single
ceiling) behave correctly.
"""

from pathlib import Path

import numpy as np
import pytest

from praatfan import Sound, get_available_backends, set_backend


FIXTURE_DIR = Path(__file__).parent / "fixtures"
TEST_WAV = FIXTURE_DIR / "one_two_three_four_five.wav"
CEILINGS = [4500.0, 5000.0, 5500.0, 6000.0, 6500.0]


@pytest.fixture(params=get_available_backends())
def backend_sound(request):
    set_backend(request.param)
    return request.param, Sound.from_file(TEST_WAV)


def _compare_formants(f_multi, f_single, n_formants=4):
    for k in range(1, n_formants + 1):
        a = np.asarray(f_multi.formant_values(k))
        b = np.asarray(f_single.formant_values(k))
        assert np.array_equal(a, b, equal_nan=True), \
            f"F{k} mismatch: multi={a[:5]} single={b[:5]}"
        a = np.asarray(f_multi.bandwidth_values(k))
        b = np.asarray(f_single.bandwidth_values(k))
        assert np.array_equal(a, b, equal_nan=True), \
            f"B{k} mismatch: multi={a[:5]} single={b[:5]}"


def test_bit_parity_with_singular(backend_sound):
    backend, snd = backend_sound
    multi = snd.to_formant_burg_multi(CEILINGS)
    assert len(multi) == len(CEILINGS)
    for hz, f_multi in zip(CEILINGS, multi):
        f_single = snd.to_formant_burg(maximum_formant=hz)
        assert f_multi.n_frames == f_single.n_frames
        _compare_formants(f_multi, f_single)


def test_empty_input_returns_empty_list(backend_sound):
    _, snd = backend_sound
    assert snd.to_formant_burg_multi([]) == []


def test_single_ceiling(backend_sound):
    _, snd = backend_sound
    multi = snd.to_formant_burg_multi([5500.0])
    assert len(multi) == 1
    single = snd.to_formant_burg(maximum_formant=5500.0)
    _compare_formants(multi[0], single)


def test_input_order_preserved(backend_sound):
    _, snd = backend_sound
    shuffled = [6500.0, 4500.0, 5500.0, 5000.0, 6000.0]
    multi = snd.to_formant_burg_multi(shuffled)
    for hz, f_multi in zip(shuffled, multi):
        f_single = snd.to_formant_burg(maximum_formant=hz)
        _compare_formants(f_multi, f_single, n_formants=1)


def test_non_default_params_propagate(backend_sound):
    _, snd = backend_sound
    kwargs = dict(time_step=0.005, max_number_of_formants=4,
                  window_length=0.030, pre_emphasis_from=75.0)
    multi = snd.to_formant_burg_multi(CEILINGS[:2], **kwargs)
    for hz, f_multi in zip(CEILINGS[:2], multi):
        f_single = snd.to_formant_burg(maximum_formant=hz, **kwargs)
        assert f_multi.n_frames == f_single.n_frames
        _compare_formants(f_multi, f_single, n_formants=kwargs["max_number_of_formants"])
