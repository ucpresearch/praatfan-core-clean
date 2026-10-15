"""Tests for the praatfan_wasm backend (praatfan_rust as WebAssembly in an
embedded V8) and the auto-selection order around it."""

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pytest

import praatfan.selector as selector
from praatfan import Sound, call, get_available_backends, set_backend
from praatfan.selector import BackendNotAvailableError

FIXTURE_DIR = Path(__file__).parent / "fixtures"
TEST_WAV = FIXTURE_DIR / "one_two_three_four_five.wav"

needs_wasm = pytest.mark.skipif("praatfan_wasm" not in get_available_backends(),
                                reason="praatfan_wasm unavailable (mini-racer not installed)")


@pytest.fixture
def restore_backend():
    saved = selector._current_backend
    yield
    selector._current_backend = saved


# --- auto-selection --------------------------------------------------------

def test_auto_order():
    assert selector.AUTO_PREFERENCE == ("praatfan_rust", "praatfan_gpl",
                                        "praatfan_wasm", "praatfan")


@pytest.mark.parametrize("available, expected", [
    (["praatfan", "parselmouth", "praatfan_wasm", "praatfan_gpl", "praatfan_rust"],
     "praatfan_rust"),
    (["praatfan", "parselmouth", "praatfan_wasm", "praatfan_gpl"], "praatfan_gpl"),
    (["praatfan", "parselmouth", "praatfan_wasm"], "praatfan_wasm"),
    (["parselmouth", "praatfan"], "praatfan"),
])
def test_auto_picks_first_preferred(monkeypatch, restore_backend, available, expected):
    monkeypatch.delenv("PRAATFAN_BACKEND", raising=False)
    monkeypatch.setattr(selector, "_read_config_file", lambda: None)
    monkeypatch.setattr(selector, "get_available_backends", lambda: list(available))
    selector._current_backend = None
    assert selector.get_backend() == expected


def test_parselmouth_never_auto_selected(monkeypatch, restore_backend):
    monkeypatch.delenv("PRAATFAN_BACKEND", raising=False)
    monkeypatch.setattr(selector, "_read_config_file", lambda: None)
    monkeypatch.setattr(selector, "get_available_backends", lambda: ["parselmouth"])
    selector._current_backend = None
    with pytest.raises(BackendNotAvailableError, match="parselmouth"):
        selector.get_backend()


def test_parselmouth_by_name(monkeypatch, restore_backend):
    monkeypatch.setenv("PRAATFAN_BACKEND", "parselmouth")
    monkeypatch.setattr(selector, "get_available_backends",
                        lambda: ["praatfan", "parselmouth"])
    selector._current_backend = None
    assert selector.get_backend() == "parselmouth"


# --- the backend itself ----------------------------------------------------

def _analyses(snd):
    p = snd.to_pitch_ac()
    pc = snd.to_pitch_cc()
    f = snd.to_formant_burg()
    m = snd.get_spectral_moments_at_times(np.array([0.3, 0.6, 1.0]))
    return {
        "times": p.xs(), "f0": p.values(), "strength": p.strengths(),
        "f0_cc": pc.values(),
        "F1": f.formant_values(1), "F2": f.formant_values(2), "F3": f.formant_values(3),
        "B1": f.bandwidth_values(1),
        "intensity": snd.to_intensity().values(),
        "hnr_ac": snd.to_harmonicity_ac().values(),
        "hnr_cc": snd.to_harmonicity_cc().values(),
        "cog": m["center_of_gravity"],
        "band": snd.get_band_energy_at_times(np.array([0.3, 0.6]), 0.0, 1000.0),
    }


@needs_wasm
@pytest.mark.parametrize("reference", ["praatfan_rust", "praatfan"])
def test_matches_reference_backend(restore_backend, reference):
    if reference not in get_available_backends():
        pytest.skip(f"{reference} unavailable")
    set_backend("praatfan_wasm")
    got = _analyses(Sound(TEST_WAV))
    set_backend(reference)
    ref = _analyses(Sound(TEST_WAV))
    for k in got:
        a, b = np.asarray(got[k], float), np.asarray(ref[k], float)
        assert a.shape == b.shape, k
        assert np.array_equal(np.isnan(a), np.isnan(b)), k
        # Same Rust code as praatfan_rust: differences are libm rounding only.
        # Against pure Python, strengths differ by ~1e-3 (as Rust's do).
        tol = 1e-3 if (reference == "praatfan" and k == "strength") else 1e-6
        assert np.nanmax(np.abs(a - b), initial=0.0) < tol, k


@needs_wasm
def test_delegated_and_derived(restore_backend):
    set_backend("praatfan_wasm")
    snd = Sound(TEST_WAV)
    assert snd.backend == "praatfan_wasm"
    part = snd.extract_part(0.2, 0.8)
    assert part.backend == "praatfan_wasm"
    assert part.to_pitch_ac().backend == "praatfan_wasm"
    # outside the pipe: served by the pure-Python implementation
    assert snd.to_spectrum().n_bins > 0
    assert snd.to_spectrogram().n_times > 0
    assert snd.to_pitch_ac_referenced(reference_peak=0.5).n_frames > 0
    with pytest.raises(NotImplementedError, match="praatfan_wasm"):
        snd.to_pitch_ac().raw_ac_strengths()
    pitch = call(snd, "To Pitch (cc)", 0, 75, 600)
    assert call(pitch, "Get number of frames") == pitch.n_frames


@needs_wasm
def test_samples_and_multi(restore_backend):
    set_backend("praatfan_wasm")
    rng = np.random.default_rng(0)
    snd = Sound(rng.standard_normal(16000) * 0.1, sampling_frequency=16000)
    assert snd.to_intensity().n_frames > 0
    multi = snd.to_formant_burg_multi([4500.0, 5500.0])
    single = snd.to_formant_burg(maximum_formant=5500.0)
    assert np.array_equal(multi[1].formant_values(1), single.formant_values(1),
                          equal_nan=True)


def _pool_work(_):
    from praatfan import _wasm
    rng = np.random.default_rng(1)
    return len(_wasm.run(rng.standard_normal(8000) * 0.1, 8000,
                         [{"type": "intensity"}])[0]["times"])


@needs_wasm
@pytest.mark.skipif(sys.platform != "linux", reason="fork start method")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_fork_after_parent_used_v8():
    """A child forked from a process that has started V8 must not touch V8
    (it would hang); it routes its calls to a helper process instead."""
    expected = _pool_work(0)                        # starts V8 in this process
    with mp.get_context("fork").Pool(2) as pool:
        result = pool.map_async(_pool_work, range(4))
        assert result.get(timeout=120) == [expected] * 4


@pytest.mark.parametrize("backend", ["praatfan_rust", "praatfan", "praatfan_wasm"])
def test_unvoiced_f0_is_nan(restore_backend, backend):
    """Every backend reports unvoiced F0 as nan, in values() and in call()."""
    if backend not in get_available_backends():
        pytest.skip(f"{backend} unavailable")
    set_backend(backend)
    pitch = Sound(TEST_WAV).to_pitch_ac()
    f0 = pitch.values()
    assert np.isnan(f0[0]) and not np.any(f0 == 0)
    assert np.isnan(call(pitch, "Get value in frame", 1, "Hertz"))


@needs_wasm
def test_bundle_is_this_version():
    """The vendored bundle must be rebuilt (rust/wasm-v8/build.sh --vendor)
    whenever the version moves; a stale one would be a different engine."""
    from praatfan import __version__, _wasm
    assert _wasm.engine().version == __version__
