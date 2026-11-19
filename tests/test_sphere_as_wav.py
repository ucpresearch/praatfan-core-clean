"""NIST SPHERE files load whatever their extension.

TIMIT ships SPHERE files named ``.WAV`` (a ``NIST_1A`` header, not RIFF).
Every loader must read them by content: the pure-Python loader (libsndfile
reads uncompressed SPHERE; desphere covers shorten) and praatfan_rust (the
statically linked desphere fallback). Files are built here, so no binary
fixtures are needed.
"""
import numpy as np
import pytest

from praatfan.sound import Sound


def _sphere(samples_i16, sr, byte_format):
    order = "<" if byte_format == "01" else ">"
    hdr = (
        "NIST_1A\n   1024\n"
        f"sample_count -i {len(samples_i16)}\n"
        f"sample_rate -i {sr}\n"
        "channel_count -i 1\n"
        "sample_n_bytes -i 2\n"
        f"sample_byte_format -s2 {byte_format}\n"
        "sample_coding -s3 pcm\n"
        "end_head\n"
    ).encode()
    hdr += b" " * (1024 - len(hdr))
    return hdr + np.asarray(samples_i16, dtype=order + "i2").tobytes()


@pytest.fixture
def pcm():
    rng = np.random.default_rng(7)
    t = np.arange(16000) / 16000
    x = 8000 * np.sin(2 * np.pi * 200 * t) + rng.normal(0, 300, t.size)
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


def _loaders():
    out = [("praatfan", lambda p: np.asarray(Sound.from_file(p).samples, dtype=float))]
    try:
        import praatfan_rust as pr
        out.append(("praatfan_rust", lambda p: np.asarray(pr.Sound(p).values(), dtype=float)))
    except ImportError:
        pass
    return out


@pytest.mark.parametrize("byte_format", ["01", "10"])
@pytest.mark.parametrize("name", ["si1673.wav", "SI1673.WAV", "si1673.sph"])
@pytest.mark.parametrize("loader", _loaders(), ids=lambda l: l[0])
def test_sphere_loads_by_content(tmp_path, pcm, byte_format, name, loader):
    path = tmp_path / name
    path.write_bytes(_sphere(pcm, 16000, byte_format))
    got = loader[1](str(path))
    assert np.array_equal(got, pcm / 32768.0)
