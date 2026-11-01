"""rust/LICENSE-THIRD-PARTY must match Cargo.lock, and every bundled copy must
match it (wheels, the pipe binary, and the WASM bundles all ship it)."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTICES = ROOT / "rust" / "LICENSE-THIRD-PARTY"


@pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo")
def test_notices_match_cargo_lock():
    r = subprocess.run(
        [sys.executable, str(ROOT / "rust" / "third-party" / "generate_notices.py"),
         "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("copy", ["src/praatfan/wasm/LICENSE-THIRD-PARTY",
                                  "r-package/inst/wasm/LICENSE-THIRD-PARTY"])
def test_vendored_copies_present(copy):
    assert (ROOT / copy).is_file(), f"{copy} missing: re-vendor the wasm bundle"
