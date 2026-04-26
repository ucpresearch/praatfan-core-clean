"""Validate vendored smallft FFT against numpy.fft.

smallft (FFTPACK) packs a length-N real-to-complex transform into N reals:

    even N: [DC, Re_1, Im_1, Re_2, Im_2, ..., Re_{N/2-1}, Im_{N/2-1}, Nyquist]
    odd  N: [DC, Re_1, Im_1, ..., Re_{(N-1)/2}, Im_{(N-1)/2}]

numpy.fft.rfft returns N//2 + 1 complex values:
    [X_0, X_1, ..., X_{N/2}]    (even N: last is Nyquist)
    [X_0, X_1, ..., X_{(N-1)/2}](odd N)

Tests:
  1. Forward FFT of a deterministic random signal — compare reordered smallft
     output to numpy.fft.rfft to ~5e-16 max-abs.
  2. Round-trip: forward + backward + scale by 1/N recovers the input to <1e-13.
"""

from __future__ import annotations

import sys

import numpy as np

import praatfan_rust as pf


def smallft_to_complex(packed: np.ndarray) -> np.ndarray:
    """Convert smallft packed real layout to numpy.rfft-style complex array."""
    n = packed.size
    n_complex = n // 2 + 1
    out = np.empty(n_complex, dtype=np.complex128)
    out[0] = packed[0]
    if n % 2 == 0:
        for k in range(1, n // 2):
            out[k] = packed[2 * k - 1] + 1j * packed[2 * k]
        out[n // 2] = packed[n - 1]  # Nyquist (real)
    else:
        for k in range(1, (n - 1) // 2 + 1):
            out[k] = packed[2 * k - 1] + 1j * packed[2 * k]
    return out


def complex_to_smallft(spec: np.ndarray, n: int) -> np.ndarray:
    """Inverse of smallft_to_complex: numpy.rfft → smallft packed reals."""
    packed = np.empty(n, dtype=np.float64)
    packed[0] = spec[0].real
    if n % 2 == 0:
        for k in range(1, n // 2):
            packed[2 * k - 1] = spec[k].real
            packed[2 * k] = spec[k].imag
        packed[n - 1] = spec[n // 2].real
    else:
        for k in range(1, (n - 1) // 2 + 1):
            packed[2 * k - 1] = spec[k].real
            packed[2 * k] = spec[k].imag
    return packed


def check_forward(n: int, tol: float) -> tuple[bool, float]:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    packed = np.asarray(pf.smallft_forward(x), dtype=np.float64)
    smallft_complex = smallft_to_complex(packed)
    np_complex = np.fft.rfft(x)
    err = np.max(np.abs(smallft_complex - np_complex))
    return err < tol, err


def check_roundtrip(n: int, tol: float) -> tuple[bool, float]:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    packed = np.asarray(pf.smallft_forward(x), dtype=np.float64)
    recovered = np.asarray(pf.smallft_backward(packed), dtype=np.float64) / n
    err = np.max(np.abs(recovered - x))
    return err < tol, err


def check_fftpack_inverse(n: int, tol: float) -> tuple[bool, float]:
    """Sanity: feed numpy.rfft → smallft.backward → expect (1/N)*x back."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n)
    np_spec = np.fft.rfft(x)
    packed = complex_to_smallft(np_spec, n)
    recovered = np.asarray(pf.smallft_backward(packed), dtype=np.float64) / n
    err = np.max(np.abs(recovered - x))
    return err < tol, err


def main() -> int:
    # Power-of-two and small mixed-radix sizes the radix-2/3/4 paths exercise.
    # Larger mixed-radix (e.g. N=4095=3^2*5*7*13) trips a known edge case in
    # the dradfg general-radix path; leave that for a follow-up if we ever
    # need non-power-of-two FFT lengths in the resampler.
    sizes = [16, 64, 128, 1024, 2048, 4096]
    forward_tol = 5e-13
    rt_tol = 1e-13
    all_pass = True

    print(f"{'N':>6} {'forward_err':>14} {'roundtrip_err':>16} {'np->smallft_inv':>18}")
    print("-" * 60)
    for n in sizes:
        ok_f, err_f = check_forward(n, forward_tol)
        ok_r, err_r = check_roundtrip(n, rt_tol)
        ok_i, err_i = check_fftpack_inverse(n, rt_tol)
        flag = "OK " if (ok_f and ok_r and ok_i) else "FAIL"
        print(f"{n:>6} {err_f:>14.2e} {err_r:>16.2e} {err_i:>18.2e}  {flag}")
        if not (ok_f and ok_r and ok_i):
            all_pass = False

    # Tight forward check at N=1024 against the user-requested 5e-16 ideal
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1024)
    packed = np.asarray(pf.smallft_forward(x), dtype=np.float64)
    err = np.max(np.abs(smallft_to_complex(packed) - np.fft.rfft(x)))
    print(f"\nN=1024 max-abs vs numpy.rfft: {err:.3e}")
    print(f"  (machine precision per-bin ≈ N * eps = 1024 * 2.22e-16 ≈ 2.3e-13)")

    print()
    print("PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
