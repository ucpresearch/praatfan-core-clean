"""Rounding shared by the pure-Python analyses.

Python's round() (and numpy's) rounds halves to even; the Rust port uses
f64::round, which rounds halves away from zero. Frame centres often land
exactly on a half sample (e.g. any file with an odd number of samples), so
the two rules picked different samples for every frame of such a file and
the backends disagreed file-wide (found 2026-11-02). Every sample-index
round in the analyses goes through rhalf() to keep Python identical to Rust.
"""
import math


def rhalf(x: float) -> int:
    """Round half away from zero, as Rust's f64::round, returned as int."""
    return int(math.copysign(math.floor(abs(x) + 0.5), x))
