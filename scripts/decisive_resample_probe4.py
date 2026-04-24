"""Probe 4: precise Hann verification.

Direct test: compute W(x) = kernel/sinc at many offsets, compare to
    Hann: 0.5 + 0.5 cos(pi x / N),  |x| <= N
with N = precision (does support really equal precision exactly?)

Also test alternative Hann parameterizations:
  A: N = precision     (full support ±N)
  B: N = precision + 1
  C: N = precision - 0.5
  D: the "half-Hann" used in _resample_wsinc: depth = precision*step, then
     0.5 + 0.5 cos(pi phi / depth)
"""
from __future__ import annotations
import numpy as np
import parselmouth
from parselmouth.praat import call


def praat_resample(samples, old_rate, new_rate, precision=50):
    snd = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=old_rate)
    out = call(snd, "Resample", new_rate, precision)
    return np.asarray(out.values[0], dtype=np.float64)


# Extract window at HUGE oversample
old_rate = 1000.0
new_rate = 200000.0   # 200x
N = 400
p = N // 2
x = np.zeros(N); x[p] = 1.0

for prec in (50, 10, 5, 3):
    y = praat_resample(x, old_rate, new_rate, precision=prec)
    m = np.arange(len(y))
    offset = (m + 0.5) * (old_rate / new_rate) - 0.5 - p

    # W(x) = y(x) / sinc(x), valid where |sinc(x)| not too small
    sv = np.sinc(offset)
    # interior of kernel
    mask = (np.abs(offset) <= prec + 0.1) & (np.abs(sv) > 0.2)
    xo = offset[mask]
    W = y[mask] / sv[mask]

    # Hann candidates
    def hann(N):
        return np.where(np.abs(xo) <= N, 0.5 + 0.5 * np.cos(np.pi * xo / N), 0.0)

    print(f"\nprecision={prec}")
    for N_try in (prec - 0.5, prec, prec + 0.5, prec + 1, prec + 0.5 * old_rate / new_rate):
        err = np.max(np.abs(W - hann(N_try)))
        rms = np.sqrt(np.mean((W - hann(N_try)) ** 2))
        print(f"   Hann N={N_try:7.4f} : max|err|={err:.3e}  rms={rms:.3e}")
    # Also try half-Hann with "depth = precision * step" where step = max(old/new, 1)
    # For upsample, step = 1, so depth = precision -> same as Hann N=precision
    # For downsample step > 1. We'll test that separately.


# --- For DOWNSAMPLE: does depth = precision * (old/new) hold? ---
print("\n=== Downsample kernel: depth scaling ===")
old_rate = 24000.0
new_rate = 8000.0  # 3x down → step=3
prec = 50
N = 2000
p = N // 2
x = np.zeros(N); x[p] = 1.0
y = praat_resample(x, old_rate, new_rate, precision=prec)
m = np.arange(len(y))
offset_in = (m + 0.5) * (old_rate / new_rate) - 0.5 - p

# For downsample, cutoff = new_rate/2 relative to old_rate -> sinc in input-time
# but normalized to 1/step (step=3). So kernel = (1/step) * sinc(offset/step) * W(offset)
# Let's compute implied kernel = y * step, divide by sinc(offset/step)
step = old_rate / new_rate
scaled_sinc = np.sinc(offset_in / step)
mask = (np.abs(offset_in) <= prec * step + 1) & (np.abs(scaled_sinc) > 0.2)
xo = offset_in[mask]
# y encodes (1/step) * sinc(x/step) * W(x)  (conjecture)
implied_W = y[mask] * step / scaled_sinc[mask]
print(f"step={step}, depth should be ≈ precision*step = {prec*step}")
print("Sample W values (offset | W):")
for xi, wi in zip(xo[::max(1, len(xo)//15)], implied_W[::max(1, len(xo)//15)]):
    print(f"  {xi:+.2f} | {wi:+.6f}")

def hann(xs, N):
    return np.where(np.abs(xs) <= N, 0.5 + 0.5 * np.cos(np.pi * xs / N), 0.0)

for N_try in (prec, prec * step, prec * step + 0.5, prec * step - 0.5):
    err = np.max(np.abs(implied_W - hann(xo, N_try)))
    print(f"  Hann N={N_try:.2f} : max|err|={err:.3e}")


# --- precision=1: special case? ---
print("\n=== precision=1 detailed kernel ===")
old_rate = 1000.0
new_rate = 200000.0
N = 400
p = N // 2
x = np.zeros(N); x[p] = 1.0
y = praat_resample(x, old_rate, new_rate, precision=1)
m = np.arange(len(y))
offset = (m + 0.5) * (old_rate / new_rate) - 0.5 - p
mask = (np.abs(offset) < 2)
print("offset | y")
for xi, yi in zip(offset[mask][::20], y[mask][::20]):
    print(f"  {xi:+.4f} | {yi:+.5f}")
