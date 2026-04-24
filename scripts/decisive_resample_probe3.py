"""Probe 3: isolate the exact window W(x) = kernel / sinc."""
from __future__ import annotations
import math
import numpy as np
import parselmouth
from parselmouth.praat import call
from numpy import i0


def praat_resample(samples, old_rate, new_rate, precision=50):
    snd = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=old_rate)
    out = call(snd, "Resample", new_rate, precision)
    return np.asarray(out.values[0], dtype=np.float64)


# =====================================================================
# Clean window extraction. Upsample 100x to get dense kernel samples
# and divide by sinc. Evaluate at offsets where sinc is not near zero.
# =====================================================================
print("=== Clean W(x) = kernel / sinc extraction ===")
old_rate = 1000.0
new_rate = 100000.0   # 100x upsample
N = 400
p = N // 2
x = np.zeros(N); x[p] = 1.0

for prec in (1, 5, 10, 50):
    y = praat_resample(x, old_rate, new_rate, precision=prec)
    m = np.arange(len(y))
    offset = (m + 0.5) * (old_rate / new_rate) - 0.5 - p  # in input samples

    # Restrict to a region centered on 0 (interior, far from edges)
    mask = (offset > -prec - 2) & (offset < prec + 2)
    xo = offset[mask]
    yo = y[mask]

    # Pick points where |sinc(xo)| > 0.05
    sv = np.sinc(xo)
    keep = np.abs(sv) > 0.05
    wsamp_x = xo[keep]
    wsamp = yo[keep] / sv[keep]

    # peak
    pk = np.max(yo)
    print(f"\nprecision={prec}: peak={pk:.6f}")
    # Show window at offsets -prec, -prec/2, 0, prec/2, prec
    for target in [-prec, -prec / 2, 0.0, prec / 2, prec]:
        idx = int(np.argmin(np.abs(wsamp_x - target)))
        print(f"  W({wsamp_x[idx]:+.3f}) ≈ {wsamp[idx]:+.6f}")

    # Fit Hann: W(x) = 0.5 + 0.5 cos(pi x / prec)   (since support is [-prec, prec])
    hann_pred = 0.5 + 0.5 * np.cos(np.pi * wsamp_x / prec)
    hann_pred = np.where(np.abs(wsamp_x) <= prec, hann_pred, 0.0)
    rms_hann = float(np.sqrt(np.mean((wsamp - hann_pred) ** 2)))

    # Kaiser with various beta
    best = (None, float("inf"))
    for beta in np.linspace(1.0, 30.0, 291):
        t = np.clip(1.0 - (wsamp_x / prec) ** 2, 0, None)
        k = i0(beta * np.sqrt(t)) / i0(beta)
        rms = float(np.sqrt(np.mean((wsamp - k) ** 2)))
        if rms < best[1]:
            best = (beta, rms)

    # Triangular
    tri = np.maximum(0, 1 - np.abs(wsamp_x) / prec)
    rms_tri = float(np.sqrt(np.mean((wsamp - tri) ** 2)))

    # Rect (boxcar)
    rect = np.where(np.abs(wsamp_x) <= prec, 1.0, 0.0)
    rms_rect = float(np.sqrt(np.mean((wsamp - rect) ** 2)))

    # Cosine-half (first lobe of cos)
    cos_half = np.where(np.abs(wsamp_x) <= prec, np.cos(np.pi * wsamp_x / (2 * prec)), 0.0)
    rms_cos = float(np.sqrt(np.mean((wsamp - cos_half) ** 2)))

    # Hamming
    hamm = 0.54 + 0.46 * np.cos(np.pi * wsamp_x / prec)
    hamm = np.where(np.abs(wsamp_x) <= prec, hamm, 0.0)
    rms_hamm = float(np.sqrt(np.mean((wsamp - hamm) ** 2)))

    print(f"  RMS window-fit error (smaller=better):")
    print(f"     rect         : {rms_rect:.4e}")
    print(f"     triangular   : {rms_tri:.4e}")
    print(f"     hann         : {rms_hann:.4e}")
    print(f"     hamming      : {rms_hamm:.4e}")
    print(f"     cos(πx/2N)   : {rms_cos:.4e}")
    print(f"     kaiser β={best[0]:.2f}: {best[1]:.4e}")


# =====================================================================
# Check if precision=50 => support=50 EXACT? i.e. W(x) = 0 for |x| >= 50?
# =====================================================================
print("\n=== Support verification for precision=50 ===")
old_rate = 1000.0
new_rate = 100000.0
N = 400
p = N // 2
x = np.zeros(N); x[p] = 1.0
y = praat_resample(x, old_rate, new_rate, precision=50)
m = np.arange(len(y))
offset = (m + 0.5) * (old_rate / new_rate) - 0.5 - p

# Scan values at various offsets
for tgt in [45, 48, 49, 49.5, 49.9, 50, 50.1, 51, 55, 60]:
    idx = int(np.argmin(np.abs(offset - tgt)))
    print(f"  offset = {offset[idx]:+.4f} : y = {y[idx]:+.3e}")


# =====================================================================
# Cutoff for down-sampling: does the kernel scale with new_rate?
# For 2x down (24000->12000), cutoff is 6000 Hz.
# Kernel in input-time units should be sinc(t*2*cutoff) * W(...)
# i.e. zero crossings at 1/cutoff = 1/6000 s = every 2 input samples at 24kHz
# =====================================================================
print("\n=== Downsample kernel (ratio 2:1) ===")
old_rate = 24000.0
new_rate = 12000.0
N = 2000
p = N // 2
x = np.zeros(N); x[p] = 1.0
y = praat_resample(x, old_rate, new_rate, precision=50)
# output index whose input-domain position is nearest p
m = np.arange(len(y))
offs = (m + 0.5) * (old_rate / new_rate) - 0.5 - p
# offs goes in steps of 2 (since old/new = 2)
print("First 20 output samples around peak (input-sample offset | value):")
near = np.argsort(np.abs(offs))[:40]
near = np.sort(near)
for idx in near[:20]:
    print(f"  offset={offs[idx]:+.4f} | y={y[idx]:+.6e}")

# Now check: are zero crossings at offset ±1, ±3, ±5, ... (i.e. every 2 input
# samples spacing in the output-sample grid)?
# In input-sample units, zero crossings of sinc(t*cutoff*2) where
# cutoff = new_rate/2 = 12000Hz. So zero crossings at t = k / cutoff = k / 12000
# In input samples (24000 Hz): 2k samples apart.
# Output sample spacing is exactly 2 input samples. So every output sample
# (away from peak) should sit on a zero crossing of the sinc part — but we see
# non-zero values because of window tails. Wait — let me check: output sample
# spacing = old/new = 2. That matches zero-crossing spacing. So *all* output
# samples far from peak should be exactly on zero crossings of the sinc kernel.
# → tails only come from window/kernel overlap.
