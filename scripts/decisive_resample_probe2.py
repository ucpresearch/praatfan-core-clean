"""Refinement probe: upsample an impulse to reveal the kernel densely."""
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


# --- Dense kernel via high upsampling ---
print("=== Dense kernel reveal (30x upsample of impulse) ===")
old_rate = 1000.0
new_rate = 30000.0  # 30x up: oversample the kernel
N = 200
p = N // 2
x = np.zeros(N); x[p] = 1.0
y = praat_resample(x, old_rate, new_rate, precision=50)

# Output sample m corresponds to input position x_in = (m+0.5)*old/new - 0.5
m = np.arange(len(y))
x_in = (m + 0.5) * (old_rate / new_rate) - 0.5
offset = x_in - p   # in input samples

# Only the central lobes
mask = (offset > -60) & (offset < 60)
kernel_x = offset[mask]
kernel_y = y[mask]

peak_idx = int(np.argmax(kernel_y))
peak_val = kernel_y[peak_idx]
# Find zero crossings
zc = []
for i in range(peak_idx, len(kernel_y) - 1):
    if kernel_y[i] * kernel_y[i + 1] < 0:
        # Linear interp
        f = kernel_y[i] / (kernel_y[i] - kernel_y[i + 1])
        zc.append(kernel_x[i] + f * (kernel_x[i + 1] - kernel_x[i]))
    if len(zc) >= 30:
        break
print(f"peak value: {peak_val:.6e} at offset {kernel_x[peak_idx]:+.4f}")
print(f"zero crossings (right of peak): {[f'{z:.4f}' for z in zc[:15]]}")

# Extrema between zero-crossings -> envelope
extrema = []
zc_idx = []
for z in zc:
    # find sample index closest to z
    idx = int(np.argmin(np.abs(kernel_x - z)))
    zc_idx.append(idx)

prev = peak_idx
env_x, env_y = [], []
for idx in zc_idx:
    sub = kernel_y[prev:idx + 1]
    if len(sub) == 0:
        continue
    k = int(np.argmax(np.abs(sub)))
    env_x.append(kernel_x[prev + k])
    env_y.append(abs(kernel_y[prev + k]))
    prev = idx
env_x = np.array(env_x); env_y = np.array(env_y)
env_y_norm = env_y / peak_val
mask = env_x > 0
ex = env_x[mask]; ey = env_y_norm[mask]

print("Envelope (input-sample offset | normalized):")
for a, b in list(zip(ex, ey))[:15]:
    print(f"  {a:+.3f} | {b:.4e}")

# Determine support N_half: first offset where envelope near 0
# Try to find last extremum above 1e-4
support_candidates = ex[ey > 1e-3]
N_half = support_candidates[-1] if len(support_candidates) else ex[-1]
print(f"Effective support (last envelope > 1e-3): N_half = {N_half:.2f}")

# Now fit canonical windows on envelope
def chi(pred):
    return float(np.sum((ey - pred) ** 2))

hann = 0.5 + 0.5 * np.cos(np.pi * ex / N_half)
hann = np.where(ex <= N_half, hann, 0.0)
hamming = 0.54 + 0.46 * np.cos(np.pi * ex / N_half)
tri = np.maximum(0.0, 1.0 - ex / N_half)
# Gaussian fit: env = exp(-0.5*(x/sigma)^2)
pos = ey > 1e-6
logy = -2.0 * np.log(ey[pos])
A = ex[pos] ** 2
sigma2 = np.sum(A * A) / np.sum(A * logy)
sigma = math.sqrt(sigma2)
gauss = np.exp(-0.5 * (ex / sigma) ** 2)

best_beta = None; best_chi = float("inf")
for beta in np.linspace(1.0, 30.0, 291):
    t = np.clip(1.0 - (ex / N_half) ** 2, 0.0, None)
    k = i0(beta * np.sqrt(t)) / i0(beta)
    c = chi(k)
    if c < best_chi:
        best_chi = c; best_beta = beta
kaiser = i0(best_beta * np.sqrt(np.clip(1 - (ex / N_half) ** 2, 0, None))) / i0(best_beta)

print(f"\nWindow fit chi² (N_half={N_half:.2f}):")
print(f"  Hann                : {chi(hann):.4e}")
print(f"  Hamming             : {chi(hamming):.4e}")
print(f"  Triangular          : {chi(tri):.4e}")
print(f"  Gaussian σ={sigma:.3f}     : {chi(gauss):.4e}")
print(f"  Kaiser β={best_beta:.3f}   : {best_chi:.4e}")

# Is the kernel sinc * window? Compute ratio y / sinc(offset) to isolate window
# kernel should be: sinc(offset) * W(offset) / 1 (for upsample, input rate is the reference)
# Actually for upsampling old=1000, new=30000: cutoff is min(new,old)/2 = 500 Hz,
# normalized to input rate. So kernel = sinc(offset) in input-sample units.
raw_x = kernel_x
raw_y = kernel_y / peak_val
sinc_vals = np.sinc(raw_x)
# where sinc is not near zero:
ok = np.abs(sinc_vals) > 0.01
window_samples = raw_y[ok] / sinc_vals[ok]
xw = raw_x[ok]
print("\nImplied window W(x) = y(x) / sinc(x):")
print("   x    |  W(x)")
for xi, wi in list(zip(xw, window_samples))[:20]:
    if abs(xi) < 30:
        print(f"  {xi:+.3f} | {wi:+.4e}")


# --- Precision support test ---
print("\n=== Precision -> support ===")
for prec in (1, 2, 3, 5, 10, 50):
    yy = praat_resample(x, old_rate, new_rate, precision=prec)
    # Find where |yy| < 1e-6 in symmetric region
    m = np.arange(len(yy))
    x_in = (m + 0.5) * (old_rate / new_rate) - 0.5
    offs = x_in - p
    # rightward: find largest offset with |yy| > 1% of peak
    pk = np.max(np.abs(yy))
    mask = np.abs(yy) > 0.01 * pk
    if mask.sum():
        support = np.max(np.abs(offs[mask]))
    else:
        support = 0.0
    # also zero crossings count in positive half
    right = yy[offs >= 0]
    zeros = 0
    for i in range(len(right) - 1):
        if right[i] * right[i + 1] < 0:
            zeros += 1
    print(f"  precision={prec:>4d} | support ≈ {support:.2f} input samples | zero crossings right = {zeros}")


# --- Bit-exactness deep dive: integer ratios with aligned inputs ---
print("\n=== Q3b: Integer-ratio bit-exactness with longer signals ===")
# Use a very long bandlimited signal to minimize edge effects
old_rate = 48000.0
Ns = 16384
t = np.arange(Ns) / old_rate
x = np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 250 * t)

for factor in [2, 3, 4, 6, 8]:
    new_rate = old_rate / factor
    y = praat_resample(x, old_rate, new_rate)
    m = np.arange(len(y))
    x_in = (m + 0.5) * (old_rate / new_rate) - 0.5
    t_out = x_in / old_rate
    ref = np.sin(2 * np.pi * 100 * t_out) + 0.5 * np.sin(2 * np.pi * 250 * t_out)
    edge = len(y) // 10
    core = slice(edge, len(y) - edge)
    mean_err = float(np.mean(np.abs(y[core] - ref[core])))
    max_err = float(np.max(np.abs(y[core] - ref[core])))
    print(f"  {factor}x down -> {new_rate:>6.0f} Hz | mean|err|={mean_err:.3e} | max|err|={max_err:.3e}")
