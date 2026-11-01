//! Real FFT on the vendored FFTPACK port (`crate::smallft`).
//!
//! Every FFT in the crate goes through here. The transform is plain scalar
//! Rust and its twiddle factors come from the pure-Rust `libm`, so results are
//! bit-identical on every host: no CPU-dependent SIMD path and no host maths
//! library. (rustfft, used before, picked a SIMD path at run time and built
//! its twiddles with the host `sin`/`cos`.)
//!
//! Lengths whose prime factors are all <= 13 run directly on FFTPACK. Its
//! general-radix kernels build twiddles for a factor p by a rotation
//! recurrence, whose rounding error grows with p (measured: ~1e-12 relative
//! at p ~ 500, ~4e-11 at p = 9973, vs ~2e-16 for smooth lengths). Lengths
//! with a larger prime factor therefore use Bluestein's chirp-z algorithm
//! (Bluestein 1970; Oppenheim & Schafer, "Discrete-Time Signal Processing",
//! the chirp transform algorithm) on power-of-two FFTPACK transforms, which
//! stays at ~2e-16.
//!
//! Packed layout (FFTPACK `rfftf`), for a length-`n` real input `x`:
//!
//! ```text
//! [ X0.re, X1.re, X1.im, X2.re, X2.im, ..., X(n/2).re ]      n even
//! [ X0.re, X1.re, X1.im, ..., X((n-1)/2).re, X((n-1)/2).im ] n odd
//! ```
//!
//! with `Xk = sum_j x[j] * exp(-2*pi*i*j*k/n)` (unnormalized). `backward`
//! (lengths with prime factors <= 13 only) is the unnormalized inverse:
//! `backward(forward(x)) = n * x`.

use crate::smallft::{drftb1, drftf1, fdrffti};

/// Largest prime factor allowed for a direct FFTPACK transform.
const MAX_DIRECT_FACTOR: usize = 13;

fn largest_prime_factor(mut n: usize) -> usize {
    let mut largest = 1;
    let mut p = 2;
    while p * p <= n {
        while n % p == 0 {
            largest = p;
            n /= p;
        }
        p += 1;
    }
    if n > 1 {
        largest = n;
    }
    largest
}

/// Direct FFTPACK real transform.
#[derive(Clone, Debug)]
struct Direct {
    n: usize,
    /// Twiddle factors (FFTPACK `wsave[n..2n]`).
    wa: Vec<f64>,
    /// Factorization of `n` (FFTPACK `ifac`).
    ifac: Vec<i32>,
}

impl Direct {
    fn new(n: usize) -> Self {
        let mut wsave = vec![0.0; 2 * n];
        let mut ifac = vec![0i32; 32];
        fdrffti(n, &mut wsave, &mut ifac);
        Direct { n, wa: wsave[n..].to_vec(), ifac }
    }

    fn forward(&self, data: &mut [f64], work: &mut [f64]) {
        if self.n > 1 {
            drftf1(self.n as i32, data, &mut work[..self.n], &self.wa, &self.ifac);
        }
    }

    fn backward(&self, data: &mut [f64], work: &mut [f64]) {
        if self.n > 1 {
            drftb1(self.n as i32, data, &mut work[..self.n], &self.wa, &self.ifac);
        }
    }
}

/// Bluestein plan for a non-power-of-two length `n`.
///
/// `Xk = conj(c_k) * sum_j (x_j conj(c_j)) c_(k-j)`, `c_j = exp(i*pi*j^2/n)`:
/// a linear convolution of length `2n-1`, done circularly at a power-of-two
/// length `m >= 2n-1` with FFTPACK.
#[derive(Clone, Debug)]
struct Bluestein {
    m: usize,
    inner: Direct,
    /// `conj(c_j)` for j in 0..n, as (cos, -sin).
    chirp: Vec<(f64, f64)>,
    /// Full complex spectrum of the circularly placed `c`, divided by `m`.
    kernel: Vec<(f64, f64)>,
}

impl Bluestein {
    fn new(n: usize) -> Self {
        let m = (2 * n - 1).next_power_of_two();
        let inner = Direct::new(m);
        // Angle pi*j^2/n with j^2 reduced mod 2n in integers, so the phase
        // stays exact for large j.
        let two_n = 2 * n as u128;
        let chirp: Vec<(f64, f64)> = (0..n)
            .map(|j| {
                let r = ((j as u128 * j as u128) % two_n) as f64;
                let ph = std::f64::consts::PI * r / n as f64;
                (libm::cos(ph), -libm::sin(ph))
            })
            .collect();
        // c_j = conj(chirp_j); placed at j and m-j.
        let mut re = vec![0.0; m];
        let mut im = vec![0.0; m];
        for j in 0..n {
            let (c, s) = chirp[j];
            re[j] = c;
            im[j] = -s;
            if j > 0 {
                re[m - j] = c;
                im[m - j] = -s;
            }
        }
        let mut work = vec![0.0; m];
        let mut kernel = complex_fft(&inner, &mut re, &mut im, &mut work);
        let inv_m = 1.0 / m as f64;
        for z in kernel.iter_mut() {
            z.0 *= inv_m;
            z.1 *= inv_m;
        }
        Bluestein { m, inner, chirp, kernel }
    }

    /// Forward transform of real `x` (length n) into packed form.
    fn forward(&self, n: usize, data: &mut [f64]) {
        let m = self.m;
        let mut re = vec![0.0; m];
        let mut im = vec![0.0; m];
        for j in 0..n {
            re[j] = data[j] * self.chirp[j].0;
            im[j] = data[j] * self.chirp[j].1;
        }
        let mut work = vec![0.0; m];
        let spec = complex_fft(&self.inner, &mut re, &mut im, &mut work);
        // Pointwise product with the kernel, then inverse via
        // IFFT(Z) = conj(FFT(conj(Z))) (the 1/m is folded into the kernel).
        for k in 0..m {
            let (a, b) = spec[k];
            let (c, d) = self.kernel[k];
            re[k] = a * c - b * d;
            im[k] = -(a * d + b * c);
        }
        let conv = complex_fft(&self.inner, &mut re, &mut im, &mut work);
        // conv holds conj(result); X_k = chirp_k * result_k.
        for k in 0..=n / 2 {
            let (pr, pi) = (conv[k].0, -conv[k].1);
            let (c, s) = self.chirp[k];
            let xr = pr * c - pi * s;
            let xi = pr * s + pi * c;
            if k == 0 {
                data[0] = xr;
            } else if 2 * k == n {
                data[n - 1] = xr;
            } else {
                data[2 * k - 1] = xr;
                data[2 * k] = xi;
            }
        }
    }
}

/// Full complex FFT of `re + i*im` (power-of-two length) from two real
/// FFTPACK transforms. Overwrites `re` and `im`.
fn complex_fft(
    p: &Direct,
    re: &mut [f64],
    im: &mut [f64],
    work: &mut [f64],
) -> Vec<(f64, f64)> {
    let m = p.n;
    p.forward(re, work);
    p.forward(im, work);
    let bin = |v: &[f64], k: usize| -> (f64, f64) {
        if k == 0 {
            (v[0], 0.0)
        } else if 2 * k == m {
            (v[m - 1], 0.0)
        } else if 2 * k < m {
            (v[2 * k - 1], v[2 * k])
        } else {
            let q = m - k; // Hermitian symmetry of a real input's spectrum
            (v[2 * q - 1], -v[2 * q])
        }
    };
    let mut out = Vec::with_capacity(m);
    for k in 0..m {
        let (ar, ai) = bin(re, k);
        let (br, bi) = bin(im, k);
        out.push((ar - bi, ai + br)); // A + i*B
    }
    out
}

#[derive(Clone, Debug)]
enum Plan {
    Direct(Direct),
    Bluestein(Box<Bluestein>),
}

/// A prepared real FFT of one length. Read-only after construction, so one
/// plan can be shared by all worker threads; each caller supplies its own
/// work buffer.
#[derive(Clone, Debug)]
pub struct RealFft {
    n: usize,
    plan: Plan,
}

impl RealFft {
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "FFT length must be at least 1");
        let plan = if largest_prime_factor(n) <= MAX_DIRECT_FACTOR {
            Plan::Direct(Direct::new(n))
        } else {
            Plan::Bluestein(Box::new(Bluestein::new(n)))
        };
        RealFft { n, plan }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    /// Forward transform in place: `data` (length n, real) becomes the
    /// packed half spectrum. `work` must have length >= n; its contents
    /// are overwritten. (Non-power-of-two lengths allocate internally.)
    pub fn forward(&self, data: &mut [f64], work: &mut [f64]) {
        assert!(data.len() == self.n && work.len() >= self.n);
        match &self.plan {
            Plan::Direct(p) => p.forward(data, work),
            Plan::Bluestein(b) => b.forward(self.n, data),
        }
    }

    /// Unnormalized inverse in place: packed half spectrum -> `n` times the
    /// real signal. Only for lengths with prime factors <= 13 (all inverse
    /// uses in the crate are powers of two). `work` as for
    /// [`forward`](Self::forward).
    pub fn backward(&self, data: &mut [f64], work: &mut [f64]) {
        assert!(data.len() == self.n && work.len() >= self.n);
        match &self.plan {
            Plan::Direct(p) => p.backward(data, work),
            Plan::Bluestein(_) => {
                panic!("RealFft::backward: length {} has a prime factor > 13", self.n)
            }
        }
    }

    /// Number of non-negative-frequency bins: `n/2 + 1`.
    pub fn n_bins(&self) -> usize {
        self.n / 2 + 1
    }

    /// Bin `k` (0 <= k <= n/2) of a packed spectrum as (re, im).
    #[inline]
    pub fn bin(&self, packed: &[f64], k: usize) -> (f64, f64) {
        if k == 0 {
            (packed[0], 0.0)
        } else if 2 * k == self.n {
            (packed[self.n - 1], 0.0)
        } else {
            (packed[2 * k - 1], packed[2 * k])
        }
    }

    /// Replace a packed spectrum by its power spectrum |Xk|^2 (imaginary
    /// parts zero), ready for [`backward`](Self::backward). Used for
    /// autocorrelation via Wiener-Khinchin.
    pub fn packed_to_power(&self, packed: &mut [f64]) {
        let n = self.n;
        packed[0] *= packed[0];
        let mut k = 1;
        while 2 * k < n {
            let re = packed[2 * k - 1];
            let im = packed[2 * k];
            packed[2 * k - 1] = re * re + im * im;
            packed[2 * k] = 0.0;
            k += 1;
        }
        if n % 2 == 0 && n > 1 {
            packed[n - 1] *= packed[n - 1];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                (((i * 7919 + 13) % 101) as f64 / 101.0 - 0.5)
                    + 0.25 * libm::sin(i as f64 * 0.37)
            })
            .collect()
    }

    /// Direct DFT with exact integer phase reduction, at bins 0, step,
    /// 2*step, ... and n/2.
    fn dft(x: &[f64], step: usize) -> Vec<(usize, (f64, f64))> {
        let n = x.len();
        let mut ks: Vec<usize> = (0..=n / 2).step_by(step).collect();
        if *ks.last().unwrap() != n / 2 {
            ks.push(n / 2);
        }
        ks.into_iter()
            .map(|k| {
                let (mut re, mut im) = (0.0, 0.0);
                for (j, &v) in x.iter().enumerate() {
                    let ph = 2.0 * std::f64::consts::PI * ((j * k) % n) as f64 / n as f64;
                    re += v * libm::cos(ph);
                    im -= v * libm::sin(ph);
                }
                (k, (re, im))
            })
            .collect()
    }

    fn check(n: usize) {
        let x = signal(n);
        let fft = RealFft::new(n);
        let mut data = x.clone();
        let mut work = vec![0.0; n];
        fft.forward(&mut data, &mut work);
        let want = dft(&x, if n > 5000 { 97 } else { 1 });
        let scale: f64 = x.iter().map(|v| v.abs()).sum::<f64>().max(1.0);
        for &(k, (re, im)) in want.iter() {
            let (gr, gi) = fft.bin(&data, k);
            let err = (gr - re).abs().max((gi - im).abs());
            assert!(err <= 1e-12 * scale, "n={n} k={k}: got ({gr},{gi}) want ({re},{im})");
        }
        if largest_prime_factor(n) <= MAX_DIRECT_FACTOR {
            fft.backward(&mut data, &mut work);
            for j in 0..n {
                let err = (data[j] / n as f64 - x[j]).abs();
                assert!(err <= 1e-12, "n={n} round trip j={j}: err {err}");
            }
        }
    }

    #[test]
    fn matches_dft_every_length_to_600() {
        for n in 1..=600 {
            check(n);
        }
    }

    #[test]
    fn matches_dft_selected_lengths() {
        for &n in &[1024, 2048, 4095, 4096, 4097, 8192, 16384, 9973, 44100, 48000] {
            check(n);
        }
    }

    #[test]
    fn power_spectrum_gives_autocorrelation() {
        let x = signal(37);
        let n = 128;
        let fft = RealFft::new(n);
        let mut data = vec![0.0; n];
        data[..x.len()].copy_from_slice(&x);
        let mut work = vec![0.0; n];
        fft.forward(&mut data, &mut work);
        fft.packed_to_power(&mut data);
        fft.backward(&mut data, &mut work);
        for lag in 0..x.len() {
            let direct: f64 = (0..x.len() - lag).map(|i| x[i] * x[i + lag]).sum();
            assert!((data[lag] / n as f64 - direct).abs() < 1e-12, "lag {lag}");
        }
    }

    /// The vendored FFTPACK kernels themselves, at every length (including
    /// the repeated odd factors that used to panic and the odd lengths whose
    /// inverse used to be wrong). Tolerance allows for the general-radix
    /// rotation recurrence at large prime factors.
    #[test]
    fn fftpack_kernels_every_length_to_700() {
        for n in 1..=700 {
            let x = signal(n);
            let p = Direct::new(n);
            let mut data = x.clone();
            let mut work = vec![0.0; n];
            p.forward(&mut data, &mut work);
            let scale: f64 = x.iter().map(|v| v.abs()).sum::<f64>().max(1.0);
            let fft = RealFft { n, plan: Plan::Direct(p.clone()) };
            for &(k, (re, im)) in dft(&x, 1).iter() {
                let (gr, gi) = fft.bin(&data, k);
                let err = (gr - re).abs().max((gi - im).abs());
                assert!(err <= 1e-11 * scale, "n={n} k={k}: err {err}");
            }
            p.backward(&mut data, &mut work);
            for j in 0..n {
                let err = (data[j] / n as f64 - x[j]).abs();
                assert!(err <= 1e-11, "n={n} round trip j={j}: err {err}");
            }
        }
        // Lengths named in the original bug reports.
        for &n in &[4095, 4097, 9, 18, 27, 45] {
            let x = signal(n);
            let p = Direct::new(n);
            let mut d = x.clone();
            let mut w = vec![0.0; n];
            p.forward(&mut d, &mut w);
            p.backward(&mut d, &mut w);
            for j in 0..n {
                assert!((d[j] / n as f64 - x[j]).abs() <= 1e-11, "n={n}");
            }
        }
    }
}
