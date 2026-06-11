//! Speech-referenced amplitude normalization (estimator).
//!
//! Rust port of `src/praatfan/speech_reference.py`, implementing the
//! estimator pinned in `DECISIONS-speech-reference-normalization.md`.
//! This is NOT a Praat algorithm — it is a project-defined estimator
//! shared across the praatfan family so one `reference_peak` (and the
//! z-norm `mean` / `std`) can flow through the whole analysis stack.
//!
//! Pitch (and harmonicity) reference every frame's amplitude against a
//! whole-file statistic (`global_peak`). On long conversational
//! recordings, one loud event anywhere depresses `local_intensity`
//! file-wide and forces quiet voiced frames unvoiced. The estimator
//! computes a norming standard from speech-like frames only, using
//! frame-level robust statistics so a short loud burst (a tiny fraction
//! of the speech frames) cannot dominate it.
//!
//! All percentiles use linear interpolation (Hyndman & Fan type 7,
//! numpy's default) so Python and Rust agree to within summation-order
//! rounding (contract: relative 1e-9 on `reference_peak`).

use crate::error::{Error, Result};

/// Result of [`estimate_speech_reference`].
#[derive(Debug, Clone)]
pub struct SpeechReference {
    /// Per-frame "this looks like speech" flags (hop grid).
    pub speech_mask: Vec<bool>,
    /// Per-frame center times in seconds.
    pub frame_times: Vec<f64>,
    /// Median per-frame mean over speech frames (DC reference).
    pub mean: f64,
    /// Median per-frame RMS over speech frames (the z-norm scale).
    pub std: f64,
    /// The norming standard: `reference_percentile`-ile of per-frame peak
    /// |x| over speech frames. 0.0 for an all-zero / empty signal.
    pub reference_peak: f64,
    /// Fraction of frames in the speech mask.
    pub speech_fraction: f64,
}

/// Percentile with linear interpolation (Hyndman & Fan type 7).
///
/// Matches `np.percentile(values, q)` exactly given the same values:
/// `h = (n-1) * q/100`, result = `a[l] + (h-l) * (a[l+1] - a[l])`.
/// Sorts the slice in place.
fn percentile_type7(values: &mut [f64], q: f64) -> f64 {
    debug_assert!(!values.is_empty());
    let n = values.len();
    if n == 1 {
        return values[0];
    }
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let h = (n - 1) as f64 * q / 100.0;
    let l = h.floor() as usize;
    if l + 1 >= n {
        return values[n - 1];
    }
    values[l] + (h - l as f64) * (values[l + 1] - values[l])
}

/// Median via the type-7 percentile at q=50 (matches `np.median`).
fn median(values: &mut [f64]) -> f64 {
    percentile_type7(values, 50.0)
}

/// Estimate a speech-scoped amplitude reference for normalization.
///
/// Pure function, no analysis side effects. Run once per recording; the
/// result is shared across pitch, HNR, and downstream normalization.
///
/// Algorithm (constants normative, DECISIONS §1–2):
/// 1. Frame the signal (`frame_s` windows, `hop_s` hop, full frames from
///    sample 0); per-frame RMS in dB.
/// 2. `speech_mask` = frames whose dB level is within `speech_floor_db`
///    of the 95th-percentile dB level (inclusive).
/// 3. Frame-level robust standards over speech frames: `std` = median
///    per-frame RMS, `mean` = median per-frame mean, `reference_peak` =
///    `reference_percentile`-ile of per-frame peak |x|.
///
/// Defaults (matching the Python implementation): `frame_s = 0.05`,
/// `hop_s = 0.01`, `speech_floor_db = 30.0`, `reference_percentile = 75.0`.
pub fn estimate_speech_reference(
    samples: &[f64],
    sample_rate: f64,
    frame_s: f64,
    hop_s: f64,
    speech_floor_db: f64,
    reference_percentile: f64,
) -> SpeechReference {
    let n = samples.len();
    if n == 0 {
        return SpeechReference {
            speech_mask: Vec::new(),
            frame_times: Vec::new(),
            mean: 0.0,
            std: 1.0,
            reference_peak: 0.0,
            speech_fraction: 0.0,
        };
    }

    // f64::round is half-away-from-zero, matching the Python
    // floor(x + 0.5) construction.
    let n_frame = ((frame_s * sample_rate).round() as usize).max(1);
    let n_hop = ((hop_s * sample_rate).round() as usize).max(1);

    if n < n_frame {
        // Short signal: one frame covering all samples; treat as speech.
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        let std = var.sqrt();
        let peak = samples.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        return SpeechReference {
            speech_mask: vec![true],
            frame_times: vec![n as f64 / (2.0 * sample_rate)],
            mean,
            std: if std > 0.0 { std } else { 1.0 },
            reference_peak: peak,
            speech_fraction: 1.0,
        };
    }

    let n_frames = 1 + (n - n_frame) / n_hop;

    // Per-frame reductions in one pass: RMS, mean, and peak |x|.
    let mut frame_times = Vec::with_capacity(n_frames);
    let mut rms = Vec::with_capacity(n_frames);
    let mut frame_mean = Vec::with_capacity(n_frames);
    let mut frame_peak = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let start = i * n_hop;
        frame_times.push((start as f64 + n_frame as f64 / 2.0) / sample_rate);
        let mut sumsq = 0.0;
        let mut sum = 0.0;
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &x in &samples[start..start + n_frame] {
            sumsq += x * x;
            sum += x;
            if x < lo {
                lo = x;
            }
            if x > hi {
                hi = x;
            }
        }
        // +1e-20 keeps silent frames finite in dB; matches Python.
        rms.push((sumsq / n_frame as f64 + 1e-20).sqrt());
        frame_mean.push(sum / n_frame as f64);
        frame_peak.push(hi.max(-lo));
    }

    let mut db: Vec<f64> = rms.iter().map(|&r| 20.0 * (r + 1e-20).log10()).collect();
    let ceiling_db = {
        let mut tmp = db.clone();
        percentile_type7(&mut tmp, 95.0)
    };
    let threshold = ceiling_db - speech_floor_db;
    let mut speech_mask: Vec<bool> = db.iter().map(|&d| d >= threshold).collect();
    if !speech_mask.iter().any(|&m| m) {
        speech_mask = vec![true; n_frames];
    }
    // db no longer needed; drop the reference so the borrow ends.
    db.clear();

    let mut masked_rms: Vec<f64> = Vec::new();
    let mut masked_mean: Vec<f64> = Vec::new();
    let mut masked_peak: Vec<f64> = Vec::new();
    let mut n_speech = 0usize;
    for (i, &m) in speech_mask.iter().enumerate() {
        if m {
            masked_rms.push(rms[i]);
            masked_mean.push(frame_mean[i]);
            masked_peak.push(frame_peak[i]);
            n_speech += 1;
        }
    }

    let std = median(&mut masked_rms);
    SpeechReference {
        speech_mask,
        frame_times,
        mean: median(&mut masked_mean),
        std: if std > 0.0 { std } else { 1.0 },
        reference_peak: percentile_type7(&mut masked_peak, reference_percentile),
        speech_fraction: n_speech as f64 / n_frames as f64,
    }
}

/// Resolve a `reference_peak` argument for the `*_referenced` calls.
///
/// Returns `Some(peak)` to substitute for the whole-file statistic, or
/// `None` meaning "use the legacy whole-file statistic" (only reachable
/// when the internal estimator returns 0, i.e. an all-zero signal).
///
/// # Errors
///
/// `Error::InvalidParameter` if an explicit reference is not finite and > 0.
pub fn resolve_reference_peak(
    samples: &[f64],
    sample_rate: f64,
    reference_peak: Option<f64>,
) -> Result<Option<f64>> {
    match reference_peak {
        Some(r) => {
            if !r.is_finite() || r <= 0.0 {
                Err(Error::InvalidParameter(format!(
                    "reference_peak must be finite and > 0, got {}",
                    r
                )))
            } else {
                Ok(Some(r))
            }
        }
        None => {
            let est = estimate_speech_reference(samples, sample_rate, 0.05, 0.01, 30.0, 75.0);
            Ok(if est.reference_peak > 0.0 {
                Some(est.reference_peak)
            } else {
                None
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_numpy_type7() {
        // np.percentile([1,2,3,4], 95) == 3.85
        let mut v = vec![4.0, 2.0, 1.0, 3.0];
        assert!((percentile_type7(&mut v, 95.0) - 3.85).abs() < 1e-12);
        // single value
        let mut one = vec![7.0];
        assert_eq!(percentile_type7(&mut one, 50.0), 7.0);
    }

    #[test]
    fn all_zero_signal_yields_zero_reference() {
        // Silent frames are finite in dB (+1e-20) and all equal, so they
        // pass the mask, but their peak |x| is 0 → reference_peak == 0.
        let s = vec![0.0; 1600];
        let r = estimate_speech_reference(&s, 16000.0, 0.05, 0.01, 30.0, 75.0);
        assert_eq!(r.reference_peak, 0.0);
    }

    #[test]
    fn short_signal_single_frame() {
        let s = vec![0.5; 100];
        let r = estimate_speech_reference(&s, 16000.0, 0.05, 0.01, 30.0, 75.0);
        assert_eq!(r.speech_mask.len(), 1);
        assert!(r.speech_mask[0]);
        assert!((r.reference_peak - 0.5).abs() < 1e-15);
        assert!((r.frame_times[0] - 100.0 / (2.0 * 16000.0)).abs() < 1e-15);
    }

    #[test]
    fn burst_has_bounded_leverage() {
        // 60 s quiet sine + 0.5 s burst: reference stays at sine scale.
        let sr = 16000usize;
        let mut s: Vec<f64> = (0..sr * 60)
            .map(|i| 0.02 * (2.0 * std::f64::consts::PI * 120.0 * i as f64 / sr as f64).sin())
            .collect();
        for x in s[sr * 30..sr * 30 + sr / 2].iter_mut() {
            *x = 0.9;
        }
        let r = estimate_speech_reference(&s, sr as f64, 0.05, 0.01, 30.0, 75.0);
        assert!(
            r.reference_peak < 0.025,
            "reference {} should stay near 0.02",
            r.reference_peak
        );
    }

    #[test]
    fn resolve_rejects_invalid_explicit() {
        let s = vec![0.1; 1000];
        assert!(resolve_reference_peak(&s, 16000.0, Some(-1.0)).is_err());
        assert!(resolve_reference_peak(&s, 16000.0, Some(f64::NAN)).is_err());
        assert_eq!(
            resolve_reference_peak(&s, 16000.0, Some(0.5)).unwrap(),
            Some(0.5)
        );
        // all-zero → reference 0 → legacy fallback (None)
        let z = vec![0.0; 1000];
        assert_eq!(resolve_reference_peak(&z, 16000.0, None).unwrap(), None);
    }
}
