//! A NaN sample must not panic any analysis. Pitch and harmonicity used to
//! panic in the whole-file amplitude statistics (`partial_cmp().unwrap()`
//! over |x|). Now frames whose window covers the NaN come out undefined and
//! the rest are computed as usual.

use praatfan_rust::Sound;

fn signal_with_nans(at: &[usize]) -> Sound {
    let sr = 16_000.0;
    let mut x: Vec<f64> = (0..48_000)
        .map(|i| {
            let t = i as f64 / sr;
            let gate = if libm::sin(2.0 * std::f64::consts::PI * t) > 0.0 { 1.0 } else { 0.0 };
            0.3 * gate * libm::sin(2.0 * std::f64::consts::PI * 150.0 * t)
                + 1e-3 * libm::sin(i as f64 * 1.37)
        })
        .collect();
    for &i in at {
        x[i] = f64::NAN;
    }
    Sound::from_slice(&x, sr)
}

#[test]
fn nan_sample_does_not_panic() {
    for at in [&[5_000usize][..], &[5_000, 20_000, 30_000][..]] {
        let s = signal_with_nans(at);
        let clean = signal_with_nans(&[]);
        for (p, c) in [
            (s.to_pitch_ac(0.0, 75.0, 600.0), clean.to_pitch_ac(0.0, 75.0, 600.0)),
            (s.to_pitch_cc(0.0, 75.0, 600.0), clean.to_pitch_cc(0.0, 75.0, 600.0)),
            (
                s.to_pitch_ac_referenced(0.0, 75.0, 600.0, None).unwrap(),
                clean.to_pitch_ac_referenced(0.0, 75.0, 600.0, None).unwrap(),
            ),
            (
                s.to_pitch_cc_referenced(0.0, 75.0, 600.0, None).unwrap(),
                clean.to_pitch_cc_referenced(0.0, 75.0, 600.0, None).unwrap(),
            ),
        ] {
            assert_eq!(p.n_frames(), c.n_frames());
            // Most frames are still voiced where the clean signal is.
            let pv = p.values();
            let cv = c.values();
            let voiced = |v: f64| v.is_finite() && v > 0.0;
            let both = pv.iter().zip(cv.iter()).filter(|(a, b)| voiced(**a) && voiced(**b)).count();
            let clean_voiced = cv.iter().filter(|b| voiced(**b)).count();
            assert!(both + 20 * at.len() >= clean_voiced, "{both} of {clean_voiced}");
        }
        let _ = s.to_harmonicity_ac(0.01, 75.0, 0.1, 4.5);
        let _ = s.to_harmonicity_cc(0.01, 75.0, 0.1, 1.0);
        let _ = s.to_harmonicity_ac_referenced(0.01, 75.0, 0.1, 4.5, None);
        let _ = s.to_harmonicity_cc_referenced(0.01, 75.0, 0.1, 1.0, None);
        let _ = s.to_intensity(100.0, 0.0);
        let _ = s.to_formant_burg(0.0, 5, 5500.0, 0.025, 50.0);
        let _ = s.to_spectrogram(0.005, 5000.0, 0.002, 20.0);
        let _ = s.resample(11_000.0);
    }
}
