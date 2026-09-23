//! Dump the exact f64 bit patterns of the per-frame analyses (pitch AC/CC,
//! harmonicity AC/CC, intensity) plus wall-clock timings.
//!
//! Used to check that parallelising the frame loops is output-identical:
//! run once per build / thread count and `cmp` the dumps.
//!
//!     cargo run --release --example frame_bitdump -- in.wav out.txt
//!     RAYON_NUM_THREADS=1 cargo run --release --example frame_bitdump -- in.wav out1.txt

use praatfan_rust::formant::sound_to_formant_burg_multi;
use praatfan_rust::{
    sound_to_formant_burg, sound_to_spectrogram, Formant,
    sound_to_harmonicity_ac, sound_to_harmonicity_cc, sound_to_intensity, sound_to_pitch_ac,
    sound_to_pitch_cc, Sound,
};
use std::fmt::Write as _;
use std::time::Instant;

fn dump(out: &mut String, name: &str, series: &[(&str, Vec<f64>)]) {
    for (label, values) in series {
        writeln!(out, "# {name} {label} n={}", values.len()).unwrap();
        for v in values {
            writeln!(out, "{:016x}", v.to_bits()).unwrap();
        }
    }
}

fn formant_series(f: &Formant) -> Vec<(String, Vec<f64>)> {
    let mut v = vec![("times".to_string(), f.times().to_vec())];
    for n in 1..=5 {
        v.push((format!("F{n}"), f.formant_values(n).to_vec()));
        v.push((format!("B{n}"), f.bandwidth_values(n).to_vec()));
    }
    v
}

fn dump_owned(out: &mut String, name: &str, series: &[(String, Vec<f64>)]) {
    let refs: Vec<(&str, Vec<f64>)> = series.iter().map(|(l, v)| (l.as_str(), v.clone())).collect();
    dump(out, name, &refs);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: frame_bitdump <in.wav> <out.txt>");
        std::process::exit(2);
    }
    let sound = Sound::from_file(&args[1]).expect("load");
    let mut out = String::new();

    let t = Instant::now();
    let p = sound_to_pitch_ac(&sound, 0.0, 75.0, 600.0);
    eprintln!("pitch_ac   {:8.1} ms", t.elapsed().as_secs_f64() * 1e3);
    dump(&mut out, "pitch_ac", &[
        ("times", p.times().to_vec()),
        ("values", p.values().to_vec()),
        ("strengths", p.strengths().to_vec()),
    ]);

    let t = Instant::now();
    let p = sound_to_pitch_cc(&sound, 0.0, 75.0, 600.0);
    eprintln!("pitch_cc   {:8.1} ms", t.elapsed().as_secs_f64() * 1e3);
    dump(&mut out, "pitch_cc", &[
        ("times", p.times().to_vec()),
        ("values", p.values().to_vec()),
        ("strengths", p.strengths().to_vec()),
    ]);

    let t = Instant::now();
    let h = sound_to_harmonicity_ac(&sound, 0.01, 75.0, 0.1, 4.5);
    eprintln!("hnr_ac     {:8.1} ms", t.elapsed().as_secs_f64() * 1e3);
    dump(&mut out, "hnr_ac", &[("times", h.times().to_vec()), ("values", h.values().to_vec())]);

    let t = Instant::now();
    let h = sound_to_harmonicity_cc(&sound, 0.01, 75.0, 0.1, 1.0);
    eprintln!("hnr_cc     {:8.1} ms", t.elapsed().as_secs_f64() * 1e3);
    dump(&mut out, "hnr_cc", &[("times", h.times().to_vec()), ("values", h.values().to_vec())]);

    let t = Instant::now();
    let i = sound_to_intensity(&sound, 100.0, 0.0);
    eprintln!("intensity  {:8.1} ms", t.elapsed().as_secs_f64() * 1e3);
    dump(&mut out, "intensity", &[("times", i.times().to_vec()), ("values", i.values().to_vec())]);

    for rate in [11000.0, 10000.0, 7350.0] {
        let t = Instant::now();
        let r = sound.resample(rate);
        eprintln!("resample {rate:>5} {:8.1} ms", t.elapsed().as_secs_f64() * 1e3);
        dump(&mut out, &format!("resample_{rate}"), &[("samples", r.samples().to_vec())]);
    }

    for ceiling in [5000.0, 5500.0, 8000.0] {
        let t = Instant::now();
        let f = sound_to_formant_burg(&sound, 0.0, 5, ceiling, 0.025, 50.0);
        eprintln!("formant {ceiling:>5} {:8.1} ms", t.elapsed().as_secs_f64() * 1e3);
        dump_owned(&mut out, &format!("formant_{ceiling}"), &formant_series(&f));
    }

    let t = Instant::now();
    let fs = sound_to_formant_burg_multi(&sound, 0.0, 5, &[4500.0, 5500.0, 6500.0], 0.025, 50.0);
    eprintln!("formant_multi x3 {:8.1} ms", t.elapsed().as_secs_f64() * 1e3);
    for (k, f) in fs.iter().enumerate() {
        dump_owned(&mut out, &format!("formant_multi_{k}"), &formant_series(f));
    }

    let t = Instant::now();
    let sg = sound_to_spectrogram(&sound, 0.005, 5000.0, 0.002, 20.0);
    eprintln!("spectrogram {:8.1} ms", t.elapsed().as_secs_f64() * 1e3);
    dump(&mut out, "spectrogram", &[("values", sg.values().iter().copied().collect())]);

    std::fs::write(&args[2], out).expect("write");
}
