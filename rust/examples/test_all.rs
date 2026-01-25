//! Test all praatfan modules against the fixture audio files.

use praatfan::Sound;
use std::path::Path;

fn test_sound_file(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(60));
    println!("Testing: {}", path.display());
    println!("{}", "=".repeat(60));

    // Load sound - try mono first, then extract channel 0 if stereo
    let sound = match Sound::from_file(path) {
        Ok(s) => s,
        Err(praatfan::Error::NotMono(channels)) => {
            println!("  (Stereo file with {} channels, extracting channel 0)", channels);
            Sound::from_file_channel(path, 0)?
        }
        Err(e) => return Err(e.into()),
    };
    println!(
        "Sound: {} samples, {} Hz, {:.3}s duration",
        sound.n_samples(),
        sound.sample_rate(),
        sound.duration()
    );

    // Test Spectrum
    print!("  Spectrum... ");
    let spectrum = sound.to_spectrum(true);
    println!(
        "OK ({} bins, df={:.2} Hz, CoG={:.1} Hz)",
        spectrum.n_bins(),
        spectrum.df(),
        spectrum.get_center_of_gravity(2.0)
    );

    // Test Intensity
    print!("  Intensity... ");
    let intensity = sound.to_intensity(75.0, 0.0);
    let mean_intensity: f64 = intensity.values().iter().sum::<f64>() / intensity.n_frames() as f64;
    println!(
        "OK ({} frames, mean={:.1} dB)",
        intensity.n_frames(),
        mean_intensity
    );

    // Test Pitch AC
    print!("  Pitch AC... ");
    let pitch_ac = sound.to_pitch_ac(0.0, 75.0, 600.0);
    let voiced_count = pitch_ac.frames().iter().filter(|f| f.voiced()).count();
    println!(
        "OK ({} frames, {} voiced)",
        pitch_ac.n_frames(),
        voiced_count
    );

    // Test Pitch CC
    print!("  Pitch CC... ");
    let pitch_cc = sound.to_pitch_cc(0.0, 75.0, 600.0);
    let voiced_count = pitch_cc.frames().iter().filter(|f| f.voiced()).count();
    println!(
        "OK ({} frames, {} voiced)",
        pitch_cc.n_frames(),
        voiced_count
    );

    // Test Harmonicity AC
    print!("  Harmonicity AC... ");
    let hnr_ac = sound.to_harmonicity_ac(0.01, 75.0, 0.1, 4.5);
    let valid_hnr: Vec<f64> = hnr_ac
        .values()
        .iter()
        .filter(|&&v| v > -100.0)
        .copied()
        .collect();
    let mean_hnr = if valid_hnr.is_empty() {
        f64::NAN
    } else {
        valid_hnr.iter().sum::<f64>() / valid_hnr.len() as f64
    };
    println!(
        "OK ({} frames, mean HNR={:.1} dB)",
        hnr_ac.n_frames(),
        mean_hnr
    );

    // Test Harmonicity CC
    print!("  Harmonicity CC... ");
    let hnr_cc = sound.to_harmonicity_cc(0.01, 75.0, 0.1, 1.0);
    let valid_hnr: Vec<f64> = hnr_cc
        .values()
        .iter()
        .filter(|&&v| v > -100.0)
        .copied()
        .collect();
    let mean_hnr = if valid_hnr.is_empty() {
        f64::NAN
    } else {
        valid_hnr.iter().sum::<f64>() / valid_hnr.len() as f64
    };
    println!(
        "OK ({} frames, mean HNR={:.1} dB)",
        hnr_cc.n_frames(),
        mean_hnr
    );

    // Test Formant
    print!("  Formant... ");
    let formant = sound.to_formant_burg(0.0, 5, 5500.0, 0.025, 50.0);
    let f1_values = formant.formant_values(1);
    let valid_f1: Vec<f64> = f1_values.iter().filter(|v| !v.is_nan()).copied().collect();
    let mean_f1 = if valid_f1.is_empty() {
        f64::NAN
    } else {
        valid_f1.iter().sum::<f64>() / valid_f1.len() as f64
    };
    println!(
        "OK ({} frames, mean F1={:.1} Hz)",
        formant.n_frames(),
        mean_f1
    );

    // Test Spectrogram
    print!("  Spectrogram... ");
    let spectrogram = sound.to_spectrogram(0.005, 5000.0, 0.002, 20.0);
    println!(
        "OK ({} time frames x {} freq bins)",
        spectrogram.n_times(),
        spectrogram.n_freqs()
    );

    Ok(())
}

fn main() {
    let fixture_dir = Path::new("../tests/fixtures");

    let test_files = [
        "one_two_three_four_five.wav",
        "one_two_three_four_five_32float.wav",
        "one_two_three_four_five-gain5.wav",
        "one_two_three_four_five_16k.wav",
        "tam-haʃaʁav-haɡadol.wav",
    ];

    println!("Praatfan Rust - Module Tests");
    println!("============================");

    let mut passed = 0;
    let mut failed = 0;

    for filename in &test_files {
        let path = fixture_dir.join(filename);
        match test_sound_file(&path) {
            Ok(()) => {
                println!("  PASSED: {}", filename);
                passed += 1;
            }
            Err(e) => {
                println!("  FAILED: {} - {}", filename, e);
                failed += 1;
            }
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("Summary: {} passed, {} failed", passed, failed);
    println!("{}", "=".repeat(60));

    if failed > 0 {
        std::process::exit(1);
    }
}
