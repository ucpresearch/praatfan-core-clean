//! Test formant computation specifically

use praatfan::Sound;
use std::time::Instant;

fn main() {
    let path = "../tests/fixtures/tam-haʃaʁav-haɡadol.wav";  // stereo

    println!("Loading: {}", path);
    let sound = match Sound::from_file(path) {
        Ok(s) => s,
        Err(praatfan::Error::NotMono(ch)) => {
            println!("Stereo ({} ch), extracting channel 0...", ch);
            Sound::from_file_channel(path, 0).unwrap()
        }
        Err(e) => panic!("{}", e),
    };
    println!("Sound: {} samples, {} Hz, {:.3}s",
             sound.n_samples(), sound.sample_rate(), sound.duration());

    // Check sample values
    let samples = sound.samples();
    let max_val = samples.iter().map(|&v| v.abs()).fold(0.0f64, f64::max);
    let min_val = samples.iter().map(|&v| v.abs()).fold(f64::MAX, f64::min);
    let has_nan = samples.iter().any(|v| v.is_nan());
    let has_inf = samples.iter().any(|v| v.is_infinite());
    println!("Sample range: [{:.6}, {:.6}], has_nan={}, has_inf={}", min_val, max_val, has_nan, has_inf);

    println!("\nComputing formant...");
    let start = Instant::now();
    let formant = sound.to_formant_burg(0.0, 5, 5500.0, 0.025, 50.0);
    let elapsed = start.elapsed();

    println!("Done in {:.2?}", elapsed);
    println!("Frames: {}", formant.n_frames());

    let f1 = formant.formant_values(1);
    let valid: Vec<f64> = f1.iter().filter(|v| !v.is_nan()).copied().collect();
    let mean_f1 = valid.iter().sum::<f64>() / valid.len() as f64;
    println!("Mean F1: {:.1} Hz", mean_f1);
}
