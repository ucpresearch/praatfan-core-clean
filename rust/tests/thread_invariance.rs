//! The per-frame loops run in parallel on native targets. Output must be
//! bit-identical regardless of thread count.
#![cfg(not(target_arch = "wasm32"))]

use praatfan_rust::formant::sound_to_formant_burg_multi;
use praatfan_rust::pitch::{sound_to_pitch_internal, FrameTiming, PitchMethod};
use praatfan_rust::{
    sound_to_harmonicity_ac, sound_to_harmonicity_cc,
    sound_to_intensity, sound_to_pitch_ac, sound_to_pitch_cc, Sound,
};

fn analyse(sound: &Sound) -> Vec<u64> {
    let mut bits = Vec::new();
    let mut push = |v: &[f64]| bits.extend(v.iter().map(|x| x.to_bits()));
    for p in [
        sound_to_pitch_ac(sound, 0.0, 75.0, 600.0),
        sound_to_pitch_cc(sound, 0.0, 75.0, 600.0),
    ] {
        push(&p.times().to_vec());
        push(&p.values().to_vec());
        push(&p.strengths().to_vec());
    }
    for h in [
        sound_to_harmonicity_ac(sound, 0.01, 75.0, 0.1, 4.5),
        sound_to_harmonicity_cc(sound, 0.01, 75.0, 0.1, 1.0),
    ] {
        push(&h.times().to_vec());
        push(&h.values().to_vec());
    }
    // Non-default parameters: explicit time step, CC at 3 periods, left timing
    let p = sound_to_pitch_internal(
        sound, 0.005, 60.0, 500.0, PitchMethod::Cc, 0.4, 0.05, 0.02, 0.3, 0.1, 3.0,
        FrameTiming::Left, false, false, None,
    );
    push(&p.values().to_vec());
    push(&p.strengths().to_vec());
    for (min_pitch, step) in [(100.0, 0.0), (60.0, 0.003)] {
        let i = sound_to_intensity(sound, min_pitch, step);
        push(&i.times().to_vec());
        push(&i.values().to_vec());
    }
    for f in sound_to_formant_burg_multi(sound, 0.0, 5, &[4500.0, 5000.0, 5500.0], 0.025, 50.0) {
        for n in 1..=3 {
            push(&f.formant_values(n).to_vec());
        }
    }
    bits
}

fn with_threads(n: usize, sound: &Sound) -> Vec<u64> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
        .install(|| analyse(sound))
}

fn assert_invariant(sound: &Sound) {
    let serial = with_threads(1, sound);
    assert!(!serial.is_empty());
    for n in [2, 3, 8] {
        assert_eq!(serial, with_threads(n, sound), "{n} threads differ from 1");
    }
    // Outside any caller pool: the crate's own pool
    assert_eq!(serial, analyse(sound), "crate pool differs from 1 thread");
}

fn fixture() -> Sound {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/fixtures/tam-haʃaʁav-haɡadol-mono.wav"
    );
    Sound::from_file(path).unwrap()
}

#[test]
fn frame_loops_are_thread_count_invariant() {
    assert_invariant(&fixture());
}

#[test]
fn short_sounds_are_thread_count_invariant() {
    // A handful of frames: rayon may not split at all
    let full = fixture();
    for dur in [0.06, 0.1, 0.25] {
        assert_invariant(&full.extract_part(0.5, 0.5 + dur));
    }
}
