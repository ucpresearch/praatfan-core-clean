
use praatfan::Sound;

fn main() {
    let path = "../tests/fixtures/tam-haʃaʁav-haɡadol-mono.wav";
    let sound = Sound::from_file(path).expect("Failed to load");
    let formant = sound.to_formant_burg(0.0, 5, 5500.0, 0.025, 50.0);
    let f = formant.formant_values(5);
    for v in f.iter() {
        println!("{}", v);
    }
}
