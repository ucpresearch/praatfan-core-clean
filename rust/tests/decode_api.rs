//! The public decoder (`decode_bytes` / `decode_file`) returns exactly what
//! `Sound`'s loaders decode, plus every channel when none is selected.

use praatfan_rust::{decode_bytes, decode_file, Sound};

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/fixtures/");

#[test]
fn stereo_interleaved_and_selected_channel() {
    let path = format!("{FIX}tam-haʃaʁav-haɡadol.wav");
    let all = decode_file(&path, None).unwrap();
    assert_eq!(all.channels, 2);
    assert!(!all.lossy && !all.sphere);
    assert_eq!(all.codec, "pcm_s16le");
    assert_eq!(all.samples.len() % 2, 0);
    for ch in 0..2 {
        let one = decode_file(&path, Some(ch)).unwrap();
        assert_eq!(one.channels, 2);
        let deint: Vec<f64> = all.samples.iter().skip(ch).step_by(2).copied().collect();
        assert_eq!(one.samples, deint);
        let snd = Sound::from_file_channel(&path, ch).unwrap();
        assert_eq!(snd.samples().to_vec(), one.samples);
        assert_eq!(snd.sample_rate(), one.sample_rate as f64);
    }
    assert!(decode_file(&path, Some(2)).is_err());
}

#[test]
fn bytes_match_file_for_each_fixture() {
    for (name, codec) in [
        ("one_two_three_four_five.wav", "pcm_s16le"),
        ("one_two_three_four_five_32float.wav", "pcm_f32le"),
        ("one_two_three_four_five.flac", "flac"),
    ] {
        let path = format!("{FIX}{name}");
        let bytes = std::fs::read(&path).unwrap();
        let ext = name.rsplit('.').next();
        let from_file = decode_file(&path, None).unwrap();
        let from_slice = decode_bytes(bytes.as_slice(), ext, None).unwrap();
        let unhinted = decode_bytes(bytes.clone(), None, None).unwrap();
        assert_eq!(from_file.codec, codec, "{name}");
        assert!(!from_file.lossy, "{name}");
        for d in [&from_slice, &unhinted] {
            assert_eq!(d.samples, from_file.samples, "{name}");
            assert_eq!((d.sample_rate, d.channels), (from_file.sample_rate, 1), "{name}");
        }
        let snd = Sound::from_bytes(bytes, ext, None).unwrap();
        assert_eq!(snd.samples().to_vec(), from_file.samples, "{name}");
    }
}

#[test]
fn sphere_bytes_are_flagged() {
    let x: Vec<i16> = (0..4000).map(|i| ((i * 37) % 20000 - 10000) as i16).collect();
    let mut b = format!(
        "NIST_1A\n   1024\nsample_count -i {}\nsample_rate -i 16000\nchannel_count -i 1\n\
         sample_n_bytes -i 2\nsample_byte_format -s2 01\nsample_coding -s3 pcm\nend_head\n",
        x.len()
    )
    .into_bytes();
    b.resize(1024, b' ');
    for s in &x {
        b.extend_from_slice(&s.to_le_bytes());
    }
    let d = decode_bytes(b, Some("wav"), None).unwrap();
    assert!(d.sphere && !d.lossy);
    assert_eq!((d.sample_rate, d.channels), (16000, 1));
    let want: Vec<f64> = x.iter().map(|&s| s as f64 / 32768.0).collect();
    assert_eq!(d.samples, want);
}

#[test]
fn garbage_is_an_error() {
    assert!(decode_bytes(vec![0u8; 100], None, None).is_err());
}
