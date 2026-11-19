//! NIST SPHERE bytes decode by content, whatever the extension hint (TIMIT
//! ships SPHERE files named `.WAV`). Covers `Sound::from_file` and
//! `Sound::from_bytes`, the path the wasm pipe (`pipe_request_bytes`) uses.

use praatfan_rust::Sound;

fn sphere(samples: &[i16], sr: u32, big_endian: bool) -> Vec<u8> {
    let fmt = if big_endian { "10" } else { "01" };
    let mut hdr = format!(
        "NIST_1A\n   1024\nsample_count -i {}\nsample_rate -i {}\nchannel_count -i 1\n\
         sample_n_bytes -i 2\nsample_byte_format -s2 {}\nsample_coding -s3 pcm\nend_head\n",
        samples.len(),
        sr,
        fmt
    )
    .into_bytes();
    hdr.resize(1024, b' ');
    for &s in samples {
        hdr.extend_from_slice(&if big_endian { s.to_be_bytes() } else { s.to_le_bytes() });
    }
    hdr
}

fn pcm() -> Vec<i16> {
    (0..8000).map(|i| ((i * 37) % 20000 - 10000) as i16).collect()
}

#[test]
fn sphere_named_wav_decodes() {
    let x = pcm();
    let want: Vec<f64> = x.iter().map(|&s| s as f64 / 32768.0).collect();
    let dir = std::env::temp_dir().join(format!("pf-sphere-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for big in [false, true] {
        let bytes = sphere(&x, 16000, big);
        for name in ["a.wav", "A.WAV", "a.sph"] {
            let p = dir.join(name);
            std::fs::write(&p, &bytes).unwrap();
            let s = Sound::from_file(&p).unwrap_or_else(|e| panic!("{name} big={big}: {e}"));
            assert_eq!(s.samples().to_vec(), want, "{name} big={big}");
        }
        for ext in [Some("wav"), Some("sph"), None] {
            let s = Sound::from_bytes(bytes.clone(), ext, None)
                .unwrap_or_else(|e| panic!("{ext:?} big={big}: {e}"));
            assert_eq!(s.samples().to_vec(), want, "{ext:?} big={big}");
            assert_eq!(s.sample_rate(), 16000.0);
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}
