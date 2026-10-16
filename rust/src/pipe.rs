//! Pipe (stdin JSON → stdout JSON) interface for praatfan_rust.
//!
//! One request runs any number of analyses against a single audio file, so
//! callers (R via processx/jsonlite, shell, etc.) pay the file load and
//! resampling cost once per file rather than once per measure.
//!
//! Request shape:
//!
//! ```json
//! { "wav_path": "audio.wav",
//!   "channel": 0,
//!   "analyses": [
//!     { "type": "pitch_ac", "pitch_floor": 75.0, "pitch_ceiling": 600.0 },
//!     { "type": "formant_burg", "max_formant_hz": 5500.0 },
//!     { "type": "spectral_moments", "times": [0.1, 0.2, 0.3] },
//!     { "type": "spectrogram", "window_length": 0.025 }
//!   ] }
//! ```
//!
//! Response: `{"ok": true, "version": ..., "duration": ..., "sample_rate": ...,
//! "results": [...]}` with one result object per analysis, in request order.
//! Each result echoes its `type` and carries parallel arrays (`times` plus
//! per-analysis value arrays). Non-finite values (absent formants, undefined
//! moments) serialize as JSON `null`, which jsonlite reads back as `NA`.
//!
//! Unknown analysis types and unknown parameter keys are hard errors — a
//! typo'd knob must not silently run with defaults.
//!
//! Transports. The request is the same JSON on each; only where the audio
//! comes from differs:
//!
//! - `handle_request` — the binary: `wav_path` is read from disk (required).
//! - `handle_request_bytes` — the file's bytes are handed in (wasm32, which has
//!   no filesystem). They are decoded exactly as from disk; `wav_path`, when
//!   given, only names the file (format hint from its extension, error text).
//! - `handle_request_samples` — decoded mono samples and their rate are handed
//!   in. `wav_path` and `channel` describe a file and are refused.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::pitch::{FrameTiming, PitchMethod};
use crate::sound::Sound;
use crate::spectrum::Spectrum;

// ── Request ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipeRequest {
    /// Required by the file transport; see the module docs for the others.
    #[serde(default)]
    pub wav_path: Option<String>,
    /// Channel to read from a multi-channel file (0-based). Omit for mono.
    #[serde(default)]
    pub channel: Option<usize>,
    pub analyses: Vec<serde_json::Value>,
}

/// Parameters for `pitch_ac` / `pitch_cc`. Defaults are Praat's command
/// defaults plus the Boersma (1993) tuning values used by
/// `sound_to_pitch_ac` / `sound_to_pitch_cc`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PitchParams {
    /// 0 = auto (0.75/floor for AC, 0.25/floor for CC).
    #[serde(default)]
    time_step: f64,
    #[serde(default = "d_75")]
    pitch_floor: f64,
    #[serde(default = "d_600")]
    pitch_ceiling: f64,
    #[serde(default = "d_0_45")]
    voicing_threshold: f64,
    #[serde(default = "d_0_03")]
    silence_threshold: f64,
    #[serde(default = "d_0_01")]
    octave_cost: f64,
    #[serde(default = "d_0_35")]
    octave_jump_cost: f64,
    #[serde(default = "d_0_14")]
    voiced_unvoiced_cost: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FormantParams {
    /// 0 = auto (25% of window length).
    #[serde(default)]
    time_step: f64,
    #[serde(default = "d_5usize")]
    max_num_formants: usize,
    #[serde(default = "d_5500")]
    max_formant_hz: f64,
    #[serde(default = "d_0_025")]
    window_length: f64,
    #[serde(default = "d_50")]
    pre_emphasis_from: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IntensityParams {
    #[serde(default = "d_100")]
    min_pitch: f64,
    /// 0 = auto (0.8/min_pitch).
    #[serde(default)]
    time_step: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HarmonicityParams {
    #[serde(default = "d_0_01_step")]
    time_step: f64,
    #[serde(default = "d_75")]
    min_pitch: f64,
    #[serde(default = "d_0_1")]
    silence_threshold: f64,
    /// Praat default: 4.5 for the AC method, 1.0 for CC. Resolved per
    /// method when omitted.
    #[serde(default)]
    periods_per_window: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpectralMomentsParams {
    times: Vec<f64>,
    #[serde(default = "d_0_025")]
    window_length: f64,
    #[serde(default = "d_2")]
    power: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BandEnergyParams {
    times: Vec<f64>,
    #[serde(default = "d_0_025")]
    window_length: f64,
    #[serde(default)]
    f_min: f64,
    /// 0 = up to Nyquist.
    #[serde(default)]
    f_max: f64,
}

/// Parameters for `spectrogram`; defaults are Praat's "Sound: To
/// Spectrogram" defaults. See docs/pipe-spectrogram.md for the schema.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpectrogramParams {
    #[serde(default = "d_0_005")]
    window_length: f64,
    /// <= 0 or above Nyquist means Nyquist.
    #[serde(default = "d_5000")]
    max_frequency: f64,
    #[serde(default = "d_0_002")]
    time_step: f64,
    #[serde(default = "d_20")]
    frequency_step: f64,
    #[serde(default = "d_gaussian")]
    window_shape: String,
    #[serde(default = "d_f64le")]
    dtype: String,
    /// Restrict the returned frames (the grid is still the full sound's).
    /// `null` = absent.
    #[serde(default)]
    start_time: Option<f64>,
    #[serde(default)]
    end_time: Option<f64>,
}

fn d_75() -> f64 { 75.0 }
fn d_100() -> f64 { 100.0 }
fn d_600() -> f64 { 600.0 }
fn d_5500() -> f64 { 5500.0 }
fn d_50() -> f64 { 50.0 }
fn d_0_45() -> f64 { 0.45 }
fn d_0_03() -> f64 { 0.03 }
fn d_0_01() -> f64 { 0.01 }
fn d_0_35() -> f64 { 0.35 }
fn d_0_14() -> f64 { 0.14 }
fn d_0_025() -> f64 { 0.025 }
fn d_0_01_step() -> f64 { 0.01 }
fn d_0_1() -> f64 { 0.1 }
fn d_2() -> f64 { 2.0 }
fn d_5usize() -> usize { 5 }
fn d_0_005() -> f64 { 0.005 }
fn d_5000() -> f64 { 5000.0 }
fn d_0_002() -> f64 { 0.002 }
fn d_20() -> f64 { 20.0 }
fn d_gaussian() -> String { "gaussian".to_string() }
fn d_f64le() -> String { "f64le".to_string() }

// ── Response ─────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct PipeResponse {
    pub ok: bool,
    pub version: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<Vec<serde_json::Value>>,
    /// Diagnostics about the request (e.g. an ill-conditioned parameter).
    /// The binary also prints them to stderr; hosts without a stderr (wasm)
    /// read them here.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Serialize)]
struct PitchResult {
    r#type: &'static str,
    times: Vec<f64>,
    /// 0.0 = unvoiced frame.
    frequencies: Vec<f64>,
    strengths: Vec<f64>,
}

#[derive(Serialize)]
struct FormantResult {
    r#type: &'static str,
    times: Vec<f64>,
    n_formants: Vec<usize>,
    /// F1..Fk frequency tracks and B1..Bk bandwidth tracks; `null` where
    /// the formant was not found in a frame.
    #[serde(flatten)]
    tracks: BTreeMap<String, Vec<f64>>,
}

#[derive(Serialize)]
struct ContourResult {
    r#type: &'static str,
    times: Vec<f64>,
    values: Vec<f64>,
}

#[derive(Serialize)]
struct SpectralMomentsResult {
    r#type: &'static str,
    times: Vec<f64>,
    center_of_gravity: Vec<f64>,
    standard_deviation: Vec<f64>,
    skewness: Vec<f64>,
    kurtosis: Vec<f64>,
}

#[derive(Serialize)]
struct SpectrogramResult {
    r#type: &'static str,
    n_times: usize,
    n_freqs: usize,
    /// The steps actually used.
    time_step: f64,
    frequency_step: f64,
    times: Vec<f64>,
    freqs: Vec<f64>,
    layout: &'static str,
    dtype: &'static str,
    /// `n_times * n_freqs` values in `dtype`, little-endian, time-major,
    /// base64 — too many for JSON numbers.
    power_b64: String,
}

// ── Handler ──────────────────────────────────────────────────────────

/// Where a request's audio comes from (see the module docs, "Transports").
enum Audio {
    File,
    Bytes(Vec<u8>),
    Samples(Vec<f64>, f64),
}

/// The file transport: `wav_path` is read from disk.
pub fn handle_request(input: &str) -> String {
    respond(input, Audio::File)
}

/// The bytes transport: `bytes` is the audio file's contents.
pub fn handle_request_bytes(bytes: Vec<u8>, input: &str) -> String {
    respond(input, Audio::Bytes(bytes))
}

/// The samples transport: mono `samples` at `sample_rate` Hz.
pub fn handle_request_samples(samples: Vec<f64>, sample_rate: f64, input: &str) -> String {
    respond(input, Audio::Samples(samples, sample_rate))
}

fn respond(input: &str, audio: Audio) -> String {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle_inner(input, audio)));
    let result = match outcome {
        Ok(r) => r,
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            Err(format!("internal error: {}", msg))
        }
    };
    match result {
        Ok(json) => json,
        Err(e) => {
            let resp = PipeResponse {
                ok: false,
                version: env!("CARGO_PKG_VERSION"),
                error: Some(e),
                duration: None,
                sample_rate: None,
                results: None,
                warnings: Vec::new(),
            };
            serde_json::to_string(&resp)
                .unwrap_or_else(|_| r#"{"ok":false,"error":"serialization failed"}"#.to_string())
        }
    }
}

fn load_sound(req: &PipeRequest, audio: Audio) -> Result<Sound, String> {
    match audio {
        Audio::File => {
            let path = req.wav_path.as_deref().ok_or("wav_path required")?;
            match req.channel {
                Some(ch) => Sound::from_file_channel(path, ch),
                None => Sound::from_file(path),
            }
            .map_err(|e| format!("failed to load {}: {}", path, e))
        }
        Audio::Bytes(bytes) => {
            let name = req.wav_path.as_deref().unwrap_or("<bytes>");
            let ext = req
                .wav_path
                .as_deref()
                .and_then(|p| std::path::Path::new(p).extension())
                .and_then(|e| e.to_str());
            Sound::from_bytes(bytes, ext, req.channel)
                .map_err(|e| format!("failed to load {}: {}", name, e))
        }
        Audio::Samples(samples, sample_rate) => {
            if req.wav_path.is_some() || req.channel.is_some() {
                return Err("wav_path and channel describe a file; the samples transport \
                            takes decoded mono samples and refuses them"
                    .to_string());
            }
            if !(sample_rate.is_finite() && sample_rate > 0.0) {
                return Err(format!("sample_rate must be positive, not {}", sample_rate));
            }
            Ok(Sound::new(ndarray::Array1::from_vec(samples), sample_rate))
        }
    }
}

fn handle_inner(input: &str, audio: Audio) -> Result<String, String> {
    let req: PipeRequest =
        serde_json::from_str(input).map_err(|e| format!("invalid JSON request: {}", e))?;

    let sound = load_sound(&req, audio)?;

    let mut results = Vec::with_capacity(req.analyses.len());
    let mut warnings = Vec::new();
    for (i, spec) in req.analyses.iter().enumerate() {
        results.push(run_analysis(&sound, spec, i, &mut warnings)?);
    }

    let resp = PipeResponse {
        ok: true,
        version: env!("CARGO_PKG_VERSION"),
        error: None,
        duration: Some(sound.duration()),
        sample_rate: Some(sound.sample_rate()),
        results: Some(results),
        warnings,
    };
    serde_json::to_string(&resp).map_err(|e| format!("serialization failed: {}", e))
}

fn run_analysis(
    sound: &Sound,
    spec: &serde_json::Value,
    index: usize,
    warnings: &mut Vec<String>,
) -> Result<serde_json::Value, String> {
    let obj = spec
        .as_object()
        .ok_or_else(|| format!("analyses[{}] must be an object", index))?;
    let atype = obj
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("analyses[{}] must have a string 'type' field", index))?
        .to_string();

    let mut params = obj.clone();
    params.remove("type");
    let params = serde_json::Value::Object(params);
    let ctx = |e: String| format!("analyses[{}] ({}): {}", index, atype, e);

    let result = match atype.as_str() {
        "pitch_ac" => pitch_analysis(sound, params, PitchMethod::Ac, "pitch_ac"),
        "pitch_cc" => pitch_analysis(sound, params, PitchMethod::Cc, "pitch_cc"),
        "formant_burg" => formant_analysis(sound, params),
        "intensity" => intensity_analysis(sound, params),
        "harmonicity_ac" | "harmonicity_cc" => {
            harmonicity_analysis(sound, params, atype == "harmonicity_ac").map(|(r, w)| {
                if let Some(w) = w {
                    warnings.push(format!("analyses[{}] ({}): {}", index, atype, w));
                }
                r
            })
        }
        "spectral_moments" => spectral_moments_analysis(sound, params),
        "band_energy" => band_energy_analysis(sound, params),
        "spectrogram" => spectrogram_analysis(sound, params),
        other => Err(format!(
            "unknown type {:?} (expected pitch_ac, pitch_cc, formant_burg, intensity, \
             harmonicity_ac, harmonicity_cc, spectral_moments, band_energy, spectrogram)",
            other
        )),
    };
    result.map_err(ctx)
}

fn parse<T: for<'de> Deserialize<'de>>(params: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(params).map_err(|e| e.to_string())
}

fn to_value<T: Serialize>(result: &T) -> Result<serde_json::Value, String> {
    serde_json::to_value(result).map_err(|e| format!("serialization failed: {}", e))
}

fn pitch_analysis(
    sound: &Sound,
    params: serde_json::Value,
    method: PitchMethod,
    label: &'static str,
) -> Result<serde_json::Value, String> {
    let p: PitchParams = parse(params)?;
    if !(p.pitch_floor > 0.0 && p.pitch_ceiling > p.pitch_floor) {
        return Err(format!(
            "requires 0 < pitch_floor < pitch_ceiling, got floor={} ceiling={}",
            p.pitch_floor, p.pitch_ceiling
        ));
    }
    let periods_per_window = match method {
        PitchMethod::Ac => 3.0,
        PitchMethod::Cc => 2.0,
    };
    let pitch = crate::pitch::sound_to_pitch_internal(
        sound,
        p.time_step,
        p.pitch_floor,
        p.pitch_ceiling,
        method,
        p.voicing_threshold,
        p.silence_threshold,
        p.octave_cost,
        p.octave_jump_cost,
        p.voiced_unvoiced_cost,
        periods_per_window,
        FrameTiming::Centered,
        true,
        true,
        None,
    );
    to_value(&PitchResult {
        r#type: label,
        times: pitch.times().to_vec(),
        frequencies: pitch.values().to_vec(),
        strengths: pitch.strengths().to_vec(),
    })
}

fn formant_analysis(sound: &Sound, params: serde_json::Value) -> Result<serde_json::Value, String> {
    let p: FormantParams = parse(params)?;
    if p.max_num_formants == 0 || p.max_formant_hz <= 0.0 || p.window_length <= 0.0 {
        return Err(format!(
            "requires max_num_formants >= 1, max_formant_hz > 0, window_length > 0, \
             got {} / {} / {}",
            p.max_num_formants, p.max_formant_hz, p.window_length
        ));
    }
    let formant = sound.to_formant_burg(
        p.time_step,
        p.max_num_formants,
        p.max_formant_hz,
        p.window_length,
        p.pre_emphasis_from,
    );
    let mut tracks = BTreeMap::new();
    for n in 1..=p.max_num_formants {
        tracks.insert(format!("F{}", n), formant.formant_values(n).to_vec());
        tracks.insert(format!("B{}", n), formant.bandwidth_values(n).to_vec());
    }
    to_value(&FormantResult {
        r#type: "formant_burg",
        times: formant.times().to_vec(),
        n_formants: formant.frames().iter().map(|f| f.n_formants()).collect(),
        tracks,
    })
}

fn intensity_analysis(
    sound: &Sound,
    params: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let p: IntensityParams = parse(params)?;
    if p.min_pitch <= 0.0 {
        return Err(format!("requires min_pitch > 0, got {}", p.min_pitch));
    }
    let intensity = sound.to_intensity(p.min_pitch, p.time_step);
    to_value(&ContourResult {
        r#type: "intensity",
        times: intensity.times().to_vec(),
        values: intensity.values().to_vec(),
    })
}

fn harmonicity_analysis(
    sound: &Sound,
    params: serde_json::Value,
    ac: bool,
) -> Result<(serde_json::Value, Option<String>), String> {
    let p: HarmonicityParams = parse(params)?;
    if p.min_pitch <= 0.0 || p.time_step <= 0.0 {
        return Err(format!(
            "requires min_pitch > 0 and time_step > 0, got {} / {}",
            p.min_pitch, p.time_step
        ));
    }
    let periods = p.periods_per_window.unwrap_or(if ac { 4.5 } else { 1.0 });
    let mut warning = None;
    if ac && crate::harmonicity::ac_periods_per_window_is_ill_conditioned(periods) {
        // Diagnostic only. See harmonicity::AC_MIN_PERIODS_PER_WINDOW; this
        // will become a hard error in a future release.
        let w = format!(
            "periods_per_window={:.1} is below {:.1} and the autocorrelation \
             method is ill-conditioned there; frames whose correlation exceeds 1 \
             are reported as null. Use periods_per_window >= {:.1} (standard value \
             4.5), or harmonicity_cc (standard value 1.0). This will become an \
             error in a future release.",
            periods,
            crate::harmonicity::AC_MIN_PERIODS_PER_WINDOW,
            crate::harmonicity::AC_MIN_PERIODS_PER_WINDOW,
        );
        // stdout carries the JSON response, so the binary's copy goes to stderr
        #[cfg(not(target_arch = "wasm32"))]
        eprintln!("warning: {}", w);
        warning = Some(w);
    }
    let harmonicity = if ac {
        sound.to_harmonicity_ac(p.time_step, p.min_pitch, p.silence_threshold, periods)
    } else {
        sound.to_harmonicity_cc(p.time_step, p.min_pitch, p.silence_threshold, periods)
    };
    let result = to_value(&ContourResult {
        r#type: if ac { "harmonicity_ac" } else { "harmonicity_cc" },
        times: harmonicity.times().to_vec(),
        values: harmonicity.values().to_vec(),
    })?;
    Ok((result, warning))
}

/// Window a slice of the sound centered at `t` and transform it — the same
/// clamp-to-signal-edges semantics as Python `Sound.get_spectrum_at_time`.
fn spectrum_at_time(sound: &Sound, t: f64, window_length: f64) -> Spectrum {
    let half = window_length / 2.0;
    let start = (t - half).max(0.0);
    let end = (t + half).min(sound.duration());
    sound.extract_part(start, end).to_spectrum(true)
}

fn check_times(times: &[f64], window_length: f64) -> Result<(), String> {
    if window_length <= 0.0 {
        return Err(format!("requires window_length > 0, got {}", window_length));
    }
    match times.iter().find(|t| !t.is_finite()) {
        Some(t) => Err(format!("times must be finite, got {}", t)),
        None => Ok(()),
    }
}

fn spectral_moments_analysis(
    sound: &Sound,
    params: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let p: SpectralMomentsParams = parse(params)?;
    check_times(&p.times, p.window_length)?;
    let n = p.times.len();
    let mut result = SpectralMomentsResult {
        r#type: "spectral_moments",
        times: p.times.clone(),
        center_of_gravity: Vec::with_capacity(n),
        standard_deviation: Vec::with_capacity(n),
        skewness: Vec::with_capacity(n),
        kurtosis: Vec::with_capacity(n),
    };
    for &t in &p.times {
        let spectrum = spectrum_at_time(sound, t, p.window_length);
        result.center_of_gravity.push(spectrum.get_center_of_gravity(p.power));
        result.standard_deviation.push(spectrum.get_standard_deviation(p.power));
        result.skewness.push(spectrum.get_skewness(p.power));
        result.kurtosis.push(spectrum.get_kurtosis(p.power));
    }
    to_value(&result)
}

fn band_energy_analysis(
    sound: &Sound,
    params: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let p: BandEnergyParams = parse(params)?;
    check_times(&p.times, p.window_length)?;
    let values = p
        .times
        .iter()
        .map(|&t| spectrum_at_time(sound, t, p.window_length).get_band_energy(p.f_min, p.f_max))
        .collect();
    to_value(&ContourResult {
        r#type: "band_energy",
        times: p.times,
        values,
    })
}

fn spectrogram_analysis(
    sound: &Sound,
    params: serde_json::Value,
) -> Result<serde_json::Value, String> {
    use crate::spectrogram::{sound_to_spectrogram_with_shape, WindowShape};

    let p: SpectrogramParams = parse(params)?;
    for (name, v) in [
        ("window_length", p.window_length),
        ("time_step", p.time_step),
        ("frequency_step", p.frequency_step),
    ] {
        if !(v.is_finite() && v > 0.0) {
            return Err(format!("requires a finite {} > 0, got {}", name, v));
        }
    }
    for (name, v) in [
        ("max_frequency", Some(p.max_frequency)),
        ("start_time", p.start_time),
        ("end_time", p.end_time),
    ] {
        if let Some(v) = v.filter(|v| !v.is_finite()) {
            return Err(format!("{} must be finite, got {}", name, v));
        }
    }
    if let (Some(a), Some(b)) = (p.start_time, p.end_time) {
        if a > b {
            return Err(format!("requires start_time <= end_time, got {} > {}", a, b));
        }
    }
    let (shape, physical_window) = match p.window_shape.as_str() {
        // the Gaussian window is twice as long as its effective length
        // (Praat manual, Sound: To Spectrogram...)
        "gaussian" => (WindowShape::Gaussian, 2.0 * p.window_length),
        "hanning" => (WindowShape::Hanning, p.window_length),
        other => return Err(format!("window_shape must be gaussian or hanning, not {:?}", other)),
    };
    let f32_out = match p.dtype.as_str() {
        "f64le" => false,
        "f32le" => true,
        other => return Err(format!("dtype must be f64le or f32le, not {:?}", other)),
    };
    let nyquist = sound.sample_rate() / 2.0;
    let max_frequency = if p.max_frequency <= 0.0 || p.max_frequency > nyquist {
        nyquist
    } else {
        p.max_frequency
    };

    let sg = sound_to_spectrogram_with_shape(
        sound,
        p.window_length,
        max_frequency,
        p.time_step,
        p.frequency_step,
        shape,
    );
    let all_times = sg.times();
    // A sound shorter than the window has no frames (the library still
    // centres one; the pipe reports none).
    let frames: Vec<usize> = if sound.duration() < physical_window {
        Vec::new()
    } else {
        (0..all_times.len())
            .filter(|&i| {
                let t = all_times[i];
                p.start_time.is_none_or(|a| t >= a) && p.end_time.is_none_or(|b| t <= b)
            })
            .collect()
    };

    let values = sg.values();
    let n_freqs = sg.n_freqs();
    let mut bytes = Vec::with_capacity(frames.len() * n_freqs * if f32_out { 4 } else { 8 });
    for &i in &frames {
        for f in 0..n_freqs {
            let v = values[[f, i]];
            if f32_out {
                bytes.extend_from_slice(&(v as f32).to_le_bytes());
            } else {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    to_value(&SpectrogramResult {
        r#type: "spectrogram",
        n_times: frames.len(),
        n_freqs,
        time_step: sg.time_step(),
        frequency_step: sg.freq_step(),
        times: frames.iter().map(|&i| all_times[i]).collect(),
        freqs: sg.frequencies(),
        layout: "time-major",
        dtype: if f32_out { "f32le" } else { "f64le" },
        power_b64: base64_encode(&bytes),
    })
}

/// Standard base64 (RFC 4648 alphabet, padded, no line breaks).
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk.len();
        let b = [chunk[0], if n > 1 { chunk[1] } else { 0 }, if n > 2 { chunk[2] } else { 0 }];
        let word = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for k in 0..4 {
            if k <= n {
                out.push(ALPHABET[(word >> (18 - 6 * k) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/fixtures/one_two_three_four_five.wav"
    );

    fn request(analyses: &str) -> serde_json::Value {
        let input = format!(r#"{{"wav_path": {:?}, "analyses": {}}}"#, FIXTURE, analyses);
        serde_json::from_str(&handle_request(&input)).expect("response must be valid JSON")
    }

    fn arr_len(result: &serde_json::Value, key: &str) -> usize {
        result[key].as_array().unwrap().len()
    }

    #[test]
    fn batch_runs_all_analysis_types() {
        let resp = request(
            r#"[
                {"type": "pitch_ac"},
                {"type": "pitch_cc", "pitch_floor": 100.0, "pitch_ceiling": 400.0},
                {"type": "formant_burg", "max_num_formants": 4},
                {"type": "intensity"},
                {"type": "harmonicity_ac"},
                {"type": "harmonicity_cc"},
                {"type": "spectral_moments", "times": [0.5, 1.0]},
                {"type": "band_energy", "times": [0.5, 1.0], "f_min": 0.0, "f_max": 1000.0}
            ]"#,
        );
        assert_eq!(resp["ok"], true, "{}", resp);
        assert!(resp["duration"].as_f64().unwrap() > 0.0);
        let results = resp["results"].as_array().unwrap();
        assert_eq!(results.len(), 8);

        // Parallel arrays stay parallel, per analysis.
        let pitch = &results[0];
        assert_eq!(pitch["type"], "pitch_ac");
        let n = arr_len(pitch, "times");
        assert!(n > 0);
        assert_eq!(arr_len(pitch, "frequencies"), n);
        assert_eq!(arr_len(pitch, "strengths"), n);

        let formant = &results[2];
        let nf = arr_len(formant, "times");
        for key in ["n_formants", "F1", "F4", "B1", "B4"] {
            assert_eq!(arr_len(formant, key), nf, "length mismatch for {}", key);
        }
        assert!(formant.get("F5").is_none(), "only max_num_formants tracks");

        let moments = &results[6];
        assert_eq!(arr_len(moments, "center_of_gravity"), 2);
        assert!(moments["center_of_gravity"][0].as_f64().unwrap() > 0.0);

        let band = &results[7];
        assert!(band["values"][0].as_f64().unwrap() >= 0.0);
    }

    /// Absent formants are NaN internally; on the wire they must be `null`
    /// (jsonlite → NA), and the response must never contain a bare NaN token.
    #[test]
    fn non_finite_values_serialize_as_null() {
        let input = format!(
            r#"{{"wav_path": {:?}, "analyses": [{{"type": "formant_burg"}}]}}"#,
            FIXTURE
        );
        let raw = handle_request(&input);
        assert!(!raw.contains("NaN"), "raw JSON must not contain NaN");
        let resp: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(resp["ok"], true);
    }

    #[test]
    fn unknown_type_and_typoed_param_are_errors() {
        let resp = request(r#"[{"type": "spectrum"}]"#);
        assert_eq!(resp["ok"], false);
        assert!(resp["error"].as_str().unwrap().contains("unknown type"));

        let resp = request(r#"[{"type": "pitch_ac", "pitch_flor": 75.0}]"#);
        assert_eq!(resp["ok"], false);
        let err = resp["error"].as_str().unwrap();
        assert!(err.contains("pitch_flor"), "error should name the bad key: {}", err);
        assert!(err.contains("analyses[0]"), "error should locate the analysis: {}", err);
    }

    #[test]
    fn missing_file_reports_error_not_panic() {
        let resp: serde_json::Value = serde_json::from_str(&handle_request(
            r#"{"wav_path": "/nonexistent.wav", "analyses": [{"type": "intensity"}]}"#,
        ))
        .unwrap();
        assert_eq!(resp["ok"], false);
        assert!(resp["error"].as_str().unwrap().contains("/nonexistent.wav"));
    }

    /// Omitted pitch parameters must resolve to the same values as the
    /// library's own AC entry point (`sound_to_pitch_ac`).
    #[test]
    fn pitch_defaults_match_library_entry_point() {
        let resp = request(r#"[{"type": "pitch_ac"}]"#);
        let sound = Sound::from_file(FIXTURE).unwrap();
        let expected = sound.to_pitch_ac(0.0, 75.0, 600.0);
        let freqs = resp["results"][0]["frequencies"].as_array().unwrap();
        assert_eq!(freqs.len(), expected.n_frames());
        for (got, want) in freqs.iter().zip(expected.values().iter()) {
            assert_eq!(got.as_f64().unwrap(), *want);
        }
    }

    /// Windows clamped at the signal edges must still produce moments
    /// (mirrors Python get_spectrum_at_time clamping).
    #[test]
    fn edge_windows_are_clamped_not_fatal() {
        let resp = request(r#"[{"type": "spectral_moments", "times": [0.0, 0.001]}]"#);
        assert_eq!(resp["ok"], true, "{}", resp);
        assert_eq!(arr_len(&resp["results"][0], "center_of_gravity"), 2);
    }

    const ALL_ANALYSES: &str = r#"[
        {"type": "pitch_ac"}, {"type": "pitch_cc"}, {"type": "formant_burg"},
        {"type": "intensity"}, {"type": "harmonicity_ac"}, {"type": "harmonicity_cc"},
        {"type": "spectral_moments", "times": [0.2, 0.5]},
        {"type": "band_energy", "times": [0.2, 0.5], "f_max": 1000},
        {"type": "spectrogram"}]"#;

    #[test]
    fn bytes_and_samples_transports_match_file_transport() {
        let with_path = format!(r#"{{"wav_path": {:?}, "analyses": {}}}"#, FIXTURE, ALL_ANALYSES);
        let without = format!(r#"{{"analyses": {}}}"#, ALL_ANALYSES);
        let from_file = handle_request(&with_path);
        assert!(from_file.starts_with(r#"{"ok":true"#), "{}", from_file);

        let bytes = std::fs::read(FIXTURE).unwrap();
        assert_eq!(handle_request_bytes(bytes.clone(), &with_path), from_file);
        assert_eq!(handle_request_bytes(bytes, &without), from_file);

        let sound = Sound::from_file(FIXTURE).unwrap();
        let samples = sound.samples().to_vec();
        assert_eq!(handle_request_samples(samples, sound.sample_rate(), &without), from_file);
    }

    #[test]
    fn samples_transport_refuses_file_fields() {
        let input = format!(r#"{{"wav_path": {:?}, "analyses": []}}"#, FIXTURE);
        let resp: serde_json::Value =
            serde_json::from_str(&handle_request_samples(vec![0.0; 100], 16000.0, &input)).unwrap();
        assert_eq!(resp["ok"], false);
        let resp: serde_json::Value =
            serde_json::from_str(&handle_request_samples(vec![0.0; 100], 0.0, r#"{"analyses": []}"#)).unwrap();
        assert_eq!(resp["ok"], false);
    }

    /// Undecodable bytes are named by wav_path, or "<bytes>" without one
    /// (docs/pipe-spectrogram.md section 3).
    #[test]
    fn undecodable_bytes_are_named() {
        let resp: serde_json::Value = serde_json::from_str(&handle_request_bytes(
            vec![1, 2, 3, 4], r#"{"analyses": [{"type": "intensity"}]}"#)).unwrap();
        assert_eq!(resp["ok"], false);
        assert!(resp["error"].as_str().unwrap().contains("<bytes>"), "{}", resp);
        let resp: serde_json::Value = serde_json::from_str(&handle_request_bytes(
            vec![1, 2, 3, 4], r#"{"wav_path": "a.wav", "analyses": []}"#)).unwrap();
        assert!(resp["error"].as_str().unwrap().contains("a.wav"), "{}", resp);
    }

    #[test]
    fn file_transport_requires_wav_path() {
        let resp: serde_json::Value =
            serde_json::from_str(&handle_request(r#"{"analyses": []}"#)).unwrap();
        assert_eq!(resp["ok"], false);
        assert!(resp["error"].as_str().unwrap().contains("wav_path required"));
    }

    #[test]
    fn ill_conditioned_ppw_is_reported_in_the_response() {
        let resp = request(r#"[{"type": "harmonicity_ac", "periods_per_window": 2.0},
                              {"type": "harmonicity_ac"}]"#);
        assert_eq!(resp["ok"], true, "{}", resp);
        let w = resp["warnings"].as_array().unwrap();
        assert_eq!(w.len(), 1);
        assert!(w[0].as_str().unwrap().starts_with("analyses[0] (harmonicity_ac): periods_per_window=2.0"));
        assert!(request(r#"[{"type": "harmonicity_ac"}]"#).get("warnings").is_none());
    }

    fn b64_decode(s: &str) -> Vec<u8> {
        const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = Vec::new();
        for chunk in s.as_bytes().chunks(4) {
            let mut w = 0u32;
            let mut n = 0;
            for &c in chunk {
                w <<= 6;
                if c != b'=' {
                    w |= A.iter().position(|&a| a == c).unwrap() as u32;
                    n += 1;
                }
            }
            w <<= 6 * (4 - chunk.len()) as u32;
            out.extend_from_slice(&w.to_be_bytes()[1..n]);
        }
        out
    }

    #[test]
    fn base64_matches_rfc4648_vectors() {
        for (raw, enc) in [("", ""), ("f", "Zg=="), ("fo", "Zm8="), ("foo", "Zm9v"),
                           ("foob", "Zm9vYg=="), ("fooba", "Zm9vYmE="), ("foobar", "Zm9vYmFy")] {
            assert_eq!(base64_encode(raw.as_bytes()), enc);
            assert_eq!(b64_decode(enc), raw.as_bytes());
        }
    }

    /// The pipe's spectrogram is the library's, laid out time-major.
    #[test]
    fn spectrogram_shape_and_values() {
        let resp = request(r#"[{"type": "spectrogram", "window_length": 0.025}]"#);
        assert_eq!(resp["ok"], true, "{}", resp);
        let r = &resp["results"][0];
        let (nt, nf) = (r["n_times"].as_u64().unwrap() as usize, r["n_freqs"].as_u64().unwrap() as usize);
        assert_eq!((arr_len(r, "times"), arr_len(r, "freqs")), (nt, nf));
        assert_eq!((r["layout"].as_str(), r["dtype"].as_str()), (Some("time-major"), Some("f64le")));
        let bytes = b64_decode(r["power_b64"].as_str().unwrap());
        assert_eq!(bytes.len(), nt * nf * 8);

        let sound = Sound::from_file(FIXTURE).unwrap();
        let sg = crate::spectrogram::sound_to_spectrogram(&sound, 0.025, 5000.0, 0.002, 20.0);
        assert_eq!((sg.n_times(), sg.n_freqs()), (nt, nf));
        assert_eq!(r["time_step"].as_f64().unwrap(), sg.time_step());
        assert_eq!(r["frequency_step"].as_f64().unwrap(), sg.freq_step());
        for (t, f) in [(0, 0), (nt / 2, nf / 3), (nt - 1, nf - 1)] {
            let k = (t * nf + f) * 8;
            let got = f64::from_le_bytes(bytes[k..k + 8].try_into().unwrap());
            assert_eq!(got, sg.values()[[f, t]]);
        }
    }

    #[test]
    fn spectrogram_range_dtype_clamp_and_empty() {
        let full = &request(r#"[{"type": "spectrogram"}]"#)["results"][0];
        let part = &request(r#"[{"type": "spectrogram", "start_time": 0.5, "end_time": 0.6,
                                 "dtype": "f32le"}]"#)["results"][0];
        let times = |r: &serde_json::Value| -> Vec<f64> {
            r["times"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
        };
        let (tf, tp) = (times(full), times(part));
        assert!(!tp.is_empty() && tp.iter().all(|&t| (0.5..=0.6).contains(&t)));
        let first = tf.iter().position(|&t| t == tp[0]).expect("range keeps the full grid");
        assert_eq!(&tf[first..first + tp.len()], &tp[..]);
        let nf = part["n_freqs"].as_u64().unwrap() as usize;
        assert_eq!(b64_decode(part["power_b64"].as_str().unwrap()).len(), tp.len() * nf * 4);

        // null = absent
        let nulls = &request(r#"[{"type": "spectrogram", "start_time": null, "end_time": null}]"#)
            ["results"][0];
        assert_eq!(nulls["power_b64"], full["power_b64"]);

        // max_frequency <= 0 or above Nyquist: clamped to Nyquist
        let nyq = &request(r#"[{"type": "spectrogram", "max_frequency": 12000}]"#)["results"][0];
        for mf in ["0", "-1", "99999"] {
            let r = &request(&format!(r#"[{{"type": "spectrogram", "max_frequency": {}}}]"#, mf))
                ["results"][0];
            assert_eq!(r["freqs"], nyq["freqs"], "max_frequency {}", mf);
        }

        // a range selecting nothing, and a sound shorter than the window: empty, ok
        let none = &request(r#"[{"type": "spectrogram", "start_time": 5.0}]"#)["results"][0];
        assert_eq!((none["n_times"].as_u64(), none["power_b64"].as_str()), (Some(0), Some("")));
        let short = serde_json::from_str::<serde_json::Value>(&handle_request_samples(
            vec![0.1; 100], 16000.0, r#"{"analyses": [{"type": "spectrogram"}]}"#)).unwrap();
        assert_eq!(short["ok"], true, "{}", short);
        let r = &short["results"][0];
        assert_eq!((r["n_times"].as_u64(), r["times"].as_array().map(|a| a.len())), (Some(0), Some(0)));
        assert!(r["n_freqs"].as_u64().unwrap() > 0);
    }

    #[test]
    fn spectrogram_errors() {
        for bad in [r#""window_length": 0"#, r#""time_step": -1"#, r#""frequency_step": 0"#,
                    r#""start_time": 0.6, "end_time": 0.5"#, r#""window_shape": "hamming""#,
                    r#""dtype": "f16""#, r#""bogus": 1"#] {
            let resp = request(&format!(r#"[{{"type": "spectrogram", {}}}]"#, bad));
            assert_eq!(resp["ok"], false, "{}", bad);
        }
    }
}
