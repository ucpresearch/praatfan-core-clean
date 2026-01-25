/* tslint:disable */
/* eslint-disable */

/**
 * Formant analysis result.
 */
export class Formant {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get bandwidth values for a specific formant number.
     */
    bandwidth_values(formant_num: number): Float64Array;
    /**
     * Get formant frequency values for a specific formant number (1=F1, 2=F2, etc.)
     */
    formant_values(formant_num: number): Float64Array;
    /**
     * Get bandwidth at a specific frame and formant number.
     */
    get_bandwidth_at_frame(frame: number, formant_num: number): number;
    /**
     * Get time at a specific frame index.
     */
    get_time_from_frame(frame: number): number;
    /**
     * Get formant frequency at a specific frame and formant number.
     */
    get_value_at_frame(frame: number, formant_num: number): number;
    /**
     * Get the maximum number of formants per frame.
     */
    max_num_formants(): number;
    /**
     * Get the number of frames.
     */
    n_frames(): number;
    /**
     * Get the time step between frames.
     */
    time_step(): number;
    /**
     * Get time points for all frames.
     */
    times(): Float64Array;
}

/**
 * Harmonicity (HNR) analysis result.
 */
export class Harmonicity {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get time at a specific frame index.
     */
    get_time_from_frame(frame: number): number;
    /**
     * Get HNR at a specific frame index in dB.
     */
    get_value_in_frame(frame: number): number;
    /**
     * Get the number of frames.
     */
    n_frames(): number;
    /**
     * Get the time step between frames.
     */
    time_step(): number;
    /**
     * Get time points for all frames.
     */
    times(): Float64Array;
    /**
     * Get HNR values for all frames in dB. Returns -200 for silent/unvoiced frames.
     */
    values(): Float64Array;
}

/**
 * Intensity analysis result.
 */
export class Intensity {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get the maximum intensity value in dB.
     */
    get_maximum(): number;
    /**
     * Get the mean intensity value in dB.
     */
    get_mean(): number;
    /**
     * Get the minimum intensity value in dB.
     */
    get_minimum(): number;
    /**
     * Get time at a specific frame index.
     */
    get_time_from_frame(frame: number): number;
    /**
     * Get intensity at a specific frame index in dB.
     */
    get_value_in_frame(frame: number): number;
    /**
     * Get the number of frames.
     */
    n_frames(): number;
    /**
     * Get the time step between frames.
     */
    time_step(): number;
    /**
     * Get time points for all frames.
     */
    times(): Float64Array;
    /**
     * Get intensity values for all frames in dB.
     */
    values(): Float64Array;
}

/**
 * Pitch (F0) analysis result.
 */
export class Pitch {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get time at a specific frame index.
     */
    get_time_from_frame(frame: number): number;
    /**
     * Get F0 at a specific frame index. Returns NaN for unvoiced frames.
     */
    get_value_in_frame(frame: number): number;
    /**
     * Get the number of frames.
     */
    n_frames(): number;
    /**
     * Get the pitch ceiling in Hz.
     */
    pitch_ceiling(): number;
    /**
     * Get the pitch floor in Hz.
     */
    pitch_floor(): number;
    /**
     * Get strength (correlation) values for all frames.
     */
    strengths(): Float64Array;
    /**
     * Get the time step between frames.
     */
    time_step(): number;
    /**
     * Get time points for all frames.
     */
    times(): Float64Array;
    /**
     * Get F0 values for all frames.
     *
     * Returns Float64Array with F0 in Hz for voiced frames, NaN for unvoiced.
     */
    values(): Float64Array;
}

/**
 * Audio samples with sample rate.
 *
 * This is the main type for acoustic analysis in WASM. Create a Sound
 * from raw samples, then call analysis methods.
 */
export class Sound {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get the duration in seconds.
     */
    duration(): number;
    /**
     * Create a Sound from WAV file bytes.
     *
     * Supports mono WAV files. For stereo files, use `from_wav_channel()`.
     *
     * # Arguments
     *
     * * `wav_bytes` - WAV file contents as Uint8Array
     *
     * # Errors
     *
     * Throws if the WAV file is invalid or has multiple channels.
     */
    static from_wav(wav_bytes: Uint8Array): Sound;
    /**
     * Create a Sound from a specific channel of a WAV file.
     *
     * # Arguments
     *
     * * `wav_bytes` - WAV file contents as Uint8Array
     * * `channel` - Channel index (0 = left, 1 = right, etc.)
     */
    static from_wav_channel(wav_bytes: Uint8Array, channel: number): Sound;
    /**
     * Get the number of samples.
     */
    n_samples(): number;
    /**
     * Create a Sound from raw audio samples.
     *
     * # Arguments
     *
     * * `samples` - Audio samples as Float64Array, typically in range [-1, 1]
     * * `sample_rate` - Sample rate in Hz (e.g., 44100, 48000)
     *
     * # Example (JavaScript)
     *
     * ```javascript
     * const samples = new Float64Array(audioContext.length);
     * const sound = new Sound(samples, 44100);
     * ```
     */
    constructor(samples: Float64Array, sample_rate: number);
    /**
     * Get the sample rate in Hz.
     */
    sample_rate(): number;
    /**
     * Get the audio samples as a Float64Array.
     */
    samples(): Float64Array;
    /**
     * Compute formants using Burg's LPC method.
     *
     * # Arguments
     *
     * * `time_step` - Time step in seconds (0 = auto)
     * * `max_num_formants` - Maximum number of formants (typically 5)
     * * `max_formant_hz` - Maximum formant frequency (5500 for male, 5000 for female)
     * * `window_length` - Window length in seconds (typically 0.025)
     * * `pre_emphasis_from` - Pre-emphasis frequency in Hz (typically 50)
     */
    to_formant_burg(time_step: number, max_num_formants: number, max_formant_hz: number, window_length: number, pre_emphasis_from: number): Formant;
    /**
     * Compute harmonicity (HNR) using autocorrelation method.
     *
     * # Arguments
     *
     * * `time_step` - Time step in seconds
     * * `min_pitch` - Minimum pitch in Hz
     * * `silence_threshold` - Silence threshold (0-1)
     * * `periods_per_window` - Number of periods per window (typically 4.5)
     */
    to_harmonicity_ac(time_step: number, min_pitch: number, silence_threshold: number, periods_per_window: number): Harmonicity;
    /**
     * Compute harmonicity (HNR) using cross-correlation method.
     */
    to_harmonicity_cc(time_step: number, min_pitch: number, silence_threshold: number, periods_per_window: number): Harmonicity;
    /**
     * Compute intensity contour.
     *
     * # Arguments
     *
     * * `min_pitch` - Minimum pitch in Hz (determines window size)
     * * `time_step` - Time step in seconds (0 = auto)
     */
    to_intensity(min_pitch: number, time_step: number): Intensity;
    /**
     * Compute pitch (F0) contour using autocorrelation method.
     *
     * # Arguments
     *
     * * `time_step` - Time step in seconds (0 = auto)
     * * `pitch_floor` - Minimum pitch in Hz (e.g., 75)
     * * `pitch_ceiling` - Maximum pitch in Hz (e.g., 600)
     */
    to_pitch_ac(time_step: number, pitch_floor: number, pitch_ceiling: number): Pitch;
    /**
     * Compute pitch (F0) contour using cross-correlation method.
     */
    to_pitch_cc(time_step: number, pitch_floor: number, pitch_ceiling: number): Pitch;
    /**
     * Compute spectrogram (time-frequency representation).
     *
     * # Arguments
     *
     * * `window_length` - Window length in seconds
     * * `max_frequency` - Maximum frequency in Hz
     * * `time_step` - Time step in seconds
     * * `frequency_step` - Frequency step in Hz
     */
    to_spectrogram(window_length: number, max_frequency: number, time_step: number, frequency_step: number): Spectrogram;
    /**
     * Compute spectrum (single-frame FFT).
     *
     * # Arguments
     *
     * * `fast` - If true, use power-of-2 FFT size for speed
     */
    to_spectrum(fast: boolean): Spectrum;
}

/**
 * Spectrogram (time-frequency) analysis result.
 */
export class Spectrogram {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get the maximum frequency in Hz.
     */
    freq_max(): number;
    /**
     * Get the minimum frequency in Hz.
     */
    freq_min(): number;
    /**
     * Get the frequency step between bins.
     */
    freq_step(): number;
    /**
     * Get frequency points for all bins.
     */
    frequencies(): Float64Array;
    /**
     * Get frequency at a specific bin index.
     */
    get_freq_from_bin(bin: number): number;
    /**
     * Get time at a specific frame index.
     */
    get_time_from_frame(frame: number): number;
    /**
     * Get power value at a specific time frame and frequency bin.
     */
    get_value_at(time_frame: number, freq_bin: number): number;
    /**
     * Get the number of frequency bins.
     */
    n_freqs(): number;
    /**
     * Get the number of time frames.
     */
    n_times(): number;
    /**
     * Get the maximum time in seconds.
     */
    time_max(): number;
    /**
     * Get the minimum time in seconds.
     */
    time_min(): number;
    /**
     * Get the time step between frames.
     */
    time_step(): number;
    /**
     * Get time points for all frames.
     */
    times(): Float64Array;
    /**
     * Get all power values as a flat array (row-major: freq × time).
     */
    values(): Float64Array;
}

/**
 * Spectrum (single-frame FFT) analysis result.
 */
export class Spectrum {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get the frequency resolution (bin width) in Hz.
     */
    df(): number;
    /**
     * Get the energy in a frequency band.
     */
    get_band_energy(f_min: number, f_max: number): number;
    /**
     * Get the center of gravity (spectral centroid) in Hz.
     */
    get_center_of_gravity(power: number): number;
    /**
     * Get frequency for a bin index.
     */
    get_freq_from_bin(bin: number): number;
    /**
     * Get the kurtosis of the spectrum.
     */
    get_kurtosis(power: number): number;
    /**
     * Get the skewness of the spectrum.
     */
    get_skewness(power: number): number;
    /**
     * Get the standard deviation (spectral spread) in Hz.
     */
    get_standard_deviation(power: number): number;
    /**
     * Get the imaginary parts of the spectrum.
     */
    imag(): Float64Array;
    /**
     * Get the maximum frequency in Hz.
     */
    max_frequency(): number;
    /**
     * Get the number of frequency bins.
     */
    n_bins(): number;
    /**
     * Get the real parts of the spectrum.
     */
    real(): Float64Array;
}

/**
 * Initialize the WASM module.
 *
 * This sets up panic hooks for better error messages in the browser console.
 * Call this once before using any other functions.
 */
export function init(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_formant_free: (a: number, b: number) => void;
    readonly __wbg_harmonicity_free: (a: number, b: number) => void;
    readonly __wbg_intensity_free: (a: number, b: number) => void;
    readonly __wbg_pitch_free: (a: number, b: number) => void;
    readonly __wbg_sound_free: (a: number, b: number) => void;
    readonly __wbg_spectrogram_free: (a: number, b: number) => void;
    readonly __wbg_spectrum_free: (a: number, b: number) => void;
    readonly formant_bandwidth_values: (a: number, b: number) => [number, number];
    readonly formant_formant_values: (a: number, b: number) => [number, number];
    readonly formant_get_bandwidth_at_frame: (a: number, b: number, c: number) => number;
    readonly formant_get_time_from_frame: (a: number, b: number) => number;
    readonly formant_get_value_at_frame: (a: number, b: number, c: number) => number;
    readonly formant_max_num_formants: (a: number) => number;
    readonly formant_n_frames: (a: number) => number;
    readonly formant_time_step: (a: number) => number;
    readonly formant_times: (a: number) => [number, number];
    readonly harmonicity_get_time_from_frame: (a: number, b: number) => number;
    readonly harmonicity_get_value_in_frame: (a: number, b: number) => number;
    readonly harmonicity_n_frames: (a: number) => number;
    readonly harmonicity_time_step: (a: number) => number;
    readonly harmonicity_times: (a: number) => [number, number];
    readonly harmonicity_values: (a: number) => [number, number];
    readonly init: () => void;
    readonly intensity_get_maximum: (a: number) => number;
    readonly intensity_get_mean: (a: number) => number;
    readonly intensity_get_minimum: (a: number) => number;
    readonly intensity_get_time_from_frame: (a: number, b: number) => number;
    readonly intensity_get_value_in_frame: (a: number, b: number) => number;
    readonly intensity_n_frames: (a: number) => number;
    readonly intensity_time_step: (a: number) => number;
    readonly intensity_times: (a: number) => [number, number];
    readonly intensity_values: (a: number) => [number, number];
    readonly pitch_get_time_from_frame: (a: number, b: number) => number;
    readonly pitch_get_value_in_frame: (a: number, b: number) => number;
    readonly pitch_n_frames: (a: number) => number;
    readonly pitch_pitch_ceiling: (a: number) => number;
    readonly pitch_pitch_floor: (a: number) => number;
    readonly pitch_strengths: (a: number) => [number, number];
    readonly pitch_time_step: (a: number) => number;
    readonly pitch_times: (a: number) => [number, number];
    readonly pitch_values: (a: number) => [number, number];
    readonly sound_duration: (a: number) => number;
    readonly sound_from_wav: (a: number, b: number) => [number, number, number];
    readonly sound_from_wav_channel: (a: number, b: number, c: number) => [number, number, number];
    readonly sound_n_samples: (a: number) => number;
    readonly sound_new: (a: number, b: number, c: number) => number;
    readonly sound_sample_rate: (a: number) => number;
    readonly sound_samples: (a: number) => [number, number];
    readonly sound_to_formant_burg: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly sound_to_harmonicity_ac: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly sound_to_harmonicity_cc: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly sound_to_intensity: (a: number, b: number, c: number) => number;
    readonly sound_to_pitch_ac: (a: number, b: number, c: number, d: number) => number;
    readonly sound_to_pitch_cc: (a: number, b: number, c: number, d: number) => number;
    readonly sound_to_spectrogram: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly sound_to_spectrum: (a: number, b: number) => number;
    readonly spectrogram_freq_max: (a: number) => number;
    readonly spectrogram_freq_min: (a: number) => number;
    readonly spectrogram_freq_step: (a: number) => number;
    readonly spectrogram_frequencies: (a: number) => [number, number];
    readonly spectrogram_get_freq_from_bin: (a: number, b: number) => number;
    readonly spectrogram_get_time_from_frame: (a: number, b: number) => number;
    readonly spectrogram_get_value_at: (a: number, b: number, c: number) => number;
    readonly spectrogram_n_freqs: (a: number) => number;
    readonly spectrogram_n_times: (a: number) => number;
    readonly spectrogram_time_max: (a: number) => number;
    readonly spectrogram_time_min: (a: number) => number;
    readonly spectrogram_time_step: (a: number) => number;
    readonly spectrogram_times: (a: number) => [number, number];
    readonly spectrogram_values: (a: number) => [number, number];
    readonly spectrum_df: (a: number) => number;
    readonly spectrum_get_band_energy: (a: number, b: number, c: number) => number;
    readonly spectrum_get_center_of_gravity: (a: number, b: number) => number;
    readonly spectrum_get_freq_from_bin: (a: number, b: number) => number;
    readonly spectrum_get_kurtosis: (a: number, b: number) => number;
    readonly spectrum_get_skewness: (a: number, b: number) => number;
    readonly spectrum_get_standard_deviation: (a: number, b: number) => number;
    readonly spectrum_imag: (a: number) => [number, number];
    readonly spectrum_max_frequency: (a: number) => number;
    readonly spectrum_n_bins: (a: number) => number;
    readonly spectrum_real: (a: number) => [number, number];
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
