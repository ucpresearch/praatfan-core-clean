/* @ts-self-types="./praatfan.d.ts" */

/**
 * Formant analysis result.
 */
export class Formant {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Formant.prototype);
        obj.__wbg_ptr = ptr;
        FormantFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FormantFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_formant_free(ptr, 0);
    }
    /**
     * Get bandwidth values for a specific formant number.
     * @param {number} formant_num
     * @returns {Float64Array}
     */
    bandwidth_values(formant_num) {
        const ret = wasm.formant_bandwidth_values(this.__wbg_ptr, formant_num);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get formant frequency values for a specific formant number (1=F1, 2=F2, etc.)
     * @param {number} formant_num
     * @returns {Float64Array}
     */
    formant_values(formant_num) {
        const ret = wasm.formant_formant_values(this.__wbg_ptr, formant_num);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get bandwidth at a specific frame and formant number.
     * @param {number} frame
     * @param {number} formant_num
     * @returns {number}
     */
    get_bandwidth_at_frame(frame, formant_num) {
        const ret = wasm.formant_get_bandwidth_at_frame(this.__wbg_ptr, frame, formant_num);
        return ret;
    }
    /**
     * Get time at a specific frame index.
     * @param {number} frame
     * @returns {number}
     */
    get_time_from_frame(frame) {
        const ret = wasm.formant_get_time_from_frame(this.__wbg_ptr, frame);
        return ret;
    }
    /**
     * Get formant frequency at a specific frame and formant number.
     * @param {number} frame
     * @param {number} formant_num
     * @returns {number}
     */
    get_value_at_frame(frame, formant_num) {
        const ret = wasm.formant_get_value_at_frame(this.__wbg_ptr, frame, formant_num);
        return ret;
    }
    /**
     * Get the maximum number of formants per frame.
     * @returns {number}
     */
    max_num_formants() {
        const ret = wasm.formant_max_num_formants(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get the number of frames.
     * @returns {number}
     */
    n_frames() {
        const ret = wasm.formant_n_frames(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get the time step between frames.
     * @returns {number}
     */
    time_step() {
        const ret = wasm.formant_time_step(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get time points for all frames.
     * @returns {Float64Array}
     */
    times() {
        const ret = wasm.formant_times(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
}
if (Symbol.dispose) Formant.prototype[Symbol.dispose] = Formant.prototype.free;

/**
 * Harmonicity (HNR) analysis result.
 */
export class Harmonicity {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Harmonicity.prototype);
        obj.__wbg_ptr = ptr;
        HarmonicityFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        HarmonicityFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_harmonicity_free(ptr, 0);
    }
    /**
     * Get time at a specific frame index.
     * @param {number} frame
     * @returns {number}
     */
    get_time_from_frame(frame) {
        const ret = wasm.harmonicity_get_time_from_frame(this.__wbg_ptr, frame);
        return ret;
    }
    /**
     * Get HNR at a specific frame index in dB.
     * @param {number} frame
     * @returns {number}
     */
    get_value_in_frame(frame) {
        const ret = wasm.harmonicity_get_value_in_frame(this.__wbg_ptr, frame);
        return ret;
    }
    /**
     * Get the number of frames.
     * @returns {number}
     */
    n_frames() {
        const ret = wasm.harmonicity_n_frames(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get the time step between frames.
     * @returns {number}
     */
    time_step() {
        const ret = wasm.harmonicity_time_step(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get time points for all frames.
     * @returns {Float64Array}
     */
    times() {
        const ret = wasm.harmonicity_times(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get HNR values for all frames in dB. Returns -200 for silent/unvoiced frames.
     * @returns {Float64Array}
     */
    values() {
        const ret = wasm.harmonicity_values(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
}
if (Symbol.dispose) Harmonicity.prototype[Symbol.dispose] = Harmonicity.prototype.free;

/**
 * Intensity analysis result.
 */
export class Intensity {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Intensity.prototype);
        obj.__wbg_ptr = ptr;
        IntensityFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IntensityFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_intensity_free(ptr, 0);
    }
    /**
     * Get the maximum intensity value in dB.
     * @returns {number}
     */
    get_maximum() {
        const ret = wasm.intensity_get_maximum(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the mean intensity value in dB.
     * @returns {number}
     */
    get_mean() {
        const ret = wasm.intensity_get_mean(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the minimum intensity value in dB.
     * @returns {number}
     */
    get_minimum() {
        const ret = wasm.intensity_get_minimum(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get time at a specific frame index.
     * @param {number} frame
     * @returns {number}
     */
    get_time_from_frame(frame) {
        const ret = wasm.intensity_get_time_from_frame(this.__wbg_ptr, frame);
        return ret;
    }
    /**
     * Get intensity at a specific frame index in dB.
     * @param {number} frame
     * @returns {number}
     */
    get_value_in_frame(frame) {
        const ret = wasm.intensity_get_value_in_frame(this.__wbg_ptr, frame);
        return ret;
    }
    /**
     * Get the number of frames.
     * @returns {number}
     */
    n_frames() {
        const ret = wasm.intensity_n_frames(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get the time step between frames.
     * @returns {number}
     */
    time_step() {
        const ret = wasm.intensity_time_step(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get time points for all frames.
     * @returns {Float64Array}
     */
    times() {
        const ret = wasm.intensity_times(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get intensity values for all frames in dB.
     * @returns {Float64Array}
     */
    values() {
        const ret = wasm.intensity_values(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
}
if (Symbol.dispose) Intensity.prototype[Symbol.dispose] = Intensity.prototype.free;

/**
 * Pitch (F0) analysis result.
 */
export class Pitch {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Pitch.prototype);
        obj.__wbg_ptr = ptr;
        PitchFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PitchFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_pitch_free(ptr, 0);
    }
    /**
     * Get time at a specific frame index.
     * @param {number} frame
     * @returns {number}
     */
    get_time_from_frame(frame) {
        const ret = wasm.pitch_get_time_from_frame(this.__wbg_ptr, frame);
        return ret;
    }
    /**
     * Get F0 at a specific frame index. Returns NaN for unvoiced frames.
     * @param {number} frame
     * @returns {number}
     */
    get_value_in_frame(frame) {
        const ret = wasm.pitch_get_value_in_frame(this.__wbg_ptr, frame);
        return ret;
    }
    /**
     * Get the number of frames.
     * @returns {number}
     */
    n_frames() {
        const ret = wasm.pitch_n_frames(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get the pitch ceiling in Hz.
     * @returns {number}
     */
    pitch_ceiling() {
        const ret = wasm.pitch_pitch_ceiling(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the pitch floor in Hz.
     * @returns {number}
     */
    pitch_floor() {
        const ret = wasm.pitch_pitch_floor(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get strength (correlation) values for all frames.
     * @returns {Float64Array}
     */
    strengths() {
        const ret = wasm.pitch_strengths(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get the time step between frames.
     * @returns {number}
     */
    time_step() {
        const ret = wasm.pitch_time_step(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get time points for all frames.
     * @returns {Float64Array}
     */
    times() {
        const ret = wasm.pitch_times(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get F0 values for all frames.
     *
     * Returns Float64Array with F0 in Hz for voiced frames, NaN for unvoiced.
     * @returns {Float64Array}
     */
    values() {
        const ret = wasm.pitch_values(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
}
if (Symbol.dispose) Pitch.prototype[Symbol.dispose] = Pitch.prototype.free;

/**
 * Audio samples with sample rate.
 *
 * This is the main type for acoustic analysis in WASM. Create a Sound
 * from raw samples, then call analysis methods.
 */
export class Sound {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Sound.prototype);
        obj.__wbg_ptr = ptr;
        SoundFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SoundFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_sound_free(ptr, 0);
    }
    /**
     * Get the duration in seconds.
     * @returns {number}
     */
    duration() {
        const ret = wasm.sound_duration(this.__wbg_ptr);
        return ret;
    }
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
     * @param {Uint8Array} wav_bytes
     * @returns {Sound}
     */
    static from_wav(wav_bytes) {
        const ptr0 = passArray8ToWasm0(wav_bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.sound_from_wav(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return Sound.__wrap(ret[0]);
    }
    /**
     * Create a Sound from a specific channel of a WAV file.
     *
     * # Arguments
     *
     * * `wav_bytes` - WAV file contents as Uint8Array
     * * `channel` - Channel index (0 = left, 1 = right, etc.)
     * @param {Uint8Array} wav_bytes
     * @param {number} channel
     * @returns {Sound}
     */
    static from_wav_channel(wav_bytes, channel) {
        const ptr0 = passArray8ToWasm0(wav_bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.sound_from_wav_channel(ptr0, len0, channel);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return Sound.__wrap(ret[0]);
    }
    /**
     * Get the number of samples.
     * @returns {number}
     */
    n_samples() {
        const ret = wasm.sound_n_samples(this.__wbg_ptr);
        return ret >>> 0;
    }
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
     * @param {Float64Array} samples
     * @param {number} sample_rate
     */
    constructor(samples, sample_rate) {
        const ptr0 = passArrayF64ToWasm0(samples, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.sound_new(ptr0, len0, sample_rate);
        this.__wbg_ptr = ret >>> 0;
        SoundFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Get the sample rate in Hz.
     * @returns {number}
     */
    sample_rate() {
        const ret = wasm.sound_sample_rate(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the audio samples as a Float64Array.
     * @returns {Float64Array}
     */
    samples() {
        const ret = wasm.sound_samples(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
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
     * @param {number} time_step
     * @param {number} max_num_formants
     * @param {number} max_formant_hz
     * @param {number} window_length
     * @param {number} pre_emphasis_from
     * @returns {Formant}
     */
    to_formant_burg(time_step, max_num_formants, max_formant_hz, window_length, pre_emphasis_from) {
        const ret = wasm.sound_to_formant_burg(this.__wbg_ptr, time_step, max_num_formants, max_formant_hz, window_length, pre_emphasis_from);
        return Formant.__wrap(ret);
    }
    /**
     * Compute harmonicity (HNR) using autocorrelation method.
     *
     * # Arguments
     *
     * * `time_step` - Time step in seconds
     * * `min_pitch` - Minimum pitch in Hz
     * * `silence_threshold` - Silence threshold (0-1)
     * * `periods_per_window` - Number of periods per window (typically 4.5)
     * @param {number} time_step
     * @param {number} min_pitch
     * @param {number} silence_threshold
     * @param {number} periods_per_window
     * @returns {Harmonicity}
     */
    to_harmonicity_ac(time_step, min_pitch, silence_threshold, periods_per_window) {
        const ret = wasm.sound_to_harmonicity_ac(this.__wbg_ptr, time_step, min_pitch, silence_threshold, periods_per_window);
        return Harmonicity.__wrap(ret);
    }
    /**
     * Compute harmonicity (HNR) using cross-correlation method.
     * @param {number} time_step
     * @param {number} min_pitch
     * @param {number} silence_threshold
     * @param {number} periods_per_window
     * @returns {Harmonicity}
     */
    to_harmonicity_cc(time_step, min_pitch, silence_threshold, periods_per_window) {
        const ret = wasm.sound_to_harmonicity_cc(this.__wbg_ptr, time_step, min_pitch, silence_threshold, periods_per_window);
        return Harmonicity.__wrap(ret);
    }
    /**
     * Compute intensity contour.
     *
     * # Arguments
     *
     * * `min_pitch` - Minimum pitch in Hz (determines window size)
     * * `time_step` - Time step in seconds (0 = auto)
     * @param {number} min_pitch
     * @param {number} time_step
     * @returns {Intensity}
     */
    to_intensity(min_pitch, time_step) {
        const ret = wasm.sound_to_intensity(this.__wbg_ptr, min_pitch, time_step);
        return Intensity.__wrap(ret);
    }
    /**
     * Compute pitch (F0) contour using autocorrelation method.
     *
     * # Arguments
     *
     * * `time_step` - Time step in seconds (0 = auto)
     * * `pitch_floor` - Minimum pitch in Hz (e.g., 75)
     * * `pitch_ceiling` - Maximum pitch in Hz (e.g., 600)
     * @param {number} time_step
     * @param {number} pitch_floor
     * @param {number} pitch_ceiling
     * @returns {Pitch}
     */
    to_pitch_ac(time_step, pitch_floor, pitch_ceiling) {
        const ret = wasm.sound_to_pitch_ac(this.__wbg_ptr, time_step, pitch_floor, pitch_ceiling);
        return Pitch.__wrap(ret);
    }
    /**
     * Compute pitch (F0) contour using cross-correlation method.
     * @param {number} time_step
     * @param {number} pitch_floor
     * @param {number} pitch_ceiling
     * @returns {Pitch}
     */
    to_pitch_cc(time_step, pitch_floor, pitch_ceiling) {
        const ret = wasm.sound_to_pitch_cc(this.__wbg_ptr, time_step, pitch_floor, pitch_ceiling);
        return Pitch.__wrap(ret);
    }
    /**
     * Compute spectrogram (time-frequency representation).
     *
     * # Arguments
     *
     * * `window_length` - Window length in seconds
     * * `max_frequency` - Maximum frequency in Hz
     * * `time_step` - Time step in seconds
     * * `frequency_step` - Frequency step in Hz
     * @param {number} window_length
     * @param {number} max_frequency
     * @param {number} time_step
     * @param {number} frequency_step
     * @returns {Spectrogram}
     */
    to_spectrogram(window_length, max_frequency, time_step, frequency_step) {
        const ret = wasm.sound_to_spectrogram(this.__wbg_ptr, window_length, max_frequency, time_step, frequency_step);
        return Spectrogram.__wrap(ret);
    }
    /**
     * Compute spectrum (single-frame FFT).
     *
     * # Arguments
     *
     * * `fast` - If true, use power-of-2 FFT size for speed
     * @param {boolean} fast
     * @returns {Spectrum}
     */
    to_spectrum(fast) {
        const ret = wasm.sound_to_spectrum(this.__wbg_ptr, fast);
        return Spectrum.__wrap(ret);
    }
}
if (Symbol.dispose) Sound.prototype[Symbol.dispose] = Sound.prototype.free;

/**
 * Spectrogram (time-frequency) analysis result.
 */
export class Spectrogram {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Spectrogram.prototype);
        obj.__wbg_ptr = ptr;
        SpectrogramFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SpectrogramFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_spectrogram_free(ptr, 0);
    }
    /**
     * Get the maximum frequency in Hz.
     * @returns {number}
     */
    freq_max() {
        const ret = wasm.spectrogram_freq_max(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the minimum frequency in Hz.
     * @returns {number}
     */
    freq_min() {
        const ret = wasm.spectrogram_freq_min(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the frequency step between bins.
     * @returns {number}
     */
    freq_step() {
        const ret = wasm.spectrogram_freq_step(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get frequency points for all bins.
     * @returns {Float64Array}
     */
    frequencies() {
        const ret = wasm.spectrogram_frequencies(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get frequency at a specific bin index.
     * @param {number} bin
     * @returns {number}
     */
    get_freq_from_bin(bin) {
        const ret = wasm.spectrogram_get_freq_from_bin(this.__wbg_ptr, bin);
        return ret;
    }
    /**
     * Get time at a specific frame index.
     * @param {number} frame
     * @returns {number}
     */
    get_time_from_frame(frame) {
        const ret = wasm.spectrogram_get_time_from_frame(this.__wbg_ptr, frame);
        return ret;
    }
    /**
     * Get power value at a specific time frame and frequency bin.
     * @param {number} time_frame
     * @param {number} freq_bin
     * @returns {number}
     */
    get_value_at(time_frame, freq_bin) {
        const ret = wasm.spectrogram_get_value_at(this.__wbg_ptr, time_frame, freq_bin);
        return ret;
    }
    /**
     * Get the number of frequency bins.
     * @returns {number}
     */
    n_freqs() {
        const ret = wasm.spectrogram_n_freqs(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get the number of time frames.
     * @returns {number}
     */
    n_times() {
        const ret = wasm.spectrogram_n_times(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get the maximum time in seconds.
     * @returns {number}
     */
    time_max() {
        const ret = wasm.spectrogram_time_max(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the minimum time in seconds.
     * @returns {number}
     */
    time_min() {
        const ret = wasm.spectrogram_time_min(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the time step between frames.
     * @returns {number}
     */
    time_step() {
        const ret = wasm.spectrogram_time_step(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get time points for all frames.
     * @returns {Float64Array}
     */
    times() {
        const ret = wasm.spectrogram_times(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get all power values as a flat array (row-major: freq × time).
     * @returns {Float64Array}
     */
    values() {
        const ret = wasm.spectrogram_values(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
}
if (Symbol.dispose) Spectrogram.prototype[Symbol.dispose] = Spectrogram.prototype.free;

/**
 * Spectrum (single-frame FFT) analysis result.
 */
export class Spectrum {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Spectrum.prototype);
        obj.__wbg_ptr = ptr;
        SpectrumFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SpectrumFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_spectrum_free(ptr, 0);
    }
    /**
     * Get the frequency resolution (bin width) in Hz.
     * @returns {number}
     */
    df() {
        const ret = wasm.spectrum_df(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the energy in a frequency band.
     * @param {number} f_min
     * @param {number} f_max
     * @returns {number}
     */
    get_band_energy(f_min, f_max) {
        const ret = wasm.spectrum_get_band_energy(this.__wbg_ptr, f_min, f_max);
        return ret;
    }
    /**
     * Get the center of gravity (spectral centroid) in Hz.
     * @param {number} power
     * @returns {number}
     */
    get_center_of_gravity(power) {
        const ret = wasm.spectrum_get_center_of_gravity(this.__wbg_ptr, power);
        return ret;
    }
    /**
     * Get frequency for a bin index.
     * @param {number} bin
     * @returns {number}
     */
    get_freq_from_bin(bin) {
        const ret = wasm.spectrum_get_freq_from_bin(this.__wbg_ptr, bin);
        return ret;
    }
    /**
     * Get the kurtosis of the spectrum.
     * @param {number} power
     * @returns {number}
     */
    get_kurtosis(power) {
        const ret = wasm.spectrum_get_kurtosis(this.__wbg_ptr, power);
        return ret;
    }
    /**
     * Get the skewness of the spectrum.
     * @param {number} power
     * @returns {number}
     */
    get_skewness(power) {
        const ret = wasm.spectrum_get_skewness(this.__wbg_ptr, power);
        return ret;
    }
    /**
     * Get the standard deviation (spectral spread) in Hz.
     * @param {number} power
     * @returns {number}
     */
    get_standard_deviation(power) {
        const ret = wasm.spectrum_get_standard_deviation(this.__wbg_ptr, power);
        return ret;
    }
    /**
     * Get the imaginary parts of the spectrum.
     * @returns {Float64Array}
     */
    imag() {
        const ret = wasm.spectrum_imag(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get the maximum frequency in Hz.
     * @returns {number}
     */
    max_frequency() {
        const ret = wasm.spectrum_max_frequency(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the number of frequency bins.
     * @returns {number}
     */
    n_bins() {
        const ret = wasm.spectrum_n_bins(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get the real parts of the spectrum.
     * @returns {Float64Array}
     */
    real() {
        const ret = wasm.spectrum_real(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
}
if (Symbol.dispose) Spectrum.prototype[Symbol.dispose] = Spectrum.prototype.free;

/**
 * Initialize the WASM module.
 *
 * This sets up panic hooks for better error messages in the browser console.
 * Call this once before using any other functions.
 */
export function init() {
    wasm.init();
}

function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_8c4e43fe74559d73: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg___wbindgen_throw_be289d5034ed271b: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_error_7534b8e9a36f1ab4: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_new_8a6f238a6ece86ea: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_stack_0ed75d68575b0f3c: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./praatfan_bg.js": import0,
    };
}

const FormantFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_formant_free(ptr >>> 0, 1));
const HarmonicityFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_harmonicity_free(ptr >>> 0, 1));
const IntensityFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_intensity_free(ptr >>> 0, 1));
const PitchFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_pitch_free(ptr >>> 0, 1));
const SoundFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_sound_free(ptr >>> 0, 1));
const SpectrogramFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_spectrogram_free(ptr >>> 0, 1));
const SpectrumFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_spectrum_free(ptr >>> 0, 1));

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('praatfan_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
