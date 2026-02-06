"""Parselmouth compatibility layer.

This module provides a `call()` function that emulates parselmouth's
functional API, allowing existing parselmouth scripts to work with
praatfan objects with minimal changes.

Usage:
    from praatfan.praat import call

    # Instead of parselmouth:
    # from parselmouth.praat import call

    sound = praatfan.Sound("audio.wav")
    pitch = call(sound, "To Pitch (ac)", 0, 75, 600)
    f0 = call(pitch, "Get value in frame", 10, "Hertz")

Note:
    Parselmouth uses 1-based frame indices while praatfan uses 0-based.
    This compatibility layer handles the conversion automatically.
"""

from typing import Any, Optional, Union
import numpy as np


class PraatCallError(Exception):
    """Raised when a call() command fails or is not supported."""
    pass


def _normalize_command(command: str) -> str:
    """Normalize command string for case-insensitive matching."""
    return command.lower().strip()


def _is_sound(obj: Any) -> bool:
    """Check if object is a Sound."""
    name = type(obj).__name__
    return name in ("Sound", "UnifiedSound") or "Sound" in name


def _is_pitch(obj: Any) -> bool:
    """Check if object is a Pitch."""
    name = type(obj).__name__
    return name in ("Pitch", "UnifiedPitch") or (name.endswith("Pitch") and "Pitch" in name)


def _is_formant(obj: Any) -> bool:
    """Check if object is a Formant."""
    name = type(obj).__name__
    return name in ("Formant", "UnifiedFormant") or (name.endswith("Formant") and "Formant" in name)


def _is_intensity(obj: Any) -> bool:
    """Check if object is an Intensity."""
    name = type(obj).__name__
    return name in ("Intensity", "UnifiedIntensity") or (name.endswith("Intensity") and "Intensity" in name)


def _is_harmonicity(obj: Any) -> bool:
    """Check if object is a Harmonicity."""
    name = type(obj).__name__
    return name in ("Harmonicity", "UnifiedHarmonicity") or (name.endswith("Harmonicity") and "Harmonicity" in name)


def _is_spectrum(obj: Any) -> bool:
    """Check if object is a Spectrum."""
    name = type(obj).__name__
    return name in ("Spectrum", "UnifiedSpectrum") or (name.endswith("Spectrum") and "Spectrum" in name)


def _is_spectrogram(obj: Any) -> bool:
    """Check if object is a Spectrogram."""
    name = type(obj).__name__
    return name in ("Spectrogram", "UnifiedSpectrogram") or (name.endswith("Spectrogram") and "Spectrogram" in name)


def _parse_bool_arg(arg: Any) -> bool:
    """Parse a boolean argument (handles 'yes'/'no' strings)."""
    if isinstance(arg, bool):
        return arg
    if isinstance(arg, str):
        return arg.lower() in ("yes", "true", "1")
    return bool(arg)


# =============================================================================
# Sound commands
# =============================================================================

def _call_sound(obj: Any, command: str, args: tuple) -> Any:
    """Handle call() for Sound objects."""
    cmd = _normalize_command(command)

    # To Pitch (ac)
    if cmd == "to pitch (ac)":
        # Parselmouth full argument order (10 args):
        #   time_step, pitch_floor, max_candidates, very_accurate,
        #   silence_threshold, voicing_threshold, octave_cost,
        #   octave_jump_cost, voiced_unvoiced_cost, pitch_ceiling
        # Short form (3 args): time_step, pitch_floor, pitch_ceiling
        time_step = args[0] if len(args) > 0 else 0.0
        pitch_floor = args[1] if len(args) > 1 else 75.0
        if len(args) > 9:
            # Full parselmouth form: pitch_ceiling is the last (10th) argument
            pitch_ceiling = args[9]
        elif len(args) > 2:
            # Short form: pitch_ceiling is the 3rd argument
            pitch_ceiling = args[2]
        else:
            pitch_ceiling = 600.0
        # Use to_pitch_ac if available, otherwise to_pitch with method parameter
        if hasattr(obj, 'to_pitch_ac'):
            return obj.to_pitch_ac(
                time_step=time_step,
                pitch_floor=pitch_floor,
                pitch_ceiling=pitch_ceiling
            )
        else:
            return obj.to_pitch(
                time_step=time_step,
                pitch_floor=pitch_floor,
                pitch_ceiling=pitch_ceiling,
                method="ac"
            )

    # To Pitch (cc)
    elif cmd == "to pitch (cc)":
        # Arguments: time_step, pitch_floor, max_candidates (ignored),
        #            very_accurate (ignored), silence_threshold (ignored),
        #            voicing_threshold (ignored), octave_cost (ignored),
        #            octave_jump_cost (ignored), voiced_unvoiced_cost (ignored),
        #            pitch_ceiling
        time_step = args[0] if len(args) > 0 else 0.0
        pitch_floor = args[1] if len(args) > 1 else 75.0
        # args[2] = max_candidates, args[3] = very_accurate, ...
        # pitch_ceiling is the last positional argument
        pitch_ceiling = args[9] if len(args) > 9 else 600.0
        # Use to_pitch_cc if available, otherwise to_pitch with method parameter
        if hasattr(obj, 'to_pitch_cc'):
            return obj.to_pitch_cc(
                time_step=time_step,
                pitch_floor=pitch_floor,
                pitch_ceiling=pitch_ceiling
            )
        else:
            return obj.to_pitch(
                time_step=time_step,
                pitch_floor=pitch_floor,
                pitch_ceiling=pitch_ceiling,
                method="cc"
            )

    # To Pitch...
    elif cmd == "to pitch" or cmd == "to pitch...":
        # Simple version: time_step, pitch_floor, pitch_ceiling
        # Uses AC method by default
        time_step = args[0] if len(args) > 0 else 0.0
        pitch_floor = args[1] if len(args) > 1 else 75.0
        pitch_ceiling = args[2] if len(args) > 2 else 600.0
        if hasattr(obj, 'to_pitch_ac'):
            return obj.to_pitch_ac(
                time_step=time_step,
                pitch_floor=pitch_floor,
                pitch_ceiling=pitch_ceiling
            )
        else:
            return obj.to_pitch(
                time_step=time_step,
                pitch_floor=pitch_floor,
                pitch_ceiling=pitch_ceiling
            )

    # To Formant (burg)
    elif cmd == "to formant (burg)" or cmd == "to formant (burg)...":
        # Arguments: time_step, max_num_formants, max_formant_hz,
        #            window_length, pre_emphasis_from
        time_step = args[0] if len(args) > 0 else 0.0
        max_num_formants = int(args[1]) if len(args) > 1 else 5
        max_formant_hz = args[2] if len(args) > 2 else 5500.0
        window_length = args[3] if len(args) > 3 else 0.025
        pre_emphasis_from = args[4] if len(args) > 4 else 50.0
        # Try both parameter name conventions
        try:
            return obj.to_formant_burg(
                time_step=time_step,
                max_number_of_formants=max_num_formants,
                maximum_formant=max_formant_hz,
                window_length=window_length,
                pre_emphasis_from=pre_emphasis_from
            )
        except TypeError:
            return obj.to_formant_burg(
                time_step=time_step,
                max_num_formants=max_num_formants,
                max_formant_hz=max_formant_hz,
                window_length=window_length,
                pre_emphasis_from=pre_emphasis_from
            )

    # To Intensity
    elif cmd == "to intensity" or cmd == "to intensity...":
        # Arguments: min_pitch, time_step, subtract_mean (ignored)
        min_pitch = args[0] if len(args) > 0 else 100.0
        time_step = args[1] if len(args) > 1 else 0.0
        # Try both parameter name conventions
        try:
            return obj.to_intensity(minimum_pitch=min_pitch, time_step=time_step)
        except TypeError:
            return obj.to_intensity(min_pitch=min_pitch, time_step=time_step)

    # To Harmonicity (ac)
    elif cmd == "to harmonicity (ac)" or cmd == "to harmonicity (ac)...":
        # Arguments: time_step, min_pitch, silence_threshold, periods_per_window
        time_step = args[0] if len(args) > 0 else 0.01
        min_pitch = args[1] if len(args) > 1 else 75.0
        silence_threshold = args[2] if len(args) > 2 else 0.1
        periods_per_window = args[3] if len(args) > 3 else 4.5
        # Try both parameter name conventions
        try:
            return obj.to_harmonicity_ac(
                time_step=time_step,
                minimum_pitch=min_pitch,
                silence_threshold=silence_threshold,
                periods_per_window=periods_per_window
            )
        except TypeError:
            return obj.to_harmonicity_ac(
                time_step=time_step,
                min_pitch=min_pitch,
                silence_threshold=silence_threshold,
                periods_per_window=periods_per_window
            )

    # To Harmonicity (cc)
    elif cmd == "to harmonicity (cc)" or cmd == "to harmonicity (cc)...":
        # Arguments: time_step, min_pitch, silence_threshold, periods_per_window
        time_step = args[0] if len(args) > 0 else 0.01
        min_pitch = args[1] if len(args) > 1 else 75.0
        silence_threshold = args[2] if len(args) > 2 else 0.1
        periods_per_window = args[3] if len(args) > 3 else 1.0
        # Try both parameter name conventions
        try:
            return obj.to_harmonicity_cc(
                time_step=time_step,
                minimum_pitch=min_pitch,
                silence_threshold=silence_threshold,
                periods_per_window=periods_per_window
            )
        except TypeError:
            return obj.to_harmonicity_cc(
                time_step=time_step,
                min_pitch=min_pitch,
                silence_threshold=silence_threshold,
                periods_per_window=periods_per_window
            )

    # To Spectrum
    elif cmd == "to spectrum" or cmd == "to spectrum...":
        # Arguments: fast (yes/no)
        fast = _parse_bool_arg(args[0]) if len(args) > 0 else True
        return obj.to_spectrum(fast=fast)

    # To Spectrogram
    elif cmd == "to spectrogram" or cmd == "to spectrogram...":
        # Arguments: window_length, max_frequency, time_step, frequency_step,
        #            window_shape (ignored)
        window_length = args[0] if len(args) > 0 else 0.005
        max_frequency = args[1] if len(args) > 1 else 5000.0
        time_step = args[2] if len(args) > 2 else 0.002
        frequency_step = args[3] if len(args) > 3 else 20.0
        # Try both parameter name conventions
        try:
            return obj.to_spectrogram(
                window_length=window_length,
                maximum_frequency=max_frequency,
                time_step=time_step,
                frequency_step=frequency_step
            )
        except TypeError:
            return obj.to_spectrogram(
                window_length=window_length,
                max_frequency=max_frequency,
                time_step=time_step,
                frequency_step=frequency_step
            )

    # Get total duration
    elif cmd == "get total duration":
        if hasattr(obj, 'duration'):
            return obj.duration
        elif hasattr(obj, 'get_total_duration'):
            return obj.get_total_duration()
        # Calculate from samples and sample rate
        if hasattr(obj, 'n_samples') and hasattr(obj, 'sampling_frequency'):
            return obj.n_samples / obj.sampling_frequency
        raise PraatCallError("Cannot determine duration of Sound object")

    # Extract part
    elif cmd == "extract part" or cmd == "extract part...":
        # Arguments: start_time, end_time, window_shape, relative_width, preserve_times
        start_time = args[0] if len(args) > 0 else 0.0
        end_time = args[1] if len(args) > 1 else None
        # window_shape = args[2] if len(args) > 2 else "rectangular"  # ignored for now
        # relative_width = args[3] if len(args) > 3 else 1.0  # ignored for now
        # preserve_times = args[4] if len(args) > 4 else "no"  # ignored for now

        if hasattr(obj, 'extract_part'):
            return obj.extract_part(start_time, end_time)
        elif hasattr(obj, 'extract'):
            return obj.extract(start_time, end_time)
        # Manual extraction using samples
        # Get samples - handle both values() method and values property
        samples = None
        if hasattr(obj, 'values'):
            samples = obj.values() if callable(obj.values) else obj.values
        if samples is not None and hasattr(obj, 'sampling_frequency'):
            sr = obj.sampling_frequency
            samples = np.array(samples)
            start_sample = round(start_time * sr)
            end_sample = round(end_time * sr) if end_time else len(samples)
            extracted_samples = samples[start_sample:end_sample]
            # Create a new Sound object using the class constructor
            # The unified Sound class accepts (samples, sampling_frequency)
            return obj.__class__(extracted_samples, sr)
        raise PraatCallError("Cannot extract part from Sound object - method not available")

    # Filter (pre-emphasis)
    elif cmd == "filter (pre-emphasis)" or cmd == "filter (pre-emphasis)...":
        # Arguments: from_frequency (Hz)
        from_frequency = args[0] if len(args) > 0 else 50.0
        if hasattr(obj, 'pre_emphasis'):
            return obj.pre_emphasis(from_frequency)
        elif hasattr(obj, 'filter_pre_emphasis'):
            return obj.filter_pre_emphasis(from_frequency)
        # Manual pre-emphasis using samples
        # Get samples - handle both values() method and values property
        samples = None
        if hasattr(obj, 'values'):
            samples = obj.values() if callable(obj.values) else obj.values
        if samples is not None and hasattr(obj, 'sampling_frequency'):
            sr = obj.sampling_frequency
            samples = np.array(samples)
            # Pre-emphasis coefficient: alpha = exp(-2 * pi * from_frequency / sr)
            alpha = np.exp(-2 * np.pi * from_frequency / sr)
            # Apply pre-emphasis: y[n] = x[n] - alpha * x[n-1]
            pre_emphasized = np.zeros_like(samples)
            pre_emphasized[0] = samples[0]
            pre_emphasized[1:] = samples[1:] - alpha * samples[:-1]
            # Create a new Sound object using the class constructor
            return obj.__class__(pre_emphasized, sr)
        raise PraatCallError("Cannot apply pre-emphasis to Sound object - method not available")

    else:
        raise PraatCallError(f"Unknown Sound command: '{command}'")


# =============================================================================
# Pitch commands
# =============================================================================

def _get_pitch_times(obj: Any) -> np.ndarray:
    """Get pitch frame times, handling both API styles."""
    if hasattr(obj, 'xs'):
        return np.array(obj.xs())
    elif hasattr(obj, 'times'):
        return obj.times()
    elif hasattr(obj, 'frames'):
        return np.array([f.time for f in obj.frames])
    return np.array([])


def _get_pitch_values(obj: Any) -> np.ndarray:
    """Get pitch F0 values, handling both API styles."""
    if hasattr(obj, 'values') and callable(obj.values):
        return np.array(obj.values())
    elif hasattr(obj, 'values'):
        return np.array(obj.values)
    elif hasattr(obj, 'frames'):
        return np.array([f.frequency for f in obj.frames])
    return np.array([])


def _get_pitch_strengths(obj: Any) -> np.ndarray:
    """Get pitch strengths, handling both API styles."""
    if hasattr(obj, 'strengths') and callable(obj.strengths):
        return np.array(obj.strengths())
    elif hasattr(obj, 'strengths'):
        return np.array(obj.strengths)
    elif hasattr(obj, 'frames'):
        return np.array([f.strength for f in obj.frames])
    return np.array([])


def _convert_pitch_unit(value: float, unit: str) -> float:
    """Convert pitch value from Hz to specified unit."""
    unit_lower = unit.lower() if isinstance(unit, str) else "hertz"
    if unit_lower == "hertz":
        return value
    elif unit_lower == "semitones":
        return 12.0 * np.log2(value / 100.0)
    elif unit_lower == "mel":
        return 1127.0 * np.log(1.0 + value / 700.0)
    elif unit_lower == "erb":
        return 21.4 * np.log10(0.00437 * value + 1.0)
    else:
        return value


def _is_undefined(value, undefined_value) -> bool:
    """Check if a value matches the undefined sentinel, handling NaN correctly."""
    if isinstance(undefined_value, float) and np.isnan(undefined_value):
        return np.isnan(value)
    return value == undefined_value


def _interpolate_at_time(times: np.ndarray, values: np.ndarray, time: float,
                          interpolation: str = "linear", undefined_value: float = 0.0) -> Optional[float]:
    """Interpolate a value at a specific time."""
    if len(times) == 0:
        return None

    # Find position in frame array
    t0 = times[0]
    time_step = times[1] - times[0] if len(times) > 1 else 0.01
    n_frames = len(times)
    idx_float = (time - t0) / time_step

    if idx_float < -0.5 or idx_float > n_frames - 0.5:
        return None

    if interpolation == "nearest":
        idx = int(round(idx_float))
        idx = max(0, min(n_frames - 1, idx))
        value = values[idx]
        if _is_undefined(value, undefined_value):
            return None
        return float(value)

    elif interpolation == "linear":
        idx = int(np.floor(idx_float))
        frac = idx_float - idx

        i1 = max(0, min(n_frames - 1, idx))
        i2 = max(0, min(n_frames - 1, idx + 1))

        v1, v2 = values[i1], values[i2]

        v1_undef = _is_undefined(v1, undefined_value)
        v2_undef = _is_undefined(v2, undefined_value)

        # Both must be defined for interpolation
        if v1_undef and v2_undef:
            return None
        elif v1_undef:
            return float(v2)
        elif v2_undef:
            return float(v1)
        else:
            return float(v1 * (1 - frac) + v2 * frac)
    else:
        return None


def _call_pitch(obj: Any, command: str, args: tuple) -> Any:
    """Handle call() for Pitch objects."""
    cmd = _normalize_command(command)

    # Get number of frames
    if cmd == "get number of frames":
        return _get_n_frames(obj)

    # Get time from frame number (1-based)
    elif cmd == "get time from frame number" or cmd == "get time from frame number...":
        frame_number = int(args[0])
        n_frames = _get_n_frames(obj)
        # Convert 1-based to 0-based
        frame_idx = frame_number - 1
        if frame_idx < 0 or frame_idx >= n_frames:
            return None
        times = _get_pitch_times(obj)
        return float(times[frame_idx])

    # Get value in frame (1-based)
    elif cmd == "get value in frame" or cmd == "get value in frame...":
        frame_number = int(args[0])
        unit = args[1] if len(args) > 1 else "Hertz"

        # Convert 1-based to 0-based
        frame_idx = frame_number - 1
        if frame_idx < 0 or frame_idx >= _get_n_frames(obj):
            return None

        values = _get_pitch_values(obj)
        value = values[frame_idx]

        # 0 means unvoiced
        if value == 0:
            return None

        return _convert_pitch_unit(float(value), unit)

    # Get value at time
    elif cmd == "get value at time" or cmd == "get value at time...":
        time = args[0]
        unit = args[1] if len(args) > 1 else "Hertz"
        interpolation = args[2] if len(args) > 2 else "linear"

        # Map parselmouth interpolation names
        interp_map = {
            "linear": "linear",
            "nearest": "nearest",
            "none": "nearest"
        }
        interp = interp_map.get(interpolation.lower() if isinstance(interpolation, str) else "linear", "linear")

        # Try native method first
        if hasattr(obj, 'get_value_at_time'):
            return obj.get_value_at_time(time, unit=unit, interpolation=interp)

        # Fall back to manual interpolation
        times = _get_pitch_times(obj)
        values = _get_pitch_values(obj)
        result = _interpolate_at_time(times, values, time, interp, undefined_value=0.0)
        if result is not None:
            return _convert_pitch_unit(result, unit)
        return None

    # Get strength in frame (1-based)
    elif cmd == "get strength in frame" or cmd == "get strength in frame...":
        frame_number = int(args[0])

        # Convert 1-based to 0-based
        frame_idx = frame_number - 1
        if frame_idx < 0 or frame_idx >= _get_n_frames(obj):
            return None

        strengths = _get_pitch_strengths(obj)
        return float(strengths[frame_idx])

    # Get strength at time
    elif cmd == "get strength at time" or cmd == "get strength at time...":
        time = args[0]
        interpolation = args[1] if len(args) > 1 else "linear"

        interp_map = {
            "linear": "linear",
            "nearest": "nearest",
            "none": "nearest"
        }
        interp = interp_map.get(interpolation.lower() if isinstance(interpolation, str) else "linear", "linear")

        # Try native method first
        if hasattr(obj, 'get_strength_at_time'):
            return obj.get_strength_at_time(time, interpolation=interp)

        # Fall back to manual interpolation
        times = _get_pitch_times(obj)
        strengths = _get_pitch_strengths(obj)
        return _interpolate_at_time(times, strengths, time, interp, undefined_value=-1.0)

    else:
        raise PraatCallError(f"Unknown Pitch command: '{command}'")


# =============================================================================
# Formant commands
# =============================================================================

def _get_formant_times(obj: Any) -> np.ndarray:
    """Get formant frame times, handling both API styles."""
    # Try xs() first for unified API
    try:
        if hasattr(obj, 'xs'):
            return np.array(obj.xs())
    except (AttributeError, TypeError):
        pass

    # Try times() for pure Python implementation
    if hasattr(obj, 'times') and callable(obj.times):
        try:
            return obj.times()
        except (AttributeError, TypeError):
            pass

    # Try frames for pure Python implementation
    if hasattr(obj, 'frames'):
        return np.array([f.time for f in obj.frames])

    # Try accessing inner object for selector wrappers
    if hasattr(obj, '_inner'):
        inner = obj._inner
        if hasattr(inner, 'times') and callable(inner.times):
            try:
                return np.array(inner.times())
            except (AttributeError, TypeError):
                pass

    return np.array([])


def _get_formant_values(obj: Any, formant_number: int) -> np.ndarray:
    """Get formant frequency values for a specific formant, handling both API styles."""
    if hasattr(obj, 'formant_values') and callable(obj.formant_values):
        return np.array(obj.formant_values(formant_number))
    elif hasattr(obj, 'frames'):
        values = []
        for frame in obj.frames:
            fp = frame.get_formant(formant_number)
            values.append(fp.frequency if fp else np.nan)
        return np.array(values)
    return np.array([])


def _get_bandwidth_values(obj: Any, formant_number: int) -> np.ndarray:
    """Get bandwidth values for a specific formant, handling both API styles."""
    if hasattr(obj, 'bandwidth_values') and callable(obj.bandwidth_values):
        return np.array(obj.bandwidth_values(formant_number))
    elif hasattr(obj, 'frames'):
        values = []
        for frame in obj.frames:
            fp = frame.get_formant(formant_number)
            values.append(fp.bandwidth if fp else np.nan)
        return np.array(values)
    return np.array([])


def _get_n_frames(obj: Any) -> int:
    """Get number of frames from an object, handling different attribute names."""
    # Try direct attributes first
    try:
        if hasattr(obj, 'n_frames'):
            return obj.n_frames
    except AttributeError:
        pass

    if hasattr(obj, 'num_frames'):
        return obj.num_frames

    # Try accessing inner object for selector wrappers
    if hasattr(obj, '_inner'):
        inner = obj._inner
        if hasattr(inner, 'num_frames'):
            return inner.num_frames
        if hasattr(inner, 'n_frames'):
            return inner.n_frames

    return 0


def _call_formant(obj: Any, command: str, args: tuple) -> Any:
    """Handle call() for Formant objects."""
    cmd = _normalize_command(command)

    # Get number of frames
    if cmd == "get number of frames":
        return _get_n_frames(obj)

    # Get time from frame number (1-based)
    elif cmd == "get time from frame number" or cmd == "get time from frame number...":
        frame_number = int(args[0])
        # Convert 1-based to 0-based
        frame_idx = frame_number - 1
        if frame_idx < 0 or frame_idx >= _get_n_frames(obj):
            return None
        times = _get_formant_times(obj)
        return float(times[frame_idx])

    # Get number of formants in frame (1-based)
    elif cmd == "get number of formants" or cmd == "get number of formants...":
        frame_number = int(args[0])
        # Convert 1-based to 0-based
        frame_idx = frame_number - 1
        if frame_idx < 0 or frame_idx >= _get_n_frames(obj):
            return 0

        # For unified API, count non-NaN formants
        if hasattr(obj, 'formant_values'):
            count = 0
            for i in range(1, 10):  # Check up to F9
                try:
                    values = obj.formant_values(i)
                    if frame_idx < len(values) and not np.isnan(values[frame_idx]):
                        count = i
                    else:
                        break
                except (IndexError, ValueError):
                    break
            return count
        elif hasattr(obj, 'frames'):
            return obj.frames[frame_idx].n_formants
        return 0

    # Get value at time
    elif cmd == "get value at time" or cmd == "get value at time...":
        formant_number = int(args[0])  # 1-based formant number (F1, F2, etc.)
        time = args[1]
        unit = args[2] if len(args) > 2 else "Hertz"
        interpolation = args[3] if len(args) > 3 else "linear"

        interp_map = {
            "linear": "linear",
            "nearest": "nearest",
            "none": "nearest"
        }
        interp = interp_map.get(interpolation.lower() if isinstance(interpolation, str) else "linear", "linear")

        # Try native method first
        if hasattr(obj, 'get_value_at_time'):
            return obj.get_value_at_time(formant_number, time, unit=unit, interpolation=interp)

        # Fall back to manual interpolation
        times = _get_formant_times(obj)
        values = _get_formant_values(obj, formant_number)
        result = _interpolate_at_time(times, values, time, interp, undefined_value=np.nan)
        if result is not None and not np.isnan(result):
            return float(result)
        return None

    # Get bandwidth at time
    elif cmd == "get bandwidth at time" or cmd == "get bandwidth at time...":
        formant_number = int(args[0])  # 1-based formant number
        time = args[1]
        unit = args[2] if len(args) > 2 else "Hertz"
        interpolation = args[3] if len(args) > 3 else "linear"

        interp_map = {
            "linear": "linear",
            "nearest": "nearest",
            "none": "nearest"
        }
        interp = interp_map.get(interpolation.lower() if isinstance(interpolation, str) else "linear", "linear")

        # Try native method first
        if hasattr(obj, 'get_bandwidth_at_time'):
            return obj.get_bandwidth_at_time(formant_number, time, unit=unit, interpolation=interp)

        # Fall back to manual interpolation
        times = _get_formant_times(obj)
        values = _get_bandwidth_values(obj, formant_number)
        result = _interpolate_at_time(times, values, time, interp, undefined_value=np.nan)
        if result is not None and not np.isnan(result):
            return float(result)
        return None

    else:
        raise PraatCallError(f"Unknown Formant command: '{command}'")


# =============================================================================
# Intensity commands
# =============================================================================

def _get_intensity_times(obj: Any) -> np.ndarray:
    """Get intensity frame times, handling both API styles."""
    # Try xs() first for unified API
    try:
        if hasattr(obj, 'xs'):
            return np.array(obj.xs())
    except (AttributeError, TypeError):
        pass

    # Try times() for pure Python implementation
    if hasattr(obj, 'times') and callable(obj.times):
        try:
            return obj.times()
        except (AttributeError, TypeError):
            pass

    # Try times property for pure Python implementation
    if hasattr(obj, 'times'):
        try:
            return np.array(obj.times)
        except (AttributeError, TypeError):
            pass

    # Try accessing inner object for selector wrappers
    if hasattr(obj, '_inner'):
        inner = obj._inner
        if hasattr(inner, 'times') and callable(inner.times):
            try:
                return np.array(inner.times())
            except (AttributeError, TypeError):
                pass

    return np.array([])


def _get_intensity_values(obj: Any) -> np.ndarray:
    """Get intensity values, handling both API styles."""
    if hasattr(obj, 'values') and callable(obj.values):
        return np.array(obj.values())
    elif hasattr(obj, 'values'):
        return np.array(obj.values)
    return np.array([])


def _call_intensity(obj: Any, command: str, args: tuple) -> Any:
    """Handle call() for Intensity objects."""
    cmd = _normalize_command(command)

    # Get number of frames
    if cmd == "get number of frames":
        return _get_n_frames(obj)

    # Get time from frame number (1-based)
    elif cmd == "get time from frame number" or cmd == "get time from frame number...":
        frame_number = int(args[0])
        # Convert 1-based to 0-based
        frame_idx = frame_number - 1
        if frame_idx < 0 or frame_idx >= _get_n_frames(obj):
            return None
        times = _get_intensity_times(obj)
        return float(times[frame_idx])

    # Get value at time
    elif cmd == "get value at time" or cmd == "get value at time...":
        time = args[0]
        interpolation = args[1] if len(args) > 1 else "cubic"

        interp_map = {
            "cubic": "cubic",
            "linear": "linear",
            "nearest": "nearest",
            "none": "nearest"
        }
        interp = interp_map.get(interpolation.lower() if isinstance(interpolation, str) else "cubic", "cubic")

        # Try native method first
        if hasattr(obj, 'get_value_at_time'):
            return obj.get_value_at_time(time, interpolation=interp)

        # Fall back to manual interpolation (linear only for now)
        times = _get_intensity_times(obj)
        values = _get_intensity_values(obj)
        effective_interp = "linear" if interp == "cubic" else interp  # Simplify cubic to linear
        return _interpolate_at_time(times, values, time, effective_interp, undefined_value=-np.inf)

    # Get value in frame (1-based)
    elif cmd == "get value in frame" or cmd == "get value in frame...":
        frame_number = int(args[0])
        # Convert 1-based to 0-based
        frame_idx = frame_number - 1
        if frame_idx < 0 or frame_idx >= _get_n_frames(obj):
            return None
        values = _get_intensity_values(obj)
        return float(values[frame_idx])

    else:
        raise PraatCallError(f"Unknown Intensity command: '{command}'")


# =============================================================================
# Harmonicity commands
# =============================================================================

def _get_harmonicity_times(obj: Any) -> np.ndarray:
    """Get harmonicity frame times, handling both API styles."""
    # Try xs() first for unified API
    try:
        if hasattr(obj, 'xs'):
            return np.array(obj.xs())
    except (AttributeError, TypeError):
        pass

    # Try times() for pure Python implementation
    if hasattr(obj, 'times') and callable(obj.times):
        try:
            return obj.times()
        except (AttributeError, TypeError):
            pass

    # Try times property for pure Python implementation
    if hasattr(obj, 'times'):
        try:
            return np.array(obj.times)
        except (AttributeError, TypeError):
            pass

    # Try accessing inner object for selector wrappers
    if hasattr(obj, '_inner'):
        inner = obj._inner
        if hasattr(inner, 'times') and callable(inner.times):
            try:
                return np.array(inner.times())
            except (AttributeError, TypeError):
                pass

    return np.array([])


def _get_harmonicity_values(obj: Any) -> np.ndarray:
    """Get harmonicity values, handling both API styles."""
    if hasattr(obj, 'values') and callable(obj.values):
        return np.array(obj.values())
    elif hasattr(obj, 'values'):
        return np.array(obj.values)
    return np.array([])


def _call_harmonicity(obj: Any, command: str, args: tuple) -> Any:
    """Handle call() for Harmonicity objects."""
    cmd = _normalize_command(command)

    # Get number of frames
    if cmd == "get number of frames":
        return _get_n_frames(obj)

    # Get time from frame number (1-based)
    elif cmd == "get time from frame number" or cmd == "get time from frame number...":
        frame_number = int(args[0])
        # Convert 1-based to 0-based
        frame_idx = frame_number - 1
        if frame_idx < 0 or frame_idx >= _get_n_frames(obj):
            return None
        times = _get_harmonicity_times(obj)
        return float(times[frame_idx])

    # Get value at time
    elif cmd == "get value at time" or cmd == "get value at time...":
        time = args[0]
        interpolation = args[1] if len(args) > 1 else "cubic"

        interp_map = {
            "cubic": "cubic",
            "linear": "linear",
            "nearest": "nearest",
            "none": "nearest"
        }
        interp = interp_map.get(interpolation.lower() if isinstance(interpolation, str) else "cubic", "cubic")

        # Try native method first
        if hasattr(obj, 'get_value_at_time'):
            return obj.get_value_at_time(time, interpolation=interp)

        # Fall back to manual interpolation (linear only for now)
        times = _get_harmonicity_times(obj)
        values = _get_harmonicity_values(obj)
        effective_interp = "linear" if interp == "cubic" else interp  # Simplify cubic to linear
        return _interpolate_at_time(times, values, time, effective_interp, undefined_value=-200.0)

    # Get value in frame (1-based)
    elif cmd == "get value in frame" or cmd == "get value in frame...":
        frame_number = int(args[0])
        # Convert 1-based to 0-based
        frame_idx = frame_number - 1
        if frame_idx < 0 or frame_idx >= _get_n_frames(obj):
            return None
        values = _get_harmonicity_values(obj)
        return float(values[frame_idx])

    else:
        raise PraatCallError(f"Unknown Harmonicity command: '{command}'")


# =============================================================================
# Spectrum commands
# =============================================================================

def _get_spectrum_values(obj: Any) -> np.ndarray:
    """Get spectrum magnitude values, handling both API styles.

    Always returns real-valued magnitude (absolute value for complex spectra).
    """
    if hasattr(obj, 'real') and hasattr(obj, 'imag'):
        # Prefer explicit real/imag to compute magnitude correctly
        real = np.array(obj.real() if callable(obj.real) else obj.real)
        imag = np.array(obj.imag() if callable(obj.imag) else obj.imag)
        return np.sqrt(real**2 + imag**2)
    if hasattr(obj, 'values') and callable(obj.values):
        vals = np.array(obj.values())
        if np.iscomplexobj(vals):
            return np.abs(vals)
        return vals
    elif hasattr(obj, 'values'):
        vals = np.array(obj.values)
        if np.iscomplexobj(vals):
            return np.abs(vals)
        return vals
    return np.array([])


def _get_spectrum_freqs(obj: Any) -> np.ndarray:
    """Get spectrum frequency values, handling both API styles."""
    if hasattr(obj, 'xs') and callable(obj.xs):
        return np.array(obj.xs())
    elif hasattr(obj, 'df'):
        n_bins = obj.n_bins
        return np.arange(n_bins) * obj.df
    return np.array([])


def _spectrum_central_moment(freqs: np.ndarray, magnitude: np.ndarray, n: int, power: float) -> float:
    """Compute nth central moment of spectrum."""
    weighted = magnitude ** power
    denominator = np.sum(weighted)
    if denominator == 0:
        return 0.0

    cog = np.sum(freqs * weighted) / denominator

    if n == 1:
        return cog

    deviation = freqs - cog
    numerator = np.sum((deviation ** n) * weighted)
    return numerator / denominator


def _call_spectrum(obj: Any, command: str, args: tuple) -> Any:
    """Handle call() for Spectrum objects."""
    cmd = _normalize_command(command)

    # Get centre of gravity (British spelling in Praat)
    if cmd == "get centre of gravity" or cmd == "get centre of gravity...":
        power = args[0] if len(args) > 0 else 2.0
        if hasattr(obj, 'get_center_of_gravity'):
            return obj.get_center_of_gravity(power=power)
        # Manual calculation
        freqs = _get_spectrum_freqs(obj)
        magnitude = _get_spectrum_values(obj)
        weighted = magnitude ** power
        denominator = np.sum(weighted)
        if denominator == 0:
            return 0.0
        return float(np.sum(freqs * weighted) / denominator)

    # Get center of gravity (American spelling)
    elif cmd == "get center of gravity" or cmd == "get center of gravity...":
        power = args[0] if len(args) > 0 else 2.0
        if hasattr(obj, 'get_center_of_gravity'):
            return obj.get_center_of_gravity(power=power)
        # Manual calculation
        freqs = _get_spectrum_freqs(obj)
        magnitude = _get_spectrum_values(obj)
        weighted = magnitude ** power
        denominator = np.sum(weighted)
        if denominator == 0:
            return 0.0
        return float(np.sum(freqs * weighted) / denominator)

    # Get standard deviation
    elif cmd == "get standard deviation" or cmd == "get standard deviation...":
        power = args[0] if len(args) > 0 else 2.0
        if hasattr(obj, 'get_standard_deviation'):
            return obj.get_standard_deviation(power=power)
        # Manual calculation
        freqs = _get_spectrum_freqs(obj)
        magnitude = _get_spectrum_values(obj)
        mu2 = _spectrum_central_moment(freqs, magnitude, 2, power)
        return float(np.sqrt(mu2))

    # Get skewness
    elif cmd == "get skewness" or cmd == "get skewness...":
        power = args[0] if len(args) > 0 else 2.0
        if hasattr(obj, 'get_skewness'):
            return obj.get_skewness(power=power)
        # Manual calculation
        freqs = _get_spectrum_freqs(obj)
        magnitude = _get_spectrum_values(obj)
        mu2 = _spectrum_central_moment(freqs, magnitude, 2, power)
        mu3 = _spectrum_central_moment(freqs, magnitude, 3, power)
        if mu2 == 0:
            return 0.0
        return float(mu3 / (mu2 ** 1.5))

    # Get kurtosis
    elif cmd == "get kurtosis" or cmd == "get kurtosis...":
        power = args[0] if len(args) > 0 else 2.0
        if hasattr(obj, 'get_kurtosis'):
            return obj.get_kurtosis(power=power)
        # Manual calculation
        freqs = _get_spectrum_freqs(obj)
        magnitude = _get_spectrum_values(obj)
        mu2 = _spectrum_central_moment(freqs, magnitude, 2, power)
        mu4 = _spectrum_central_moment(freqs, magnitude, 4, power)
        if mu2 == 0:
            return 0.0
        return float(mu4 / (mu2 ** 2) - 3.0)

    # Get band energy
    elif cmd == "get band energy" or cmd == "get band energy...":
        f_min = args[0] if len(args) > 0 else 0.0
        f_max = args[1] if len(args) > 1 else 0.0
        if hasattr(obj, 'get_band_energy'):
            return obj.get_band_energy(f_min=f_min, f_max=f_max)
        # Manual calculation with negative frequency correction
        freqs = _get_spectrum_freqs(obj)
        magnitude = _get_spectrum_values(obj)
        if f_max <= 0:
            f_max = freqs[-1] if len(freqs) > 0 else 0
        df = obj.df if hasattr(obj, 'df') else (freqs[1] - freqs[0] if len(freqs) > 1 else 1)
        n_bins = len(freqs)
        # Sum energy with 2x factor for interior bins (one-sided spectrum)
        # DC (bin 0) and Nyquist (last bin) are not doubled
        energy = 0.0
        for i in range(n_bins):
            if freqs[i] < f_min or freqs[i] > f_max:
                continue
            bin_energy = magnitude[i] ** 2 * df
            if i == 0 or i == n_bins - 1:
                energy += bin_energy
            else:
                energy += 2.0 * bin_energy
        return float(energy)

    else:
        raise PraatCallError(f"Unknown Spectrum command: '{command}'")


# =============================================================================
# Spectrogram commands
# =============================================================================

def _get_spectrogram_times(obj: Any) -> np.ndarray:
    """Get spectrogram time values, handling both API styles."""
    if hasattr(obj, 'xs') and callable(obj.xs):
        return np.array(obj.xs())
    elif hasattr(obj, 'times') and callable(obj.times):
        return np.array(obj.times())
    return np.array([])


def _get_spectrogram_freqs(obj: Any) -> np.ndarray:
    """Get spectrogram frequency values, handling both API styles."""
    if hasattr(obj, 'ys') and callable(obj.ys):
        return np.array(obj.ys())
    elif hasattr(obj, 'frequencies') and callable(obj.frequencies):
        return np.array(obj.frequencies())
    return np.array([])


def _get_spectrogram_n_times(obj: Any) -> int:
    """Get number of time frames from a spectrogram object."""
    if hasattr(obj, 'n_times'):
        n = obj.n_times
        return n() if callable(n) else n
    if hasattr(obj, 'num_frames'):
        return obj.num_frames
    return 0


def _get_spectrogram_n_freqs(obj: Any) -> int:
    """Get number of frequency bins from a spectrogram object."""
    if hasattr(obj, 'n_freqs'):
        n = obj.n_freqs
        return n() if callable(n) else n
    if hasattr(obj, 'num_freq_bins'):
        return obj.num_freq_bins
    return 0


def _get_spectrogram_values(obj: Any) -> np.ndarray:
    """Get spectrogram power values as 2D array (n_freqs × n_times)."""
    if hasattr(obj, 'values'):
        vals = obj.values() if callable(obj.values) else obj.values
        return np.array(vals)
    return np.array([])


def _call_spectrogram(obj: Any, command: str, args: tuple) -> Any:
    """Handle call() for Spectrogram objects."""
    cmd = _normalize_command(command)

    # Get number of frames (n_times)
    if cmd == "get number of frames":
        return _get_spectrogram_n_times(obj)

    # Get number of frequencies
    elif cmd == "get number of frequencies":
        return _get_spectrogram_n_freqs(obj)

    # Get time from frame number (1-based)
    elif cmd == "get time from frame number" or cmd == "get time from frame number...":
        frame_number = int(args[0])
        # Convert 1-based to 0-based
        frame_idx = frame_number - 1
        n_times = _get_spectrogram_n_times(obj)
        if frame_idx < 0 or frame_idx >= n_times:
            return None
        times = _get_spectrogram_times(obj)
        return float(times[frame_idx])

    # Get power at (time, frequency)
    elif cmd == "get power at" or cmd == "get power at...":
        time = args[0]
        freq = args[1]

        times = _get_spectrogram_times(obj)
        freqs = _get_spectrogram_freqs(obj)

        if len(times) == 0 or len(freqs) == 0:
            return None

        n_times = _get_spectrogram_n_times(obj)
        n_freqs = _get_spectrogram_n_freqs(obj)

        # Find nearest time frame
        time_step = times[1] - times[0] if len(times) > 1 else getattr(obj, 'time_step', 0.002)
        time_idx = int(round((time - times[0]) / time_step))
        time_idx = max(0, min(n_times - 1, time_idx))

        # Find nearest frequency bin
        freq_step = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        freq_idx = int(round((freq - freqs[0]) / freq_step))
        freq_idx = max(0, min(n_freqs - 1, freq_idx))

        values = _get_spectrogram_values(obj)
        if values.ndim == 2:
            return float(values[freq_idx, time_idx])
        return None

    else:
        raise PraatCallError(f"Unknown Spectrogram command: '{command}'")


# =============================================================================
# Main call() function
# =============================================================================

def call(obj: Any, command: str, *args) -> Any:
    """
    Execute a Praat command on an object.

    This function emulates parselmouth's `call()` function, allowing
    existing parselmouth scripts to work with praatfan objects.

    Args:
        obj: A praatfan object (Sound, Pitch, Formant, etc.)
        command: The Praat command string (e.g., "To Pitch (ac)")
        *args: Additional arguments for the command

    Returns:
        The result of the command (varies by command type)

    Raises:
        PraatCallError: If the command is not recognized or fails

    Examples:
        >>> sound = praatfan.Sound.from_file("audio.wav")
        >>> pitch = call(sound, "To Pitch (ac)", 0, 75, 600)
        >>> f0 = call(pitch, "Get value in frame", 10, "Hertz")
        >>> n_frames = call(pitch, "Get number of frames")

    Note:
        Frame numbers in commands are 1-based (like Praat/parselmouth),
        while praatfan's internal API is 0-based. This function handles
        the conversion automatically.
    """
    # Dispatch based on object type
    if _is_sound(obj):
        return _call_sound(obj, command, args)
    elif _is_pitch(obj):
        return _call_pitch(obj, command, args)
    elif _is_formant(obj):
        return _call_formant(obj, command, args)
    elif _is_intensity(obj):
        return _call_intensity(obj, command, args)
    elif _is_harmonicity(obj):
        return _call_harmonicity(obj, command, args)
    elif _is_spectrum(obj):
        return _call_spectrum(obj, command, args)
    elif _is_spectrogram(obj):
        return _call_spectrogram(obj, command, args)
    else:
        raise PraatCallError(
            f"Unknown object type: {type(obj).__name__}. "
            f"Expected Sound, Pitch, Formant, Intensity, Harmonicity, "
            f"Spectrum, or Spectrogram."
        )
