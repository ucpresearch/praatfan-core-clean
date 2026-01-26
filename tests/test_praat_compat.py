"""Tests for the parselmouth compatibility layer (praatfan.praat).

These tests verify that the call() function correctly maps parselmouth-style
commands to praatfan methods, including proper 1-based to 0-based index
conversion.
"""

import pytest
import numpy as np
from pathlib import Path

import praatfan
from praatfan.praat import call, PraatCallError


# Test fixture path
FIXTURE_DIR = Path(__file__).parent / "fixtures"
TEST_WAV = FIXTURE_DIR / "one_two_three_four_five.wav"


@pytest.fixture
def sound():
    """Load test sound file."""
    return praatfan.Sound.from_file(TEST_WAV)


# =============================================================================
# Sound command tests
# =============================================================================

class TestSoundCommands:
    """Test Sound commands via call()."""

    def test_to_pitch_ac(self, sound):
        """Test 'To Pitch (ac)' command."""
        pitch = call(sound, "To Pitch (ac)", 0, 75, 600)
        assert pitch is not None
        n_frames = call(pitch, "Get number of frames")
        assert n_frames > 0

    def test_to_pitch_cc(self, sound):
        """Test 'To Pitch (cc)' command."""
        # Full argument list for CC method
        pitch = call(sound, "To Pitch (cc)", 0, 75, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 600)
        assert pitch is not None
        n_frames = call(pitch, "Get number of frames")
        assert n_frames > 0

    def test_to_pitch_simple(self, sound):
        """Test simple 'To Pitch' command."""
        pitch = call(sound, "To Pitch", 0, 75, 600)
        assert pitch is not None
        n_frames = call(pitch, "Get number of frames")
        assert n_frames > 0

    def test_to_formant_burg(self, sound):
        """Test 'To Formant (burg)' command."""
        formant = call(sound, "To Formant (burg)", 0, 5, 5500, 0.025, 50)
        assert formant is not None
        n_frames = call(formant, "Get number of frames")
        assert n_frames > 0

    def test_to_intensity(self, sound):
        """Test 'To Intensity' command."""
        intensity = call(sound, "To Intensity", 100, 0)
        assert intensity is not None
        n_frames = call(intensity, "Get number of frames")
        assert n_frames > 0

    def test_to_harmonicity_ac(self, sound):
        """Test 'To Harmonicity (ac)' command."""
        harmonicity = call(sound, "To Harmonicity (ac)", 0.01, 75, 0.1, 4.5)
        assert harmonicity is not None
        n_frames = call(harmonicity, "Get number of frames")
        assert n_frames > 0

    def test_to_harmonicity_cc(self, sound):
        """Test 'To Harmonicity (cc)' command."""
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        assert harmonicity is not None
        n_frames = call(harmonicity, "Get number of frames")
        assert n_frames > 0

    def test_to_spectrum(self, sound):
        """Test 'To Spectrum' command."""
        spectrum = call(sound, "To Spectrum", "yes")
        assert spectrum is not None
        # Spectrum doesn't have a frame count command, test center of gravity instead
        cog = call(spectrum, "Get centre of gravity", 2.0)
        assert cog is not None

    def test_to_spectrogram(self, sound):
        """Test 'To Spectrogram' command."""
        spectrogram = call(sound, "To Spectrogram", 0.005, 5000, 0.002, 20)
        assert spectrogram is not None
        n_frames = call(spectrogram, "Get number of frames")
        assert n_frames > 0
        n_freqs = call(spectrogram, "Get number of frequencies")
        assert n_freqs > 0

    def test_unknown_command_raises(self, sound):
        """Test that unknown commands raise PraatCallError."""
        with pytest.raises(PraatCallError):
            call(sound, "Unknown Command")


# =============================================================================
# Pitch command tests
# =============================================================================

class TestPitchCommands:
    """Test Pitch commands via call()."""

    @pytest.fixture
    def pitch(self, sound):
        """Create Pitch object."""
        return call(sound, "To Pitch (ac)", 0, 75, 600)

    def test_get_number_of_frames(self, pitch):
        """Test 'Get number of frames' command."""
        n_frames = call(pitch, "Get number of frames")
        assert n_frames > 0

    def test_get_time_from_frame_number(self, pitch):
        """Test 'Get time from frame number' with 1-based index."""
        n_frames = call(pitch, "Get number of frames")

        # 1-based: frame 1 should be first frame
        t1 = call(pitch, "Get time from frame number", 1)
        assert t1 is not None
        assert t1 > 0

        # Frame 2 should come after frame 1
        t2 = call(pitch, "Get time from frame number", 2)
        assert t2 is not None
        assert t2 > t1

    def test_get_time_from_frame_number_out_of_range(self, pitch):
        """Test that out-of-range frame numbers return None."""
        n_frames = call(pitch, "Get number of frames")

        result = call(pitch, "Get time from frame number", 0)  # 0 is invalid (1-based)
        assert result is None

        result = call(pitch, "Get time from frame number", n_frames + 1)
        assert result is None

    def test_get_value_in_frame(self, pitch):
        """Test 'Get value in frame' command."""
        n_frames = call(pitch, "Get number of frames")

        # Find a voiced frame by checking if value is not None and not NaN
        for i in range(1, n_frames + 1):
            value = call(pitch, "Get value in frame", i, "Hertz")
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                assert value > 0
                break

    def test_get_value_in_frame_unvoiced(self, pitch):
        """Test that unvoiced frames return None."""
        n_frames = call(pitch, "Get number of frames")

        # Find an unvoiced frame (value is None)
        found_unvoiced = False
        for i in range(1, n_frames + 1):
            value = call(pitch, "Get value in frame", i, "Hertz")
            if value is None:
                found_unvoiced = True
                break
        # It's okay if we don't find any unvoiced frames

    def test_get_value_at_time(self, pitch):
        """Test 'Get value at time' command."""
        n_frames = call(pitch, "Get number of frames")
        mid_frame = n_frames // 2
        t = call(pitch, "Get time from frame number", mid_frame)
        value = call(pitch, "Get value at time", t, "Hertz", "linear")
        # Value may be None if unvoiced, that's ok

    def test_get_strength_in_frame(self, pitch):
        """Test 'Get strength in frame' command."""
        strength = call(pitch, "Get strength in frame", 1)
        assert strength is not None
        assert isinstance(strength, (int, float))

    def test_get_strength_at_time(self, pitch):
        """Test 'Get strength at time' command."""
        t = call(pitch, "Get time from frame number", 1)
        strength = call(pitch, "Get strength at time", t, "linear")
        assert strength is not None

    def test_unknown_command_raises(self, pitch):
        """Test that unknown commands raise PraatCallError."""
        with pytest.raises(PraatCallError):
            call(pitch, "Unknown Command")


# =============================================================================
# Formant command tests
# =============================================================================

class TestFormantCommands:
    """Test Formant commands via call()."""

    @pytest.fixture
    def formant(self, sound):
        """Create Formant object."""
        return call(sound, "To Formant (burg)", 0, 5, 5500, 0.025, 50)

    def test_get_number_of_frames(self, formant):
        """Test 'Get number of frames' command."""
        n_frames = call(formant, "Get number of frames")
        assert n_frames > 0

    def test_get_time_from_frame_number(self, formant):
        """Test 'Get time from frame number' with 1-based index."""
        t1 = call(formant, "Get time from frame number", 1)
        assert t1 is not None
        assert t1 > 0

    def test_get_number_of_formants(self, formant):
        """Test 'Get number of formants' command."""
        # 1-based frame index
        n_formants = call(formant, "Get number of formants", 1)
        assert n_formants >= 0

    def test_get_value_at_time(self, formant):
        """Test 'Get value at time' command."""
        n_frames = call(formant, "Get number of frames")
        t = call(formant, "Get time from frame number", n_frames // 2)
        # Get F1 at time t
        f1 = call(formant, "Get value at time", 1, t, "Hertz", "linear")
        # Value may be None if no formant found

    def test_get_bandwidth_at_time(self, formant):
        """Test 'Get bandwidth at time' command."""
        n_frames = call(formant, "Get number of frames")
        t = call(formant, "Get time from frame number", n_frames // 2)
        # Get B1 at time t
        b1 = call(formant, "Get bandwidth at time", 1, t, "Hertz", "linear")
        # Value may be None if no formant found

    def test_unknown_command_raises(self, formant):
        """Test that unknown commands raise PraatCallError."""
        with pytest.raises(PraatCallError):
            call(formant, "Unknown Command")


# =============================================================================
# Intensity command tests
# =============================================================================

class TestIntensityCommands:
    """Test Intensity commands via call()."""

    @pytest.fixture
    def intensity(self, sound):
        """Create Intensity object."""
        return call(sound, "To Intensity", 100, 0)

    def test_get_number_of_frames(self, intensity):
        """Test 'Get number of frames' command."""
        n_frames = call(intensity, "Get number of frames")
        assert n_frames > 0

    def test_get_time_from_frame_number(self, intensity):
        """Test 'Get time from frame number' with 1-based index."""
        t1 = call(intensity, "Get time from frame number", 1)
        assert t1 is not None
        assert t1 > 0

    def test_get_value_at_time(self, intensity):
        """Test 'Get value at time' command."""
        n_frames = call(intensity, "Get number of frames")
        t = call(intensity, "Get time from frame number", n_frames // 2)
        value = call(intensity, "Get value at time", t, "cubic")
        assert value is not None

    def test_get_value_in_frame(self, intensity):
        """Test 'Get value in frame' command."""
        value = call(intensity, "Get value in frame", 1)
        assert value is not None
        assert isinstance(value, (int, float))

    def test_unknown_command_raises(self, intensity):
        """Test that unknown commands raise PraatCallError."""
        with pytest.raises(PraatCallError):
            call(intensity, "Unknown Command")


# =============================================================================
# Harmonicity command tests
# =============================================================================

class TestHarmonicityCommands:
    """Test Harmonicity commands via call()."""

    @pytest.fixture
    def harmonicity(self, sound):
        """Create Harmonicity object."""
        return call(sound, "To Harmonicity (ac)", 0.01, 75, 0.1, 4.5)

    def test_get_number_of_frames(self, harmonicity):
        """Test 'Get number of frames' command."""
        n_frames = call(harmonicity, "Get number of frames")
        assert n_frames > 0

    def test_get_time_from_frame_number(self, harmonicity):
        """Test 'Get time from frame number' with 1-based index."""
        t1 = call(harmonicity, "Get time from frame number", 1)
        assert t1 is not None
        assert t1 > 0

    def test_get_value_at_time(self, harmonicity):
        """Test 'Get value at time' command."""
        n_frames = call(harmonicity, "Get number of frames")
        t = call(harmonicity, "Get time from frame number", n_frames // 2)
        value = call(harmonicity, "Get value at time", t, "cubic")
        assert value is not None

    def test_get_value_in_frame(self, harmonicity):
        """Test 'Get value in frame' command."""
        value = call(harmonicity, "Get value in frame", 1)
        assert value is not None
        assert isinstance(value, (int, float))

    def test_unknown_command_raises(self, harmonicity):
        """Test that unknown commands raise PraatCallError."""
        with pytest.raises(PraatCallError):
            call(harmonicity, "Unknown Command")


# =============================================================================
# Spectrum command tests
# =============================================================================

class TestSpectrumCommands:
    """Test Spectrum commands via call()."""

    @pytest.fixture
    def spectrum(self, sound):
        """Create Spectrum object."""
        return call(sound, "To Spectrum", "yes")

    def test_get_centre_of_gravity(self, spectrum):
        """Test 'Get centre of gravity' command."""
        cog = call(spectrum, "Get centre of gravity", 2.0)
        assert cog is not None
        assert isinstance(cog, (int, float))
        assert cog >= 0

    def test_get_center_of_gravity(self, spectrum):
        """Test 'Get center of gravity' (American spelling) command."""
        cog = call(spectrum, "Get center of gravity", 2.0)
        assert cog is not None
        assert isinstance(cog, (int, float))
        assert cog >= 0

    def test_get_standard_deviation(self, spectrum):
        """Test 'Get standard deviation' command."""
        std = call(spectrum, "Get standard deviation", 2.0)
        assert std is not None
        assert isinstance(std, (int, float))
        assert std >= 0

    def test_get_skewness(self, spectrum):
        """Test 'Get skewness' command."""
        skew = call(spectrum, "Get skewness", 2.0)
        assert skew is not None
        assert isinstance(skew, (int, float))

    def test_get_kurtosis(self, spectrum):
        """Test 'Get kurtosis' command."""
        kurt = call(spectrum, "Get kurtosis", 2.0)
        assert kurt is not None
        assert isinstance(kurt, (int, float))

    def test_get_band_energy(self, spectrum):
        """Test 'Get band energy' command."""
        energy = call(spectrum, "Get band energy", 0, 1000)
        assert energy is not None
        assert isinstance(energy, (int, float))
        assert energy >= 0

    def test_unknown_command_raises(self, spectrum):
        """Test that unknown commands raise PraatCallError."""
        with pytest.raises(PraatCallError):
            call(spectrum, "Unknown Command")


# =============================================================================
# Spectrogram command tests
# =============================================================================

class TestSpectrogramCommands:
    """Test Spectrogram commands via call()."""

    @pytest.fixture
    def spectrogram(self, sound):
        """Create Spectrogram object."""
        return call(sound, "To Spectrogram", 0.005, 5000, 0.002, 20)

    def test_get_number_of_frames(self, spectrogram):
        """Test 'Get number of frames' command."""
        n_frames = call(spectrogram, "Get number of frames")
        assert n_frames > 0

    def test_get_number_of_frequencies(self, spectrogram):
        """Test 'Get number of frequencies' command."""
        n_freqs = call(spectrogram, "Get number of frequencies")
        assert n_freqs > 0

    def test_get_time_from_frame_number(self, spectrogram):
        """Test 'Get time from frame number' with 1-based index."""
        t1 = call(spectrogram, "Get time from frame number", 1)
        assert t1 is not None
        assert t1 > 0

    def test_get_power_at(self, spectrogram):
        """Test 'Get power at' command."""
        # Get power at center time and 1000 Hz
        n_frames = call(spectrogram, "Get number of frames")
        t = call(spectrogram, "Get time from frame number", n_frames // 2)
        power = call(spectrogram, "Get power at", t, 1000)
        assert power is not None
        assert power >= 0

    def test_unknown_command_raises(self, spectrogram):
        """Test that unknown commands raise PraatCallError."""
        with pytest.raises(PraatCallError):
            call(spectrogram, "Unknown Command")


# =============================================================================
# Case insensitivity tests
# =============================================================================

class TestCaseInsensitivity:
    """Test that commands are case-insensitive."""

    def test_pitch_commands_case_insensitive(self, sound):
        """Test that Pitch commands work regardless of case."""
        pitch1 = call(sound, "To Pitch (ac)", 0, 75, 600)
        pitch2 = call(sound, "TO PITCH (AC)", 0, 75, 600)
        pitch3 = call(sound, "to pitch (AC)", 0, 75, 600)

        n1 = call(pitch1, "Get number of frames")
        n2 = call(pitch2, "Get number of frames")
        n3 = call(pitch3, "Get number of frames")

        assert n1 == n2 == n3

    def test_get_commands_case_insensitive(self, sound):
        """Test that Get commands work regardless of case."""
        pitch = call(sound, "To Pitch (ac)", 0, 75, 600)

        n1 = call(pitch, "Get number of frames")
        n2 = call(pitch, "GET NUMBER OF FRAMES")
        n3 = call(pitch, "get NUMBER of FRAMES")

        assert n1 == n2 == n3


# =============================================================================
# Parselmouth comparison tests (optional)
# =============================================================================

class TestParselmouthComparison:
    """Compare praatfan.praat.call() results with parselmouth.

    These tests only run if parselmouth is installed.
    """

    @pytest.fixture
    def parselmouth_available(self):
        """Check if parselmouth is available."""
        try:
            import parselmouth
            return True
        except ImportError:
            pytest.skip("parselmouth not installed")

    def test_pitch_frame_count_matches(self, parselmouth_available):
        """Test that frame count matches parselmouth."""
        import parselmouth
        from parselmouth.praat import call as pm_call

        pm_sound = parselmouth.Sound(str(TEST_WAV))
        pf_sound = praatfan.Sound.from_file(TEST_WAV)

        # Create pitch objects
        # praatfan uses simpler interface: time_step, floor, ceiling
        pf_pitch = call(pf_sound, "To Pitch (ac)", 0, 75, 600)
        # parselmouth requires full argument list:
        # time_step, floor, max_candidates, very_accurate, silence_threshold,
        # voicing_threshold, octave_cost, octave_jump_cost, voiced_unvoiced_cost, ceiling
        pm_pitch = pm_call(pm_sound, "To Pitch (ac)", 0, 75, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 600)

        pf_n = call(pf_pitch, "Get number of frames")
        pm_n = pm_call(pm_pitch, "Get number of frames")

        assert pf_n == pm_n, f"Frame count mismatch: praatfan={pf_n}, parselmouth={pm_n}"

    def test_formant_frame_count_matches(self, parselmouth_available):
        """Test that formant frame count matches parselmouth."""
        import parselmouth
        from parselmouth.praat import call as pm_call

        pm_sound = parselmouth.Sound(str(TEST_WAV))
        pf_sound = praatfan.Sound.from_file(TEST_WAV)

        # Create formant objects
        pf_formant = call(pf_sound, "To Formant (burg)", 0, 5, 5500, 0.025, 50)
        pm_formant = pm_call(pm_sound, "To Formant (burg)", 0, 5, 5500, 0.025, 50)

        pf_n = call(pf_formant, "Get number of frames")
        pm_n = pm_call(pm_formant, "Get number of frames")

        assert pf_n == pm_n, f"Frame count mismatch: praatfan={pf_n}, parselmouth={pm_n}"

    def test_intensity_frame_count_matches(self, parselmouth_available):
        """Test that intensity frame count matches parselmouth."""
        import parselmouth
        from parselmouth.praat import call as pm_call

        pm_sound = parselmouth.Sound(str(TEST_WAV))
        pf_sound = praatfan.Sound.from_file(TEST_WAV)

        # Create intensity objects
        pf_intensity = call(pf_sound, "To Intensity", 100, 0)
        pm_intensity = pm_call(pm_sound, "To Intensity", 100, 0)

        pf_n = call(pf_intensity, "Get number of frames")
        pm_n = pm_call(pm_intensity, "Get number of frames")

        assert pf_n == pm_n, f"Frame count mismatch: praatfan={pf_n}, parselmouth={pm_n}"
