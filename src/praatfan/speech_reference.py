"""
Speech-referenced amplitude normalization (estimator).

Implements the estimator from SPEC-speech-reference-normalization.md with
the exact algorithm pinned in DECISIONS-speech-reference-normalization.md.
This is NOT a Praat algorithm — it is a project-defined estimator shared
across praatfan / praatfan-rust / praatfan-gpl and sibling analysis
packages so one ``reference_peak`` (and the z-norm ``mean`` / ``std``) can
flow through the whole analysis stack.

Why it exists: pitch (and harmonicity) reference every frame's amplitude
against a whole-file statistic (``global_peak``). On long conversational
recordings, one loud event anywhere depresses ``local_intensity`` file-wide
and forces quiet voiced frames unvoiced. The estimator computes a norming
standard from speech-like frames only, using **frame-level robust
statistics** so a short loud burst (a tiny fraction of the speech frames)
cannot dominate it.

Algorithm (all constants normative, see DECISIONS §1–2):
1. Frame the signal (``frame_s`` windows, ``hop_s`` hop, full frames from
   sample 0); per-frame RMS in dB.
2. ``speech_mask`` = frames whose dB level is within ``speech_floor_db``
   of the 95th-percentile dB level (inclusive). Silence drops out; loud
   bursts stay in but get only bounded leverage in step 3.
3. Standards from frame-level robust statistics over the speech-masked
   frames (NOT sample-level moments, which stay burst-dominated):
   - ``std`` = median per-frame RMS (the z-norm scale)
   - ``mean`` = median per-frame mean (DC reference)
   - ``reference_peak`` = ``reference_percentile``-ile (default 75) of
     the per-frame peak |x| — a TYPICAL-speech peak, deliberately not
     the loudest. The contamination cliff is (100−p)% of speech-masked
     time, so p=75 tolerates 25% loud non-speech (vs the retired v1
     sample-domain p=98, whose cliff was only 2%).

All percentiles use linear interpolation (Hyndman & Fan type 7, numpy's
default) so the Rust port can reproduce them exactly.
"""

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechReference:
    """Result of :func:`estimate_speech_reference`.

    Attributes:
        speech_mask: Per-frame bool array — "this frame looks like speech".
        frame_times: Per-frame center times in seconds.
        mean: Median per-frame mean over speech frames (DC reference).
        std: Median per-frame RMS over speech frames (the z-norm scale).
        reference_peak: ``reference_percentile``-ile of per-frame peak |x|
            over speech frames (the norming standard for Praat-style
            consumers). 0.0 for an all-zero / empty signal.
        speech_fraction: Fraction of frames in the speech mask.
    """
    speech_mask: np.ndarray
    frame_times: np.ndarray
    mean: float
    std: float
    reference_peak: float
    speech_fraction: float


def estimate_speech_reference(
    samples,
    sample_rate: float,
    *,
    frame_s: float = 0.05,
    hop_s: float = 0.01,
    speech_floor_db: float = 30.0,
    reference_percentile: float = 75.0,
) -> SpeechReference:
    """Estimate a speech-scoped amplitude reference for normalization.

    Pure function, no analysis side effects. Run once per recording; the
    result is shared across pitch, HNR, and downstream normalization.

    Robustness: silence is masked out (step 2) so it cannot dilute the
    scale; a short loud burst is only a few frames, so the frame-level
    median / percentile in step 3 give it bounded leverage.

    Args:
        samples: Mono audio samples (converted to float64).
        sample_rate: Sample rate in Hz.
        frame_s: Analysis frame length in seconds.
        hop_s: Hop between frame starts in seconds.
        speech_floor_db: Frames within this many dB of the 95th-percentile
            frame level count as speech.
        reference_percentile: Percentile of per-frame peak |x| (over
            speech frames) used as the reference peak. The contamination
            cliff is (100−p)% of speech-masked time.

    Returns:
        SpeechReference(speech_mask, frame_times, mean, std, reference_peak,
        speech_fraction)
    """
    samples = np.asarray(samples, dtype=np.float64)
    sample_rate = float(sample_rate)
    n = len(samples)
    if n == 0:
        return SpeechReference(
            np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64),
            0.0, 1.0, 0.0, 0.0,
        )

    # Round half away from zero (Python round() is half-to-even; the Rust
    # port uses f64::round which is half-away — keep them identical).
    n_frame = max(1, int(np.floor(frame_s * sample_rate + 0.5)))
    n_hop = max(1, int(np.floor(hop_s * sample_rate + 0.5)))

    if n < n_frame:
        # Short signal: one frame covering all samples; treat as speech.
        std = float(samples.std())
        return SpeechReference(
            np.ones(1, dtype=bool),
            np.array([n / (2.0 * sample_rate)]),
            float(samples.mean()),
            std if std > 0.0 else 1.0,
            float(np.abs(samples).max()),
            1.0,
        )

    n_frames = 1 + (n - n_frame) // n_hop
    starts = np.arange(n_frames, dtype=np.int64) * n_hop
    frame_times = (starts + n_frame / 2.0) / sample_rate

    # Per-frame reductions over the strided view without materializing the
    # n_frames × n_frame matrix (einsum/reduce iterate the view in place).
    windows = np.lib.stride_tricks.sliding_window_view(samples, n_frame)[::n_hop]
    sumsq = np.einsum("ij,ij->i", windows, windows)
    frame_mean = windows.mean(axis=1)
    frame_peak = np.maximum(windows.max(axis=1), -windows.min(axis=1))

    # +1e-20 keeps silent frames finite in dB (matches the cross-package
    # reference; negligible for any real speech frame).
    rms = np.sqrt(sumsq / n_frame + 1e-20)
    db = 20.0 * np.log10(rms + 1e-20)

    ceiling_db = float(np.percentile(db, 95.0))
    speech_mask = db >= (ceiling_db - speech_floor_db)
    if not speech_mask.any():
        speech_mask = np.ones_like(speech_mask)

    sp = speech_mask
    std = float(np.median(rms[sp]))
    return SpeechReference(
        speech_mask,
        frame_times,
        float(np.median(frame_mean[sp])),
        std if std > 0.0 else 1.0,
        float(np.percentile(frame_peak[sp], reference_percentile)),
        float(sp.mean()),
    )


def resolve_reference_peak(samples, sample_rate, reference_peak):
    """Resolve a ``reference_peak`` argument for the ``*_referenced`` calls.

    Returns a positive float to substitute for the whole-file statistic,
    or None meaning "use the legacy whole-file statistic" (only reachable
    when the internal estimator returns 0, i.e. an all-zero signal).

    Raises:
        ValueError: If an explicit reference_peak is not finite and > 0.
    """
    if reference_peak is None:
        ref = estimate_speech_reference(samples, sample_rate).reference_peak
        return ref if ref > 0.0 else None
    ref = float(reference_peak)
    if not np.isfinite(ref) or ref <= 0.0:
        raise ValueError(
            f"reference_peak must be finite and > 0, got {reference_peak!r}"
        )
    return ref
