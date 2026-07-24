#' praatfan — Praat-compatible acoustic analysis in R
#'
#' R wrapper around the praatfan Rust engine, a clean-room reimplementation
#' of Praat's core acoustic analysis algorithms validated frame-by-frame
#' against Praat.  All analysis runs in the \code{praatfan-open-pipe} binary
#' (JSON stdin/stdout); no Python required.
#'
#' Quick start:
#'   library(praatfan)
#'   pf_setup()                          # one-time install
#'   f0  <- pf_pitch("audio.wav")
#'   fmt <- pf_formants("audio.wav")
#'
#' All analysis functions accept audio in any format the engine reads
#' (WAV, FLAC, MP3, OGG, AIFF, NIST SPHERE incl. shorten).  Mono only;
#' pass \code{channel} to select one channel of a multi-channel file.
#'
#' @author praatfan contributors

# ---------------------------------------------------------------------------
# Batch interface
# ---------------------------------------------------------------------------

#' Run a batch of analyses against one audio file.
#'
#' The audio file is loaded once and every analysis in \code{analyses} runs
#' against it — the cheapest way to compute several measures per file.  Most
#' users want the typed wrappers (\code{\link{pf_pitch}},
#' \code{\link{pf_formants}}, ...) instead; this is the raw interface.
#'
#' @param wav      Path to an audio file.
#' @param analyses A list of analysis specs, each a named list with a
#'   \code{type} field (\code{"pitch_ac"}, \code{"pitch_cc"},
#'   \code{"formant_burg"}, \code{"intensity"}, \code{"harmonicity_ac"},
#'   \code{"harmonicity_cc"}, \code{"spectral_moments"},
#'   \code{"band_energy"}) plus that analysis's parameters.  Omitted
#'   parameters take Praat's defaults; unknown parameter names are errors.
#'   Vector-valued fields (e.g. \code{times}) should be wrapped in
#'   \code{I()} so length-1 vectors stay JSON arrays.
#' @param channel  0-based channel to read from a multi-channel file.
#'   Default \code{NULL} (file must be mono).
#' @param engine   Engine for this call: \code{"open"} or \code{"gpl"}.
#'   Default \code{NULL} = the session engine (see \code{\link{pf_engine}}).
#'   Both engines speak the same schema, so this is handy for side-by-side
#'   comparisons.
#' @return The parsed response: a list with \code{duration},
#'   \code{sample_rate}, \code{version}, and \code{results} — one entry per
#'   analysis, in request order, each a list of parallel vectors.
#' @export
pf_analyze <- function(wav, analyses, channel = NULL, engine = NULL) {
  .pf_ensure_ready(.pf_engine_of(engine))
  req <- list(
    wav_path = normalizePath(wav, mustWork = TRUE),
    analyses = analyses
  )
  if (!is.null(channel)) req$channel <- channel
  .pf_pipe_call(req, engine)
}

# ---------------------------------------------------------------------------
# Typed wrappers
# ---------------------------------------------------------------------------

#' Pitch (F0) contour.
#'
#' Equivalent to Praat's \emph{Sound: To Pitch (ac)} / \emph{(cc)}.
#'
#' @param wav           Path to an audio file.
#' @param method        \code{"ac"} (autocorrelation, default) or \code{"cc"}
#'   (cross-correlation).
#' @param time_step     Frame step in seconds; 0 = auto (0.75/floor for ac,
#'   0.25/floor for cc).
#' @param pitch_floor   Minimum pitch in Hz (default 75).
#' @param pitch_ceiling Maximum pitch in Hz (default 600).
#' @param voicing_threshold,silence_threshold,octave_cost,octave_jump_cost,voiced_unvoiced_cost
#'   Optional tuning parameters (Boersma 1993).  \code{NULL} (default) uses
#'   the engine defaults 0.45 / 0.03 / 0.01 / 0.35 / 0.14.
#' @param channel       0-based channel for multi-channel files.
#' @return A data.frame with columns \code{file}, \code{time}, \code{f0}
#'   (Hz; \code{NA} for unvoiced frames), \code{strength}, \code{voiced}.
#' @export
pf_pitch <- function(wav, method = c("ac", "cc"), time_step = 0,
                     pitch_floor = 75, pitch_ceiling = 600,
                     voicing_threshold = NULL, silence_threshold = NULL,
                     octave_cost = NULL, octave_jump_cost = NULL,
                     voiced_unvoiced_cost = NULL, channel = NULL) {
  method <- match.arg(method)
  spec <- list(
    type = paste0("pitch_", method),
    time_step = time_step,
    pitch_floor = pitch_floor,
    pitch_ceiling = pitch_ceiling
  )
  spec$voicing_threshold <- voicing_threshold
  spec$silence_threshold <- silence_threshold
  spec$octave_cost <- octave_cost
  spec$octave_jump_cost <- octave_jump_cost
  spec$voiced_unvoiced_cost <- voiced_unvoiced_cost

  r <- pf_analyze(wav, list(spec), channel)$results[[1]]
  f0 <- r$frequencies
  voiced <- f0 > 0
  f0[!voiced] <- NA_real_
  data.frame(file = basename(wav), time = r$times, f0 = f0,
             strength = r$strengths, voiced = voiced,
             stringsAsFactors = FALSE)
}


#' Formant tracks (Burg LPC).
#'
#' Equivalent to Praat's \emph{Sound: To Formant (burg)}.
#'
#' @param wav              Path to an audio file.
#' @param time_step        Frame step in seconds; 0 = auto (25\% of window).
#' @param max_num_formants Maximum formants per frame (default 5).
#' @param max_formant_hz   Formant ceiling in Hz (default 5500; use ~5000
#'   for a typical male speaker).
#' @param window_length    Analysis window parameter in seconds (default
#'   0.025; the effective window is twice this, per Praat).
#' @param pre_emphasis_from Pre-emphasis onset frequency in Hz (default 50).
#' @param channel          0-based channel for multi-channel files.
#' @return A data.frame with columns \code{file}, \code{time},
#'   \code{n_formants}, then \code{F1..Fk} and \code{B1..Bk} (Hz; \code{NA}
#'   where a formant was not found).
#' @export
pf_formants <- function(wav, time_step = 0, max_num_formants = 5,
                        max_formant_hz = 5500, window_length = 0.025,
                        pre_emphasis_from = 50, channel = NULL) {
  spec <- list(
    type = "formant_burg",
    time_step = time_step,
    max_num_formants = max_num_formants,
    max_formant_hz = max_formant_hz,
    window_length = window_length,
    pre_emphasis_from = pre_emphasis_from
  )
  r <- pf_analyze(wav, list(spec), channel)$results[[1]]

  df <- data.frame(file = basename(wav), time = r$times,
                   n_formants = r$n_formants, stringsAsFactors = FALSE)
  for (n in seq_len(max_num_formants)) {
    df[[paste0("F", n)]] <- r[[paste0("F", n)]]
  }
  for (n in seq_len(max_num_formants)) {
    df[[paste0("B", n)]] <- r[[paste0("B", n)]]
  }
  df
}


#' Intensity contour.
#'
#' Equivalent to Praat's \emph{Sound: To Intensity}.
#'
#' @param wav       Path to an audio file.
#' @param min_pitch Minimum pitch in Hz; sets the analysis window length
#'   (default 100).
#' @param time_step Frame step in seconds; 0 = auto (0.8/min_pitch).
#' @param channel   0-based channel for multi-channel files.
#' @return A data.frame with columns \code{file}, \code{time},
#'   \code{intensity} (dB).
#' @export
pf_intensity <- function(wav, min_pitch = 100, time_step = 0,
                         channel = NULL) {
  spec <- list(type = "intensity", min_pitch = min_pitch,
               time_step = time_step)
  r <- pf_analyze(wav, list(spec), channel)$results[[1]]
  data.frame(file = basename(wav), time = r$times, intensity = r$values,
             stringsAsFactors = FALSE)
}


#' Harmonicity (HNR) contour.
#'
#' Equivalent to Praat's \emph{Sound: To Harmonicity (ac)} / \emph{(cc)}.
#'
#' @param wav                Path to an audio file.
#' @param method             \code{"ac"} (default) or \code{"cc"}.
#' @param time_step          Frame step in seconds (default 0.01).
#' @param min_pitch          Minimum pitch in Hz (default 75).
#' @param silence_threshold  Silence threshold, 0-1 (default 0.1).
#' @param periods_per_window Periods per analysis window.  \code{NULL}
#'   (default) uses Praat's default for the method: 4.5 for ac, 1.0 for cc.
#' @param channel            0-based channel for multi-channel files.
#' @return A data.frame with columns \code{file}, \code{time}, \code{hnr}
#'   (dB).
#' @export
pf_harmonicity <- function(wav, method = c("ac", "cc"), time_step = 0.01,
                           min_pitch = 75, silence_threshold = 0.1,
                           periods_per_window = NULL, channel = NULL) {
  method <- match.arg(method)
  spec <- list(
    type = paste0("harmonicity_", method),
    time_step = time_step,
    min_pitch = min_pitch,
    silence_threshold = silence_threshold
  )
  spec$periods_per_window <- periods_per_window
  r <- pf_analyze(wav, list(spec), channel)$results[[1]]
  data.frame(file = basename(wav), time = r$times, hnr = r$values,
             stringsAsFactors = FALSE)
}


#' Spectral moments at given times.
#'
#' Windows the signal around each time point (clamped at the signal edges)
#' and computes Praat's spectral moments of that window's spectrum —
#' equivalent to \emph{Spectrum: Get centre of gravity} and friends on an
#' extracted slice.
#'
#' @param wav           Path to an audio file.
#' @param times         Numeric vector of measurement times in seconds.
#' @param window_length Window length in seconds (default 0.025).
#' @param power         Moment power parameter (default 2, as Praat).
#' @param channel       0-based channel for multi-channel files.
#' @return A data.frame with columns \code{file}, \code{time},
#'   \code{center_of_gravity}, \code{standard_deviation} (Hz),
#'   \code{skewness}, \code{kurtosis}.
#' @export
pf_spectral_moments <- function(wav, times, window_length = 0.025,
                                power = 2, channel = NULL) {
  spec <- list(type = "spectral_moments", times = I(as.numeric(times)),
               window_length = window_length, power = power)
  r <- pf_analyze(wav, list(spec), channel)$results[[1]]
  data.frame(file = basename(wav), time = r$times,
             center_of_gravity = r$center_of_gravity,
             standard_deviation = r$standard_deviation,
             skewness = r$skewness, kurtosis = r$kurtosis,
             stringsAsFactors = FALSE)
}


#' Band energy at given times.
#'
#' Windows the signal around each time point (clamped at the signal edges)
#' and integrates spectral energy over \code{[f_min, f_max]} — equivalent
#' to Praat's \emph{Spectrum: Get band energy} on an extracted slice.
#'
#' @param wav           Path to an audio file.
#' @param times         Numeric vector of measurement times in seconds.
#' @param f_min         Lower band edge in Hz (default 0).
#' @param f_max         Upper band edge in Hz; 0 (default) = Nyquist.
#' @param window_length Window length in seconds (default 0.025).
#' @param channel       0-based channel for multi-channel files.
#' @return A data.frame with columns \code{file}, \code{time},
#'   \code{energy} (Pa^2 s).
#' @export
pf_band_energy <- function(wav, times, f_min = 0, f_max = 0,
                           window_length = 0.025, channel = NULL) {
  spec <- list(type = "band_energy", times = I(as.numeric(times)),
               f_min = f_min, f_max = f_max,
               window_length = window_length)
  r <- pf_analyze(wav, list(spec), channel)$results[[1]]
  data.frame(file = basename(wav), time = r$times, energy = r$values,
             stringsAsFactors = FALSE)
}
