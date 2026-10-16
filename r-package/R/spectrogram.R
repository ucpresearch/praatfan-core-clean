#' Spectrogram.
#'
#' A short-time Fourier transform of the file, with the parameters of
#' Praat's \emph{Sound: To Spectrogram}.  The open engines (\code{"open"},
#' \code{"wasm"}) compute a standard Gaussian/Hanning STFT whose grid and
#' values are close to, but not exactly, Praat's; the \code{"gpl"} engine
#' aims for Praat's exact values.  Always read the grid from the result
#' (\code{times}, \code{freqs}); it depends on the engine and the settings.
#'
#' @param wav            Path to an audio file.
#' @param window_length  Effective window length: \code{"wide"} (wide-band,
#'   0.005 s, the default; resolves formants), \code{"narrow"} (narrow-band,
#'   0.025 s; resolves harmonics), or a number of seconds.
#' @param max_frequency  Ceiling in Hz (default 5000).  0, negative or above
#'   the Nyquist frequency means the Nyquist frequency.
#' @param time_step      Requested time step in s (default 0.002).  The step
#'   actually used is in the result.
#' @param frequency_step Requested frequency step in Hz (default 20).  The
#'   step actually used is in the result.
#' @param window_shape   \code{"gaussian"} (default) or \code{"hanning"}.
#' @param start_time,end_time Optional: return only the frames whose time
#'   lies in this range (the grid is still the whole file's).  Use them, or
#'   \code{precision = "single"}, to keep long files small: the defaults
#'   give about 125,000 values per second of audio.
#' @param precision      \code{"double"} (default) or \code{"single"}: how the
#'   engine sends the values (single halves the transfer; the matrix in R is
#'   double either way).
#' @param channel        0-based channel for multi-channel files.
#' @param engine         Engine for this call (see \code{\link{pf_engine}}).
#' @return An object of class \code{pf_spectrogram}: a list with
#'   \code{file}, \code{times} (frame centres, s), \code{freqs} (Hz),
#'   \code{power} (a \code{length(times)} x \code{length(freqs)} matrix of
#'   linear power), \code{time_step}, \code{frequency_step} (as used) and
#'   \code{settings}.  It has \code{plot()}, \code{print()} and
#'   \code{as.data.frame()} methods.
#' @examples
#' \dontrun{
#' sg <- pf_spectrogram("audio.wav")
#' plot(sg)
#' plot(pf_spectrogram("audio.wav", "narrow", max_frequency = 3000))
#' }
#' @export
pf_spectrogram <- function(wav, window_length = "wide", max_frequency = 5000,
                           time_step = 0.002, frequency_step = 20,
                           window_shape = c("gaussian", "hanning"),
                           start_time = NULL, end_time = NULL,
                           precision = c("double", "single"),
                           channel = NULL, engine = NULL) {
  window_shape <- match.arg(window_shape)
  precision <- match.arg(precision)
  if (is.character(window_length)) {
    window_length <- switch(match.arg(window_length, c("wide", "narrow")),
                            wide = 0.005, narrow = 0.025)
  }
  stopifnot(is.numeric(window_length), length(window_length) == 1)
  spec <- list(type = "spectrogram", window_length = window_length,
               max_frequency = max_frequency, time_step = time_step,
               frequency_step = frequency_step, window_shape = window_shape,
               dtype = if (precision == "single") "f32le" else "f64le")
  if (!is.null(start_time)) spec$start_time <- start_time
  if (!is.null(end_time)) spec$end_time <- end_time
  r <- tryCatch(
    pf_analyze(wav, list(spec), channel, engine)$results[[1]],
    error = function(e) {
      msg <- conditionMessage(e)
      if (grepl("unknown type \"spectrogram\"", msg, fixed = TRUE)) {
        msg <- paste0(msg, "\nThis engine's installed version predates ",
                      "spectrograms; update it with pf_setup(engine = \"",
                      pf_engine(), "\", force = TRUE) (or the engine you ",
                      "passed), or check what PRAATFAN_PIPE / PATH points to.")
      }
      stop(msg, call. = FALSE)
    })

  times <- as.numeric(unlist(r$times))
  freqs <- as.numeric(unlist(r$freqs))
  nt <- length(times)
  nf <- length(freqs)
  size <- if (identical(r$dtype, "f32le")) 4L else 8L
  raw <- if (nzchar(r$power_b64)) jsonlite::base64_dec(r$power_b64) else raw(0)
  if (!identical(r$layout, "time-major") || length(raw) != nt * nf * size) {
    stop("praatfan: malformed spectrogram from the engine (", length(raw),
         " bytes for ", nt, " x ", nf, " ", r$dtype, ")", call. = FALSE)
  }
  values <- readBin(raw, "double", n = nt * nf, size = size, endian = "little")
  structure(list(
    file = basename(wav),
    times = times,
    freqs = freqs,
    # time-major on the wire: frame by frame, so each row is one frame
    power = matrix(values, nrow = nt, ncol = nf, byrow = TRUE),
    time_step = r$time_step,
    frequency_step = r$frequency_step,
    settings = list(window_length = window_length,
                    max_frequency = max_frequency,
                    window_shape = window_shape)
  ), class = "pf_spectrogram")
}

#' @export
print.pf_spectrogram <- function(x, ...) {
  span <- function(v, unit) {
    if (length(v)) sprintf("%.4g-%.4g %s", min(v), max(v), unit) else "empty"
  }
  cat(sprintf("<pf_spectrogram> %s: %d frames x %d bins, %s (step %.4g s), %s (step %.4g Hz), %s window %.4g s\n",
              x$file, length(x$times), length(x$freqs),
              span(x$times, "s"), x$time_step, span(x$freqs, "Hz"),
              x$frequency_step, x$settings$window_shape,
              x$settings$window_length))
  invisible(x)
}

#' Long-format data frame: one row per frame and frequency bin.
#'
#' @param x A \code{pf_spectrogram}.
#' @param row.names,optional Ignored.
#' @param ... Ignored.
#' @return A data.frame with \code{file}, \code{time}, \code{frequency} and
#'   \code{power} (linear) — for ggplot2's \code{geom_raster()} and the like.
#' @export
as.data.frame.pf_spectrogram <- function(x, row.names = NULL, optional = FALSE, ...) {
  data.frame(file = rep(x$file, length(x$power)),
             time = rep(x$times, times = length(x$freqs)),
             frequency = rep(x$freqs, each = length(x$times)),
             power = as.vector(x$power),
             stringsAsFactors = FALSE)
}

#' Draw a spectrogram.
#'
#' Grey-scale, dark = loud, as Praat paints it.  The power is converted to
#' dB (\code{10 * log10(power)}), given a pre-emphasis of
#' \code{pre_emphasis} dB per octave (0 dB at 1000 Hz) so that the weaker
#' high frequencies show, and everything more than \code{dynamic_range} dB
#' below \code{max_db} is drawn white.
#'
#' @param x             A \code{pf_spectrogram}.
#' @param dynamic_range dB range shown (default 50).
#' @param pre_emphasis  dB per octave (default 6; 0 = none).
#' @param max_db        Level drawn darkest, on the same dB scale; \code{NULL}
#'   (default) = the loudest point after pre-emphasis.
#' @param col           Colours from quiet to loud (default grey, white to
#'   black).
#' @param xlab,ylab,main Axis labels and title.
#' @param smooth        Display smoothing: each cell shows the mean linear
#'   power of the cells within \code{smooth} frames and bins of it (default
#'   1, a 3 x 3 neighbourhood; 0 = none).  A fine STFT has isolated, very
#'   deep nulls (between harmonics, in noise) that otherwise show as white
#'   specks.  The data in \code{x} are not changed.
#' @param useRaster     Draw as a bitmap (default \code{TRUE}; fast, small).
#' @param ...           Passed on to \code{\link[graphics]{image}}.
#' @export
plot.pf_spectrogram <- function(x, dynamic_range = 50, pre_emphasis = 6,
                                max_db = NULL,
                                col = grDevices::grey(seq(1, 0, length.out = 256)),
                                xlab = "Time (s)", ylab = "Frequency (Hz)",
                                main = x$file, smooth = 1, useRaster = TRUE, ...) {
  if (length(x$times) < 2 || length(x$freqs) < 2) {
    stop("praatfan: nothing to plot (", length(x$times), " frames)", call. = FALSE)
  }
  # More cells than pixels: each pixel shows the mean power of the cells it
  # covers.  Sampling one cell per pixel instead would turn the isolated
  # nulls of a fine STFT grid into white speckles.
  px <- .pf_plot_pixels()
  times <- .pf_bin_mean(x$times, px[1])
  freqs <- .pf_bin_mean(x$freqs, px[2])
  power <- .pf_box_mean(x$power, smooth)
  power <- .pf_bin_mean(power, px[1])
  power <- t(.pf_bin_mean(t(power), px[2]))
  db <- 10 * log10(pmax(power, .Machine$double.xmin))
  if (pre_emphasis != 0) {
    # the 0 Hz bin gets the boost of the first nonzero bin
    f <- pmax(freqs, freqs[freqs > 0][1])
    db <- sweep(db, 2, pre_emphasis * log2(f / 1000), `+`)
  }
  if (is.null(max_db)) max_db <- max(db)
  z <- pmin(pmax(db, max_db - dynamic_range), max_db)
  graphics::image(times, freqs, z, col = col,
                  zlim = c(max_db - dynamic_range, max_db),
                  xlab = xlab, ylab = ylab, main = main,
                  useRaster = useRaster, ...)
  invisible(x)
}

# Plot-region size in device pixels (time, frequency); generous when the
# device does not say (vector devices draw every cell anyway).
.pf_plot_pixels <- function() {
  px <- tryCatch({
    if (grDevices::dev.cur() == 1) stop("no device")
    grDevices::dev.size("px") * diff(graphics::par("plt"))[c(1, 3)]
  }, error = function(e) c(NA, NA))
  px <- ceiling(px)
  px[!is.finite(px) | px < 50] <- 2000
  px
}

# Average a vector, or a matrix's rows, over groups of k consecutive
# entries, k chosen so there are at most n groups.  Equal group sizes keep
# the grid regular (needed for raster drawing); a short last group is
# averaged too, but its coordinate continues the regular spacing.
.pf_bin_mean <- function(v, n) {
  m <- NROW(v)
  if (m <= n) return(v)
  k <- ceiling(m / n)
  g <- (seq_len(m) - 1) %/% k
  out <- rowsum(v, g, reorder = FALSE) / as.vector(table(g))
  if (is.null(dim(v))) {
    out <- as.vector(out)
    out[length(out)] <- out[1] + (length(out) - 1) * (out[2] - out[1])
  }
  out
}

# Mean of each cell's (2r+1) x (2r+1) neighbourhood, truncated at the edges.
.pf_box_mean <- function(m, r) {
  r <- as.integer(r)
  if (r < 1) return(m)
  run <- function(v) {          # moving mean along a vector
    n <- length(v)
    cs <- c(0, cumsum(v))
    lo <- pmax(seq_len(n) - r, 1)
    hi <- pmin(seq_len(n) + r, n)
    (cs[hi + 1] - cs[lo]) / (hi - lo + 1)
  }
  m <- apply(m, 2, run)
  t(apply(m, 1, run))
}
