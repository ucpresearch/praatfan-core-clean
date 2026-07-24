#' Internal pipe interface to the analysis engine binary
#'
#' Sends JSON requests to a pipe engine binary via stdin and parses JSON
#' responses from stdout.  Two interchangeable engines speak the same
#' request/response schema:
#'
#' \describe{
#'   \item{\code{"open"}}{\code{praatfan-open-pipe} — the clean-room MIT
#'     engine (default).}
#'   \item{\code{"gpl"}}{\code{praatfan-gpl-pipe} — the GPL engine, which
#'     tracks Praat's values even more closely.  Choosing it means running
#'     a GPL-licensed binary (as a subprocess; it does not affect this
#'     package's license).}
#' }
#'
#' @name pipe
#' @keywords internal
NULL

# --- Engine registry ---------------------------------------------------------

# Both engine repos publish release assets under the shared naming
# convention <bin>-<os>-<arch>[.exe] with Rust-target arch spellings
# (linux-x86_64, linux-aarch64, macos-aarch64, macos-x86_64,
# windows-x86_64.exe).
.pf_engines <- list(
  open = list(
    name = "open",
    bin  = "praatfan-open-pipe",
    repo = "ucpresearch/praatfan-core-clean",
    src  = "praatfan-core-clean"
  ),
  gpl = list(
    name = "gpl",
    bin  = "praatfan-gpl-pipe",
    repo = "ucpresearch/praatfan-core-rs",
    src  = "praatfan-core-rs"
  )
)

#' Get or set the analysis engine.
#'
#' With no argument, returns the name of the engine currently in use.
#' With an argument, selects the engine for all subsequent analysis calls
#' in this session (stored in \code{options(praatfan.engine = ...)}).
#'
#' The default is \code{"open"}; the \code{PRAATFAN_ENGINE} environment
#' variable overrides the default, and an explicit \code{pf_engine(...)}
#' call overrides both.  Independently of the engine choice, the
#' \code{PRAATFAN_PIPE} environment variable — a path to a specific binary
#' — always wins.
#'
#' @param engine \code{NULL} to query, or one of \code{"open"},
#'   \code{"gpl"} to select.
#' @return The current engine name (invisibly, when setting).
#' @export
pf_engine <- function(engine = NULL) {
  if (is.null(engine)) {
    current <- getOption("praatfan.engine")
    if (is.null(current)) {
      env <- Sys.getenv("PRAATFAN_ENGINE", "")
      current <- if (nzchar(env)) env else "open"
    }
    return(.pf_engine_of(current)$name)
  }
  engine <- .pf_engine_of(engine)$name
  options(praatfan.engine = engine)
  invisible(engine)
}

#' Resolve an engine name (or NULL = current) to its registry entry.
#' @keywords internal
.pf_engine_of <- function(engine = NULL) {
  if (is.null(engine)) engine <- pf_engine()
  eng <- .pf_engines[[engine]]
  if (is.null(eng)) {
    stop("Unknown engine ", deparse(engine), "; expected one of: ",
         paste(names(.pf_engines), collapse = ", "), call. = FALSE)
  }
  eng
}

# --- Low-level pipe call -----------------------------------------------------

.pf_pipe_call <- function(request, engine = NULL) {
  eng <- .pf_engine_of(engine)
  bin <- .pf_pipe_bin(eng)
  json_in <- jsonlite::toJSON(request, auto_unbox = TRUE, digits = NA)

  err_file <- tempfile()
  on.exit(unlink(err_file), add = TRUE)

  json_out <- system2(bin, input = json_in, stdout = TRUE, stderr = err_file)

  status <- attr(json_out, "status")
  if (!is.null(status) && status != 0L) {
    err_msg <- paste(readLines(err_file, warn = FALSE), collapse = "\n")
    stop(eng$bin, " failed (exit ", status, "):\n", err_msg, call. = FALSE)
  }

  resp <- jsonlite::fromJSON(paste(json_out, collapse = "\n"),
                             simplifyVector = TRUE,
                             simplifyDataFrame = FALSE)
  if (!isTRUE(resp$ok)) {
    stop(eng$bin, " error: ", resp$error %||% "unknown error", call. = FALSE)
  }
  resp
}

# --- Binary location ---------------------------------------------------------

.pf_pipe_bin <- function(eng = .pf_engine_of()) {
  # 1. Environment variable override (any engine)
  env <- Sys.getenv("PRAATFAN_PIPE", "")
  if (nzchar(env) && file.exists(env)) return(env)

  # 2. Installed location
  candidate <- file.path(.pf_bin_dir, eng$bin)
  if (file.exists(candidate)) return(candidate)
  if (file.exists(paste0(candidate, ".exe"))) return(paste0(candidate, ".exe"))

  # 3. PATH
  found <- Sys.which(eng$bin)
  if (found != "") return(found)

  stop(eng$bin, " not found. Run pf_setup(engine = \"", eng$name,
       "\") first, or set PRAATFAN_PIPE=/path/to/binary.", call. = FALSE)
}

# --- Null-coalescing operator (base R) ---------------------------------------

`%||%` <- function(x, y) if (is.null(x)) y else x
