#' Internal pipe interface to the analysis engine binary
#'
#' Sends JSON requests to an engine and parses its JSON responses.  Three
#' interchangeable engines speak the same request/response schema:
#'
#' \describe{
#'   \item{\code{"open"}}{\code{praatfan-open-pipe} — the clean-room MIT
#'     engine, as a native binary (JSON over stdin/stdout).}
#'   \item{\code{"wasm"}}{the same open engine compiled to WebAssembly,
#'     shipped inside this package and run in R through \pkg{V8}.  Nothing
#'     to install; slower than the binary (one thread per call).}
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
  ),
  # the open engine as WebAssembly, run in V8 (wasm.R); ships with the
  # package, no binary
  wasm = list(
    name = "wasm",
    bin  = NULL,
    repo = NULL,
    src  = NULL,
    wasm = "open"
  ),
  # the GPL engine as WebAssembly; GPL-licensed, so downloaded on request by
  # pf_setup(engine = "gpl_wasm") rather than shipped
  gpl_wasm = list(
    name  = "gpl_wasm",
    bin   = NULL,
    repo  = "ucpresearch/praatfan-core-rs",
    src   = NULL,
    wasm  = "gpl",
    asset = "praatfan-gpl-wasm-v8.zip"
  )
)

#' Get or set the analysis engine.
#'
#' With no argument, returns the name of the engine currently in use.
#' With an argument, selects the engine for all subsequent analysis calls
#' in this session (stored in \code{options(praatfan.engine = ...)}).
#'
#' An explicit \code{pf_engine(...)} call wins, then the
#' \code{PRAATFAN_ENGINE} environment variable.  With neither, the engine is
#' \code{"open"} when its native binary is available (installed by
#' \code{\link{pf_setup}}, on PATH, or named by \code{PRAATFAN_PIPE}), and
#' otherwise \code{"wasm"} — the same engine running inside R, so analysis
#' works with nothing installed.  Independently of the engine choice, the
#' \code{PRAATFAN_PIPE} environment variable — a path to a specific binary
#' — always wins for the native engines.
#'
#' The GPL engine also comes as WebAssembly (\code{"gpl_wasm"}, downloaded
#' by \code{pf_setup(engine = "gpl_wasm")}).  Choosing \code{"gpl"} uses
#' the GPL binary when it is installed and otherwise the GPL WASM build when
#' that is, so \code{"gpl"} works with either.
#'
#' @param engine \code{NULL} to query, or one of \code{"open"},
#'   \code{"wasm"}, \code{"gpl"}, \code{"gpl_wasm"} to select.
#' @return The current engine name (invisibly, when setting).
#' @export
pf_engine <- function(engine = NULL) {
  if (is.null(engine)) {
    current <- .pf_engine_requested()
    if (is.null(current)) current <- .pf_auto_engine()
    return(.pf_engine_of(current)$name)
  }
  engine <- .pf_engine_of(engine)$name
  options(praatfan.engine = engine)
  invisible(engine)
}

#' The engine the user asked for (option, then environment), or NULL.
#' @keywords internal
.pf_engine_requested <- function() {
  current <- getOption("praatfan.engine")
  if (is.null(current)) {
    env <- Sys.getenv("PRAATFAN_ENGINE", "")
    if (nzchar(env)) current <- env
  }
  current
}

#' No engine requested: the open binary when it is available, else WASM
#' (announced once per session), else "open" (its auto-setup installs it).
#' @keywords internal
.pf_auto_engine <- function() {
  if (.pf_is_ready(.pf_engines$open)) return("open")
  if (!is.null(.pf_wasm_unavailable())) return("open")
  if (!isTRUE(.pf_wasm_env$announced)) {
    .pf_wasm_env$announced <- TRUE
    message("praatfan: no praatfan-open-pipe binary found, so the engine runs ",
            "inside R (WASM; same numbers, slower). Run pf_setup() for native ",
            "speed; pf_engine(\"wasm\") silences this.")
  }
  "wasm"
}

#' The engine a call actually runs on: \code{"gpl"} without its binary but
#' with the GPL WASM build installed runs on that (announced once).
#' @keywords internal
.pf_runtime_engine <- function(eng) {
  if (eng$name == "gpl" && !.pf_is_ready(eng) &&
      .pf_is_ready(.pf_engines$gpl_wasm)) {
    if (!isTRUE(.pf_wasm_env$announced_gpl)) {
      .pf_wasm_env$announced_gpl <- TRUE
      message("praatfan: no praatfan-gpl-pipe binary found; using the GPL ",
              "engine's WASM build (same engine, slower). ",
              "pf_setup(engine = \"gpl\") installs the binary.")
    }
    return(.pf_engines$gpl_wasm)
  }
  eng
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
  if (!is.null(eng$wasm)) {
    who <- paste0("praatfan (", eng$name, " engine)")
    resp <- .pf_parse_response(.pf_wasm_text(request, eng$wasm), who)
    # the binary prints these to stderr, which a WASM engine has none of
    for (w in resp$warnings) warning(who, ": ", w, call. = FALSE)
    return(resp)
  }
  bin <- .pf_pipe_bin(eng)
  json_in <- jsonlite::toJSON(request, auto_unbox = TRUE, digits = NA)

  err_file <- tempfile()
  on.exit(unlink(err_file), add = TRUE)

  json_out <- system2(bin, input = json_in, stdout = TRUE, stderr = err_file)

  status <- attr(json_out, "status")
  err_msg <- paste(readLines(err_file, warn = FALSE), collapse = "\n")
  if (!is.null(status) && status != 0L) {
    stop(eng$bin, " failed (exit ", status, "):\n", err_msg, call. = FALSE)
  }
  # The binary also writes diagnostics to stderr on a successful run (for
  # example the ill-conditioned periods_per_window warning). Without this an R
  # caller receives a column of NAs with no explanation at all.
  if (nzchar(err_msg)) {
    warning(eng$bin, ": ", err_msg, call. = FALSE)
  }

  .pf_parse_response(paste(json_out, collapse = "\n"), eng$bin)
}

.pf_parse_response <- function(txt, who) {
  resp <- jsonlite::fromJSON(txt, simplifyVector = TRUE, simplifyDataFrame = FALSE)
  if (!isTRUE(resp$ok)) {
    stop(who, " error: ", resp$error %||% "unknown error", call. = FALSE)
  }
  resp
}

# --- Binary location ---------------------------------------------------------

.pf_pipe_bin <- function(eng = .pf_engine_of()) {
  # 1. Environment variable override (any engine)
  env <- Sys.getenv("PRAATFAN_PIPE", "")
  if (nzchar(env) && file.exists(env)) return(env)

  # 2. Installed location
  candidate <- file.path(.pf_bin_dir(), eng$bin)
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
