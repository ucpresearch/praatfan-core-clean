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
#' works with nothing installed.  Independently of the engine choice,
#' \code{options(praatfan.pipe = )} and then the \code{PRAATFAN_PIPE}
#' environment variable — a path to a specific binary — always win for the
#' native engines; one that names a missing file is an error.  Otherwise a
#' native engine is looked up in the caller's directory, then in the
#' installed package (a shared install, see \code{\link{pf_setup}}), then
#' on PATH.
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
    .pf_check_engine_version(eng, resp$version)
    return(resp)
  }
  bin <- .pf_pipe_bin(eng)
  json_in <- .pf_to_json(request)

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

  resp <- .pf_parse_response(paste(json_out, collapse = "\n"), eng$bin)
  .pf_check_engine_version(eng, resp$version)
  resp
}

#' Warn (once per session) when an open engine's version differs from the
#' package's
#'
#' The \code{"wasm"} engine ships inside the package, but the
#' \code{"open"} binary is installed separately by \code{pf_setup()} (or
#' named by \code{PRAATFAN_PIPE}). After an upgrade of either, the two
#' open engines can be different versions and give different numbers.
#' The GPL engines have their own version numbers and are not checked.
#' @keywords internal
.pf_check_engine_version <- function(eng, version) {
  if (is.null(version)) return(invisible())
  if (eng$name %in% c("gpl", "gpl_wasm")) {
    return(.pf_check_gpl_pair(eng$name, as.character(version)))
  }
  if (!eng$name %in% c("open", "wasm")) return(invisible())
  pkg <- as.character(utils::packageVersion("praatfan"))
  if (identical(as.character(version), pkg)) return(invisible())
  key <- paste(eng$name, version)
  if (isTRUE(.pf_version_warned[[key]])) return(invisible())
  assign(key, TRUE, envir = .pf_version_warned)
  fix <- if (eng$name == "open") {
    paste("Run pf_setup() (as the library's owner for a shared install),",
          "or set PRAATFAN_PIPE to a matching binary.")
  } else {
    "Reinstall the praatfan R package."
  }
  warning("praatfan: the \"", eng$name, "\" engine is version ", version,
          " but the R package is ", pkg, ", so its results can differ from ",
          "the other open engine. ", fix, call. = FALSE)
  invisible()
}

.pf_version_warned <- new.env(parent = emptyenv())

#' The GPL engines have their own version line, so the binary
#' (\code{"gpl"}) and the WASM build (\code{"gpl_wasm"}) are compared with
#' each other, not with the package: warn once when this session has seen
#' both and they differ.
#' @keywords internal
.pf_check_gpl_pair <- function(name, version) {
  assign(name, version, envir = .pf_gpl_seen)
  other <- if (name == "gpl") "gpl_wasm" else "gpl"
  if (!exists(other, envir = .pf_gpl_seen, inherits = FALSE)) return(invisible())
  v <- c(gpl = get("gpl", envir = .pf_gpl_seen),
         gpl_wasm = get("gpl_wasm", envir = .pf_gpl_seen))
  if (identical(v[[1]], v[[2]])) return(invisible())
  key <- paste("gpl-pair", v[[1]], v[[2]])
  if (isTRUE(.pf_version_warned[[key]])) return(invisible())
  assign(key, TRUE, envir = .pf_version_warned)
  warning("praatfan: the \"gpl\" engine is version ", v[["gpl"]],
          " but \"gpl_wasm\" is ", v[["gpl_wasm"]], ", so their results can ",
          "differ. pf_setup(engine = \"gpl\") and pf_setup(engine = ",
          "\"gpl_wasm\") install the latest of each.", call. = FALSE)
  invisible()
}

.pf_gpl_seen <- new.env(parent = emptyenv())

.pf_parse_response <- function(txt, who) {
  resp <- jsonlite::fromJSON(txt, simplifyVector = TRUE, simplifyDataFrame = FALSE)
  if (!isTRUE(resp$ok)) {
    stop(who, " error: ", resp$error %||% "unknown error", call. = FALSE)
  }
  resp
}

# --- Binary location ---------------------------------------------------------

.pf_pipe_bin <- function(eng = .pf_engine_of()) {
  # 1. options(praatfan.pipe), then PRAATFAN_PIPE (any engine); a pin naming
  #    a missing file stops rather than falling through to another binary
  pinned <- .pf_pinned_pipe()
  if (!is.null(pinned)) return(pinned)

  # 2. Installed: the caller's directory, then the shared in-package one
  installed <- .pf_find_installed_bin(eng)
  if (!is.null(installed)) return(installed)

  # 3. PATH
  found <- Sys.which(eng$bin)
  if (found != "") return(found)

  stop(eng$bin, " not found. Run pf_setup(engine = \"", eng$name,
       "\") first, or set PRAATFAN_PIPE=/path/to/binary.", call. = FALSE)
}

# --- Null-coalescing operator (base R) ---------------------------------------

`%||%` <- function(x, y) if (is.null(x)) y else x

#' Serialise a pipe request to JSON so that every double round-trips
#'
#' \code{digits = NA} writes only 15 significant digits, which changes about
#' one double in four by an ulp (e.g. a phone midpoint 1.0619687500000001
#' becomes 1.06196875 and parses back as 1.0619687499999999). At a window
#' edge that moves a whole sample. 17 significant digits always round-trip,
#' and the engines parse JSON numbers exactly.
#' @keywords internal
.pf_to_json <- function(request) {
  jsonlite::toJSON(request, auto_unbox = TRUE, digits = I(17))
}
