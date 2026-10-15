#' The WASM engine: the open engine running inside R
#'
#' The package ships the open engine twice: as the native
#' \code{praatfan-open-pipe} binary that \code{pf_setup()} installs, and as a
#' WebAssembly build of the same Rust code under \code{inst/wasm/}, run in an
#' embedded V8 (the \pkg{V8} package).  Both answer the same JSON pipe
#' protocol with the same numbers (to floating-point rounding); the binary is
#' faster (several threads per call), WASM needs nothing installed beyond this
#' package.
#'
#' The file is read in R as raw bytes and decoded inside the engine, by the
#' same decoder the binary uses, so every format the binary reads works here
#' too.  The bytes of the last file are kept in the V8 context, so several
#' analyses of one file send it once.
#'
#' Parallel work: the engine runs one thread per call, so analysing several
#' files at once is the way to use more cores.  Each R process builds its own
#' V8 context; PSOCK clusters (\code{parallel::makePSOCKcluster},
#' \pkg{future}'s multisession) are the portable choice (Windows included).
#'
#' @name wasm
#' @keywords internal
NULL

.pf_wasm_env <- new.env(parent = emptyenv())

.pf_wasm_file <- function(name) {
  p <- system.file("wasm", name, package = "praatfan")
  if (nzchar(p)) return(p)
  # a source tree loaded with pkgload: inst/ is not flattened
  file.path(system.file(package = "praatfan"), "inst", "wasm", name)
}

#' Why the WASM engine cannot run here, or \code{NULL} when it can.
#' @keywords internal
.pf_wasm_unavailable <- function() {
  if (!requireNamespace("V8", quietly = TRUE)) {
    return("the V8 package is not installed (install.packages(\"V8\"))")
  }
  if (!file.exists(.pf_wasm_file("praatfan_bg.wasm"))) {
    return("this installation has no WASM bundle (inst/wasm/)")
  }
  NULL
}

#' The V8 context holding the engine: created on first use, then cached.
#'
#' The glue is wasm-bindgen's \code{--target no-modules} output, which defines
#' a global \code{wasm_bindgen}; \code{shim.js} supplies the UTF-8
#' TextEncoder/TextDecoder an embedded V8 lacks and a base64 decoder.
#' @keywords internal
.pf_wasm_ctx <- function() {
  ctx <- .pf_wasm_env$ctx
  if (!is.null(ctx)) return(ctx)
  why <- .pf_wasm_unavailable()
  if (!is.null(why)) {
    stop("praatfan: the WASM engine cannot run: ", why, ". Run pf_setup() ",
         "to install the native binary instead.", call. = FALSE)
  }
  ctx <- V8::v8()
  ctx$source(.pf_wasm_file("shim.js"))
  ctx$source(.pf_wasm_file("praatfan.js"))
  f <- .pf_wasm_file("praatfan_bg.wasm")
  ctx$assign("pf_wasm_b64", jsonlite::base64_enc(readBin(f, "raw", file.size(f))))
  ctx$eval(paste0("wasm_bindgen.initSync({module: praatfan_b64bytes(",
                  "pf_wasm_b64.replace(/\\s/g, ''))}); delete globalThis.pf_wasm_b64;"))
  .pf_wasm_env$ctx <- ctx
  .pf_wasm_env$audio_key <- NULL
  ctx
}

#' Drop the context (after an engine crash, or to free its memory).
#' @keywords internal
.pf_wasm_reset <- function() {
  .pf_wasm_env$ctx <- NULL
  .pf_wasm_env$audio_key <- NULL
  invisible(NULL)
}

#' Put a file's bytes into the context as \code{pf_audio}, unless it already
#' holds that file (keyed by path, size and modification time).
#' @keywords internal
.pf_wasm_load_audio <- function(ctx, path) {
  info <- file.info(path)
  if (is.na(info$size)) stop("failed to load ", path, ": no such file", call. = FALSE)
  key <- paste(path, info$size, format(info$mtime, "%Y-%m-%d %H:%M:%OS6"), sep = "\r")
  if (identical(.pf_wasm_env$audio_key, key)) return(invisible(NULL))
  .pf_wasm_env$audio_key <- NULL
  ctx$assign("pf_audio", readBin(path, "raw", info$size))
  .pf_wasm_env$audio_key <- key
  invisible(NULL)
}

#' One pipe request through the WASM engine; the response JSON text.
#'
#' The request is exactly the binary's: \code{wav_path} names the file, which
#' is read here and handed over as bytes (its extension is the format hint).
#' @keywords internal
.pf_wasm_text <- function(request) {
  ctx <- .pf_wasm_ctx()
  .pf_wasm_load_audio(ctx, request$wav_path)
  ctx$assign("pf_req", jsonlite::toJSON(request, auto_unbox = TRUE, digits = NA))
  txt <- tryCatch(
    ctx$eval("wasm_bindgen.pipe_request_bytes(pf_audio, pf_req)"),
    error = function(e) {
      # a Rust panic surfaces as a wasm trap; the instance is not reusable
      .pf_wasm_reset()
      stop("praatfan (WASM engine) failed: ", conditionMessage(e), call. = FALSE)
    })
  Encoding(txt) <- "UTF-8"
  txt
}
