#' The WASM engines: an engine running inside R
#'
#' The package ships the open engine twice: as the native
#' \code{praatfan-open-pipe} binary that \code{pf_setup()} installs, and as a
#' WebAssembly build of the same Rust code under \code{inst/wasm/}, run in an
#' embedded V8 (the \pkg{V8} package).  Both answer the same JSON pipe
#' protocol with the same numbers (to floating-point rounding); the binary is
#' faster (several threads per call), WASM needs nothing installed beyond this
#' package.
#'
#' The GPL engine has a WASM build too.  It is GPL-licensed, so it is not
#' shipped here: \code{pf_setup(engine = "gpl_wasm")} downloads it from the
#' GPL engine's releases into the user's data directory, and the
#' \code{"gpl_wasm"} engine runs it.  Both bundles load the same host shim
#' (\code{inst/wasm/shim.js}), each in its own V8 context.
#'
#' The file is read in R as raw bytes and decoded inside the engine, by the
#' same decoder the binary uses, so every format the binary reads works here
#' too.  The bytes of the last file are kept in each V8 context, so several
#' analyses of one file send it once.
#'
#' Parallel work: an engine runs one thread per call, so analysing several
#' files at once is the way to use more cores.  Each R process builds its own
#' V8 context; PSOCK clusters (\code{parallel::makePSOCKcluster},
#' \pkg{future}'s multisession) are the portable choice (Windows included).
#'
#' @name wasm
#' @keywords internal
NULL

.pf_wasm_env <- new.env(parent = emptyenv())
# per bundle ("open", "gpl"): the V8 context and the audio it holds
.pf_wasm_env$ctx <- list()
.pf_wasm_env$audio_key <- list()

.pf_wasm_file <- function(name) {
  p <- system.file("wasm", name, package = "praatfan")
  if (nzchar(p)) return(p)
  # a source tree loaded with pkgload: inst/ is not flattened
  file.path(system.file(package = "praatfan"), "inst", "wasm", name)
}

#' Where the downloaded GPL bundle lives.
#' @keywords internal
.pf_gpl_wasm_dir <- function() file.path(.pf_home(), "wasm", "gpl")

#' The glue script and module of a bundle: \code{"open"} (shipped) or
#' \code{"gpl"} (downloaded by \code{pf_setup(engine = "gpl_wasm")}).
#' @keywords internal
.pf_wasm_bundle <- function(which = "open") {
  switch(which,
    open = list(glue = .pf_wasm_file("praatfan.js"),
                wasm = .pf_wasm_file("praatfan_bg.wasm")),
    gpl  = list(glue = file.path(.pf_gpl_wasm_dir(), "praatfan_gpl.js"),
                wasm = file.path(.pf_gpl_wasm_dir(), "praatfan_gpl_bg.wasm")),
    stop("unknown WASM bundle ", deparse(which), call. = FALSE))
}

#' Why a WASM engine cannot run here, or \code{NULL} when it can.
#' @keywords internal
.pf_wasm_unavailable <- function(which = "open") {
  if (!requireNamespace("V8", quietly = TRUE)) {
    return("the V8 package is not installed (install.packages(\"V8\"))")
  }
  b <- .pf_wasm_bundle(which)
  if (!file.exists(b$glue) || !file.exists(b$wasm)) {
    return(if (which == "gpl") {
      "the GPL WASM engine is not installed (pf_setup(engine = \"gpl_wasm\"))"
    } else {
      "this installation has no WASM bundle (inst/wasm/)"
    })
  }
  NULL
}

#' A fresh V8 context with a bundle loaded and initialised.
#'
#' The glue is wasm-bindgen's \code{--target no-modules} output, which defines
#' a global \code{wasm_bindgen}; \code{shim.js} supplies the UTF-8
#' TextEncoder/TextDecoder an embedded V8 lacks and a base64 decoder.
#' @keywords internal
.pf_wasm_load <- function(glue, wasm) {
  ctx <- V8::v8()
  ctx$source(.pf_wasm_file("shim.js"))
  ctx$source(glue)
  ctx$assign("pf_wasm_b64", jsonlite::base64_enc(readBin(wasm, "raw", file.size(wasm))))
  ctx$eval(paste0("wasm_bindgen.initSync({module: praatfan_b64bytes(",
                  "pf_wasm_b64.replace(/\\s/g, ''))}); delete globalThis.pf_wasm_b64;"))
  ctx
}

#' The V8 context holding a bundle: created on first use, then cached.
#' @keywords internal
.pf_wasm_ctx <- function(which = "open") {
  ctx <- .pf_wasm_env$ctx[[which]]
  if (!is.null(ctx)) return(ctx)
  why <- .pf_wasm_unavailable(which)
  if (!is.null(why)) {
    stop("praatfan: the ", if (which == "gpl") "GPL " else "",
         "WASM engine cannot run: ", why, ".", call. = FALSE)
  }
  b <- .pf_wasm_bundle(which)
  ctx <- .pf_wasm_load(b$glue, b$wasm)
  .pf_wasm_env$ctx[[which]] <- ctx
  .pf_wasm_env$audio_key[[which]] <- NULL
  ctx
}

#' Drop a context (after an engine crash, a reinstall, or to free memory).
#' @keywords internal
.pf_wasm_reset <- function(which = "open") {
  .pf_wasm_env$ctx[[which]] <- NULL
  .pf_wasm_env$audio_key[[which]] <- NULL
  invisible(NULL)
}

#' Put a file's bytes into the context as \code{pf_audio}, unless it already
#' holds that file (keyed by path, size and modification time).
#' @keywords internal
.pf_wasm_load_audio <- function(ctx, path, which) {
  info <- file.info(path)
  if (is.na(info$size)) stop("failed to load ", path, ": no such file", call. = FALSE)
  key <- paste(path, info$size, format(info$mtime, "%Y-%m-%d %H:%M:%OS6"), sep = "\r")
  if (identical(.pf_wasm_env$audio_key[[which]], key)) return(invisible(NULL))
  .pf_wasm_env$audio_key[[which]] <- NULL
  ctx$assign("pf_audio", readBin(path, "raw", info$size))
  .pf_wasm_env$audio_key[[which]] <- key
  invisible(NULL)
}

#' One pipe request through a WASM engine; the response JSON text.
#'
#' The request is exactly the binary's: \code{wav_path} names the file, which
#' is read here and handed over as bytes (its extension is the format hint).
#' @keywords internal
.pf_wasm_text <- function(request, which = "open") {
  ctx <- .pf_wasm_ctx(which)
  .pf_wasm_load_audio(ctx, request$wav_path, which)
  ctx$assign("pf_req", jsonlite::toJSON(request, auto_unbox = TRUE, digits = NA))
  txt <- tryCatch(
    ctx$eval("wasm_bindgen.pipe_request_bytes(pf_audio, pf_req)"),
    error = function(e) {
      # a Rust panic surfaces as a wasm trap; the instance is not reusable
      .pf_wasm_reset(which)
      stop("praatfan (", which, " WASM engine) failed: ", conditionMessage(e),
           call. = FALSE)
    })
  Encoding(txt) <- "UTF-8"
  txt
}
