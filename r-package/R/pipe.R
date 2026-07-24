#' Internal pipe interface to the praatfan-open-pipe binary
#'
#' Sends JSON requests to the Rust \code{praatfan-open-pipe} binary via stdin
#' and parses JSON responses from stdout.
#'
#' @name pipe
#' @keywords internal
NULL

# --- Low-level pipe call -----------------------------------------------------

.pf_pipe_call <- function(request) {
  bin <- .pf_pipe_bin()
  json_in <- jsonlite::toJSON(request, auto_unbox = TRUE, digits = NA)

  err_file <- tempfile()
  on.exit(unlink(err_file), add = TRUE)

  json_out <- system2(bin, input = json_in, stdout = TRUE, stderr = err_file)

  status <- attr(json_out, "status")
  if (!is.null(status) && status != 0L) {
    err_msg <- paste(readLines(err_file, warn = FALSE), collapse = "\n")
    stop("praatfan-open-pipe failed (exit ", status, "):\n", err_msg,
         call. = FALSE)
  }

  resp <- jsonlite::fromJSON(paste(json_out, collapse = "\n"),
                             simplifyVector = TRUE,
                             simplifyDataFrame = FALSE)
  if (!isTRUE(resp$ok)) {
    stop("praatfan-open-pipe error: ",
         resp$error %||% "unknown error", call. = FALSE)
  }
  resp
}

# --- Binary location ---------------------------------------------------------

.pf_pipe_bin <- function() {
  # 1. Environment variable override
  env <- Sys.getenv("PRAATFAN_PIPE", "")
  if (nzchar(env) && file.exists(env)) return(env)

  # 2. Installed location
  candidate <- file.path(.pf_bin_dir, "praatfan-open-pipe")
  if (file.exists(candidate)) return(candidate)
  if (file.exists(paste0(candidate, ".exe"))) return(paste0(candidate, ".exe"))

  # 3. PATH
  found <- Sys.which("praatfan-open-pipe")
  if (found != "") return(found)

  stop("praatfan-open-pipe not found. Run pf_setup() first, or set ",
       "PRAATFAN_PIPE=/path/to/binary.", call. = FALSE)
}

# --- Null-coalescing operator (base R) ---------------------------------------

`%||%` <- function(x, y) if (is.null(x)) y else x
