#' praatfan — installation of the native backend
#'
#' Installs the \code{praatfan-open-pipe} binary (JSON stdin/stdout) that all
#' analysis functions call.  No Python, pip, or conda required.
#'
#' @name setup
#' @keywords internal
NULL

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

.pf_home <- if (.Platform$OS.type == "windows") {
  file.path(Sys.getenv("LOCALAPPDATA", Sys.getenv("APPDATA")), "praatfan")
} else {
  file.path(Sys.getenv("XDG_DATA_HOME", path.expand("~/.local/share")),
            "praatfan")
}
.pf_bin_dir <- file.path(.pf_home, "bin")
.pf_src_dir <- file.path(.pf_home, "src")

# Upstream Rust engine.  HTTPS is tried first (works anonymously for a public
# repo, and with a credential helper for a private one); SSH is the fallback.
.pf_repo_slug <- "ucpresearch/praatfan-core-clean"
.pf_repo_urls <- c(
  paste0("https://github.com/", .pf_repo_slug, ".git"),
  paste0("git@github.com:", .pf_repo_slug, ".git")
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

#' Install the praatfan-open-pipe binary.
#'
#' Fetches a pre-built \code{praatfan-open-pipe} binary; compiling is a last
#' resort.  No Python, pip, or venv required.  Total footprint ~7 MB.
#'
#' Three routes, tried in this order by \code{method = "auto"}:
#' \describe{
#'   \item{\code{"binary"}}{Download the binary for this platform from GitHub
#'     Releases.  If the repository is private and \code{gh} is installed and
#'     authenticated, the download is retried through \code{gh release
#'     download}, which carries the user's token.}
#'   \item{\code{"git"}}{Clone the source repository and build with
#'     \code{cargo}.  Cloning inherits the user's git credentials (SSH key or
#'     credential helper), so this reaches a private repository without
#'     \code{gh}.  Requires a Rust toolchain.}
#'   \item{\code{"source"}}{Build a local source tree with \code{cargo}.}
#' }
#'
#' @param source  Path to a praatfan source tree — either the repository root
#'   (containing \code{rust/Cargo.toml}) or the \code{rust/} directory itself.
#'   If given, it is built directly and the other routes are skipped.  If
#'   \code{NULL} (default), the \code{"source"} route auto-detects a checkout
#'   relative to the current working directory and the installed package.
#' @param repo    Git URL for the \code{"git"} route.  Defaults to the public
#'   HTTPS URL with an SSH fallback.
#' @param ref     Version to install: a release tag for the \code{"binary"}
#'   route, a branch or tag for the \code{"git"} route.  Default \code{NULL} —
#'   the latest release, and the repository's default branch.
#' @param method  Which install route to take: \code{"auto"} (default), or one
#'   of \code{"binary"}, \code{"git"}, \code{"source"} to force a single route.
#' @param force   Re-install even if already present.
#' @export
pf_setup <- function(source = NULL, repo = NULL, ref = NULL,
                     method = c("auto", "binary", "git", "source"),
                     force = FALSE) {
  method <- match.arg(method)

  if (!force && .pf_is_ready()) {
    env <- Sys.getenv("PRAATFAN_PIPE", "")
    bin <- if (nzchar(env) && file.exists(env)) env
           else if (file.exists(file.path(.pf_bin_dir, "praatfan-open-pipe")))
             file.path(.pf_bin_dir, "praatfan-open-pipe")
           else Sys.which("praatfan-open-pipe")
    message("praatfan-open-pipe already available: ", bin)
    return(invisible(TRUE))
  }

  dir.create(.pf_bin_dir, showWarnings = FALSE, recursive = TRUE)

  if (!is.null(source)) {
    # Explicit source tree: use it, and let its errors surface directly.
    .pf_build_from_source(.pf_resolve_manifest_dir(source))
  } else {
    routes <- switch(method,
                     auto   = c("binary", "git", "source"),
                     binary = "binary",
                     git    = "git",
                     source = "source")
    failures <- list()
    for (route in routes) {
      err <- tryCatch({
        switch(route,
               binary = .pf_download_binary(ref),
               git    = .pf_clone_and_build(repo, ref),
               source = {
                 found <- .pf_find_source()
                 if (is.null(found)) {
                   stop("no local praatfan source tree found", call. = FALSE)
                 }
                 .pf_build_from_source(found)
               })
        NULL
      }, error = function(e) conditionMessage(e))

      if (is.null(err)) break
      failures[[route]] <- err
      if (length(routes) > 1L) message("  ", route, ": ", err)
    }

    if (length(failures) == length(routes)) {
      stop("Could not install praatfan-open-pipe. Tried:\n",
           paste0("  - ", names(failures), ": ", unlist(failures),
                  collapse = "\n"),
           call. = FALSE)
    }
  }

  if (.pf_is_ready()) {
    message("praatfan-open-pipe ready at ", .pf_bin_dir)
  } else {
    stop("Installation failed. Check the output above for errors.",
         call. = FALSE)
  }
  invisible(TRUE)
}


#' Remove the praatfan-open-pipe binary.
#'
#' @param sources  Also remove the cached git clone created by
#'   \code{pf_setup(method = "git")}.  Default \code{TRUE}.
#' @export
pf_uninstall <- function(sources = TRUE) {
  if (dir.exists(.pf_bin_dir)) {
    message("Removing: ", .pf_bin_dir)
    unlink(.pf_bin_dir, recursive = TRUE)
  } else {
    message("Nothing to remove at ", .pf_bin_dir)
  }
  if (sources && dir.exists(.pf_src_dir)) {
    message("Removing: ", .pf_src_dir)
    unlink(.pf_src_dir, recursive = TRUE)
  }
  remaining <- list.files(.pf_home, all.files = TRUE, no.. = TRUE)
  if (length(remaining) == 0L && dir.exists(.pf_home)) {
    unlink(.pf_home, recursive = TRUE)
  }
  invisible(TRUE)
}

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

#' Platform naming for this machine (GitHub Releases asset names).
#' @keywords internal
.pf_platform <- function() {
  os <- tolower(Sys.info()[["sysname"]])
  arch <- Sys.info()[["machine"]]

  asset <- if (os == "darwin" && arch == "arm64") {
    "praatfan-open-pipe-macos-aarch64"
  } else if (os == "darwin") {
    "praatfan-open-pipe-macos-x86_64"
  } else if (os == "linux" && arch == "aarch64") {
    "praatfan-open-pipe-linux-aarch64"
  } else if (os == "linux") {
    "praatfan-open-pipe-linux-x86_64"
  } else if (os == "windows") {
    "praatfan-open-pipe-windows-x86_64.exe"
  } else {
    stop("Unsupported platform: ", os, " ", arch, call. = FALSE)
  }

  list(os = os, arch = arch, asset = asset,
       exe = if (os == "windows") "praatfan-open-pipe.exe"
             else "praatfan-open-pipe")
}


#' Copy a binary into the install directory and make it executable.
#' @keywords internal
.pf_install_file <- function(src, plat = .pf_platform()) {
  dest <- file.path(.pf_bin_dir, plat$exe)
  if (!file.copy(src, dest, overwrite = TRUE)) {
    stop("could not copy ", src, " to ", dest, call. = FALSE)
  }
  if (plat$os != "windows") Sys.chmod(dest, "0755")
  message("  Installed: ", dest)
  invisible(dest)
}


#' Accept either a repository root or the rust/ crate directory.
#' @keywords internal
.pf_resolve_manifest_dir <- function(tree) {
  tree <- normalizePath(tree, mustWork = TRUE)
  if (file.exists(file.path(tree, "rust", "Cargo.toml"))) {
    return(file.path(tree, "rust"))
  }
  if (file.exists(file.path(tree, "Cargo.toml"))) {
    return(tree)
  }
  stop("No Cargo.toml under ", tree, " or ", file.path(tree, "rust"),
       call. = FALSE)
}


#' @keywords internal
.pf_find_source <- function() {
  candidates <- c(
    normalizePath("rust", mustWork = FALSE),
    normalizePath("../praatfan-core-clean/rust", mustWork = FALSE),
    normalizePath(file.path(system.file(package = "praatfan"),
                            "..", "..", "..", "rust"),
                  mustWork = FALSE)
  )
  for (cand in candidates) {
    if (file.exists(file.path(cand, "Cargo.toml"))) return(cand)
  }
  NULL
}


#' @keywords internal
.pf_git <- function(args) {
  system2("git", args, stdout = "", stderr = "")
}


#' @keywords internal
.pf_git_out <- function(args) {
  out <- suppressWarnings(
    tryCatch(system2("git", args, stdout = TRUE, stderr = FALSE),
             error = function(e) character(0)))
  if (!is.null(attr(out, "status")) && attr(out, "status") != 0L) {
    return("")
  }
  paste(out, collapse = "")
}


#' Clone the source repository with git and build its binary.
#'
#' The clone is cached under \code{<data dir>/src} and updated in place on
#' later calls.  git is used rather than a release download because it picks
#' up the user's own credentials, so a private repository works.
#'
#' @keywords internal
.pf_clone_and_build <- function(repo = NULL, ref = NULL) {
  if (Sys.which("git") == "") {
    stop("git not found on PATH", call. = FALSE)
  }
  if (Sys.which("cargo") == "") {
    stop("cargo (Rust toolchain) not found on PATH. ",
         "Install from https://rustup.rs/", call. = FALSE)
  }

  dir.create(.pf_src_dir, showWarnings = FALSE, recursive = TRUE)
  dest <- file.path(.pf_src_dir, "praatfan-core-clean")
  urls <- if (is.null(repo)) .pf_repo_urls else repo

  # A cached clone of some other remote must not silently satisfy an explicit
  # repo = ...; drop it and clone the requested one instead.
  if (dir.exists(file.path(dest, ".git"))) {
    origin <- .pf_git_out(c("-C", shQuote(dest), "remote", "get-url", "origin"))
    if (!origin %in% urls) {
      message("Cached clone points at ", origin, "; re-cloning.")
      unlink(dest, recursive = TRUE)
    }
  }

  # Update an existing clone in place; fall back to a fresh clone if that
  # fails (e.g. the cached checkout is corrupt or the ref has gone away).
  if (dir.exists(file.path(dest, ".git"))) {
    message("Updating clone: ", dest)
    ok <- .pf_git(c("-C", shQuote(dest), "fetch", "--depth", "1", "--tags",
                    "origin", ref %||% "HEAD")) == 0L
    if (ok) {
      ok <- .pf_git(c("-C", shQuote(dest), "checkout", "--force",
                      "FETCH_HEAD")) == 0L
    }
    if (!ok) {
      message("  Update failed; re-cloning.")
      unlink(dest, recursive = TRUE)
    }
  }

  if (!dir.exists(file.path(dest, ".git"))) {
    cloned <- FALSE
    for (url in urls) {
      message("Cloning ", url,
              if (is.null(ref)) "" else paste0(" (", ref, ")"), " ...")
      args <- c("clone", "--depth", "1")
      if (!is.null(ref)) args <- c(args, "--branch", ref)
      if (.pf_git(c(args, shQuote(url), shQuote(dest))) == 0L) {
        cloned <- TRUE
        break
      }
      unlink(dest, recursive = TRUE)
    }
    if (!cloned) {
      stop("git clone failed for ", paste(urls, collapse = " and "), ". ",
           "For a private repository, check that your git credentials ",
           "(SSH key or credential helper) are set up, or pass an explicit ",
           "repo = \"git@github.com:owner/repo.git\".", call. = FALSE)
    }
  }

  .pf_build_from_source(file.path(dest, "rust"))
}


#' @keywords internal
.pf_build_from_source <- function(source) {
  cargo <- Sys.which("cargo")
  if (cargo == "") {
    stop("cargo (Rust toolchain) not found on PATH.\n",
         "Install from https://rustup.rs/ or provide a pre-built binary.",
         call. = FALSE)
  }

  plat <- .pf_platform()
  source <- normalizePath(source, mustWork = TRUE)
  manifest <- file.path(source, "Cargo.toml")
  message("Building praatfan-open-pipe from source: ", source)
  status <- system2(cargo,
                    c("build", "--release", "--features", "pipe",
                      "--bin", "praatfan-open-pipe",
                      "--manifest-path", shQuote(manifest)),
                    stdout = "", stderr = "")
  if (status != 0L) {
    stop("cargo build failed (exit ", status, ")", call. = FALSE)
  }

  built <- file.path(source, "target", "release", plat$exe)
  if (!file.exists(built)) {
    stop("Build succeeded but binary not found at: ", built, call. = FALSE)
  }
  .pf_install_file(built, plat)
}


#' Download a pre-built binary from GitHub Releases.
#'
#' @param ref  Release tag to pull from; \code{NULL} uses the latest release.
#' @keywords internal
.pf_download_binary <- function(ref = NULL) {
  plat <- .pf_platform()
  name <- plat$asset

  url <- if (is.null(ref)) {
    paste0("https://github.com/", .pf_repo_slug,
           "/releases/latest/download/", name)
  } else {
    paste0("https://github.com/", .pf_repo_slug,
           "/releases/download/", ref, "/", name)
  }

  message("Downloading praatfan-open-pipe for ", plat$os, "/", plat$arch,
          if (is.null(ref)) "" else paste0(" (", ref, ")"), "...")

  tmp <- tempfile()
  on.exit(unlink(tmp, recursive = TRUE), add = TRUE)
  src <- tmp
  got <- tryCatch({
    suppressWarnings(
      utils::download.file(url, tmp, mode = "wb", quiet = TRUE))
    file.exists(tmp) && file.size(tmp) > 0L
  }, error = function(e) FALSE)

  # A private repository serves 404 to an unauthenticated download; gh carries
  # the user's token, so retry through it when it is available.
  if (!got && Sys.which("gh") != "") {
    message("  Direct download failed; retrying via gh.")
    unlink(tmp)
    dir.create(tmp, showWarnings = FALSE, recursive = TRUE)
    status <- system2("gh",
                      c("release", "download", ref, "--repo", .pf_repo_slug,
                        "--pattern", shQuote(name), "--dir", shQuote(tmp),
                        "--clobber"),
                      stdout = "", stderr = "")
    src <- file.path(tmp, name)
    got <- status == 0L && file.exists(src)
  }

  if (!got) {
    stop("could not download ", name, " from ", url,
         " (a private repository needs an authenticated `gh`, or use ",
         "method = \"git\")", call. = FALSE)
  }

  .pf_install_file(src, plat)
}


.pf_is_ready <- function() {
  env <- Sys.getenv("PRAATFAN_PIPE", "")
  if (nzchar(env) && file.exists(env)) return(TRUE)
  candidate <- file.path(.pf_bin_dir, "praatfan-open-pipe")
  if (file.exists(candidate) || file.exists(paste0(candidate, ".exe"))) {
    return(TRUE)
  }
  Sys.which("praatfan-open-pipe") != ""
}

.pf_ensure_ready <- function() {
  if (!.pf_is_ready()) {
    stop("praatfan-open-pipe not installed. Run pf_setup() first.",
         call. = FALSE)
  }
}
