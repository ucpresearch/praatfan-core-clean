#' praatfan — installation of the native backend
#'
#' Installs the pipe engine binary that all analysis functions call
#' (\code{praatfan-open-pipe} by default; see \code{\link{pf_engine}} for
#' the GPL alternative).  No Python, pip, or conda required.
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

# HTTPS is tried first (works anonymously for a public repo, and with a
# credential helper for a private one); SSH is the fallback.
.pf_repo_urls <- function(slug) {
  c(paste0("https://github.com/", slug, ".git"),
    paste0("git@github.com:", slug, ".git"))
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

#' Install a pipe engine binary.
#'
#' Fetches a pre-built engine binary; compiling is a last resort.  No
#' Python, pip, or venv required.  Total footprint ~7 MB per engine.
#'
#' Three routes, tried in this order by \code{method = "auto"}:
#' \describe{
#'   \item{\code{"binary"}}{Download the binary for this platform from the
#'     engine repository's GitHub Releases.  If the repository is private
#'     and \code{gh} is installed and authenticated, the download is
#'     retried through \code{gh release download}, which carries the
#'     user's token.}
#'   \item{\code{"git"}}{Clone the engine's source repository and build
#'     with \code{cargo}.  Cloning inherits the user's git credentials
#'     (SSH key or credential helper), so this reaches a private
#'     repository without \code{gh}.  Requires a Rust toolchain.}
#'   \item{\code{"source"}}{Build a local source tree with \code{cargo}.}
#' }
#'
#' @param engine  Which engine to install: \code{"open"} (default) or
#'   \code{"gpl"}.  See \code{\link{pf_engine}}.  Both can be installed
#'   side by side.
#' @param source  Path to a source tree — either a repository root
#'   (containing \code{rust/Cargo.toml}) or the crate directory itself.
#'   If given, it is built directly and the other routes are skipped.  If
#'   \code{NULL} (default), the \code{"source"} route auto-detects a
#'   checkout relative to the current working directory and the installed
#'   package.
#' @param repo    Git URL for the \code{"git"} route.  Defaults to the
#'   engine's public HTTPS URL with an SSH fallback.
#' @param ref     Version to install: a release tag for the \code{"binary"}
#'   route, a branch or tag for the \code{"git"} route.  Default \code{NULL} —
#'   the latest release, and the repository's default branch.
#' @param method  Which install route to take: \code{"auto"} (default), or one
#'   of \code{"binary"}, \code{"git"}, \code{"source"} to force a single route.
#' @param force   Re-install even if already present.
#' @export
pf_setup <- function(engine = NULL, source = NULL, repo = NULL, ref = NULL,
                     method = c("auto", "binary", "git", "source"),
                     force = FALSE) {
  method <- match.arg(method)
  eng <- .pf_engine_of(engine)

  if (!force && .pf_is_ready(eng)) {
    bin <- tryCatch(.pf_pipe_bin(eng), error = function(e) "")
    message(eng$bin, " already available: ", bin)
    return(invisible(TRUE))
  }

  dir.create(.pf_bin_dir, showWarnings = FALSE, recursive = TRUE)

  if (!is.null(source)) {
    # Explicit source tree: use it, and let its errors surface directly.
    .pf_build_from_source(.pf_resolve_manifest_dir(source), eng)
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
               binary = .pf_download_binary(eng, ref),
               git    = .pf_clone_and_build(eng, repo, ref),
               source = {
                 found <- .pf_find_source(eng)
                 if (is.null(found)) {
                   stop("no local ", eng$src, " source tree found",
                        call. = FALSE)
                 }
                 .pf_build_from_source(found, eng)
               })
        NULL
      }, error = function(e) conditionMessage(e))

      if (is.null(err)) break
      failures[[route]] <- err
      if (length(routes) > 1L) message("  ", route, ": ", err)
    }

    if (length(failures) == length(routes)) {
      stop("Could not install ", eng$bin, ". Tried:\n",
           paste0("  - ", names(failures), ": ", unlist(failures),
                  collapse = "\n"),
           call. = FALSE)
    }
  }

  if (.pf_is_ready(eng)) {
    message(eng$bin, " ready at ", .pf_bin_dir)
  } else {
    stop("Installation failed. Check the output above for errors.",
         call. = FALSE)
  }
  invisible(TRUE)
}


#' Remove installed engine binaries.
#'
#' @param sources  Also remove the cached git clones created by
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
#'
#' Asset names are \code{<binary>-<os>-<arch>[.exe]}, built from the
#' engine's binary name.
#'
#' @keywords internal
.pf_platform <- function(eng) {
  os <- tolower(Sys.info()[["sysname"]])
  arch <- Sys.info()[["machine"]]

  suffix <- if (os == "darwin" && arch == "arm64") {
    "macos-aarch64"
  } else if (os == "darwin") {
    "macos-x86_64"
  } else if (os == "linux" && arch == "aarch64") {
    "linux-aarch64"
  } else if (os == "linux") {
    "linux-x86_64"
  } else if (os == "windows") {
    "windows-x86_64.exe"
  } else {
    stop("Unsupported platform: ", os, " ", arch, call. = FALSE)
  }

  list(os = os, arch = arch,
       asset = paste0(eng$bin, "-", suffix),
       exe = if (os == "windows") paste0(eng$bin, ".exe") else eng$bin)
}


#' Copy a binary into the install directory and make it executable.
#' @keywords internal
.pf_install_file <- function(src, plat) {
  dest <- file.path(.pf_bin_dir, plat$exe)
  if (!file.copy(src, dest, overwrite = TRUE)) {
    stop("could not copy ", src, " to ", dest, call. = FALSE)
  }
  if (plat$os != "windows") Sys.chmod(dest, "0755")
  message("  Installed: ", dest)
  invisible(dest)
}


#' Accept either a repository root or the Rust crate directory.
#' @keywords internal
.pf_resolve_manifest_dir <- function(tree) {
  tree <- normalizePath(tree, mustWork = TRUE)
  found <- .pf_try_manifest_dir(tree)
  if (is.null(found)) {
    stop("No Cargo.toml under ", tree, " or ", file.path(tree, "rust"),
         call. = FALSE)
  }
  found
}


#' Non-throwing variant: manifest dir under `tree`, or NULL.
#' @keywords internal
.pf_try_manifest_dir <- function(tree) {
  if (file.exists(file.path(tree, "rust", "Cargo.toml"))) {
    return(file.path(tree, "rust"))
  }
  if (file.exists(file.path(tree, "Cargo.toml"))) {
    return(tree)
  }
  NULL
}


#' @keywords internal
.pf_find_source <- function(eng) {
  roots <- c(
    normalizePath(".", mustWork = FALSE),
    normalizePath(file.path("..", eng$src), mustWork = FALSE),
    normalizePath(file.path(system.file(package = "praatfan"),
                            "..", "..", ".."),
                  mustWork = FALSE)
  )
  for (root in roots) {
    found <- .pf_try_manifest_dir(root)
    if (!is.null(found)) return(found)
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


#' Clone the engine's source repository with git and build its binary.
#'
#' The clone is cached under \code{<data dir>/src} and updated in place on
#' later calls.  git is used rather than a release download because it picks
#' up the user's own credentials, so a private repository works.
#'
#' @keywords internal
.pf_clone_and_build <- function(eng, repo = NULL, ref = NULL) {
  if (Sys.which("git") == "") {
    stop("git not found on PATH", call. = FALSE)
  }
  if (Sys.which("cargo") == "") {
    stop("cargo (Rust toolchain) not found on PATH. ",
         "Install from https://rustup.rs/", call. = FALSE)
  }

  dir.create(.pf_src_dir, showWarnings = FALSE, recursive = TRUE)
  dest <- file.path(.pf_src_dir, eng$src)
  urls <- if (is.null(repo)) .pf_repo_urls(eng$repo) else repo

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

  .pf_build_from_source(.pf_resolve_manifest_dir(dest), eng)
}


#' @keywords internal
.pf_build_from_source <- function(source, eng) {
  cargo <- Sys.which("cargo")
  if (cargo == "") {
    stop("cargo (Rust toolchain) not found on PATH.\n",
         "Install from https://rustup.rs/ or provide a pre-built binary.",
         call. = FALSE)
  }

  plat <- .pf_platform(eng)
  source <- normalizePath(source, mustWork = TRUE)
  manifest <- file.path(source, "Cargo.toml")
  message("Building ", eng$bin, " from source: ", source)
  status <- system2(cargo,
                    c("build", "--release", "--features", "pipe",
                      "--bin", eng$bin,
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


#' Download a pre-built binary from the engine's GitHub Releases.
#'
#' @param ref  Release tag to pull from; \code{NULL} uses the latest release.
#' @keywords internal
.pf_download_binary <- function(eng, ref = NULL) {
  plat <- .pf_platform(eng)
  name <- plat$asset

  url <- if (is.null(ref)) {
    paste0("https://github.com/", eng$repo,
           "/releases/latest/download/", name)
  } else {
    paste0("https://github.com/", eng$repo,
           "/releases/download/", ref, "/", name)
  }

  message("Downloading ", eng$bin, " for ", plat$os, "/", plat$arch,
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
                      c("release", "download", ref, "--repo", eng$repo,
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


.pf_is_ready <- function(eng = .pf_engine_of()) {
  env <- Sys.getenv("PRAATFAN_PIPE", "")
  if (nzchar(env) && file.exists(env)) return(TRUE)
  candidate <- file.path(.pf_bin_dir, eng$bin)
  if (file.exists(candidate) || file.exists(paste0(candidate, ".exe"))) {
    return(TRUE)
  }
  Sys.which(eng$bin) != ""
}

#' Ensure the engine binary is available, bootstrapping it if needed.
#'
#' On first use after `install_github`, the binary is fetched automatically
#' (the standard R pattern for packages wrapping a native tool — the GitHub
#' tarball that `install_github` installs contains no binaries).  Disable
#' with `options(praatfan.auto_setup = FALSE)` for a hard error instead.
#'
#' @keywords internal
.pf_ensure_ready <- function(eng = .pf_engine_of()) {
  if (.pf_is_ready(eng)) return(invisible(TRUE))
  if (isFALSE(getOption("praatfan.auto_setup", TRUE))) {
    stop(eng$bin, " not installed. Run pf_setup(engine = \"", eng$name,
         "\") first.", call. = FALSE)
  }
  message(eng$bin, " not found; installing it now. ",
          "(Disable with options(praatfan.auto_setup = FALSE).)")
  pf_setup(engine = eng$name)
}
