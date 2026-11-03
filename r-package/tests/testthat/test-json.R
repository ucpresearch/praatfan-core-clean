test_that("request doubles survive the JSON round trip bit for bit", {
  x <- c(1.0619687500000001, 0.1 + 0.2, 1 / 3, 0.025, 5500, 1e-300,
         123456.789012345678, (2.1625 + 2.36125) / 2, .Machine$double.eps)
  req <- list(wav_path = "a.wav", analyses = list(list(type = "band_energy",
                                                      times = x)))
  back <- jsonlite::fromJSON(praatfan:::.pf_to_json(req),
                             simplifyVector = TRUE)
  expect_identical(back$analyses$times[[1]], x)
})

test_that("scalars stay scalars and integers stay plain", {
  j <- praatfan:::.pf_to_json(list(type = "formant_burg",
                                   max_number_of_formants = 5L,
                                   window_length = 0.025))
  expect_match(j, '"max_number_of_formants":5[,}]')
  expect_identical(jsonlite::fromJSON(j)$window_length, 0.025)
})

test_that("an open engine of another version warns once per session", {
  eng <- praatfan:::.pf_engines$open
  rm(list = ls(praatfan:::.pf_version_warned), envir = praatfan:::.pf_version_warned)
  pkg <- as.character(utils::packageVersion("praatfan"))
  expect_silent(praatfan:::.pf_check_engine_version(eng, pkg))
  expect_warning(praatfan:::.pf_check_engine_version(eng, "0.0.1"), "version 0.0.1")
  expect_silent(praatfan:::.pf_check_engine_version(eng, "0.0.1"))
  expect_silent(praatfan:::.pf_check_engine_version(praatfan:::.pf_engines$gpl, "0.0.1"))
})

test_that("gpl and gpl_wasm of different versions warn once", {
  rm(list = ls(praatfan:::.pf_version_warned), envir = praatfan:::.pf_version_warned)
  rm(list = ls(praatfan:::.pf_gpl_seen), envir = praatfan:::.pf_gpl_seen)
  g <- praatfan:::.pf_engines
  expect_silent(praatfan:::.pf_check_engine_version(g$gpl, "0.1.16"))
  expect_silent(praatfan:::.pf_check_engine_version(g$gpl_wasm, "0.1.16"))
  expect_warning(praatfan:::.pf_check_engine_version(g$gpl_wasm, "0.1.15"), "gpl_wasm")
  expect_silent(praatfan:::.pf_check_engine_version(g$gpl_wasm, "0.1.15"))
})

test_that("a pinned binary that does not exist is an error, not a fallback", {
  withr_env <- Sys.getenv("PRAATFAN_PIPE", unset = NA)
  on.exit(if (is.na(withr_env)) Sys.unsetenv("PRAATFAN_PIPE") else Sys.setenv(PRAATFAN_PIPE = withr_env))
  Sys.setenv(PRAATFAN_PIPE = file.path(tempdir(), "no-such-binary"))
  expect_error(praatfan:::.pf_pipe_bin(praatfan:::.pf_engines$open), "does not exist")
  Sys.unsetenv("PRAATFAN_PIPE")
  old <- options(praatfan.pipe = file.path(tempdir(), "no-such-binary"))
  on.exit(options(old), add = TRUE)
  expect_error(praatfan:::.pf_is_ready(praatfan:::.pf_engines$open), "does not exist")
})

test_that("where = 'auto' tells home from elsewhere", {
  expect_equal(praatfan:::.pf_resolve_where("user"), "user")
  expect_equal(praatfan:::.pf_resolve_where("package"), "package")
  expect_true(praatfan:::.pf_under_home(file.path(path.expand("~"), "x")))
  skip_on_os("windows")
  expect_false(praatfan:::.pf_under_home("/opt/site-library/praatfan"))
})
