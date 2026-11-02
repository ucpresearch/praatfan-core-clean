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
