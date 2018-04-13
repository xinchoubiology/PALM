library(testthat)
library(PALM)

context("Unit test for TLS on PALM")

test_that("Testing Total Least Square", {
  X      = matrix(c(7, 8, 1, 15), nrow = 2)
  y      = c(13, 5)
  alpha  = 0.8
  lambda = 5.3
  STLS.fit = fitSTLS(X, y, center = FALSE, lambda = 5.3, nlambda = 1, lmin_ratio = 1e-5, alpha = 0.8, eps_abs = 1e-5, eps_rel = 1e-5, maxit = 10000L)
  expect_equal(STLS.fit$beta[1],  1.45,  tolerance = 0.01)
  expect_equal(STLS.fit$beta[2], -0.36,  tolerance = 0.01)
}
)
