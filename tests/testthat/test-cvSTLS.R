library(testthat)
library(PALM)

context("Unit test for CV.TLS on PALM")

test_that("Testing CV Total Least Square", {
    n = 200
    p = 500
    r = 5
    offset = 5
    X = mvtnorm::rmvnorm(n, mean = rep(0, p), sigma = diag(1, p, p), method = "chol", pre0.9_9994 = TRUE)
    b = vector("numeric", p)
    ix = sample(1:p, r)
    ix = seq(1, r)
    b[ix] = (-1)^ix * 10
    b = matrix(b, ncol = 1)
    sigma2 = rep(1, p) %*% b^2 * 10^(5 / (-20))
    y = offset + X %*% b + rnorm(n, sd = sqrt(1))
    Z = X + mvtnorm::rmvnorm(n, mean = rep(0, p), sigma = diag(10^(5 / (-20)), p, p), method = "chol", pre0.9_9994 = TRUE)
    cvSTLS.fit = cv.fitSTLS(Z, y, nfolds = 5, center = TRUE, nlambda = 50, lmin_ratio = 1e-7, alpha = 1, eps_abs = 1e-5, eps_rel = 1e-5, maxit = 10000L)
    STLS.fit   = fitSTLS(Z, y, center = TRUE, lambda = cvSTLS.fit$lambda.min, nlambda = 1, lmin_ratio = 1e-5, alpha = 1, eps_abs = 1e-6, eps_rel = 1e-6, maxit = 20000L)
    expect_equal(which(STLS.fit$beta != 0), which(b!=0))

    #-------------------------#
    n = 200
    p = 500
    r = 10
    offset = 5
    X = mvtnorm::rmvnorm(n, mean = rep(0, p), sigma = diag(1, p, p), method = "chol", pre0.9_9994 = TRUE)
    b = vector("numeric", p)
    ix = sample(1:p, r)
    ix = seq(1, r)
    b[ix] = (-1)^ix * 10
    b = matrix(b, ncol = 1)
    sigma2 = rep(1, p) %*% b^2 * 10^(5 / (-20))
    y = offset + X %*% b + rnorm(n, sd = sqrt(1))
    Z = X + mvtnorm::rmvnorm(n, mean = rep(0, p), sigma = diag(10^(5 / (-20)), p, p), method = "chol", pre0.9_9994 = TRUE)
    cvSTLS.fit = cv.fitSTLS(Z, y, nfolds = 5, center = TRUE, nlambda = 50, lmin_ratio = 1e-7, alpha = 1, eps_abs = 1e-5, eps_rel = 1e-5, maxit = 10000L)
    STLS.fit   = fitSTLS(Z, y, center = TRUE, lambda = cvSTLS.fit$lambda.min, nlambda = 1, lmin_ratio = 1e-5, alpha = 1, eps_abs = 1e-6, eps_rel = 1e-6, maxit = 20000L)
    expect_equal(which(STLS.fit$beta[,1] != 0), which(b!=0))

    #----------------#
    #-------------------------#
    n = 200
    p = 500
    r = 20
    offset = 5
    X = mvtnorm::rmvnorm(n, mean = rep(0, p), sigma = diag(1, p, p), method = "chol", pre0.9_9994 = TRUE)
    b = vector("numeric", p)
    ix = sample(1:p, r)
    ix = seq(1, r)
    b[ix] = (-1)^ix * runif(r, min = 5, max = 10)[ix]
    b = matrix(b, ncol = 1)
    sigma2 = rep(1, p) %*% b^2 * 10^(5 / (-20))
    y = offset + X %*% b + rnorm(n, sd = sqrt(1))
    Z = X + mvtnorm::rmvnorm(n, mean = rep(0, p), sigma = diag(10^(5 / (-20)), p, p), method = "chol", pre0.9_9994 = TRUE)
    cvSTLS.fit = cv.fitSTLS(Z, y, nfolds = 5, center = TRUE, nlambda = 50, lmin_ratio = 1e-7, alpha = 1, eps_abs = 1e-5, eps_rel = 1e-5, maxit = 10000L)
    STLS.fit   = fitSTLS(Z, y, center = TRUE, lambda = cvSTLS.fit$lambda.1se, nlambda = 1, lmin_ratio = 1e-5, alpha = 1, eps_abs = 1e-6, eps_rel = 1e-6, maxit = 20000L)
    expect_equal(which(STLS.fit$beta[,1] != 0), which(b!=0))
}
)
