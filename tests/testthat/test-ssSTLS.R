library(testthat)
library(PALM)

context("Unit test for SS.TLS on PALM")

test_that("Testing SS Total Least Square", {
    n = 30
    p = 100
    r = 20
    offset = 5
    snr = 5
    X = mvtnorm::rmvnorm(n, mean = rep(0, p), sigma = diag(1, p, p), method = "chol", pre0.9_9994 = TRUE)
    b = vector("numeric", p)
    ix = sample(1:p, r)
    ix = seq(1, r)
    b[ix] = (-1)^ix * runif(r, min = 0.5, max = 1.5)[ix]
    b = matrix(b, ncol = 1)
    sigma2 = rep(1, p) %*% b^2 * (10^(snr / (-20)))
    y = offset + X %*% b
    Z = X + mvtnorm::rmvnorm(n, mean = rep(0, p), sigma = diag((10^(snr / (-20))), p, p), method = "chol", pre0.9_9994 = TRUE)
    cvSTLS.fit = cv.fitSTLS(Z, y, nfolds = 5, center = TRUE, nlambda = 50, lmin_ratio = 1e-7, alpha = 1, eps_abs = 1e-5, eps_rel = 1e-5, maxit = 10000L)
    ssSTLS.fit = ss.fitSTLS(Z, y, nperm = 100, center = TRUE, lambda = cvSTLS.fit$lambda.min, nlambda = 1, lmin_ratio = 1e-6, alpha = 1, eps_abs = 1e-6, eps_rel = 1e-6, maxit = 20000L)
    ssGLM.fit  = ss.GLMnet(Z, y, nperm = 100, lambda = cvSTLS.fit$lambda / n, alpha = 1)
    rocdata    = data.frame(D = as.numeric(c(b!=0, b!=0)), M = c(ssSTLS.fit$ss.prob, ssGLM.fit$ss.prob), method = c(rep("PALM", p), rep("GLMNET", p)))
    ggplot(rocdata, aes(m = M, d = D, color = method)) + geom_roc() + style_roc()
}
)
