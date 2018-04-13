library(PALM)
library(glmnet)
library(mvtnorm)
library(ggplot2)
library(optparse)

genx = function(n, p, rho = 1) {
  X = mvtnorm::rmvnorm(n, mean = rep(0, p), sigma = diag(rho, p, p), method = "chol", pre0.9_9994 = TRUE)
  X
}

generry = function(X, beta, p, snr, rho, b0) {
  sigma2 = rep(rho, p) %*% beta^2 * (10^(snr / (-20)))
  y = b0 + X %*% beta + rnorm(nrow(X), sd = sqrt(sigma2))
  y
}

generrx = function(X, snr, rho) {
  Z = X + mvtnorm::rmvnorm(nrow(X), mean = rep(0, ncol(X)), sigma = diag(rho * 10^(snr / (-20)), ncol(X), ncol(X)), method = "chol", pre0.9_9994 = TRUE)
  Z
}

## simulation generation X matrix
genmodel = function(nobs = 200, nvars = 400, rho = 1, snr = 10, supp = 10, offset = 5, scale = 10) {
  X    = genx(nobs, nvars, rho)
  beta = vector("numeric", nvars)
  ix   = sample(1:nvars, supp)
  beta[ix] = (-1)^ix * runif(nvars, min = scale / 2, max = scale)[ix]
  y    = generry(X, beta, nvars, snr, rho, offset)
  Z    = generrx(X, snr, rho)

  ## PALM cross-validation
  cvSTLS.fit = cv.fitSTLS(Z, y, nfolds = 5, center = TRUE, nlambda = 50, lmin_ratio = 1e-6, alpha = 1, eps_abs = 1e-5, eps_rel = 1e-5, maxit = 10000L)
  STLS.fit   = fitSTLS(Z, y, center = TRUE, lambda = cvSTLS.fit$lambda.min, nlambda = 1, lmin_ratio = 1e-6, alpha = 1, eps_abs = 1e-6, eps_rel = 1e-6, maxit = 20000L)

  power.palm = sum(STLS.fit$beta[ix] != 0) / supp
  fdr.palm   = length(setdiff(which(STLS.fit$beta != 0), ix)) / sum(STLS.fit$beta != 0)
  if(is.nan(fdr.palm)) {
    fdr.palm = 0
  }

  ## err.palm estimation error for coefficient
  err.palm = (sum((STLS.fit$beta - beta)^2) + (offset - STLS.fit$intercept)^2) / (sum(beta^2) + offset^2)

  ## GLMNET cross-validation
  lambda     = cvSTLS.fit$lambda
  cvSLS.fit  = cv.glmnet(Z, y, nfolds = 5, alpha = 1, nlambda = 50, lambda = lambda / nobs)
  SLS.fit    = glmnet(Z, y, alpha = 1, lambda = cvSLS.fit$lambda.1se)

  power.glm  = sum(SLS.fit$beta[ix] != 0) / supp
  fdr.glm    = length(setdiff(which(SLS.fit$beta != 0), ix)) / sum(SLS.fit$beta != 0)
  if(is.nan(fdr.glm)) {
    fdr.glm  = 0
  }

  ## err.glm estimation error for coefficient
  err.glm = (sum((SLS.fit$beta - beta)^2) + (offset - as.numeric(SLS.fit$a0))^2) / (sum(beta^2) + offset^2)

  data.frame(n = nobs, p = nvars, sp = supp, snr = snr,
             power.palm = power.palm, power.glm = power.glm,
             fdr.palm = fdr.palm, fdr.glm = fdr.glm,
             rerr.palm = err.palm, rerr.glm = err.glm)
}

set.seed(as.numeric(Sys.time()))

PALM4snr = list()
for(snr in c(40, 30, 20, 10, 5, 2)) {
  ### replication
  tmp = list()
  for(i in 1:50) {
    tmp[[i]] = genmodel(nobs = 200, nvars = 400, rho = 1, snr = snr, supp = 20, offset = 5, scale = 10)
  }
  PALM4snr[[as.character(snr)]] = colMeans(do.call(rbind, tmp))
}

PALM4snr = do.call(rbind, PALM4snr)



## stability selection
ss.GLMnet = function(X, y, nperm = 100, lambda = NULL, alpha = 1) {
  n = nrow(X)
  k = floor(n / 2)
  index   = rep(c(0, 1), c(n-k, k))
  permB   = replicate(nperm, sample(index))[sample(1:n), , drop = FALSE]
  permix  = llply(1:nperm, function(x) { which(permB[,x] != 0) })
  glmobj  = cv.glmnet(X, y, alpha = alpha, lambda = lambda, nfolds = 5)
  ss.glmnet = list()
  for(i in 1:nperm) {
    tmp   = glmnet(X[permix[[i]], ], y[permix[[i]]], alpha = alpha, lambda = glmobj$lambda.min)
    ss.glmnet[[i]] = tmp$beta
  }
  ssfit = list(lambda = lambda, ss.beta = ss.glmnet)
  ssfit[["ss.prob"]] = Reduce(`+`, llply(ssfit$ss.beta, function(x){as.matrix(x != 0)}))
  ssfit[["ss.prob"]] = as.numeric(rowSums(ssfit[["ss.prob"]]) / nperm)
  ssfit
}



