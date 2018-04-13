## PALM_STLS Class' methods
PALM_STLS$methods(
  initialize = function(X, y, ...) {
    if(nrow(X) != length(y)) {
      stop("nrow(X) should be equal to size of y.")
    }
    .self$X                = as.matrix(X)
    .self$y                = as.numeric(y)
    .self$lambda           = numeric(0)
    .self$nlambda          = 100L
    .self$lmin_ratio       = 1e-5
    .self$alpha            = 1.0
    .self$maxit            = 10000L
    .self$eps_abs          = 1e-6
    .self$eps_rel          = 1e-6
    .self$center           = TRUE
    .self$warm_start       = FALSE
  }
)

## print PALM_STLS object
PALM_STLS$methods(
  show_common = function() {
    cat(sprintf("X := %d x %d Matrix", nrow(.self$X), ncol(.self$X)))
    cat(sprintf("y := %d Vector"), length(.self$y))

    fields = setdiff(names(.refClassDef@fieldClasses), c("X", "y"))
    for(field in fields) {
      cat(field, " : ", paste(.self$field(field), collapse = " "), "\n", sep = "")
    }
  },
  show = function()
  {
    cat("Sparse total least square model in PALM \n \n")
    show_common()
  }
)

## penalty parameter set-up
PALM_STLS$methods(
  penalty = function(lambda = NULL, nlambda = 100, lmin_ratio = 1e-5, alpha = 1.0, ...) {
    lambda_val = sort(as.numeric(lambda), decreasing = TRUE)
    if(length(lambda_val) > 0 & any(lambda_val < 0)) {
      stop("lambda must be non-negative value.")
    }
    if(nlambda <= 0) {
      stop("nlambda must be a positive integer.")
    }
    if(alpha < 0 | alpha > 1) {
      stop("alpha must be located in (0, 1].")
    }

    .self$lambda        = lambda_val
    .self$nlambda       = nlambda
    .self$lmin_ratio    = lmin_ratio
    .self$alpha         = alpha

    invisible(.self)
  }
)


## options configuration
PALM_STLS$methods(
  options = function(maxit = 1000L, eps_abs = 1e-6, eps_rel = 1e-6, center = TRUE, warm_start = FALSE, ...) {
    if(maxit <= 0) {
      stop("maxit should be positive integer.")
    }
    if(eps_abs < 0 | eps_rel < 0) {
      stop("eps_abs/rel should be positive.")
    }

    .self$maxit      = maxit
    .self$eps_abs    = eps_abs
    .self$eps_rel    = eps_rel
    .self$center     = center
    .self$warm_start = warm_start

    invisible(.self)
  }
)


## fit
PALM_STLS$methods(
  fit = function(...) {
    res = .Call("PALM_STLS", .self$X, .self$y, .self$lambda, .self$nlambda, .self$lmin_ratio, .self$alpha,
                list(maxit       = .self$maxit,
                     eps_abs     = .self$eps_abs,
                     eps_rel     = .self$eps_rel,
                     center      = .self$center,
                     warm_start  = .self$warm_start),
                PACKAGE = "PALM")
    do.call(STLS_fit, res)
  }
)


### STLS_fit object

STLS_fit$methods(
  show_common = function() {
    cat("Lambda \n")
    print(.self$lambda)
    cat("\n")
    cat("Intercept \n")
    cat(.self$intercept)
    cat("\n")
    cat("Coef \n")
    cat(sprintf("%d x %d Sparse Matrix: \n", nrow(.self$beta), ncol(.self$beta)))
    cat("\n")
    cat("Niter \n")
    print(.self$niter)
  },
  show = function() {
    cat("PALM based Total Least Square fitting results \n")
    show_common()
  }
)


#' Fitting A sparse Total Least Square model Using PALM algorithm
#'
#' @title fitSTLS
#' @param X The observation data matrix
#' @param y The observation response vector
#' @param center Y = Y - mean(Y) and X = X - mean(X)
#' @param lambda The regularized lambda vector
#' @param nlambda The Number of lambda
#' @param lmin_ratio lambda_max / lambda_min
#' @param alpha alpha parameter for elastic net penalty (0,1](Ridge Regression --> Lasso)
#' @param maxit The maximal iteration of PALM algorithm
#' @param eps_abs Absolutely epsilon for generated varaince vector
#' @param eps_rel Relative epsilon for objective function value of generated vector
#' @param warm_start Warm start for regularizer parameter tuning
#' @return \code{STLS_fit} object
#' @import methods
#' @import Rcpp
#' @import ggplot2
#' @importClassesFrom Matrix dgCMatrix
#' @importMethodsFrom Matrix dim as.matrix coerce
#' @export
fitSTLS = function(X, y,
                   center     = TRUE,
                   lambda     = NULL,
                   nlambda    = 100,
                   lmin_ratio = 1e-4,
                   alpha      = 1,
                   eps_abs    = 1e-4,
                   eps_rel    = 1e-4,
                   maxit      = 1000L,
                   warm_start = TRUE) {
  fit = PALM_STLS(X, y)
  fit$penalty(lambda, nlambda, lmin_ratio, alpha)
  fit$options(maxit, eps_abs, eps_rel, center, warm_start)
  res = fit$fit()
  res
}



### cross-validation configuration
cv.PALM_STLS$methods(
  initialize = function(X, y, folds, ...) {
    if (nrow(X) != length(y)) {
      stop("nrow(X) should be equal to size of y.")
    }
    .self$foldix = llply(seq(max(folds)), function(x) {
                                             which(folds != x) - 1
                                           })
    .self$X                = as.matrix(X)
    .self$y                = as.numeric(y)
    .self$lambda           = numeric(0)
    .self$nlambda          = 100L
    .self$lmin_ratio       = 1e-5
    .self$alpha            = 1.0
    .self$maxit            = 10000L
    .self$eps_abs          = 1e-6
    .self$eps_rel          = 1e-6
    .self$center           = TRUE
    .self$warm_start       = FALSE
  }
)

## print cv.PALM_STLS
cv.PALM_STLS$methods(
  show_common = function() {
    cat(sprintf("X := %d x %d Matrix", nrow(.self$X), ncol(.self$X)))
    cat(sprintf("y := %d Vector"), length(.self$y))

    fields = setdiff(names(.refClassDef@fieldClasses), c("X", "y"))
    for(field in fields) {
      cat(field, " : ", paste(.self$field(field), collapse = " "), "\n", sep = "")
    }
  },
  show = function()
  {
    cat("CV TLS model in PALM \n \n")
    show_common()
  }
)

## penalty parameter set-up
cv.PALM_STLS$methods(
  penalty = function(lambda = NULL, nlambda = 100, lmin_ratio = 1e-5, alpha = 1.0, ...) {
    lambda_val = sort(as.numeric(lambda), decreasing = TRUE)
    if(length(lambda_val) > 0 & any(lambda_val < 0)) {
      stop("lambda must be non-negative value.")
    }
    if(nlambda <= 0) {
      stop("nlambda must be a positive integer.")
    }
    if(alpha < 0 | alpha > 1) {
      stop("alpha must be located in (0, 1].")
    }

    .self$lambda        = lambda_val
    .self$nlambda       = nlambda
    .self$lmin_ratio    = lmin_ratio
    .self$alpha         = alpha

    invisible(.self)
  }
)


## options configuration
cv.PALM_STLS$methods(
  options = function(maxit = 1000L, eps_abs = 1e-6, eps_rel = 1e-6, center = TRUE, warm_start = FALSE, ...) {
    if(maxit <= 0) {
      stop("maxit should be positive integer.")
    }
    if(eps_abs < 0 | eps_rel < 0) {
      stop("eps_abs/rel should be positive.")
    }

    .self$maxit      = maxit
    .self$eps_abs    = eps_abs
    .self$eps_rel    = eps_rel
    .self$center     = center
    .self$warm_start = warm_start

    invisible(.self)
  }
)

cv.PALM_STLS$methods(
  fit = function(...) {
    res = .Call("CVPALM_STLS", .self$X, .self$y, .self$foldix, .self$lambda, .self$nlambda, .self$lmin_ratio, .self$alpha,
                list(maxit       = .self$maxit,
                     eps_abs     = .self$eps_abs,
                     eps_rel     = .self$eps_rel,
                     center      = .self$center,
                     warm_start  = .self$warm_start),
                PACKAGE = "PALM")
    cvtest = list()
    n = nrow(.self$X)
    for(i in 1:length(res$cv.result)) {
      ix    = setdiff(seq(n), res$cv.train[[i]] + 1)
      xtest = .self$X[ix, , drop =]
      ytest = .self$y[ix]
      tmp   = xtest %*% res$cv.result[[i]]$beta
      obj   = vector("numeric", length(res$cv.result[[i]]$intercept))
      for(j in 1:length(res$cv.result[[i]]$intercept)) {
        obj[j] = crossprod(ytest - (tmp[,j] + res$cv.result[[i]]$intercept[j]))
        obj[j] = 1/2 * obj[j] / (1 + sum((res$cv.result[[i]]$beta[,j])^2))
      }
      cvtest[[i]] = obj
    }
    cvtest = do.call(rbind, cvtest)
    res[["cvm"]]  = colMeans(cvtest)
    res[["cvsd"]] = apply(cvtest, 2, sd)
    res
  }
)

#' Cross-validation for Sparse Total Least Square model Using PALM algorithm
#'
#' @title cv.fitSTLS
#' @param X The observation data matrix
#' @param y The observation response vector
#' @param nfolds Number of folds in cross-validation
#' @param foldid Folds id in N-fold cross-validation
#' @param center Y = Y - mean(Y) and X = X - mean(X)
#' @param lambda The regularized lambda vector
#' @param nlambda The Number of lambda
#' @param lmin_ratio lambda_max / lambda_min
#' @param alpha alpha parameter for elastic net penalty (0,1](Ridge Regression --> Lasso)
#' @param maxit The maximal iteration of PALM algorithm
#' @param eps_abs Absolutely epsilon for generated varaince vector
#' @param eps_rel Relative epsilon for objective function value of generated vector
#' @param warm_start Warm start for regularizer parameter tuning
#' @return \code{STLS_fit} object
#' @import methods
#' @import Rcpp
#' @import ggplot2
#' @importClassesFrom Matrix dgCMatrix
#' @importMethodsFrom Matrix dim as.matrix coerce
#' @export
cv.fitSTLS = function(X, y,
                   nfolds     = 5,
                   foldid     = NULL,
                   center     = TRUE,
                   lambda     = NULL,
                   nlambda    = 100,
                   lmin_ratio = 1e-4,
                   alpha      = 1,
                   eps_abs    = 1e-4,
                   eps_rel    = 1e-4,
                   maxit      = 1000L,
                   warm_start = TRUE) {
  if(nfolds >= 3 & is.null(foldid)) {
    foldid = sample(seq(nfolds), size = nrow(X), replace = T)
  } else if(!is.null(foldid)) {
    nfolds = max(foldid)
  }
  cvfit = cv.PALM_STLS(X, y, foldid)
  cvfit$penalty(lambda, nlambda, lmin_ratio, alpha)
  cvfit$options(maxit, eps_abs, eps_rel, center, warm_start)
  cvfit = cvfit$fit()
  # get lambda.1se
  opt.lambda = minLambda(cvfit$lambda, cvfit$cvm, cvfit$cvsd)
  res   = list(lambda = cvfit$lambda, cvres = cvfit$cv.result, cvtrain = cvfit$cv.train, cvm = cvfit$cvm, cvsd = cvfit$cvsd,
               lambda.min = cvfit$lambda[which.min(cvfit$cvm)], lambda.1se = opt.lambda$lambda.1se, lambda.opt = opt.lambda$lambda.opt)
  do.call(cv.STLS_fit, res)
}


##' plot cross-validation curive and search for lambda.min
##'
##' @rdname plot.cv.STLS_fit
##' @param x fitted cv.STLS_fit
##' @method plot cv.STLS_fit
##' @export
plot.cv.STLS_fit = function(x, ...) {
  cvSTLS = data.frame(lambda = log10(x$lambda), cvm = x$cvm, cerr = x$cvsd)
  ggplot(cvSTLS, aes(x = lambda, y = cvm)) + geom_errorbar(aes(ymin = cvm - cerr, ymax = cvm + cerr)) +
    geom_line() + geom_point() + scale_y_log10() +
    xlab(expression(log(lambda))) + ylab("objective value") +
    ggtitle("cv.obj ~ lambda") +
    geom_vline(xintercept = log10(x$lambda.min), linetype = "dashed") +
    scale_colour_PALM() + theme_PALM()
}


### stability-selection configuration
ss.PALM_STLS$methods(
  initialize = function(X, y, nperm = 100, ...) {
    if (nrow(X) != length(y)) {
      stop("nrow(X) should be equal to size of y.")
    }
    .self$X                = as.matrix(X)
    .self$y                = as.numeric(y)
    n = nrow(X)
    k = floor(n / 2)
    index = rep(c(0, 1), c(n-k, k))
    permB = replicate(nperm, sample(index))[sample(1:n), , drop = FALSE]
    .self$permix           = llply(1:nperm, function(x) {
                                              which(permB[,x] != 0)
                                            })
    .self$lambda           = numeric(0)
    .self$nlambda          = 100L
    .self$lmin_ratio       = 1e-5
    .self$alpha            = 1.0
    .self$maxit            = 10000L
    .self$eps_abs          = 1e-6
    .self$eps_rel          = 1e-6
    .self$center           = TRUE
    .self$warm_start       = FALSE
  }
)

## print ss.PALM_STLS
ss.PALM_STLS$methods(
  show_common = function() {
    cat(sprintf("X := %d x %d Matrix", nrow(.self$X), ncol(.self$X)))
    cat(sprintf("y := %d Vector"), length(.self$y))

    fields = setdiff(names(.refClassDef@fieldClasses), c("X", "y"))
    for(field in fields) {
      cat(field, " : ", paste(.self$field(field), collapse = " "), "\n", sep = "")
    }
  },
  show = function()
  {
    cat("SS TLS model in PALM \n \n")
    show_common()
  }
)

## penalty parameter set-up
ss.PALM_STLS$methods(
  penalty = function(lambda = NULL, nlambda = 100, lmin_ratio = 1e-5, alpha = 1.0, ...) {
    lambda_val = sort(as.numeric(lambda), decreasing = TRUE)
    if(length(lambda_val) > 0 & any(lambda_val < 0)) {
      stop("lambda must be non-negative value.")
    }
    if(nlambda <= 0) {
      stop("nlambda must be a positive integer.")
    }
    if(alpha < 0 | alpha > 1) {
      stop("alpha must be located in (0, 1].")
    }

    .self$lambda        = lambda_val
    .self$nlambda       = nlambda
    .self$lmin_ratio    = lmin_ratio
    .self$alpha         = alpha

    invisible(.self)
  }
)


## options configuration
ss.PALM_STLS$methods(
  options = function(maxit = 1000L, eps_abs = 1e-6, eps_rel = 1e-6, center = TRUE, warm_start = FALSE, ...) {
    if(maxit <= 0) {
      stop("maxit should be positive integer.")
    }
    if(eps_abs < 0 | eps_rel < 0) {
      stop("eps_abs/rel should be positive.")
    }

    .self$maxit      = maxit
    .self$eps_abs    = eps_abs
    .self$eps_rel    = eps_rel
    .self$center     = center
    .self$warm_start = warm_start

    invisible(.self)
  }
)

ss.PALM_STLS$methods(
  fit = function(...) {
    res = .Call("SSPALM_STLS", .self$X, .self$y, .self$permix, .self$lambda, .self$nlambda, .self$lmin_ratio, .self$alpha,
                list(maxit       = .self$maxit,
                     eps_abs     = .self$eps_abs,
                     eps_rel     = .self$eps_rel,
                     center      = .self$center,
                     warm_start  = .self$warm_start),
                PACKAGE = "PALM")
    res
  }
)


#' Stability-Selection for Sparse Total Least Square model Using PALM algorithm
#'
#' @title ss.fitSTLS
#' @param X The observation data matrix
#' @param y The observation response vector
#' @param nperm Number of permutaiton of stabilioty-selection
#' @param center Y = Y - mean(Y) and X = X - mean(X)
#' @param lambda The regularized lambda vector
#' @param nlambda The Number of lambda
#' @param lmin_ratio lambda_max / lambda_min
#' @param alpha alpha parameter for elastic net penalty (0,1](Ridge Regression --> Lasso)
#' @param maxit The maximal iteration of PALM algorithm
#' @param eps_abs Absolutely epsilon for generated varaince vector
#' @param eps_rel Relative epsilon for objective function value of generated vector
#' @param warm_start Warm start for regularizer parameter tuning
#' @return \code{STLS_fit} object
#' @import methods
#' @import Rcpp
#' @import ggplot2
#' @importClassesFrom Matrix dgCMatrix
#' @importMethodsFrom Matrix dim as.matrix coerce
#' @export
ss.fitSTLS = function(X, y,
                      nperm      = 100,
                      center     = TRUE,
                      lambda     = NULL,
                      nlambda    = 100,
                      lmin_ratio = 1e-4,
                      alpha      = 1,
                      eps_abs    = 1e-4,
                      eps_rel    = 1e-4,
                      maxit      = 1000L,
                      warm_start = TRUE) {
  ssfit = ss.PALM_STLS(X, y, nperm)
  ssfit$penalty(lambda, nlambda, lmin_ratio, alpha)
  ssfit$options(maxit, eps_abs, eps_rel, center, warm_start)
  ssfit = ssfit$fit()
  ssfit[["ss.prob"]] = llply(ssfit$ss.beta, `[[`, "beta")
  ssfit[["ss.prob"]] = Reduce(`+`, llply(ssfit[["ss.prob"]], function(x){as.matrix(x != 0)}))
  ssfit[["ss.prob"]] = rowSums(ssfit[["ss.prob"]]) / (nperm * nlambda)
  ssfit
}








