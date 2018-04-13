## PALM based total least square
PALM_STLS = setRefClass("PALM_STLS",
                        fields = list(X             = "matrix",
                                      y             = "numeric",
                                      lambda        = "numeric",
                                      nlambda       = "numeric",
                                      lmin_ratio    = "numeric",
                                      alpha         = "numeric",
                                      maxit         = "integer",
                                      eps_abs       = "numeric",
                                      eps_rel       = "numeric",
                                      center        = "logical",
                                      warm_start    = "logical"))

## Fitting result object
STLS_fit  = setRefClass("STLS_fit",
                        fields = list(lambda    = "numeric",
                                      intercept = "numeric",
                                      beta      = "dgCMatrix",
                                      niter     = "integer"))


## PALM-CV
cv.PALM_STLS = setRefClass("cv.PALM_STLS",
                           fields = list(X             = "matrix",
                                         y             = "numeric",
                                         foldix        = "list",
                                         lambda        = "numeric",
                                         nlambda       = "numeric",
                                         lmin_ratio    = "numeric",
                                         alpha         = "numeric",
                                         maxit         = "integer",
                                         eps_abs       = "numeric",
                                         eps_rel       = "numeric",
                                         center        = "logical",
                                         warm_start    = "logical"))


cv.STLS_fit = setRefClass("cv.STLS_fit",
                          fields = list(lambda      = "numeric",
                                        cvres       = "list",
                                        cvtrain     = "list",
                                        cvm         = "numeric",
                                        cvsd        = "numeric",
                                        lambda.min  = "numeric",
                                        lambda.1se  = "numeric",
                                        lambda.opt  = "numeric"))


## SS-PALM
ss.PALM_STLS = setRefClass("ss.PALM_STLS",
                           fields = list(X             = "matrix",
                                         y             = "numeric",
                                         permix        = "list",
                                         lambda        = "numeric",
                                         nlambda       = "numeric",
                                         lmin_ratio    = "numeric",
                                         alpha         = "numeric",
                                         maxit         = "integer",
                                         eps_abs       = "numeric",
                                         eps_rel       = "numeric",
                                         center        = "logical",
                                         warm_start    = "logical"))

ss.STLS_fit = setRefClass("ss.STLS_fit",
                          fields = list(ss.prob  = "numeric"))


