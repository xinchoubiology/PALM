minLambda = function(lambda, cvm, cvsd) {
  cvmin = min(cvm, na.rm = TRUE)
  ixmin = cvm <= cvmin
  lambda.min = max(lambda[ixmin], na.rm = TRUE)
  ixmin  = match(lambda.min, lambda)
  min1se = (cvm + cvsd)[ixmin]
  ix1se  = cvm <= min1se
  lambda.1se = max(lambda[ix1se], na.rm = TRUE)
  ixopt  = which(diff(cvm) > 0)
  ixopt  = max(ixopt)
  lambda.opt = max(lambda[ixopt])
  list(lambda.min = lambda.min, lambda.1se = lambda.1se, lambda.opt = lambda.opt)
}
