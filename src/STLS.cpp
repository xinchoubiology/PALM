#include "STLS.h"

RcppExport SEXP PALM_STLS(SEXP x_, SEXP y_, SEXP lambda_, SEXP nlambda_, SEXP lmin_ratio_, SEXP alpha_, SEXP options_)
{
BEGIN_RCPP

    Rcpp::NumericMatrix X(x_);
    Rcpp::NumericVector y(y_);

    const int n = X.rows();
    const int p = X.cols();

    MatrixXf dataX(n, p);
    VectorXf dataY(n);

    std::copy(X.begin(), X.end(), dataX.data());
    std::copy(y.begin(), y.end(), dataY.data());

    ArrayXd lambda(as<ArrayXd>(lambda_));
    int nlambda = lambda.size();

    const double alpha = as<double>(alpha_);
    const double lmin_ratio = as<double>(lmin_ratio_);

    List options(options_);
    const int maxit       = as<int>(options["maxit"]);
    const double eps_abs  = as<double>(options["eps_abs"]);
    const double eps_rel  = as<double>(options["eps_rel"]);
    const bool center     = as<bool>(options["center"]);
    const bool warm_start = as<bool>(options["warm_start"]);

    if(nlambda < 1)
    {
        lambda  = get_Lambda(dataX, dataY, as<int>(nlambda_), lmin_ratio, alpha);
        nlambda = lambda.size();
    }

    List result = STLS_solver(dataX, dataY, lambda, nlambda, eps_abs, eps_rel, alpha, center, warm_start, maxit);

    result["lambda"] = lambda;
    return result;

END_RCPP
}

RcppExport SEXP CVPALM_STLS(SEXP x_, SEXP y_, SEXP foldix_, SEXP lambda_, SEXP nlambda_, SEXP lmin_ratio_, SEXP alpha_, SEXP options_)
{
BEGIN_RCPP
    Rcpp::NumericMatrix X(x_);
    Rcpp::NumericVector y(y_);

    const int n = X.rows();
    const int p = X.cols();

    MatrixXf dataX(n, p);
    VectorXf dataY(n);

    std::copy(X.begin(), X.end(), dataX.data());
    std::copy(y.begin(), y.end(), dataY.data());

    ArrayXd lambda(as<ArrayXd>(lambda_));
    int nlambda = lambda.size();

    const double alpha = as<double>(alpha_);
    const double lmin_ratio = as<double>(lmin_ratio_);

    List options(options_);
    const int maxit       = as<int>(options["maxit"]);
    const double eps_abs  = as<double>(options["eps_abs"]);
    const double eps_rel  = as<double>(options["eps_rel"]);
    const bool center     = as<bool>(options["center"]);
    const bool warm_start = as<bool>(options["warm_start"]);

    if(nlambda < 1)
    {
        lambda  = get_Lambda(dataX, dataY, as<int>(nlambda_), lmin_ratio, alpha);
        nlambda = lambda.size();
    }

    // cross-validation chunk
    List foldix(foldix_);

    const int nfold = foldix.size();
    List cvres(nfold);

    for(int i = 0; i < nfold; i++)
    {
        ArrayXd subidx(as<ArrayXd>(foldix[i]));
        MatrixXf subX(get_Rows(dataX, subidx));
        VectorXf suby(get_Idx(dataY, subidx));
        cvres[i] = STLS_solver(subX, suby, lambda, nlambda, eps_abs, eps_rel, alpha, center, warm_start, maxit);
    }

    return List::create(Named("lambda") = lambda, Named("cv.result") = cvres, Named("cv.train") = foldix);

END_RCPP
}


// stability selection on all lambda_max -> lambda_min
RcppExport SEXP SSPALM_STLS(SEXP x_, SEXP y_, SEXP permix_, SEXP lambda_, SEXP nlambda_, SEXP lmin_ratio_, SEXP alpha_, SEXP options_)
{
    Rcpp::NumericMatrix X(x_);
    Rcpp::NumericVector y(y_);

    const int n = X.rows();
    const int p = X.cols();

    MatrixXf dataX(n, p);
    VectorXf dataY(n);

    std::copy(X.begin(), X.end(), dataX.data());
    std::copy(y.begin(), y.end(), dataY.data());

    ArrayXd lambda(as<ArrayXd>(lambda_));
    int nlambda = lambda.size();

    const double alpha = as<double>(alpha_);
    const double lmin_ratio = as<double>(lmin_ratio_);

    List options(options_);
    const int maxit       = as<int>(options["maxit"]);
    const double eps_abs  = as<double>(options["eps_abs"]);
    const double eps_rel  = as<double>(options["eps_rel"]);
    const bool center     = as<bool>(options["center"]);
    const bool warm_start = as<bool>(options["warm_start"]);

    if(nlambda < 1)
    {
        lambda  = get_Lambda(dataX, dataY, as<int>(nlambda_), lmin_ratio, alpha);
        nlambda = lambda.size();
    }

    // stability-selection on all permutation 
    List permix(permix_);

    const int nperm = permix.size();
    List permres(nperm);

    for(int i = 0; i < nperm; i++)
    {
        ArrayXd subidx(as<ArrayXd>(permix[i]));
        MatrixXf subX(get_Rows(dataX, subidx));
        VectorXf suby(get_Idx(dataY, subidx));
        permres[i] = STLS_solver(subX, suby, lambda, nlambda, eps_abs, eps_rel, alpha, center, warm_start, maxit);
    }

    return List::create(Named("lambda") = lambda, Named("ss.beta") = permres, Named("ss.perm") = permix);
}