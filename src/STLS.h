#ifndef STLS_H
#define STLS_H

#include <Eigen/Core>
#include "./PALMSTls.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::ArrayXd;

using Rcpp::List;
using Rcpp::IntegerVector;
using Rcpp::Named;
using Rcpp::as;

// utility function
template<class MatType, class RowIndexType>
inline MatType get_Rows(const MatType& X, const RowIndexType& ix)
{
    MatType subRows(ix.size(), X.cols());
    for(unsigned int i = 0; i < ix.size(); ++i)
    {
        subRows.row(i) = X.row(ix(i));
    }
    return subRows;
}

template<class VecType, class IndexType>
inline VecType get_Idx(const VecType& y, const IndexType& ix)
{
    VecType subIdx(ix.size());
    for(unsigned int i = 0; i < ix.size(); ++i)
    {
        subIdx(i) = y(ix(i));
    }
    return subIdx;
}


List STLS_solver(MatrixXf &dataX, VectorXf &dataY, ArrayXd &lambda,
                 int nlambda, double eps_abs, double eps_rel, double alpha,
                 bool center, bool warm_start, int maxit)
{
    const int n = dataX.rows();
    const int p = dataX.cols();

    double meany = 0;
    ArrayXd meanX;
    if(center){
      meany  = dataY.mean();
      meanX.resize(p);
      for(int i = 0; i < p; i++)
      {
        float *begin = &dataX(0, i);
        float *end   = begin + n;
        meanX[i]     = dataX.col(i).mean();
        std::transform(begin, end, begin, std::bind2nd(std::minus<double>(), meanX[i]));
      }
    }

    PALMSTls *PALM_STls_Solver;
    PALM_STls_Solver = new PALMSTls(dataX, dataY, eps_abs, eps_rel, alpha);

    Eigen::SparseMatrix<float> beta(p, nlambda);
    VectorXf intercept(nlambda);
    IntegerVector niter(nlambda);

    double ilambda = 0;

    for(int i = 0; i < nlambda; i++)
    {
        ilambda = lambda[i];
        if(i == 0) {
            PALM_STls_Solver->init(ilambda, false);
        } else {
            PALM_STls_Solver->init(ilambda, warm_start);
        }

        niter[i]                        = PALM_STls_Solver->solve(maxit);
        Eigen::SparseVector<double> res = PALM_STls_Solver->get_x();

        intercept[i] = meany;

        for(Eigen::SparseVector<double>::InnerIterator iter(res); iter; ++iter)
        {
            if(iter.value() != 0)
            {
              beta.insert(iter.index(), i) = iter.value();
              if(center)
              {
                intercept[i] -= (iter.value() * meanX[iter.index()]);
              }
            }
        }
    }
    beta.makeCompressed();

    delete PALM_STls_Solver;
    return List::create(Named("beta") = beta, Named("niter") = niter, Named("intercept") = intercept);
}

ArrayXd get_Lambda(MatrixXf dataX, VectorXf dataY, int nlambda, double lmin_ratio = 1e-5, double alpha = 1.0, bool center = true)
{
    ArrayXd lambda(nlambda);

    const int n = dataX.rows();
    const int p = dataX.cols();

    double meany = 0;
    ArrayXd meanX;
    if(center){
      meany  = dataY.mean();
      meanX.resize(p);
      for(int i = 0; i < p; i++)
      {
        float *begin = &dataX(0, i);
        float *end   = begin + n;
        meanX[i]     = dataX.col(i).mean();
        std::transform(begin, end, begin, std::bind2nd(std::minus<double>(), meanX[i]));
      }
    }

    PALMSTls *PALM_STls_Solver;
    PALM_STls_Solver = new PALMSTls(dataX, dataY, 1e-6, 1e-6, alpha);

    double lambda_max = PALM_STls_Solver->get_lambda0();
    Rcpp::Rcout << "LAMBDA MAX = " << lambda_max << std::endl;
    double lambda_min = lambda_max * lmin_ratio;

    lambda.setLinSpaced(nlambda, std::log(lambda_max), std::log(lambda_min));
    lambda = lambda.exp();

    return lambda;
}

#endif // STLS_H
