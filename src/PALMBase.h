#ifndef PALMBASE_H
#define PALMBASE_H

#include <RcppEigen.h>

// General problem setting
//    minimize H(X) + sum_{i=1}^p f_i(X_i)
//    minimize nonconvex function + separable regularization terms
template<typename VecTypeX, typename VecTypeG>
class PALMBase
{
protected:
  typedef float Scalar;
  const int dim_main;   // dimension of x parameter vector

  VecTypeX main_x;      // optimized parameter coefficients
  VecTypeG aux_gamma;   // auxiliary multipliers of lipschitz modulus in proximal operator
  VecTypeG grad_x;      // previous gradient of lambda_{k-1} parameter coefficients

  double eps_abs;       // absolute tol
  double eps_rel;       // relative tol

  double eps_iter;      // iteration updated gap tol (params)
  double eps_obj;       // primal objective function updated gap tol (obj)

  double iter_gap;      // ||x_{k+1} - x_k||
  double obj_gap;       // ||f_{k+1} - f_k|| / ||f_k||

  double obj_val;       // objective function value
  double update_obj;    // updated objective value kept in stack

  int iterator;         // iteration index counter

  // objective function operator overload
  virtual double obj(const VecTypeX &x) = 0;

  // block-wise update objective function
  // res_x = (x1_{k+1}, x2_{k+1},..., x?_{k+1}, x(?+1)_k, ..., xp_k)
  virtual void next(VecTypeX &update_x, int offset) = 0;

  // sequential strong rule build
  virtual void ssr(VecTypeG &update_g, double lambda_prev, double lambda) = 0;

  // get iteration updated gap
  virtual double get_eps_iter()
  {
    return eps_abs;
  }

  // get primal objective function updated gap tol (obj)
  virtual double get_eps_obj()
  {
    return eps_obj;
  }

  // ||x_{k+1} - x_k||
  virtual double get_iter_gap(const VecTypeX &update_x)
  {
    return (update_x - main_x).norm();
  }

  //  ||f_{k+1} - f_k|| / ||f_k||
  virtual double get_obj_gap(double update_obj) {
    return std::abs(obj_val - update_obj) / std::abs(obj_val);
  }

  // get lipschitz modulus for proximal operator for block-wise updated x
  virtual double get_lipschitz(int offset) = 0;

  /* prox_{c?_k}^{f_i}(x?^k - \frac{1}{c_k}\nabla H(x1_{k+1}, ..., x(?-1)_{k+1}, x?_k, ..., xp_k))
  */
  // get \nabla_i H(X)
  virtual double get_subgrad(const VecTypeX &update_x, int offset) = 0;

  // proximal operator of specific function
  // proximal operator for f_i(X_i) is an analytic closed form function
  virtual double prox_operator(const double rho, double x) = 0;

  // debugging console
  void print_header(std::string title)
  {
    const int width = 80;
    const char sep  = ' ';

    Rcpp::Rcout << std::endl << std::string(width, '=') << std::endl;
    Rcpp::Rcout << std::string((width - title.length()) / 2, ' ') << title << std::endl;
    Rcpp::Rcout << std::string(width, '-') << std::endl;

    Rcpp::Rcout << std::left << std::setw(7)  << std::setfill(sep) << "iter";
    Rcpp::Rcout << std::left << std::setw(13) << std::setfill(sep) << "iteration gap";
    Rcpp::Rcout << std::left << std::setw(13) << std::setfill(sep) << "objective";
    Rcpp::Rcout << std::endl;

    Rcpp::Rcout << std::string(width, '-') << std::endl;
  }

  void print_row(int iter)
  {
    const char sep = ' ';

    Rcpp::Rcout << std::left << std::setw(7)  << std::setfill(sep) << iter;
    Rcpp::Rcout << std::left << std::setw(13) << std::setfill(sep) << iter_gap;
    Rcpp::Rcout << std::left << std::setw(13) << std::setfill(sep) << obj_val;
    Rcpp::Rcout << std::endl;
  }

  void print_foot()
  {
    const int width = 80;
    const char sep = ' ';
    Rcpp::Rcout << std::string(width, '=') << std::endl << std::endl;
    Rcpp::Rcout << std::left << std::setw(7)  << std::setfill(sep) << ' ';
    Rcpp::Rcout << std::left << std::setw(13) << std::setfill(sep) << eps_iter << std::endl;
    Rcpp::Rcout << std::left << std::setw(13) << std::setfill(sep) << eps_obj  << std::endl;
  }

public:
  PALMBase(int n_, double eps_abs_ = 1e-6, double eps_rel_ = 1e-6) :
    dim_main(n_),
    main_x(n_),
    aux_gamma(n_),
    grad_x(n_),
    eps_abs(eps_abs_),
    eps_obj(eps_rel_)
  {}

  virtual ~PALMBase() {}

  void update()
  {
    VecTypeX update_x(main_x);
    for(int ix = 0; ix < dim_main; ix++)
    {
      if(grad_x.coeff(ix) == 0) {
        continue;
      }
      next(update_x, ix);
    }
    iter_gap   = get_iter_gap(update_x);
    update_obj = obj(update_x);
    obj_gap    = get_obj_gap(update_obj);
    obj_val    = update_obj;
    main_x.swap(update_x);
  }

  bool converged()
  {
    return (obj_gap <= eps_obj);
  }

  int solve(int maxit)
  {
    eps_iter = get_eps_iter();
    eps_obj  = get_eps_obj();

    int i;
    // print_header("PALM iterations");
    for(i = 0; i < maxit; i++)
    {
      update();

      // print_row(i);

      if(converged())
      {
        break;
      }
    }

    iterator = i + 1;
    return iterator;
  }

  VecTypeX get_x()
  {
    return main_x;
  }
};


#endif // PALMBASE_H
