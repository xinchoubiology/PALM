#ifndef PALMSTLS_H
#define PALMSTLS_H

#include "./PALMBase.h"

/*
// argmin 1/2 ||y - X \beta||_2^2 / (1 + ||\beta||_2^2) + \lambda [(1 - \alpha) ||\beta||_2^2 / 2 + \alpha ||\beta||_1^1]
// s.t. \alpha \in (0, 1]
*/
class PALMSTls: public PALMBase<Eigen::SparseVector<float>, Eigen::VectorXf>
{
protected:
    typedef float Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map<const Vector> MapVec;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;
    typedef Eigen::SparseVector<Scalar> SpVector;

    MapMat datX;                         // data X observation matrix
    MapVec datY;                         // data Y response vector
    Scalar lambda;                       // Elastic Net Penalty Parameter
    Scalar lambda0;                      // minimal-maximize lambda value for \beta => 0 Vector
    Scalar alpha;                        // alpha for Elastic Net

    // inter-mediate parameters
    Vector Xi;                           // i-th column of datX
    Scalar bi;                           // i-th current updated coeffienct \beta_i
    Vector Res;                          // full residual y - X \beta
    Vector Resi;                         // i-th coeffienct's partial residual y - X \beta + Xi * \beta_i
    Scalar coef;                         // i-th coeffienct's partial denominator 1 + ||\beta||_2^2 - bi^2
    Scalar XRi;                          // Xi.dot(Resi)
    Scalar RRi;                          // Resi.dot(Resi)
    Vector XXi;                          // cache for Xi.dot(Xi)

    double Lips;                         // Lipschitz modulus for current updated coefficient
    double subGrad;                      // Subgradient value of current updated coefficient's H(X) 1/2 ||y - X \beta||_2^2 / (1 + ||\beta||_2^2) + \lambda * (1 - \alpha) ||\beta||_2^2

    double obj(const SpVector &x)
    {
        return 1/2 * Res.squaredNorm() / (1 + x.squaredNorm()) + lambda * ((1 - alpha) * x.squaredNorm() / 2 + alpha * x.cwiseAbs().sum());
    }

    double get_lipschitz(int offset)
    {
        //return 4 * RRi * std::pow(coef, -2) + 13/4 * std::abs(XRi) * std::pow(coef, -3/2) + 4 * XXi(offset) * std::pow(coef, -1) + lambda * (1 - alpha);
        return 4 * std::abs(XXi(offset) * coef - RRi) * std::pow(coef, -2) + 13/4 * std::abs(XRi) * std::pow(coef, -3/2) + lambda * (1 - alpha);
    }

    double get_subgrad(const SpVector &update_x, int offset)
    {
        return (XRi * (std::pow(bi, 2) - coef) + (XXi(offset) * coef - RRi) * bi) / std::pow(1 + update_x.squaredNorm(), 2) + lambda * (1 - alpha) * bi;
    }

    // argmin lambda * alpha * |u| + rho / 2 * ||u - x||_2^2
    double prox_operator(const double rho, double x)
    {
      return fmax(0, x - 1 / rho) - fmax(0, -x - 1 / rho);
    }

    // PALM alternative proximal update
    // argmin lambda * alpha * |u| + c_k / 2||u - (u_{k-1} - 1/c_k * \nabla H(X_{k-1}))||_2^2
    void next(SpVector &update_x, int offset)
    {
      if(lambda > lambda0 - 1e-5) {
        return;
      }

      // varaible assignment
      Xi          = datX.col(offset);
      bi          = update_x.coeff(offset);
      Resi        = Res + Xi * bi;
      coef        = 1 + update_x.squaredNorm() - std::pow(bi, 2);
      XRi         = Xi.dot(Resi);
      RRi         = Resi.dot(Resi);

      Lips        = aux_gamma(offset) * get_lipschitz(offset);
      subGrad     = get_subgrad(update_x, offset);

      update_x.coeffRef(offset)  = prox_operator(Lips / (lambda * alpha), bi - subGrad / Lips);
      Res         = Resi - Xi * update_x.coeff(offset);
    }

    // sequential strong rule build
    void ssr(Eigen::VectorXf &update_g, double lambda_prev, double lambda)
    {
        for(int ix = 0; ix < dim_main; ix++)
        {
            Xi      = datX.col(ix);
            bi      = main_x.coeff(ix);
            Resi    = Res + Xi * bi;
            coef    = 1 + main_x.squaredNorm() - std::pow(bi, 2);
            XRi     = Xi.dot(Resi);
            RRi     = Resi.dot(Resi);
            subGrad = get_subgrad(main_x, ix);
            update_g.coeffRef(ix) = (subGrad <= alpha * (2 * lambda - lambda_prev)) ? 0 : 1;
        }
    }

public:
    PALMSTls(ConstGenericMatrix &datX_, ConstGenericVector &datY_, double eps_abs_ = 1e-6, double eps_rel_ = 1e-6, double alpha_ = 1) :
        PALMBase<Eigen::SparseVector<float>, Eigen::VectorXf>(datX_.cols(), eps_abs_, eps_rel_),
        datX(datX_.data(), datX_.rows(), datX_.cols()),
        datY(datY_.data(), datY_.rows(), datY_.cols()),
        alpha(std::max(alpha_, 0.001))
    {
        lambda0 = (datX.transpose() * datY).cwiseAbs().maxCoeff() / alpha;
        lambda  = lambda0;
        main_x.setZero();
        aux_gamma.setOnes();
        XXi     = datX.colwise().squaredNorm();
    }

    double get_lambda0() const
    {
        return lambda0;
    }

    /* warm start in lambda search
    */
    void init(double lambda_, bool warm_start = true)
    {
        if(warm_start == false)
        {
            main_x.setZero();
            Res   = datY - datX * main_x;
        }

        if(lambda_ > lambda0 - 1e-5) {
            grad_x.setOnes();
        } else {
            ssr(grad_x, lambda, lambda_);
        }
        
        lambda    = lambda_;
        iter_gap  = 1e6;
        obj_gap   = 1e6;
        iterator  = 0;
        obj_val   = obj(main_x);
    }
};
#endif // PALMSTLS_H
