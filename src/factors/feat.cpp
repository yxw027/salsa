#include "factors/feat.h"

FeatFunctor::FeatFunctor(const Xformd& x_b2c, const Matrix2d& cov,
                         const Vector3d &zetai, const Vector3d &zetaj, int to_idx) :
  to_idx_(to_idx),
  zetai_(zetai),
  zetaj_(zetaj),
  x_b2c_(x_b2c)
{
  Xi_ = cov.inverse().llt().matrixL().transpose();
  Pz_.block<1,3>(0,0) = zetaj_.cross(e_x).transpose().normalized();
  Pz_.block<1,3>(1,0) = zetaj_.cross(Pz_.block<1,3>(0,0).transpose()).transpose().normalized();
  R_b2c = x_b2c_.q().R();
}

template <typename T>
bool FeatFunctor::operator ()(const T* _xi, const T* _xj, const T* _rho, T* _res) const
{
    typedef Matrix<T,2,1> Vec2;
    typedef Matrix<T,3,1> Vec3;
    Map<Vec2> res(_res);
    Xform<T> xi(_xi);
    Xform<T> xj(_xj);
    const T& rho(*_rho);
    Vec3 zi = 1.0/rho * zetai_;
//    Vec3 zj_hat = x_b2c_.rotp<T>(xj.rotp(xi.t_ + xi.rota(x_b2c_.rota<T>(zi) + x_b2c_.t_) - (xj.rota(x_b2c_.t_) + xj.t_)));
    Vec3 zj_hat = x_b2c_.rotp<T>(xj.rotp(xi.transforma(x_b2c_.transforma<T>(zi)) - xj.transforma(x_b2c_.t_)));
    zj_hat.normalize();


    res = Xi_ * Pz_ * (zetaj_ - zj_hat);
    return true;
}

template bool FeatFunctor::operator ()<double>(const double* _xi, const double* _xj, const double* _rho, double* _res) const;
typedef ceres::Jet<double, 15> jactype;
template bool FeatFunctor::operator ()<jactype>(const jactype* _xi, const jactype* _xj, const jactype* _rho, jactype* _res) const;
