#include "factors/feat.h"

using namespace Eigen;
using namespace xform;


namespace salsa
{

FeatFunctor::FeatFunctor(const Matrix2d& cov, const Vector3d &zetai,
                         const Vector3d &zetaj, int to_idx) :
  to_idx_(to_idx),
  zetai_(zetai),
  zetaj_(zetaj)
{
  Xi_ = cov.inverse().llt().matrixL().transpose();
  Pz_.block<1,3>(0,0) = zetaj_.cross(e_x).transpose().normalized();
  Pz_.block<1,3>(1,0) = zetaj_.cross(Pz_.block<1,3>(0,0).transpose()).transpose().normalized();
}

template <typename T>
bool FeatFunctor::operator ()(const T* _xi, const T* _xj, const T* _rho,
                              const T* _xb2c, T* _res) const
{
    typedef Matrix<T,2,1> Vec2;
    typedef Matrix<T,3,1> Vec3;
    Map<Vec2> res(_res);
    Xform<T> xi(_xi);
    Xform<T> xj(_xj);
    Xform<T> x_b2c(_xb2c);
    const T& rho(*_rho);
    Vec3 zi = 1.0/rho * zetai_;
    Vec3 zj_hat = x_b2c.rotp(xj.rotp(xi.transforma(x_b2c.transforma(zi)) - xj.transforma(x_b2c.t_)));
    zj_hat.normalize();


    res = Xi_ * Pz_ * (zetaj_ - zj_hat);
    return true;
}

template bool FeatFunctor::operator ()<double>(const double* _xi, const double* _xj,
                                               const double* _rho, const double* _xb2c,
                                               double* _res) const;
typedef ceres::Jet<double, 22> jactype;
template bool FeatFunctor::operator ()<jactype>(const jactype* _xi, const jactype* _xj,
                                                const jactype* _rho, const jactype* _xb2c,
                                                jactype* _res) const;

#define Tr transpose()
bool FeatFactor::Evaluate(double * const * parameters, double *residuals, double **jacobians) const
{
    Map<Vector2d> res(residuals);
    Xformd xi(parameters[0]);
    Xformd xj(parameters[1]);
    const double& rho(*parameters[2]);
    Xformd x_b2c(parameters[3]);
    Vector3d zi = 1.0/rho * zetai_;
    Vector3d zjhat = x_b2c.rotp(xj.rotp(xi.transforma(x_b2c.transforma(zi)) - xj.transforma(x_b2c.t_)));
    zjhat.normalize();
    res = Xi_ * Pz_ * (zetaj_ - zjhat);

    Matrix3d R_b2c = x_b2c.q().R();
    Matrix3d R_I2i = xi.q().R();
    Matrix3d R_I2j = xj.q().R();
    if (jacobians[0])
    {
        Map<Matrix<double, 2, 6>, RowMajor> dres_dxi(jacobians[0]);
        dres_dxi.block<2,3>(0, 0) = Pz_ * R_b2c * R_I2j;
        dres_dxi.block<2,3>(0, 3) = Pz_ * R_b2c * R_I2j * R_I2i.Tr * skew(x_b2c.transforma(zi));
    }

    if (jacobians[1])
    {
        Map<Matrix<double, 2, 6>, RowMajor> dres_dxj(jacobians[0]);
        dres_dxj.block<2,3>(0, 0) = -Pz_ * R_b2c * R_I2j;
//        dres_dxj.block<2,3>(0, 0) = -Pz_ * R_b2c * skew(R_I2j*(xi.t() + R_I2i.Tr*()));
    }


}
}
