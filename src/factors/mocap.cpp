#include "factors/mocap.h"

MocapFunctor::MocapFunctor(double& dt_m, Xformd& x_u2m) :
    dt_m_{dt_m},
    x_u2m_{x_u2m}

{
    active_ = false;

}

void MocapFunctor::init(const Vector7d& _xm, const Vector6d& _xmdot, const Matrix6d& _P)
{
    Xi_ = _P.inverse().llt().matrixL().transpose();
    xmdot_ = _xmdot;
    xm_ = _xm;
    active_ = true;
}

template<typename T>
bool MocapFunctor::operator()(const T* _xu, T* _res) const
{
    typedef Matrix<T,6,1> Vec6;
    Map<Vec6> res(_res);
    Xform<T> xu(_xu);
    res = Xi_ * ((xm_ + (dt_m_ * xmdot_)) - (xu.template otimes<T,double>(x_u2m_)));
    // res = Xi_ * (x_ - x);
    return true;
}

template bool MocapFunctor::operator()<double>(const double* _xu, double* _res) const;
typedef ceres::Jet<double, 7> jactype;
template bool MocapFunctor::operator()<jactype>(const jactype* _xu, jactype* _res) const;

