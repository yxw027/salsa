#include "factors/mocap.h"

using namespace Eigen;
using namespace xform;

namespace salsa
{

MocapFunctor::MocapFunctor(const double& dt_m, const Xformd& x_u2m, const Vector7d& _xm,
                           const Vector6d& _xmdot, const Matrix6d& _Xi, int idx, int node, int kf) :
    dt_m_(dt_m),
    x_u2m_{x_u2m},
    idx_{idx},
    node_{node},
    kf_{kf}
{
    xmdot_ = _xmdot;
    xm_ = _xm;
    Xi_ = _Xi;
}

template<typename T>
bool MocapFunctor::operator()(const T* _xu, T* _res) const
{
    typedef Matrix<T,6,1> Vec6;
    Map<Vec6> res(_res);
    Xform<T> xu(_xu);

    res = Xi_ * (xm_ - (xu.otimes(x_u2m_)));

    return true;
}

template bool MocapFunctor::operator()<double>(const double* _xu, double* _res) const;
typedef ceres::Jet<double, 7> jactype;
template bool MocapFunctor::operator()<jactype>(const jactype* _xu, jactype* _res) const;

}
