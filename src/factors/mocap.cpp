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

//    res.setZero();
//    res.template topRows<3>() = Xi_.topLeftCorner<3,3>() * (xm_.t() - xu.t());
//    double yaw = xm_.q().yaw();
//    T yawhat = xu.q().yaw();
//    T diff = yaw - yawhat;
//    if (diff > M_PI)
//        diff -= (T)2.0*M_PI;
//    if (diff < -M_PI)
//        diff += (T)2.0*M_PI;
//    res(3) = Xi_(5,5) * diff;
    res = Xi_ * (xm_ - (xu.otimes(x_u2m_)));
//    res.topRows<3>().setZero();

    return true;
}

template bool MocapFunctor::operator()<double>(const double* _xu, double* _res) const;
typedef ceres::Jet<double, 7> jactype;
template bool MocapFunctor::operator()<jactype>(const jactype* _xu, jactype* _res) const;

}
