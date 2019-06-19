#include "factors/zero_vel.h"

using namespace xform;
using namespace Eigen;

namespace salsa
{

ZeroVelFunctor::ZeroVelFunctor(const xform::Xformd& x0, const Eigen::Vector3d& v0,
                                       const Matrix7d& _Xi) :
    v0_(v0),
    Xi_(_Xi)
{
    p0_ = x0.t();
    yaw0_ =  x0.q().yaw();
}

template <typename T>
bool ZeroVelFunctor::operator()(const T* _x, const T* _v, T* _r) const
{
    typedef Matrix<T,3,1> Vec3;
    typedef Matrix<T,7,1> Vec7;
    Xform<T> x(_x);
    Map<const Vec3> v(_v);
    Map<Vec7> res(_r);

    res.template head<3>() = p0_ - x.t();
    res.template tail<3>() = v0_ - v;

    T delta_yaw = yaw0_ - x.q().yaw();
    // handle angle wrap in yaw error
    if (delta_yaw > (T)M_PI)
        delta_yaw -= (T)2.0*M_PI;
    else if (delta_yaw < -(T)M_PI)
        delta_yaw += (T)2.0*M_PI;
    res(3) = delta_yaw;

    res = Xi_ * res;
    return true;
}

template bool ZeroVelFunctor::operator()<double>(const double* _xj, const double* _vj, double* _res) const;
typedef ceres::Jet<double, 10> jactype;
template bool ZeroVelFunctor::operator()<jactype>(const jactype* _xj, const jactype* _vj, jactype* _res) const;

}
