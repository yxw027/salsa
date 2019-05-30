#include "factors/anchor.h"

using namespace Eigen;
using namespace xform;

namespace salsa
{


XformAnchor::XformAnchor(const Matrix6d &Xi) :
    Xi_(Xi)
{
    x_ = xform::Xformd::Identity();
}

void XformAnchor::set(const Xformd& x)
{
    x_ = x;
    SALSA_ASSERT(std::abs(1.0 - x.q().arr_.norm()) < 1e-8, "Quat Left Manifold");
}

template <typename T>
bool XformAnchor::operator ()(const T* _x, T* _res) const
{
    Xform<T> x(_x);
    Map<Matrix<T, 6, 1>> res(_res);

    res = Xi_ * x_.boxminus(x);
    return true;
}
template bool XformAnchor::operator ()<double>(const double* _x, double* _res) const;
typedef ceres::Jet<double, 7> jt1;
template bool XformAnchor::operator ()<jt1>(const jt1* _x, jt1* _res) const;


StateAnchor::StateAnchor(const State::dxMat &Xi):
    Xi_(Xi)
{}

void StateAnchor::set(const salsa::State& x)
{
    x_ = x;
    SALSA_ASSERT(std::abs(1.0 - x.x.q().arr_.norm()) < 1e-8, "Quat Left Manifold");
}

template<typename T>
bool StateAnchor::operator()(const T* _x, const T* _v, const T* _tau, T* _res) const
{
    Map<Matrix<T, State::dxSize, 1>> res(_res);
    Xform<T> x(_x);
    Map<const Matrix<T, 3, 1>>v (_v);
    Map<const Matrix<T, 2, 1>>tau (_tau);

    res.template segment<6>(0) = x_.x.boxminus(x);
    res.template segment<3>(6) = v - x_.v;
    res.template segment<2>(9) = 1e9*(tau - x_.tau); // convert to ns to avoid scaling issues

    res = Xi_ * res;
    return true;
}

template bool StateAnchor::operator()<double>(const double* _x, const double* v, const double* _tau, double* _res) const;
typedef ceres::Jet<double, State::xSize> jactype;
template bool StateAnchor::operator()<jactype>(const jactype* _x, const jactype* _v, const jactype* _tau, jactype* _res) const;


ImuBiasAnchor::ImuBiasAnchor(const Vector6d& bias_prev, const Matrix6d& xi) :
    bias_prev_(bias_prev),
    Xi_(xi)
{}
void ImuBiasAnchor::setBias(const Vector6d& bias_prev)
{
    bias_prev_ = bias_prev;
}

template <typename T>
bool ImuBiasAnchor::operator()(const T* _b, T* _res) const
{
    typedef Matrix<T,6,1> Vec6;
    Map<const Vec6> b(_b);
    Map<Vec6> res(_res);
    res = Xi_ * (b - bias_prev_);
    return true;
}
template bool ImuBiasAnchor::operator ()<double>(const double* _b, double* _res) const;
typedef ceres::Jet<double, 6> jactype2;
template bool ImuBiasAnchor::operator ()<jactype2>(const jactype2* _b, jactype2* _res) const;

}
