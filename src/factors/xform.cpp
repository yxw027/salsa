#include <factors/xform.h>
#include <salsa/misc.h>

using namespace Eigen;
using namespace xform;


bool XformParam::Plus(const double* _x, const double* delta, double* x_plus_delta) const
{
    Xformd x(_x);
    Map<const Vector6d> d(delta);
    Xformd xp(x_plus_delta);
    xp = x + d;
    SALSA_ASSERT( abs(1 - xp.q().arr_.norm()) < 1e-8 , "Quaternion Left Manifold");
    return true;
}

bool XformParam::ComputeJacobian(const double* _x, double* jacobian) const
{
    Map<Matrix<double, 7, 6, RowMajor>> J(jacobian);
    Xformd x(_x);
    J.topLeftCorner<3,3>() = x.q().R().transpose();
    J.topRightCorner<3,3>().setZero();
    J.bottomLeftCorner<4,3>().setZero();
    J.bottomRightCorner<4,3>() << -_x[4]/2.0, -_x[5]/2.0, -_x[6]/2.0,
                                   _x[3]/2.0, -_x[6]/2.0,  _x[5]/2.0,
                                   _x[6]/2.0,  _x[3]/2.0, -_x[4]/2.0,
                                  -_x[5]/2.0,  _x[4]/2.0,  _x[3]/2.0;
    return true;
}
int XformParam::GlobalSize() const
{
    return 7;
}
int XformParam::LocalSize() const
{
    return 6;
}

template<typename T>
bool XformPlus::operator()(const T* x, const T* delta, T* x_plus_delta) const
{
	Xform<T> q(x);
	Map<const Matrix<T,6,1>> d(delta);
    Xform<T> xp(x_plus_delta);
    xp = q + d;
    SALSA_ASSERT( abs((T)1.0 - xp.q().arr_.norm()) < 1e-8 , "Quaternion Left Manifold");
	return true;
}
template bool XformPlus::operator()<double>(const double* x, const double* delta, double* x_plus_delta) const;
typedef ceres::Jet<double, 13> jactype;
template bool XformPlus::operator()<jactype>(const jactype* x, const jactype* delta, jactype* x_plus_delta) const;

