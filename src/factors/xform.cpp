#include <factors/xform.h>

using namespace Eigen;
using namespace xform;

template<typename T>
bool XformPlus::operator()(const T* x, const T* delta, T* x_plus_delta) const
{
	Xform<T> q(x);
	Map<const Matrix<T,6,1>> d(delta);
	Map<Matrix<T,7,1>> qp(x_plus_delta);
	qp = (q + d).elements();
	return true;
}
template bool XformPlus::operator()<double>(const double* x, const double* delta, double* x_plus_delta) const;
typedef ceres::Jet<double, 13> jactype;
template bool XformPlus::operator()<jactype>(const jactype* x, const jactype* delta, jactype* x_plus_delta) const;
