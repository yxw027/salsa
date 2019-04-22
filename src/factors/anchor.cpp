#include "factors/anchor.h"

using namespace Eigen;
using namespace xform;

namespace salsa
{

AnchorFunctor::AnchorFunctor(const Matrix11d &Xi):
    Xi_(Xi)
{
    x_ = nullptr;
}

void AnchorFunctor::set(const salsa::State *x)
{
    x_ = x;
}


template<typename T>
bool AnchorFunctor::operator()(const T* _x, const T* _v, const T* _tau, T* _res) const
{
    Map<Matrix<T, 11, 1>> res(_res);
    Xform<T> x(_x);
    Map<const Matrix<T, 3, 1>>v (_v);
    Map<const Matrix<T, 2, 1>>tau (_tau);

    res.template segment<6>(0) = x_->x.boxminus(x);
    res.template segment<3>(6) = v - x_->v;
    res.template segment<2>(9) = 1e9*(tau - x_->tau);

    res = Xi_ * res;
    return true;
}

template bool AnchorFunctor::operator()<double>(const double* _x, const double* v, const double* _tau, double* _res) const;
typedef ceres::Jet<double, 12> jactype;
template bool AnchorFunctor::operator()<jactype>(const jactype* _x, const jactype* _v, const jactype* _tau, jactype* _res) const;

}
