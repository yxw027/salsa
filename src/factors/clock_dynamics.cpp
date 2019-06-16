#include "factors/clock_dynamics.h"

using namespace Eigen;

namespace salsa {


ClockBiasFunctor::ClockBiasFunctor(const Matrix2d& Xi, int from_idx, int from_node)
{
    Xi_ = Xi;
    from_idx_ = from_idx;
    from_node_ = from_node;
    to_idx_ = -1;
}

bool ClockBiasFunctor::finished(double dt, int to_idx)
{
    dt_ = dt;
    to_idx_ = to_idx;
}

ClockBiasFunctor ClockBiasFunctor::split(double t)
{
    return ClockBiasFunctor(Xi_, from_idx_, from_node_);
}


template <typename T>
bool ClockBiasFunctor::operator()(const T* _taui, const T* _tauj, T* _res) const
{
    typedef Matrix<T,2,1> Vec2;

    Map<const Vec2> tau_i(_taui);
    Map<const Vec2> tau_j(_tauj);
    Map<Vec2> res(_res);

    res(0) = (tau_i(0) + tau_i(1) * (T)dt_) - tau_j(0);
    res(1) = (tau_i(1)) - tau_j(1);

    res = Xi_ * (T)1e9/dt_ * res;
    return true;
}
template bool ClockBiasFunctor::operator()<double>(const double* _taui, const double* _tauj, double* _res) const;
typedef ceres::Jet<double, 4> jactype;
template bool ClockBiasFunctor::operator()<jactype>(const jactype* _taui, const jactype* _tauj, jactype* _res) const;


ClockBiasFactor::ClockBiasFactor(ClockBiasFunctor *_ptr)
{
    ptr = _ptr;
}

bool ClockBiasFactor::Evaluate(const double * const *parameters, double *residuals, double **jacobians) const
{
    typedef Matrix<double,2,1> Vec2;

    Map<const Vec2> tau_i(parameters[0]);
    Map<const Vec2> tau_j(parameters[1]);
    Map<Vec2> res(residuals);

    res(0) = (tau_i(0) + tau_i(1) * ptr->dt_) - tau_j(0);
    res(1) = (tau_i(1)) - tau_j(1);

    res = ptr->Xi_ * 1e9/ptr->dt_ * res;

    if (jacobians)
    {
        if (jacobians[0])
        {
            Map<Matrix<double, 2, 2, RowMajor>> drdtaui(jacobians[0]);
            drdtaui << 1, ptr->dt_,
                       0, 1;
            drdtaui = ptr->Xi_ * 1e9 * drdtaui;
        }
        if (jacobians[1])
        {
            Map<Matrix<double, 2, 2, RowMajor>> drdtauj(jacobians[1]);
            drdtauj << -1,  0,
                        0, -1;
            drdtauj = ptr->Xi_ * 1e9 * drdtauj;
        }
    }
    return true;
}
}
