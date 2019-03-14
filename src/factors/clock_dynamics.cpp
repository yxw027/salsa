#include "factors/clock_dynamics.h"

ClockBiasFunctor::ClockBiasFunctor(const Matrix2d& Xi, double dt,
                                   int from_idx, int from_node, int to_idx)
{
    Xi_ = Xi;
    from_idx_ = from_idx;
    dt_ = dt;
    to_idx_ = to_idx;
    from_node_ = from_node;
    std::cout << "create clk bias from "<< from_node << " to " << to_idx << std::endl;
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

    res = Xi_ * (T)1e9 * res;
    return true;
}
template bool ClockBiasFunctor::operator()<double>(const double* _taui, const double* _tauj, double* _res) const;
typedef ceres::Jet<double, 4> jactype;
template bool ClockBiasFunctor::operator()<jactype>(const jactype* _taui, const jactype* _tauj, jactype* _res) const;



//ClockBiasPinFunctor::ClockBiasPinFunctor(const Vector2d &tau_prev, const Matrix2d &Xi)
//{
//    Xi_ = Xi;
//    tau_prev_ = tau_prev;
//}

//void ClockBiasPinFunctor::setTauPrev(const Vector2d& tau_prev)
//{
//    tau_prev_ = tau_prev;
//}

//template <typename T>
//bool ClockBiasPinFunctor::operator()(const T* _tau, T* _res) const
//{
//    typedef Matrix<T,2,1> Vec2;
//    Map<const Vec2> tau(_tau);
//    Map<Vec2> res(_res);

//    res = Xi_ * (tau - tau_prev_);
//    return true;
//}

//template bool ClockBiasPinFunctor::operator()<double>(const double* tau, double* res) const;
//typedef ceres::Jet<double, 2> jactype2;
//template bool ClockBiasPinFunctor::operator()<jactype2>(const jactype2* tau, jactype2* res) const;
