#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "factors/shield.h"

#include "salsa/misc.h"
#include "geometry/xform.h"

using namespace Eigen;
using namespace xform;

class ImuFunctor
{
public:
    enum
    {
        NRES = 9
    };
    typedef Quat<double> QuatT;
    typedef Xform<double> XformT;
    typedef Matrix<double, 3, 1> Vec3;
    typedef Matrix<double, 6, 1> Vec6;
    typedef Matrix<double, 9, 1> Vec9;
    typedef Matrix<double, 10, 1> Vec10;

    typedef Matrix<double, 6, 6> Mat6;
    typedef Matrix<double, 9, 9> Mat9;
    typedef Matrix<double, 9, 6> Mat96;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ImuFunctor();
    ImuFunctor(const double& _t0, const Vec6& b0);

    void reset(const double& _t0, const Vec6& b0, int from_idx=-1);
    void errorStateDynamics(const Vec10& y, const Vec9& dy, const Vec6& u, const Vec6& eta, Vec9& dydot);
    void dynamics(const Vec10& y, const Vec6& u, Vec9& ydot, Mat9& A, Mat96&B);
    void boxplus(const Vec10& y, const Vec9& dy, Vec10& yp);
    void boxminus(const Vec10& y1, const Vec10& y2, Vec9& d);
    void integrate(const double& _t, const Vec6& u, const Mat6& cov);
    void estimateXj(const double* _xi, const double* _vi, double* _xj, double* _vj) const;
    void finished();

    template<typename T>
    bool operator()(const T* _xi, const T* _xj, const T* _vi, const T* _vj, const T* _b, T* residuals) const;

    enum : int
    {
        ALPHA = 0,
        BETA = 3,
        GAMMA = 6,
    };

    enum :int
    {
        ACC = 0,
        OMEGA = 3
    };

    enum : int
    {
        P = 0,
        V = 3,
        Q = 6,
    };

    bool active_;
    double t0_;
    double delta_t_;
    Vec6 b_;
    int n_updates_;

    Mat9 P_;
    Mat9 Xi_;
    Vec10 y_;
    Vec6 u_;
    Mat6 cov_;

    Mat96 J_;
    Vec3 gravity_ = (Vec3() << 0, 0, 9.80665).finished();
};
typedef ceres::AutoDiffCostFunction<FunctorShield<ImuFunctor>, 9, 7, 7, 3, 3, 6> ImuFactorAD;
typedef ceres::AutoDiffCostFunction<ImuFunctor, 9, 7, 7, 3, 3, 6> UnshieldedImuFactorAD;


class ImuBiasDynamicsFunctor
{
public:
    ImuBiasDynamicsFunctor(const Vector6d& bias_prev, const Matrix6d& xi);
    void setBias(const Vector6d& bias_prev);
    template <typename T>
    bool operator() (const T* _b, T* _res) const;

    Vector6d bias_prev_;
    const Matrix6d Xi_;
};
typedef ceres::AutoDiffCostFunction<FunctorShield<ImuBiasDynamicsFunctor>, 6, 6> ImuBiasFactorAD;
typedef ceres::AutoDiffCostFunction<ImuBiasDynamicsFunctor, 6, 6> UnshieldedImuBiasFactorAD;
