#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "factors/shield.h"

#include "salsa/misc.h"
#include "geometry/xform.h"

namespace salsa
{

class ImuFunctor
{
public:
    typedef Eigen::Matrix<double, 9, 6> Matrix96;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ImuFunctor(const double& _t0, const Vector6d& b0, int from_idx, int from_node);

    void errorStateDynamics(const Vector10d& y, const Vector9d& dy,
                            const Vector6d& u, const Vector6d& eta, Vector9d& dydot);
    void dynamics(const Vector10d& y, const Vector6d& u, Vector9d& ydot, Matrix9d& A, Matrix96&B);
    static void boxplus(const Vector10d& y, const Vector9d& dy, Vector10d& yp);
    static void boxminus(const Vector10d& y1, const Vector10d& y2, Vector9d& d);
    void integrate(const double& _t, const Vector6d& u, const Matrix6d& cov);
    void estimateXj(const double* _xi, const double* _vi, double* _xj, double* _vj) const;
    void finished(int to_idx);

    template<typename T>
    bool operator()(const T* _xi, const T* _xj, const T* _vi, const T* _vj,
                    const T* _b, T* residuals) const;

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

    int from_idx_;
    int to_idx_;
    int from_node_;

    double t0_;
    double delta_t_;
    Vector6d b_;
    int n_updates_;

    Matrix9d P_;
    Matrix9d Xi_;
    Vector10d y_;
    Vector6d u_;
    Matrix6d cov_;

    Matrix96 J_;
    static Eigen::Vector3d gravity_;
};
typedef ceres::AutoDiffCostFunction<FunctorShield<ImuFunctor>, 9, 7, 7, 3, 3, 6> ImuFactorAD;
typedef ceres::AutoDiffCostFunction<ImuFunctor, 9, 7, 7, 3, 3, 6> UnshieldedImuFactorAD;

}
