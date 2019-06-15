#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "factors/shield.h"

#include "salsa/misc.h"
#include "salsa/meas.h"
#include "geometry/xform.h"

namespace salsa
{

class ImuIntegrator
{
public:
    ImuIntegrator();
    void reset(const double& _t);
    void dynamics(const Vector10d& y, const Vector6d& u, Vector9d& ydot);
    void estimateXj(const double* _xi, const double* _vi, double* _xj, double* _vj) const;
    void integrateStateOnly(const double& _t, const Vector6d& u);
    static void boxplus(const Vector10d& y, const Vector9d& dy, Vector10d& yp);
    static void boxminus(const Vector10d& y1, const Vector10d& y2, Vector9d& d);

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

    double t;
    double delta_t_;
    double t0_;
    Vector6d b_;
    Vector10d y_;
    static Eigen::Vector3d gravity_;
};

class ImuFunctor : public ImuIntegrator
{
public:
    typedef Eigen::Matrix<double, 9, 6> Matrix96;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ImuFunctor(const double& _t0, const Vector6d& b0, int from_idx, int from_node);


    void errorStateDynamics(const Vector10d& y, const Vector9d& dy,
                            const Vector6d& u, const Vector6d& eta, Vector9d& dydot);
    void dynamics(const Vector10d& y, const Vector6d& u, Vector9d& ydot, Matrix9d& A, Matrix96&B);
    void integrate(const double& _t, const Vector6d& u, const Matrix6d& cov, bool save=true);
    void integrate(const meas::Imu& z, bool save=true);
    void finished(int to_idx);
    ImuFunctor split(double t);

    template<typename T>
    bool operator()(const T* _xi, const T* _xj, const T* _vi, const T* _vj,
                    const T* _b, T* residuals) const;

    int from_idx_;
    int to_idx_;
    int from_node_;

    Vector6d u_;
    Matrix6d cov_;
    int n_updates_;

    Matrix9d P_;
    Matrix9d Xi_;

    Matrix96 J_;

    std::deque<meas::Imu, Eigen::aligned_allocator<meas::Imu>> meas_hist_;
};
typedef ceres::AutoDiffCostFunction<FunctorShield<ImuFunctor>, 9, 7, 7, 3, 3, 6> ImuFactorAD;
typedef ceres::AutoDiffCostFunction<ImuFunctor, 9, 7, 7, 3, 3, 6> UnshieldedImuFactorAD;

}
