#pragma once

#include <ceres/ceres.h>

#include "geometry/xform.h"
#include "factors/shield.h"

namespace salsa
{

class ZeroVelFunctor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ZeroVelFunctor(const xform::Xformd& x0, const Eigen::Vector3d& v0, const Matrix7d& _Xi);

    template <typename T>
    bool operator() (const T* _x, const T* _v, T* _r) const;

    double yaw0_;
    Eigen::Vector3d p0_;
    Eigen::Vector3d v0_;
    Matrix7d Xi_;
};
typedef ceres::AutoDiffCostFunction<ZeroVelFunctor, 7, 7, 3> ZeroVelFactorAD;

class ZeroVelFactor : public ceres::SizedCostFunction<7, 7, 3>
{
    ZeroVelFactor(const ZeroVelFunctor* functor);
    bool Evaluate(const double* const* parameters, double* residuals, double ** jacobians) const;
    const ZeroVelFunctor* ptr;
};

}

