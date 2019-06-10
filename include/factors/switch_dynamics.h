#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "factors/shield.h"

namespace salsa
{

class SwitchFunctor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SwitchFunctor(const double Xi);

    template <typename T>
    bool operator()(const T* _si, const T* _sj, T* _res) const;

    double Xi_;
};
typedef ceres::AutoDiffCostFunction<FunctorShield<SwitchFunctor>, 1, 1, 1> SwitchBiasFactorAD;


class SwitchFactor : public ceres::SizedCostFunction<1,1,1>
{
public:
    SwitchFactor(double Xi);

    bool Evaluate(const double * const *parameters, double *residuals, double **jacobians) const override;

    double Xi_;
};
}
