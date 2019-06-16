#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "factors/shield.h"

namespace salsa
{

class ClockBiasFunctor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ClockBiasFunctor(const Eigen::Matrix2d& Xi, int from_idx, int from_node);
    bool finished(double dt, int to_idx);
    ClockBiasFunctor split(double t);

    template <typename T>
    bool operator()(const T* _taui, const T* _tauj, T* _res) const;

    int from_node_;
    int from_idx_;
    int to_idx_;
    double dt_;
    Eigen::Matrix2d Xi_;
};
typedef ceres::AutoDiffCostFunction<FunctorShield<ClockBiasFunctor>, 2, 2, 2> ClockBiasFactorAD;

class ClockBiasFactor : public ceres::SizedCostFunction<2,2,2>
{
public:
    ClockBiasFunctor* ptr;

    ClockBiasFactor(ClockBiasFunctor* _ptr);
    bool Evaluate(const double * const *parameters, double *residuals, double **jacobians) const override;
};

}
