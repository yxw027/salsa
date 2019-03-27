#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "factors/shield.h"

using namespace Eigen;

class ClockBiasFunctor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ClockBiasFunctor(const Matrix2d& Xi, int from_idx, int from_node);
    bool finished(double dt, int to_idx);

    template <typename T>
    bool operator()(const T* _taui, const T* _tauj, T* _res) const;

    int from_node_;
    int from_idx_;
    int to_idx_;
    double dt_;
    Matrix2d Xi_;
};
typedef ceres::AutoDiffCostFunction<FunctorShield<ClockBiasFunctor>, 2, 2, 2> ClockBiasFactorAD;


//class ClockBiasPinFunctor
//{
//public:
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//    ClockBiasPinFunctor(const Vector2d& tau_prev, const Matrix2d& Xi);
//    void setTauPrev(const Vector2d& tau_prev);

//    template <typename T>
//    bool operator()(const T* _tau, T* res) const;

//    Vector2d tau_prev_;
//    Matrix2d Xi_;
//};
//typedef ceres::AutoDiffCostFunction<FunctorShield<ClockBiasPinFunctor>, 2, 2> ClockBiasPinFunctorAD;
