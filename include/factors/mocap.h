#pragma once

#include <ceres/ceres.h>
#include "geometry/xform.h"
#include "factors/shield.h"

using namespace Eigen;
using namespace xform;

struct MocapFunctor
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MocapFunctor(const double &dt_m, const Xformd &x_u2m, const Vector7d& _xm,
               const Vector6d& _xmdot, const Matrix6d& _Xi, int idx, int node, int kf);

  template<typename T>
  bool operator()(const T* _xu, T* _res) const;

  int idx_;
  int node_;
  int kf_;
  Xformd xm_;
  Vector6d xmdot_;
  Matrix6d Xi_;
  const double& dt_m_;
  const Xformd& x_u2m_;
};



typedef ceres::AutoDiffCostFunction<FunctorShield<MocapFunctor>, 6, 7> MocapFactorAD;
typedef ceres::AutoDiffCostFunction<MocapFunctor, 6, 7> UnshiedledMocapFactorAD;
