#pragma once

#include <ceres/ceres.h>
#include "geometry/xform.h"
#include "factors/shield.h"

using namespace Eigen;
using namespace xform;

struct MocapFunctor
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MocapFunctor(double& dt_m, Xformd& x_u2m);
  void init(const Vector7d& _xm, const Vector6d& _xmdot, const Matrix6d& _P);

  template<typename T>
  bool operator()(const T* _xu, T* _res) const;

  bool active_;
  Xformd xm_;
  Vector6d xmdot_;
  Matrix6d Xi_;
  double& dt_m_;
  Xformd& x_u2m_;
};



typedef ceres::AutoDiffCostFunction<FunctorShield<MocapFunctor>, 6, 7> MocapFactorAD;
typedef ceres::AutoDiffCostFunction<MocapFunctor, 6, 7> UnshiedledMocapFactorAD;
