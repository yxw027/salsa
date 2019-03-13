#pragma once

#include <ceres/ceres.h>
#include "geometry/xform.h"
#include "factors/shield.h"
#include "salsa/state.h"

typedef Matrix<double, 11, 11> Matrix11d;

struct AnchorFunctor
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  AnchorFunctor(const Matrix11d& Xi);
  void set(const salsa::State *x);

  template<typename T>
  bool operator()(const T* _x, const T* _v, const T* _tau, T* _res) const;

  const salsa::State* x_;
  const Matrix11d& Xi_;

};
typedef ceres::AutoDiffCostFunction<FunctorShield<AnchorFunctor>, 11, 7, 3, 2> AnchorFactorAD;
typedef ceres::AutoDiffCostFunction<AnchorFunctor, 11, 7, 3, 2> UnshiedledAnchorFactorAD;
