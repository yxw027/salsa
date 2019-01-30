#pragma once

#include <ceres/ceres.h>
#include "geometry/xform.h"
#include "factors/shield.h"

using namespace Eigen;
using namespace xform;

struct MocapFunctor
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MocapFunctor()
  {
      active_ = false;
  }

  MocapFunctor(const Vector7d& _x, const Vector6d& _xdot, const Matrix6d& _P)
  {
      init(_x, _xdot, _P, 0);
  }

  void init(const Vector7d& _x, const Vector6d& _xdot, const Matrix6d& _P, int x_idx = -1)
  {
    Xi_ = _P.inverse().llt().matrixL().transpose();
    xdot_ = _xdot;
    x_ = _x;
    x_idx_ = x_idx;
    active_ = true;
  }

  template<typename T>
  bool operator()(const T* _x, const T* _toff, T* _res) const
  {
    typedef Matrix<T,6,1> Vec6;
    Map<Vec6> res(_res);
    Xform<T> x(_x);
    res = Xi_ * ((x_.boxplus<T,T>((*_toff) * xdot_)) - x);
//    res = Xi_ * (x_ - x);
    return true;
  }

  int x_idx_; // state index that this measurement goes with
  bool active_;
  Xformd x_;
  Vector6d xdot_;
  Matrix6d Xi_;
};



typedef ceres::AutoDiffCostFunction<FunctorShield<MocapFunctor>, 6, 7, 1> MocapFactorAD;
typedef ceres::AutoDiffCostFunction<MocapFunctor, 6, 7, 1> UnshiedledMocapFactorAD;
