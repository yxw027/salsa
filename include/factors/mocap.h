#pragma once

#include <ceres/ceres.h>
#include "geometry/xform.h"

using namespace Eigen;
using namespace xform;

struct MocapFunctor
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MocapFunctor(const Vector7d& _x, const Vector6d& _xdot, const Matrix6d& _P)
  {
    Xi_ = _P.inverse().llt().matrixL().transpose();
    xdot_ = _xdot;
    x_ = _x;
  }

  template<typename T>
  bool operator()(const T* _x, const T* _toff, T* _res) const
  {
    typedef Matrix<T,6,1> Vec6;
    Map<Vec6> res(_res);
    Xform<T> x(_x);
//    res = Xi_ * ((x_.boxplus<T,T>((*_toff) * xdot_)) - x);
    res = Xi_ * (x_ - x);
    return true;
  }

  Xformd x_;
  Vector6d xdot_;
  Matrix6d Xi_;
};
typedef ceres::AutoDiffCostFunction<MocapFunctor, 6, 7, 1> MocapFactorAD;
