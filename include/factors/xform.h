#pragma once

#include <ceres/ceres.h>
#include "geometry/xform.h"

using namespace Eigen;
using namespace xform;

struct XformPlus {
  template<typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const
  {
    Xform<T> q(x);
    Map<const Matrix<T,6,1>> d(delta);
    Map<Matrix<T,7,1>> qp(x_plus_delta);
    qp = (q + d).elements();
    return true;
  }
};
typedef ceres::AutoDiffLocalParameterization<XformPlus, 7, 6> XformParamAD;
