#pragma once

#include <ceres/ceres.h>
#include "geometry/xform.h"

struct XformPlus {
  template<typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const;
};
typedef ceres::AutoDiffLocalParameterization<XformPlus, 7, 6> XformParamAD;

class XformParam : public ceres::LocalParameterization
{
public:
  bool Plus(const double* _x, const double* delta, double* x_plus_delta) const override;
  bool ComputeJacobian(const double* _x, double* jacobian) const override;
  int GlobalSize() const override;
  int LocalSize() const override;
};
