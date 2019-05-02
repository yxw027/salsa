#pragma once

#include <ceres/ceres.h>
#include "geometry/xform.h"

struct XformPlus {
  template<typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const;
};
typedef ceres::AutoDiffLocalParameterization<XformPlus, 7, 6> XformParamAD;

//class XformParam : public ceres::LocalParameterization
//{
//public:
//  bool Plus(const double* x, const double* delta, double* x_plus_delta);
//  bool ComputeJacobian(const double* x, double* jacobian) const;
//  bool MultiplyByJacobian(const double* x,
//                          const int num_rows,
//                          const double* global_matrix,
//                          double* local_matrix) const;
//  int GlobalSize() const;
//  int LocalSize() const;

//};
