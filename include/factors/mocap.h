#pragma once

#include <ceres/ceres.h>
#include "geometry/xform.h"
#include "factors/shield.h"

using namespace Eigen;
using namespace xform;

struct MocapFunctor
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MocapFunctor(double& dt_m, Xformd& x_u2m) :
    dt_m_{dt_m},
    x_u2m_{x_u2m}

  {
    active_ = false;

  }

  void init(const Vector7d& _xm, const Vector6d& _xmdot, const Matrix6d& _P, int x_idx = -1)
  {
    Xi_ = _P.inverse().llt().matrixL().transpose();
    xmdot_ = _xmdot;
    xm_ = _xm;
    x_idx_ = x_idx;
    active_ = true;
  }

  template<typename T>
  bool operator()(const T* _xu, T* _res) const
  {
    typedef Matrix<T,6,1> Vec6;
    Map<Vec6> res(_res);
    Xform<T> xu(_xu);
    res = Xi_ * ((xm_ + (dt_m_ * xmdot_)) - (xu.template otimes<T,double>(x_u2m_)));
    // res = Xi_ * (x_ - x);
    return true;
  }

  int x_idx_; // state index that this measurement goes with
  bool active_;
  Xformd xm_;
  Vector6d xmdot_;
  Matrix6d Xi_;
  double& dt_m_;
  Xformd& x_u2m_;
};



typedef ceres::AutoDiffCostFunction<FunctorShield<MocapFunctor>, 6, 7> MocapFactorAD;
typedef ceres::AutoDiffCostFunction<MocapFunctor, 6, 7> UnshiedledMocapFactorAD;
