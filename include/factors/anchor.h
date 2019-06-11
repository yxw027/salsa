#pragma once

#include <ceres/ceres.h>
#include "geometry/xform.h"
#include "factors/shield.h"
#include "salsa/state.h"
#include "salsa/misc.h"

namespace salsa
{

struct XformAnchor
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    XformAnchor(const Matrix6d& Xi);
    void set(const xform::Xformd &x);

    template <typename T>
    bool operator()(const T* _x, T* _res) const;

    xform::Xformd x_;
    const Matrix6d& Xi_;
};
typedef ceres::AutoDiffCostFunction<FunctorShield<XformAnchor>, 6, 7> XformAnchorFactorAD;

struct StateAnchor
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StateAnchor(const State::dxMat& Xi);
  void set(const State &x);

  template<typename T>
  bool operator()(const T* _x, const T* _v, const T* _tau, T* _res) const;

  salsa::State x_;
  const State::dxMat& Xi_;

};
typedef ceres::AutoDiffCostFunction<FunctorShield<StateAnchor>, 11, 7, 3, 2> StateAnchorFactorAD;


class ImuBiasAnchor
{
public:
    ImuBiasAnchor(const Vector6d& bias_prev, const Matrix6d& xi);
    void setBias(const Vector6d& bias_prev);
    template <typename T>
    bool operator() (const T* _b, T* _res) const;

    Vector6d bias_prev_;
    const Matrix6d Xi_;
};
typedef ceres::AutoDiffCostFunction<FunctorShield<ImuBiasAnchor>, 6, 6> ImuBiasAnchorFactorAD;

}
