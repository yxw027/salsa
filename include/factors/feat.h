#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "factors/shield.h"

#include "salsa/misc.h"
#include "geometry/xform.h"
#include "geometry/cam.h"

using namespace Eigen;
using namespace xform;

class FeatFunctor
{
public:
    FeatFunctor(const Xformd& x_b2c, const Matrix2d& cov,
                const Vector3d &zetai_, const Vector3d &zetaj_,
                int to_idx);

    template<typename T>
    bool operator() (const T* _xi, const T* _xj, const T* _rho, T* _res) const;

    int to_idx_;
    const Vector3d& zetai_;
    Vector3d zetaj_;
    const Xformd& x_b2c_;
    Matrix2d Xi_;
    Matrix<double, 2, 3> Pz_;

    double rho_true_;

    Matrix3d R_b2c;
};


typedef ceres::AutoDiffCostFunction<FunctorShield<FeatFunctor>, 2, 7, 7, 1> FeatFactorAD;
typedef ceres::AutoDiffCostFunction<FeatFunctor, 2, 7, 7, 1> UnshieldedFeatFactorAD;

class FeatFactor : public FeatFunctor,  public ceres::CostFunction
{
public:
    using FeatFunctor::FeatFunctor;
    bool Evaluate(double * const * parameters, double *residuals, double **jacobians) const;
};
