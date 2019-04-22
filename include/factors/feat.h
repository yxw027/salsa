#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "factors/shield.h"

#include "salsa/misc.h"
#include "geometry/xform.h"
#include "geometry/cam.h"

namespace salsa
{

class FeatFunctor
{
public:
    FeatFunctor(const xform::Xformd& x_b2c, const Eigen::Matrix2d& cov,
                const Eigen::Vector3d &zetai_, const Eigen::Vector3d &zetaj_,
                int to_idx);

    template<typename T>
    bool operator() (const T* _xi, const T* _xj, const T* _rho, T* _res) const;

    int to_idx_;
    const Eigen::Vector3d& zetai_;
    Eigen::Vector3d zetaj_;
    const xform::Xformd& x_b2c_;
    Eigen::Matrix2d Xi_;
    Eigen::Matrix<double, 2, 3> Pz_;

    double rho_true_;

    Eigen::Matrix3d R_b2c;
};


typedef ceres::AutoDiffCostFunction<FunctorShield<FeatFunctor>, 2, 7, 7, 1> FeatFactorAD;
typedef ceres::AutoDiffCostFunction<FeatFunctor, 2, 7, 7, 1> UnshieldedFeatFactorAD;

class FeatFactor : public FeatFunctor,  public ceres::CostFunction
{
public:
    using FeatFunctor::FeatFunctor;
    bool Evaluate(double * const * parameters, double *residuals, double **jacobians) const;
};

}
