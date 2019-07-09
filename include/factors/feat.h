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
    FeatFunctor(double t, const Eigen::Matrix2d& cov, const xform::Xformd& x_b2c,
                const Eigen::Vector3d &zetai_, const Eigen::Vector3d &zetaj_,
                const Eigen::Vector2d &pixj, int to_idx);

    template<typename T>
    bool operator() (const T* _xi, const T* _xj, const T* _rho, T* _res) const;

    double t_;
    int to_idx_;
    const Eigen::Vector3d& zetai_;
    Eigen::Vector3d zetaj_;
    Eigen::Matrix2d Xi_;
    Eigen::Vector2d pixj_;
    Eigen::Matrix<double, 2, 3> Pz_;
    const xform::Xformd& x_b2c_;

    double rho_true_;
};


typedef ceres::AutoDiffCostFunction<FunctorShield<FeatFunctor>, 2, 7, 7, 1> FeatFactorAD;
typedef ceres::AutoDiffCostFunction<FeatFunctor, 2, 7, 7, 1> UnshieldedFeatFactorAD;

class FeatFactor : public ceres::SizedCostFunction<2, 7, 7, 1>
{
public:
    FeatFactor(const FeatFunctor* functor);
    bool Evaluate(const double * const * parameters, double *residuals, double **jacobians) const;
    const FeatFunctor* ptr;
};

}
