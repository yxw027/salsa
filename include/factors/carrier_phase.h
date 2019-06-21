#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "gnss_utils/satellite.h"
#include "gnss_utils/wgs84.h"

#include "factors/shield.h"

namespace salsa
{

class CarrierPhaseFunctor
{
public:
    CarrierPhaseFunctor(const double& _Xi, const double& _Phi0, double Phi1,
                        gnss_utils::Satellite& sat, const Eigen::Vector3d& p_b2g,
                        const gnss_utils::GTime g0, const gnss_utils::GTime g1,
                        int node, int from_idx, int to_idx);

    const double& Phi0_;
    const double Phi1_;
    double dPhi_bar_;
    const Eigen::Vector3d& p_b2g_;
    double Xi_;
    const double lambda_;
    int node_;
    int from_idx_;
    int to_idx_;
    gnss_utils::GTime g0_;
    gnss_utils::GTime g1_;

    Eigen::Vector3d sat_pos0_;
    Eigen::Vector3d sat_vel0_;
    Eigen::Vector2d sat_clk0_;

    Eigen::Vector3d sat_pos1_;
    Eigen::Vector3d sat_vel1_;
    Eigen::Vector2d sat_clk1_;

    template<typename T>
    bool operator()(const T* _x0, const T*_x1, const T* _clk0, const T* _clk1, const T* _x_e2n, T* _res) const;

};
typedef ceres::AutoDiffCostFunction<FunctorShield<CarrierPhaseFunctor>, 1, 7, 7, 2, 2, 7> CarrierPhaseFactorAD;


class CarrierPhaseFactor : public ceres::SizedCostFunction<1,7,7,2,2,7>
{
public:
    CarrierPhaseFactor(const CarrierPhaseFunctor* functor);
    bool Evaluate(const double * const *parameters, double *residuals, double **jacobians) const;
    const CarrierPhaseFunctor* ptr;
};


}
