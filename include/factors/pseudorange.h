#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "gnss_utils/satellite.h"
#include "gnss_utils/wgs84.h"

#include "factors/shield.h"


namespace salsa
{

class PseudorangeFunctor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PseudorangeFunctor();
    void init(const gnss_utils::GTime& _t, const Eigen::Vector2d& _rho, gnss_utils::Satellite &sat,
              const Eigen::Vector3d& _rec_pos_ecef, const Eigen::Matrix2d& cov,
              int node, int idx);
    template <typename T>
    bool operator()(const T* _x, const T* _v, const T* _clk,
                    const T* _x_e2n, T* _res) const;

    bool active_ = false;
    int node_;
    int idx_;
    gnss_utils::GTime t;
    Eigen::Vector2d rho;
    Eigen::Vector3d sat_pos;
    Eigen::Vector3d sat_vel;
    Eigen::Vector2d sat_clk;
    double ion_delay;
    double trop_delay;
    double sagnac_comp;
    Eigen::Vector3d rec_pos;
    Eigen::Matrix2d Xi_;
    double sw;
};

typedef ceres::AutoDiffCostFunction<FunctorShield<PseudorangeFunctor>, 2, 7, 3, 2, 7> PseudorangeFactorAD;

}
