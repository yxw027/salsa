#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "multirotor_sim/satellite.h"
#include "multirotor_sim/wsg84.h"

#include "factors/shield.h"


using namespace Eigen;

class PseudorangeFunctor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PseudorangeFunctor();
    void init(const GTime& _t, const Vector2d& _rho, const Satellite& sat,
              const Vector3d& _rec_pos_ecef, const Matrix2d& cov,
              int node, int kf, int idx);
    template <typename T>
    bool operator()(const T* _x, const T* _v, const T* _clk,
                    const T* _x_e2n, T* _res) const;

    bool active_ = false;
    int node_;
    int kf_;
    int idx_;
    GTime t;
    Vector2d rho;
    Vector3d sat_pos;
    Vector3d sat_vel;
    Vector2d sat_clk_bias;
    double ion_delay;
    Vector3d rec_pos;
    Matrix2d Xi_;
    double sw;
};

typedef ceres::AutoDiffCostFunction<FunctorShield<PseudorangeFunctor>, 2, 7, 3, 2, 7> PseudorangeFactorAD;
