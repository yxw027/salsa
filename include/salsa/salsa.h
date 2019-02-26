#pragma once

#include <memory>

#include <Eigen/Core>
#include "geometry/xform.h"
#include "multirotor_sim/estimator_base.h"
#include "multirotor_sim/satellite.h"
#include "multirotor_sim/wsg84.h"
#include "multirotor_sim/utils.h"

#include "factors/imu.h"
#include "factors/mocap.h"
#include "factors/xform.h"
#include "factors/pseudorange.h"
#include "factors/clock_dynamics.h"
#include "factors/carrier_phase.h"
#include "factors/clock_dynamics.h"

#include "salsa/logger.h"

using namespace std;
using namespace Eigen;
using namespace xform;

#ifndef SALSA_WINDOW_SIZE
#define SALSA_WINDOW_SIZE 10
#endif

#ifndef SALSA_NUM_FEATURES
#define SALSA_NUM_FEATURES 120
#endif

#ifndef SALSA_NUM_SATELLITES
#define SALSA_NUM_SATELLITES 20
#endif

class MTLogger;
class Logger;
class Salsa : public multirotor_sim::EstimatorBase
{
public:

  enum
  {
    N = SALSA_WINDOW_SIZE,
    N_FEAT = SALSA_NUM_FEATURES,
    N_SAT = SALSA_NUM_SATELLITES,

  };


  Salsa();

  void init(const std::string& filename);

  void load(const std::string& filename);
  void initState();
  void initFactors();
  void initialize(const double& t, const Xformd &x0, const Vector3d& v0, const Vector2d& tau0);
  void initSolverOptions();
  void initLog();

  void finishNode(const double& t);

  void solve();
  void addResidualBlocks(ceres::Problem& problem);
  void addImuFactors(ceres::Problem& problem);
  void addMocapFactors(ceres::Problem& problem);
  void addRawGnssFactors(ceres::Problem& problem);

  void renderGraph(const std::string& filename);

  void pointPositioning(const GTime& t, const VecVec3& z, std::vector<Satellite>& sat, Vector3d &xhat) const;

  void imuCallback(const double &t, const Vector6d &z, const Matrix6d &R) override;
  void mocapCallback(const double &t, const Xformd &z, const Matrix6d &R) override;
  void rawGnssCallback(const GTime& t, const VecVec3& z, const VecMat3& R, std::vector<Satellite>& sat, const std::vector<bool>& slip) override;

  double current_t_;
  Xformd current_x_;
  Vector3d current_v_;

  bool initialized_[N];
  Matrix<double, N, 1> t_;
  Matrix<double, 7, N> x_; int x_idx_;
  Matrix<double, 3, N> v_;
  Matrix<double, 2, N> tau_;
  Matrix<double, N_SAT, 1> s_;
  Vector6d imu_bias_;
  int current_node_;

  double switch_weight_;
  double acc_wander_weight_;
  double gyro_wander_weight_;
  Matrix6d acc_bias_xi_;

  std::vector<ImuFunctor> imu_;
  ImuBiasDynamicsFunctor* bias_;
  std::vector<ClockBiasFunctor> clk_;
  std::vector<MocapFunctor> mocap_;
  std::vector<std::vector<PseudorangeFunctor>> prange_;
  std::vector<ClockBiasFunctor> clock_bias_;

  ceres::Solver::Options options_;
  ceres::Solver::Summary summary_;

  Logger* state_log_ = nullptr;
  Logger* opt_log_ = nullptr;

  std::string log_prefix_;
  Xformd x_u2m_; // transform from imu to mocap frame
  Xformd x_u2b_; // transform from imu to body frame
  Xformd x_u2c_; // transform from imu to camera frame
  Xformd x_e2n_; // transform from ECEF to NED (inertial) frame
  double dt_m_; // time offset of mocap  (t(stamped) - dt_m = t(true))
  double dt_c_; // time offset of camera (t(stamped) - dt_m = t(true))
  GTime start_time_;
  Matrix2d clk_bias_R_;


};
