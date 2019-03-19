 #pragma once

#include <memory>
#include <deque>
#include <experimental/filesystem>

#include <Eigen/Core>
#include <Eigen/StdVector>

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
#include "factors/anchor.h"
#include "factors/feat.h"

#include "salsa/logger.h"
#include "salsa/state.h"

using namespace std;
using namespace Eigen;
using namespace xform;
using multirotor_sim::VecMat3;
using multirotor_sim::VecVec3;
using multirotor_sim::ImageFeat;

#define STATE_BUF_SIZE 50

namespace salsa
{

typedef Matrix<double, 11, 1> Vector11d;
typedef std::deque<MocapFunctor, aligned_allocator<MocapFunctor>> MocapDeque;
typedef std::vector<PseudorangeFunctor, aligned_allocator<PseudorangeFunctor>> PseudorangeVec;
typedef std::deque<PseudorangeVec, aligned_allocator<PseudorangeVec>> PseudorangeDeque;
typedef std::deque<ImuFunctor, aligned_allocator<ImuFunctor>> ImuDeque;
typedef std::deque<ClockBiasFunctor, aligned_allocator<ClockBiasFunctor>> ClockBiasDeque;
typedef std::deque<FeatFunctor, aligned_allocator<FeatFunctor>> FeatDeque;
struct Feat
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int idx0;
    int node0;
    int kf0;
    Vector3d z;
    double rho;
    Feat(int _idx, int _node, int _kf0, const Vector3d& _z, const double& _rho) :
        idx0(_idx), node0(_node), kf0(_kf0), z(_z), rho(_rho){}
};
typedef std::map<int, Feat, std::less<int>,
                 aligned_allocator<std::pair<const int, Feat>>> FeatMap;

class MTLogger;
class Logger;
class Salsa : public multirotor_sim::EstimatorBase
{
public:
  Salsa();

  void init(const std::string& filename);

  void load(const std::string& filename);
  void initState();
  void initFactors();
  void initialize(const double& t, const Xformd &x0, const Vector3d& v0, const Vector2d& tau0);
  void initSolverOptions();

  void initLog();
  void logRawGNSSRes();
  void logOptimizedWindow();
  void renderGraph(const std::string& filename);

  void finishNode(const double& t, bool new_keyframe);
  void cleanUpSlidingWindow();
  const State& lastKfState();
  bool calcNewKeyframeCondition(const Features& z);

  void solve();
  void addParameterBlocks(ceres::Problem& problem);
  void addImuFactors(ceres::Problem& problem);
  void addMocapFactors(ceres::Problem& problem);
  void addOriginConstraint(ceres::Problem& problem);
  void addRawGnssFactors(ceres::Problem& problem);


  void pointPositioning(const GTime& t, const VecVec3& z,
                        std::vector<Satellite>& sat, Vector8d &xhat) const;

  void imuCallback(const double &t, const Vector6d &z, const Matrix6d &R) override;
  void mocapCallback(const double &t, const Xformd &z, const Matrix6d &R) override;
  void rawGnssCallback(const GTime& t, const VecVec3& z, const VecMat3& R,
                       std::vector<Satellite>& sat, const std::vector<bool>& slip) override;
  void imageCallback(const double& t, const ImageFeat& z, const Matrix2d& R_pix,
                     const Matrix1d& R_depth) override;
  void imageCallback(const double& t, const Features& z, const Matrix2d& R_pix, const Matrix1d& R_depth);

  bool isTrackedFeature(int id) const;

  State current_state_;

  int xbuf_head_, xbuf_tail_;
  salsa::State xbuf_[STATE_BUF_SIZE];
  vector<double> s_;
  Vector6d imu_bias_;
  int current_node_;
  int oldest_node_;
  int current_kf_;

  double switch_weight_;
  double acc_wander_weight_;
  double gyro_wander_weight_;
  Matrix6d acc_bias_xi_;
  Matrix11d anchor_xi_;

  ImuDeque imu_;
  ImuBiasDynamicsFunctor* bias_;
  AnchorFunctor* anchor_;
  MocapDeque mocap_;
  PseudorangeDeque prange_;
  ClockBiasDeque clk_;
  FeatDeque feat_;
  FeatMap xfeat_;

  ceres::Solver::Options options_;
  ceres::Solver::Summary summary_;

  Logger* state_log_ = nullptr;
  Logger* opt_log_ = nullptr;
  Logger* raw_gnss_res_log_ = nullptr;

  std::string log_prefix_;
  Camera<double> cam_;
  Xformd x_u2m_; // transform from imu to mocap frame
  Xformd x_u2b_; // transform from imu to body frame
  Xformd x_u2c_; // transform from imu to camera frame
  Xformd x_e2n_; // transform from ECEF to NED (inertial) frame
  double dt_m_; // time offset of mocap  (t(stamped) - dt_m = t(true))
  double dt_c_; // time offset of camera (t(stamped) - dt_m = t(true))
  GTime start_time_;
  Matrix2d clk_bias_Xi_;

  double kf_parallax_thresh_ = 20.0;
  double kf_feature_thresh_ = 40;
  Features kf_feat_;
  int kf_Nmatch_feat_;
  double kf_parallax_;

  int N_;


};
}
