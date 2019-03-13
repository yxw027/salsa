#include <Eigen/Core>
#include "geometry/xform.h"
#include "salsa/salsa.h"
#include <experimental/filesystem>


using namespace std;
using namespace Eigen;
using namespace xform;

namespace fs = std::experimental::filesystem;

namespace salsa
{

Salsa::Salsa()
{}

void Salsa::init(const string& filename)
{
  load(filename);
  initLog();
  initState();
  initFactors();
  initSolverOptions();
}

void Salsa::load(const string& filename)
{
  get_yaml_eigen("X_u2m", filename, x_u2m_.arr());
  get_yaml_eigen("q_u2b", filename, x_u2c_.q().arr_);
  get_yaml_eigen("X_u2c", filename, x_u2b_.arr());
  get_yaml_node("tm", filename, dt_m_);
  get_yaml_node("tc", filename, dt_c_);
  get_yaml_node("log_prefix", filename, log_prefix_);
  get_yaml_node("switch_weight", filename, switch_weight_);
  get_yaml_node("acc_wander_weight", filename, acc_wander_weight_);
  get_yaml_node("gyro_wander_weight", filename, gyro_wander_weight_);
  get_yaml_diag("anchor_xi", filename, anchor_xi_);

  Vector6d cov_diag;
  cov_diag << acc_wander_weight_*acc_wander_weight_,
      acc_wander_weight_*acc_wander_weight_,
      acc_wander_weight_*acc_wander_weight_,
      gyro_wander_weight_*gyro_wander_weight_,
      gyro_wander_weight_*gyro_wander_weight_,
      gyro_wander_weight_*gyro_wander_weight_;
  acc_bias_xi_ = cov_diag.cwiseInverse().asDiagonal();

  Vector2d clk_bias_diag;
  get_yaml_eigen("R_clock_bias", filename, clk_bias_diag);
  clk_bias_Xi_ = clk_bias_diag.asDiagonal();
  clk_bias_Xi_ = clk_bias_Xi_.inverse().llt().matrixL().transpose();
}

void Salsa::initState()
{
  xbuf_head_ = xbuf_tail_ = 0;
  imu_bias_.setZero();

  current_node_ = -1;
  current_kf_ = -1;
  current_state_.t = NAN;
  current_state_.x.arr().setConstant(NAN);
  current_state_.v.setConstant(NAN);
}

void Salsa::initFactors()
{
  bias_ = new ImuBiasDynamicsFunctor(imu_bias_, acc_bias_xi_);
  anchor_ = new AnchorFunctor(anchor_xi_);
}

void Salsa::initLog()
{
  if (!fs::exists(fs::path(log_prefix_).parent_path()))
    fs::create_directories(fs::path(log_prefix_).parent_path());

  state_log_ = new Logger(log_prefix_ + "State.log");
  opt_log_ = new Logger(log_prefix_ + "Opt.log");
  raw_gnss_res_log_ = new Logger(log_prefix_+ "RawRes.log");
}

void Salsa::addParameterBlocks(ceres::Problem &problem)
{
  problem.AddParameterBlock(x_e2n_.data(), 7, new XformParamAD);
  problem.SetParameterBlockConstant(x_e2n_.data());
  int idx = xbuf_tail_;
  int prev_idx = idx;
  while (prev_idx != xbuf_head_)
  {
    problem.AddParameterBlock(xbuf_[idx].x.data(), 7, new XformParamAD());
    problem.AddParameterBlock(xbuf_[idx].v.data(), 3);
    problem.AddParameterBlock(xbuf_[idx].tau.data(), 2);

    if (std::abs(1.0 - xbuf_[idx].x.q().arr_.norm()) > 1e-4)
      std::cout << "bad quaternion " << idx << ": " << xbuf_[idx].x.q().arr_.norm() << std::endl;
    prev_idx = idx;
    idx = (idx+1) % STATE_BUF_SIZE;
  }
  for (int s = 0; s < s_.size(); s++)
  {
    problem.AddParameterBlock(s_.data() + s, 1);
    problem.SetParameterBlockConstant(s_.data() + s);
  }
  problem.AddParameterBlock(imu_bias_.data(), 6);
}

void Salsa::addOriginConstraint(ceres::Problem &problem)
{
    anchor_->set(&xbuf_[xbuf_tail_]);
    FunctorShield<AnchorFunctor>* ptr = new FunctorShield<AnchorFunctor>(anchor_);
    problem.AddResidualBlock(new AnchorFactorAD(ptr), NULL, xbuf_[xbuf_tail_].x.data(),
                             xbuf_[xbuf_tail_].v.data(), xbuf_[xbuf_tail_].tau.data());
}

void Salsa::addImuFactors(ceres::Problem &problem)
{
  for (auto it = imu_.begin(); it != imu_.end(); it++)
  {
    // ignore unfinished IMU factors
    if (it->to_idx_ < 0)
      continue;

    FunctorShield<ImuFunctor>* ptr = new FunctorShield<ImuFunctor>(&*it);
    problem.AddResidualBlock(new ImuFactorAD(ptr),
                             NULL,
                             xbuf_[it->from_idx_].x.data(),
                             xbuf_[it->to_idx_].x.data(),
                             xbuf_[it->from_idx_].v.data(),
                             xbuf_[it->to_idx_].v.data(),
                             imu_bias_.data());
  }
  bias_->setBias(imu_bias_);
  FunctorShield<ImuBiasDynamicsFunctor>* ptr = new FunctorShield<ImuBiasDynamicsFunctor>(bias_);
  problem.AddResidualBlock(new ImuBiasFactorAD(ptr),
                           NULL,
                           imu_bias_.data());
}

void Salsa::addMocapFactors(ceres::Problem &problem)
{
  for (auto it = mocap_.begin(); it != mocap_.end(); it++)
  {
      FunctorShield<MocapFunctor>* ptr = new FunctorShield<MocapFunctor>(&*it);
      problem.AddResidualBlock(new MocapFactorAD(ptr),
                               NULL,
                               xbuf_[it->idx_].x.data());

      Vector6d residual;
      (*it)(xbuf_[it->idx_].x.data(), residual.data());

      volatile int debug = 0;
  }
}

//void Salsa::addRawGnssFactors(ceres::Problem &problem)
//{
//  for (int n = 0; n < N; n++)
//  {
//    for (int s = 0; s < N_SAT; s++)
//    {
//      if (prange_[n][s].active_)
//      {
//        FunctorShield<PseudorangeFunctor>* ptr = new FunctorShield<PseudorangeFunctor>(&prange_[n][s]);
//        problem.AddResidualBlock(new PseudorangeFactorAD(ptr),
//                                 new ceres::HuberLoss(2.0),
//                                 xbuf_ + n*7,
//                                 v_.data() + n*3,
//                                 tau_.data() + n*2,
//                                 x_e2n_.data());
//      }
//    }
//    if (clk_[n].active_)
//    {
//      int from_idx = n;
//      int to_idx = (from_idx+1)%N;
//      FunctorShield<ClockBiasFunctor>* ptr = new FunctorShield<ClockBiasFunctor>(&clk_[n]);
//      problem.AddResidualBlock(new ClockBiasFactorAD(ptr),
//                               NULL,
//                               tau_.data() + from_idx*2,
//                               tau_.data() + to_idx*2);
//    }
//  }
//}

void Salsa::initSolverOptions()
{
  options_.max_num_iterations = 100;
  options_.num_threads = 6;
  options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options_.minimizer_progress_to_stdout = false;
}

void Salsa::solve()
{
  ceres::Problem problem;

  addParameterBlocks(problem);
  addImuFactors(problem);
  addMocapFactors(problem);
//  addRawGnssFactors(problem);

  ceres::Solve(options_, &problem, &summary_);

  logOptimizedWindow();
  logRawGNSSRes();
}

void Salsa::imuCallback(const double &t, const Vector6d &z, const Matrix6d &R)
{
  current_state_.t = t;
  if (imu_.empty())
    return;

  ImuFunctor& imu(imu_.back());

  imu.integrate(t, z, R);
  imu.estimateXj(xbuf_[imu.from_idx_].x.data(),
                 xbuf_[imu.from_idx_].v.data(),
                 current_state_.x.data(),
                 current_state_.v.data());
  current_state_.tau = xbuf_[imu.from_idx_].tau;

  SALSA_ASSERT((current_state_.x.arr().array() == current_state_.x.arr().array()).all()
               || (current_state_.v.array() == current_state_.v.array()).all(),
               "NaN Detected in propagation");

  if (state_log_)
  {
    state_log_->log(current_state_.t);
    state_log_->logVectors(current_state_.x.arr(), current_state_.v, imu_bias_, current_state_.tau);
  }
}


void Salsa::finishNode(const double& t, bool new_keyframe)
{
  int to_idx = (xbuf_head_+1) % STATE_BUF_SIZE;

  ImuFunctor& imu(imu_.back());
  int from = imu.from_idx_;

  // Finish the transition factors
  imu.integrate(t, imu.u_, imu.cov_);
  imu.finished(to_idx);
  clk_.push_back(ClockBiasFunctor(clk_bias_Xi_));
  clk_.back().init(imu.delta_t_);

  // Set up the next node
  xbuf_[to_idx].t = t;
  imu.estimateXj(xbuf_[from].x.data(), xbuf_[from].v.data(),
                 xbuf_[to_idx].x.data(), xbuf_[to_idx].v.data());
  xbuf_[to_idx].tau(0) = xbuf_[from].tau(0) + xbuf_[from].tau(1) * imu.delta_t_;
  xbuf_[to_idx].tau(1) = xbuf_[from].tau(1);

  ++current_node_;

  // Prepare the next imu factor
  imu_.push_back(ImuFunctor(t, imu_bias_, to_idx, current_node_));

  // increment the keyframe counter if we are supposed to
  xbuf_[to_idx].node = current_node_;
  if (new_keyframe)
  {
      xbuf_[to_idx].kf = ++current_kf_;
      cleanUpSlidingWindow();
  }
  else
  {
      xbuf_[to_idx].kf = -1;
  }

  xbuf_head_ = to_idx;
  SALSA_ASSERT(xbuf_head_ != xbuf_tail_, "Ran out of xbuf_");
}

void Salsa::cleanUpSlidingWindow()
{
    if (current_kf_ <= N)
        return;

    int oldest_desired_kf = current_kf_ - N;
    while (xbuf_[xbuf_tail_].kf != oldest_desired_kf)
    {
        SALSA_ASSERT(xbuf_tail_ != xbuf_head_, "Cleaned up too much!");
        xbuf_tail_ = (xbuf_tail_ + 1) % STATE_BUF_SIZE;
    }
    oldest_node_ = xbuf_[xbuf_tail_].node;

    while (imu_.begin()->from_node_ < oldest_node_)
        imu_.pop_front();

    while (mocap_.begin()->node_ < oldest_node_)
        mocap_.pop_front();
}

void Salsa::initialize(const double& t, const Xformd &x0, const Vector3d& v0, const Vector2d& tau0)
{
  current_state_.t = t;
  current_state_.x = x0;
  current_state_.v = v0;

  xbuf_tail_ = 0;
  xbuf_head_ = 0;
  xbuf_[0].t = t;
  xbuf_[0].x = x0;
  xbuf_[0].v = v0;
  xbuf_[0].tau = tau0;
  xbuf_[0].kf = current_kf_ = 0;
  xbuf_[0].node = current_node_ = 0;
  oldest_node_ = 0;

  imu_.emplace_back(t, imu_bias_, 0, current_node_);
}

void Salsa::pointPositioning(const GTime &t, const VecVec3 &z,
                             std::vector<Satellite> &sats, Vector3d &xhat) const
{
  const int nsat = sats.size();
  MatrixXd A, b;
  A.resize(nsat, 4);
  b.resize(nsat, 1);
  Matrix<double, 4, 1> dx;
  GTime that = t;
  ColPivHouseholderQR<MatrixXd> solver;

  int iter = 0;
  do
  {
    iter++;
    int i = 0;
    for (Satellite sat : sats)
    {
      Vector3d sat_pos, sat_vel;
      Vector2d sat_clk_bias;
      sat.computePositionVelocityClock(t, sat_pos, sat_vel, sat_clk_bias);

      Vector3d zhat ;
      sat.computeMeasurement(t, xhat, Vector3d::Zero(), Vector2d::Zero(), zhat);
      b(i) = z[i](0) - zhat(0);

      A.block<1,3>(i,0) = (xhat - sat_pos).normalized().transpose();
      A(i,3) = Satellite::C_LIGHT;
      i++;
    }

    solver.compute(A);
    dx = solver.solve(b);

    xhat += dx.topRows<3>();
    that += dx(3);
  } while (dx.norm() > 1e-4);
}

void Salsa::mocapCallback(const double &t, const Xformd &z, const Matrix6d &R)
{
  if (current_node_ == -1)
  {
    SD("Initialized Mocap\n");
    initialize(t, z, Vector3d::Zero(), Vector2d::Zero());
    mocap_.emplace_back(dt_m_, x_u2m_, z.arr(), Vector6d::Zero(),
                        R.inverse().llt().matrixL().transpose(),
                        xbuf_head_, current_node_, current_kf_);
    return;
  }
  else
  {
    int prev_x_idx = xbuf_head_;

    finishNode(t, true);

    xbuf_[xbuf_head_].kf = current_node_;
    xbuf_[xbuf_head_].x = z.elements();
    Vector6d zdot = (Xformd(xbuf_[xbuf_head_].x) - Xformd(xbuf_[prev_x_idx].x))
                    / (xbuf_[xbuf_head_].t - xbuf_[prev_x_idx].t);
    mocap_.emplace_back(dt_m_, x_u2m_, z.arr(), zdot, R.inverse().llt().matrixL().transpose(),
                        xbuf_head_, current_node_, current_kf_);

    solve();

  }
}

void Salsa::rawGnssCallback(const GTime &t, const VecVec3 &z, const VecMat3 &R,
                            std::vector<Satellite> &sats, const std::vector<bool>& slip)
{
//  if (current_node_ == -1 && sats.size() > 8)
//  {
//    SD("Initialized Raw GNSS\n");
//    Vector3d p_ecef = Vector3d::Zero();
//    /// TODO: Velocity Least-Squares
//    pointPositioning(t, z, sats, p_ecef);
//    x_e2n_ = WSG84::x_ecef2ned(p_ecef);
//    start_time_ = t - current_state_.t;
//    initialize(current_state_.t, Xformd::Identity(), Vector3d::Zero(), Vector2d::Zero());

//    prange_.push_back(std::vector<PseudorangeFunctor>(sats.size()));
//    for (int s = 0; s < sats.size(); s++)
//    {
//      prange_[0][s].init(t, z[s].topRows<2>(), sats[s], p_ecef, R[s].topLeftCorner<2,2>());
//    }
//    return;
//  }
//  else
//  {
//    finishNode((t-start_time_).toSec());

//    if (sats.size() > 8)
//    {
//      Vector3d p_ecef = Vector3d::Zero(); /// TODO: use IMU position estimate
//      /// TODO: Velocity Least-Squares
//      pointPositioning(t, z, sats, p_ecef);
//      xbuf_[xbuf_head_].x.t() = WSG84::ecef2ned(x_e2n_, p_ecef);

//      prange_.push_back(std::vector<PseudorangeFunctor>(sats.size()));
//      for (int s = 0; s < sats.size(); s++)
//      {
//        prange_.back()[s].init(t, z[s].topRows<2>(), sats[s], p_ecef, R[s].topLeftCorner<2,2>());
//      }

//      solve();
//    }
//  }
}

void Salsa::logOptimizedWindow()
{
  if (opt_log_)
  {
    opt_log_->log((xbuf_head_ + STATE_BUF_SIZE - xbuf_tail_) % STATE_BUF_SIZE);

    int i = xbuf_tail_;
    while (i != xbuf_head_)
    {
      i = (i+1) % STATE_BUF_SIZE;
      opt_log_->log(xbuf_[i].t);
      opt_log_->logVectors(xbuf_[i].x.arr());
      opt_log_->logVectors(xbuf_[i].v);
      opt_log_->logVectors(xbuf_[i].tau);
    }

    opt_log_->log(s_.size());
    for (int i = 0; i < s_.size(); i++)
      opt_log_->log(s_[i]);

    opt_log_->logVectors(imu_bias_);
  }
}


void Salsa::logRawGNSSRes()
{
//  for (int n = 0; n < N; n++)
//  {
//    raw_gnss_res_log_->log(t_[n]);
//  }
//  for (int n = 0; n < N; n++)
//  {
//    for (int s = 0; s < N_SAT; s++)
//    {
//      Vector2d res = Vector2d::Constant(NAN);
//      if (prange_[n][s].active_)
//      {
//        prange_[n][s](x_.data() + n*7, v_.data() + n*3, tau_.data() + n*2, x_e2n_.data(), res.data());
//      }
//      raw_gnss_res_log_->logVectors(res);
//    }
//  }
//  for (int n = 0; n < N; n++)
//  {
//    for (int s = 0; s < N_SAT; s++)
//    {
//      int id = -1;
//      if (prange_[n][s].active_)
//      {
//        id = s;
//      }
//      raw_gnss_res_log_->log(id);
//    }
//  }
}

}
