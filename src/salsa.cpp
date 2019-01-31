#include <Eigen/Core>
#include "geometry/xform.h"
#include "salsa/salsa.h"


using namespace std;
using namespace Eigen;
using namespace xform;

Salsa::Salsa()
{
  initLog();
  initState();
  initSolverOptions();
}

void Salsa::initState()
{
  for (int i = 0; i < N; i++)
  {
    initialized_[i] = false;
    t_[i] = NAN;
    x_.col(i).setConstant(NAN);
    v_.col(i).setConstant(NAN);
    tau_.col(i).setConstant(NAN);
  }
  imu_bias_.setZero();
  dt_mocap_ = 0.0;

  x_idx_ = -1;
  current_node_ = -1;
}

void Salsa::initLog()
{
  state_log_ = new Logger("/tmp/Salsa.State.log");
  opt_log_ = new Logger("/tmp/Salsa.Opt.log");
}

void Salsa::addResidualBlocks(ceres::Problem &problem)
{
  for (int n = 0; n < N; n++)
  {
    if (initialized_[n])
    {
      problem.AddParameterBlock(x_.data() + n*7, 7, new XformParamAD());
      problem.AddParameterBlock(v_.data() + n*3, 3);
      problem.AddParameterBlock(tau_.data() + n*2, 2);
    }
  }
  problem.AddParameterBlock(imu_bias_.data(), 6);
  problem.AddParameterBlock(&dt_mocap_, 1);
}

void Salsa::addImuFactors(ceres::Problem &problem)
{
  for (int i = 0; i < N; i++)
  {
    if (imu_[i].active_)
    {
      int from_idx = imu_[i].from_idx_;
      int to_idx = (from_idx+1)%N;
      FunctorShield<ImuFunctor>* ptr = new FunctorShield<ImuFunctor>(&imu_[i]);
      problem.AddResidualBlock(new ImuFactorAD(ptr),
                               NULL,
                               x_.data() + from_idx*7,
                               x_.data() + to_idx*7,
                               v_.data() + from_idx*3,
                               v_.data() + to_idx*3,
                               imu_bias_.data());
    }
  }
}

void Salsa::addMocapFactors(ceres::Problem &problem)
{
  for (int n = 0; n < N; n++)
  {
    if (mocap_[n].active_)
    {
      FunctorShield<MocapFunctor>* ptr = new FunctorShield<MocapFunctor>(&mocap_[n]);
      problem.AddResidualBlock(new MocapFactorAD(ptr),
                               NULL,
                               x_.data() + mocap_[n].x_idx_*7,
                               &dt_mocap_);
    }
  }
}

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

  addResidualBlocks(problem);
  addImuFactors(problem);
  addMocapFactors(problem);

  ceres::Solve(options_, &problem, &summary_);

  if (opt_log_)
  {
    opt_log_->logVectors(t_, x_, v_, tau_, imu_bias_);
    opt_log_->log(dt_mocap_);
  }
}

void Salsa::imuCallback(const double &t, const Vector6d &z, const Matrix6d &R)
{
  if (x_idx_ < 0)
    return;

  imu_[imu_idx_].integrate(t, z, R);
  current_t_ = t;
  imu_[imu_idx_].estimateXj(x_.data() + imu_[imu_idx_].from_idx_*7,
                            v_.data() + imu_[imu_idx_].from_idx_*7*3,
                            current_x_.data(),
                            current_v_.data());

  if (state_log_)
  {
    state_log_->log(current_t_);
    state_log_->logVectors(current_x_.arr(), current_v_);
  }
}


void Salsa::finishNode(const double& t)
{
  int next_imu_idx = (imu_idx_ + 1) % N;
  int next_x_idx = (x_idx_ + 1) % N;

  imu_[imu_idx_].integrate(t, imu_[imu_idx_].u_, imu_[imu_idx_].cov_);
  imu_[imu_idx_].finished();

  // Best Guess of next state
  int from_idx = imu_[imu_idx_].from_idx_;
  int to_idx = (from_idx + 1) % N;
  imu_[imu_idx_].estimateXj(x_.data() + 7*from_idx,
                            v_.data() + 3*from_idx,
                            x_.data() + 7*to_idx,
                            v_.data() + 3*to_idx);
  tau_(0, to_idx) = tau_(0, from_idx) + imu_[imu_idx_].delta_t_ * tau_(1, from_idx);
  tau_(1, to_idx) = tau_(1, from_idx);
  t_(to_idx) = t;
  initialized_[to_idx] = true;

  // turn off all other factors (they get turned on later)
  mocap_[mocap_idx_].active_ = false;

  // Prepare the next imu factor
  imu_[next_imu_idx].reset(t, imu_bias_, to_idx);
  imu_idx_ = next_imu_idx;
  x_idx_ = next_x_idx;
  current_node_++;
}

void Salsa::initialize(const double& t, const Xformd &x0, const Vector3d& v0, const Vector2d& tau0)
{
  imu_idx_ = 0;
  mocap_idx_ = 0;
  current_node_ = 0;

  initialized_[0] = true;
  t_[0] = t;
  x_.col(0) = x0.arr();
  v_.col(0) = v0;
  tau_.col(0) = tau0;

  imu_[imu_idx_].reset(t, imu_bias_, 0);

  x_idx_ = 0;
}

void Salsa::mocapCallback(const double &t, const Xformd &z, const Matrix6d &R)
{
  if (x_idx_ < 0)
  {
    initialize(t, z, Vector3d::Zero(), Vector2d::Zero());
    mocap_[mocap_idx_].init(z.arr(), Vector6d::Zero(), R, 0);
    return;
  }
  else
  {
    int next_mocap_idx = (mocap_idx_ + 1) % N;
    int prev_x_idx = x_idx_;
    mocap_idx_ = next_mocap_idx;

    finishNode(t);

    x_.col(x_idx_) = z.elements();
    Vector6d zdot = (Xformd(x_.col(x_idx_)) - Xformd(x_.col(prev_x_idx))) / (t - t_[prev_x_idx]);
    mocap_[mocap_idx_].init(z.arr(), zdot, R, x_idx_);

    solve();
  }
}
