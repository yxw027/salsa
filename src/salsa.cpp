#include <Eigen/Core>
#include "geometry/xform.h"
#include "salsa/salsa.h"


using namespace std;
using namespace Eigen;
using namespace xform;

Salsa::Salsa()
{
  initState();
  initSolverOptions();
}

void Salsa::initState()
{
  for (int i = 0; i < N; i++)
  {
    x_.col(i) = Xformd::Identity().elements();
    v_.col(i).setZero();
    tau_.col(i).setZero();
  }
  imu_bias_.setZero();
  dt_mocap_ = 0.0;

  x_idx_ = -1;
  current_node_ = -1;
}

void Salsa::addResidualBlocks(ceres::Problem &problem)
{
  for (int n = 0; n < N; n++)
  {
    problem.AddParameterBlock(x_.data() + n*7, 7, &xform_param_);
    problem.AddParameterBlock(v_.data() + n*3, 3);
    problem.AddParameterBlock(tau_.data() + n*2, 2);
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

  addImuFactors(problem);
  addMocapFactors(problem);

  ceres::Solve(options_, &problem, &summary_);
}

void Salsa::imuCallback(const double &t, const Vector6d &z, const Matrix6d &R)
{
  if (x_idx_ < 0)
    return;
  else
    imu_[imu_idx_].integrate(t, z, R);
}


void Salsa::finishNode(const double& t)
{
    int next_imu_idx = (imu_idx_ + 1) % N;
    int next_x_idx = (x_idx_ + 1) % N;

    imu_[imu_idx_].integrate(t, imu_[imu_idx_].u_, imu_[imu_idx_].cov_);
    imu_[imu_idx_].finished();

    // Best Guess of next state
    imu_[imu_idx_].estimateXj(x_.data() + 7*x_idx_, v_.data() + 3*x_idx_, x_.data() + 7*next_x_idx, v_.data() + 3*next_x_idx);
    tau_(0, next_x_idx) = tau_(0, imu_idx_) + imu_[imu_idx_].delta_t_ * tau_(1, imu_idx_);
    tau_(1, next_x_idx) = tau_(1, imu_idx_);
    t_(next_x_idx) = t;

    // turn off all other factors (they get turned on later)
    mocap_[mocap_idx_].active_ = false;

    // Prepare the next imu factor
    imu_[next_imu_idx].reset(t, imu_bias_, x_idx_);
    imu_idx_ = next_imu_idx;
    x_idx_ = next_x_idx;
    current_node_++;
}

void Salsa::initialize(const double& t, const Xformd &x0, const Vector3d& v0, const Vector2d& tau0)
{
  imu_idx_ = 0;
  mocap_idx_ = 0;
  current_node_ = 0;

  t_[0] = t;
  x_.col(0) = x0.arr();
  v_.col(0) = v0;
  tau_.col(0) = tau0;

  imu_[imu_idx_].reset(t, imu_bias_, 0);

  x_idx_ = 1;
}

void Salsa::mocapCallback(const double &t, const Xformd &z, const Matrix6d &R)
{
  if (x_idx_ < 0)
  {
    initialize(t, z, Vector3d::Zero(), Vector2d::Zero());
    mocap_[mocap_idx_].init(z.arr(), Vector6d::Zero(), R, 0);
    mocap_idx_ = 1;
    return;
  }
  else
  {
    int next_mocap_idx = (mocap_idx_ + 1) % N;
    int prev_x_idx = x_idx_;

    finishNode(t);

    x_.col(x_idx_) = z.elements();
    Vector6d zdot = (Xformd(x_.col(x_idx_)) - Xformd(x_.col(prev_x_idx))) / (t - t_[prev_x_idx]);
    mocap_[mocap_idx_].init(z.arr(), zdot, R, prev_x_idx);
    mocap_idx_ = next_mocap_idx;

    solve();
  }
}

void Salsa::featCallback(const double& t, const Vector2d& z, const Matrix2d& R, int id, double depth) {}
void Salsa::rawGnssCallback(const GTime& t, const Vector3d& z, const Matrix3d& R, Satellite& sat) { }

