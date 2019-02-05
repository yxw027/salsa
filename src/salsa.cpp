#include <Eigen/Core>
#include "geometry/xform.h"
#include "salsa/salsa.h"


using namespace std;
using namespace Eigen;
using namespace xform;

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
  get_yaml_eigen("x_u2m", filename, x_u2m_.arr());
  get_yaml_eigen("x_u2c", filename, x_u2c_.arr());
  get_yaml_eigen("x_u2b", filename, x_u2b_.arr());
  get_yaml_node("dt_m", filename, dt_m_);
  get_yaml_node("dt_c", filename, dt_c_);
  get_yaml_node("log_prefix", filename, log_prefix_);
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

  x_idx_ = -1;
  current_node_ = -1;
  current_t_ = NAN;
  current_x_.arr().setConstant(NAN);
  current_v_.setConstant(NAN);
}

void Salsa::initFactors()
{
  imu_.reserve(N);
  for (int i = 0; i < N; i++)
    imu_.push_back(ImuFunctor());

  mocap_.reserve(N);
  for (int i = 0; i < N; i++)
    mocap_.push_back(MocapFunctor(dt_m_, x_u2m_));

  prange_.resize(N);
  for (int i = 0; i < N; i++)
  {
    prange_[i].reserve(N_SAT);
    for (int j = 0; j < N_SAT; j++)
      prange_[i].push_back(PseudorangeFunctor());
  }
}

void Salsa::initLog()
{
  state_log_ = new Logger(log_prefix_ + ".State.log");
  opt_log_ = new Logger(log_prefix_ + ".Opt.log");
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
                               x_.data() + mocap_[n].x_idx_*7);
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
  }
}

void Salsa::imuCallback(const double &t, const Vector6d &z, const Matrix6d &R)
{

  if (x_idx_ < 0)
    return;

  current_t_ = t;
  imu_[x_idx_].integrate(t, z, R);
  imu_[x_idx_].estimateXj(x_.data() + imu_[x_idx_].from_idx_*7,
                          v_.data() + imu_[x_idx_].from_idx_*3,
                          current_x_.data(),
                          current_v_.data());

  SALSA_ASSERT((current_x_.arr().array() == current_x_.arr().array()).all()
               || (current_v_.array() == current_v_.array()).all(),
               "NaN Detected in propagation");

  if (state_log_)
  {
    state_log_->log(current_t_);
    state_log_->logVectors(current_x_.arr(), current_v_);
  }
}


void Salsa::finishNode(const double& t)
{
  int next_x_idx = (x_idx_ + 1) % N;

  imu_[x_idx_].integrate(t, imu_[x_idx_].u_, imu_[x_idx_].cov_);
  imu_[x_idx_].finished();

  // Best Guess of next state
  int from_idx = imu_[x_idx_].from_idx_;
  int to_idx = (from_idx + 1) % N;
  imu_[x_idx_].estimateXj(x_.data() + 7*from_idx,
                          v_.data() + 3*from_idx,
                          x_.data() + 7*to_idx,
                          v_.data() + 3*to_idx);
  tau_(0, to_idx) = tau_(0, from_idx) + imu_[x_idx_].delta_t_ * tau_(1, from_idx);
  tau_(1, to_idx) = tau_(1, from_idx);
  t_(to_idx) = t;
  initialized_[to_idx] = true;

  // turn off all other factors (they get turned on later)
  mocap_[x_idx_].active_ = false;
  for (int i = 0; i < N_SAT; i++)
    prange_[x_idx_][i].active_ = false;

  // Prepare the next imu factor
  imu_[next_x_idx].reset(t, imu_bias_, to_idx);
  x_idx_ = next_x_idx;
  current_node_++;
}

void Salsa::initialize(const double& t, const Xformd &x0, const Vector3d& v0, const Vector2d& tau0)
{
  current_node_ = 0;
  current_t_ = t;
  current_x_ = x0;
  current_v_ = v0;

  initialized_[0] = true;
  t_[0] = t;
  x_.col(0) = x0.arr();
  v_.col(0) = v0;
  tau_.col(0) = tau0;

  imu_[0].reset(t, imu_bias_, 0);

  x_idx_ = 0;
}

void Salsa::pointPositioning(const GTime &t, const VecVec3 &z, std::vector<Satellite> &sats, Vector3d &xhat) const
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
  if (x_idx_ < 0)
  {
    initialize(t, z, Vector3d::Zero(), Vector2d::Zero());
    mocap_[x_idx_].init(z.arr(), Vector6d::Zero(), R, 0);
    return;
  }
  else
  {
    int prev_x_idx = x_idx_;

    finishNode(t);

    x_.col(x_idx_) = z.elements();
    Vector6d zdot = (Xformd(x_.col(x_idx_)) - Xformd(x_.col(prev_x_idx))) / (t - t_[prev_x_idx]);
    mocap_[x_idx_].init(z.arr(), zdot, R, x_idx_);

    solve();
  }
}

void Salsa::rawGnssCallback(const GTime &t, const VecVec3 &z, const VecMat3 &R, std::vector<Satellite> &sat)
{
  if (x_idx_ < 0)
  {
    Vector3d p_ecef;
    pointPositioning(t, z, sat, p_ecef);
    x_e2n_ = WSG84::x_ecef2ned(p_ecef);
    start_time_ = t;
    initialize(0, Xformd::Identity(), Vector3d::Zero(), Vector2d::Zero());
    for (int i = 0; i < sat.size(); i++)
    {
      prange_[0][i].init(t, z[i].topRows<2>(), sat[i], p_ecef, R[i].topLeftCorner<2,2>());
    }
    x_idx_ = 0;
    return;
  }


}
