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
    get_yaml_node("N", filename, N_);
    get_yaml_node("log_prefix", filename, log_prefix_);
    get_yaml_node("switch_weight", filename, switch_weight_);

    Vector11d anchor_cov;
    get_yaml_eigen("anchor_cov", filename, anchor_cov);
    anchor_xi_ = anchor_cov.cwiseInverse().cwiseSqrt().asDiagonal();

    Vector6d cov_diag;
    get_yaml_node("gyro_wander_weight", filename, gyro_wander_weight_);
    get_yaml_node("acc_wander_weight", filename, acc_wander_weight_);
    cov_diag << acc_wander_weight_*acc_wander_weight_,
                acc_wander_weight_*acc_wander_weight_,
                acc_wander_weight_*acc_wander_weight_,
                gyro_wander_weight_*gyro_wander_weight_,
                gyro_wander_weight_*gyro_wander_weight_,
                gyro_wander_weight_*gyro_wander_weight_;
    acc_bias_xi_ = cov_diag.cwiseInverse().cwiseSqrt().asDiagonal();

    Vector2d clk_bias_diag;
    get_yaml_eigen("R_clock_bias", filename, clk_bias_diag);
    clk_bias_Xi_ = clk_bias_diag.cwiseInverse().cwiseSqrt().asDiagonal();
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
    }
}

void Salsa::addRawGnssFactors(ceres::Problem &problem)
{
    for (auto pvec = prange_.begin(); pvec != prange_.end(); pvec++)
    {
        for (auto it = pvec->begin(); it != pvec->end(); it++)
        {
            FunctorShield<PseudorangeFunctor>* ptr = new FunctorShield<PseudorangeFunctor>(&*it);
            problem.AddResidualBlock(new PseudorangeFactorAD(ptr),
                                     new ceres::HuberLoss(2.0),
                                     xbuf_[it->idx_].x.data(),
                                     xbuf_[it->idx_].v.data(),
                                     xbuf_[it->idx_].tau.data(),
                                     x_e2n_.data());
        }
    }
    for (auto it = clk_.begin(); it != clk_.end(); it++)
    {
        FunctorShield<ClockBiasFunctor>* ptr = new FunctorShield<ClockBiasFunctor>(&*it);
        problem.AddResidualBlock(new ClockBiasFactorAD(ptr),
                                 NULL,
                                 xbuf_[it->from_idx_].tau.data(),
                                 xbuf_[it->to_idx_].tau.data());
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

    addParameterBlocks(problem);
    addOriginConstraint(problem);
    addImuFactors(problem);
    addMocapFactors(problem);
    addRawGnssFactors(problem);

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
    clk_.emplace_back(clk_bias_Xi_, imu.delta_t_, current_node_, imu.from_idx_, to_idx);

    // Set up the next node
    xbuf_[to_idx].t = t;
    imu.estimateXj(xbuf_[from].x.data(), xbuf_[from].v.data(),
                   xbuf_[to_idx].x.data(), xbuf_[to_idx].v.data());
    xbuf_[to_idx].tau(0) = xbuf_[from].tau(0) + xbuf_[from].tau(1) * imu.delta_t_;
    xbuf_[to_idx].tau(1) = xbuf_[from].tau(1);

    ++current_node_;

    // Prepare the next imu factor
    imu_.emplace_back(t, imu_bias_, to_idx, current_node_);

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
    if (current_kf_ <= N_)
        return;

    int oldest_desired_kf = current_kf_ - N_;
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

    while (prange_.front().front().node_ < oldest_node_)
        prange_.pop_front();

    while (clk_.front().from_node_ < oldest_node_)
        clk_.pop_front();
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
                             std::vector<Satellite> &sats, Vector8d &xhat) const
{
    const int nsat = sats.size();
    MatrixXd A, b;
    A.setZero(nsat*2, 8);
    b.setZero(nsat*2, 1);
    Vector8d dx;
    ColPivHouseholderQR<MatrixXd> solver;

    enum
    {
        POS = 0,
        VEL = 3,
        TAU = 6,
        TAUDOT = 7
    };

    int iter = 0;
    do
    {
        iter++;
        int i = 0;
        for (Satellite sat : sats)
        {
            Vector3d sat_pos, sat_vel;
            Vector2d sat_clk_bias;
            auto phat = xhat.segment<3>(0);
            auto vhat = xhat.segment<3>(3);
            auto that = xhat.segment<2>(6);
            GTime tnew = t + that(0);
            sat.computePositionVelocityClock(tnew, sat_pos, sat_vel, sat_clk_bias);

            Vector3d zhat ;
            sat.computeMeasurement(tnew, phat, vhat, that, zhat);
            b.block<2,1>(2*i,0) = z[i].topRows<2>() - zhat.topRows<2>();

            Vector3d e_i = (sat_pos - phat).normalized();
            A.block<1,3>(2*i,0) = -e_i.transpose();
            A(2*i,6) = Satellite::C_LIGHT;
            A.block<1,3>(2*i+1,3) = -e_i.transpose();
            A(2*i+1,7) = Satellite::C_LIGHT;

            i++;
        }

        solver.compute(A);
        dx = solver.solve(b);

        xhat += dx;
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
    if (current_node_ == -1)
    {
        if (sats.size() < 8)
        {
            SD("Waiting for GNSS\n");
            return;
        }

        SD("Initialized Raw GNSS\n");
        Vector8d pp_sol = Vector8d::Zero();
        pointPositioning(t, z, sats, pp_sol);
        auto phat = pp_sol.segment<3>(0);
        auto vhat = pp_sol.segment<3>(3);
        auto that = pp_sol.segment<2>(6);
        x_e2n_ = WSG84::x_ecef2ned(phat);
        start_time_ = t - current_state_.t;
        initialize(current_state_.t, Xformd::Identity(), x_e2n_.q().rotp(vhat), that);

        prange_.emplace_back(sats.size());
        for (int s = 0; s < sats.size(); s++)
        {
            prange_[0][s].init(t, z[s].topRows<2>(), sats[s], pp_sol.topRows<3>(), R[s].topLeftCorner<2,2>(),
                               current_node_, xbuf_head_, current_kf_);
        }
        return;
    }
    else
    {
        finishNode((t-start_time_).toSec(), true);

        if (sats.size() > 8)
        {
            Vector8d pp_sol = Vector8d::Zero();
            pointPositioning(t, z, sats, pp_sol);
            xbuf_[xbuf_head_].x.t() = WSG84::ecef2ned(x_e2n_, pp_sol.topRows<3>());

            prange_.emplace_back(sats.size());
            for (int s = 0; s < sats.size(); s++)
            {
                prange_.back()[s].init(t, z[s].topRows<2>(), sats[s], pp_sol.topRows<3>(), R[s].topLeftCorner<2,2>(),
                                       current_node_, xbuf_head_, current_kf_);
            }

            solve();
        }
    }
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
