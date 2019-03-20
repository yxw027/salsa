#include "salsa/salsa.h"

using namespace std;
using namespace Eigen;
using namespace xform;
namespace fs = experimental::filesystem;

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

    get_yaml_eigen("focal_len", filename, cam_.focal_len_);
    get_yaml_eigen("distortion", filename, cam_.distortion_);
    get_yaml_eigen("cam_center", filename, cam_.cam_center_);
    get_yaml_node("cam_skew", filename, cam_.s_);
    get_yaml_node("kf_parallax_thresh", filename, kf_parallax_thresh_);
    get_yaml_node("kf_feature_thresh", filename, kf_feature_thresh_);
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
    clk_.emplace_back(clk_bias_Xi_, imu.delta_t_, imu.from_idx_, current_node_, to_idx);

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

    cleanUpFeatureTracking(oldest_node_, oldest_desired_kf);
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

const State& Salsa::lastKfState()
{
    int it = xbuf_tail_;
    while (xbuf_[it].kf < 0)
        it = (it - 1 + STATE_BUF_SIZE) % STATE_BUF_SIZE;
    return xbuf_[it];
}




}
