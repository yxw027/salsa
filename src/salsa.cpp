#include "salsa/salsa.h"

using namespace std;
using namespace Eigen;
using namespace xform;
namespace fs = experimental::filesystem;

namespace salsa
{

Salsa::Salsa()
{
    bias_ = nullptr;
    state_anchor_ = nullptr;
    x_e2n_anchor_ = nullptr;
    x_u2c_anchor_ = nullptr;
    disable_solver_ = false;
    start_time_.tow_sec = -1.0;  // flags as uninitialized
    x_e2n_ = xform::Xformd::Identity();
    x_b2c_ = xform::Xformd::Identity();
}

Salsa::~Salsa()
{
    if (bias_) delete bias_;
    if (state_anchor_) delete state_anchor_;
    if (x_e2n_anchor_) delete x_e2n_anchor_;
    if (x_u2c_anchor_) delete x_u2c_anchor_;

    for (auto&& ptr : logs_) delete ptr;
}

void Salsa::init(const string& filename)
{
    load(filename);
    initImg(filename);
    initGNSS(filename);
    initLog(filename);
    initState();
    initFactors();
    initSolverOptions();
    x0_ = Xformd::Identity();
}

void Salsa::load(const string& filename)
{
    get_yaml_eigen("x_b2m", filename, x_b2m_.arr());
    get_yaml_eigen("x_b2c", filename, x_b2c_.arr());
    get_yaml_node("tm", filename, dt_m_);
    get_yaml_node("tc", filename, dt_c_);
    get_yaml_node("node_window", filename, node_window_);
    get_yaml_node("num_sat", filename, ns_);
    get_yaml_node("num_feat", filename, nf_);
    get_yaml_node("state_buf_size", filename, STATE_BUF_SIZE);
    get_yaml_node("use_measured_depth", filename, use_measured_depth_);
    get_yaml_node("disable_solver", filename, disable_solver_);
    get_yaml_node("disable_mocap", filename, disable_mocap_);

    xbuf_.resize(STATE_BUF_SIZE);
    s_.reserve(ns_);

    Vector11d state_anchor_cov;
    get_yaml_eigen("state_anchor_cov", filename, state_anchor_cov);
    state_anchor_xi_ = state_anchor_cov.cwiseInverse().cwiseSqrt().asDiagonal();

    Vector6d cov_diag;
    get_yaml_eigen("x_b2c_anchor_cov", filename, cov_diag);
    x_b2c_anchor_xi_ = cov_diag.cwiseInverse().cwiseSqrt().asDiagonal();

    get_yaml_eigen("x_e2n_anchor_cov", filename, cov_diag);
    x_e2n_anchor_xi_ = cov_diag.cwiseInverse().cwiseSqrt().asDiagonal();

    get_yaml_eigen("bias_anchor_cov", filename, cov_diag);
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
    current_state_.x = Xformd::Identity();
    current_state_.v.setZero();
}

void Salsa::setInitialState(const Xformd &x0)
{
    x0_ = x0;
}

void Salsa::initFactors()
{
    bias_ = new ImuBiasAnchor(imu_bias_, acc_bias_xi_);
    state_anchor_ = new StateAnchor(state_anchor_xi_);
    x_e2n_anchor_ = new XformAnchor(x_e2n_anchor_xi_);
    x_u2c_anchor_ = new XformAnchor(x_b2c_anchor_xi_);
}

void Salsa::addParameterBlocks(ceres::Problem &problem)
{
    problem.AddParameterBlock(x_e2n_.data(), 7, new XformParamAD);
    problem.AddParameterBlock(x_b2c_.data(), 7, new XformParamAD());
    problem.AddParameterBlock(imu_bias_.data(), 6);

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

    for (auto& feat : xfeat_)
    {
        Feat& ft(feat.second);
        if (ft.funcs.size() > 0)
        {
            problem.AddParameterBlock(&ft.rho, 1);
            problem.SetParameterLowerBound(&ft.rho, 0, 0.01);
        }
    }
}

void Salsa::setAnchors(ceres::Problem &problem)
{
    if (xbuf_tail_ == xbuf_head_)
        return;

    if (!estimate_origin_)
    {
        problem.SetParameterBlockConstant(x_e2n_.data());
        state_anchor_->set(&xbuf_[xbuf_tail_]);
        FunctorShield<StateAnchor>* ptr = new FunctorShield<StateAnchor>(state_anchor_);
        problem.AddResidualBlock(new StateAnchorFactorAD(ptr), NULL, xbuf_[xbuf_tail_].x.data(),
                                 xbuf_[xbuf_tail_].v.data(), xbuf_[xbuf_tail_].tau.data());
    }
    else
    {
        problem.SetParameterBlockConstant(xbuf_[xbuf_tail_].x.data());
        problem.SetParameterBlockConstant(xbuf_[xbuf_tail_].v.data());
        problem.SetParameterBlockConstant(xbuf_[xbuf_tail_].tau.data());
        x_e2n_anchor_->set(&x_e2n_);
        FunctorShield<XformAnchor>* ptr = new FunctorShield<XformAnchor>(x_e2n_anchor_);
        problem.AddResidualBlock(new XformAnchorFactorAD(ptr), NULL, x_e2n_.data());
    }

    bias_->setBias(imu_bias_);
    FunctorShield<ImuBiasAnchor>* imu_ptr = new FunctorShield<ImuBiasAnchor>(bias_);
    problem.AddResidualBlock(new ImuBiasAnchorFactorAD(imu_ptr), NULL, imu_bias_.data());

    x_u2c_anchor_->set(&x_b2c_);
    FunctorShield<XformAnchor>* u2c_ptr = new FunctorShield<XformAnchor>(x_u2c_anchor_);
    problem.AddResidualBlock(new XformAnchorFactorAD(u2c_ptr), NULL, x_b2c_.data());
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
}

void Salsa::addMocapFactors(ceres::Problem &problem)
{
    for (auto it = mocap_.begin(); it != mocap_.end(); it++)
    {
        FunctorShield<MocapFunctor>* ptr = new FunctorShield<MocapFunctor>(&*it);
        problem.AddResidualBlock(new MocapFactorAD(ptr),
                                 NULL,
                                 xbuf_[it->idx_].x.data());
    }
}

void Salsa::addRawGnssFactors(ceres::Problem &problem)
{
    if (disable_gnss_)
        return;

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
        if (it->to_idx_ < 0)
            continue;
        FunctorShield<ClockBiasFunctor>* ptr = new FunctorShield<ClockBiasFunctor>(&*it);
        problem.AddResidualBlock(new ClockBiasFactorAD(ptr),
                                 NULL,
                                 xbuf_[it->from_idx_].tau.data(),
                                 xbuf_[it->to_idx_].tau.data());
    }

}

void Salsa::addFeatFactors(ceres::Problem &problem)
{
    FeatMap::iterator ft = xfeat_.begin();
    while (ft != xfeat_.end())
    {
        if (ft->second.funcs.size() < 2)
        {
            ft++;
            continue;
        }
        FeatDeque::iterator func = ft->second.funcs.begin();
        while (func != ft->second.funcs.end())
        {
            FunctorShield<FeatFunctor>* ptr = new FunctorShield<FeatFunctor>(&*func);
            problem.AddResidualBlock(new FeatFactorAD(ptr),
                                     new ceres::HuberLoss(3.0),
                                     xbuf_[ft->second.idx0].x.data(),
                                     xbuf_[func->to_idx_].x.data(),
                                     &ft->second.rho,
                                     x_b2c_.data());
            func++;
        }
        ft++;
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
    setAnchors(problem);
    addImuFactors(problem);
    addFeatFactors(problem);
    addMocapFactors(problem);
    addRawGnssFactors(problem);

    if (!disable_solver_)
        ceres::Solve(options_, &problem, &summary_);
//    std::cout << summary_.FullReport() << std::endl;

    logState();
    logOptimizedWindow();
    logRawGNSSRes();
    logFeatRes();
    logMocapRes();
    logFeatures();
    logSatPos();
    logPrangeRes();
    logXe2n();
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

    logCurrentState();
    logImu();
}

void Salsa::endInterval(double t)
{
    // shortcuts to the relevant transition factors
    ImuFunctor& imu(imu_.back());
    ClockBiasFunctor& clk(clk_.back());
    const int from = imu.from_idx_;
    int to = imu.to_idx_;

    // see if this interval is pointing anywhere
    bool do_cleanup = false;
    if (to < 0)
    {
        assert(from == xbuf_head_);
        // if it's not, then set up a new node
        to = (xbuf_head_+1) % STATE_BUF_SIZE;
        ++current_node_;
        do_cleanup = true;
    }

    // Finish the transition factors
    imu.integrate(t, imu.u_, imu.cov_);
    imu.finished(to);
    clk.finished(imu.delta_t_, to);

    // Initialize the estimated state at the end of the interval
    xbuf_[to].t = t;
    imu.estimateXj(xbuf_[from].x.data(), xbuf_[from].v.data(),
                   xbuf_[to].x.data(), xbuf_[to].v.data());
    xbuf_[to].tau(0) = xbuf_[from].tau(0) + xbuf_[from].tau(1) * imu.delta_t_;
    xbuf_[to].tau(1) = xbuf_[from].tau(1);
    xbuf_[to].kf = -1;
    xbuf_[to].node = current_node_;
    xbuf_head_ = to;

    if (do_cleanup)
        cleanUpSlidingWindow();

    assert(xbuf_head_ < xbuf_.size());
    assert(xbuf_head_ != xbuf_tail_);
}


void Salsa::startNewInterval(double t)
{
    imu_.emplace_back(t, imu_bias_, xbuf_head_, current_node_);
    clk_.emplace_back(clk_bias_Xi_, xbuf_head_, current_node_);
}

void Salsa::cleanUpSlidingWindow()
{
    if (current_node_ < node_window_)
        return;

    oldest_node_ = current_node_ - node_window_;
    while (xbuf_[xbuf_tail_].node < oldest_node_)
        xbuf_tail_ = (xbuf_tail_ + 1) % STATE_BUF_SIZE;

    while (imu_.begin()->from_node_ < oldest_node_)
        imu_.pop_front();

    while (clk_.front().from_node_ < oldest_node_)
        clk_.pop_front();

    while (mocap_.size() > 0 && mocap_.begin()->node_ < oldest_node_)
        mocap_.pop_front();

    while (prange_.size() > 0 && prange_.front().front().node_ < oldest_node_)
        prange_.pop_front();

    cleanUpFeatureTracking();
}

void Salsa::initialize(const double& t, const Xformd &x0, const Vector3d& v0, const Vector2d& tau0)
{
    xbuf_tail_ = 0;
    xbuf_head_ = 0;
    xbuf_[0].t = current_state_.t = t;
    xbuf_[0].x = x0;
    current_state_.x = x0;
    xbuf_[0].v = current_state_.v = v0;
    xbuf_[0].tau = current_state_.tau =tau0;
    xbuf_[0].kf = current_state_.kf = current_kf_ = -1;
    xbuf_[0].node = current_state_.node = current_node_ = 0;
    oldest_node_ = 0;

    assert((current_state_.x.arr().array() == current_state_.x.arr().array()).all()
            && (current_state_.v.array() == current_state_.v.array()).all());
}

void Salsa::setNewKeyframe()
{
    current_kf_++;
    xbuf_[xbuf_head_].kf = current_kf_;
    current_state_.kf = current_kf_;
    if (new_kf_cb_)
        new_kf_cb_(current_kf_, kf_condition_);
}


const State& Salsa::lastKfState()
{
    assert (current_kf_ >= 0);
    int it = xbuf_tail_;
    while (xbuf_[it].kf < 0)
        it = (it - 1 + STATE_BUF_SIZE) % STATE_BUF_SIZE;
    return xbuf_[it];
}




}
