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
    anchor_ = nullptr;
}

Salsa::~Salsa()
{
    if (bias_) delete bias_;
    if (anchor_) delete anchor_;

    if (current_state_log_) delete current_state_log_;
    if (opt_log_) delete opt_log_;
    if (raw_gnss_res_log_) delete raw_gnss_res_log_;
    if (feat_res_log_) delete feat_res_log_;
    if (feat_log_) delete feat_log_;
    if (state_log_) delete state_log_;
    if (cb_log_) delete cb_log_;
}

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
    get_yaml_node("num_sat", filename, ns_);
    get_yaml_node("num_feat", filename, nf_);
    get_yaml_node("log_prefix", filename, log_prefix_);
    get_yaml_node("switch_weight", filename, switch_weight_);
    get_yaml_node("state_buf_size", filename, STATE_BUF_SIZE);

    xbuf_.resize(STATE_BUF_SIZE);
    s_.reserve(ns_);

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
    get_yaml_eigen("image_size", filename, cam_.image_size_);
    get_yaml_node("cam_skew", filename, cam_.s_);
    get_yaml_node("kf_parallax_thresh", filename, kf_parallax_thresh_);
    get_yaml_node("kf_feature_thresh", filename, kf_feature_thresh_);

    loadKLT(filename);
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

void Salsa::initFactors()
{
    bias_ = new ImuBiasDynamicsFunctor(imu_bias_, acc_bias_xi_);
    anchor_ = new AnchorFunctor(anchor_xi_);
}

void Salsa::initLog()
{
    if (!fs::exists(fs::path(log_prefix_).parent_path()))
        fs::create_directories(fs::path(log_prefix_).parent_path());

    current_state_log_ = new Logger(log_prefix_ + "CurrentState.log");
    opt_log_ = new Logger(log_prefix_ + "Opt.log");
    raw_gnss_res_log_ = new Logger(log_prefix_+ "RawRes.log");
    feat_res_log_ = new Logger(log_prefix_ + "FeatRes.log");
    feat_log_ = new Logger(log_prefix_ + "Feat.log");
    state_log_ = new Logger(log_prefix_ + "State.log");
    cb_log_ = new Logger(log_prefix_ + "CB.log");
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
    for (auto& feat : xfeat_)
    {
        Feat& ft(feat.second);
        if (ft.funcs.size() > 0)
        {
            problem.AddParameterBlock(&ft.rho, 1);
            problem.SetParameterLowerBound(&ft.rho, 0, 0.01);
        }
    }
    problem.AddParameterBlock(imu_bias_.data(), 6);
}

void Salsa::addOriginConstraint(ceres::Problem &problem)
{
    if (xbuf_tail_ == xbuf_head_)
        return;
//    problem.SetParameterBlockConstant(xbuf_[xbuf_tail_].x.data());
//    problem.SetParameterBlockConstant(xbuf_[xbuf_tail_].v.data());
//    problem.SetParameterBlockConstant(xbuf_[xbuf_tail_].tau.data());
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
            Vector2d res;
            (*func)(xbuf_[ft->second.idx0].x.data(), xbuf_[func->to_idx_].x.data(),&ft->second.rho,
                    res.data());
            FunctorShield<FeatFunctor>* ptr = new FunctorShield<FeatFunctor>(&*func);
            problem.AddResidualBlock(new FeatFactorAD(ptr),
                                     new ceres::HuberLoss(3.0),
                                     xbuf_[ft->second.idx0].x.data(),
                                     xbuf_[func->to_idx_].x.data(),
                                     &ft->second.rho);
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
    addOriginConstraint(problem);
    addImuFactors(problem);
    addFeatFactors(problem);
    addMocapFactors(problem);
    addRawGnssFactors(problem);


    ceres::Solve(options_, &problem, &summary_);
//    std::cout << summary_.FullReport() << std::endl;

    logState();
    logOptimizedWindow();
    logRawGNSSRes();
    logFeatRes();
    logFeatures();
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
}


void Salsa::finishNode(const double& t, bool new_node, bool new_keyframe)
{
    int to_idx = xbuf_head_;
    if (new_node)
    {
        to_idx = (xbuf_head_+1) % STATE_BUF_SIZE;
        ++current_node_;
    }

    ImuFunctor& imu(imu_.back());
    ClockBiasFunctor& clk(clk_.back());
    int from = imu.from_idx_;

    // Finish the transition factors
    imu.integrate(t, imu.u_, imu.cov_);
    imu.finished(to_idx);
    clk.finished(imu.delta_t_, to_idx);

    // Set up the next node
    xbuf_[to_idx].t = t;
    imu.estimateXj(xbuf_[from].x.data(), xbuf_[from].v.data(),
                   xbuf_[to_idx].x.data(), xbuf_[to_idx].v.data());
    xbuf_[to_idx].tau(0) = xbuf_[from].tau(0) + xbuf_[from].tau(1) * imu.delta_t_;
    xbuf_[to_idx].tau(1) = xbuf_[from].tau(1);
    xbuf_[to_idx].node = current_node_;

    if (new_keyframe)
    {
        imu_.emplace_back(t, imu_bias_, to_idx, current_node_);
        clk_.emplace_back(clk_bias_Xi_, imu.from_idx_, current_node_);
        xbuf_[to_idx].kf = ++current_kf_;

        cleanUpSlidingWindow();
        if (new_kf_cb_)
            new_kf_cb_(current_kf_, kf_condition_);
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

    while (clk_.front().from_node_ < oldest_node_)
        clk_.pop_front();

    while (mocap_.size() > 0 && mocap_.begin()->node_ < oldest_node_)
        mocap_.pop_front();

    while (prange_.size() > 0 && prange_.front().front().node_ < oldest_node_)
        prange_.pop_front();


    cleanUpFeatureTracking(xbuf_tail_, oldest_desired_kf);
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
    xbuf_[0].kf = current_state_.kf = current_kf_ = 0;
    xbuf_[0].node = current_state_.node = current_node_ = 0;
    oldest_node_ = 0;

    imu_.emplace_back(t, imu_bias_, 0, current_node_);
    clk_.emplace_back(clk_bias_Xi_, 0, current_node_);
    kf_condition_ = FIRST_KEYFRAME;
    if (new_kf_cb_)
        new_kf_cb_(current_kf_, kf_condition_);
}


const State& Salsa::lastKfState()
{
    int it = xbuf_tail_;
    while (xbuf_[it].kf < 0)
        it = (it - 1 + STATE_BUF_SIZE) % STATE_BUF_SIZE;
    return xbuf_[it];
}




}
