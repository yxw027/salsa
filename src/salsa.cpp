#include "salsa/salsa.h"

using namespace std;
using namespace Eigen;
using namespace xform;
namespace fs = experimental::filesystem;

namespace salsa
{

Salsa::Salsa() :
    new_meas_(meas::basecmp)
{
    bias_ = nullptr;
    state_anchor_ = nullptr;
    x_e2n_anchor_ = nullptr;
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
    v0_ = Vector3d::Zero();
    current_state_.x = x0_;
    current_state_.v = v0_;
}

void Salsa::load(const string& filename)
{
    get_yaml_eigen("x_b2m", filename, x_b2m_.arr());
    get_yaml_eigen("x_b2c", filename, x_b2c_.arr());
    get_yaml_eigen("p_b2g", filename, p_b2g_);
    get_yaml_node("tm", filename, dt_m_);
    get_yaml_node("tc", filename, dt_c_);
    get_yaml_node("max_node_window", filename, max_node_window_);
    get_yaml_node("max_kf_window", filename, max_kf_window_);
    get_yaml_node("num_sat", filename, ns_);
    get_yaml_node("num_feat", filename, nf_);
    get_yaml_node("state_buf_size", filename, STATE_BUF_SIZE);
    get_yaml_node("use_measured_depth", filename, use_measured_depth_);
    get_yaml_node("disable_solver", filename, disable_solver_);
    get_yaml_node("disable_mocap", filename, disable_mocap_);
    get_yaml_node("max_solver_time", filename, options_.max_solver_time_in_seconds);
    get_yaml_node("max_iter", filename, options_.max_num_iterations);

    xbuf_.resize(STATE_BUF_SIZE);

    Vector11d state_anchor_cov;
    get_yaml_eigen("state_anchor_cov", filename, state_anchor_cov);
    state_anchor_xi_ = state_anchor_cov.cwiseInverse().cwiseSqrt().asDiagonal();

    Vector6d cov_diag;
    get_yaml_eigen("x_e2n_anchor_cov", filename, cov_diag);
    x_e2n_anchor_xi_ = cov_diag.cwiseInverse().cwiseSqrt().asDiagonal();

    get_yaml_eigen("bias_anchor_cov", filename, cov_diag);
    imu_bias_xi_ = cov_diag.cwiseInverse().cwiseSqrt().asDiagonal();

    Vector2d clk_bias_diag;
    get_yaml_eigen("R_clock_bias", filename, clk_bias_diag);
    clk_bias_Xi_ = clk_bias_diag.cwiseInverse().cwiseSqrt().asDiagonal();

    get_yaml_node("update_on_camera", filename, update_on_camera_);
    get_yaml_node("update_on_gnss", filename, update_on_gnss_);
    get_yaml_node("update_on_mocap", filename, update_on_mocap_);
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
    bias_ = new ImuBiasAnchor(imu_bias_, imu_bias_xi_);
    state_anchor_ = new StateAnchor(state_anchor_xi_);
    x_e2n_anchor_ = new XformAnchor(x_e2n_anchor_xi_);
}

void Salsa::addParameterBlocks(ceres::Problem &problem)
{
    problem.AddParameterBlock(x_e2n_.data(), 7, new XformParamAD());
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

    x_e2n_anchor_->set(x_e2n_);
    FunctorShield<XformAnchor>* xe2n_ptr = new FunctorShield<XformAnchor>(x_e2n_anchor_);
    problem.AddResidualBlock(new XformAnchorFactorAD(xe2n_ptr), NULL, x_e2n_.data());

    state_anchor_->set(xbuf_[xbuf_tail_]);
    FunctorShield<StateAnchor>* state_ptr = new FunctorShield<StateAnchor>(state_anchor_);
    problem.AddResidualBlock(new StateAnchorFactorAD(state_ptr), NULL, xbuf_[xbuf_tail_].x.data(),
                             xbuf_[xbuf_tail_].v.data(), xbuf_[xbuf_tail_].tau.data());

    bias_->setBias(imu_bias_);
    FunctorShield<ImuBiasAnchor>* imu_ptr = new FunctorShield<ImuBiasAnchor>(bias_);
    problem.AddResidualBlock(new ImuBiasAnchorFactorAD(imu_ptr), NULL, imu_bias_.data());
}

void Salsa::addImuFactors(ceres::Problem &problem)
{
    for (auto it = imu_.begin(); it != imu_.end(); it++)
    {
        // ignore unfinished IMU factors
        if (it->to_idx_ < 0)
            continue;

        SALSA_ASSERT(inWindow(it->to_idx_), "Trying to add factor to node outside of window");
        SALSA_ASSERT(inWindow(it->from_idx_), "Trying to add factor to node outside of window");
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
        SALSA_ASSERT(inWindow(it->idx_), "Trying to add factor to node outside of window");
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
            SALSA_ASSERT(inWindow(it->idx_), "Trying to add factor to node outside of window");
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
            SALSA_ASSERT(inWindow(ft->second.idx0), "Trying to add factor to node outside of window: %d", ft->second.idx0);
            SALSA_ASSERT(inWindow(func->to_idx_), "Trying to add factor to node outside of window: %d", func->to_idx_);
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
    options_.num_threads = 6;
    options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options_.minimizer_progress_to_stdout = false ;
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
}

void Salsa::imuCallback(const double &t, const Vector6d &z, const Matrix6d &R)
{
    current_state_.t = t;
    if (std::isnan(current_state_integrator_.t))
        current_state_integrator_.reset(t);

    if (imu_.empty())
    {
        return;
    }

    SD(1, "Got IMU t: %.3f", t);
    imu_meas_buf_.push_back({t, z, R});


    current_state_integrator_.b_ = imu_bias_;
    for (auto& z : imu_meas_buf_)
    {
        if (z.t > current_state_integrator_.t)
            current_state_integrator_.integrateStateOnly(z.t, z.z);
    }

    current_state_integrator_.estimateXj(xbuf_[xbuf_head_].x.data(),
                                         xbuf_[xbuf_head_].v.data(),
                                         current_state_.x.data(),
                                         current_state_.v.data());
    current_state_.tau = xbuf_[xbuf_head_].tau;
    SALSA_ASSERT((current_state_.x.arr().array() == current_state_.x.arr().array()).all()
                 || (current_state_.v.array() == current_state_.v.array()).all(),
                 "NaN Detected in propagation");

    //    ImuFunctor& imu(imu_.back());
    //    imu.integrate(t, z, R);

    logCurrentState();
    logImu();
}

//void Salsa::endInterval(double t)
//{
//    // shortcuts to the relevant transition factors
//    ImuFunctor& imu(imu_.back());
//    ClockBiasFunctor& clk(clk_.back());
//    const int from = imu.from_idx_;
//    int to = imu.to_idx_;

//    // see if this interval is pointing anywhere
//    bool do_cleanup = false;
//    if (to < 0)
//    {
//        assert(from == xbuf_head_);
//        // if it's not, then set up a new node
//        to = (xbuf_head_+1) % STATE_BUF_SIZE;
//        ++current_node_;
//        do_cleanup = true;
//    }

//    // Finish the transition factors
//    while (imu_meas_buf_.size() > 0 && imu_meas_buf_.front().t <= t)
//    {
//        auto& z(imu_meas_buf_.front());
//        imu.integrate(z.t, z.z, z.R);
//        imu_meas_buf_.pop_front();
//    }
//    imu.integrate(t, imu.u_, imu.cov_);
//    imu.finished(to);
//    clk.finished(imu.delta_t_, to);
//    current_state_integrator_.reset(t);

//    // Initialize the estimated state at the end of the interval
//    xbuf_[to].t = t;
//    imu.estimateXj(xbuf_[from].x.data(), xbuf_[from].v.data(),
//                   xbuf_[to].x.data(), xbuf_[to].v.data());
//    xbuf_[to].tau(0) = xbuf_[from].tau(0) + xbuf_[from].tau(1) * imu.delta_t_;
//    xbuf_[to].tau(1) = xbuf_[from].tau(1);
//    xbuf_[to].kf = -1;
//    xbuf_[to].node = current_node_;
//    xbuf_head_ = to;

//    if (do_cleanup)
//        cleanUpSlidingWindow();

//    assert(xbuf_head_ < xbuf_.size());
//    assert(xbuf_head_ != xbuf_tail_);
//}


void Salsa::cleanUpSlidingWindow()
{
    if (current_node_ < max_node_window_ && current_kf_ < max_kf_window_)
        return;

    // start by placing the tail at the oldest node
    oldest_node_ = current_node_ - max_node_window_;
    oldest_kf_ = current_kf_ - max_kf_window_;

    // Move the tail to the oldest node
    while (xbuf_[xbuf_tail_].node < oldest_node_)
        xbuf_tail_ = (xbuf_tail_ + 1) % STATE_BUF_SIZE;

    int kf_idx = xbuf_tail_;

    // Figure out which condition needs to be used by traversing the window and see where we
    // meet our criteria
    while (kf_idx != xbuf_head_ && xbuf_[kf_idx].kf < oldest_kf_)
    {
        kf_idx = (kf_idx + 1) % STATE_BUF_SIZE;
    }

    // If we have keyframes, we should check to see if this is a more conservative limit than the
    // maximum number of nodes (keyframes are expensive to optimize)
    if (kf_idx != xbuf_head_ && xbuf_[kf_idx].node > oldest_node_)
    {
        SD(2, "Using Max Keyframe Constraint To determine number of nodes to optimize");
        oldest_node_ = xbuf_[kf_idx].node;
        xbuf_tail_ = kf_idx;
    }
    else
    {
        SD(2, "Using Max Node Constraint To determine number of nodes to optimize");
    }

    SD(2, "Clean Up Sliding Window, oldest_node = %d, oldest_kf = %d", oldest_node_, oldest_kf_);


    while (imu_.begin()->from_node_ < oldest_node_)
    {
        SD(1, "removing IMU %d->%d", imu_.front().from_idx_, imu_.front().to_idx_);
        imu_.pop_front();
    }

    while (clk_.front().from_node_ < oldest_node_)
    {
        SD(1, "removing clk %d->%d", clk_.front().from_idx_, clk_.front().to_idx_);
        clk_.pop_front();
    }

    while (mocap_.size() > 0 && mocap_.begin()->node_ < oldest_node_)
    {
        SD(1, "removing mocap ->%d", mocap_.front().idx_);
        mocap_.pop_front();
    }


    while (prange_.size() > 0 && prange_.front().front().node_ < oldest_node_)
    {
        SD(1, "removing %lu prange factors all pointing at ->%d", prange_.front().size(), prange_.front().front().idx_);
        prange_.pop_front();
    }

    cleanUpFeatureTracking(kf_idx);
}

void Salsa::initialize(const double& t, const Xformd &x0, const Vector3d& v0, const Vector2d& tau0)
{
    SD(3, "Initialize State");
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
    SD(2, "Creating new keyframe %d", current_kf_);
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

void Salsa::handleMeas()
{
    std::multiset<meas::Base*>::iterator mit = new_meas_.begin();

    if (current_node_ == -1)
    {
        SD(5, "Initializing with Meas Type %d", (*mit)->type);
        initialize(*mit);
    }

    SALSA_ASSERT((*mit)->t >= xbuf_[xbuf_head_].t - 1e-6, \
                 "Unable to handle stale %s measurement.  State Time: %.3f, Meas Time: %.3f", \
                 (*mit)->Type().c_str(), xbuf_[xbuf_head_].t, (*mit)->t);

    while (mit != new_meas_.end())
    {
        integrateTransition((*mit)->t);

        switch ((*mit)->type)
        {
        case meas::Base::IMG:
        {
            initializeNodeWithImu();
            meas::Img* z = dynamic_cast<meas::Img*>(*mit);
            imageUpdate(*z);
            if (z->new_keyframe)
                startNewInterval(z->t);
            break;
        }
        case meas::Base::GNSS:
        {
            meas::Gnss* z = dynamic_cast<meas::Gnss*>(*mit);
            initializeNodeWithGnss(*z);
            gnssUpdate(*z);
            startNewInterval(z->t);
            break;
        }
        case meas::Base::MOCAP:
        {
            meas::Mocap* z = dynamic_cast<meas::Mocap*>(*mit);
            initializeNodeWithMocap(*z);
            mocapUpdate(*z);
            startNewInterval(z->t);
            break;
        }
        default:
            SALSA_ASSERT(false, "Unknown Measurement Type");
            break;
        }
        mit = new_meas_.erase(mit); // The measurement has been handled, we don't need it anymore
    }
    solve();
}

void Salsa::initialize(const meas::Base *m)
{
    switch (m->type)
    {
    case meas::Base::IMG:
    {
        SD(3, "Initialized Image\n");
        initialize(m->t, x0_, v0_, Vector2d::Zero());
        startNewInterval(m->t);
        break;
    }
    case meas::Base::GNSS:
    {
        SD(3, "Initialized GNSS\n");
        const meas::Gnss* z = dynamic_cast<const meas::Gnss*>(m);
        initializeStateGnss(*z);
        startNewInterval(z->t);
        break;
    }
    case meas::Base::MOCAP:
    {
        SD(3, "Initialized Mocap\n");
        const meas::Mocap* z = dynamic_cast<const meas::Mocap*>(m);
        initialize(z->t, z->z, Vector3d::Zero(), Vector2d::Zero());
        break;
    }
    default:
        SALSA_ASSERT(false, "Unkown Measurement Type");
        break;
    }
}

void Salsa::integrateTransition(double t)
{
    ImuFunctor& imu(imu_.back());
    int num_imu = 0;
    while (imu_meas_buf_.size() > 0 && imu_meas_buf_.front().t-1e-6 <= t)
    {
        num_imu++;
        auto& z(imu_meas_buf_.front());
        SD(1, "Integrate to t=%.3f", z.t);
        imu.integrate(z.t, z.z, z.R);
        imu_meas_buf_.pop_front();
    }

    // If the transition factors are done, then finish them
    if (num_imu >= 2)
    {
        bool do_cleanup = false;
        const int from = imu.from_idx_;
        int to = imu.to_idx_;

        if (to < 0)
        {
            assert(from == xbuf_head_);
            // if it's not, then set up a new node
            to = (xbuf_head_+1) % STATE_BUF_SIZE;
            xbuf_head_ = to;
            ++current_node_;
            SD(1, "Create a new node %d", current_node_);
            SD(1, "Advance head %d", xbuf_head_);
            do_cleanup = true;
            xbuf_[xbuf_head_].type = State::None;
            xbuf_[xbuf_head_].n_cam = 0;
            xbuf_[xbuf_head_].node = current_node_;
            xbuf_[xbuf_head_].kf = -1;
        }

        SD(1, "Integrate to t=%.2f", t);
        imu.integrate(t, imu.u_, imu.cov_);
        SD(1, "End Imu Interval at idx=%d", to);
        imu.finished(to);
        ClockBiasFunctor& clk(clk_.back());
        clk.finished(imu.delta_t_, to);
        current_state_integrator_.reset(t);

        if (do_cleanup)
            cleanUpSlidingWindow();

        SALSA_ASSERT(xbuf_head_ < xbuf_.size(), "Memory Overrun");
        SALSA_ASSERT(xbuf_head_ != xbuf_tail_, "Cleaned up too much");
    }
    else // otherwise we gotta remove this transition, because we are going to double-up this node
    {
        SD(2, "Remove Imu interval");
        if (imu.n_updates_ == 1)
        {
            SD(3, "Push singleton IMU measurement back onto queue");
            imu_meas_buf_.push_front(meas::Imu(imu.t, imu.u_, imu.cov_));
        }
        imu_.pop_back();
        clk_.pop_back();
    }
}

void Salsa::startNewInterval(double t)
{
    SD(2, "Starting a new interval. imu_.size()=%lu and xbuf_head=%d", imu_.size(), xbuf_head_);
    imu_.emplace_back(t, imu_bias_, xbuf_head_, current_node_);
    clk_.emplace_back(clk_bias_Xi_, xbuf_head_, current_node_);

    // The following makes sure that we don't plot uninitialized memory
    if (imu_.size() >= 2)
    {
        imu_.back().u_ = imu_[imu_.size() -2].u_;
        imu_.back().cov_ = imu_[imu_.size() -2].cov_;
    }
    else
    {
        imu_.back().u_.setZero();
        imu_.back().u_[2] = -9.80665;
        imu_.back().cov_ = Matrix6d::Identity();
    }
}

void Salsa::initializeNodeWithImu()
{
    if (imu_.size() == 0)
    {
        SD(5, "Tried to initialize a node from IMU with not IMU measurements");
        return;
    }

    ImuFunctor& imu(imu_.back());
    const int from = imu.from_idx_;
    int to = imu.to_idx_;

    SALSA_ASSERT(to >= 0, "Need to have a valid destination");

    SD(2, "Initialize Node %d with IMU factor %lu by integrating from %d", to, imu_.size(), from);
    xbuf_[to].t = imu.t;
    imu.estimateXj(xbuf_[from].x.data(), xbuf_[from].v.data(),
                   xbuf_[to].x.data(), xbuf_[to].v.data());
    xbuf_[to].tau(0) = xbuf_[from].tau(0) + xbuf_[from].tau(1) * imu.delta_t_;
    xbuf_[to].tau(1) = xbuf_[from].tau(1);
}

void Salsa::addMeas(const meas::Mocap &&mocap)
{
    SD(2, "Got new Mocap measurement t: %.2f", mocap.t);
    mocap_meas_buf_.push_back(mocap);
    new_meas_.insert(new_meas_.end(), &mocap_meas_buf_.back());
    if (update_on_mocap_)
        handleMeas();
}

void Salsa::addMeas(const meas::Gnss &&gnss)
{
    SD(2, "Got new GNSS measurement t: %.2f", gnss.t);
    gnss_meas_buf_.push_back(gnss);
    new_meas_.insert(new_meas_.end(), &gnss_meas_buf_.back());
    if (update_on_gnss_)
        handleMeas();
}

void Salsa::addMeas(const meas::Img &&img)
{
    SD(2, "Got new IMG measurement t: %.2f", img.t);
    img_meas_buf_.push_back(img);
    new_meas_.insert(new_meas_.end(), &img_meas_buf_.back());
    if (update_on_camera_)
        handleMeas();
}

bool Salsa::inWindow(int idx)
{
    if (idx < 0 || idx >= STATE_BUF_SIZE)
        return false;
    else if (xbuf_head_ > xbuf_tail_)
        return (idx <= xbuf_head_ && idx >= xbuf_tail_);
    else
        return (idx <= xbuf_head_ || idx >= xbuf_tail_);
}


}
