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
    last_kf_id_ = -1;
    state_anchor_ = nullptr;
    x_e2n_anchor_ = nullptr;
    disable_solver_ = false;
    start_time_.tow_sec = -1.0;  // flags as uninitialized
    x_e2n_ = xform::Xformd::Identity();
    x_b2c_ = xform::Xformd::Identity();
    static_start_end_ = INFINITY;
    normalized_imu_ = false;
}

Salsa::~Salsa()
{
    if (state_anchor_) delete state_anchor_;
    if (x_e2n_anchor_) delete x_e2n_anchor_;
    if (video_)
    {
        video_->release();
        delete video_;
    }

    for (auto&& ptr : logs_) delete ptr;
}

void Salsa::init(const string& filename)
{
    load(filename);
    initLog(filename);
    initImg(filename);
    initGNSS(filename);
    initState();
    initFactors();
    initSolverOptions();
    x0_ = x_b2o_.inverse();
    v0_ = Vector3d::Zero();
    current_state_.x = x0_;
    current_state_.v = v0_;
}

void Salsa::load(const string& filename)
{
    get_yaml_eigen("x_b2m", filename, x_b2m_.arr());
    get_yaml_eigen("x_b2c", filename, x_b2c_.arr());
    get_yaml_eigen("x_b2o", filename, x_b2o_.arr());
    get_yaml_eigen("p_b2g", filename, p_b2g_);
    get_yaml_node("mocap_offset", filename, dt_m_);
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
    get_yaml_node("num_threads", filename, options_.num_threads);
    get_yaml_eigen("bias0", filename, bias0_);
    get_yaml_node("max_depth", filename, max_depth_);

    xbuf_.resize(STATE_BUF_SIZE);

    get_yaml_diag("state_anchor_xi", filename, state_anchor_xi_);
    get_yaml_diag("x_e2n_anchor_xi", filename, x_e2n_anchor_xi_);
    get_yaml_diag("imu_bias_xi", filename, imu_bias_Xi_);
    get_yaml_diag("clk_bias_xi", filename, clk_bias_Xi_);
    get_yaml_diag("prange_xi", filename, prange_Xi_);

    get_yaml_node("update_on_camera", filename, update_on_camera_);
    get_yaml_node("update_on_gnss", filename, update_on_gnss_);
    get_yaml_node("update_on_mocap", filename, update_on_mocap_);

    get_yaml_node("switch_xi", filename, switch_Xi_);
    get_yaml_node("switchdot_xi", filename, switchdot_Xi_);
    get_yaml_node("enable_switching_factors", filename, enable_switching_factors_);

    get_yaml_node("enable_static_start", filename, enable_static_start_);
    get_yaml_diag("static_start_Xi", filename, zero_vel_Xi_);
    get_yaml_node("static_start_imu_thresh", filename, static_start_imu_thresh_);
    get_yaml_node("camera_start_delay", filename, camera_start_delay_);
    get_yaml_node("static_start_freq", filename, static_start_freq_);
    get_yaml_node("static_start_imu_bias_relaxation", filename, static_start_imu_bias_relaxation_);
    get_yaml_node("static_start_fix_delay", filename, static_start_fix_delay_);
}

void Salsa::initState()
{
    xbuf_head_ = xbuf_tail_ = 0;

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

void Salsa::imuCallback(const double &t, const Vector6d &z, const Matrix6d &R)
{
    current_state_.t = t;
    if (std::isnan(current_state_integrator_.t))
        current_state_integrator_.reset(t);

    SD(1, "Got IMU t: %.3f", t);
    imu_meas_buf_.push_back({t, z, R});

    if (enable_static_start_ && le(t, static_start_end_))
    {
        if (z.head<3>().norm() > static_start_imu_thresh_)
        {
            SD(5, "Static Start end: t%.3f", t);
            static_start_end_ = t;
        }
        else if(gt(t, xhead().t + static_start_freq_) || (current_node_ == -1 && t > 1.0))
        {
            addMeas(meas::ZeroVel(t));
        }
    }

    if (imu_.empty())
    {
        return;
    }

    current_state_integrator_.b_ = xhead().bias;
    for (auto& z : imu_meas_buf_)
    {
        if (z.t > current_state_integrator_.t)
            current_state_integrator_.integrateStateOnly(z.t, z.z);
    }

    current_state_integrator_.estimateXj(xbuf_[xbuf_head_].x,  xbuf_[xbuf_head_].v,
                                         current_state_.x, current_state_.v);
    current_state_.tau = xbuf_[xbuf_head_].tau;
    current_state_.bias = xbuf_[xbuf_head_].bias;
    SALSA_ASSERT((current_state_.x.arr().array() == current_state_.x.arr().array()).all()
                 || (current_state_.v.array() == current_state_.v.array()).all(),
                 "NaN Detected in propagation");



    logCurrentState();
    logImu();
}



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

    int kf_idx = current_kf_ >= 0 ? xbuf_tail_ : xbuf_head_; // if we haven't declared keyframes, skip keyframe checking

    // Figure out which condition needs to be used by traversing the window and see where we
    // meet our criteria
    while (kf_idx != xbuf_head_ && (xbuf_[kf_idx].kf <= oldest_kf_ || xbuf_[kf_idx].kf < 0))
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

    SD(2, "Clean Up Sliding Window, oldest_node=n%d/i%d, oldest_kf=k%d/i%d", oldest_node_, xbuf_tail_, oldest_kf_, kf_idx);


    while (imu_.begin()->from_node_ < oldest_node_)
    {
        SD(1, "removing IMU %d->%d", imu_.front().from_idx_, imu_.front().to_idx_);
        imu_.pop_front();
        printImuIntervals();
    }
    SALSA_ASSERT(checkIMUString(), "IMU lost order");

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
      SD_S(3, "Initialize State: pos = " << x0.t_.transpose() << " euler = "
           << 180.0/M_PI * x0.q_.euler().transpose() << " q = " << x0.q_);
    xbuf_tail_ = 0;
    xbuf_head_ = 0;
    xbuf_[0].t = current_state_.t = t;
    xbuf_[0].x = x0;
    current_state_.x = x0;
    xbuf_[0].v = current_state_.v = v0;
    xbuf_[0].tau = current_state_.tau =tau0;
    xbuf_[0].kf = current_state_.kf = current_kf_ = -1;
    xbuf_[0].node = current_state_.node = current_node_ = 0;
    xbuf_[0].bias = current_state_.bias = bias0_;
    oldest_node_ = 0;
    x0_ = x0;
    v0_ = v0;
    current_state_integrator_.reset(t);

    if (enable_static_start_)
        static_start_end_ = INFINITY;
    else
        static_start_end_ = -INFINITY;

    assert((current_state_.x.arr().array() == current_state_.x.arr().array()).all()
           && (current_state_.v.array() == current_state_.v.array()).all());
}

void Salsa::setNewKeyframe(int idx)
{
    current_kf_++;
    SD(2, "Creating new keyframe %d", current_kf_);
    xbuf_[idx].kf = current_kf_;
    current_state_.kf = current_kf_;
    if (new_kf_cb_)
        new_kf_cb_(current_kf_, kf_condition_);
    last_kf_id_ = idx;
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
        initialize(*mit);
        new_meas_.erase(mit); // this measurement gets handled in the `initialize` function
        return;
    }

    while (mit != new_meas_.end())
    {
        double t = (*mit)->t;
        double t_max_node = xhead().t;
        double t_min_node = xtail().t;
        double t_max_imu = imu_meas_buf_.size() > 0 ? imu_meas_buf_.back().t : t_max_node;
        int node_idx = -1; // idx to apply the update to

        // This happens a lot, so catch it specifically
        if (eq(t, t_max_node))
        {
            node_idx = xbuf_head_;
        }
        // If our next measurement is less than our max IMU meas, but greater than
        // our most recent node
        else if (gt(t, t_max_node) && le(t, t_max_imu))
        {
            // If this is an image node, and if last node is also and image-only node,
            // and not a keyframe
            if ((xhead().type == State::Camera) && (xhead().kf < 0))
            {
                node_idx = moveNode(t);
            }
            else
            {
                node_idx = newNode(t);
                current_state_integrator_.reset(t);
            }
        }
        else if (lt(t, t_min_node))
        {
            node_idx = -1;
            SD(5, "Unable to handle stale measurements at t%.3f, Oldest: t%.3f", t, t_min_node);
        }
        // otherwise The measurement occurs either on or before our current node
        else if (lt(t, t_max_node))
        {
            node_idx = insertNode(t);
        }
        // the measurement is too far in the future to handle (wait for IMU)
        else
        {
            break;  // We were unable to handle this measurement,
        }

        // If we have a valid index to apply the measurement
        if (node_idx >= 0)
        {
            update(*mit, node_idx);
        }
        // This measurement had an error somehwere.
        else
        {
            SD(5, "Error in handling %s measurement at t%.3f", (*mit)->Type().c_str(), t);
        }
        mit = new_meas_.erase(mit); // Measurement handled, pop it!
        SALSA_ASSERT(checkGraph(), "Graph got messed up!");
    }
    if (!disable_solver_)
        solve();
}

void Salsa::update(meas::Base *m, int idx)
{
    switch(m->type)
    {
    case meas::Base::IMG:
    {
        const meas::Img* z = dynamic_cast<const meas::Img*>(m);
        imageUpdate(*z, idx);
        break;
    }
    case meas::Base::GNSS:
    {
        const meas::Gnss* z = dynamic_cast<const meas::Gnss*>(m);
        gnssUpdate(*z, idx);
        break;
    }
    case meas::Base::MOCAP:
    {
        const meas::Mocap* z = dynamic_cast<const meas::Mocap*>(m);
        mocapUpdate(*z, idx);
        break;
    }
    case meas::Base::ZERO_VEL:
    {
        const meas::ZeroVel* z = dynamic_cast<const meas::ZeroVel*>(m);
        zeroVelUpdate(*z, idx);
        break;
    }
    default:
        SD(5, "Unknown measurement type %d at t%.3f", m->type, m->t);
        break;
    }
}


bool Salsa::initialize(const meas::Base *m)
{
    switch (m->type)
    {
    case meas::Base::IMG:
    {
        SD(5, "Initialized Using Image\n");
        const meas::Img* z = dynamic_cast<const meas::Img*>(m);
        initializeStateImage(*z);
        imageUpdate(*z, xbuf_head_);
        return true;
    }
    case meas::Base::GNSS:
    {
        SD(5, "Initialized Using GNSS\n");
        const meas::Gnss* z = dynamic_cast<const meas::Gnss*>(m);
        if (initializeStateGnss(*z))
        {
            gnssUpdate(*z, xbuf_head_);
            return true;
        }
        else
            return false;
    }
    case meas::Base::MOCAP:
    {
        SD(5, "Initialized Using Mocap\n");
        const meas::Mocap* z = dynamic_cast<const meas::Mocap*>(m);
        initializeStateMocap(*z);
        mocapUpdate(*z, xbuf_head_);
        return true;
    }
    case meas::Base::ZERO_VEL:
    {
        SD(5, "Initialized Using ZeroVel\n");
        const meas::ZeroVel* z = dynamic_cast<const meas::ZeroVel*>(m);
        initializeStateZeroVel(*z);
        return false;
    }
    default:
        SALSA_ASSERT(false, "Unknown Measurement Type %d at t%.3f", m->type, m->t);
        return false;
    }
    current_state_integrator_.reset(m->t);
}
void Salsa::addMeas(const meas::Mocap &&mocap)
{
    SD(2, "Got new Mocap measurement t: %.3f", mocap.t);
    mocap_meas_buf_.push_back(mocap);
    new_meas_.insert(new_meas_.end(), &mocap_meas_buf_.back());
    if (update_on_mocap_)
        handleMeas();
}

void Salsa::addMeas(const meas::Gnss &&gnss)
{
    SD(2, "Got new GNSS measurement t: %.3f", gnss.t);
    if (!std::isfinite(gnss.t))
    {
        SD(2, "NaN Time on GNSS meas t: %.3f, skipping", gnss.t);
        return;
    }
    gnss_meas_buf_.push_back(gnss);
    new_meas_.insert(new_meas_.end(), &gnss_meas_buf_.back());
    if (update_on_gnss_ || (enable_static_start_ && current_state_.t < static_start_end_))
        handleMeas();
}

void Salsa::addMeas(const meas::Img &&img)
{
    SD(2, "Got new IMG measurement t: %.3f", img.t);
    img_meas_buf_.push_back(img);
    new_meas_.insert(new_meas_.end(), &img_meas_buf_.back());
    if (update_on_camera_)
        handleMeas();
}

void Salsa::addMeas(const meas::ZeroVel &&zv)
{
    SD(2, "Got new ZeroVel Measurement t: %.3f", zv.t);
    zv_meas_buf_.push_back(zv);
    new_meas_.insert(new_meas_.end(), &zv_meas_buf_.back());
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

bool Salsa::checkIMUString()
{
    if (imu_.size() == 0)
        return xbuf_head_ == xbuf_tail_;
    int from = xbuf_tail_;
    auto it = imu_.begin();
    double t0 = imu_.begin()->t0_;
    while(it != imu_.end())
    {
        if (it->to_idx_ == -1)
            return it+1 == imu_.end();
        if ((from != it->from_idx_) || (it->to_idx_ != (from + 1) %STATE_BUF_SIZE))
        {
            SD(5, "Index Gap in IMU String. End: %d, start: %d, SIZE %d", it->from_idx_, it->to_idx_, STATE_BUF_SIZE);
            return false;
        }
        if (xbuf_[it->from_idx_].node != it->from_node_)
        {
            SD(5, "Misaligned from_node in IMU");
            return false;
        }
        if (ne(it->t0_, t0))
        {
            SD(5, "Time Gap in IMU String end: %.8f, start: %.8f", it->t0_, t0);
            return false;
        }
        if (ne(it->t-it->t0_, it->delta_t_))
        {
            SD(5, "Time Gap in IMU String start: %.3f, end: %.8f, dt %.8f", it->t0_, it->t, it->delta_t_);
            return false;
        }
        from = it->to_idx_;
        t0 = it->t;
        it++;
    }
    return it == imu_.end();
}

bool Salsa::checkFeatures()
{
    for (auto& ft : xfeat_)
    {
        if (ft.second.kf0 < 0 || xbuf_[ft.second.idx0].kf < 0)
        {
            SD(5, "bad anchor for feature %d, n%d, i%d, k%d", ft.first, xbuf_[ft.second.idx0].node, ft.second.idx0, ft.second.kf0);
            return false;
        }
        else if (xbuf_[ft.second.idx0].kf != ft.second.kf0)
        {
            SD(5, "mismatched keyframes, ft.k%d, buf.k%d", ft.second.kf0, xbuf_[ft.second.idx0].kf);
            return false;
        }
        else if (ne(xbuf_[ft.second.idx0].t, ft.second.t0))
        {
            SD(5, "Times don't match for ft %d.  x.t%.3f, ft.t0%.3f", ft.first, xbuf_[ft.second.idx0].t, ft.second.t0);
            return false;
        }
        else
        {
            for (auto& f : ft.second.funcs)
            {
                if (xbuf_[f.to_idx_].kf < 0)
                {
                    SD(5, "Feat Functor pointing at non-keyframe f%d, n%d, i%d", ft.first, xbuf_[f.to_idx_].node, f.to_idx_);
                    return false;
                }
                else if (xbuf_[f.to_idx_].kf < ft.second.kf0)
                {
                    SD(5, "Feat Functor pointing at keyframe before anchor, %d->%d", ft.second.kf0, xbuf_[f.to_idx_].kf);
                    return false;
                }
                else if (ne(xbuf_[f.to_idx_].t, f.t_))
                {
                    SD(5, "Times don't match for ft %d.  x.t%.3f, f.t%.3f", ft.first, xbuf_[f.to_idx_].t, f.t_);
                    return false;
                }
            }
        }
    }
    return true;
}

bool Salsa::checkPrange()
{
    for (auto& pvec : prange_)
    {
        for (auto& p : pvec)
        {
            if (xbuf_[p.idx_].node != p.node_)
            {
                SD(5, "Prange nodes don't match: x.n%d, p.n%d", xbuf_[p.idx_].node, p.node_);
                return false;
            }
            else if (ne(xbuf_[p.idx_].t, (p.t - start_time_).toSec()))
            {
                SD(5, "Prange time doesn't match: x.t%.3f, p.t%.3f, idx:%d",
                   xbuf_[p.idx_].t, (p.t - start_time_).toSec(), p.idx_);
                return false;
            }
        }
    }
    return true;
}

bool Salsa::checkClkString()
{
    if (clk_.size() == 0)
        return xbuf_head_ == xbuf_tail_;
    int from = xbuf_tail_;
    auto it = clk_.begin();
    double t0 = xtail().t;
    while (it != clk_.end())
    {
        if (it->to_idx_ == -1)
            return it+1 == clk_.end();
        double tf = xbuf_[it->to_idx_].t;
        if ((from != it->from_idx_) || (it->to_idx_ != (from + 1) %STATE_BUF_SIZE))
        {
            SD(5, "Index Gap in Clk String. End: %d, start: %d, SIZE %d", it->from_idx_, it->to_idx_, STATE_BUF_SIZE);
            return false;
        }
        if (xbuf_[it->from_idx_].node != it->from_node_)
        {
            SD(5, "Misaligned from_node in Clk");
            return false;
        }
        if (ne(tf, t0 + it->dt_))
        {
            SD(5, "Time Gap in Clk String end: %.3f, start: %.3f, dt: %.3f", tf, t0, it->dt_);
            return false;
        }
        from = it->to_idx_;
        t0 = tf;
        it++;
    }
    return it == clk_.end();
}

bool Salsa::checkGraph()
{
    return checkIMUString() && checkClkString() && checkFeatures() && checkPrange();
}

int Salsa::newNode(double t)
{
    // Sanity Checks
    SALSA_ASSERT(le(t, imu_meas_buf_.back().t) , "Not enough IMU to create node"); // t <= t[imu_max]
    SALSA_ASSERT(gt(t, xhead().t), "Trying to double up node"); // t > t[node_max]

    SD(2, "New Transition Factors from %d", xhead().node);
    ClockBiasFunctor& clk = clk_.emplace_back(clk_bias_Xi_, xbuf_head_, xhead().node);
    ImuFunctor& imu = imu_.emplace_back(xhead().t, xhead().bias, imu_bias_Xi_, xbuf_head_, xhead().node);

    if (imu_.size() > 1)
    {
        SD(2, "Using previous IMU meas to initialize new imu functor");
        imu.u_ = (imu_.end()-2)->u_;
        imu.cov_ = (imu_.end()-2)->cov_;
    }
    else
    {
        SD(2, "Using next IMU meas to initialize new imu functor");
        imu.u_ = imu_meas_buf_.front().z;
        imu.cov_ = imu_meas_buf_.front().R;
    }

    bool single_imu = ge(imu_meas_buf_.front().t, t);

    // Integrate to the measurement
    while (lt(imu.t, t))
    {
        meas::Imu& next_imu(imu_meas_buf_.front());
        Vector6d z = (imu.u_ + next_imu.z)/2.0; // use trapezoidal integration

        if (le(next_imu.t, t))
        {
            if (le(next_imu.t, imu.t))
                SD((current_node_ > 1 ? 5 : 2), "Trying to integrate backwards. z.t: %.3f, imu.t: %.3f", next_imu.t, imu.t);
            else
                imu.integrate(next_imu.t, z, next_imu.R);
            imu_meas_buf_.pop_front();
        }
        else
        {
            // interpolate
            if (single_imu)
            {
                SD(1, "Handling Single IMU Interpolation");
                // IMU intervals need at least two updates otherwise we'll get NaNs when when
                // invert the covariance, so we can just apply half the measurement, twice.
                double dt = t - imu.t;
                imu.integrate(imu.t+dt/2.0, z, next_imu.R);
                imu.integrate(t, z, next_imu.R);
            }
            else
            {
                imu.integrate(t, z, next_imu.R);
            }
        }
    }

    // Create new Node
    int next_idx = (xbuf_head_ + 1) % STATE_BUF_SIZE;
    SALSA_ASSERT(next_idx != xbuf_tail_, "Overfull State Buffer");

    xbuf_[next_idx].type = State::None;
    xbuf_[next_idx].kf = -1;
    xbuf_[next_idx].node = ++current_node_;
    xbuf_[next_idx].t = t;
    imu.estimateXj(xhead().x, xhead().v, xbuf_[next_idx].x, xbuf_[next_idx].v);
    xbuf_[next_idx].tau(0) = xhead().tau(0) + imu.delta_t_*xhead().tau(1);
    xbuf_[next_idx].tau(1) = xhead().tau(1);
    xbuf_[next_idx].bias = xhead().bias;

    imu.finished(next_idx);
    clk.finished(imu.delta_t_, next_idx);

    SD(3, "Creating new node %d in idx %d", current_node_, next_idx);
    SD(2, "Advancing Head to %d", next_idx);
    xbuf_head_ = next_idx;
    printImuIntervals();
    cleanUpSlidingWindow();

    return xbuf_head_;
}

int Salsa::moveNode(double t)
{
    // Sanity Checks
    SALSA_ASSERT(le(t, imu_meas_buf_.back().t) , "Not enough IMU to move node"); // t <= t[imu_max]

    // Grab the Transition Factors
    ImuFunctor& imu = imu_.back();

    // Integrate to the measurement
    SD(2, "Extending IMU Factor n%d->n%d from to t%.3f", imu.from_node_, xhead().node, t);
    while (lt(imu.t, t))
    {
        meas::Imu& next_imu(imu_meas_buf_.front());
        Vector6d z = (imu.u_ + next_imu.z)/2.0; // use trapezoidal integration

        if (le(next_imu.t, t))
        {
            imu.integrate(next_imu.t, z, next_imu.R);
            imu_meas_buf_.pop_front();
        }
        else
        {
            // interpolate
            imu.integrate(t, z, next_imu.R);
        }
    }

    imu.finished(xbuf_head_);
    clk_.back().finished(imu.delta_t_, xbuf_head_);

    SD(2, "Sliding node %d from t%.3f -> t%.3f", xhead().node, xhead().t, t);
    int from_idx = imu.from_idx_;
    xhead().t = t;
    imu.estimateXj(xbuf_[from_idx].x, xbuf_[from_idx].v, xhead().x, xhead().v);
    xhead().tau(0) = xbuf_[from_idx].tau(0) + imu.delta_t_*xhead().tau(1);
    xhead().tau(1) = xbuf_[from_idx].tau(1);
    xhead().bias = xbuf_[from_idx].bias;
    xhead().type = State::None;
    current_state_integrator_.reset(t);

    return xbuf_head_;
}

int Salsa::insertNode(double t)
{
    // Sanity Checks
    SALSA_ASSERT(le(t, xhead().t), "Trying to insert a future node"); // t <= t[node_max]
    if (lt(t, xtail().t))
    {
        SD(5, "Unable to fuse stale Measurement:  oldest_node t%.3f, z.t=t%.3f", xtail().t, t);
        return -1;
    }

    SD(2, "Inserting Node at t%.3f", t);

    auto imu_it = imu_.end()-1;
    auto clk_it = clk_.end()-1;
    while (le(t, imu_it->t0_)) // t < imu.t_start
    {
        if (imu_it == imu_.begin())
        {
            SD(5, "Unable to find proper IMU interval to split for node insertion at %.3f", t);
            return -1;
        }
        // work backwards through buffer
        --imu_it;
        --clk_it;
    }

    if (eq(xbuf_[imu_it->to_idx_].t, t))
    {
        SD(2, "Lucky Insert on top of previous node");
        return imu_it->to_idx_;
    }
    else
    {
        SD(2, "Splitting Interval t%.3f->t%.3f at t%3.f", imu_it->t0_, imu_it->t, t);
        int new_node_idx = imu_it->to_idx_;
        // insert new node
        State& new_node(insertNodeIntoBuffer(new_node_idx));

        // split transition functions
        imu_it = imu_.insert(imu_it, std::move(imu_it->split(t)));
        clk_it = clk_.insert(clk_it, std::move(clk_it->split(t)));

        // Initialize the new node
        State& from_node(xbuf_[imu_it->from_idx_]);
        new_node.kf = -1;
        new_node.t = t;
        new_node.type = State::None;
        imu_it->estimateXj(from_node.x, from_node.v, new_node.x, new_node.v);
        new_node.tau(0) = from_node.tau(0) + from_node.tau(1) * imu_it->delta_t_;
        new_node.tau(1) = from_node.tau(1);
        new_node.bias = from_node.bias;

        // Fix all the indices
        reWireTransitionFactors(new_node_idx, imu_it, clk_it);
        reWireGnssFactors(new_node_idx);
        reWireFeatFactors(new_node_idx);

        // Cleanup/Make sure it worked correctly
        printImuIntervals();
        checkIMUString();
        printGraph();

        return new_node_idx;
    }
}

void Salsa::reWireTransitionFactors(int new_node_idx, ImuDeque::iterator& imu_it, ClockBiasDeque::iterator& clk_it)
{
    // -- finish the first half of the split
    int to_idx = new_node_idx;
    imu_it->finished(to_idx);
    clk_it->finished(imu_it->delta_t_, to_idx);

    // -- finish the second half of split
    int from_idx = new_node_idx;
    ++imu_it; ++clk_it;
    to_idx = (to_idx + 1) % STATE_BUF_SIZE;
    clk_it->from_idx_ = imu_it->from_idx_ = from_idx;
    clk_it->from_node_ = imu_it->from_node_ = xbuf_[from_idx].node;
    imu_it->finished(to_idx);
    clk_it->finished(imu_it->delta_t_, to_idx);

    // -- The rest of the factors don't need to be finished, they just need updated idx-es
    ++imu_it; ++clk_it;
    from_idx = to_idx;
    while (imu_it != imu_.end())
    {
        to_idx = (to_idx + 1) % STATE_BUF_SIZE;
        clk_it->from_idx_ = imu_it->from_idx_ = from_idx;
        clk_it->from_node_ = imu_it->from_node_ = xbuf_[from_idx].node;
        clk_it->to_idx_ = imu_it->to_idx_ = to_idx;
        from_idx = to_idx;
        ++imu_it; ++clk_it;
    }
    SALSA_ASSERT(to_idx == xbuf_head_, "Misalinged nodes/IMUs after insertion");
    SALSA_ASSERT(clk_it == clk_.end(), "Misalinged clk/IMUs after insertion");
}

void Salsa::reWireGnssFactors(int inserted_idx)
{
    for (auto& pvec : prange_)
    {
        for (auto& p : pvec)
        {
            if (stateIdxGe(p.idx_, inserted_idx))
            {
                SD(2, "Moving gnss %d->%d because of inserted %d. hd:%d, tl:%d", p.idx_, nextIdx(p.idx_), inserted_idx, xbuf_head_, xbuf_tail_);
                p.idx_ = nextIdx(p.idx_);
                p.node_ += 1;
            }
        }
    }
}

void Salsa::reWireFeatFactors(int inserted_idx)
{
    for (auto& ft : xfeat_)
    {
        if (stateIdxGe(ft.second.idx0, inserted_idx))
        {
            SD(2, "Moving anchor %d->%d", ft.second.idx0, nextIdx(ft.second.idx0));
            ft.second.idx0 = nextIdx(ft.second.idx0);
        }
        for (auto& f : ft.second.funcs)
        {
            if (stateIdxGe(f.to_idx_, inserted_idx))
            {
                SD(2, "Moving feat %d->%d because of inserted %d. hd:%d, tl:%d", ft.second.idx0, nextIdx(ft.second.idx0), inserted_idx, xbuf_head_, xbuf_tail_);
                f.to_idx_ = nextIdx(f.to_idx_);
            }
        }
    }
}

State& Salsa::insertNodeIntoBuffer(int idx)
{
    SALSA_ASSERT((xbuf_head_ + 1) % STATE_BUF_SIZE != xbuf_tail_, "Buffer Overrun");
    SALSA_ASSERT(inWindow(idx), "Trying to insert outside of buffer");

    // shift all nodes up one, meanwhile incrementing node id
    int src = xbuf_head_;
    int dst = (xbuf_head_ +1 ) % STATE_BUF_SIZE;
    xbuf_head_ = dst;
    do
    {
        xbuf_[dst] = xbuf_[src];
        xbuf_[dst].node += 1;
        src = (src - 1 + STATE_BUF_SIZE) % STATE_BUF_SIZE;
        dst = (dst - 1 + STATE_BUF_SIZE) % STATE_BUF_SIZE;
    } while (dst != idx);
    ++current_node_;


    // return reference to new node
    return xbuf_[idx];
}

int Salsa::lastKfId() const
{
    return last_kf_id_;
}

void Salsa::zeroVelUpdate(const meas::ZeroVel& m, int idx)
{
    (void)idx;
    SD(2, "ZeroVel Update, t=%.3f", m.t);
    xbuf_[idx].x = x0_;
    xbuf_[idx].v = v0_;
    if (!normalized_imu_)
    {
      Vector6d avg_imu = imu_.back().avgImuOverInterval();
      double mag = avg_imu.head<3>().norm();
      double scale_error = mag/ImuIntegrator::G;
      xtail().bias.head<3>() = xhead().bias.head<3>() = -avg_imu.head<3>()/mag * (scale_error - 1.0);
      xtail().bias.tail<3>() = xhead().bias.tail<3>() = -avg_imu.tail<3>();
      SD_S(4, "Setting Scale Bias to " << xhead().bias.transpose() << " scale error = " << scale_error);
      normalized_imu_ = true;
    }
}

void Salsa::initializeStateZeroVel(const meas::ZeroVel &m)
{
    initialize(m.t, x0_, v0_, Vector2d::Zero());
}

bool Salsa::stateIdxGe(int idx0, int idx1)
{
    SALSA_ASSERT(inWindow(idx0) && inWindow(idx1), "Cannot check idx not in window");
    return  (xbuf_head_ >= xbuf_tail_ && idx0 >= idx1) ? true  :
            (xbuf_head_ >= xbuf_tail_ && idx0 < idx1)  ? false :
            (idx0 <= xbuf_head_ && idx1 >= xbuf_tail_) ? true  :
            (idx1 <= xbuf_head_ && idx0 >= xbuf_tail_) ? false :
            (idx0 >= xbuf_tail_ && idx0 >= idx1)       ? true  :
            (idx0 <= xbuf_head_ && idx0 >= idx1)       ? true  :
                                                         false;
}

int Salsa::nextIdx(int idx)
{
    return (idx + 1) % STATE_BUF_SIZE;
}

}
