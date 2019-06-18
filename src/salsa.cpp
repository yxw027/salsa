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
    get_yaml_node("num_threads", filename, options_.num_threads);
    get_yaml_eigen("bias0", filename, imu_bias_);

    xbuf_.resize(STATE_BUF_SIZE);

    Vector11d state_anchor_cov;
    get_yaml_eigen("state_anchor_cov", filename, state_anchor_cov);
    state_anchor_xi_ = state_anchor_cov.cwiseInverse().cwiseSqrt().asDiagonal();

    Vector6d cov_diag;
    get_yaml_eigen("x_e2n_anchor_cov", filename, cov_diag);
    x_e2n_anchor_xi_ = cov_diag.cwiseInverse().cwiseSqrt().asDiagonal();

    get_yaml_eigen("bias_anchor_cov", filename, cov_diag);
    imu_bias_xi_ = cov_diag.cwiseInverse().cwiseSqrt().asDiagonal();

    get_yaml_diag("clk_bias_xi", filename, clk_bias_Xi_);

    get_yaml_node("update_on_camera", filename, update_on_camera_);
    get_yaml_node("update_on_gnss", filename, update_on_gnss_);
    get_yaml_node("update_on_mocap", filename, update_on_mocap_);

    get_yaml_node("switch_xi", filename, switch_Xi_);
    get_yaml_node("switchdot_xi", filename, switchdot_Xi_);
    get_yaml_node("enable_switching_factors", filename, enable_switching_factors_);
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

    if (imu_.empty())
    {
        return;
    }

    current_state_integrator_.b_ = imu_bias_;
    for (auto& z : imu_meas_buf_)
    {
        if (z.t > current_state_integrator_.t)
            current_state_integrator_.integrateStateOnly(z.t, z.z);
    }

    current_state_integrator_.estimateXj(xbuf_[xbuf_head_].x,  xbuf_[xbuf_head_].v,
                                         current_state_.x, current_state_.v);
    current_state_.tau = xbuf_[xbuf_head_].tau;
    SALSA_ASSERT((current_state_.x.arr().array() == current_state_.x.arr().array()).all()
                 || (current_state_.v.array() == current_state_.v.array()).all(),
                 "NaN Detected in propagation");

    handleMeas();
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

    SD(2, "Clean Up Sliding Window, oldest_node = %d, oldest_kf = %d", oldest_node_, oldest_kf_);


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
    SD_S(4, "Initialize State: pos = " << x0.t_.transpose() << " euler = "
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
    oldest_node_ = 0;

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
            }
        }
        else if (lt(t, t_min_node))
        {
            node_idx = -1;
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
    }
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
    default:
        SALSA_ASSERT(false, "Unknown Measurement Type %d at t%.3f", m->type, m->t);
        return false;
    }
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
    gnss_meas_buf_.push_back(gnss);
    new_meas_.insert(new_meas_.end(), &gnss_meas_buf_.back());
    if (update_on_gnss_)
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
            SD(5, "Misaligned from_node in Clk");
            return false;
        }
        if (std::abs(it->t0_-t0) > 1e-8)
        {
            SD(5, "Time Gap in IMU String end: %.3f, start: %.3f", it->t0_, t0);
            return false;
        }
        from = it->to_idx_;
        t0 = it->t;
        it++;
    }
    return it == imu_.end();
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

int Salsa::newNode(double t)
{
    // Sanity Checks
    SALSA_ASSERT(le(t, imu_meas_buf_.back().t) , "Not enough IMU to create node"); // t <= t[imu_max]
    SALSA_ASSERT(gt(t, xhead().t), "Trying to double up node"); // t > t[node_max]

    SD(2, "New Transition Factors from %d", xhead().node);
    ClockBiasFunctor& clk = clk_.emplace_back(clk_bias_Xi_, xbuf_head_, xhead().node);
    ImuFunctor& imu = imu_.emplace_back(xhead().t, imu_bias_, xbuf_head_, xhead().node);
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


    // figure out how many imu measurements we have to integrate before our measurement
    int num_imu = 1; // Count any interpolation we have to take care of
    for (auto& it : imu_meas_buf_)
    {
        if (lt(it.t, t)) // imu.t < t
        {
            ++num_imu;
            break;
        }
        else // imu.t >= t
            break;
    }
    SD(1, "We have %d imu measurements between t%.3f and t%.3f", num_imu, xhead().t, t);

    // Integrate to the measurement
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
            if (num_imu == 1)
            {
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

    xbuf_[next_idx].kf = -1;
    xbuf_[next_idx].node = ++current_node_;
    xbuf_[next_idx].t = t;
    imu.estimateXj(xhead().x, xhead().v, xbuf_[next_idx].x, xbuf_[next_idx].v);
    xbuf_[next_idx].tau(0) = xhead().tau(0) + imu.delta_t_*xhead().tau(1);
    xbuf_[next_idx].tau(1) = xhead().tau(1);

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
    ClockBiasFunctor& clk = clk_.back();
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
    clk.finished(imu.delta_t_, xbuf_head_);

    SD(2, "Sliding node %d from t%.3f -> t%.3f", xhead().node, xhead().t, t);
    int from_idx = imu.from_idx_;
    xhead().t = t;
    imu.estimateXj(xbuf_[from_idx].x, xbuf_[from_idx].v, xhead().x, xhead().v);
    xhead().tau(0) = xbuf_[from_idx].tau(0) + imu.delta_t_*xhead().tau(1);
    xhead().tau(1) = xbuf_[from_idx].tau(1);
    xhead().type = State::None;

    return xbuf_head_;
}

int Salsa::insertNode(double t)
{
    // Sanity Checks
    SALSA_ASSERT(le(t, xhead().t), "Trying to insert a future node"); // t <= t[node_max]
    if (ge(t, xtail().t))
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

        // Fix all the indices
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

        // Cleanup/Make sure it worked correctly
        printImuIntervals();
        checkIMUString();
        printGraph();

        SALSA_ASSERT(to_idx == xbuf_head_, "Misalinged nodes/IMUs after insertion");
        SALSA_ASSERT(clk_it == clk_.end(), "Misalinged clk/IMUs after insertion");
        return new_node_idx;
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
//

//void Salsa::integrateTransition(double t)
//{
//    ImuFunctor& imu(imu_.back());
//    int num_imu = 0;
//    while (imu_meas_buf_.size() > 0 && imu_meas_buf_.front().t-1e-6 <= t)
//    {
//        auto& z(imu_meas_buf_.front());
//        if (z.t < imu.t)
//        {
//            SD(4, "Removing unusable imu measurement with t=%.3f, (Integrator Time = %.3f)", z.t, imu.t);
//            imu_meas_buf_.pop_front();
//        }
//        else
//        {
//            num_imu++;
//            SD(1, "Integrate to t=%.3f", z.t);
//            imu.integrate(z.t, z.z, z.R);
//            imu_meas_buf_.pop_front();
//        }
//    }

//    // If the transition factors are done, then finish them
//    if (num_imu >= 2)
//    {
//        bool do_cleanup = false;
//        const int from = imu.from_idx_;
//        int to = imu.to_idx_;

//        if (to < 0)
//        {
//            assert(from == xbuf_head_);
//            // if it's not, then set up a new node
//            to = (xbuf_head_+1) % STATE_BUF_SIZE;
//            xbuf_head_ = to;
//            ++current_node_;
//            SD(1, "Create a new node %d", current_node_);
//            SD(1, "Advance head %d", xbuf_head_);
//            do_cleanup = true;
//            xbuf_[xbuf_head_].type = State::None;
//            xbuf_[xbuf_head_].n_cam = 0;
//            xbuf_[xbuf_head_].node = current_node_;
//            xbuf_[xbuf_head_].kf = -1;
//        }

//        SD(1, "Integrate to t=%.3f", t);
//        imu.integrate(t, imu.u_, imu.cov_);
//        SD(1, "End Imu Interval at idx=%d", to);
//        imu.finished(to);
//        ClockBiasFunctor& clk(clk_.back());
//        clk.finished(imu.delta_t_, to);
//        current_state_integrator_.reset(t);

//        if (do_cleanup)
//            cleanUpSlidingWindow();

//        SALSA_ASSERT(xbuf_head_ < xbuf_.size(), "Memory Overrun");
//        SALSA_ASSERT(xbuf_head_ != xbuf_tail_, "Cleaned up too much");
//    }
//    else if (imu.n_updates_ <= 1) // otherwise we gotta remove this transition, because we are going to double-up this node
//    {
//        SD(2, "Remove Imu interval, so we can double up");
//        if (imu.n_updates_ == 1)
//        {
//            SD(3, "Push singleton IMU measurement into previous interval");
//            //            imu_meas_buf_.push_front(meas::Imu(imu.t, imu.u_, imu.cov_));
//            ImuFunctor& prev_imu(*(imu_.end()-2));
//            prev_imu.integrate(imu.t, imu.u_, imu.cov_);
//            prev_imu.finished(prev_imu.to_idx_);
//        }
//        imu_.pop_back();
//        printImuIntervals();
//        SALSA_ASSERT(checkIMUString(), "IMU lost order");
//        clk_.pop_back();
//    }
//    else
//    {
//        printImuIntervals();
//        SD(2, "Two measurements on node %d", current_node_);
//        SALSA_ASSERT(checkIMUString(), "IMU lost order");
//    }
//}



//void Salsa::startNewInterval(double t)
//{
//    SD(2, "Starting a new interval. imu_.size()=%lu and xbuf_head=%d", imu_.size(), xbuf_head_);
//    imu_.emplace_back(t, imu_bias_, xbuf_head_, current_node_);
//    printImuIntervals();
//    SALSA_ASSERT(checkIMUString(), "IMU lost order");
//    clk_.emplace_back(clk_bias_Xi_, xbuf_head_, current_node_);

//    // The following makes sure that we don't plot uninitialized memory
//    if (imu_.size() > 1)
//    {
//        imu_.back().u_ = imu_[imu_.size()-2].u_;
//        imu_.back().cov_ = imu_[imu_.size()-2].cov_;
//    }
//    else
//    {
//        imu_.back().u_.setZero();
//        imu_.back().u_[2] = -9.80665;
//        imu_.back().cov_ = Matrix6d::Identity();
//    }
//}

//void Salsa::initializeNodeWithImu()
//{
//    if (imu_.size() == 0)
//    {
//        return;
//    }

//    const ImuFunctor& imu(imu_.back());
//    const int from = imu.from_idx_;
//    int to = imu.to_idx_;

//    SALSA_ASSERT(to >= 0, "Need to have a valid destination");
//    SALSA_ASSERT(std::abs(1.0 - xbuf_[from].x.q().arr_.norm()) < 1e-8, "Quat left manifold");

//    SD(2, "Initialize Node %d with IMU factor %lu by integrating from %d", to, imu_.size(), from);
//    xbuf_[to].t = imu.t;
//    imu.estimateXj(xbuf_[from].x.data(), xbuf_[from].v.data(),
//                   xbuf_[to].x.data(), xbuf_[to].v.data());
//    xbuf_[to].tau(0) = xbuf_[from].tau(0) + xbuf_[from].tau(1) * imu.delta_t_;
//    xbuf_[to].tau(1) = xbuf_[from].tau(1);
//    SALSA_ASSERT(std::abs(1.0 - xbuf_[to].x.q().arr_.norm()) < 1e-8, "Quat left manifold");
//}


//void Salsa::handleMeas()
//{
//    if (new_meas_.size() == 0)
//        return;

//    std::multiset<meas::Base*>::iterator mit = new_meas_.begin();

//    if (current_node_ == -1)
//    {
//        initialize(*mit);
//        new_meas_.erase(mit);
//        return;
//    }

//    if((*mit)->t < xbuf_[xbuf_tail_].t - 1e-6)
//    {
//        SD(5, "Unable to handle stale %s measurement.  State Time: %.3f, Meas Time: %.3f", \
//           (*mit)->Type().c_str(), xbuf_[xbuf_tail_].t, (*mit)->t);
//        mit = new_meas_.erase(mit);
//    }

//    // figure out the most recent measurement we can handle (in case IMU is delayed)
//    double t_max = xbuf_[xbuf_head_].t;
//    if (imu_meas_buf_.size() > 0)
//        t_max = imu_meas_buf_.back().t;

//    while (mit != new_meas_.end())
//    {
//        if ((*mit)->t-1e-6 > t_max)
//        {
//            SD(3, "Unable to handle %s measurement, because the IMU isn't here yet.  z.t: %.3f, Imu.t: %.3f",
//               (*mit)->Type().c_str(), (*mit)->t, t_max);
//            return;
//        }
//        integrateTransition((*mit)->t);

//        switch ((*mit)->type)
//        {
//        case meas::Base::IMG:
//        {
//            initializeNodeWithImu();
//            meas::Img* z = dynamic_cast<meas::Img*>(*mit);
//            imageUpdate(*z);
//            if (z->new_keyframe)
//                startNewInterval(z->t);
//            break;
//        }
//        case meas::Base::GNSS:
//        {
//            meas::Gnss* z = dynamic_cast<meas::Gnss*>(*mit);
//            initializeNodeWithGnss(*z);
//            gnssUpdate(*z);
//            startNewInterval(z->t);
//            break;
//        }
//        case meas::Base::MOCAP:
//        {
//            meas::Mocap* z = dynamic_cast<meas::Mocap*>(*mit);
//            initializeNodeWithMocap(*z);
//            mocapUpdate(*z);
//            startNewInterval(z->t);
//            break;
//        }
//        default:
//            SALSA_ASSERT(false, "Unknown Measurement Type");
//            break;
//        }
//        mit = new_meas_.erase(mit); // The measurement has been handled, we don't need it anymore
//    }
//    solve();
//    SD(2, "Finished New Measurements\n\n\n");
//}


}
