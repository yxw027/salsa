#include "salsa/salsa.h"

using namespace std;
using namespace Eigen;
using namespace xform;
using namespace gnss_utils;
namespace fs = std::experimental::filesystem;

namespace salsa
{

void Salsa::initGNSS(const std::string& filename)
{
    get_yaml_eigen("x_e2n", filename, x_e2n_.arr_);
    get_yaml_node("min_satellite_elevation", filename, min_satellite_elevation_);
    min_satellite_elevation_ = deg2rad(min_satellite_elevation_);
    get_yaml_node("use_point_positioning", filename, use_point_positioning_);
    get_yaml_node("disable_gnss", filename, disable_gnss_);
    get_yaml_node("min_sats", filename, min_sats_);
    get_yaml_node("dt_gnss", filename, dt_gnss_);
}

void Salsa::ephCallback(const GTime& t, const eph_t &eph)
{
    if (disable_gnss_)
        return;

    if (eph.sat > 90)
        return;

    auto s = sats_.begin();
    while (s != sats_.end())
    {
        if (s->id_ == eph.sat)
            break;
        s++;
    }
    bool new_sat = (s == sats_.end());

    if (start_time_.tow_sec < 0)
        return;

    GTime now = t;
    Vector3d rec_pos_ecef = WGS84::ned2ecef(x_e2n_, current_state_.x.t());
    Satellite sat(eph, sats_.size());
    bool high_enough = sat.azimuthElevation(now, rec_pos_ecef)(1) > min_satellite_elevation_;
    if (new_sat)
    {
        if (high_enough)
            sats_.push_back(sat);
    }
    else
    {
        if (high_enough)
            s->addEphemeris(eph);
        else
            sats_.erase(s);
    }
    refreshSatIdx();
}

void Salsa::refreshSatIdx()
{
    for (int i = 0; i < sats_.size(); i++)
    {
        sats_[i].idx_ = i;
    }
}

void Salsa::filterObs(const ObsVec &obs)
{
    filtered_obs_.clear();
    GTime time;
    time.tow_sec = 0.0;

    // Only add observations we have the satellite for
    for (auto& o : obs)
    {
        assert(time.tow_sec == 0.0 || time.tow_sec == o.t.tow_sec);
        time = o.t;
        int idx = getSatIdx(o.sat);
        if (idx >= 0)
        {
            filtered_obs_.push_back(o);
            filtered_obs_.back().sat_idx = idx;
        }
    }
}

int Salsa::getSatIdx(int sat_id) const
{
    for (auto& s : sats_)
    {
        if (s.id_ == sat_id)
            return s.idx_;
    }
    return -1;
}



void Salsa::rawGnssCallback(const GTime &t, const VecVec3 &z, const VecMat3 &R,
                            SatVec &sats, const std::vector<bool>& slip)
{
    if (disable_gnss_)
        return;

    sats_ = sats;
    ObsVec obsvec;
    for (int i = 0; i < z.size(); i++)
    {
        Obs new_obs;
        new_obs.z = z[i];
        new_obs.LLI = slip[i];
        new_obs.t = t;
        new_obs.qualL = R[i](2,2);
        new_obs.qualP = R[i](0,0);
        new_obs.sat = sats[i].id_;
        new_obs.sat_idx = sats_[i].idx_ = i;
        obsvec.push_back(new_obs);
    }
    obsCallback(obsvec);
}

void Salsa::initializeNodeWithGnss(const meas::Gnss& m, int idx)
{
    if (filtered_obs_.size() >= min_sats_ && use_point_positioning_)
    {
        // Use Iterated Least-Squares to estimate x_e2n and time offset
        Vector8d pp_sol = Vector8d::Zero();
        pp_sol.topRows<3>() = x_e2n_.t();
        pointPositioning(m.obs[0].t, m.obs, sats_, pp_sol);
        auto phat = pp_sol.segment<3>(0);
        auto vhat = pp_sol.segment<3>(3);
        auto that = pp_sol.segment<2>(6);
        pp_lla_ = phat;
        logPointPosLla();
        xbuf_[idx].tau = that;
    }
}

void Salsa::gnssUpdate(const meas::Gnss &m, int idx)
{
    SD(2, "Gnss Update on node %d, t=%.3f", xbuf_[idx].node, m.t);

    // Sanity Checks
    SALSA_ASSERT((xbuf_[idx].type & State::Gnss) == 0, "Cannot double-up with Gnss nodes");
    initializeNodeWithGnss(m, idx);

    Vector3d rec_pos_ecef = WGS84::ned2ecef(x_e2n_, xbuf_[idx].p);
    prange_.emplace_back(m.obs.size());
    int i = 0;
    for (auto& ob : m.obs)
    {
        Matrix2d xi = prange_Xi_;
        xi(0,0) *= std::sqrt(1.0/ob.qualP);
        xi(1,1) *= std::sqrt(1.0/ob.qualP);
        prange_.back()[i++].init(m.obs[0].t, ob.z.topRows<2>(), sats_[ob.sat_idx], rec_pos_ecef, prange_Xi_,
                                 switch_Xi_, p_b2g_, xbuf_[idx].node, idx);
    }
    xbuf_[idx].type |= State::Gnss;
}

bool Salsa::initializeStateGnss(const meas::Gnss &m)
{
    if (sats_.size() < min_sats_)
    {
        SD(5, "Waiting for Ephemeris, got %lu sats\n", sats_.size());
        return false;
    }

    if (use_point_positioning_)
    {
        Vector8d pp_sol = Vector8d::Zero();
        pp_sol.topRows<3>() = x_e2n_.t();
        pointPositioning(m.obs[0].t, m.obs, sats_, pp_sol);
        auto phat = pp_sol.segment<3>(0);
        auto vhat = pp_sol.segment<3>(3);
        auto that = pp_sol.segment<2>(6);
        pp_lla_ = phat;
        logPointPosLla();

        Xformd x_e2bn = gnss_utils::WGS84::x_ecef2ned(phat);
        Xformd x_bn2b(Vector3d::Zero(), x0_.q());
        x_e2n_ = x_e2bn * x_bn2b * x0_.inverse();
    }
    initialize(m.t, x0_, Vector3d::Zero(), Vector2d::Zero());
    return true;
}

void Salsa::obsCallback(const ObsVec &obs)
{
    if (!std::isfinite(current_state_.t))
        return;

    if (start_time_.tow_sec < 0)
        start_time_ = obs[0].t - current_state_.t - dt_gnss_;

    filterObs(obs);
    if (filtered_obs_.size() > 0)
    {
      GTime& t(filtered_obs_[0].t);
      addMeas(meas::Gnss((t - start_time_).toSec(), filtered_obs_));
    }
}

void Salsa::pointPositioning(const GTime &t, const ObsVec &obs, SatVec &sats, Vector8d &xhat) const
{
    const int nobs = obs.size();
    MatrixXd A, b;
    A.setZero(nobs*2, 8);
    b.setZero(nobs*2, 1);
    Vector8d dx;
    ColPivHouseholderQR<MatrixXd> solver;

    int iter = 0;
    do
    {
        int i = 0;
        for (auto&& o : obs)
        {
            Satellite& sat(sats[o.sat_idx]);
            Vector3d sat_pos, sat_vel;
            Vector2d sat_clk_bias;
            auto phat = xhat.segment<3>(0);
            auto vhat = xhat.segment<3>(3);
            auto that = xhat.segment<2>(6);
            GTime tnew = t + that(0);
            sat.computePositionVelocityClock(tnew, sat_pos, sat_vel, sat_clk_bias);

            Vector3d zhat ;
            sat.computeMeasurement(tnew, phat, vhat, that, zhat);
            assert ((zhat.array() == zhat.array()).all());
            b(2*i) = o.z(0) - zhat(0);
            b(2*i + 1) = o.z(1) - zhat(1);

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
    } while (dx.norm() > 1e-4 && ++iter < 10);
    assert ((xhat.array() == xhat.array()).all());
}

}
