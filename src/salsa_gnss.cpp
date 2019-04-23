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
    get_yaml_node("doppler_cov", filename, doppler_cov_);
    get_yaml_node("estimate_origin", filename, estimate_origin_);
    get_yaml_eigen("x_e2n", filename, x_e2n_.arr_);
    get_yaml_node("min_satellite_elevation", filename, min_satellite_elevation_);
    get_yaml_node("switch_weight", filename, switch_weight_);
    get_yaml_node("use_point_positioning", filename, use_point_positioning_);
    get_yaml_node("disable_gnss", filename, disable_gnss_);
}

void Salsa::ephCallback(const GTime& t, const eph_t &eph)
{
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

//void Salsa::obsCallback(const ObsVec &obs)
//{
//  filterObs(obs);
//}

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
        new_obs.sat_idx = sats[i].idx_;
        obsvec.push_back(new_obs);
    }
    obsCallback(obsvec);
}

void Salsa::obsCallback(const ObsVec &obs)
{
    last_callback_ = GNSS;

    if (start_time_.tow_sec < 0)
        start_time_ = obs[0].t - current_state_.t;

    filterObs(obs);
    GTime& t(filtered_obs_[0].t);

    if (sats_.size() < 8)
    {
        SD("Waiting for Ephemeris\n");
        return;
    }

    if (current_node_ == -1)
    {
        SD("Initialized Raw GNSS\n");
        if (use_point_positioning_)
        {
            Vector8d pp_sol = Vector8d::Zero();
            pp_sol.topRows<3>() = x_e2n_.t();
            pointPositioning(t, filtered_obs_, sats_, pp_sol);
            auto phat = pp_sol.segment<3>(0);
            auto vhat = pp_sol.segment<3>(3);
            auto that = pp_sol.segment<2>(6);

            Xformd xhat = Xformd::Identity();
            if (estimate_origin_)
                x_e2n_ = WGS84::x_ecef2ned(phat);
            else
                xhat.t() = WGS84::ecef2ned(x_e2n_, phat);

            initialize(current_state_.t, xhat, x_e2n_.q().rotp(vhat), that);
        }
        else
        {
            initialize(current_state_.t, current_state_.x, current_state_.v, Vector2d::Zero());
        }

        Vector3d rec_pos_ecef = WGS84::ned2ecef(x_e2n_, xbuf_[xbuf_head_].p);
        prange_.emplace_back(filtered_obs_.size());
        int i = 0;
        for (auto& ob : filtered_obs_)
        {
            Matrix2d R = (Vector2d() << ob.qualP, doppler_cov_).finished().asDiagonal();
            prange_.back()[i++].init(t, ob.z.topRows<2>(), sats_[ob.sat_idx], rec_pos_ecef, R,
                                     current_node_, current_kf_, xbuf_head_);
        }
        return;
    }
    else
    {
        finishNode((filtered_obs_[0].t-start_time_).toSec(), true, true);

        if (filtered_obs_.size() > 8)
        {
            if (use_point_positioning_)
            {
                Vector8d pp_sol = Vector8d::Zero();
                pp_sol.topRows<3>() = x_e2n_.t();
                pointPositioning(t, filtered_obs_, sats_, pp_sol);
                auto phat = pp_sol.segment<3>(0);
                auto vhat = pp_sol.segment<3>(3);
                auto that = pp_sol.segment<2>(6);
                xbuf_[xbuf_head_].x.t() = WGS84::ecef2ned(x_e2n_, phat);
//                xbuf_[xbuf_head_].x.q() = quat::Quatd::Identity();
                xbuf_[xbuf_head_].v = xbuf_[xbuf_head_].x.q().rotp(x_e2n_.q().rotp(vhat));
//                std::cout << "vhat " << xbuf_[xbuf_head_].v.transpose() << std::endl;
                xbuf_[xbuf_head_].tau = that;
            }

            Vector3d rec_pos_ecef = WGS84::ned2ecef(x_e2n_, xbuf_[xbuf_head_].p);
            prange_.emplace_back(filtered_obs_.size());
            int i = 0;
            for (auto& ob : filtered_obs_)
            {
                Matrix2d R = (Vector2d() << ob.qualP, doppler_cov_).finished().asDiagonal();
                prange_.back()[i++].init(t, ob.z.topRows<2>(), sats_[ob.sat_idx], rec_pos_ecef, R,
                                         current_node_, current_kf_, xbuf_head_);
            }
            solve();
        }
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
}

}
