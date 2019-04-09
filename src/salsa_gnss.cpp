#include "salsa/salsa.h"

using namespace std;
using namespace Eigen;
using namespace xform;
namespace fs = std::experimental::filesystem;

namespace salsa
{

void Salsa::ephCallback(const eph_t &eph)
{
    auto s = sats_.begin();
    while (s != sats_.end())
    {
        if (s->id_ == eph.sat)
            break;
        s++;
    }
    bool new_sat = (s == sats_.end());

    GTime now = start_time_ + current_state_.t;
    Vector3d rec_pos_ecef = WSG84::ned2ecef(x_e2n_, current_state_.x.t());
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
    filterObs(obs);

    GTime& t(filtered_obs_[0].t);
    if (current_node_ == -1)
    {
        if (sats_.size() < 8)
        {
            SD("Waiting for GNSS\n");
            return;
        }

        SD("Initialized Raw GNSS\n");
        Vector8d pp_sol = Vector8d::Zero();
        pointPositioning(t, filtered_obs_, sats_, pp_sol);
        auto phat = pp_sol.segment<3>(0);
        auto vhat = pp_sol.segment<3>(3);
        auto that = pp_sol.segment<2>(6);
        x_e2n_ = WSG84::x_ecef2ned(phat);
        start_time_ = t - current_state_.t;
        initialize(current_state_.t, Xformd::Identity(), x_e2n_.q().rotp(vhat), that);

        prange_.emplace_back(filtered_obs_.size());
        int i = 0;
        for (auto& ob : filtered_obs_)
        {
            Matrix2d R = (Vector2d() << ob.qualP, doppler_cov_).finished().asDiagonal();
            prange_.back()[i++].init(t, ob.z.topRows<2>(), sats_[ob.sat_idx], pp_sol.topRows<3>(),
                    R, current_node_, current_kf_, xbuf_head_);
        }
        return;
    }
    else
    {
        finishNode((filtered_obs_[0].t-start_time_).toSec(), true, false);

        if (obs.size() > 8)
        {
            Vector8d pp_sol = Vector8d::Zero();
            pointPositioning(t, filtered_obs_, sats_, pp_sol);
            auto phat = pp_sol.segment<3>(0);
            auto vhat = pp_sol.segment<3>(3);
            auto that = pp_sol.segment<2>(6);
            xbuf_[xbuf_head_].x.t() = WSG84::ecef2ned(x_e2n_, phat);
            xbuf_[xbuf_head_].v = xbuf_[xbuf_head_].x.q().rotp(x_e2n_.q().rotp(vhat));
            xbuf_[xbuf_head_].tau = that;

            prange_.emplace_back(filtered_obs_.size());
            int i = 0;
            for (auto& ob : filtered_obs_)
            {
                Matrix2d R = (Vector2d() << ob.qualP, doppler_cov_).finished().asDiagonal();
                prange_.back()[i++].init(t, ob.z.topRows<2>(), sats_[ob.sat_idx],
                        pp_sol.topRows<3>(), R, current_node_, current_kf_, xbuf_head_);
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
            b.block<2,1>(2*i,0) = o.z.topRows<2>() - zhat.topRows<2>();

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

}
