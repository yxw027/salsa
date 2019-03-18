#include "salsa/salsa.h"

using namespace std;
using namespace Eigen;
using namespace xform;
namespace fs = std::experimental::filesystem;

namespace salsa
{

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
            prange_[0][s].init(t, z[s].topRows<2>(), sats[s], pp_sol.topRows<3>(),
                               R[s].topLeftCorner<2,2>(),
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
            auto phat = pp_sol.segment<3>(0);
            auto vhat = pp_sol.segment<3>(3);
            auto that = pp_sol.segment<2>(6);
            xbuf_[xbuf_head_].x.t() = WSG84::ecef2ned(x_e2n_, phat);
            xbuf_[xbuf_head_].v = xbuf_[xbuf_head_].x.q().rotp(x_e2n_.q().rotp(vhat));
            xbuf_[xbuf_head_].tau = that;

            prange_.emplace_back(sats.size());
            for (int s = 0; s < sats.size(); s++)
            {
                prange_.back()[s].init(t, z[s].topRows<2>(), sats[s], pp_sol.topRows<3>(),
                                       R[s].topLeftCorner<2,2>(),
                                       current_node_, xbuf_head_, current_kf_);
            }

            solve();
        }
    }
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
