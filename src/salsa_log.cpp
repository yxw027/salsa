#include "salsa/salsa.h"
namespace fs = experimental::filesystem;
namespace salsa
{

void Salsa::initLog(const std::string& filename)
{
    get_yaml_node("log_prefix", filename, log_prefix_);
    if (!fs::exists(fs::path(log_prefix_).parent_path()))
        fs::create_directories(fs::path(log_prefix_).parent_path());

    logs_.resize(log::NumLogs);
    logs_[log::CurrentState] = new Logger(log_prefix_ + "CurrentState.log");
    logs_[log::Opt] = new Logger(log_prefix_ + "Opt.log");
    logs_[log::RawRes] = new Logger(log_prefix_+ "RawRes.log");
    logs_[log::FeatRes] = new Logger(log_prefix_ + "FeatRes.log");
    logs_[log::Feat] = new Logger(log_prefix_ + "Feat.log");
    logs_[log::State] = new Logger(log_prefix_ + "State.log");
    logs_[log::CB] = new Logger(log_prefix_ + "CB.log");
    logs_[log::MocapRes] = new Logger(log_prefix_ + "MocapRes.log");
    logs_[log::SatPos] = new Logger(log_prefix_ + "SatPos.log");
    logs_[log::PRangeRes] = new Logger(log_prefix_ + "PRangeRes.log");
}


void Salsa::logState()
{
    logs_[log::State]->log(xbuf_[xbuf_head_].t);
    logs_[log::State]->logVectors(xbuf_[xbuf_head_].x.arr());
    logs_[log::State]->logVectors(xbuf_[xbuf_head_].v);
    logs_[log::State]->logVectors(xbuf_[xbuf_head_].tau);
    logs_[log::State]->log(xbuf_[xbuf_head_].kf);
    logs_[log::State]->log(xbuf_[xbuf_head_].node);

    logs_[log::CB]->log(current_state_.t, last_callback_);
}

void Salsa::logOptimizedWindow()
{
    logs_[log::Opt]->log(STATE_BUF_SIZE, xbuf_head_, xbuf_tail_);
    for (int i = 0; i < STATE_BUF_SIZE; i++)
    {
        logs_[log::Opt]->log(xbuf_[i].node, xbuf_[i].kf, xbuf_[i].t);
        logs_[log::Opt]->logVectors(xbuf_[i].x.arr(), xbuf_[i].v, xbuf_[i].tau);
    }
    logs_[log::Opt]->logVectors(imu_bias_);
}

void Salsa::logCurrentState()
{
    logs_[log::CurrentState]->log(current_state_.t);
    logs_[log::CurrentState]->logVectors(current_state_.x.arr(), current_state_.v, imu_bias_, current_state_.tau);
}


void Salsa::logFeatRes()
{
    logs_[log::FeatRes]->log(current_state_.t);
    FeatMap::iterator ft = xfeat_.begin();
    logs_[log::FeatRes]->log((int)xfeat_.size());
    for (int i = 0; i < nf_; i++)
    {
        if (ft != xfeat_.end())
        {
            logs_[log::FeatRes]->log(ft->first, (int)ft->second.funcs.size(), xbuf_[ft->second.idx0].node);
            FeatDeque::iterator func = ft->second.funcs.begin();
            for (int j = 0; j < N_; j++)
            {
                Vector2d res;
                if (func != ft->second.funcs.end())
                {
                    (*func)(xbuf_[ft->second.idx0].x.data(), xbuf_[func->to_idx_].x.data(),
                            &ft->second.rho, res.data());
                    logs_[log::FeatRes]->log(xbuf_[func->to_idx_].node, xbuf_[func->to_idx_].t);
                    logs_[log::FeatRes]->logVectors(res);
                    func++;
                }
                else
                {
                    res.setConstant(NAN);
                    logs_[log::FeatRes]->log((int)(-1), (double)NAN);
                    logs_[log::FeatRes]->logVectors(res);
                }
            }
            ft++;
        }
        else
        {
            logs_[log::FeatRes]->log((int)-1, ((int)0), (int)-1);
            for (int j = 0; j < N_; j++)
            {
                Vector2d res;
                res.setConstant(NAN);
                logs_[log::FeatRes]->log((int)-1, (double)NAN);
                logs_[log::FeatRes]->logVectors(res);
            }
        }
    }
}

void Salsa::logFeatures()
{
    logs_[log::Feat]->log(current_state_.t);
    FeatMap::iterator ft = xfeat_.begin();
    logs_[log::Feat]->log(xfeat_.size());
    for (int i = 0; i < nf_; i++)
    {
        if (ft != xfeat_.end())
        {
            logs_[log::Feat]->log(ft->first);
            Xformd x_I2i(xbuf_[ft->second.idx0].x);
            Vector3d p_I2l = x_I2i.t() + x_I2i.q().rota(x_u2c_.q().rota(1.0/ft->second.rho * ft->second.z0) + x_u2c_.t());
            logs_[log::Feat]->logVectors(p_I2l);
            logs_[log::Feat]->log(ft->second.rho, ft->second.rho_true, ft->second.slide_count);
            ft++;
        }
        else
        {
            logs_[log::Feat]->log(-1);
            Vector3d p_I2l = Vector3d::Constant(NAN);
            logs_[log::Feat]->logVectors(p_I2l);
            logs_[log::Feat]->log((double)NAN, (double)NAN, (int)-1);
        }
    }
}

void Salsa::logMocapRes()
{
    logs_[log::MocapRes]->log(current_state_.t);
    logs_[log::MocapRes]->log((int)mocap_.size());
    for (int i = 0; i < N_; i++)
    {
        double t;
        Vector6d residual;
        if (i < mocap_.size())
        {
            mocap_[i](xbuf_[mocap_[i].idx_].x.data(), residual.data());
            t = xbuf_[mocap_[i].idx_].t;
        }
        else
        {
            t = NAN;
            residual.setConstant(NAN);
        }
        logs_[log::MocapRes]->log(t);
        logs_[log::MocapRes]->logVectors(residual);
    }
}

void Salsa::logRawGNSSRes()
{
    logs_[log::RawRes]->log(current_state_.t);
    logs_[log::RawRes]->log((int)prange_.size());
    for (int i = 0; i < N_; i++)
    {
        if (i < prange_.size())
        {
            logs_[log::RawRes]->log((int)prange_[i].size());
            for (int j = 0; j < ns_; j++)
            {
                double t;
                Vector2d res;
                if (j < prange_[i].size())
                {
                    int idx = prange_[i][j].idx_;
                    prange_[i][j](xbuf_[idx].x.data(), xbuf_[idx].v.data(), xbuf_[idx].tau.data(),
                            x_e2n_.data(), res.data());
                    t = xbuf_[idx].t;
                }
                else
                {
                    t = NAN;
                    res.setConstant(NAN);
                }
                logs_[log::RawRes]->log(t);
                logs_[log::RawRes]->logVectors(res);
            }
        }
        else
        {
            logs_[log::RawRes]->log((int)-1);
            for (int j = 0; j < ns_; j++)
            {
                double t = NAN;
                Vector2d res = Vector2d::Constant(NAN);
                logs_[log::RawRes]->log(t);
                logs_[log::RawRes]->logVectors(res);
            }
        }
    }
}

void Salsa::logSatPos()
{
    logs_[log::SatPos]->log(current_state_.t, (int)sats_.size());
    for (int i = 0; i < ns_; i++)
    {
        if (i < sats_.size())
        {
            Vector3d pos, vel;
            Vector2d clk;
            GTime now = start_time_ + current_state_.t;
            Vector2d azel = sats_[i].azimuthElevation(now, WSG84::ned2ecef(x_e2n_, xbuf_[xbuf_head_].x.t()));
            sats_[i].computePositionVelocityClock(now, pos, vel, clk);
            logs_[log::SatPos]->log(sats_[i].id_);
            logs_[log::SatPos]->logVectors(pos, vel, clk, azel);
        }
        else
        {
            Matrix<double, 10, 1> padding = Matrix<double, 10, 1>::Constant(NAN);
            logs_[log::SatPos]->log((int)-1);
            logs_[log::SatPos]->logVectors(padding);
        }
    }
    cout << DateTime(start_time_ + current_state_.t) << endl;
}

void Salsa::logPrangeRes()
{
    x_e2n_ = WSG84::x_ecef2ned(WSG84::lla2ecef(Vector3d(40.24650*M_PI/180.0, -111.64623*M_PI/180.0, 1410)));
    logs_[log::PRangeRes]->log(current_state_.t, (int)sats_.size());
    Vector3d rec_pos = WSG84::ned2ecef(x_e2n_, xbuf_[xbuf_head_].x.t());
    Vector3d rec_vel = x_e2n_.rota(xbuf_[xbuf_head_].v);
    Vector2d clk = xbuf_[xbuf_head_].tau;

    GTime& t(filtered_obs_[0].t);
    if (filtered_obs_.size() >= 8)
    {
        Vector8d pp_sol;
        pp_sol.setZero();
        pp_sol.topRows<3>() = x_e2n_.t();
        pointPositioning(t, filtered_obs_, sats_, pp_sol);
        rec_pos = pp_sol.head<3>();
        rec_vel = pp_sol.segment<3>(3);
        clk = pp_sol.tail<2>();

        cout << DateTime(filtered_obs_[0].t) << ": ";
        printLla(WSG84::ecef2lla(rec_pos));
        cout << " vs ";
        printLla(WSG84::ecef2lla(x_e2n_.t()));
        cout << " err = " << WSG84::ecef2ned(x_e2n_, rec_pos).transpose() << endl;
    }
    for (int i = 0; i < ns_; i++)
    {
        if (i < filtered_obs_.size())
        {
            Vector3d z;
            Vector3d zhat = Vector3d::Constant(30);
            Vector3d sat_pos, sat_vel;
            Vector2d sat_clk;


            sats_[filtered_obs_[i].sat_idx].computePositionVelocityClock(t, sat_pos, sat_vel, sat_clk);
            sats_[filtered_obs_[i].sat_idx].computeMeasurement(t, rec_pos, rec_vel, clk, zhat);
            cout << "Sat: " << (int)filtered_obs_[i].sat;
            cout << "  Dist: " << setprecision(10) << (sat_pos-rec_pos).norm();
            cout << "  Zhat: " << setprecision(10) << zhat(0);
            cout << "  Z: " << setprecision(10) << filtered_obs_[i].z(0);
            cout << "  Zhat - Dist: " << setprecision(10) << zhat(0) - (sat_pos-rec_pos).norm();
            cout << "  z - Zhat: " << setprecision(10) << filtered_obs_[i].z(0) - zhat(0);
            cout << "  Az: " << sats_[filtered_obs_[i].sat_idx].azimuthElevation(t, rec_pos)(0)*180.0/M_PI;
            cout << "  El: " << sats_[filtered_obs_[i].sat_idx].azimuthElevation(t, rec_pos)(1)*180.0/M_PI;
            cout << endl;
            range_t gnss_sdr;
            ionoutc_t ion = {true, true,
                             0.1118E-07,-0.7451E-08,-0.5961E-07, 0.1192E-06,
                             0.1167E+06,-0.2294E+06,-0.1311E+06, 0.1049E+07};
//            computeRange(&gnss_sdr, sats_[filtered_obs_[i].sat_idx], &ion, t, rec_pos);
//            Vector3d e_sat = (sat_pos - rec_pos).normalized();
//            zhat << (rec_pos - sat_pos).norm(),
//                    (sat_pos - rec_pos).dot(e_sat),
//                    1.0/Satellite::LAMBDA_L1 * (rec_pos - sat_pos).norm();
//            zhat << gnss_sdr.range, gnss_sdr.rate, gnss_sdr.d;
            z << filtered_obs_[i].z;
            Vector3d res = z - zhat;
//            cout << res.transpose() << endl;
            logs_[log::PRangeRes]->log((int)sats_[filtered_obs_[i].sat_idx].id_);
            logs_[log::PRangeRes]->logVectors(res, z, zhat);
        }
        else
        {
            Vector9d padding = Vector9d::Constant(NAN);
            logs_[log::PRangeRes]->log((int)-1);
            logs_[log::PRangeRes]->logVectors(padding);
        }
    }
    cout << WSG84::ecef2lla(rec_pos).transpose() << endl;
}




}
