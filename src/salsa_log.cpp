#include "salsa/salsa.h"

namespace fs = std::experimental::filesystem;
using namespace gnss_utils;
using namespace Eigen;
using namespace xform;

namespace salsa
{

void Salsa::initLog(const std::string& filename)
{
    if (log_prefix_.empty())
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
    logs_[log::Imu] = new Logger(log_prefix_ + "Imu.log");
    logs_[log::Xe2n] = new Logger(log_prefix_ + "Xe2n.log");
    logs_[log::Graph] = new Logger(log_prefix_ + "Graph.log");
}


void Salsa::logState()
{
    logs_[log::State]->log(xbuf_[xbuf_head_].t);
    logs_[log::State]->logVectors(xbuf_[xbuf_head_].x.arr());
    logs_[log::State]->logVectors(xbuf_[xbuf_head_].v);
    logs_[log::State]->logVectors(xbuf_[xbuf_head_].tau);
    logs_[log::State]->log(xbuf_[xbuf_head_].kf);
    logs_[log::State]->log(xbuf_[xbuf_head_].node);
}

void Salsa::logOptimizedWindow()
{
    logs_[log::Opt]->log(STATE_BUF_SIZE, xbuf_head_, xbuf_tail_);
    for (int i = 0; i < STATE_BUF_SIZE; i++)
    {
        logs_[log::Opt]->log(xbuf_[i].node, xbuf_[i].kf, xbuf_[i].t);
        logs_[log::Opt]->logVectors(xbuf_[i].x.arr(), xbuf_[i].v, xbuf_[i].tau);
    }
    logs_[log::Opt]->logVectors(imu_bias_, x_b2c_.arr(), x_e2n_.arr());
    if (DEBUGLOGLEVEL <= 4)
        printGraph();
    if (DEBUGLOGLEVEL <= 2)
        printFeat();
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
            for (int j = 0; j < node_window_; j++)
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
            for (int j = 0; j < node_window_; j++)
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
            Vector3d p_I2l = x_I2i.t() + x_I2i.q().rota(x_b2c_.q().rota(1.0/ft->second.rho * ft->second.z0) + x_b2c_.t());
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
    for (int i = 0; i < node_window_; i++)
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
    for (int i = 0; i < node_window_; i++)
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
            Vector2d azel = sats_[i].azimuthElevation(now, WGS84::ned2ecef(x_e2n_, xbuf_[xbuf_head_].x.t()));
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
}

void Salsa::logPrangeRes()
{
    logs_[log::PRangeRes]->log(current_state_.t, (int)sats_.size());
    Vector3d rec_pos = WGS84::ned2ecef(x_e2n_, xbuf_[xbuf_head_].x.t());
    Vector3d rec_vel = x_e2n_.rota(xbuf_[xbuf_head_].v);
    Vector2d clk = xbuf_[xbuf_head_].tau;

    for (int i = 0; i < ns_; i++)
    {
        if (i < filtered_obs_.size())
        {
            GTime& t(filtered_obs_[i].t);
            Vector3d zhat;
            sats_[filtered_obs_[i].sat_idx].computeMeasurement(t, rec_pos, rec_vel, clk, zhat);
            logs_[log::PRangeRes]->log((int)sats_[filtered_obs_[i].sat_idx].id_);
            Vector3d res = zhat - filtered_obs_[i].z;
            logs_[log::PRangeRes]->logVectors(res, filtered_obs_[i].z, zhat);
        }
        else
        {
            Vector9d padding = Vector9d::Constant(NAN);
            logs_[log::PRangeRes]->log((int)-1);
            logs_[log::PRangeRes]->logVectors(padding);
        }
    }
}

void Salsa::logImu()
{
    logs_[log::Imu]->log(current_state_.t);
    logs_[log::Imu]->logVectors(imu_.back().u_);
}

void Salsa::printGraph()
{
    // head: 10   tail: 5   kf_condition: 1  nkf_feat: 50   nft_window: 150  nfeat_meas_window: 250 nfeat_tot:
    // t -- 000.2 -- 000.4 -- 000.5 --
    // K --   0   --       --   1   --
    // N --   0   --   1   --   2   --
    // G --       --   X   --       --
    // F --   X   --       --   X   --
    // M --       --       --       --
    logs_[log::Graph]->file_ << "\n\nhead: " << xbuf_head_ << "\ttail: " << xbuf_tail_;
    logs_[log::Graph]->file_ << "\tkf_cond: " << kf_condition_ << "\tft_kf: " << current_feat_.size();
    logs_[log::Graph]->file_ << "\tft_win: " << xfeat_.size() << "\t  ft_ms_win: " << numTotalFeat() << "\t   ft_tot: " << next_feature_id_;
    logs_[log::Graph]->file_ << "\tparallax: " << kf_parallax_ << "\t  kf_match: " << kf_Nmatch_feat_;


    logs_[log::Graph]->file_ << "\nt -- ";
    int tmp = xbuf_tail_;
    int end = (xbuf_head_+1)%STATE_BUF_SIZE;
    while (tmp != end)
    {
        logs_[log::Graph]->file_ << std::fixed << std::setw(5) << std::setprecision(1) << xbuf_[tmp].t << " -- ";
        tmp = (tmp + 1) % STATE_BUF_SIZE;
    }

    logs_[log::Graph]->file_ << "\nK -- ";
    tmp = xbuf_tail_;
    while (tmp != end)
    {
        if (xbuf_[tmp].kf >= 0)
            logs_[log::Graph]->file_ << std::fixed << std::setw(5) << xbuf_[tmp].kf << " -- ";
        else
            logs_[log::Graph]->file_ << "     " << " -- ";
        tmp = (tmp + 1) % STATE_BUF_SIZE;
    }

    logs_[log::Graph]->file_ << "\nN -- ";
    tmp = xbuf_tail_;
    while (tmp != end)
    {
        logs_[log::Graph]->file_ << std::fixed << std::setw(5) << xbuf_[tmp].node << " -- ";
        tmp = (tmp + 1) % STATE_BUF_SIZE;
    }

    logs_[log::Graph]->file_ << "\nG -- ";
    tmp = xbuf_tail_;
    while (tmp != end)
    {
        if (xbuf_[tmp].type & State::Gnss)
            logs_[log::Graph]->file_ << "  X   -- ";
        else
            logs_[log::Graph]->file_ << "      -- ";
        tmp = (tmp + 1) % STATE_BUF_SIZE;
    }

    logs_[log::Graph]->file_ << "\nF -- ";
    tmp = xbuf_tail_;
    while (tmp != end)
    {
        if (xbuf_[tmp].type & State::Camera)
            logs_[log::Graph]->file_ << std::fixed << std::setw(5) << (int)xbuf_[tmp].n_cam << " -- ";
        else
            logs_[log::Graph]->file_ << "      -- ";
        tmp = (tmp + 1) % STATE_BUF_SIZE;
    }

    logs_[log::Graph]->file_ << "\nM -- ";
    tmp = xbuf_tail_;
    while (tmp != end)
    {
        if (xbuf_[tmp].type & State::Mocap)
            logs_[log::Graph]->file_ << "  X   -- ";
        else
            logs_[log::Graph]->file_ << "      -- ";
        tmp = (tmp + 1) % STATE_BUF_SIZE;
    }
    logs_[log::Graph]->file_ << "\n";
}

void Salsa::printFeat()
{
    // idx:   65   66   67   68   69   70   71  ...
    // 0    |  O |  X |  X |  X |    |    |    |
    // 1    |  O |  X |  X |    |    |    |    |
    // 2    |    |  O |  X |  X |  X |  X |    |
    // 3    |    |    |    |    |  O |  X |    |
    // 4    |    |    |    |    |    |  O |  X |
    // 5    |    |    |    |    |    |  O |  X |
    // 6    |    |    |    |    |    |  O |  X |

    logs_[log::Graph]->file_ << "idx:  ";
    int tmp = xbuf_tail_;
    int end = (xbuf_head_+1)%STATE_BUF_SIZE;
    while (tmp != end)
    {
        logs_[log::Graph]->file_ << std::fixed << std::setw(4) << tmp << " ";
        tmp = (tmp + 1) % STATE_BUF_SIZE;
    }

    logs_[log::Graph]->file_ << "\n";
    for (FeatMap::iterator f = xfeat_.begin(); f != xfeat_.end(); f++)
    {
        logs_[log::Graph]->file_ << std::fixed << std::setw(4) << f->first << " |";
        tmp = xbuf_tail_;
        while (tmp != end)
        {
            if (tmp == f->second.idx0)
            {
                logs_[log::Graph]->file_ << "  O |";
            }
            else
            {
                bool is_to = false;
                for (auto& func : f->second.funcs)
                {
                    if (func.to_idx_ == tmp)
                    {
                        logs_[log::Graph]->file_ << "  X |";
                        is_to = true;
                        break;
                    }
                }
                if (!is_to)
                {
                    logs_[log::Graph]->file_ << "    |";
                }
            }
            tmp = (tmp + 1) % STATE_BUF_SIZE;
        }
        logs_[log::Graph]->file_ << "\n";
    }
}


}
