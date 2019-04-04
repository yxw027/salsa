#include "salsa/salsa.h"

namespace salsa
{


void Salsa::logState()
{
    if (state_log_)
    {
        state_log_->log(xbuf_[xbuf_head_].t);
        state_log_->logVectors(xbuf_[xbuf_head_].x.arr());
        state_log_->logVectors(xbuf_[xbuf_head_].v);
        state_log_->logVectors(xbuf_[xbuf_head_].tau);
        state_log_->log(xbuf_[xbuf_head_].kf);
        state_log_->log(xbuf_[xbuf_head_].node);
    }
    if (cb_log_)
    {
        cb_log_->log(current_state_.t, last_callback_);
    }
}

void Salsa::logOptimizedWindow()
{
    if (opt_log_)
    {
        opt_log_->log(STATE_BUF_SIZE, xbuf_head_, xbuf_tail_);
        for (int i = 0; i < STATE_BUF_SIZE; i++)
        {
            opt_log_->log(xbuf_[i].node, xbuf_[i].kf, xbuf_[i].t);
            opt_log_->logVectors(xbuf_[i].x.arr(), xbuf_[i].v, xbuf_[i].tau);
        }
        opt_log_->logVectors(imu_bias_);
    }
}

void Salsa::logCurrentState()
{
    if (current_state_log_)
    {
        current_state_log_->log(current_state_.t);
        current_state_log_->logVectors(current_state_.x.arr(), current_state_.v, imu_bias_, current_state_.tau);
    }
}


void Salsa::logFeatRes()
{
    feat_res_log_->log(current_state_.t);
    FeatMap::iterator ft = xfeat_.begin();
    feat_res_log_->log((int)xfeat_.size());
    for (int i = 0; i < nf_; i++)
    {
        if (ft != xfeat_.end())
        {
            feat_res_log_->log(ft->first, (int)ft->second.funcs.size(), xbuf_[ft->second.idx0].node);
            FeatDeque::iterator func = ft->second.funcs.begin();
            for (int j = 0; j < N_; j++)
            {
                Vector2d res;
                if (func != ft->second.funcs.end())
                {
                    (*func)(xbuf_[ft->second.idx0].x.data(), xbuf_[func->to_idx_].x.data(),
                            &ft->second.rho, res.data());
                    feat_res_log_->log(xbuf_[func->to_idx_].node, xbuf_[func->to_idx_].t);
                    feat_res_log_->logVectors(res);
                    func++;
                }
                else
                {
                    res.setConstant(NAN);
                    feat_res_log_->log((int)(-1), (double)NAN);
                    feat_res_log_->logVectors(res);
                }
            }
            ft++;
        }
        else
        {
            feat_res_log_->log((int)-1, ((int)0), (int)-1);
            for (int j = 0; j < N_; j++)
            {
                Vector2d res;
                res.setConstant(NAN);
                feat_res_log_->log((int)-1, (double)NAN);
                feat_res_log_->logVectors(res);
            }
        }
    }
}

void Salsa::logFeatures()
{
    feat_log_->log(current_state_.t);
    FeatMap::iterator ft = xfeat_.begin();
    feat_log_->log(xfeat_.size());
    for (int i = 0; i < nf_; i++)
    {
        if (ft != xfeat_.end())
        {
            feat_log_->log(ft->first);
            Xformd x_I2i(xbuf_[ft->second.idx0].x);
            Vector3d p_I2l = x_I2i.t() + x_I2i.q().rota(x_u2c_.q().rota(1.0/ft->second.rho * ft->second.z0) + x_u2c_.t());
            feat_log_->logVectors(p_I2l);
            feat_log_->log(ft->second.rho, ft->second.rho_true, ft->second.slide_count);
            ft++;
        }
        else
        {
            feat_log_->log(-1);
            Vector3d p_I2l = Vector3d::Constant(NAN);
            feat_log_->logVectors(p_I2l);
            feat_log_->log((double)NAN, (double)NAN, (int)-1);
        }
    }
}

void Salsa::logMocapRes()
{
    if (mocap_res_log_)
    {
        mocap_res_log_->log(current_state_.t);
        mocap_res_log_->log((int)mocap_.size());
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
            mocap_res_log_->log(t);
            mocap_res_log_->logVectors(residual);
        }
    }
}

void Salsa::logRawGNSSRes()
{
    if (raw_gnss_res_log_)
    {
        raw_gnss_res_log_->log(current_state_.t);
        raw_gnss_res_log_->log((int)prange_.size());
        for (int i = 0; i < N_; i++)
        {
            if (i < prange_.size())
            {
                raw_gnss_res_log_->log((int)prange_[i].size());
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
                    raw_gnss_res_log_->log(t);
                    raw_gnss_res_log_->logVectors(res);
                }
            }
            else
            {
                raw_gnss_res_log_->log((int)-1);
                for (int j = 0; j < ns_; j++)
                {
                    double t = NAN;
                    Vector2d res = Vector2d::Constant(NAN);
                    raw_gnss_res_log_->log(t);
                    raw_gnss_res_log_->logVectors(res);
                }
            }
        }
    }
}




}
