#include "salsa/salsa.h"

using namespace std;
using namespace Eigen;
using namespace xform;
using namespace quat;

namespace salsa
{

bool Salsa::isTrackedFeature(int id) const
{
    FeatMap::const_iterator it = xfeat_.begin();
    while (it != xfeat_.end() && id != it->first)
        ++it;
    return it != xfeat_.end();
}

void Salsa::imageCallback(const double& t, const ImageFeat& z,
                          const Matrix2d& R_pix, const Matrix1d& R_depth)
{
  if (sim_KLT_)
  {
    if (current_img_.empty())
        current_img_.create(cv::Size(cam_.image_size_(0), cam_.image_size_(1)), 0);
    current_img_ = 0;
    for (auto pix : z.pixs)
    {
        int d = 2;
        cv::Point2d xp(pix(0)+d, pix(1));
        cv::Point2d xm(pix(0)-d, pix(1));
        cv::Point2d yp(pix(0), pix(1)+d);
        cv::Point2d ym(pix(0), pix(1)-d);
        cv::line(current_img_, xp, xm, 255);
        cv::line(current_img_, yp, ym, 255);
    }
    imageCallback(t, current_img_, R_pix);
  }
  else
  {
    Features zfeat;
    zfeat.id = z.id;
    zfeat.t = z.t;
    zfeat.feat_ids = z.feat_ids;
    zfeat.depths = z.depths;
    zfeat.zetas.reserve(z.pixs.size());
    zfeat.pix.reserve(z.pixs.size());
    for (auto pix : z.pixs)
    {
      zfeat.zetas.emplace_back(cam_.invProj(pix, 1.0));
      zfeat.pix.emplace_back(pix.x(), pix.y());
    }
    bool new_keyframe = calcNewKeyframeCondition(zfeat);
    imageCallback(t, zfeat, R_pix, new_keyframe);
  }
}

void Salsa::imageCallback(const double& t, const Features& z, const Matrix2d& R_pix,
                          bool new_keyframe)
{
    last_callback_ = IMG;

    if (current_node_ == -1)
    {
        initialize(t, current_state_.x, current_state_.v, Vector2d::Zero());
    }
    else
    {
        endInterval(t);
    }

    for (auto& ft : xfeat_)
        ft.second.updated_in_last_image_ = false;

    for (int i = 0; i < z.zetas.size(); i++)
    {
        if (isTrackedFeature(z.feat_ids[i]))
        {
            Feat& ft(xfeat_.at(z.feat_ids[i]));
            if (ft.funcs.size() == 0 || (xbuf_[ft.funcs.back().to_idx_].kf >= 0))
                ft.addMeas(xbuf_head_, R_pix, z.zetas[i]);
            else
                ft.moveMeas(xbuf_head_, z.zetas[i]);
            ft.updated_in_last_image_ = true;
            ft.funcs.back().rho_true_ = 1.0/z.depths[i];
        }
        else if (new_keyframe)
        {
            double rho0 = 0.1;
            if (use_measured_depth_)
                rho0 = 1.0/z.depths[i];
            xfeat_.insert({z.feat_ids[i], Feat(xbuf_head_, current_kf_+1, z.zetas[i], rho0, 1.0/z.depths[i])});
        }
    }

    if (new_keyframe)
    {
        setNewKeyframe();
        startNewInterval(t);
    }
    rmLostFeatFromKf();
    solve();
}

bool Salsa::calcNewKeyframeCondition(const Features &z)
{
    if (current_kf_ == -1)
    {
        kf_condition_ = FIRST_KEYFRAME;
        kf_feat_ = z;
        kf_num_feat_ = z.feat_ids.size();
        return true;
    }

    kf_parallax_ = 0;
    kf_Nmatch_feat_ = 0;
    Quatd q_I2i(lastKfState().x.q());
    Quatd q_I2j(current_state_.x.q());
    Quatd q_b2c(x_u2c_.q());
    Quatd q_cj2ci = q_b2c.inverse() * q_I2j.inverse() * q_I2i * q_b2c;
    Matrix3d R_cj2ci = q_cj2ci.R();
    int ni = 0;
    int nj = 0;
    while (ni < kf_feat_.zetas.size() && nj < z.zetas.size())
    {
        int idi = kf_feat_.feat_ids[ni];
        int idj = z.feat_ids[nj];
        if (idi == idj)
        {
            Vector3d zihat = R_cj2ci * z.zetas[nj];
            Vector2d lihat =  cam_.proj(zihat);
            Vector2d li(kf_feat_.pix[ni].x, kf_feat_.pix[ni].y);
            double err = (lihat - li).norm();
            kf_parallax_ += err;
            ni++;
            nj++;
            kf_Nmatch_feat_++;
        }
        else if (idi < idj)
            ni++;
        else if (idi > idj)
            nj++;
    }
    kf_parallax_ /= kf_Nmatch_feat_;

    if (kf_parallax_ > kf_parallax_thresh_)
    {
        kf_condition_ = TOO_MUCH_PARALLAX;
        kf_feat_ = z;
        kf_num_feat_ = z.feat_ids.size();
        return true;
    }
    else if(kf_Nmatch_feat_ <= std::round(kf_feature_thresh_ * kf_num_feat_) + 0.001)
    {
        kf_condition_ = INSUFFICIENT_MATCHES;
        kf_feat_ = z;
        kf_num_feat_ = z.feat_ids.size();
        return true;
    }
    else
        return false;
}

void Salsa::cleanUpFeatureTracking()
{
    // Find the oldest keyframe still in the window
    int tmp = xbuf_tail_;
    while (tmp != xbuf_head_ && xbuf_[tmp].kf < 0)
        tmp = (tmp + 1) % STATE_BUF_SIZE;

    FeatMap::iterator fit = xfeat_.begin();
    while (fit != xfeat_.end())
    {
        if (!fit->second.slideAnchor(tmp, xbuf_, x_u2c_))
            fit = xfeat_.erase(fit);
        else
            fit++;
    }
}

void Salsa::createNewKeyframe()
{
    filterFeaturesTooClose();
    collectNewfeatures();
    kf_feat_ = current_feat_;
    kf_num_feat_ = kf_feat_.size();
    current_img_.copyTo(kf_img_);
}

void Salsa::rmLostFeatFromKf()
{
    FeatMap::iterator ftpair = xfeat_.begin();
    while (ftpair != xfeat_.end())
    {
        Feat& ft(ftpair->second);
        if (ft.funcs.size() == 0)
        {
            ftpair++;
            continue;
        }
        else if (!ft.updated_in_last_image_
           && (ft.funcs.back().to_idx_ == xbuf_head_ || xbuf_[ft.funcs.back().to_idx_].kf < 0))
        {
            ft.funcs.pop_back();
            if (ft.funcs.size() == 0)
            {
                SD(2, "removing feature %d", ftpair->first);
                ftpair = xfeat_.erase(ftpair);
                continue;
            }
        }
        ftpair++;
    }
}



}
