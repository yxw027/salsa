#include "salsa/salsa.h"

using namespace std;
using namespace Eigen;
using namespace xform;

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
    Features zfeat;
    zfeat.id = z.id;
    zfeat.t = z.t;
    zfeat.feat_ids = z.feat_ids;
    zfeat.depths = z.depths;
    zfeat.zetas.reserve(z.pixs.size());
    for (auto pix : z.pixs)
    {
        zfeat.zetas.emplace_back(cam_.invProj(pix, 1.0));
    }
    imageCallback(t, zfeat, R_pix, R_depth);
}

void Salsa::imageCallback(const double& t, const Features& z, const Matrix2d& R_pix,
                          const Matrix1d& R_depth)
{
    last_callback_ = IMG;
//    FeatMap::iterator fit = xfeat_.begin();
//    while (fit != xfeat_.end())
//    {
//        if (fit->second.rho < 0.1)
//            fit = xfeat_.erase(fit);
//        else
//            fit++;
//    }
    int prev_keyframe = xbuf_[xbuf_head_].kf;
    bool new_keyframe = calcNewKeyframeCondition(z);
    if (current_node_ == -1)
        initialize(t, current_state_.x, current_state_.v, Vector2d::Zero());
    else
        finishNode(t, prev_keyframe != -1, new_keyframe);

    for (auto& ft : xfeat_)
        ft.second.updated_in_last_image_ = false;

    for (int i = 0; i < z.zetas.size(); i++)
    {
        if (isTrackedFeature(z.feat_ids[i]))
        {
            Feat& ft(xfeat_.at(z.feat_ids[i]));
            if (prev_keyframe != -1)
                ft.addMeas(xbuf_head_, x_u2c_, R_pix, z.zetas[i]);
            else
                ft.moveMeas(xbuf_head_, z.zetas[i]);
            ft.updated_in_last_image_ = true;
            ft.funcs.back().rho_true_ = 1.0/z.depths[i];
        }
        else if (new_keyframe)
        {
//            double rho0 = 1.0/z.depths[i]; /// TODO Better depth initialization
            double rho0 = 0.1;
            xfeat_.insert({z.feat_ids[i], Feat(xbuf_head_, current_kf_, z.zetas[i], rho0, 1.0/z.depths[i])});
        }
    }       
    rmLostFeatFromKf();
    solve();
}

bool Salsa::calcNewKeyframeCondition(const Features &z)
{
    if (current_node_ == -1)
    {
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
            /// TODO - calculate parallax wrt zetas
            Vector3d zihat = R_cj2ci * z.zetas[nj];
            Vector2d lihat =  cam_.proj(zihat);
            Vector3d zi = kf_feat_.zetas[ni];
            Vector2d li = cam_.proj(zi);
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
    else if(kf_Nmatch_feat_ < std::round(kf_feature_thresh_ * kf_num_feat_))
    {
        kf_condition_ = INSUFFICIENT_MATCHES;
        kf_feat_ = z;
        kf_num_feat_ = z.feat_ids.size();
        return true;
    }
    else
        return false;
}

void Salsa::cleanUpFeatureTracking(int new_from_idx, int oldest_desired_kf)
{
    FeatMap::iterator fit = xfeat_.begin();
    while (fit != xfeat_.end())
    {
        if (!fit->second.slideAnchor(new_from_idx, oldest_desired_kf, xbuf_, x_u2c_))
            fit = xfeat_.erase(fit);
        else
            fit++;
    }
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
        else if (ft.funcs.back().to_idx_ == xbuf_head_ && !ft.updated_in_last_image_)
        {
            ft.funcs.pop_back();
            if (ft.funcs.size() == 0)
            {
                ftpair = xfeat_.erase(ftpair);
                continue;
            }
        }
        ftpair++;
    }
}

}
