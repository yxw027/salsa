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
    int prev_keyframe = xbuf_[xbuf_head_].kf;
    bool new_keyframe = calcNewKeyframeCondition(z);
    if (current_node_ == -1)
    {
        initialize(t, Xformd::Identity(), Vector3d::Zero(), Vector2d::Zero());
    }
    else
    {
        finishNode(t, new_keyframe);
    }

    for (int i = 0; i < z.zetas.size(); i++)
    {
        if (isTrackedFeature(z.feat_ids[i]))
        {
            Feat& ft(xfeat_.at(z.feat_ids[i]));
            if (prev_keyframe != -1)
                ft.addMeas(xbuf_head_, current_node_, x_u2c_, R_pix, z.zetas[i]);
            else
                ft.moveMeas(xbuf_head_, current_node_, z.zetas[i]);
        }
        else if (new_keyframe)
        {
            double rho0 = 1.0/2.0; /// TODO Better depth initialization
            xfeat_.insert(xfeat_.end(), {z.feat_ids[i],
                                         Feat(xbuf_head_, current_kf_, current_node_, z.zetas[i], rho0)});
        }
    }
}

bool Salsa::calcNewKeyframeCondition(const Features &z)
{
    if (current_node_ == -1)
    {
        kf_feat_ = z;
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

    if (kf_parallax_ > kf_parallax_thresh_ || kf_Nmatch_feat_ < kf_feature_thresh_)
    {
        kf_feat_ = z;
        return true;
    }
    else
        return false;
}

void Salsa::cleanUpFeatureTracking(int oldest_node, int oldest_desired_kf)
{

}


}
