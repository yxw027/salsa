#include "salsa/feat.h"

using namespace Eigen;
using namespace xform;

namespace salsa
{

void Features::reserve(const int &N)
{
    zetas.reserve(N);
    depths.reserve(N);
    feat_ids.reserve(N);
    pix.reserve(N);
}

int Features::size() const
{
    return zetas.size();
}

void Features::resize(const int &N)
{
    zetas.resize(N);
    depths.resize(N);
    feat_ids.resize(N);
    pix.resize(N);
}

void Features::clear()
{
    zetas.clear();
    depths.clear();
    feat_ids.clear();
    pix.clear();
}

void Features::rm(const int &idx)
{
    zetas.erase(zetas.begin() + idx);
    depths.erase(depths.begin() + idx);
    feat_ids.erase(feat_ids.begin() + idx);
    pix.erase(pix.begin() + idx);
}

Feat::Feat(double _t0, int _idx, int _kf0, const Vector3d &_z0, const Vector2d &_pix0, double _rho, double _rho_true) :
    t0(_t0), kf0(_kf0), idx0(_idx), z0(_z0), pix0(_pix0), rho(_rho), rho_true(_rho_true), slide_count(0)
{}

void Feat::addMeas(double t, int to_idx, const Matrix2d &cov, const xform::Xformd& x_b2c, const Vector3d &zj, const Vector2d & pixj)
{
    SALSA_ASSERT(to_idx != idx0, "Cannot Point to the same id");
    SALSA_ASSERT(funcs.size() > 0 ? funcs.back().to_idx_ != to_idx : true, "cannot add duplicate feat meas");
    funcs.emplace_back(t, cov, x_b2c, z0, zj, pixj, to_idx);
}

void Feat::moveMeas(double t, int to_idx, const Vector3d &zj)
{
    funcs.back().to_idx_ = to_idx;
    funcs.back().zetaj_ = zj;
    funcs.back().t_ = t;
}

bool Feat::slideAnchor(int new_from_idx, const StateVec &xbuf, const Xformd &x_b2c)
{
    int new_from_kf = xbuf[new_from_idx].kf;
    assert (new_from_kf >= 0);
    if (new_from_kf <= kf0)
        return true; // Don't need to slide, this one is anchored ahead of the slide

    while (idx0 != new_from_idx)
    {
        if (funcs.size() <= 1)
            return false; // can't slide, no future measurements

        Xformd x_I2i(xbuf[idx0].x);
        Xformd x_I2j(xbuf[new_from_idx].x);

        Vector3d p_I2l_i = x_I2i.transforma(x_b2c.transforma(1.0/rho*z0));
        Vector3d zi_j = x_b2c.transformp(x_I2j.transformp(p_I2l_i));
        rho = 1.0/zi_j.norm();
        zi_j.normalize();
        z0 = funcs.front().zetaj_;
        rho_true = funcs.front().rho_true_;
        idx0 = funcs.front().to_idx_;
        pix0 = funcs.front().pixj_;
        t0 = funcs.front().t_;
        funcs.pop_front();
        slide_count++;
    }
    kf0 = new_from_kf;
    SALSA_ASSERT (funcs.front().to_idx_ != idx0, "Trying to slide to zero");
    return true;
}

Vector3d Feat::pos(const StateVec& xbuf, const Xformd& x_b2c)
{
    Xformd x_I2i(xbuf[idx0].x);
    return x_I2i.transforma(x_b2c.transforma(1.0/rho*z0));
}
}
