#include "salsa/state.h"

using namespace Eigen;
using namespace xform;

namespace salsa
{


State::State() :
    t(buf_[0]),
    x(buf_+1),
    p(buf_+1),
    v(buf_+8),
    tau(buf_+11)
{
    for (int i = 0; i < 13; i++)
        buf_[i] = NAN;
    kf = -1;
}

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

Feat::Feat(int _idx, int _kf0, const Vector3d &_z0, double _rho, double _rho_true) :
    kf0(_kf0), idx0(_idx), z0(_z0), rho(_rho), rho_true(_rho_true), slide_count(0)
{}

void Feat::addMeas(int to_idx, const Xformd &x_b2c, const Matrix2d &cov, const Vector3d &zj)
{
    funcs.emplace_back(x_b2c, cov, z0, zj, to_idx);
}

void Feat::moveMeas(int to_idx, const Vector3d &zj)
{
    funcs.back().to_idx_ = to_idx;
    funcs.back().zetaj_ = zj;
}

bool Feat::slideAnchor(int new_from_idx, const StateVec &xbuf, const Xformd &x_b2c)
{
    int new_from_kf = xbuf[new_from_idx].kf;
    assert (new_from_kf >= 0);
    if (new_from_kf <= kf0)
        return true; // Don't need to slide, this one is anchored ahead of the slide
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
    idx0 = new_from_idx;
    kf0 = new_from_kf;
    funcs.pop_front();
    slide_count++;
    return true;
}

Vector3d Feat::pos(const StateVec& xbuf, const Xformd& x_b2c)
{
    Xformd x_I2i(xbuf[idx0].x);
    return x_I2i.transforma(x_b2c.transforma(1.0/rho*z0));
}

Obs::Obs()
{
    sat_idx = -1;
}
bool Obs::operator <(const Obs& other)
{
    return sat_idx < other.sat_idx;
}






}
