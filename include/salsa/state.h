#pragma once
#include <deque>
#include <Eigen/Core>

#include "geometry/xform.h"
#include "factors/feat.h"

using namespace Eigen;
using namespace xform;

namespace salsa
{

class State
{
public:
    double buf_[13];
    int kf;
    int node;
    Xformd x;
    double& t;
    Map<Vector3d> v;
    Map<Vector2d> tau;

    State() :
        t(buf_[0]),
        x(buf_+1),
        v(buf_+8),
        tau(buf_+11)
    {
        for (int i = 0; i < 13; i++)
            buf_[i] = NAN;
        kf = -1;
    }
};

class Features
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int id; // image label
    double t; // time stamp of this image
    std::vector<Vector3d, aligned_allocator<Vector3d>> zetas; // unit vectors to features
    std::vector<double> depths; // feature distances corresponding to feature measurements
    std::vector<int> feat_ids; // feature ids corresonding to pixel measurements

    void reserve(const int& N)
    {
        zetas.reserve(N);
        depths.reserve(N);
        feat_ids.reserve(N);
    }

    void clear()
    {
        zetas.clear();
        depths.clear();
        feat_ids.clear();
    }
};

typedef std::deque<FeatFunctor, aligned_allocator<FeatFunctor>> FeatDeque;

struct Feat
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int kf0;
    int idx0;
    double rho;
    Vector3d z0;
    FeatDeque funcs;

    Feat(int _idx, int _kf0, const Vector3d& _z0, double _rho) :
         kf0(_kf0), idx0(_idx), z0(_z0), rho(_rho) {}

    void addMeas(int to_idx, const Xformd& x_b2c, const Matrix2d& cov, const Vector3d& zj)
    {
        funcs.emplace_back(x_b2c, cov, z0, zj, to_idx);
    }

    void moveMeas(int to_idx, const Vector3d& zj)
    {
        funcs.back().to_idx_ = to_idx;
        funcs.back().zetaj_ = zj;
    }

    bool slideAnchor(int new_from_idx, int new_from_kf,
                     const State* xbuf, const Xformd& x_b2c)
    {
        if (new_from_kf <= kf0)
            return true; // Don't need to slide, this one is anchored ahead of the slide
        if (funcs.size() == 1)
            return false; // can't slide, no future measurements

        Xformd x_I2i(xbuf[idx0].x);
        Xformd x_I2j(xbuf[new_from_idx].x);


        Vector3d p_I2l_i = x_I2i.t() + x_I2i.q().rota(x_b2c.q().rota(1.0/rho * z0) + x_b2c.t());
        Vector3d zi_j = x_b2c.q().rotp(x_I2j.q().rotp(p_I2l_i - x_I2j.t()) - x_b2c.t());
        rho = 1.0/zi_j.norm();
        z0 = funcs.front().zetai_;
        idx0 = new_from_idx;
        kf0 = new_from_kf;
        funcs.pop_front();
        return true;
    }
};

}
