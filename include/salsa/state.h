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
    int node0;
    double rho;
    Vector3d z0;
    FeatDeque funcs;

    Feat(int _idx, int _kf0, int _node0, const Vector3d& _z0, double _rho) :
        kf0(_kf0), idx0(_idx), node0(_node0), z0(_z0), rho(_rho) {}

    void addMeas(int to_idx, int to_node,
                 const Xformd& x_b2c, const Matrix2d& cov, const Vector3d& zj)
    {
        funcs.emplace_back(x_b2c, cov, z0, zj, to_idx);
    }

    void moveMeas(int to_idx, int to_node, const Vector3d& zj)
    {
        funcs.back().to_idx_ = to_idx;
        funcs.back().to_idx_ = to_node;
        funcs.back().zetaj_ = zj;
    }
};
typedef std::map<int, Feat, std::less<int>,
                 aligned_allocator<std::pair<const int, Feat>>> FeatMap;

}
