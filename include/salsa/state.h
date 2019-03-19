#pragma once
#include <Eigen/Core>

#include "geometry/xform.h"

using namespace Eigen;
using namespace xform;

namespace salsa
{

class State
{
public:
    enum
    {
      UNINITIALIZED,
      IS_KEYFRAME,
      NOT_KEYFRAME
    };
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
        kf = UNINITIALIZED;
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

}
