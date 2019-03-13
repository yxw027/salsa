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

}
