#include "salsa/meas.h"

namespace  salsa {
namespace  meas {

Base::Base()
{
    type = BASE;
}

std::string Base::Type() const
{
    switch (type)
    {
    case BASE:
        return "Base";
        break;
    case GNSS:
        return "Gnss";
        break;
    case IMU:
        return "Imu";
        break;
    case MOCAP:
        return "Mocap";
        break;
    case IMG:
        return "Img";
        break;
    case ZERO_VEL:
        return "ZeroVel";
        break;
    }
}

bool basecmp(const Base* a, const Base* b)
{
    return a->t < b->t;
}



Imu::Imu(double _t, const Vector6d &_z, const Matrix6d &_R)
{
    t = _t;
    z = _z;
    R = _R;
    type = IMU;
}

Gnss::Gnss(double _t, const ObsVec& _z)
{
    t = _t;
    type = GNSS;
    obs = _z;
}

Mocap::Mocap(double _t, const xform::Xformd &_z, const Matrix6d &_R) :
    z(_z),
    R(_R)
{
    t = _t;
    type = MOCAP;
}

Img::Img(double _t, const Features &_z, const Eigen::Matrix2d &_R, bool _new_keyframe)
{
    t = _t;
    z = _z;
    R_pix = _R;
    new_keyframe = _new_keyframe;
    type = IMG;
}

ZeroVel::ZeroVel(double _t)
{
    t = _t;
    type = ZERO_VEL;
}

}
}
