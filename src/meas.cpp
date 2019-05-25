#include "salsa/meas.h"

namespace  salsa {
namespace  meas {

Base::Base()
{
    type = BASE;
}

bool Base::operator <(const Base& other)
{
    return t < other.t;
}

Imu::Imu(double _t, const Vector6d &_z, const Matrix6d &_R)
{
    t = _t;
    z = _z;
    R = _R;
    type = IMU;
}

Gnss::Gnss()
{
    type = GNSS;
}

Mocap::Mocap()
{
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
Img::Img(double _t, Features &&_z, const Eigen::Matrix2d &_R, bool _new_keyframe) :
    z(std::move(_z))
{
    t = _t;
    R_pix = _R;
    new_keyframe = _new_keyframe;
    type = IMG;
}

}
}
