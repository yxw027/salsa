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

Img::Img()
{
    type = IMG;
}

}
}
