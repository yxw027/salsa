#include <Eigen/Core>

#include <geometry/xform.h>
#include <gnss_utils/gtime.h>

#include "salsa/state.h"

namespace salsa
{

namespace meas
{
struct Base
{
    Base();
    enum
    {
        BASE,
        GNSS,
        IMU,
        MOCAP,
        IMG
    };
    double t;
    int type;
    bool operator < (const Base& other);
};

struct Gnss : public Base
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Gnss();
    gnss_utils::GTime gtime;
    Eigen::Vector3d z;
    Eigen::Matrix3d R;
};

struct Imu : public Base
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Imu(double _t, const Vector6d& _z, const Matrix6d& _R);
    Vector6d z;
    Matrix6d R;
};

struct Mocap : public Base
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Mocap();
    xform::Xformd z;
    Matrix6d R;
};

struct Img : public Base
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Img(double _t, Features &&_z, const Eigen::Matrix2d& _R, bool _new_keyframe);
    Img(double _t, const Features &_z, const Eigen::Matrix2d& _R, bool _new_keyframe);
    Features z;
    Eigen::Matrix2d R_pix;
    bool new_keyframe;
};
}


}
