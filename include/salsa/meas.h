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
    virtual ~Base() = default;
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
};

bool basecmp(const Base *a, const Base *b);

struct Gnss : public Base
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Gnss(double _t, const ObsVec& _z);
    Gnss(double _t, ObsVec&& _z);
    ObsVec obs;
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
