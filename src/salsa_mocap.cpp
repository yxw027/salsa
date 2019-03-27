#include "salsa/salsa.h"

using namespace std;
using namespace Eigen;
using namespace xform;
using namespace salsa;

void Salsa::mocapCallback(const double &t, const Xformd &z, const Matrix6d &R)
{
    last_callback_ = MOCAP;
    if (current_node_ == -1)
    {
        SD("Initialized Mocap\n");
        initialize(t, z, Vector3d::Zero(), Vector2d::Zero());
        mocap_.emplace_back(dt_m_, x_u2m_, z.arr(), Vector6d::Zero(),
                            R.inverse().llt().matrixL().transpose(),
                            xbuf_head_, current_node_, current_kf_);
        return;
    }
    else
    {
        int prev_x_idx = xbuf_head_;

        finishNode(t, true, true);

        xbuf_[xbuf_head_].kf = current_node_;
        xbuf_[xbuf_head_].x = z.elements();
        Vector6d zdot = (Xformd(xbuf_[xbuf_head_].x) - Xformd(xbuf_[prev_x_idx].x))
                / (xbuf_[xbuf_head_].t - xbuf_[prev_x_idx].t);
        mocap_.emplace_back(dt_m_, x_u2m_, z.arr(), zdot, R.inverse().llt().matrixL().transpose(),
                            xbuf_head_, current_node_, current_kf_);

        solve();

    }
}
