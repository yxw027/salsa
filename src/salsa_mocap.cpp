#include "salsa/salsa.h"

using namespace std;
using namespace Eigen;
using namespace xform;
using namespace salsa;

void Salsa::mocapCallback(const double &t, const Xformd &z, const Matrix6d &R)
{
    if(disable_mocap_)
        return;
    addMeas(meas::Mocap(t, z, R));
}

void Salsa::mocapUpdate(const meas::Mocap &m)
{
    SD(1, "Mocap Update, t=%.2f", m.t);
    int prev_x_idx = (xbuf_head_ + STATE_BUF_SIZE - 1) % STATE_BUF_SIZE;
    Vector6d zdot = (Xformd(xbuf_[xbuf_head_].x) - Xformd(xbuf_[prev_x_idx].x))
            / (xbuf_[xbuf_head_].t - xbuf_[prev_x_idx].t);
    mocap_.emplace_back(dt_m_, x_b2m_, m.z.arr(), zdot, m.R.inverse().llt().matrixL().transpose(),
                        xbuf_head_, current_node_, current_kf_);
    SALSA_ASSERT((xbuf_[xbuf_head_].type & State::Mocap) == 0, "Cannot double-up with Mocap nodes");
    xbuf_[xbuf_head_].type |= State::Mocap;
}

void Salsa::initializeNodeWithMocap(const meas::Mocap& mocap)
{
    initializeNodeWithImu(); // get velocity from IMU
    xbuf_[xbuf_head_].t = mocap.t;
    xbuf_[xbuf_head_].x = mocap.z * x_b2m_.inverse();
}


void Salsa::initializeStateMocap(const meas::Mocap &m)
{
    SD_S(5, "x_b2m_" << 180.0/M_PI *x_b2m_.q_.euler().transpose() << " z_I2m " << 180.0/M_PI*m.z.q_.euler().transpose());
    initialize(m.t, m.z*x_b2m_.inverse(), Vector3d::Zero(), Vector2d::Zero());
}
