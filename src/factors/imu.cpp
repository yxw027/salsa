#include "factors/imu.h"

using namespace Eigen;
using namespace xform;
using namespace quat;

namespace salsa
{

Vector3d ImuFunctor::gravity_{0, 0, 9.80665};

ImuFunctor::ImuFunctor(const double& _t0, const Vector6d& b0, int from_idx, int from_node)
{
    delta_t_ = 0.0;
    t0_ = _t0;
    b_ = b0;

    n_updates_ = 0;
    y_.setZero();
    y_(Q) = 1.0;
    P_.setZero();
    J_.setZero();

    from_idx_ = from_idx;
    to_idx_ = -1;
    from_node_ = from_node;
}

void ImuFunctor::errorStateDynamics(const Vector10d& y, const Vector9d& dy, const Vector6d& u,
                                    const Vector6d& eta, Vector9d& dydot)
{
    VectorBlock<const Vector9d, 3> dbeta = dy.segment<3>(BETA);
    VectorBlock<const Vector9d, 3> dgamma = dy.segment<3>(GAMMA);

    Quatd gamma(y.data()+GAMMA);
    VectorBlock<const Vector6d, 3> a = u.segment<3>(ACC);
    VectorBlock<const Vector6d, 3> w = u.segment<3>(OMEGA);
    VectorBlock<Vector6d, 3> ba = b_.segment<3>(ACC);
    VectorBlock<Vector6d, 3> bw = b_.segment<3>(OMEGA);

    VectorBlock<const Vector6d, 3> eta_a = eta.segment<3>(ACC);
    VectorBlock<const Vector6d, 3> eta_w = eta.segment<3>(OMEGA);

    dydot.segment<3>(ALPHA) = dbeta;
    dydot.segment<3>(BETA) = -gamma.rota((a - ba).cross(dgamma) + eta_a);
    dydot.segment<3>(GAMMA) = -skew(w - bw)*dgamma - eta_w;
}


// ydot = f(y, u) <-- nonlinear dynamics (reference state)
// A = d(dydot)/d(dy) <-- error state
// B = d(dydot)/d(eta) <-- error state
// Because of the error state, ydot != Ay+Bu
void ImuFunctor::dynamics(const Vector10d& y, const Vector6d& u,
                          Vector9d& ydot, Matrix9d& A, Matrix96&B)
{
    VectorBlock<const Vector10d, 3> alpha = y.segment<3>(ALPHA);
    VectorBlock<const Vector10d, 3> beta = y.segment<3>(BETA);
    Quatd gamma(y.data()+GAMMA);
    VectorBlock<Vector6d, 3> ba = b_.segment<3>(ACC);
    VectorBlock<Vector6d, 3> bw = b_.segment<3>(OMEGA);
    Vector3d a = u.segment<3>(ACC) - ba;
    Vector3d w = u.segment<3>(OMEGA)- bw;

    ydot.segment<3>(ALPHA) = beta;
    ydot.segment<3>(BETA) = gamma.rota(a);
    ydot.segment<3>(GAMMA) = w;

    A.setZero();
    A.block<3,3>(ALPHA, BETA) = I_3x3;
    A.block<3,3>(BETA, GAMMA) = -gamma.R().transpose() * skew(a);
    A.block<3,3>(GAMMA, GAMMA) = -skew(w);

    B.setZero();
    B.block<3,3>(BETA, ACC) = -gamma.R().transpose();
    B.block<3,3>(GAMMA, OMEGA) = -I_3x3;
}


void ImuFunctor::boxplus(const Vector10d& y, const Vector9d& dy, Vector10d& yp)
{
    yp.segment<3>(P) = y.segment<3>(P) + dy.segment<3>(P);
    yp.segment<3>(V) = y.segment<3>(V) + dy.segment<3>(V);
    yp.segment<4>(Q) = (Quatd(y.segment<4>(Q)) + dy.segment<3>(Q)).elements();
}


void ImuFunctor::boxminus(const Vector10d& y1, const Vector10d& y2, Vector9d& d)
{
    d.segment<3>(P) = y1.segment<3>(P) - y2.segment<3>(P);
    d.segment<3>(V) = y1.segment<3>(V) - y2.segment<3>(V);
    d.segment<3>(Q) = Quatd(y1.segment<4>(Q)) - Quatd(y2.segment<4>(Q));
}


void ImuFunctor::integrate(const double& _t, const Vector6d& u, const Matrix6d& cov)
{
    SALSA_ASSERT((cov.array() == cov.array()).all(), "NaN detected in covariance on propagation");
    SALSA_ASSERT((u.array() == u.array()).all(), "NaN detected in covariance on propagation");
    n_updates_++;
    double dt = _t - (t0_ + delta_t_);
    delta_t_ = _t - t0_;
    Vector9d ydot;
    Matrix9d A;
    Matrix96 B;
    Vector10d yp;
    u_ = u;
    cov_ = cov;
    dynamics(y_, u, ydot, A, B);
    boxplus(y_, ydot * dt, yp);
    y_ = yp;

    A = Matrix9d::Identity() + A*dt + 1/2.0 * A*A*dt*dt;
    B = B*dt;

    Matrix9d P_prev = P_;
    P_ = A*P_*A.transpose() + B*cov*B.transpose();
    J_ = A*J_ + B;

    SALSA_ASSERT((A.array() == A.array()).all(), "NaN detected in covariance on propagation");
    SALSA_ASSERT((P_.array() == P_.array()).all(), "NaN detected in covariance on propagation");
}


void ImuFunctor::estimateXj(const double* _xi, const double* _vi, double* _xj, double* _vj) const
{
    VectorBlock<const Vector10d, 3> alpha = y_.segment<3>(ALPHA);
    VectorBlock<const Vector10d, 3> beta = y_.segment<3>(BETA);
    Quatd gamma(y_.data()+GAMMA);
    Xformd xi(_xi);
    Xformd xj(_xj);
    Map<const Vector3d> vi(_vi);
    Map<Vector3d> vj(_vj);

    xj.t_ = xi.t_ + xi.q_.rota(vi*delta_t_) + 1/2.0 * gravity_*delta_t_*delta_t_ + xi.q_.rota(alpha);
    xj.t_ = xi.t_ + xi.q_.rota(vi*delta_t_) + 1/2.0 * gravity_*delta_t_*delta_t_ + xi.q_.rota(alpha);
    vj = gamma.rotp(vi + xi.q_.rotp(gravity_)*delta_t_ + beta);
    xj.q_ = xi.q_ * gamma;
}


void ImuFunctor::finished(int to_idx)
{
  to_idx_ = to_idx;
  if (n_updates_ < 2)
  {
    P_ = P_ + Matrix9d::Identity() * 1e-10;
  }
  Xi_ = P_.inverse().llt().matrixL().transpose();
  SALSA_ASSERT((Xi_.array() == Xi_.array()).all(), "NaN detected in IMU information matrix");
}

template<typename T>
bool ImuFunctor::operator()(const T* _xi, const T* _xj, const T* _vi, const T* _vj,
                            const T* _b, T* residuals) const
{
    typedef Matrix<T,3,1> VecT3;
    typedef Matrix<T,6,1> VecT6;
    typedef Matrix<T,9,1> VecT9;
    typedef Matrix<T,10,1> VecT10;

    Xform<T> xi(_xi);
    Xform<T> xj(_xj);
    Map<const VecT3> vi(_vi);
    Map<const VecT3> vj(_vj);
    Map<const VecT6> b(_b);

    VecT9 dy = J_ * (b - b_);
    VecT10 y;
    y.template segment<6>(0) = y_.template segment<6>(0) + dy.template segment<6>(0);
    Quat<T> q_dy;
    q_dy.arr_(0) = (T)1.0;
    q_dy.arr_.template segment<3>(1) = 0.5 * dy.template segment<3>(GAMMA);
    y.template segment<4>(6) = (Quatd(y_.template segment<4>(6)).otimes<T,T>(q_dy)).elements();
//        y.template segment<4>(6) = (Quatd(y_.template segment<4>(6)).otimes<T,T>(Quat<T>::exp(dy.template segment<3>(6)))).elements();

    Map<VecT3> alpha(y.data()+ALPHA);
    Map<VecT3> beta(y.data()+BETA);
    Quat<T> gamma(y.data()+GAMMA);
    Map<VecT9> r(residuals);

    r.template block<3,1>(ALPHA, 0) = xi.q_.rotp(xj.t_ - xi.t_ - 1/2.0*gravity_*delta_t_*delta_t_) - vi*delta_t_ - alpha;
    r.template block<3,1>(BETA, 0) = gamma.rota(vj) - vi - xi.q_.rotp(gravity_)*delta_t_ - beta;
    r.template block<3,1>(GAMMA, 0) = (xi.q_.inverse() * xj.q_) - gamma;

    r = Xi_ * r;

    return true;
}
template bool ImuFunctor::operator()<double>(const double* _xi, const double* _xj, const double* _vi, const double* _vj, const double* _b, double* residuals) const;
typedef ceres::Jet<double, 26> jactype;
template bool ImuFunctor::operator()<jactype>(const jactype* _xi, const jactype* _xj, const jactype* _vi, const jactype* _vj, const jactype* _b, jactype* residuals) const;


}
