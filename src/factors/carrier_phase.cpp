#include "factors/carrier_phase.h"

using namespace Eigen;
using namespace xform;
using namespace gnss_utils;

namespace salsa
{

CarrierPhaseFunctor::CarrierPhaseFunctor(const double &Xi, const double &Phi0, double Phi1,
                                         Satellite &sat, const Vector3d &p_b2g,
                                         const gnss_utils::GTime g0, const gnss_utils::GTime g1,
                                         int node, int from_idx, int to_idx):
    Phi0_(Phi0),
    Phi1_(Phi1),
    p_b2g_(p_b2g),
    Xi_(Xi),
    node_(node_),
    from_idx_(from_idx),
    to_idx_(to_idx),
    g0_(g0),
    g1_(g1),
    lambda_(sat.LAMBDA_L1)
{
    dPhi_bar_ = Phi1_ - Phi0_;
    sat.computePositionVelocityClock(g0, sat_pos0_, sat_vel0_, sat_clk0_);
    sat.computePositionVelocityClock(g1, sat_pos1_, sat_vel1_, sat_clk1_);
}

template<typename T>
bool CarrierPhaseFunctor::operator()(const T* _x0, const T*_x1, const T* _clk0,
                                     const T* _clk1, const T* _x_e2n, T* _res) const
{
    typedef Matrix<T, 3, 1> Vec3;
    typedef Matrix<T, 2, 1> Vec2;

    Xform<T> x0(_x0);
    Xform<T> x1(_x1);
    Xform<T> x_e2n(_x_e2n);

    Map<const Vec2> clk0(_clk0);
    Map<const Vec2> clk1(_clk1);

    Vec3 p_e2g0 = x_e2n.transforma(x0.t()+x0.rota(p_b2g_));
    Vec3 p_e2g1 = x_e2n.transforma(x1.t()+x1.rota(p_b2g_));

    /// TODO: Use clock bias estimate to adjust satellite positions

    T dist0 = (sat_pos0_ - p_e2g0).norm();
    T dist1 = (sat_pos1_ - p_e2g1).norm();
    T dtau0 = (clk0(0) - sat_clk0_(0));
    T dtau1 = (clk1(0) - sat_clk1_(0));

    T dPhi_hat = (T)gnss_utils::Satellite::C_LIGHT/lambda_ * ((dist1 + dtau1) - (dist0 + dtau0));
    (*_res) = Xi_ * dPhi_hat;
    return true;
}

/// TODO: Switch to analytical, because this jacobian is awful to calculate with forward mode
template bool CarrierPhaseFunctor::operator()<double>(const double* _x0, const double*_x1, const double* _clk0, const double* _clk1, const double* _x_e2n, double* _res) const;
typedef ceres::Jet<double, 25> jactype; /// <- see, 25 inputs, and 1 output?  FAD is awful
template bool CarrierPhaseFunctor::operator()<jactype>(const jactype* _x0, const jactype*_x1, const jactype* _clk0, const jactype* _clk1, const jactype* _x_e2n, jactype* _res) const;


}
