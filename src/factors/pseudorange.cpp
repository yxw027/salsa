#include "factors/pseudorange.h"

using namespace Eigen;
using namespace xform;
using namespace gnss_utils;

#define Tr transpose()

namespace salsa
{

PseudorangeFunctor::PseudorangeFunctor()
{
    active_ = false;
    sw = 1.0;
}

void PseudorangeFunctor::init(const GTime& _t, const Vector2d& _rho, Satellite& sat,
                              const Vector3d& _rec_pos_ecef, const Matrix2d& cov,
                              const Vector3d &_p_b2g, int node, int idx)
{
    node_ = node;
    idx_ = idx;
    p_b2g = _p_b2g;

    // We don't have ephemeris for this satellite, we can't do anything with it yet
    if (sat.eph_.A == 0)
        return;

    t = _t;
    rho = _rho;
    rec_pos = _rec_pos_ecef;
    sat.computePositionVelocityClock(t, sat_pos, sat_vel, sat_clk);

    Vector3d los_to_sat = sat_pos - rec_pos;
    double range = (sat_pos - rec_pos).norm();
    sagnac_comp = Satellite::OMEGA_EARTH * (sat_pos.x()*rec_pos.y() - sat_pos.y()*rec_pos.x())/Satellite::C_LIGHT;
    range += sagnac_comp;
    double tau = range / Satellite::C_LIGHT;
    sat_pos -= sat_vel * tau;

    // Earth rotation correction. The change in velocity can be neglected.
    Vector3d earth_rot = sat_pos.cross(e_z*Satellite::OMEGA_EARTH * tau);
    sat_pos += earth_rot;

    // pre-calculate the (basically) constant adjustments to pseudorange we will have to make
    los_to_sat = sat_pos - rec_pos;
    sagnac_comp = Satellite::OMEGA_EARTH * (sat_pos.x()*rec_pos.y() - sat_pos.y()*rec_pos.x())/Satellite::C_LIGHT;
    Vector2d az_el = sat.los2azimuthElevation(rec_pos, los_to_sat);
    Vector3d lla = WGS84::ecef2lla(rec_pos);
    ion_delay = sat.ionosphericDelay(t, lla, az_el);
    trop_delay = sat.troposphericDelay(t, lla, az_el);
    sat_clk = sat.clk;
    Xi_ = cov.inverse().llt().matrixL().transpose();
    active_ = true;
}

//#include <iostream>
//#define DBG(x) printf(#x": %6.6f\n", x); std::cout << std::flush
template <typename T>
bool PseudorangeFunctor::operator()(const T* _x, const T* _v, const T* _clk,
                                    const T* _x_e2n, T* _res) const
{
    typedef Matrix<T,3,1> Vec3;
    typedef Matrix<T,2,1> Vec2;


    Xform<T> x(_x);
    Map<const Vec3> v_b(_v);
    Map<const Vec2> clk(_clk);
    Xform<T> x_e2n(_x_e2n);
    Map<Vec2> res(_res);


    Vec3 v_ECEF = x_e2n.rota(x.rota(v_b));
    Vec3 p_ECEF = x_e2n.transforma(x.t()+x.rota(p_b2g));
    Vec3 los_to_sat = sat_pos - p_ECEF;

    Vec2 rho_hat;
    T range = los_to_sat.norm() + sagnac_comp;
    rho_hat(0) = range + ion_delay + trop_delay + (T)Satellite::C_LIGHT*(clk(0)- sat_clk(0));
    rho_hat(1) = (sat_vel - v_ECEF).dot(los_to_sat/range)
            + Satellite::OMEGA_EARTH / Satellite::C_LIGHT * (sat_vel[1]*p_ECEF[0] + sat_pos[1]*v_ECEF[0] - sat_vel[0]*p_ECEF[1] - sat_pos[0]*v_ECEF[1])
                   + (T)Satellite::C_LIGHT*(clk(1) - sat_clk(1));

    res = Xi_ * (rho_hat - rho);

    /// TODO: Check if time or rec_pos have deviated too much
    /// and re-calculate ion_delay and earth rotation effect

    return true;
}

template bool PseudorangeFunctor::operator()<double>(const double* _x, const double* _v, const double* _clk, const double* _x_e2n, double* _res) const;
typedef ceres::Jet<double, 19> jactype;
template bool PseudorangeFunctor::operator()<jactype>(const jactype* _x, const jactype* _v, const jactype* _clk, const jactype* _x_e2n, jactype* _res) const;

PseudorangeFactor::PseudorangeFactor(const PseudorangeFunctor *functor) :
    ptr(functor)
{}

bool PseudorangeFactor::Evaluate(const double * const *parameters, double *residuals, double **jacobians) const
{
    Xformd x(parameters[0]);
    Map<const Vector3d> v_b(parameters[1]);
    Map<const Vector2d> clk(parameters[2]);
    Xformd x_e2n(parameters[3]);
    Map<Vector2d> res(residuals);
    const double& C(Satellite::C_LIGHT);
    const double& W(Satellite::OMEGA_EARTH);

    Vector3d v_ECEF = x_e2n.rota(x.rota(v_b));
    Vector3d p_ECEF = x_e2n.transforma(x.t()+x.rota(ptr->p_b2g));
    Vector3d los = ptr->sat_pos - p_ECEF;
    double los_norm = los.stableNorm();

    Vector2d rho_hat;
    double range = los_norm + ptr->sagnac_comp;
    rho_hat(0) = range
                 + ptr->ion_delay
                 + ptr->trop_delay
                 + C*(clk(0)- ptr->sat_clk(0));
    rho_hat(1) = (ptr->sat_vel - v_ECEF).dot(los/range)
                 + W / C * (ptr->sat_vel[1]*ptr->rec_pos[0] + ptr->sat_pos[1]*v_ECEF[0]
                            - ptr->sat_vel[0]*ptr->rec_pos[1] - ptr->sat_pos[0]*v_ECEF[1])
                 + C*(clk(1) - ptr->sat_clk(1));

    res = ptr->Xi_ * (rho_hat -ptr->rho);

    if (jacobians)
    {
        Vector3d e = los.Tr/los_norm;
        Matrix3d RE2I = x_e2n.q().R();
        Matrix3d RI2b = x.q().R();
        const double& xi0(ptr->Xi_(0,0));
        const double& xi1(ptr->Xi_(1,1));
//        Matrix3d Z = (los*e.Tr - I_3x3*norm)/(norm*norm);

        if (jacobians[0])
        {
            Map<Matrix<double, 2, 7, RowMajor>> dres_dx(jacobians[0]);
            Matrix<double, 3, 4> dqdd;
            quat::Quatd& q(x.q());
            dqdd << -q.x()*2.0,  q.w()*2.0,  q.z()*2.0, -q.y()*2.0,
                    -q.y()*2.0, -q.z()*2.0,  q.w()*2.0,  q.x()*2.0,
                    -q.z()*2.0,  q.y()*2.0, -q.x()*2.0,  q.w()*2.0;

            dres_dx.block<1,3>(0,0) = xi0 * (-e.Tr * RE2I.Tr);
            dres_dx.block<1,4>(0,3) = xi0 * (e.Tr * RE2I.Tr * RI2b.Tr * skew(ptr->p_b2g) * dqdd);
            dres_dx.block<1,3>(1,0).setZero(); //approx
            dres_dx.block<1,4>(1,3) = xi1 * (e.Tr*RE2I.Tr*RI2b.Tr*skew(v_b) * dqdd); // approx
        }

        if (jacobians[1])
        {
            Map<Matrix<double, 2, 3, RowMajor>> dres_dv(jacobians[1]);
            dres_dv.topRows<1>().setZero();
            dres_dv.bottomRows<1>() = xi1*(-e.Tr*RE2I.Tr*RI2b.Tr);
        }

        if (jacobians[2])
        {
            Map<Matrix<double, 2, 2, RowMajor>> dres_dclk(jacobians[2]);
            dres_dclk << xi0*C, 0,
                         0, xi1*C;

        }

        if (jacobians[3])
        {
            Map<Matrix<double, 2, 7, RowMajor>> dres_dx2en(jacobians[3]);
            Matrix<double, 3, 4> dqdd;
            quat::Quatd& q(x_e2n.q());
            dqdd << -q.x()*2.0,  q.w()*2.0,  q.z()*2.0, -q.y()*2.0,
                    -q.y()*2.0, -q.z()*2.0,  q.w()*2.0,  q.x()*2.0,
                    -q.z()*2.0,  q.y()*2.0, -q.x()*2.0,  q.w()*2.0;

            dres_dx2en.block<1,3>(0,0) = xi0 * (-e.Tr);
            dres_dx2en.block<1,4>(0,3) = xi0 * (e.Tr*RE2I.Tr*skew(x.t() + RI2b.Tr*ptr->p_b2g) * dqdd);
            dres_dx2en.block<1,3>(1,0).setZero(); //approx
            dres_dx2en.block<1,4>(1,3) = xi1 * (e.Tr*RE2I.Tr*skew(RI2b.Tr*v_b)*dqdd); //approx
        }
    }

    return true;
}


}
