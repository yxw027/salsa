#include <gtest/gtest.h>

#include "gnss_utils/gtime.h"
#include "gnss_utils/satellite.h"
#include "salsa/test_common.h"

#include "factors/pseudorange.h"
#include "factors/xform.h"

using namespace salsa;
using namespace gnss_utils;
using namespace Eigen;

TEST (PrangeFactor, ANvsADRes)
{
    PseudorangeFunctor func;
    PseudorangeFactor an(&func);
    PseudorangeFactorAD ad(new FunctorShield<PseudorangeFunctor>(&func));

    Satellite sat(3, 0);
    sat.readFromRawFile(SALSA_DIR"/sample/eph.dat");

    Vector3d p_b2g(0, 0, 0.3);
    GTime log_start = GTime::fromUTC(1541454646,  0.993);
    Vector3d p_E2b(-1798904.13, -4532227.1 ,  4099781.95);
    xform::Xformd x_e2n = WGS84::x_ecef2ned(p_E2b);
    Vector3d p_E2g = WGS84::ned2ecef(x_e2n, p_b2g);
    Vector3d vel(1, 2,3);
    Vector2d clk(1e-5, 1e-8);
    double s(0.5);

    Vector3d z;
    sat.computeMeasurement(log_start, p_E2g, x_e2n.rota(vel), clk, z);

    func.init(log_start, z.topRows<2>(), sat, p_E2g, Matrix2d::Identity(), 0.3, p_b2g, 0, 0);

    xform::Xformd x = xform::Xformd::Identity();
    Vector3d res_ad, res_an;

    x += Vector6d::Random();
    vel += Vector3d::Random();
    clk += Vector2d::Random();
    x_e2n += Vector6d::Random();

    double* p[5] {x.data(), vel.data(), clk.data(), x_e2n.data(), &s};
    ad.Evaluate(p, res_ad.data(), NULL);
    an.Evaluate(p, res_an.data(), NULL);

    EXPECT_MAT_NEAR(res_ad, res_an, 1e-8);
}

TEST (PrangeFactor, Jac1)
{
    PseudorangeFunctor func;
    PseudorangeFactor an(&func);

    Satellite sat(3, 0);
    sat.readFromRawFile(SALSA_DIR"/sample/eph.dat");

    Vector3d p_b2g(0, 0, 0.3);
    GTime log_start = GTime::fromUTC(1541454646,  0.993);
    Vector3d p_E2b(-1798904.13, -4532227.1 ,  4099781.95);
    xform::Xformd x_e2n = WGS84::x_ecef2ned(p_E2b);
    Vector3d p_E2g = WGS84::ned2ecef(x_e2n, p_b2g);
    Vector3d vel(1, 2,3);
    Vector2d clk(1e-5, 1e-8);
    double s(0.5);
    XformParam param;

    Vector3d z;
    sat.computeMeasurement(log_start, p_E2g, x_e2n.rota(vel), clk, z);

    func.init(log_start, z.topRows<2>(), sat, p_E2g, Matrix2d::Identity(), 1.0, p_b2g, 0, 0);

    xform::Xformd x = xform::Xformd::Identity();

    x += Vector6d::Random();
    vel += Vector3d::Random();
    clk += Vector2d::Random();
    x_e2n += Vector6d::Random();

    Matrix<double, 3, 6, RowMajor> drdxfd, drdxa;
    double eps = 1e-5;
    for (int i = 0; i < 6; i++)
    {
        xform::Xformd x_plus;
        xform::Xformd x_minus;
        Vector6d e_ip = Vector6d::Unit(i)*eps;
        Vector6d e_im = -1.0*Vector6d::Unit(i)*eps;

        param.Plus(x.data(), e_ip.data(), x_plus.data());
        param.Plus(x.data(), e_im.data(), x_minus.data());
        double* p_plus[5] = {x_plus.data(), vel.data(), clk.data(), x_e2n.data(), &s};
        double* p_minus[5] = {x_minus.data(), vel.data(), clk.data(), x_e2n.data(), &s};
        Vector3d res_plus, res_minus;
        an.Evaluate(p_plus, res_plus.data(), NULL);
        an.Evaluate(p_minus, res_minus.data(), NULL);
        drdxfd.col(i) = (res_plus - res_minus)/(2.0*eps);
    }
    Matrix<double, 3, 7, RowMajor> drdxa_global;
    double* p[5] = {x.data(), vel.data(), clk.data(), x_e2n.data(), &s};
    double* j[5] = {drdxa_global.data(), NULL, NULL, NULL, NULL};
    Vector3d res;
    an.Evaluate(p, res.data(), j);
    Matrix<double, 7,6, RowMajor> param_jac;
    param.ComputeJacobian(x.data(), param_jac.data());
    drdxa = drdxa_global * param_jac;

    std::cout << "FD:\n" << drdxfd << std::endl;
    std::cout << "AN:\n" << drdxa << std::endl;

    EXPECT_MAT_NEAR(drdxfd, drdxa, 2e-3);
}

TEST (PrangeFactor, Jac2)
{
    PseudorangeFunctor func;
    PseudorangeFactor an(&func);

    Satellite sat(3, 0);
    sat.readFromRawFile(SALSA_DIR"/sample/eph.dat");

    Vector3d p_b2g(0, 0, 0.3);
    GTime log_start = GTime::fromUTC(1541454646,  0.993);
    Vector3d p_E2b(-1798904.13, -4532227.1 ,  4099781.95);
    xform::Xformd x_e2n = WGS84::x_ecef2ned(p_E2b);
    Vector3d p_E2g = WGS84::ned2ecef(x_e2n, p_b2g);
    Vector3d vel(1, 2,3);
    Vector2d clk(1e-5, 1e-8);
    double s(0.5);

    Vector3d z;
    sat.computeMeasurement(log_start, p_E2g, x_e2n.rota(vel), clk, z);

    func.init(log_start, z.topRows<2>(), sat, p_E2g, Matrix2d::Identity(), 1.0, p_b2g, 0, 0);

    xform::Xformd x = xform::Xformd::Identity();

    x += Vector6d::Random();
    vel += Vector3d::Random();
    clk += Vector2d::Random();
    x_e2n += Vector6d::Random();

    Matrix<double, 3, 3, RowMajor> drdvfd, drdva;
    double eps = 1e-5;
    for (int i = 0; i < 3; i++)
    {
        Vector3d v_plus;
        Vector3d v_minus;
        v_plus = vel + Vector3d::Unit(i)*eps;
        v_minus = vel - Vector3d::Unit(i)*eps;

        double* p_plus[5] = {x.data(), v_plus.data(), clk.data(), x_e2n.data(), &s};
        double* p_minus[5] = {x.data(), v_minus.data(), clk.data(), x_e2n.data(), &s};
        Vector3d res_plus, res_minus;
        an.Evaluate(p_plus, res_plus.data(), NULL);
        an.Evaluate(p_minus, res_minus.data(), NULL);
        drdvfd.col(i) = (res_plus - res_minus)/(2.0*eps);
    }
    double* p[5] = {x.data(), vel.data(), clk.data(), x_e2n.data(), &s};
    double* j[5] = {NULL, drdva.data(), NULL, NULL, NULL};
    Vector2d res;
    an.Evaluate(p, res.data(), j);

    std::cout << "FD:\n" << drdvfd << std::endl;
    std::cout << "AN:\n" << drdva << std::endl;

    EXPECT_MAT_NEAR(drdvfd, drdva, 1e-3);
}

TEST (PrangeFactor, Jac3)
{
    PseudorangeFunctor func;
    PseudorangeFactor an(&func);

    Satellite sat(3, 0);
    sat.readFromRawFile(SALSA_DIR"/sample/eph.dat");

    Vector3d p_b2g(0, 0, 0.3);
    GTime log_start = GTime::fromUTC(1541454646,  0.993);
    Vector3d p_E2b(-1798904.13, -4532227.1 ,  4099781.95);
    xform::Xformd x_e2n = WGS84::x_ecef2ned(p_E2b);
    Vector3d p_E2g = WGS84::ned2ecef(x_e2n, p_b2g);
    Vector3d vel(1, 2,3);
    Vector2d clk(1e-5, 1e-8);
    double s(0.5);

    Vector3d z;
    sat.computeMeasurement(log_start, p_E2g, x_e2n.rota(vel), clk, z);

    func.init(log_start, z.topRows<2>(), sat, p_E2g, Matrix2d::Identity(), 1.0, p_b2g, 0, 0);

    xform::Xformd x = xform::Xformd::Identity();

    x += Vector6d::Random();
    vel += Vector3d::Random();
    clk += Vector2d::Random();
    x_e2n += Vector6d::Random();

    Matrix<double, 3, 2, RowMajor> drdclkfd, drdclka;
    double eps = 1e-5;
    for (int i = 0; i < 2; i++)
    {
        Vector2d clk_plus;
        Vector2d clk_minus;
        clk_plus = clk + Vector2d::Unit(i)*eps;
        clk_minus = clk - Vector2d::Unit(i)*eps;

        double* p_plus[5] = {x.data(), vel.data(), clk_plus.data(), x_e2n.data(), &s};
        double* p_minus[5] = {x.data(), vel.data(), clk_minus.data(), x_e2n.data(), &s};
        Vector3d res_plus, res_minus;
        an.Evaluate(p_plus, res_plus.data(), NULL);
        an.Evaluate(p_minus, res_minus.data(), NULL);
        drdclkfd.col(i) = (res_plus - res_minus)/(2.0*eps);
    }
    double* p[5] = {x.data(), vel.data(), clk.data(), x_e2n.data(), &s};
    double* j[5] = {NULL, NULL, drdclka.data(), NULL, NULL};
    Vector3d res;
    an.Evaluate(p, res.data(), j);

    std::cout << "FD:\n" << drdclkfd << std::endl;
    std::cout << "AN:\n" << drdclka << std::endl;

    EXPECT_MAT_NEAR(drdclkfd, drdclka, 3e-3);
}


TEST (PrangeFactor, Jac4)
{
    PseudorangeFunctor func;
    PseudorangeFactor an(&func);

    Satellite sat(3, 0);
    sat.readFromRawFile(SALSA_DIR"/sample/eph.dat");

    Vector3d p_b2g(0, 0, 0.3);
    GTime log_start = GTime::fromUTC(1541454646,  0.993);
    Vector3d p_E2b(-1798904.13, -4532227.1 ,  4099781.95);
    xform::Xformd x_e2n = WGS84::x_ecef2ned(p_E2b);
    Vector3d p_E2g = WGS84::ned2ecef(x_e2n, p_b2g);
    Vector3d vel(1, 2,3);
    Vector2d clk(1e-5, 1e-8);
    XformParam param;
    double s(0.5);

    Vector3d z;
    sat.computeMeasurement(log_start, p_E2g, x_e2n.rota(vel), clk, z);

    func.init(log_start, z.topRows<2>(), sat, p_E2g, Matrix2d::Identity(), 1.0, p_b2g, 0, 0);

    xform::Xformd x = xform::Xformd::Identity();

    x += Vector6d::Random();
    vel += Vector3d::Random();
    clk += Vector2d::Random();
    x_e2n += Vector6d::Random();

    Matrix<double, 3, 6, RowMajor> drdxe2nfd, drdxe2na;
    double eps = 1e-5;
    for (int i = 0; i < 6; i++)
    {
        xform::Xformd xe2n_plus;
        xform::Xformd xe2n_minus;
        Vector6d e_ip = Vector6d::Unit(i)*eps;
        Vector6d e_im = -1.0*Vector6d::Unit(i)*eps;

        param.Plus(x_e2n.data(), e_ip.data(), xe2n_plus.data());
        param.Plus(x_e2n.data(), e_im.data(), xe2n_minus.data());
        double* p_plus[5] = {x.data(), vel.data(), clk.data(), xe2n_plus.data(), &s};
        double* p_minus[5] = {x.data(), vel.data(), clk.data(), xe2n_minus.data(), &s};
        Vector3d res_plus, res_minus;
        an.Evaluate(p_plus, res_plus.data(), NULL);
        an.Evaluate(p_minus, res_minus.data(), NULL);
        drdxe2nfd.col(i) = (res_plus - res_minus)/(2.0*eps);
    }
    Matrix<double, 3, 7, RowMajor> drdxe2na_global;
    double* p[5] = {x.data(), vel.data(), clk.data(), x_e2n.data(), &s};
    double* j[5] = {NULL, NULL, NULL, drdxe2na_global.data(), NULL};
    Vector2d res;
    an.Evaluate(p, res.data(), j);
    Matrix<double, 7,6, RowMajor> param_jac;
    param.ComputeJacobian(x_e2n.data(), param_jac.data());
    drdxe2na = drdxe2na_global * param_jac;

    std::cout << "FD:\n" << drdxe2nfd << std::endl;
    std::cout << "AN:\n" << drdxe2na << std::endl;

    EXPECT_MAT_NEAR(drdxe2nfd, drdxe2na, 2e-3);
}

TEST (PrangeFactor, Jac5)
{
    PseudorangeFunctor func;
    PseudorangeFactor an(&func);

    Satellite sat(3, 0);
    sat.readFromRawFile(SALSA_DIR"/sample/eph.dat");

    Vector3d p_b2g(0, 0, 0.3);
    GTime log_start = GTime::fromUTC(1541454646,  0.993);
    Vector3d p_E2b(-1798904.13, -4532227.1 ,  4099781.95);
    xform::Xformd x_e2n = WGS84::x_ecef2ned(p_E2b);
    Vector3d p_E2g = WGS84::ned2ecef(x_e2n, p_b2g);
    Vector3d vel(1, 2,3);
    Vector2d clk(1e-5, 1e-8);
    double s(0.5);

    Vector3d z;
    sat.computeMeasurement(log_start, p_E2g, x_e2n.rota(vel), clk, z);

    func.init(log_start, z.topRows<2>(), sat, p_E2g, Matrix2d::Identity(), 6.0, p_b2g, 0, 0);

    xform::Xformd x = xform::Xformd::Identity();

    x += Vector6d::Random();
    vel += Vector3d::Random();
    clk += Vector2d::Random();
    x_e2n += Vector6d::Random();

    Matrix<double, 3, 1> drdsfd, drdsa;
    double eps = 1e-5;

    double s_plus;
    double s_minus;
    s_plus = s + eps;
    s_minus = s - eps;

    double* p_plus[5] = {x.data(), vel.data(), clk.data(), x_e2n.data(), &s_plus};
    double* p_minus[5] = {x.data(), vel.data(), clk.data(), x_e2n.data(), &s_minus};
    Vector3d res_plus, res_minus;
    an.Evaluate(p_plus, res_plus.data(), NULL);
    an.Evaluate(p_minus, res_minus.data(), NULL);
    drdsfd = (res_plus - res_minus)/(2.0*eps);

    double* p[5] = {x.data(), vel.data(), clk.data(), x_e2n.data(), &s};
    double* j[5] = {NULL, NULL, NULL, NULL, drdsa.data()};
    Vector3d res;
    an.Evaluate(p, res.data(), j);

    std::cout << "FD:\n" << drdsa << std::endl;
    std::cout << "AN:\n" << drdsa << std::endl;

    EXPECT_MAT_NEAR(drdsfd, drdsa, 3e-3);
}


TEST (PrangeFactor, InitZeroResidual)
{
    PseudorangeFunctor func;

    Satellite sat(3, 0);
    sat.readFromRawFile(SALSA_DIR"/sample/eph.dat");

    Vector3d p_b2g(0, 0, 0.3);
    GTime log_start = GTime::fromUTC(1541454646,  0.993);
    Vector3d p_E2b(-1798904.13, -4532227.1 ,  4099781.95);
    xform::Xformd x_e2n = WGS84::x_ecef2ned(p_E2b);
    Vector3d p_E2g = WGS84::ned2ecef(x_e2n, p_b2g);
    Vector3d vel(1, 2,3);
    Vector2d clk(1e-5, 1e-8);
    double k = 1.0;

    Vector3d z;
    sat.computeMeasurement(log_start, p_E2g, x_e2n.rota(vel), clk, z);

    func.init(log_start, z.topRows<2>(), sat, p_E2g, Matrix2d::Identity(), 0.4, p_b2g, 0, 0);

    xform::Xformd x = xform::Xformd::Identity();
    Vector3d res;
    func(x.data(), vel.data(), clk.data(), x_e2n.data(), &k, res.data());

    EXPECT_MAT_NEAR(res, Vector3d::Zero(), 1e-8);
}
