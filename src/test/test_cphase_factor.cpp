#include <gtest/gtest.h>

#include "factors/carrier_phase.h"
#include "gnss_utils/satellite.h"
#include "gnss_utils/gtime.h"

using namespace gnss_utils;
using namespace salsa;
using namespace Eigen;
using namespace xform;
using namespace quat;



TEST (CPhaseFactor, InitZeroResidual)
{

    Satellite sat(3, 0);
    sat.readFromRawFile(SALSA_DIR"/sample/eph.dat");
    GTime log_start = GTime::fromUTC(1541454646,  0.993);

    Vector3d p_E2I(-1798904.13, -4532227.1 ,  4099781.95);
    xform::Xformd x_e2I = WGS84::x_ecef2ned(p_E2I);

    Vector3d p_b2g(0, 0, 0.3);

    xform::Xformd x_I2b0;
    x_I2b0.t_ << 1, 2, 3;
    x_I2b0.q_ = Quatd::from_euler(0.1, -0.1, 1.0);

    xform::Xformd x_I2b1;
    x_I2b1.t_ << 3, 6, 1;
    x_I2b1.q_ = Quatd::from_euler(-0.2, 0.2, -0.3);

    Vector3d p_e2g0 = x_e2I.transforma(x_I2b0.t()+x_I2b0.rota(p_b2g));
    Vector3d p_e2g1 = x_e2I.transforma(x_I2b1.t()+x_I2b1.rota(p_b2g));

    GTime t0 = log_start + 0.2;
    GTime t1 = log_start + 12.5;

    Vector2d clk0(1e-8, 1e-9);
    Vector2d clk1(1.1e-8, 1.1e-9);

    Vector3d z0, z1;
    sat.computeMeasurement(t0, p_e2g0, Vector3d::Zero(), clk0, z0);
    sat.computeMeasurement(t1, p_e2g1, Vector3d::Zero(), clk1, z1);

    CarrierPhaseFunctor func(1.0, z0(2), z1(2), sat, p_b2g, t0, t1, 0, 0, 1);

    double residual;

    func(x_I2b0.data(), x_I2b1.data(), clk0.data(), clk1.data(), x_e2I.data(), &residual);

    EXPECT_NEAR(residual, 0, 0.2);
}

