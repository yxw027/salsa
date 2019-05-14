#include <gtest/gtest.h>

#include "gnss_utils/gtime.h"
#include "gnss_utils/satellite.h"
#include "salsa/test_common.h"

#include "factors/pseudorange.h"

using namespace salsa;
using namespace gnss_utils;
using namespace Eigen;

TEST (PrangeFactor, InitZeroResidual)
{
    PseudorangeFunctor func;

    Satellite sat(3, 0);
    sat.readFromRawFile(SALSA_DIR"/sample/eph.dat");

    GTime log_start = GTime::fromUTC(1541454646,  0.993);
    Vector3d rec_pos {-1798904.13, -4532227.1 ,  4099781.95};
    xform::Xformd x_e2n = WGS84::x_ecef2ned(rec_pos);
    Vector3d vel(1, 2,3);
    Vector2d clk(1e-5, 1e-8);

    Vector3d z;
    sat.computeMeasurement(log_start, rec_pos, x_e2n.rota(vel), clk, z);

    func.init(log_start, z.topRows<2>(), sat, rec_pos, Matrix2d::Identity(), 0, 0);

    xform::Xformd x = xform::Xformd::Identity();
    Vector2d res;
    func(x.data(), vel.data(), clk.data(), x_e2n.data(), res.data());

    EXPECT_MAT_NEAR(res, Vector2d::Zero(), 1e-8);
}
