#include <gtest/gtest.h>

#include "salsa/test_common.h"
#include "factors/feat.h"

TEST (FeatFactor, InitZeroResidual)
{
    Xformd xi;
    xi.t() = Vector3d(0, 0, 0);
    xi.q() = Quatd::Identity();
    Xformd xj;
    xj.t() = Vector3d(1, 0, 0);
    xj.q() = Quatd::Identity();
    Vector3d l(0, 0, 1);
    Xformd xb2c = Xformd::Identity();

    Vector3d zi = l - xi.t();
    Vector3d zj = l - xj.t();
    double rho = zi.norm();
    zi.normalize();
    zj.normalize();
    FeatFunctor f(xb2c, Matrix2d::Identity(), zi, zj);

    Vector2d res;
    f(xi.data(), xj.data(), &rho, res.data());

    EXPECT_MAT_NEAR(res, Vector2d::Zero(), 1e-8);
}


TEST (FeatFactor, Withc2bTransform)
{
    Xformd xi;
    xi.t() = Vector3d(0, 0, 0);
    xi.q() = Quatd::Identity();
    Xformd xj;
    xj.t() = Vector3d(1, 0, 0);
    xj.q() = Quatd::Identity();
    Vector3d l(0, 0, 1);
    Xformd xb2c;
    xb2c.t() = Vector3d(0.1, 0, 0);
    xb2c.q() = Quatd::from_euler(DEG2RAD*5, 0, 0);

    Vector3d zi = xb2c.transformp(l - xi.t());
    Vector3d zj = xb2c.transformp(l - xj.t());
    double rho = zi.norm();
    zi.normalize();
    zj.normalize();
    FeatFunctor f(xb2c, Matrix2d::Identity(), zi, zj);

    Vector2d res;
    f(xi.data(), xj.data(), &rho, res.data());

    EXPECT_MAT_NEAR(res, Vector2d::Zero(), 1e-8);
}

TEST (FeatFactor, Withc2bTransformAndNoise)
{
    Xformd xi;
    xi.t() = Vector3d(0, 0, 0);
    xi.q() = Quatd::Identity();
    Xformd xj;
    xj.t() = Vector3d(1, 0, 0);
    xj.q() = Quatd::Identity();
    Vector3d l(0, 0, 1);
    Xformd xb2c;
    xb2c.t() = Vector3d(0.1, 0, 0);
    xb2c.q() = Quatd::from_euler(DEG2RAD*5, 0, 0);

    Vector3d zi = xb2c.transformp(l - xi.t());
    Vector3d zj = xb2c.transformp(l - xj.t()) + Vector3d{0.01, -0.02, 0.03};
    double rho = zi.norm();
    zi.normalize();
    zj.normalize();
    FeatFunctor f(xb2c, Matrix2d::Identity(), zi, zj);

    Vector2d res;
    f(xi.data(), xj.data(), &rho, res.data());
    cout << res << endl;
    EXPECT_MAT_NEAR(res, Vector2d(-0.0150401, -0.013848), 1e-4);
}
