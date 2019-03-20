#include <gtest/gtest.h>

#include "salsa/test_common.h"
#include "factors/feat.h"
#include "multirotor_sim/simulator.h"
#include "multirotor_sim/estimator_wrapper.h"

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
    FeatFunctor f(xb2c, Matrix2d::Identity(), zi, zj, 0);

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
    double rho = 1.0/zi.norm();
    zi.normalize();
    zj.normalize();
    FeatFunctor f(xb2c, Matrix2d::Identity(), zi, zj, 0);

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
    double rho = 1.0/zi.norm();
    zi.normalize();
    zj.normalize();
    FeatFunctor f(xb2c, Matrix2d::Identity(), zi, zj, 0);

    Vector2d res;
    f(xi.data(), xj.data(), &rho, res.data());
    cout << res << endl;
    EXPECT_MAT_NEAR(res, Vector2d(-0.0150401, -0.013848), 1e-4);
}

TEST (FeatFactor, SimulatedFeatures)
{

    ImageFeat z_;
    int feat_cb_called = 0;
    auto feat_cb = [&z_, &feat_cb_called](const double& t, const ImageFeat& z, const Matrix2d& R_pix, const Matrix1d& R_depth)
    {
        z_ = z;
        feat_cb_called += 1;
    };

    EstimatorWrapper est;
    est.register_feat_cb(feat_cb);

    Simulator sim;
    sim.load(imu_feat());
    sim.register_estimator(&est);

    Camera<double> cam;
    cam.focal_len_(0) = sim.cam_F_(0,0);
    cam.focal_len_(1) = sim.cam_F_(1,1);
    cam.cam_center_ = sim.cam_center_;

    Xformd x_b2c(sim.p_b2c_, sim.q_b2c_);



    while (feat_cb_called ==  0)
        sim.run();

    ImageFeat z0 = z_;
    Xformd x0 = sim.state().X;

    while (feat_cb_called == 1)
        sim.run();

    for (int i = 0; i < z_.pixs.size(); i++)
    {
        Vector3d zeta0, zeta1;
        cam.invProj(z0.pixs[i], 1.0, zeta0);
        cam.invProj(z_.pixs[i], 1.0, zeta1);

        FeatFunctor f(x_b2c, Matrix2d::Identity(), zeta0, zeta1, 0);
        double rho = 1.0/z0.depths[i];
        Vector2d res;
        f(x0.data(), sim.state().X.data(), &rho, res.data());

        EXPECT_MAT_NEAR(res, Vector2d::Zero(), 1e-8);
    }
}
