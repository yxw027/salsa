#include <gtest/gtest.h>

#include "salsa/test_common.h"
#include "factors/feat.h"
#include "factors/xform.h"
#include "factors/imu.h"
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
    EXPECT_MAT_NEAR(res, Vector2d(-0.014743, -0.0186242), 1e-4);
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

TEST (FeatFactor, OptOnePointOneView)
{
    Vector3d l{0.5, 0, 1};

    xform::Xformd x1 = xform::Xformd::Identity();
    xform::Xformd x2 = xform::Xformd::Identity() + (Vector6d() << 1, 0, 0, 0, 0, 0).finished();
    xform::Xformd x_b2c = xform::Xformd::Identity();

    Vector3d zeta_i = (l - x1.t()).normalized();
    Vector3d zeta_j = (l - x2.t()).normalized();
    double rho = 1.0/((l-x1.t()).norm());
    double rhohat = 0.001;
    FeatFunctor f(x_b2c, Matrix2d::Identity(), zeta_i, zeta_j, 1);

    ceres::Problem problem;
    problem.AddParameterBlock(x1.data(), 7, new XformParamAD);
    problem.AddParameterBlock(x2.data(), 7, new XformParamAD);
    problem.SetParameterBlockConstant(x1.data());
    problem.SetParameterBlockConstant(x2.data());
    FunctorShield<FeatFunctor>* ptr = new FunctorShield<FeatFunctor>(&f);
    problem.AddResidualBlock(new FeatFactorAD(ptr),
                             NULL,
                             x1.data(), x2.data(), &rhohat);

    Vector2d res1, res2;
    f(x1.data(), x2.data(), &rhohat, res1.data());
    f(x1.data(), x2.data(), &rho, res2.data());

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.max_num_iterations = 100;
    options.num_threads = 6;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;

    ceres::Solve(options, &problem, &summary);
    f(x1.data(), x2.data(), &rhohat, res2.data());

    EXPECT_NEAR(rhohat, rho, 1e-3);
}

TEST (FeatFactor, OptOnePointOneViewC2B)
{
    Vector3d l{0.5, 0, 1};

    xform::Xformd x1 = xform::Xformd::Identity();
    xform::Xformd x2 = xform::Xformd::Identity() + (Vector6d() << 1, 0, 0, 0, 0, 0).finished();
    xform::Xformd x_b2c = xform::Xformd::Identity();
    x_b2c.t() << 0.1, -0.3, -0.1;
    x_b2c.q() = quat::Quatd::from_euler(0, DEG2RAD * -45, 0);

    Vector3d zeta_i = (l - x1.t()).normalized();
    Vector3d zeta_j = (l - x2.t()).normalized();
    double rho = 1.0/((l-x1.t()).norm());
    double rhohat = 0.001;
    FeatFunctor f(x_b2c, Matrix2d::Identity(), zeta_i, zeta_j, 1);

    ceres::Problem problem;
    problem.AddParameterBlock(x1.data(), 7, new XformParamAD);
    problem.AddParameterBlock(x2.data(), 7, new XformParamAD);
    problem.SetParameterBlockConstant(x1.data());
    problem.SetParameterBlockConstant(x2.data());
    FunctorShield<FeatFunctor>* ptr = new FunctorShield<FeatFunctor>(&f);
    problem.AddResidualBlock(new FeatFactorAD(ptr),
                             NULL,
                             x1.data(), x2.data(), &rhohat);

    Vector2d res1, res2;
    f(x1.data(), x2.data(), &rhohat, res1.data());
    f(x1.data(), x2.data(), &rho, res2.data());

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.max_num_iterations = 100;
    options.num_threads = 6;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;

    ceres::Solve(options, &problem, &summary);
    f(x1.data(), x2.data(), &rhohat, res2.data());

    EXPECT_NEAR(rhohat, rho, 1e-3);
}

TEST (FeatFactor, OptImuWindow)
{
    Matrix<double, 3, 4> l;
    l << 0.5, 0, -0.5, 0,
         0, 0.5, 0, -0.5,
         1, 1, 1, 1;

    xform::Xformd x1 = xform::Xformd::Identity();
    Vector3d v1 = Vector3d::Zero();
    xform::Xformd x_b2c;
    x_b2c = xform::Xformd::Identity();

    Vector6d imu_bias;
    imu_bias << 0.01, 0, 0, 0, 0, 0;
    Vector6d imu_bias_hat = Vector6d::Zero();

    Vector6d imu;
    imu << 0.1, 0, 0, 0, 0, 0;

    ImuFunctor fi(0, Vector6d::Zero(), 0, 0);

    double dt = 0.002;
    double t = 0.0;
    while (t < 0.1)
    {
        t += dt;
        fi.integrate(t, imu+imu_bias, Matrix6d::Identity() * 0.01);
    }

    xform::Xformd x2 = x1 + 0.5*t*t*imu;
    Vector3d v2 = v1 + t * imu.topRows<3>();
    xform::Xformd x2hat;
    Vector3d v2hat;
    fi.finished(1);
    fi.estimateXj(x1.data(), v1.data(), x2hat.data(), v2hat.data());

    ceres::Problem problem;
    problem.AddParameterBlock(x1.data(), 7, new XformParamAD);
    problem.AddParameterBlock(v1.data(), 3);
    problem.AddParameterBlock(x2hat.data(), 7, new XformParamAD);
    problem.AddParameterBlock(v2hat.data(), 3);
    problem.SetParameterBlockConstant(x1.data());
    problem.SetParameterBlockConstant(v1.data());
//    problem.SetParameterBlockConstant(x2hat.data());
//    problem.SetParameterBlockConstant(v2hat.data());

    FunctorShield<ImuFunctor>* ptr = new FunctorShield<ImuFunctor>(&fi);
    problem.AddResidualBlock(new ImuFactorAD(ptr), NULL, x1.data(), x2hat.data(), v1.data(), v2hat.data(), imu_bias_hat.data());

    double rho[4];
    double rhohat[4];
    std::vector<FeatFunctor*> f;
    std::vector<Vector3d, aligned_allocator<Vector3d>> zetai(4), zetaj(4);
    for (int i = 0; i < 4; i++)
    {
        zetai[i] = (l.col(i) - x1.t()).normalized();
        zetaj[i] = (l.col(i) - x2.t()).normalized();
        rho[i] = (l.col(i) - x1.t()).norm();
        f.push_back(new FeatFunctor(x_b2c, Matrix2d::Identity(), zetai[i], zetaj[i], 1));
        FunctorShield<FeatFunctor>* ptr = new FunctorShield<FeatFunctor>(f.back());
        problem.AddResidualBlock(new FeatFactorAD(ptr), NULL, x1.data(), x2.data(), rho+i);
    }


    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.max_num_iterations = 100;
    options.num_threads = 6;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(rho[i], rhohat[i], 1e-3);
    EXPECT_MAT_NEAR(x2.arr(), x2hat.arr(), 1e-3);
    EXPECT_MAT_NEAR(imu_bias, imu_bias_hat, 1e-3);

    for (int i = 0; i < 4; i++)
        delete f[i];
}
