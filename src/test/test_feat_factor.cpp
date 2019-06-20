#include <gtest/gtest.h>

#include "salsa/test_common.h"
#include "factors/feat.h"
#include "factors/xform.h"
#include "factors/imu.h"
#include "multirotor_sim/simulator.h"
#include "multirotor_sim/estimator_wrapper.h"

using namespace salsa;
using namespace multirotor_sim;
using namespace xform;
using namespace quat;

TEST (FeatFactor, ADvsANEval)
{
    Xformd xb2c = Xformd::Random();
    Vector3d zi = Vector3d::Random();
    Vector3d zj = Vector3d::Random();
    double rho = zi.norm();
    zi.normalize();
    zj.normalize();
    xform::Xformd xi  = Xformd::Random();
    xform::Xformd xj  = Xformd::Random();

    FeatFunctor f(3.0*Matrix2d::Identity(), xb2c, zi, zj, 0);
    FeatFactor an(&f);
    FeatFactorAD ad(new FunctorShield<FeatFunctor>(&f));

    Vector2d res1, res2;
    double* parameters[3] {xi.data(), xj.data(), &rho};
    double* jac[3] {NULL, NULL, NULL};
    ad.Evaluate(parameters, res1.data(), jac);
    an.Evaluate(parameters, res2.data(), jac);

    EXPECT_MAT_NEAR(res1, res2, 1e-8);
}

TEST (FeatFactor, FDvsANJac1)
{
    Xformd xb2c = Xformd::Random();
    Vector3d zi = Vector3d::Random();
    Vector3d zj = Vector3d::Random();
    double rho = zi.norm();
    zi.normalize();
    zj.normalize();
    FeatFunctor f(Matrix2d::Identity(), xb2c, zi, zj, 0);
    FeatFactor an(&f);
    xform::Xformd xi  = Xformd::Random();
    xform::Xformd xj  = Xformd::Random();

    XformParam param;
    Matrix<double, 2, 6, RowMajor> drdx1fd, drdx1a;
    double eps = 1e-8;
    for (int i = 0; i < 6; i++)
    {
        Xformd xi_plus;
        Xformd xi_minus;
        Vector6d e_ip = Vector6d::Unit(i)*eps;
        Vector6d e_im = -1.0*Vector6d::Unit(i)*eps;

        param.Plus(xi.data(), e_ip.data(), xi_plus.data());
        param.Plus(xi.data(), e_im.data(), xi_minus.data());
        double* p_plus[3] = {xi_plus.data(), xj.data(), &rho};
        double* p_minus[3] = {xi_minus.data(), xj.data(), &rho};
        Vector2d res_plus, res_minus;
        an.Evaluate(p_plus, res_plus.data(), NULL);
        an.Evaluate(p_minus, res_minus.data(), NULL);
        drdx1fd.col(i) = (res_plus - res_minus)/(2.0*eps);
    }
    Matrix<double, 2, 7, RowMajor> drdx1a_global;
    double* p[3] = {xi.data(), xj.data(), &rho};
    double* j[3] = {drdx1a_global.data(), NULL, NULL};
    Vector2d res;
    an.Evaluate(p, res.data(), j);
    Matrix<double, 7,6, RowMajor> param_jac;
    param.ComputeJacobian(xi.data(), param_jac.data());
    drdx1a = drdx1a_global * param_jac;

//    std::cout << "FD:\n" << drdx1fd << std::endl;
//    std::cout << "AN:\n" << drdx1a << std::endl;

    EXPECT_MAT_NEAR(drdx1fd, drdx1a, 1e-6);
}

TEST (FeatFactor, FDvsANJac2)
{
    Xformd xb2c = Xformd::Random();
    Vector3d zi = Vector3d::Random();
    Vector3d zj = Vector3d::Random();
    double rho = zi.norm();
    zi.normalize();
    zj.normalize();
    FeatFunctor f(Matrix2d::Identity(), xb2c, zi, zj, 0);
    FeatFactor an(&f);
    xform::Xformd xi  = Xformd::Random();
    xform::Xformd xj  = Xformd::Random();

    XformParam param;
    Matrix<double, 2, 6, RowMajor> drdxjfd, drdxja;
    double eps = 1e-8;
    for (int i = 0; i < 6; i++)
    {
        Xformd xj_plus;
        Xformd xj_minus;
        Vector6d e_ip = Vector6d::Unit(i)*eps;
        Vector6d e_im = -1.0*Vector6d::Unit(i)*eps;

        param.Plus(xj.data(), e_ip.data(), xj_plus.data());
        param.Plus(xj.data(), e_im.data(), xj_minus.data());
        double* p_plus[3] = {xi.data(), xj_plus.data(), &rho};
        double* p_minus[3] = {xi.data(), xj_minus.data(), &rho};
        Vector2d res_plus, res_minus;
        an.Evaluate(p_plus, res_plus.data(), NULL);
        an.Evaluate(p_minus, res_minus.data(), NULL);
        drdxjfd.col(i) = (res_plus - res_minus)/(2.0*eps);
    }
    Matrix<double, 2, 7, RowMajor> drdxja_global;
    double* p[3] = {xi.data(), xj.data(), &rho};
    double* j[3] = {NULL, drdxja_global.data(), NULL};
    Vector2d res;
    an.Evaluate(p, res.data(), j);
    Matrix<double, 7,6, RowMajor> param_jac;
    param.ComputeJacobian(xj.data(), param_jac.data());
    drdxja = drdxja_global * param_jac;

//    std::cout << "FD:\n" << drdxjfd << std::endl;
//    std::cout << "AN:\n" << drdxja << std::endl;

    EXPECT_MAT_NEAR(drdxjfd, drdxja, 1e-6);
}

TEST (FeatFactor, FDvsANJac3)
{
    Xformd xb2c = Xformd::Random();
    Vector3d zi = Vector3d::Random();
    Vector3d zj = Vector3d::Random();
    double rho = zi.norm();
    zi.normalize();
    zj.normalize();
    FeatFunctor f(Matrix2d::Identity(), xb2c, zi, zj, 0);
    FeatFactor an(&f);
    xform::Xformd xi  = Xformd::Random();
    xform::Xformd xj  = Xformd::Random();

    Matrix<double, 2, 1> drdxjfd, drdxja;
    double eps = 1e-8;
    double rho_plus(rho+eps), rho_minus(rho-eps);

    double* p_plus[3] = {xi.data(), xj.data(), &rho_plus};
    double* p_minus[3] = {xi.data(), xj.data(), &rho_minus};
    Vector2d res_plus, res_minus;
    an.Evaluate(p_plus, res_plus.data(), NULL);
    an.Evaluate(p_minus, res_minus.data(), NULL);
    drdxjfd = (res_plus - res_minus)/(2.0*eps);

    double* j[3] = {NULL, NULL, drdxja.data()};
    double* p[3] = {xi.data(), xj.data(), &rho};
    Vector2d res;

    an.Evaluate(p, res.data(), j);

//    std::cout << "FD:\n" << drdxjfd << std::endl;
//    std::cout << "AN:\n" << drdxja << std::endl;

    EXPECT_MAT_NEAR(drdxjfd, drdxja, 1e-6);
}


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
    FeatFunctor f(Matrix2d::Identity(), xb2c, zi, zj, 0);

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
    xb2c.t() = Vector3d(0.0, 0, 0);
    xb2c.q() = Quatd::from_euler(DEG2RAD*10, 0, 0);

    Vector3d zi = xb2c.rotp(xi.rotp(l - (xi.rota(xb2c.t()) + xi.t())));
    Vector3d zj = xb2c.rotp(xj.rotp(l - (xj.rota(xb2c.t()) + xj.t())));

    double rho = 1.0/zi.norm();
    zi.normalize();
    zj.normalize();
    FeatFunctor f(Matrix2d::Identity(), xb2c, zi, zj, 0);

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
    FeatFunctor f(Matrix2d::Identity(), xb2c, zi, zj, 0);

    Vector2d res;
    f(xi.data(), xj.data(), &rho, res.data());
//    cout << res << endl;
//    EXPECT_MAT_NEAR(res, Vector2d(-0.014743, -0.0186242), 1e-4);
}

TEST (FeatFactor, SimulatedFeatures)
{
    Simulator sim;
    ImageFeat z_;
    Xformd x_;
    int feat_cb_called = 0;
    auto feat_cb = [&z_, &feat_cb_called, &sim, &x_](const double& t, const ImageFeat& z, const Matrix2d& R_pix, const Matrix1d& R_depth)
    {
        z_ = z;
        feat_cb_called += 1;
        x_ = sim.state().X;
    };

    EstimatorWrapper est;
    est.register_feat_cb(feat_cb);


    sim.load(imu_feat(false));
    sim.register_estimator(&est);

    Camera<double> cam = sim.cam_;
    sim.x_b2c_.t() = Vector3d(0.1, 0.2, -0.05);
    sim.x_b2c_.q() = quat::Quatd::from_euler(DEG2RAD * 45, 0, 0);
    Xformd x_b2c = sim.x_b2c_;

    while (feat_cb_called ==  0)
        sim.run();

    ImageFeat z0 = z_;
    Xformd x0 = x_;

    while (feat_cb_called == 1)
        sim.run();

    for (int i = 0; i < z_.pixs.size(); i++)
    {
        Vector3d zeta0 = cam.invProj(z0.pixs[i], 1.0);
        Vector3d zeta1 = cam.invProj(z_.pixs[i], 1.0);

        Xformd xI2c0 = x0 * x_b2c;
        Xformd xI2c1 = x_ * x_b2c;
        Vector3d test0 = xI2c0.transformp(sim.env_.get_points()[z0.feat_ids[i]]).normalized();
        Vector3d test1 = xI2c1.transformp(sim.env_.get_points()[z_.feat_ids[i]]).normalized();

        Vector2d pix0 = cam.proj(test0);
        Vector2d pix1 = cam.proj(test1);

        FeatFunctor f(Matrix2d::Identity(), x_b2c, zeta0, zeta1, 0);
        double rho = 1.0/z0.depths[i];
        Vector2d res;
        f(x0.data(), x_.data(), &rho, res.data());

        EXPECT_MAT_NEAR(res, Vector2d::Zero(), 1e-8);
    }
}

TEST (FeatFactor, OptOnePointOneView)
{
    Vector3d l{0.5, 0, 1};

    xform::Xformd x1 = xform::Xformd::Identity();
    xform::Xformd x2 = xform::Xformd::Identity() + (Vector6d() << 1, 0, 0, 0.2, 0.1, -0.4).finished();
    xform::Xformd x_b2c = xform::Xformd::Identity();

    Vector3d zeta_i = (l - x1.t()).normalized();
    Vector3d zeta_j = x2.rotp(l - x2.t()).normalized();
    double rho = 1.0/((l-x1.t()).norm());
    double rhohat = 0.001;
    FeatFunctor f(Matrix2d::Identity(), x_b2c, zeta_i, zeta_j, 1);

    ceres::Problem problem;
    problem.AddParameterBlock(x1.data(), 7, new XformParamAD);
    problem.AddParameterBlock(x2.data(), 7, new XformParamAD);
    problem.AddParameterBlock(x_b2c.data(), 7, new XformParamAD);
    problem.SetParameterBlockConstant(x1.data());
    problem.SetParameterBlockConstant(x2.data());
    problem.SetParameterBlockConstant(x_b2c.data());
    FunctorShield<FeatFunctor>* ptr = new FunctorShield<FeatFunctor>(&f);
    problem.AddResidualBlock(new FeatFactorAD(ptr),
                             NULL, x1.data(), x2.data(), &rhohat);

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

    EXPECT_NEAR(rhohat, rho, 1e-8);
}

TEST (FeatFactor, OptOnePointOneViewC2B)
{
    Vector3d l{2, 0, 2.0};

    xform::Xformd x1 = xform::Xformd::Identity();
    xform::Xformd x2 = xform::Xformd::Identity() + (Vector6d() << 1, .2, .3, 0.1, -0.3, -0.4).finished();
    xform::Xformd x_b2c = xform::Xformd::Identity();
    x_b2c.t() << 0.1, -0.3, -0.1;
    x_b2c.q() = quat::Quatd::from_euler(0, DEG2RAD * 45, 0);

    Vector3d zeta_i = x_b2c.transformp(x1.transformp(l)).normalized();
    Vector3d zeta_j = x_b2c.transformp(x2.transformp(l)).normalized();
    double rho = 1.0/(x_b2c.transformp(x1.transformp(l)).norm());
    double rhohat = 0.001;
    FeatFunctor f(Matrix2d::Identity(), x_b2c, zeta_i, zeta_j, 1);

    ceres::Problem problem;
    problem.AddParameterBlock(x1.data(), 7, new XformParamAD);
    problem.AddParameterBlock(x2.data(), 7, new XformParamAD);
    problem.AddParameterBlock(x_b2c.data(), 7, new XformParamAD);
    problem.SetParameterBlockConstant(x1.data());
    problem.SetParameterBlockConstant(x2.data());
    problem.SetParameterBlockConstant(x_b2c.data());
    FunctorShield<FeatFunctor>* ptr = new FunctorShield<FeatFunctor>(&f);
    problem.AddResidualBlock(new FeatFactorAD(ptr),
                             NULL, x1.data(), x2.data(), &rhohat);

    Vector2d res1, res2;
    f(x1.data(), x2.data(), &rhohat, res1.data());
    f(x1.data(), x2.data(), &rho, res2.data());

//    std::cout << "res1 " << res1.transpose() << std::endl;
//    std::cout << "res2 " << res2.transpose() << std::endl;

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.max_num_iterations = 100;
    options.num_threads = 6;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;

    ceres::Solve(options, &problem, &summary);
    f(x1.data(), x2.data(), &rhohat, res2.data());
//    std::cout << "res3 " << res2.transpose() << std::endl;

    EXPECT_NEAR(rhohat, rho, 1e-8);
}

//TEST (FeatFactor, Jacobian)
//{
//    Xformd xi;
//    xi.t() = Vector3d(0, 0, 0);
//    xi.q() = Quatd::Identity();
//    Xformd xj;
//    xj.t() = Vector3d(1, 0, 0);
//    xj.q() = Quatd::Identity();
//    Vector3d l(0, 0, 1);
//    Xformd xb2c = Xformd::Identity();

//    Vector3d zi = l - xi.t();
//    Vector3d zj = l - xj.t();
//    double rho = zi.norm();
//    zi.normalize();
//    zj.normalize();

//    Vector2d res;
//    Matrix<double, 2, 6, RowMajor> dres_dxiAD, dres_dxi;
//    Matrix<double, 2, 6, RowMajor> dres_dxjAD, dres_dxj;
//    Matrix<double, 2, 1, RowMajor> dres_drhoAD, dres_drho;

//    double* param[3] = {xi.data(), xj.data(), &rho};
//    double* r = res.data();
//    double* jacAD[3] = {dres_dxiAD.data(), dres_dxjAD.data(), dres_drhoAD.data()};
//    double* jac[3] = {dres_dxi.data(), dres_dxj.data(), dres_drho.data()};

//    FeatFunctor func(xb2c, Matrix2d::Identity(), zi, zj, 0);
//    FeatFactorAD fAD(FunctorShield<FeatFunctor>(&func));
//    fAD.Evaluate(param, r, jacAD);

//    FeatFactor f(xb2c, Matrix2d::Identity(), zi, zj, 0);
//    f.Evaluate(param, r, jac);
//}
