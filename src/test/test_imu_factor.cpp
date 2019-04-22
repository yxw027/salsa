#include <fstream>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <gtest/gtest.h>

#include "factors/shield.h"
#include "factors/xform.h"
#include "factors/imu.h"
#include "factors/mocap.h"

#include "geometry/xform.h"
#include "geometry/support.h"
#include "multirotor_sim/simulator.h"
#include "multirotor_sim/controller.h"
#include "multirotor_sim/estimator_wrapper.h"
#include "salsa/num_diff.h"
#include "salsa/logger.h"
#include "salsa/test_common.h"

using namespace ceres;
using namespace Eigen;
using namespace std;
using namespace xform;
using namespace multirotor_sim;
using namespace salsa;

TEST(ImuFactor, compile)
{
    ImuFunctor imu(0.0, Vector6d::Zero(), 0, 0);
}

TEST(ImuFactor, Propagation)
{
    Simulator multirotor(false, 1);
    multirotor.load(imu_only());


    typedef ImuFunctor IMU;
    Vector6d b0;
    b0.setZero();
    Matrix6d cov = Matrix6d::Identity() * 1e-3;

    multirotor.run();
    IMU* imu = new IMU(multirotor.t_, b0, 0, 0);
    Xformd x0 = multirotor.state().X;
    Vector3d v0 = multirotor.state().v;

    salsa::Logger log("/tmp/ImuFactor.CheckPropagation.log");

    Xformd xhat = multirotor.state().X;
    Vector3d vhat = multirotor.state().v;
    log.log(multirotor.t_);
    log.logVectors(xhat.elements(), vhat, multirotor.state().X.elements(),
                   multirotor.state().v, multirotor.imu());

    double next_reset = 1.0;
    multirotor.tmax_ = 10.0;
    while (multirotor.run())
    {
        imu->integrate(multirotor.t_, multirotor.imu(), cov);

        if (std::abs(multirotor.t_ - next_reset) <= multirotor.dt_ /2.0)
        {
            delete imu;
            imu = new IMU(multirotor.t_, b0, 0, 0);
            x0 = multirotor.state().X;
            v0 = multirotor.state().v;
            next_reset += 1.0;
        }

        imu->estimateXj(x0.data(), v0.data(), xhat.data(), vhat.data());
        log.log(multirotor.t_);
        log.logVectors(xhat.elements(), vhat, multirotor.state().X.elements(), multirotor.state().v, multirotor.imu());
        EXPECT_MAT_NEAR(xhat.t(), multirotor.state().X.t(), 0.076);
        EXPECT_QUAT_NEAR(xhat.q(), multirotor.state().X.q(), 0.01);
        EXPECT_MAT_NEAR(vhat, multirotor.state().v, 0.05);
    }
    delete imu;
}


TEST(ImuFactor, ErrorStateDynamics)
{
    typedef ImuFunctor IMU;
    Vector9d dy;
    double t = 0;
    const double Tmax = 10.0;
    static const double dt = 0.001;

    Vector6d bias;
    bias.setZero();
    IMU y(t, bias, 0, 0);
    IMU yhat(t, bias, 0, 0);
    IMU::boxplus(y.y_, Vector9d::Constant(0.01), yhat.y_);
    IMU::boxminus(y.y_, yhat.y_, dy);

    Vector10d y_check;
    IMU::boxplus(yhat.y_, dy, y_check);
    ASSERT_MAT_NEAR(y.y_, y_check, 1e-8);

    Vector6d u, eta;
    u.setZero();
    eta.setZero();

    std::default_random_engine gen;
    std::normal_distribution<double> normal;

    salsa::Logger log("/tmp/ImuFactor.CheckDynamics.log");


    Matrix6d cov = Matrix6d::Identity() * 1e-3;
    Vector9d dydot;
    log.log(t);
    log.logVectors(y.y_, yhat.y_, dy, y_check, u);
    for (int i = 0; i < Tmax/dt; i++)
    {
        u += dt * randomNormal<Vector6d>(normal, gen);
        t += dt;
        y.errorStateDynamics(y.y_, dy, u, eta, dydot);
        dy += dydot * dt;

        y.integrate(t, u, cov);
        yhat.integrate(t, u, cov);
        IMU::boxplus(yhat.y_, dy, y_check);
        log.log(t);
        log.logVectors(y.y_, yhat.y_, dy, y_check, u);
        ASSERT_MAT_NEAR(y.y_, y_check, t > 0.3 ? 5e-6*t*t : 2e-7);
    }
}


TEST(ImuFactor, DynamicsJacobians)
{
    Matrix6d cov = Matrix6d::Identity()*1e-3;

    Vector6d b0;
    Vector10d y0;
    Vector6d u0;
    Vector6d eta0;
    Vector9d ydot;
    Vector9d dy0;

    Matrix9d A;
    Eigen::Matrix<double, 9, 6> B;

    for (int i = 0; i < 100; i++)
    {
        b0.setRandom();
        y0.setRandom();
        y0.segment<4>(6) = Quatd::Random().elements();
        u0.setRandom();

        eta0.setZero();
        dy0.setZero();

        ImuFunctor f(0, b0, 0, 0);
        f.dynamics(y0, u0, ydot, A, B);
        Vector9d dy0 = Vector9d::Zero();

        auto yfun = [&y0, &cov, &b0, &u0, &eta0](const Vector9d& dy)
        {
            ImuFunctor f(0, b0, 0, 0);
            Vector9d dydot;
            f.errorStateDynamics(y0, dy, u0, eta0, dydot);
            return dydot;
        };
        auto etafun = [&y0, &cov, &b0, &dy0, &u0](const Vector6d& eta)
        {
            ImuFunctor f(0, b0, 0, 0);
            Vector9d dydot;
            f.errorStateDynamics(y0, dy0, u0, eta, dydot);
            return dydot;
        };

        Matrix9d AFD;
        AFD.setZero();
        AFD = calc_jac(yfun, dy0);
        Eigen::Matrix<double, 9, 6> BFD = calc_jac(etafun, u0);

//        cout << "A\n" << A << "\nAFD\n" << AFD << "\n\n";
//        cout << "B\n" << B << "\nBFD\n" << BFD << "\n\n";

        ASSERT_MAT_NEAR(AFD, A, 1e-7);
        ASSERT_MAT_NEAR(BFD, B, 1e-7);
    }
}

TEST(ImuFactor, BiasJacobians)
{
    Simulator multirotor(false, 2);
    multirotor.load(imu_only());
    std::vector<Vector6d,Eigen::aligned_allocator<Vector6d>> meas;
    std::vector<double> t;
    multirotor.dt_ = 0.001;

    while (multirotor.t_ < 0.1)
    {
        multirotor.run();
        meas.push_back(multirotor.imu());
        t.push_back(multirotor.t_);
    }

    Matrix6d cov = Matrix6d::Identity()*1e-3;
    Vector6d b0;
    Eigen::Matrix<double, 9, 6> J, JFD;

    b0.setZero();
    ImuFunctor f(0, b0, 0, 0);
    Vector10d y0 = f.y_;
    for (int i = 0; i < meas.size(); i++)
    {
        f.integrate(t[i], meas[i], cov);
    }
    J = f.J_;

    auto fun = [&cov, &meas, &t, &y0](const Vector6d& b0)
    {
        ImuFunctor f(0, b0, 0, 0);
        for (int i = 0; i < meas.size(); i++)
        {
            f.integrate(t[i], meas[i], cov);
        }
        return f.y_;
    };
    auto bm = [](const MatrixXd& x1, const MatrixXd& x2)
    {
        Vector9d dx;
        ImuFunctor::boxminus(x1, x2, dx);
        return dx;
    };

    JFD = calc_jac(fun, b0, nullptr, nullptr, bm, 1e-5);
//    std::cout << "FD:\n" << JFD << "\nA:\n" << J <<std::endl;
    ASSERT_MAT_NEAR(J, JFD, 1e-4);
}
