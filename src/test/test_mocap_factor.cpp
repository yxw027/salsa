#include <gtest/gtest.h>

#include "factors/mocap.h"
#include "factors/xform.h"


#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include "geometry/xform.h"
#include "geometry/support.h"
#include "factors/xform.h"
#include "factors/mocap.h"
#include "factors/imu.h"

#include "salsa/estimator_wrapper.h"
#include "salsa/logger.h"

#include "multirotor_sim/controller.h"
#include "multirotor_sim/simulator.h"

using namespace ceres;
using namespace Eigen;
using namespace std;
using namespace xform;

TEST(DISABLED_MocapFactor, MultirotorPoseGraphOptimization)
{
    ReferenceController cont;
    cont.load("../lib/multirotor_sim/params/sim_params.yaml");

    Simulator multirotor(cont, cont, false, 2);
    multirotor.load("../lib/multirotor_sim/params/sim_params.yaml");

    const int N = 100;

    Vector6d b, bhat;
    b.block<3,1>(0,0) = multirotor.get_accel_bias();
    b.block<3,1>(3,0) = multirotor.get_gyro_bias();
    bhat.setZero();

    double dt, dthat;
    dt = 0.010;
    dthat = 0.0;

    multirotor.mocap_transmission_time_ = dt;
    multirotor.mocap_update_rate_ = 5;


    Eigen::MatrixXd xhat, x;
    Eigen::MatrixXd vhat, v;
    xhat.resize(7, N+1);
    x.resize(7, N+1);
    vhat.resize(3, N+1);
    v.resize(3, N+1);
    xhat.setZero();
    xhat.row(3).setConstant(1.0);
    vhat.setZero();

    Problem problem;
    problem.AddParameterBlock(&dthat, 1);
    problem.AddParameterBlock(bhat.data(), 6);
    for (int n = 0; n < N; n++)
    {
        problem.AddParameterBlock(xhat.data()+7*n, 7, new XformParamAD());
        problem.AddParameterBlock(vhat.data()+3*n, 3);
    }

    xhat.col(0) = multirotor.get_pose().arr_;
    vhat.col(0) = multirotor.dyn_.get_state().v;
    x.col(0) = multirotor.get_pose().arr_;
    v.col(0) = multirotor.dyn_.get_state().v;

    std::vector<ImuFunctor*> factors;
    Matrix6d cov =  multirotor.get_imu_noise_covariance();
    factors.push_back(new ImuFunctor(0, bhat));

    // Integrate for N frames
    int node = 0;
    ImuFunctor* factor = factors[node];
    std::vector<double> t;
    t.push_back(multirotor.t_);
    Xformd prev2_pose, prev_pose, current_pose;
    double prev2_t, prev_t, current_t;
    Matrix6d pose_cov = Matrix6d::Constant(0);

    bool new_node;
    auto imu_cb = [&factor](const double& t, const Vector6d& z, const Matrix6d& R)
    {
        factor->integrate(t, z, R);
    };
    auto pos_cb = [&pose_cov, &new_node, &current_pose](const double& t, const Vector3d& z, const Matrix3d& R)
    {
        (void)t;
        new_node = true;
        pose_cov.block<3,3>(0,0) = R;
        current_pose.t_ = z;
    };
    auto att_cb = [&pose_cov, &new_node, &current_pose](const double& t, const Quatd& z, const Matrix3d& R)
    {
        (void)t;
        new_node = true;
        current_pose.q_ = z;
        pose_cov.block<3,3>(3,3) = R;
    };
    EstimatorWrapper est;
    est.register_imu_cb(imu_cb);
    est.register_att_cb(att_cb);
    est.register_pos_cb(pos_cb);
    multirotor.register_estimator(&est);

    while (node < N)
    {
        new_node = false;
        multirotor.run();
        current_t = multirotor.t_;

        if (new_node)
        {
            t.push_back(multirotor.t_);
            node += 1;

            // estimate next node pose and velocity with IMU preintegration
            factor->estimateXj(xhat.data()+7*(node-1), vhat.data()+3*(node-1),
                               xhat.data()+7*(node), vhat.data()+3*(node));
            // Calculate the Information Matrix of the IMU factor
            factor->finished();

            // Save off True Pose and Velocity for Comparison
            x.col(node) = multirotor.get_pose().arr_;
            v.col(node) = multirotor.dyn_.get_state().v;

            // Add IMU factor to graph
            problem.AddResidualBlock(new ImuFactorAD(factor),
                                     NULL, xhat.data()+7*(node-1), xhat.data()+7*node, vhat.data()+3*(node-1), vhat.data()+3*node, bhat.data());

            // Start a new Factor
            factors.push_back(new ImuFunctor(multirotor.t_, bhat));
            factor = factors[node];

            Vector6d current_pose_dot;
            current_pose_dot.segment<3>(0) = v.col(node);
            current_pose_dot.segment<3>(3) = multirotor.dyn_.get_state().w;
            problem.AddResidualBlock(new MocapFactorAD(new MocapFunctor(current_pose.elements(), current_pose_dot, pose_cov)), NULL, xhat.data()+7*node, &dthat);
        }
    }

    Solver::Options options;
    options.max_num_iterations = 100;
    options.num_threads = 6;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    Solver::Summary summary;
    Logger<double> log("/tmp/TimeOffset.MultiWindowConstantBias.log");

    MatrixXd xhat0 = xhat;
    MatrixXd vhat0 = vhat;
    //  cout.flush();

    //    cout << "xhat0\n" << xhat.transpose() << endl;
    //    cout << "bhat0\n" << bhat.transpose() << endl;
    //    cout << "dthat0: " << dthat << endl;

    ceres::Solve(options, &problem, &summary);
    double error = (b - bhat).norm();

    //  cout << summary.FullReport();
    //  cout << "x\n" << x.transpose() << endl;
    //  cout << "xhat0\n" << xhat.transpose() << endl;
    //  cout << "b:     " << b.transpose() << endl;
    //  cout << "bhat:  " << bhat.transpose() << endl;
    //  cout << "err:   " << (b - bhat).transpose() << endl;

    //  cout << "dt:    " << dt << endl;
    //  cout << "dthatf: " << dthat << endl;
    //  cout << "e " << error << endl;
    EXPECT_LE(error, 0.2);
    EXPECT_LE(fabs(dt - dthat), 0.01);


    for (int i = 0; i <= N; i++)
    {
        log.log(t[i]);
        log.logVectors(xhat0.col(i), vhat0.col(i), xhat.col(i), vhat.col(i), x.col(i), v.col(i));
    }
}
