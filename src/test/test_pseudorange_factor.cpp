#include <fstream>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <gtest/gtest.h>

#include "geometry/xform.h"
#include "multirotor_sim/simulator.h"
#include "multirotor_sim/controller.h"
#include "multirotor_sim/satellite.h"
#include "multirotor_sim/estimator_wrapper.h"
#include "salsa/misc.h"
#include "salsa/test_common.h"
#include "salsa/logger.h"

#include "factors/pseudorange.h"
#include "factors/clock_dynamics.h"
#include "factors/xform.h"
#include "factors/shield.h"


class TestPseudorange : public ::testing::Test
{
protected:
  TestPseudorange() :
    sat(1, 0)
  {}
  void SetUp() override
    {
      time.week = 86400.00 / DateTime::SECONDS_IN_WEEK;
      time.tow_sec = 86400.00 - (time.week * DateTime::SECONDS_IN_WEEK);

      eph.sat = 1;
      eph.A = 5153.79589081 * 5153.79589081;
      eph.toe.week = 93600.0 / DateTime::SECONDS_IN_WEEK;
      eph.toe.tow_sec = 93600.0 - (eph.toe.week * DateTime::SECONDS_IN_WEEK);
      eph.toes = 93600.0;
      eph.deln =  0.465376527657e-08;
      eph.M0 =  1.05827953357;
      eph.e =  0.00223578442819;
      eph.omg =  2.06374037770;
      eph.cus =  0.177137553692e-05;
      eph.cuc =  0.457651913166e-05;
      eph.crs =  88.6875000000;
      eph.crc =  344.96875;
      eph.cis = -0.856816768646e-07;
      eph.cic =  0.651925802231e-07;
      eph.idot =  0.342514267094e-09;
      eph.i0 =  0.961685061380;
      eph.OMG0 =  1.64046615454;
      eph.OMGd = -0.856928551657e-08;
      sat.addEphemeris(eph);

  }
  eph_t eph;
  GTime time;
  Satellite sat;
};

TEST_F (TestPseudorange, CheckResidualAtInit)
{
    Vector3d provo_lla{40.246184 * DEG2RAD , -111.647769 * DEG2RAD, 1387.997511};
    Vector3d rec_pos = WSG84::lla2ecef(provo_lla);
    Xformd x_e2n = WSG84::x_ecef2ned(rec_pos);

    Vector3d z;
    Vector2d rho;
    sat.computeMeasurement(time, rec_pos, Vector3d::Zero(), Vector2d::Zero(), z);
    rho = z.topRows<2>();
    Matrix2d cov = (Vector2d{3.0, 0.4}).asDiagonal();

    PseudorangeFunctor prange_factor;
    prange_factor.init(time, rho, sat, rec_pos, cov, 1.0);

    Xformd x = Xformd::Identity();
    Vector3d v = Vector3d::Zero();
    Vector2d clk = Vector2d::Zero();
    Vector2d res = Vector2d::Zero();
    double s = 1.0;

    prange_factor(x.data(), v.data(), clk.data(), x_e2n.data(), &s, res.data());

    EXPECT_MAT_NEAR(res, Vector2d::Zero(), 1e-4);
}

//TEST_F (TestPseudorange, CheckResidualAfterMoving)
//{
//    Vector3d provo_lla{40.246184 * DEG2RAD , -111.647769 * DEG2RAD, 1387.997511};
//    Vector3d rec_pos = WSG84::lla2ecef(provo_lla);
//    Xformd x_e2n = WSG84::x_ecef2ned(rec_pos);
//    Vector2d clk_bias{0, 0};

//    Vector3d z;
//    Vector2d rho;
//    sat.computeMeasurement(time, rec_pos, Vector3d::Zero(), clk_bias, z);
//    rho = z.topRows<2>();
//    Matrix2d cov = (Vector2d{3.0, 0.4}).asDiagonal();

//    PseudorangeFunctor prange_factor;
//    prange_factor.init(time, rho, sat, rec_pos, cov, 3.0);

//    Xformd x = Xformd::Identity();
//    x.t() << 1, 0, 0;
//    Vector3d p_ecef = WSG84::ned2ecef(x_e2n, x.t());
//    Vector3d znew;
//    sat.computeMeasurement(time, p_ecef, Vector3d::Zero(), clk_bias, znew);
//    Vector3d true_res;
//    true_res <<  Vector2d{std::sqrt(1/3.0), std::sqrt(1/0.4)}.asDiagonal() * (z - znew).topRows<2>(), 0;

//    Vector3d v = Vector3d::Zero();
//    Vector3d res = Vector3d::Zero();
//    double s = 1.0;

//    prange_factor(x.data(), v.data(), clk_bias.data(), x_e2n.data(), &s, res.data());

//    cout << res.transpose() << endl;
//    cout << true_res.transpose() << endl;

//    EXPECT_MAT_NEAR(true_res, res, 1e-4);
//}

TEST(Pseudorange, TrajectoryClockDynamics)
{
    Simulator sim(false, 2);
    sim.load("../lib/multirotor_sim/params/sim_params.yaml");

    const int N = 100;
    int n = 0;

    Eigen::Matrix<double, 7, N> xhat, x;
    Eigen::Matrix<double, 3, N> vhat, v;
    Eigen::Matrix<double, 2, N> tauhat, tau;
    Eigen::Matrix<double, 15, 1> shat, s;
    std::vector<double> t;
    Xformd x_e2n_hat = sim.X_e2n_;


    std::default_random_engine rng;
    std::normal_distribution<double> normal;

    std::vector<Vector2d, aligned_allocator<Vector2d>> measurements;
    std::vector<Matrix2d, aligned_allocator<Matrix2d>> cov;
    std::vector<GTime> gtimes;
    measurements.resize(sim.satellites_.size());
    gtimes.resize(sim.satellites_.size());
    cov.resize(sim.satellites_.size());

    Matrix2d tau_cov = Vector2d{1e-5, 1e-6}.asDiagonal();

    ceres::Problem problem;

    s.setConstant(1.0);
    shat.setConstant(1.0);
    for (int i = 0; i < N; i++)
    {
        xhat.col(i) = Xformd::Identity().elements();
        problem.AddParameterBlock(xhat.data() + i*7, 7, new XformParamAD());
        vhat.setZero();
        problem.AddParameterBlock(vhat.data() + i*3, 3);
        tauhat.setZero();
        problem.AddParameterBlock(tauhat.data() + i*2, 2);
    }
    for (int i = 0; i < 15; i++)
        problem.AddParameterBlock(shat.data() + i, 1);
    problem.AddParameterBlock(x_e2n_hat.data(), 7, new XformParamAD());
    problem.SetParameterBlockConstant(x_e2n_hat.data());

    bool new_node = false;
    auto raw_gnss_cb = [&measurements, &new_node, &cov, &gtimes]
            (const GTime& t, const VecVec3& z, const VecMat3& R, std::vector<Satellite>& sats,
             const std::vector<bool>& slip)
    {
        int i = 0;
        for (Satellite sat : sats)
        {
          measurements[sat.idx_] = z[i].topRows<2>();
          cov[sat.idx_] = R[i].topLeftCorner<2,2>();
          gtimes[sat.idx_]=t;
          i++;
        }
        new_node = true;
    };

    EstimatorWrapper est;
    est.register_raw_gnss_cb(raw_gnss_cb);
    sim.register_estimator(&est);

    std::vector<PseudorangeFunctor*> prange_funcs;
    std::vector<ClockBiasFunctor*> clock_funcs;
    while (n < N)
    {
        sim.run();

        if (new_node)
        {
            new_node = false;
            for (int i = 0; i < measurements.size(); i++)
            {
                prange_funcs.push_back(new PseudorangeFunctor());
                prange_funcs.back()->init(gtimes[i], measurements[i], sim.satellites_[i], sim.get_position_ecef(), cov[i], 3.0);
                problem.AddResidualBlock(new PseudorangeFactorAD(new FunctorShield<PseudorangeFunctor>(prange_funcs.back())),
                                         NULL,
                                         xhat.data() + n*7,
                                         vhat.data() + n*3,
                                         tauhat.data() + n*2,
                                         x_e2n_hat.data(),
                                         s.data() + i);
                if (n > 0)
                {
                    clock_funcs.push_back(new ClockBiasFunctor(tau_cov));
                    clock_funcs.back()->init(sim.t_ - t.back());
                    problem.AddResidualBlock(new ClockBiasFactorAD(new FunctorShield<ClockBiasFunctor>(clock_funcs.back())),
                                         NULL, tauhat.data() + (n-1)*2, tauhat.data() + n*2);
                }
            }
            x.col(n) = sim.dyn_.get_state().X.elements();
            v.col(n) = sim.dyn_.get_state().q.rota(sim.dyn_.get_state().v);
            tau.col(n) << sim.clock_bias_, sim.clock_bias_rate_;
            t.push_back(sim.t_);
            n++;

        }
    }


    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.num_threads = 6;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;

    MatrixXd xhat0 = xhat;
    MatrixXd vhat0 = vhat;
    MatrixXd tauhat0 = tauhat;

    ceres::Solve(options, &problem, &summary);

    Logger log("/tmp/Pseudorange.TrajectoryClockDynamics.log");

    for (int i = 0; i < N; i++)
    {
        log.log(t[i]);
        log.logVectors(xhat0.col(i),
                       vhat0.col(i),
                       xhat.col(i),
                       vhat.col(i),
                       x.col(i),
                       v.col(i),
                       tauhat0.col(i),
                       tauhat.col(i),
                       tau.col(i));
    }

    for (auto func : prange_funcs)
      delete func;

    for (auto func: clock_funcs)
      delete func;

}

TEST(Pseudorange, ImuTrajectory)
{
  Simulator sim;
  sim.load("../lib/multirotor_sim/params/sim_params.yaml");
  sim.vo_enabled_ = false;
  sim.mocap_enabled_ = false;
  sim.alt_enabled_ = false;
  sim.gnss_enabled_ = false;
  sim.raw_gnss_enabled_ = true;
  sim.tmax_ = 10.0;

  string filename = "/tmp/Salsa.tmp.yaml";
  ofstream tmp(filename);
  YAML::Node node;
  node["x_u2m"] = std::vector<double>{0, 0, 0, 1, 0, 0, 0};
  node["x_u2c"] = std::vector<double>{0, 0, 0, 1, 0, 0, 0};
  node["x_u2b"] = std::vector<double>{0, 0, 0, 1, 0, 0, 0};
  node["dt_m"] = 0.0;
  node["dt_c"] = 0.0;
  node["log_prefix"] = "/tmp/Salsa.Pseudorange.ImuTrajectory";
  node["R_clock_bias"] = std::vector<double>{1e-6, 1e-8};
  tmp << node;
  tmp.close();

  Salsa salsa;
  salsa.init(filename);

  sim.register_estimator(&salsa);

  Logger log("/tmp/Salsa.Pseudorange.ImuTrajectory.Truth.log");
  while(sim.run())
  {
    Vector3d sim_ecef = sim.get_position_ecef();
    Vector3d salsa_ecef = WSG84::ned2ecef(salsa.x_e2n_, salsa.current_x_.t());
    if (salsa.current_node_ >= 0)
    {
      log.log(sim.t_);
      log.logVectors(WSG84::ecef2ned(salsa.x_e2n_, sim.get_position_ecef()), sim.state().q.arr_, sim.state().v);
    }
  }
}
