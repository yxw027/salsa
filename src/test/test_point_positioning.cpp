#include <gtest/gtest.h>

#include "multirotor_sim/simulator.h"
#include "multirotor_sim/controller.h"
#include "multirotor_sim/utils.h"
#include "multirotor_sim/wsg84.h"
#include "multirotor_sim/raw_gnss.h"
#include "salsa/test_common.h"
#include "salsa/salsa.h"

TEST (Salsa, PointPositioningInit)
{
  Simulator sim;
  Salsa salsa;
  salsa.init("../params/salsa.yaml");

  std::string filename = "tmp.params.yaml";
  ofstream tmp_file(filename);
  YAML::Node node;
  node["ref_LLA"] = std::vector<double>{40.247082 * DEG2RAD, -111.647776 * DEG2RAD, 1387.998309};
  node["gnss_update_rate"] = 5;
  node["use_raw_gnss_truth"] = false;
  node["pseudorange_stdev"] = 3.0;
  node["pseudorange_rate_stdev"] = 0.1;
  node["carrier_phase_stdev"] = 0.01;
  node["ephemeris_filename"] = "../sample/eph.dat";
  node["start_time_week"] = 2026;
  node["start_time_tow_sec"] = 165029;
  node["clock_init_stdev"] = 1e-4;
  node["clock_walk_stdev"] = 1e-7;
  tmp_file << node;
  tmp_file.close();

  sim.param_filename_ = filename;
  sim.init_raw_gnss();
  sim.register_estimator(&salsa);
  sim.t_ = 0.0;

  State x;
  x.p << 1000, 0, 0;
  sim.dyn_.set_state(x);
  sim.t_ = 0.3;
  sim.update_raw_gnss_meas();

  Vector3d p_ecef_true = WSG84::ned2ecef(sim.X_e2n_, x.p);
  Vector3d p_ecef_hat = WSG84::ned2ecef(salsa.x_e2n_, salsa.x_.col(0).topRows<3>());

  EXPECT_NEAR(p_ecef_hat.x(), p_ecef_true.x(), 3.0);
  EXPECT_NEAR(p_ecef_hat.y(), p_ecef_true.y(), 3.0);
  EXPECT_NEAR(p_ecef_hat.z(), p_ecef_true.z(), 5.0);
}
