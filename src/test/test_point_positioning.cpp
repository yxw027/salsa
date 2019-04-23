#include <gtest/gtest.h>

#include "multirotor_sim/simulator.h"
#include "multirotor_sim/controller.h"
#include "multirotor_sim/utils.h"
#include "gnss_utils//wgs84.h"
#include "salsa/test_common.h"
#include "salsa/salsa.h"


using namespace Eigen;
using namespace xform;
using namespace multirotor_sim;
using namespace salsa;
using namespace gnss_utils;

TEST (Salsa, PointPositioningInit)
{
  Simulator sim;
  salsa::Salsa salsa;
  salsa.init("../params/salsa.yaml");

  std::string filename = "tmp.params.yaml";
  ofstream tmp_file(filename);
  YAML::Node node;
  node["ref_LLA"] = std::vector<double>{40.247082 * DEG2RAD, -111.647776 * DEG2RAD, 1387.998309};
  node["gnss_update_rate"] = 5;
  node["use_raw_gnss_truth"] = true;
  node["pseudorange_stdev"] = 3.0;
  node["pseudorange_rate_stdev"] = 0.1;
  node["carrier_phase_stdev"] = 0.01;
  node["ephemeris_filename"] = "../sample/eph.dat";
  node["start_time_week"] = 2026;
  node["start_time_tow_sec"] = 165029;
  node["clock_init_stdev"] = 1e-4;
  node["clock_walk_stdev"] = 1e-7;
  node["multipath_prob"] = 0;
  node["cycle_slip_prob"] = 0;
  node["multipath_error_range"] = 0;
  tmp_file << node;
  tmp_file.close();

  sim.param_filename_ = filename;
  sim.init_raw_gnss();
  sim.register_estimator(&salsa);
  sim.t_ = 0.0;

  multirotor_sim::State x;
  x.p << 16, 4, 8;
  x.v << 3, -4, 5;
  sim.dyn_.set_state(x);
  sim.clock_bias_ = 1.4e-8;
  sim.clock_bias_rate_ = 2.6e-9;
  sim.t_ = 0.3;
  sim.update_raw_gnss_meas();

  Vector3d p_ecef_true = WGS84::ned2ecef(sim.X_e2n_, x.p);
  Vector3d v_ecef_true = sim.X_e2n_.q().rota(x.v);
  Vector2d tau_true(sim.clock_bias_, sim.clock_bias_rate_);
  Vector3d p_ecef_hat = WGS84::ned2ecef(salsa.x_e2n_, salsa.xbuf_[0].x.t());
  Vector3d v_ecef_hat = salsa.x_e2n_.q().rota(salsa.xbuf_[0].v);
  Vector2d tau_hat = salsa.xbuf_[0].tau;

  EXPECT_MAT_NEAR(p_ecef_hat, p_ecef_true, 1e-5);
  EXPECT_MAT_NEAR(v_ecef_hat, v_ecef_true, 1e-5);
  EXPECT_MAT_NEAR(tau_hat, tau_true, 1e-5);

  cout << "ptrue " << p_ecef_true.transpose() << endl;
  cout << "phat  " << p_ecef_hat.transpose() << endl;
  cout << "vtrue " << v_ecef_true.transpose() << endl;
  cout << "vhat  " << v_ecef_hat.transpose() << endl;
  cout << "tautrue " << tau_true.transpose() << endl;
  cout << "tauhat  " << tau_hat.transpose() << endl;
}
