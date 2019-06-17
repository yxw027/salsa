#include <gtest/gtest.h>

#include "multirotor_sim/simulator.h"
#include "multirotor_sim/controller.h"
#include "multirotor_sim/utils.h"
#include "gnss_utils/wgs84.h"
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
  salsa.init(default_params("/tmp/Salsa/PPInit/"));

  sim.load(imu_raw_gnss(false));
  sim.register_estimator(&salsa);
  salsa.p_b2g_ = sim.p_b2g_;
  salsa.update_on_gnss_ = true;
  salsa.disable_solver_ = true;

  multirotor_sim::State x;
  x.p.setZero();
  x.v.setZero();
  sim.dyn_.set_state(x);
  sim.clock_bias_ = 1.4e-8;
  sim.clock_bias_rate_ = 2.6e-9;

  sim.t_ = 0.3;
  sim.update_raw_gnss_meas();

  EXPECT_MAT_NEAR(sim.X_e2n_.arr(), salsa.x_e2n_.arr(), 1e-1); // PP is used for getting the first x_e2n estimate
  EXPECT_MAT_NEAR(salsa.xbuf_[0].x.t(), Vector3d::Zero(), 1e-8);
  EXPECT_MAT_NEAR(salsa.xbuf_[0].v, Vector3d::Zero(), 1e-8);
  EXPECT_NEAR(sim.clock_bias_, salsa.xbuf_[0].tau[0], 1e-5);
  EXPECT_NEAR(sim.clock_bias_rate_, salsa.xbuf_[0].tau[1], 1e-5);
}
