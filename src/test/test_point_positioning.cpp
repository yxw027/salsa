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
  salsa.init("../params/salsa.yaml");

  sim.load(imu_raw_gnss(false));
  sim.register_estimator(&salsa);
  salsa.x_e2n_ = sim.X_e2n_;
  salsa.update_on_gnss_ = true;

  multirotor_sim::State x;
  x.p << 16, 4, 8;
  x.v << 3, -4, 5;
  sim.dyn_.set_state(x);
  sim.clock_bias_ = 1.4e-8;
  sim.clock_bias_rate_ = 2.6e-9;
  while (salsa.current_node_ == -1)
  {
    sim.run();
  }

  EXPECT_MAT_NEAR(sim.state().X.t(), salsa.xbuf_[0].x.t(), 1e-5);
  EXPECT_MAT_NEAR(sim.state().q.rota(sim.state().v), salsa.xbuf_[0].v, 1e-5);
  EXPECT_NEAR(sim.clock_bias_, salsa.xbuf_[0].tau[0], 1e-5);
  EXPECT_NEAR(sim.clock_bias_rate_, salsa.xbuf_[0].tau[1], 1e-5);
}
