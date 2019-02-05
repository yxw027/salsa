#include <gtest/gtest.h>
#include <ceres/ceres.h>

#include "multirotor_sim/simulator.h"
#include "multirotor_sim/controller.h"

#include "salsa/salsa.h"
#include "salsa/test_common.h"


TEST (Salsa, MocapSimulation)
{
  Salsa salsa;
  salsa.init("../params/salsa.yaml");

  Simulator sim(true);
  sim.load("../lib/multirotor_sim/params/sim_params.yaml");
  sim.register_estimator(&salsa);

  Logger truth_log("/tmp/Salsa.MocapSimulation.Truth.log");

  while (sim.run())
  {
    truth_log.log(sim.t_);
    truth_log.logVectors(sim.state().X.arr(), sim.state().v);
  }
}
