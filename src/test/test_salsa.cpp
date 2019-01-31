#include <gtest/gtest.h>
#include <ceres/ceres.h>

#include "multirotor_sim/simulator.h"
#include "multirotor_sim/controller.h"

#include "salsa/salsa.h"
#include "salsa/test_common.h"


TEST (Salsa, MocapSimulation)
{
  Salsa salsa;

  Simulator sim(true);
  sim.load("../lib/multirotor_sim/params/sim_params.yaml");
  sim.register_estimator(&salsa);

  sim.tmax_ = 1.0;


  while (sim.run()){}

}
