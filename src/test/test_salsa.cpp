#include <gtest/gtest.h>
#include <ceres/ceres.h>

#include "multirotor_sim/simulator.h"
#include "multirotor_sim/controller.h"

#include "salsa/salsa.h"


TEST (Salsa, MocapSimulation)
{
  Salsa salsa;

  Simulator sim;
  sim.register_estimator(&salsa);


  while (sim.run())
  {

  }

}
