#include <gtest/gtest.h>
#include <ceres/ceres.h>

#include "multirotor_sim/simulator.h"
#include "multirotor_sim/controller.h"

#include "salsa/salsa.h"

TEST (Salsa, MocapSimulation)
{
    Salsa<10> salsa;

    ReferenceController cont;
    cont.load("../lib/multirotor_sim/params/sim_params.yaml");
    Simulator sim(cont, cont, true);
    sim.load("../lib/multirotor_sim/params/sim_params.yaml");

    sim.register_estimator(&salsa);

    while (sim.run()) {}
}
