#include <gtest/gtest.h>
#include <ceres/ceres.h>

#include "multirotor_sim/simulator.h"
#include "multirotor_sim/controller.h"

#include "salsa/salsa.h"
#include "salsa/test_common.h"


TEST (Salsa, MocapSimulation)
{
  Simulator sim(true);
  sim.load("../lib/multirotor_sim/params/sim_params.yaml");
  sim.vo_enabled_ = false;
  sim.mocap_enabled_ = true;
  sim.alt_enabled_ = false;
  sim.gnss_enabled_ = false;
  sim.raw_gnss_enabled_ = false;
  sim.tmax_ = 10.0;

  string filename = "/tmp/Salsa.tmp.yaml";
  ofstream tmp(filename);
  YAML::Node node;
  node["x_u2m"] = std::vector<double>{0, 0, 0, 1, 0, 0, 0};
  node["x_u2c"] = std::vector<double>{0, 0, 0, 1, 0, 0, 0};
  node["x_u2b"] = std::vector<double>{0, 0, 0, 1, 0, 0, 0};
  node["dt_m"] = 0.0;
  node["dt_c"] = 0.0;
  node["log_prefix"] = "/tmp/Salsa.MocapSimulation";
  node["R_clock_bias"] = std::vector<double>{1e-6, 1e-8};
  node["switch_weight"] = 10.0;
  tmp << node;
  tmp.close();

  Salsa salsa;
  salsa.init(filename);

  sim.register_estimator(&salsa);

  Logger truth_log(salsa.log_prefix_ + ".Truth.log");

  while (sim.run())
  {
    truth_log.log(sim.t_);
    truth_log.logVectors(sim.state().X.arr(), sim.state().v);
  }
}
