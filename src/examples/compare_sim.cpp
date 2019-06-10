#include "salsa/salsa.h"
#include "salsa/test_common.h"
#include "salsa/sim_common.h"

#include "multirotor_sim/simulator.h"

using namespace salsa;
using namespace std;
using namespace multirotor_sim;

int main()
{
    Simulator sim(true);
    sim.load(imu_raw_gnss());
//#ifndef NDEBUG
//    sim.tmax_ = 10;
//#endif
    std::string prefix = "/tmp/Salsa/compareSimulation/";

    Salsa switching_salsa;
    switching_salsa.init(default_params(prefix + "Switching/", "switching"));
    switching_salsa.x0_ = sim.state().X;
    switching_salsa.x_e2n_ = sim.X_e2n_;
    switching_salsa.update_on_gnss_ = true;

    Salsa vanilla_salsa;
    vanilla_salsa.init(default_params(prefix + "NoSwitching/", "no switching"));
    vanilla_salsa.x0_ = sim.state().X;
    vanilla_salsa.x_e2n_ = sim.X_e2n_;
    vanilla_salsa.update_on_gnss_ = true;
    vanilla_salsa.enable_switching_factors_ = false;

    sim.register_estimator(&switching_salsa);
    sim.register_estimator(&vanilla_salsa);

    Logger true_state_log(prefix + "Truth.log");

    while (sim.run())
    {
        vanilla_salsa.x0_ = switching_salsa.x0_ = sim.state().X;
        vanilla_salsa.v0_ = switching_salsa.v0_ = sim.state().v;
        vanilla_salsa.x_e2n_ = switching_salsa.x_e2n_ = sim.X_e2n_;
        logTruth(true_state_log, sim, switching_salsa);
    }
}
