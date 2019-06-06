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

    Salsa salsa;
    salsa.init(default_params("/tmp/Salsa/RawGNSSSimulation/"));
    salsa.x0_ = sim.state().X;
    salsa.x_e2n_ = sim.X_e2n_;
    salsa.update_on_gnss_ = true;

    sim.register_estimator(&salsa);

    Logger true_state_log(salsa.log_prefix_ + "Truth.log");

    while (sim.run())
    {
        salsa.x0_ = sim.state().X;
        salsa.v0_ = sim.state().v;
        salsa.x_e2n_ = sim.X_e2n_;
        logTruth(true_state_log, sim);
    }
}
