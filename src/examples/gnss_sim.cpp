#include "salsa/salsa.h"
#include "salsa/misc.h"

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
        salsa.x_e2n_ = sim.X_e2n_;
        true_state_log.log(sim.t_);
        true_state_log.logVectors(sim.state().X.arr(), sim.state().v, sim.accel_bias_,
                                  sim.gyro_bias_, Vector2d{sim.clock_bias_, sim.clock_bias_rate_},
                                  sim.X_e2n_.arr(), sim.x_b2c_.arr());
    }
}
