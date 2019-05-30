#include "salsa/salsa.h"
#include "salsa/misc.h"

#include "multirotor_sim/simulator.h"

using namespace salsa;
using namespace std;
using namespace multirotor_sim;

int main()
{
    Simulator sim(true);
    sim.load(imu_feat_gnss());

    Salsa salsa;
    salsa.init(default_params("/tmp/Salsa/MixedSimulation/"));
    salsa.x_b2c_ = sim.x_b2c_;
    salsa.x_e2n_ = sim.X_e2n_;
    salsa.cam_ = sim.cam_;

    sim.register_estimator(&salsa);

    Logger true_state_log(salsa.log_prefix_ + "Truth.log");

    while (sim.run())
    {
        salsa.x0_ = sim.state().X;
        salsa.v0_ = sim.state().v;
        true_state_log.log(sim.t_);
        true_state_log.logVectors(sim.state().X.arr(), sim.state().v, sim.accel_bias_,
                                  sim.gyro_bias_, Vector2d{sim.clock_bias_, sim.clock_bias_rate_},
                                  sim.X_e2n_.arr(), sim.x_b2c_.arr());
    }

    Logger true_feat_log(salsa.log_prefix_ + "TrueFeat.log");
    for (int i = 0; i < sim.env_.get_points().size(); i++)
    {
        true_feat_log.logVectors(sim.env_.get_points()[i]);
    }
}
