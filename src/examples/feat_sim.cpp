#include "salsa/salsa.h"
#include "salsa/misc.h"

#include "multirotor_sim/simulator.h"

using namespace salsa;
using namespace std;

int main()
{
    Simulator sim(true);
    sim.load(imu_feat(false));

    Salsa salsa;
    salsa.init(default_params("/tmp/Salsa/FeatSimulation/"));
    salsa.x_u2c_.q() = sim.q_b2c_;
    salsa.x_u2c_.t() = sim.p_b2c_;
    salsa.cam_ = sim.cam_;

    sim.register_estimator(&salsa);

    Logger true_state_log(salsa.log_prefix_ + "Truth.log");

    while (sim.run())
    {
        if (salsa.current_node_ < 0)
        {
            salsa.current_state_.x = sim.state().X;
            salsa.current_state_.v = sim.state().v;
        }
        true_state_log.log(sim.t_);
        true_state_log.logVectors(sim.state().X.arr(), sim.state().v, sim.accel_bias_,
                                  sim.gyro_bias_, Vector2d{sim.clock_bias_, sim.clock_bias_rate_});
    }

    Logger true_feat_log(salsa.log_prefix_ + "TrueFeat.log");
    for (int i = 0; i < sim.env_.get_points().size(); i++)
    {
        true_feat_log.logVectors(sim.env_.get_points()[i]);
    }
}
