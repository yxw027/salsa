#pragma once

#include "salsa/logger.h"
#include "multirotor_sim/simulator.h"

inline void logTruth(salsa::Logger& log, multirotor_sim::Simulator& sim, salsa::Salsa& salsa)
{
    log.log(sim.t_);
    int32_t multipath = sim.multipath_area_.inside(sim.get_gps_position_ned());
    int32_t denied = sim.gps_denied_area_.inside(sim.get_gps_position_ned());
    log.logVectors(sim.state().X.arr(), sim.state().v, sim.accel_bias_,
                   sim.gyro_bias_, Vector2d{sim.clock_bias_, sim.clock_bias_rate_},
                   sim.X_e2n_.arr(), sim.x_b2c_.arr());
    log.log(multipath, denied);
    for (int i = 0; i < salsa.ns_; i++)
    {
        if (i < sim.satellites_.size())
        {
            log.log((double)(sim.multipath_offset_[i] == 0));
        }
        else
        {
            log.log((double)NAN);
        }
    }
}
