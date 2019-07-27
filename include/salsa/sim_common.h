#pragma once

#include "salsa/salsa.h"
#include "salsa/test_common.h"

#include "multirotor_sim/simulator.h"
using namespace salsa;
using namespace multirotor_sim;

Salsa* initSalsa(const std::string& prefix, const std::string& label, Simulator& sim)
{
    Salsa* salsa = new Salsa;
    salsa->init(default_params(prefix, label));
    salsa->x0_ = sim.state().X;
    salsa->v0_ = sim.state().v;
    salsa->x_e2n_ = sim.X_e2n_;
    salsa->x_b2c_ = sim.x_b2c_;
    salsa->x_b2o_ = xform::Xformd::Identity();

    salsa->cam_ = sim.cam_;
    salsa->mask_.create(cv::Size(salsa->cam_.image_size_(0), salsa->cam_.image_size_(1)), CV_8UC1);
    salsa->mask_ = 255;
    sim.register_estimator(salsa);
    return salsa;
}

void setInit(Salsa* salsa, Simulator& sim)
{
    salsa->x0_ = sim.state().X;
    salsa->v0_ = sim.state().v;
    salsa->x_e2n_ = sim.X_e2n_;
}

inline void logTruth(salsa::Logger& log, multirotor_sim::Simulator& sim, salsa::Salsa& salsa)
{
    log.log(sim.t_);
    Eigen::Vector3d lla = Eigen::Vector3d::Ones() * NAN;
    int32_t multipath = sim.multipath_area_.inside(sim.get_gps_position_ned());
    int32_t denied = sim.gps_denied_area_.inside(sim.get_gps_position_ned());
    log.logVectors(sim.state().X.arr(), sim.state().X.q().euler(), sim.state().v, sim.accel_bias_,
                   sim.gyro_bias_, Vector2d{sim.clock_bias_, sim.clock_bias_rate_},
                   sim.X_e2n_.arr(), sim.x_b2c_.arr(), lla);
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
