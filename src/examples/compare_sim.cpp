#include <experimental/filesystem>

#include "salsa/salsa.h"
#include "salsa/test_common.h"
#include "salsa/sim_common.h"

#include "multirotor_sim/simulator.h"

using namespace salsa;
using namespace std;
using namespace multirotor_sim;

Salsa* initSalsa(const std::string& prefix, const std::string& label, Simulator& sim)
{
    Salsa* salsa = new Salsa;
    salsa->init(default_params(prefix, label));
    salsa->x0_ = sim.state().X;
    salsa->v0_ = sim.state().v;
    salsa->x_e2n_ = sim.X_e2n_;
    salsa->x_b2c_ = sim.x_b2c_;
    salsa->cam_ = sim.cam_;
    sim.register_estimator(salsa);
    return salsa;
}

void setInit(Salsa* salsa, Simulator& sim)
{
    salsa->x0_ = sim.state().X;
    salsa->v0_ = sim.state().v;
    salsa->x_e2n_ = sim.X_e2n_;
}

int main()
{
    Simulator sim(true);
    sim.load(imu_feat_gnss());
    std::string prefix = "/tmp/Salsa/compareSimulation/";
    std::experimental::filesystem::remove_all(prefix);

    Salsa* gnss_only = initSalsa(prefix + "GNSSOnly/", "G", sim);
    gnss_only->update_on_gnss_ = true;
    gnss_only->enable_switching_factors_ = false;
    gnss_only->disable_vision_ = true;

    Salsa* switching_salsa = initSalsa(prefix + "Switching/", "G+$\\kappa$", sim);
    switching_salsa->disable_vision_ = true;
    switching_salsa->update_on_gnss_ = true;

    Salsa* vision_salsa = initSalsa(prefix + "Vision/", "V", sim);
    vision_salsa->disable_vision_ = false;
    vision_salsa->disable_gnss_ = true;
    vision_salsa->update_on_gnss_ = false;

    Salsa* vision_gnss_salsa = initSalsa(prefix + "GNSSVision/", "GV", sim);
    vision_gnss_salsa->disable_vision_ = false;
    vision_gnss_salsa->disable_gnss_ = false;
    vision_gnss_salsa->update_on_gnss_ = false;
    vision_gnss_salsa->enable_switching_factors_ = false;

    Salsa* vision_switching_gnss_salsa = initSalsa(prefix + "GNSSSwitchingVision/", "GV+$\\kappa$", sim);
    vision_switching_gnss_salsa->disable_vision_ = false;
    vision_switching_gnss_salsa->disable_gnss_ = false;
    vision_switching_gnss_salsa->update_on_gnss_ = false;
    vision_switching_gnss_salsa->enable_switching_factors_ = true;


    Logger true_state_log(prefix + "Truth.log");
    while (sim.run())
    {
        logTruth(true_state_log, sim, *vision_salsa);

        setInit(gnss_only, sim);
        setInit(switching_salsa, sim);
        setInit(vision_salsa, sim);
        setInit(vision_gnss_salsa, sim);
        setInit(vision_switching_gnss_salsa, sim);
    }

    delete gnss_only;
    delete switching_salsa;
    delete vision_salsa;
    delete vision_gnss_salsa;
    delete vision_switching_gnss_salsa;
}
