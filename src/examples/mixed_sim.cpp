#include <experimental/filesystem>

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
    sim.load(imu_feat_gnss());
    std::string prefix = "/tmp/Salsa/mixedSimulation/";
    std::experimental::filesystem::remove_all(prefix);

    Salsa* salsa = initSalsa(prefix + "GNSSSwitchingVision/", "GV+$\\kappa$", sim);
    salsa->disable_vision_ = false;
    salsa->disable_gnss_ = false;
    salsa->update_on_gnss_ = false;
    salsa->enable_switching_factors_ = true;


    Logger true_state_log(prefix + "Truth.log");
    while (sim.run())
    {
        logTruth(true_state_log, sim, *salsa);

        setInit(salsa, sim);
    }
    delete salsa;
}
