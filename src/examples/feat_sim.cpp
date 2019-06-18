#include "salsa/salsa.h"
#include "salsa/test_common.h"
#include "salsa/sim_common.h"

#include "multirotor_sim/simulator.h"

using namespace salsa;
using namespace std;
using namespace multirotor_sim;

int main()
{
    std::string prefix = "/tmp/Salsa/featSimulation/";
    std::experimental::filesystem::remove_all(prefix);

    Simulator sim(true);
    sim.load(imu_mocap());

    Salsa* salsa = initSalsa(prefix + "Feat/", "$\\hat{x}$", sim);
    salsa->update_on_camera_ = true;
    salsa->disable_vision_ = false;
    salsa->disable_solver_ = false;

    Logger true_state_log(prefix + "Truth.log");

    while (sim.run())
    {
        logTruth(true_state_log, sim, *salsa);
        setInit(salsa, sim);
    }
    delete salsa;
}
