#include "salsa/state.h"
#include "salsa/misc.h"

using namespace Eigen;
using namespace xform;

namespace salsa
{

State::State() :
    t(buf_[0]),
    x(buf_+1),
    p(buf_+1),
    v(buf_+8),
    tau(buf_+11)
{
    for (int i = 0; i < 13; i++)
        buf_[i] = NAN;
    kf = -1;
    type = None;
}

State& State::operator=(const State& other)
{
    t = other.t;
    x = other.x;
    v = other.v;
    tau = other.tau;
    kf = other.kf;
    node = other.node;
}

Obs::Obs()
{
    sat_idx = -1;
}
bool Obs::operator <(const Obs& other)
{
    return sat_idx < other.sat_idx;
}






}
