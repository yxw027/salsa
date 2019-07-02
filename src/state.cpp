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
    tau(buf_+11),
    bias(buf_+13)
{
    for (int i = 0; i < 19; i++)
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
    bias = other.bias;
    kf = other.kf;
    node = other.node;
    type = other.type;
    n_cam = other.n_cam;
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
