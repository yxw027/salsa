#pragma once
#include <vector>
#include <Eigen/Core>

#include "geometry/xform.h"

#include "gnss_utils/gtime.h"

namespace salsa
{

class State
{
public:
    enum {
        xSize = 12,
        dxSize = 11
    };
    enum {
        None = 0x00,
        Gnss = 0x01,
        Mocap = 0x02,
        Camera = 0x04,
    };
    typedef Eigen::Matrix<double, dxSize, dxSize> dxMat;
    double buf_[13];
    int kf;
    int node;
    uint8_t type;
    uint8_t n_cam;
    xform::Xformd x;
    double& t;
    Eigen::Map<Eigen::Vector3d> p;
    Eigen::Map<Eigen::Vector3d> v;
    Eigen::Map<Eigen::Vector2d> tau;

    State();
    State& operator=(const State& other);
};
typedef std::vector<salsa::State, Eigen::aligned_allocator<salsa::State>> StateVec;

struct Obs
{
    gnss_utils::GTime t;
    uint8_t sat_idx; // index in sats_ SatVec
    uint8_t sat;
    uint8_t rcv;
    uint8_t SNR;
    uint8_t LLI; // loss-of-lock indicator
    uint8_t code;
    double qualL; // carrier phase cov
    double qualP; // psuedorange cov
    Eigen::Vector3d z; // [prange, doppler, cphase]

    Obs();

    bool operator < (const Obs& other);
};
typedef std::vector<Obs> ObsVec;

}
