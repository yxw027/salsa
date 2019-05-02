#pragma once
#include <deque>
#include <Eigen/Core>

#include "geometry/xform.h"
#include "factors/feat.h"
#include "opencv2/opencv.hpp"
#include "gnss_utils/gtime.h"

namespace salsa
{

class State
{
public:
    double buf_[13];
    int kf;
    int node;
    xform::Xformd x;
    double& t;
    Eigen::Map<Eigen::Vector3d> p;
    Eigen::Map<Eigen::Vector3d> v;
    Eigen::Map<Eigen::Vector2d> tau;

    State();
};
typedef std::vector<salsa::State, Eigen::aligned_allocator<salsa::State>> StateVec;

class Features
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int id; // image label
    double t; // time stamp of this image
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> zetas; // unit vectors to features
    std::vector<double> depths; // feature distances corresponding to feature measurements
    std::vector<int> feat_ids; // feature ids corresonding to pixel measurements
    std::vector<cv::Point2f> pix; // pixel measurements

    void reserve(const int& N);
    void resize(const int& N);
    void rm(const int& idx);
    void clear();
    int size() const;
};

typedef std::deque<FeatFunctor, Eigen::aligned_allocator<FeatFunctor>> FeatDeque;

struct Feat
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int kf0;
    int idx0;
    double rho;
    double rho_true;
    int slide_count;
    Eigen::Vector3d z0;
    FeatDeque funcs;
    bool updated_in_last_image_;

    Feat(int _idx, int _kf0, const Eigen::Vector3d& _z0, double _rho, double _rho_true=NAN);

    void addMeas(int to_idx, const Eigen::Matrix2d& cov, const Eigen::Vector3d& zj);
    void moveMeas(int to_idx, const Eigen::Vector3d& zj);
    bool slideAnchor(int new_from_idx, const StateVec& xbuf, const xform::Xformd& x_b2c);
    Eigen::Vector3d pos(const StateVec& xbuf, const xform::Xformd& x_b2c);
};

struct Obs
{
    gnss_utils::GTime t;
    uint8_t sat_idx; // index in sats_ SatVec
    uint8_t sat;
    uint8_t rcv;
    uint8_t SNR;
    uint8_t LLI; // loss-of-lock indicator
    uint8_t code;
    uint8_t qualL; // carrier phase cov
    uint8_t qualP; // psuedorange cov
    Eigen::Vector3d z; // [prange, doppler, cphase]

    Obs();

    bool operator < (const Obs& other);
};
typedef std::vector<Obs> ObsVec;

}
