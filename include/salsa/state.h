#pragma once
#include <deque>
#include <Eigen/Core>

#include "geometry/xform.h"
#include "factors/feat.h"
#include "opencv2/opencv.hpp"

using namespace Eigen;
using namespace xform;

namespace salsa
{

class State
{
public:
    double buf_[13];
    int kf;
    int node;
    Xformd x;
    double& t;
    Map<Vector3d> v;
    Map<Vector2d> tau;

    State();
};
typedef std::vector<salsa::State, aligned_allocator<salsa::State>> StateVec;

class Features
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int id; // image label
    double t; // time stamp of this image
    std::vector<Vector3d, aligned_allocator<Vector3d>> zetas; // unit vectors to features
    std::vector<double> depths; // feature distances corresponding to feature measurements
    std::vector<int> feat_ids; // feature ids corresonding to pixel measurements
    std::vector<cv::Point2f> pix; // pixel measurements

    void reserve(const int& N);
    void resize(const int& N);
    void rm(const int& idx);
    void clear();
    int size() const;
};

typedef std::deque<FeatFunctor, aligned_allocator<FeatFunctor>> FeatDeque;

struct Feat
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int kf0;
    int idx0;
    double rho;
    double rho_true;
    int slide_count;
    Vector3d z0;
    FeatDeque funcs;
    bool updated_in_last_image_;

    Feat(int _idx, int _kf0, const Vector3d& _z0, double _rho, double _rho_true=NAN);

    void addMeas(int to_idx, const Xformd& x_b2c, const Matrix2d& cov, const Vector3d& zj);
    void moveMeas(int to_idx, const Vector3d& zj);
    bool slideAnchor(int new_from_idx, int new_from_kf, const StateVec& xbuf, const Xformd& x_b2c);
    Vector3d pos(const StateVec& xbuf, const Xformd& x_b2c);
};

}
