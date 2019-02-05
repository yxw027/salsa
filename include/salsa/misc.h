#pragma once

#include <cmath>
#include <gtest/gtest.h>

#ifndef NDEBUG
#define SALSA_ASSERT(condition, ...) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << printf(__VA_ARGS__) << std::endl; \
            assert(condition); \
        } \
    } while (false)
#else
#   define SALSA_ASSERT(...)
#endif


#pragma once

#include <functional>
#include <Eigen/Core>

#include "multirotor_sim/estimator_base.h"
#include "multirotor_sim/state.h"
#include "geometry/quat.h"
#include "geometry/xform.h"

using namespace Eigen;
using namespace quat;
using namespace xform;
using namespace multirotor_sim;

//class EstimatorWrapper : public EstimatorBase
//{
//public:
//    inline void imuCallback(const double& t, const Vector6d& z, const Matrix6d& R) override { if (imu_cb_) imu_cb_(t, z, R); }
//    inline void altCallback(const double& t, const Vector1d& z, const Matrix1d& R) override { if (alt_cb_) alt_cb_(t, z, R); }
//    inline void mocapCallback(const double& t, const Xformd& z, const Matrix6d& R) override { if (mocap_cb_) mocap_cb_(t, z, R); }
//    inline void voCallback(const double& t, const Xformd& z, const Matrix6d& R) override { if (vo_cb_) vo_cb_(t, z, R); }
//    inline void imageCallback(const double& t, const Image& z, const Matrix2d& R_pix, const Matrix1d& R_depth) override { if (image_cb_) image_cb_(t, z, R_pix, R_depth); }
//    inline void gnssCallback(const double& t, const Vector6d& z, const Matrix6d& R) override { if (gnss_cb_) gnss_cb_(t, z, R); }
//    inline void rawGnssCallback(const GTime& t, const Vector3d& z, const Matrix3d& R, Satellite& sat) override { if (raw_gnss_cb_) raw_gnss_cb_(t, z, R, sat); }

//    std::function<void(const double& t, const Vector6d& z, const Matrix6d& R)> imu_cb_;
//    std::function<void(const double& t, const Vector1d& z, const Matrix1d& R)> alt_cb_;
//    std::function<void(const double& t, const Xformd& z, const Matrix6d& R)> mocap_cb_;
//    std::function<void(const double& t, const Xformd& z, const Matrix6d& R)> vo_cb_;
//    std::function<void(const double& t, const Image& z, const Matrix2d& R_pix, const Matrix1d& R_depth)> image_cb_;
//    std::function<void(const double& t, const Vector6d& z, const Matrix6d& R)> gnss_cb_;
//    std::function<void(const GTime& t, const Vector3d& z, const Matrix3d& R, Satellite& sat)> raw_gnss_cb_;

//    inline void register_imu_cb(std::function<void(const double& t, const Vector6d& z, const Matrix6d& R)> imu_cb) {imu_cb_ = imu_cb;}
//    inline void register_alt_cb(std::function<void(const double& t, const Vector1d& z, const Matrix1d& R)> alt_cb) {alt_cb_ = alt_cb;}
//    inline void register_pos_cb(std::function<void(const double& t, const Xformd& z, const Matrix6d& R)> mocap_cb) {mocap_cb_ = mocap_cb;}
//    inline void register_vo_cb(std::function<void(const double& t, const Xformd& z, const Matrix6d& R)> vo_cb) {vo_cb_ = vo_cb;}
//    inline void register_image_cb(std::function<void(const double& t, const Image& z, const Matrix2d& R_pix, const Matrix1d& R_depth)> image_cb) {image_cb_ = image_cb;}
//    inline void register_gnss_cb(std::function<void(const double& t, const Vector6d& z, const Matrix6d& R)> gnss_cb) {gnss_cb_ = gnss_cb;}
//    inline void register_raw_gnss_cb(std::function<void(const GTime& t, const Vector3d& z, const Matrix3d& R, Satellite& sat)> raw_gnss_cb) {raw_gnss_cb_ = raw_gnss_cb;}
//};

