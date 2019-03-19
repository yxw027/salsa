#include <gtest/gtest.h>

#include "salsa/salsa.h"
#include "salsa/test_common.h"

#include "multirotor_sim/simulator.h"

using namespace salsa;

TEST (Vision, isTrackedFeature)
{
    Salsa salsa;

    salsa.xfeat_.insert({0, Feat(0, 0, 0, Vector3d(1, 0, 0), 1.0)});
    salsa.xfeat_.insert({1, Feat(0, 0, 0, Vector3d(1, 0, 0), 1.0)});
    salsa.xfeat_.insert({2, Feat(0, 0, 0, Vector3d(1, 0, 0), 1.0)});

    EXPECT_TRUE(salsa.isTrackedFeature(0));
    EXPECT_TRUE(salsa.isTrackedFeature(1));
    EXPECT_TRUE(salsa.isTrackedFeature(2));
    EXPECT_FALSE(salsa.isTrackedFeature(3));
    EXPECT_FALSE(salsa.isTrackedFeature(12032));
    EXPECT_FALSE(salsa.isTrackedFeature(-10));
}

TEST (Vision, AddsNewFeatures)
{
    Simulator sim(false);
    sim.load(imu_feat(false, 10.0));

    Salsa salsa;
    salsa.init(default_params("/tmp/Salsa/FeatSimulation/"));

    sim.register_estimator(&salsa);

    while (salsa.xfeat_.size() == 0)
    {
        sim.run();
    }

    EXPECT_EQ(sim.env_.get_points().size(), salsa.xfeat_.size());
    for (int i = 0; i < 3; i++)
    {
        Xformd x = sim.state().X;
        double d = (sim.p_I2c_ - sim.env_.get_points()[i]).norm();

        Vector3d pt_hat = x.t() + x.q().rota(sim.q_b2c_.rota(salsa.xfeat_.at(i).z * d) + sim.p_b2c_);
        EXPECT_MAT_NEAR(pt_hat, sim.env_.get_points()[i], 1e-8);
    }
}

TEST (Vision, NewKFRotate)
{
    Simulator sim(false);
    sim.load(imu_feat(false, 10.0));
    sim.p_b2c_.setZero();

    Salsa salsa;
    salsa.init(default_params("/tmp/Salsa/FeatSimulation/"));
    salsa.x_u2c_.q() = sim.q_b2c_;
    salsa.x_u2c_.t() = sim.p_b2c_;

    sim.register_estimator(&salsa);

    sim.t_ = 1.0;
    sim.update_camera_meas();
    EXPECT_GT(salsa.kf_feat_.zetas.size(), 50);

    int rot = 0;
    while (salsa.current_kf_ == 0)
    {
        sim.state().X.q_ += Vector3d(0, -DEG2RAD * 5.0, 0);
        salsa.current_state_.x = sim.state().X;
        sim.t_ += 2;
        sim.update_camera_meas();
        EXPECT_LE(salsa.kf_parallax_, 1e-3);
        rot++;
    }

    EXPECT_GE (rot, 10);
    EXPECT_LE(salsa.kf_Nmatch_feat_, salsa.kf_feature_thresh_);
    EXPECT_LE(salsa.kf_parallax_, 1e-3);
}


TEST (Vision, NewKFTranslate)
{
    Simulator sim(false);
    sim.load(imu_feat(false, 10.0));

    Salsa salsa;
    salsa.init(default_params("/tmp/Salsa/FeatSimulation/"));
    salsa.x_u2c_.q() = sim.q_b2c_;
    salsa.x_u2c_.t() = sim.p_b2c_;

    sim.register_estimator(&salsa);

    sim.t_ = 1.0;
    sim.update_camera_meas();
    EXPECT_GT(salsa.kf_feat_.zetas.size(), 50);

    int step = 0;
    while (salsa.current_kf_ == 0)
    {
        sim.state().X.t_ += Vector3d(0.1, 0, 0);
        salsa.current_state_.x = sim.state().X;
        sim.t_ += 2;
        sim.update_camera_meas();
        step++;
    }

    EXPECT_GE(step, 3);
    EXPECT_GE(salsa.kf_Nmatch_feat_, salsa.kf_feature_thresh_);
    EXPECT_GE(salsa.kf_parallax_, salsa.kf_parallax_thresh_);
}

