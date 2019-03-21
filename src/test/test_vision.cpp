#include <gtest/gtest.h>

#include "salsa/salsa.h"
#include "salsa/test_common.h"

#include "multirotor_sim/simulator.h"
#include "multirotor_sim/estimator_wrapper.h"

using namespace salsa;

TEST (Vision, isTrackedFeature)
{
    Salsa salsa;

    salsa.xfeat_.insert({0, Feat(0, 0, Vector3d(1, 0, 0), 1.0)});
    salsa.xfeat_.insert({1, Feat(0, 0, Vector3d(1, 0, 0), 1.0)});
    salsa.xfeat_.insert({2, Feat(0, 0, Vector3d(1, 0, 0), 1.0)});

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

        Vector3d pt_hat = x.t() + x.q().rota(sim.q_b2c_.rota(salsa.xfeat_.at(i).z0 * d) + sim.p_b2c_);
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

    salsa.imu_[0].cov_= sim.imu_R_;
    sim.t_ = 1.0;
    sim.update_camera_meas();
    EXPECT_GT(salsa.kf_feat_.zetas.size(), 50);

    int rot = 0;
    double t = 1.0;
    while (salsa.current_kf_== 0)
    {
        Vector3d w = Vector3d(0, -DEG2RAD * 5.0, 0);
        for (int i = 0; i < 5; i++)
        {
            sim.t_ += 0.02;
            sim.state().X.q_ += w * 0.02;
            Vector6d imu;
            imu << sim.state().X.q_.rotp(gravity_), w;
            salsa.imuCallback(sim.t_, imu, sim.imu_R_);
        }
        sim.update_camera_meas();
        salsa.current_state_.x = sim.state().X;
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
    salsa.imu_[0].cov_ = sim.imu_R_;

    sim.t_ = 1.0;
    sim.update_camera_meas();
    EXPECT_GT(salsa.kf_feat_.zetas.size(), 50);

    int step = 0;
    while (salsa.current_kf_ == 0)
    {
        for (int i = 0; i < 5; i++)
        {
            sim.t_ += 0.02;
            sim.state().X.t_ += Vector3d(0.02, 0, 0);
            Vector6d imu;
            imu << gravity_, Vector3d(0,0,0);
            salsa.imuCallback(sim.t_, imu, sim.imu_R_);
        }
        salsa.current_state_.x = sim.state().X;
        sim.update_camera_meas();
        step++;
    }

    EXPECT_GE(step, 3);
    EXPECT_GE(salsa.kf_Nmatch_feat_, salsa.kf_feature_thresh_);
    EXPECT_GE(salsa.kf_parallax_, salsa.kf_parallax_thresh_);
}

TEST (Vision, SlideAnchor)
{
    Simulator sim(false);
    sim.load(imu_feat(false, 10.0));

    Feat* feat = nullptr;
    int kf = 0;
    int idx = 0;
    Salsa salsa;
    salsa.init(default_params("/tmp/Salsa/FeatSimulation/"));
    salsa.x_u2c_.q() = sim.q_b2c_;
    salsa.x_u2c_.t() = sim.p_b2c_;
    Camera<double> cam = salsa.cam_;

    salsa::State xbuf[10];
    double rho[10];

    EstimatorWrapper est;
    auto img_cb = [&rho, &feat, &kf, &idx, &cam, &salsa, &sim, &xbuf]
            (const double& t, const ImageFeat& z, const Matrix2d& R_pix, const Matrix1d& R_depth)
    {
        Vector3d zeta = cam.invProj(z.pixs[0], 1.0);
        if (!feat)
            feat = new Feat(idx, kf, zeta, 1.0/z.depths[0]);
        else
            feat->addMeas(idx, salsa.x_u2c_, R_pix, zeta);
        xbuf[idx].x = sim.state().X;
        xbuf[idx].v = sim.state().v;
        xbuf[idx].t = sim.t_;
        xbuf[idx].kf = kf;
        xbuf[idx].tau.setZero();
        rho[idx] = 1.0/z.depths[0];
        kf += 1;
        idx += 1;
    };
    est.register_feat_cb(img_cb);
    sim.register_estimator(&est);

    while (kf < 10)
    {
        sim.run();
    }

    Vector3d pt = sim.env_.get_points()[0];
    EXPECT_MAT_NEAR(pt, feat->pos(xbuf, salsa.x_u2c_), 1e-5);
    for (int i = 1; i < 9; i++)
    {
        EXPECT_TRUE(feat->slideAnchor(i, i, xbuf, salsa.x_u2c_));
        EXPECT_NEAR(feat->rho, rho[i], 1e-5);
        EXPECT_MAT_NEAR(pt, feat->pos(xbuf, salsa.x_u2c_), 1e-5);
    }
    EXPECT_FALSE(feat->slideAnchor(9, 9, xbuf, salsa.x_u2c_));
    delete feat;
}

TEST (Vision, KeyframeCleanup)
{
    Simulator sim(false);
    sim.load(imu_feat(false, 10.0));

    Salsa salsa;
    salsa.init(default_params("/tmp/Salsa/FeatSimulation/"));
    salsa.x_u2c_.q() = sim.q_b2c_;
    salsa.x_u2c_.t() = sim.p_b2c_;

    sim.register_estimator(&salsa);

    while (salsa.current_kf_ <= salsa.N_)
    {
        sim.run();
    }

    for (auto ft : salsa.xfeat_)
    {
        EXPECT_NE(ft.second.kf0, 0);
    }

}

