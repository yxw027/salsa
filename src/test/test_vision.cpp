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
        double d = (sim.x_I2c_.t() - sim.env_.get_points()[i]).norm();

        Vector3d pt_hat = x.t() + x.q().rota(sim.x_b2c_.rota(salsa.xfeat_.at(i).z0 * d) + sim.x_b2c_.t());
        EXPECT_MAT_NEAR(pt_hat, sim.env_.get_points()[i], 1e-8);
    }
}

TEST (Vision, NewKFRotate)
{
    Simulator sim(false);
    sim.load(imu_feat(false, 10.0));
    sim.x_b2c_.t().setZero();

    Salsa salsa;
    salsa.init(default_params("/tmp/Salsa/FeatSimulation/"));
    salsa.x_u2c_ = sim.x_b2c_;


    sim.register_estimator(&salsa);

    salsa.imu_[0].cov_= sim.imu_R_;
    sim.t_ = 1.0;
    sim.update_camera_meas();
    EXPECT_EQ(salsa.kf_feat_.zetas.size(), sim.num_features_);

    int kf_condition;
    auto kf_cb = [&kf_condition](int kf, int condition)
    {
        kf_condition = condition;
    };
    salsa.new_kf_cb_ = kf_cb;

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

    EXPECT_EQ(kf_condition, Salsa::INSUFFICIENT_MATCHES);
    EXPECT_LE(salsa.kf_Nmatch_feat_, salsa.kf_feature_thresh_ * salsa.kf_num_feat_);
    EXPECT_LE(salsa.kf_parallax_, 1e-3);
}


TEST (Vision, NewKFTranslate)
{
    Simulator sim(false);
    sim.load(imu_feat(false, 10.0));

    Salsa salsa;
    salsa.init(default_params("/tmp/Salsa/FeatSimulation/"));
    salsa.x_u2c_ = sim.x_b2c_;

    sim.register_estimator(&salsa);
    salsa.imu_[0].cov_ = sim.imu_R_;

    sim.t_ = 1.0;
    sim.update_camera_meas();
    EXPECT_EQ(salsa.kf_feat_.zetas.size(), sim.num_features_);

    int kf_condition;
    auto kf_cb = [&kf_condition](int kf, int condition)
    {
        kf_condition = condition;
    };
    salsa.new_kf_cb_ = kf_cb;

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

    EXPECT_EQ(kf_condition, Salsa::TOO_MUCH_PARALLAX);
    EXPECT_GE(salsa.kf_Nmatch_feat_, salsa.kf_feature_thresh_ * salsa.kf_num_feat_);
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
    salsa.x_u2c_ = sim.x_b2c_;
    Camera<double> cam = salsa.cam_;

    StateVec xbuf(10);
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
    Vector2d res;
    feat->funcs.front()(xbuf[feat->idx0].x.data(), xbuf[feat->funcs.front().to_idx_].x.data(), &feat->rho, res.data());
    for (int i = 1; i < 9; i++)
    {
        EXPECT_TRUE(feat->slideAnchor(i, i, xbuf, salsa.x_u2c_));
        feat->funcs.front()(xbuf[feat->idx0].x.data(), xbuf[feat->funcs.front().to_idx_].x.data(), &feat->rho, res.data());
        EXPECT_MAT_NEAR(res, Vector2d::Zero(), 1e-5);
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
    salsa.x_u2c_ = sim.x_b2c_;

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

TEST (Vision, HandleFeatureHandoff)
{
    Salsa salsa;
    salsa.init(small_feat_test("/tmp/Salsa/ManualKFTest/"));
    salsa.x_u2c_.q() = quat::Quatd::Identity();
    salsa.x_u2c_.t() = Vector3d::Zero();

    MatrixXd l;
    l.resize(4,3);
    l << 1,  0, 1,
         0,  1, 1,
         0, -1, 1,
        -1,  0, 1;


    Features feat;
    feat.resize(4);
    for (int k = 0; k < 4; k++)
    {
        feat.zetas[k] = l.row(k).transpose().normalized();
        Vector2d pix = salsa.cam_.proj(feat.zetas[k]);
        feat.pix[k].x = pix.x();
        feat.pix[k].y = pix.y();
        feat.depths[k] = l.row(k).transpose().norm();
        feat.feat_ids[k] = k;
    }
    feat.id = 0;
    feat.t = 0.0;

    Matrix2d R_pix = Matrix2d::Identity();
    Matrix1d R_depth = Matrix1d::Identity();
    Vector6d imu = (Vector6d() << 0.1, 0, -9.80665, Vector3d::Zero()).finished();
    Matrix6d R_imu = Matrix6d::Identity();

    double t = 0.0;
    double dt = 0.02;

    salsa.imageCallback(t, feat, R_pix, salsa.calcNewKeyframeCondition(feat));
    EXPECT_LT(salsa.summary_.initial_cost, 1e-18);

    EXPECT_EQ(salsa.xbuf_head_, 0);
    EXPECT_EQ(salsa.xbuf_tail_, 0);
    EXPECT_MAT_NEAR(salsa.xbuf_[salsa.xbuf_head_].x.arr(), xform::Xformd::Identity().arr(), 1e-8);
    EXPECT_MAT_NEAR(salsa.xbuf_[salsa.xbuf_head_].v, Vector3d::Zero(), 1e-8);
    EXPECT_MAT_NEAR(salsa.xbuf_[salsa.xbuf_head_].tau, Vector2d::Zero(), 1e-8);
    EXPECT_FLOAT_EQ(salsa.xbuf_[salsa.xbuf_head_].t, 0.0);
    EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].node, 0);
    EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, 0);
    for (auto& f : salsa.xfeat_)
    {
        EXPECT_EQ(f.second.funcs.size(), 0);
        EXPECT_EQ(f.second.kf0, 0);
        EXPECT_EQ(f.second.idx0, 0);
    }

    for (int j = 0; j < 10; j++)
    {
        for (int i = 0; i < 3; i++)
        {
            t += dt;
            salsa.imuCallback(t, imu, R_imu);
        }

        feat.id += 1;
        feat.t = t;
        for (int k = 0; k < 4; k++)
        {
            feat.zetas[k] = (l.row(k).transpose() - salsa.current_state_.x.t()).normalized();
            feat.depths[k] = (l.row(k).transpose() - salsa.current_state_.x.t()).norm();
        }
        salsa.imageCallback(t, feat, R_pix, salsa.calcNewKeyframeCondition(feat));
        EXPECT_LT(salsa.summary_.initial_cost, 1e-18);

        EXPECT_EQ(salsa.xbuf_head_, 1);
        EXPECT_EQ(salsa.xbuf_tail_, 0);
        EXPECT_MAT_NEAR(salsa.xbuf_[salsa.xbuf_head_].tau, Vector2d::Zero(), 1e-8);
        EXPECT_FLOAT_EQ(salsa.xbuf_[salsa.xbuf_head_].t, t);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].node, 1);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, -1);
        for (auto& f : salsa.xfeat_)
        {
            EXPECT_EQ(f.second.funcs.size(), 1);
            EXPECT_EQ(f.second.kf0, 0);
            EXPECT_EQ(f.second.idx0, 0);
            for (auto& func : f.second.funcs)
            {
                EXPECT_EQ(func.to_idx_, 1);
                EXPECT_MAT_NEAR(func.zetai_, f.second.z0, 1e-16);
            }
        }
    }

    // Deliberately cause a new keyframe
    feat.feat_ids[0] = 4;
    feat.feat_ids[1] = 5;
    feat.feat_ids[2] = 2;
    feat.feat_ids[3] = 3;
    for (int i = 0; i < 3; i++)
    {
        t += dt;
        salsa.imuCallback(t, imu, R_imu);
    }

    feat.id += 1;
    feat.t = t;
    for (int k = 0; k < 4; k++)
    {
        feat.zetas[k] = (l.row(k).transpose() - salsa.current_state_.x.t()).normalized();
        feat.depths[k] = (l.row(k).transpose() - salsa.current_state_.x.t()).norm();
    }
    salsa.imageCallback(t, feat, R_pix, salsa.calcNewKeyframeCondition(feat));
    EXPECT_LT(salsa.summary_.initial_cost, 1e-18);

    EXPECT_EQ(salsa.xbuf_head_, 1);
    EXPECT_EQ(salsa.xbuf_tail_, 0);
    EXPECT_MAT_NEAR(salsa.xbuf_[salsa.xbuf_head_].tau, Vector2d::Zero(), 1e-8);
    EXPECT_FLOAT_EQ(salsa.xbuf_[salsa.xbuf_head_].t, t);
    EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].node, 1);
    EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, 1);
    EXPECT_EQ(salsa.xfeat_.size(), 4);
    for (int i = 2; i < 6; i++)
    {
        if (i < 4)
        {
            EXPECT_EQ(salsa.xfeat_.at(i).funcs.size(), 1);
            EXPECT_EQ(salsa.xfeat_.at(i).kf0, 0);
            EXPECT_EQ(salsa.xfeat_.at(i).idx0, 0);
            EXPECT_EQ(salsa.xfeat_.at(i).slide_count, 0);
        }
        else
        {
            EXPECT_EQ(salsa.xfeat_.at(i).funcs.size(), 0);
            EXPECT_EQ(salsa.xfeat_.at(i).kf0, 1);
            EXPECT_EQ(salsa.xfeat_.at(i).idx0, 1);
        }

    }
}

TEST (Vision, HandleWindowSlide)
{
    Salsa salsa;
    salsa.init(small_feat_test("/tmp/Salsa/ManualKFTest/"));
    salsa.x_u2c_.q() = quat::Quatd::Identity();
    salsa.x_u2c_.t() = Vector3d::Zero();

    Matrix<double, 4, 3> l;
    l << 1,  0, 1,
         0,  1, 1,
         0, -1, 1,
        -1,  0, 1;


    Features feat;
    feat.resize(4);
    for (int j = 0; j < 4; j++)
    {
        feat.zetas[j] = l.row(j).transpose().normalized();
        Vector2d pix = salsa.cam_.proj(feat.zetas[j]);
        feat.pix[j].x = pix.x();
        feat.pix[j].y = pix.y();
        feat.depths[j] = l.row(j).transpose().norm();
        feat.feat_ids[j] = j;
    }
    feat.id = 0;
    feat.t = 0.0;

    Matrix2d R_pix = Matrix2d::Identity();
    Matrix1d R_depth = Matrix1d::Identity();
    std::vector<Vector6d, aligned_allocator<Vector6d>> imu;
    imu.resize(3);
    imu[0] = (Vector6d() << 0.05, 0.0, -9.80665, 0.1, 0, 0).finished();
    imu[1] = (Vector6d() << -0.05, 0.05, -9.80665, -0.1, 0.1, 0).finished();
    imu[2] = (Vector6d() << 0.0, -0.05, -9.80665, 0.0, -0.1, 0).finished();
    Matrix6d R_imu = Matrix6d::Identity();

    double t = 0.0;
    double dt = 0.01;

    auto imuit = imu.begin();

    int kf_cb_id, kf_cb_cond;
    salsa.new_kf_cb_ = [&kf_cb_id, &kf_cb_cond] (int kf_id, int kf_cond)
    {
        kf_cb_id = kf_id;
        kf_cb_cond = kf_cond;
    };

    for (int k = 0; k < salsa.N_*2; k++)
    {
        for (int i = 0; i < 10; i++)
        {
            salsa.imageCallback(t, feat, R_pix, salsa.calcNewKeyframeCondition(feat));
            EXPECT_LT(salsa.summary_.initial_cost, 1e-8);
            EXPECT_EQ(salsa.xbuf_tail_, k <= salsa.N_ ? 0 : k - salsa.N_);
            if (i == 0)
            {
                EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].node, k);
                EXPECT_EQ(salsa.current_node_, k);
                EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, k);
                EXPECT_EQ(salsa.current_kf_, k);
                EXPECT_EQ(salsa.xbuf_head_, k);
            }
            else
            {
                EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].node, k+1);
                EXPECT_EQ(salsa.current_node_, k+1);
                EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, -1);
                EXPECT_EQ(salsa.current_kf_, k);
                EXPECT_EQ(salsa.xbuf_head_, k+1);
            }

            for (int ii = 0; ii < 3; ii++)
            {
                t += dt;
                salsa.imuCallback(t, *imuit, R_imu);
            }

            feat.id += 1;
            feat.t = t;
            for (int j = 0; j < 4; j++)
            {
                feat.zetas[j] = salsa.current_state_.x.transformp(l.row(j).transpose()).normalized();
                feat.depths[j] = (l.row(j).transpose() - salsa.current_state_.x.t()).norm();
            }
        }

        EXPECT_EQ(kf_cb_id, k);
        if (k == 0)
            EXPECT_EQ(kf_cb_cond, Salsa::FIRST_KEYFRAME);
        else
            EXPECT_EQ(kf_cb_cond, Salsa::INSUFFICIENT_MATCHES);

        if (k > salsa.N_)
        {
            for (auto feat : salsa.xfeat_)
            {
                Feat& ft(feat.second);
                if (feat.first < 2*(k - salsa.N_))
                {
                    EXPECT_EQ(ft.kf0, k - salsa.N_);
                }
            }
        }


        imuit++;
        if (imuit == imu.end())
            imuit = imu.begin();

        feat.feat_ids[(2*k) % feat.feat_ids.size()] += feat.feat_ids.size();
        feat.feat_ids[(2*k+1) % feat.feat_ids.size()] += feat.feat_ids.size();
    }
}

