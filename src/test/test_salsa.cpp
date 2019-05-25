#include <gtest/gtest.h>

#include "gnss_utils/satellite.h"

#include "salsa/salsa.h"
#include "salsa/test_common.h"

using namespace gnss_utils;
using namespace salsa;
using namespace multirotor_sim;
using namespace Eigen;
using namespace xform;

class SalsaFeatGNSSTest : public ::testing::Test
{
public:
    void initGNSS()
    {
        std::vector<int> sat_ids = {3, 8, 10, 11, 14, 18, 22, 31, 32};
        log_start = GTime(2026, 165029);
        log_start += 200;

        for (int i = 1; i < sat_ids.size(); i++)
        {
            sats.emplace_back(sat_ids[i], i);
            sats.back().readFromRawFile(SALSA_DIR"/sample/eph.dat");
        }
    }

    void initFeat()
    {
        l << 1,  0, 1,
                0,  1, 1,
                0, -1, 1,
                -1,  0, 1;


        feat.resize(4);
        keyframe_ = 0;
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

        R_pix = Matrix2d::Identity();
        R_depth = Matrix1d::Identity();
    }

    void initSalsa()
    {
        salsa.init(small_feat_test("/tmp/Salsa/ManualKFTest/"));
        salsa.x_b2c_.q() = quat::Quatd::Identity();
        salsa.x_b2c_.t() = Vector3d::Zero();
        Vector3d provo_lla{40.246184 * M_PI/180.0 , -111.647769 * M_PI/180.0, 1387.997511};
        Vector3d provo_ecef = WGS84::lla2ecef(provo_lla);
        salsa.x_e2n_ = WGS84::x_ecef2ned(provo_ecef);
        salsa.new_kf_cb_ = [this](int kf_id, int kf_cond) {this->new_kf_cb(kf_id, kf_cond);};
    }

    void initIMU()
    {
        imu.resize(4);
        imu[0] = (Vector6d() << 0.00, 0.0, -9.80665, 0.0, 0, 0).finished();
        imu[0] = (Vector6d() << 0.05, 0.0, -9.80665, 0.1, 0, 0).finished();
        imu[1] = (Vector6d() << -0.05, 0.05, -9.80665, -0.1, 0.1, 0).finished();
        imu[2] = (Vector6d() << 0.0, -0.05, -9.80665, 0.0, -0.1, 0).finished();
        R_imu = Matrix6d::Identity();
        imuit = imu.begin();
    }

    void new_kf_cb(int kf_id, int kf_cond)
    {
        kf_cb_id = kf_id;
        kf_cb_cond = kf_cond;
        EXPECT_TRUE(expected_kf);
        expected_kf = false;
    }

    void SetUp() override
    {
        initSalsa();
        initIMU();
        initFeat();
        initGNSS();
        t = 0.0;
        dt = 0.002;
    }

    void simulateFeat()
    {
        feat.id += 1;
        if (feat.id == 1)
            expected_kf = true;

        feat.t = t;
        for (int j = 0; j < 4; j++)
        {
            feat.zetas[j] = salsa.current_state_.x.transformp(l.row(j).transpose()).normalized();
            feat.depths[j] = (l.row(j).transpose() - salsa.current_state_.x.t()).norm();
        }
        salsa.addMeas(meas::Img(t, feat, R_pix, salsa.calcNewKeyframeCondition(feat)));
        salsa.handleMeas();
    }

    void createNewKeyframe()
    {
        feat.feat_ids[keyframe_ % feat.feat_ids.size()] += feat.feat_ids.size();
        keyframe_++;
        expected_kf = true;
        simulateFeat();
        EXPECT_FALSE(expected_kf);
    }

    void incrementImuIt()
    {
        imuit++;
        if (imuit == imu.end())
            imuit = imu.begin();
    }

    void simulateIMU()
    {
        t += dt/2.0;
        salsa.imuCallback(t, *imuit, R_imu);
        t += dt/2.0;
        salsa.imuCallback(t, *imuit, R_imu);
    }

    void simulateGNSS()
    {
        VecVec3 zvec;
        VecMat3 Rvec;
        Vector3d rec_pos_ecef = WGS84::ned2ecef(salsa.x_e2n_, salsa.xbuf_[salsa.xbuf_head_].x.t());
        Vector3d rec_vel_ecef = salsa.x_e2n_.rota(salsa.xbuf_[salsa.xbuf_head_].v);
        Vector2d rec_clk = salsa.xbuf_[salsa.xbuf_head_].tau;
        if (salsa.current_node_ < 0)
        {
            rec_pos_ecef = salsa.x_e2n_.t();
            rec_vel_ecef = Vector3d::Zero();
            rec_clk = Vector2d::Zero();
        }
        GTime gtime = log_start + t;
        std::vector<bool> slip(sats.size(), false);
        for (auto& sat: sats)
        {
            Vector3d z;
            sat.computeMeasurement(gtime, rec_pos_ecef, rec_vel_ecef, rec_clk, z);
            zvec.push_back(z);
            Rvec.push_back(Matrix3d::Identity());
        }
        salsa.rawGnssCallback(gtime, zvec, Rvec, sats, slip);
        salsa.handleMeas();
    }

    void runGNSSFirst()
    {
        simulateIMU(); // initialize everything
        simulateGNSS(); // initialize with GNSS pntpos

        EXPECT_EQ(salsa.xbuf_head_, 0);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, -1);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].node, 0);
        EXPECT_MAT_FINITE(salsa.xbuf_[salsa.xbuf_head_].x.arr());
        EXPECT_MAT_FINITE(salsa.xbuf_[salsa.xbuf_head_].v);
        EXPECT_MAT_FINITE(salsa.current_state_.x.arr());
        EXPECT_MAT_FINITE(salsa.current_state_.v);
    }

    void runFeatFirst()
    {
        simulateIMU(); // initialize everything
        expected_kf = true;
        simulateFeat(); // initialize with Features
        EXPECT_EQ(salsa.xbuf_head_, 0);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, 0);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].node, 0);
        EXPECT_MAT_FINITE(salsa.xbuf_[salsa.xbuf_head_].x.arr());
        EXPECT_MAT_FINITE(salsa.xbuf_[salsa.xbuf_head_].v);
        EXPECT_MAT_FINITE(salsa.current_state_.x.arr());
        EXPECT_MAT_FINITE(salsa.current_state_.v);
    }

    void runGNSSThenKF()
    {
        // K  -1     0                    1
        // N   0     1    2    2    3     3
        //     G -- KF -- F -- G -- F -- KF
        simulateIMU(); simulateGNSS();
        EXPECT_EQ(salsa.xbuf_head_, 0);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, -1);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].node, 0);

        simulateIMU(); simulateFeat();
        EXPECT_EQ(salsa.xbuf_head_, 1);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, 0);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].node, 1);

        simulateIMU(); simulateFeat();
        EXPECT_EQ(salsa.xbuf_head_, 2);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, -1);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].node, 2);

        simulateIMU(); simulateGNSS();
        EXPECT_EQ(salsa.xbuf_head_, 2);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, -1);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].node, 2);

        simulateIMU(); simulateFeat();
        EXPECT_EQ(salsa.xbuf_head_, 3);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, -1);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].node, 3);

        simulateIMU(); createNewKeyframe();
        EXPECT_EQ(salsa.xbuf_head_, 3);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, 1);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].node, 3);

        // make sure everything in the history is right
        EXPECT_EQ(salsa.xbuf_[0].kf, -1);
        EXPECT_EQ(salsa.xbuf_[0].node, 0);
        EXPECT_EQ(salsa.xbuf_[1].kf, 0);
        EXPECT_EQ(salsa.xbuf_[1].node, 1);
        EXPECT_EQ(salsa.xbuf_[2].kf, -1);
        EXPECT_EQ(salsa.xbuf_[2].node, 2);
        EXPECT_EQ(salsa.xbuf_[3].kf, 1);
        EXPECT_EQ(salsa.xbuf_[3].node, 3);
    }

    void runKFThenGNSS()
    {
        // K    0                         1                         2               3
        // N    0    1    2               3    4              5     6    7    8     9
        //     KF -- G -- G -- F -- F -- KF -- G -- F -- F -- G -- KF -- G -- F -- KF
        //
        // 0    o              o    o     x
        // 1    o -----------  o -- o --  o         o    o          x
        // 2    o -----------  o -- o --  o ------  o -- o -------- o         o    x
        // 3    o -----------  o -- o --  o ------  o -- o -------- o ------- o -- o
        // 4                              o ------  o -- o -------- o ------- o -- o
        // 5                                                        o ------- o -- o
        // 6                                                                       o

        simulateIMU(); simulateFeat();
        simulateIMU(); simulateGNSS();
        simulateIMU(); simulateGNSS();
        simulateIMU(); simulateFeat();
        simulateIMU(); simulateFeat();
        simulateIMU(); createNewKeyframe();
        EXPECT_EQ(salsa.xfeat_.size(), 4);
        EXPECT_EQ(salsa.xfeat_.at(1).kf0, 0);
        EXPECT_EQ(salsa.xfeat_.at(2).kf0, 0);
        EXPECT_EQ(salsa.xfeat_.at(3).kf0, 0);
        EXPECT_EQ(salsa.xfeat_.at(4).kf0, 1);
        EXPECT_EQ(salsa.xfeat_.at(1).funcs.size(), 1);
        EXPECT_EQ(salsa.xfeat_.at(2).funcs.size(), 1);
        EXPECT_EQ(salsa.xfeat_.at(3).funcs.size(), 1);
        EXPECT_EQ(salsa.xfeat_.at(4).funcs.size(), 0);


        simulateIMU(); simulateGNSS();
        simulateIMU(); simulateFeat();
        simulateIMU(); simulateFeat();
        simulateIMU(); simulateGNSS();
        simulateIMU(); createNewKeyframe();
        EXPECT_EQ(salsa.xfeat_.size(), 5);
        EXPECT_EQ(salsa.xfeat_.at(1).kf0, 0);
        EXPECT_EQ(salsa.xfeat_.at(2).kf0, 0);
        EXPECT_EQ(salsa.xfeat_.at(3).kf0, 0);
        EXPECT_EQ(salsa.xfeat_.at(4).kf0, 1);
        EXPECT_EQ(salsa.xfeat_.at(5).kf0, 2);
        EXPECT_EQ(salsa.xfeat_.at(1).funcs.size(), 1);
        EXPECT_EQ(salsa.xfeat_.at(2).funcs.size(), 2);
        EXPECT_EQ(salsa.xfeat_.at(3).funcs.size(), 2);
        EXPECT_EQ(salsa.xfeat_.at(4).funcs.size(), 1);
        EXPECT_EQ(salsa.xfeat_.at(5).funcs.size(), 0);

        simulateIMU(); simulateFeat();
        simulateIMU(); simulateGNSS();
        simulateIMU(); createNewKeyframe();
        EXPECT_EQ(salsa.xfeat_.size(), 6);
        EXPECT_EQ(salsa.xfeat_.at(1).kf0, 0);
        EXPECT_EQ(salsa.xfeat_.at(2).kf0, 0);
        EXPECT_EQ(salsa.xfeat_.at(3).kf0, 0);
        EXPECT_EQ(salsa.xfeat_.at(4).kf0, 1);
        EXPECT_EQ(salsa.xfeat_.at(5).kf0, 2);
        EXPECT_EQ(salsa.xfeat_.at(6).kf0, 3);
        EXPECT_EQ(salsa.xfeat_.at(1).funcs.size(), 1);
        EXPECT_EQ(salsa.xfeat_.at(2).funcs.size(), 2);
        EXPECT_EQ(salsa.xfeat_.at(3).funcs.size(), 3);
        EXPECT_EQ(salsa.xfeat_.at(4).funcs.size(), 2);
        EXPECT_EQ(salsa.xfeat_.at(5).funcs.size(), 1);
        EXPECT_EQ(salsa.xfeat_.at(6).funcs.size(), 0);

        for (int i = 0; i < 8; i++)
        {
            EXPECT_EQ(salsa.xbuf_[i].node, i);
            if (i == 0)
                EXPECT_EQ(salsa.xbuf_[i].kf, 0);
            else if (i == 3)
                EXPECT_EQ(salsa.xbuf_[i].kf, 1);
            else if (i == 6)
                EXPECT_EQ(salsa.xbuf_[i].kf, 2);
            else
                EXPECT_EQ(salsa.xbuf_[i].kf, -1);
        }

    }

    void runMixedCleanup()
    {
        // K    |  0                   |  1                   | 2                    |  3
        // N    |  0    1    1    2    |  2    3    3    4    | 4     5    5    6    |  6
        // head |  0    1    1    2    |  2    3    3    4    | 4     5    5    6    |  6
        //      | KF -- F -- G -- F -- | KF -- F -- G -- F -- | KF -- F -- G -- F -- | KF ...
        salsa.node_window_ = 16;
        for (int i = 0; i < salsa.node_window_; i += 2)
        {
            simulateIMU(); createNewKeyframe();
            EXPECT_EQ(salsa.xbuf_head_, i);
            EXPECT_EQ(salsa.current_node_, i);
            EXPECT_EQ(salsa.current_kf_, i/2);
            EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, i/2);
            simulateIMU(); simulateFeat();
            EXPECT_EQ(salsa.xbuf_head_, i+1);
            EXPECT_EQ(salsa.current_node_, i+1);
            EXPECT_EQ(salsa.current_kf_, i/2);
            EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, -1);
            simulateIMU(); simulateGNSS();
            EXPECT_EQ(salsa.xbuf_head_, i+1);
            EXPECT_EQ(salsa.current_node_, i+1);
            EXPECT_EQ(salsa.current_kf_, i/2);
            EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, -1);
            simulateIMU(); simulateFeat();
            EXPECT_EQ(salsa.xbuf_head_, i+2);
            EXPECT_EQ(salsa.current_node_, i+2);
            EXPECT_EQ(salsa.current_kf_, i/2);
            EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, -1);
        }
        for (int i = 0; i < salsa.node_window_; i+=2)
        {
            EXPECT_EQ(salsa.xbuf_[i].node, i);
            EXPECT_EQ(salsa.xbuf_[i+1].node, i+1);
            EXPECT_EQ(salsa.xbuf_[i].kf, i/2);
            EXPECT_EQ(salsa.xbuf_[i+1].kf, -1);
        }

        simulateIMU(); createNewKeyframe();
        EXPECT_EQ(salsa.current_node_, salsa.node_window_);
        EXPECT_EQ(salsa.xfeat_.size(), salsa.node_window_/2 + 3);
        EXPECT_EQ(salsa.prange_.size(), salsa.node_window_/2);
        for (const std::pair<const int,Feat>& ft: salsa.xfeat_)
        {
            EXPECT_GT(ft.first, 1);
            if (ft.first == 12 )
                EXPECT_EQ(ft.second.funcs.size(), 0);
            else if (ft.first == 2 || ft.first == 11)
                EXPECT_EQ(ft.second.funcs.size(), 1);
            else if (ft.first == 3 || ft.first == 10)
                EXPECT_EQ(ft.second.funcs.size(), 2);
            else
                EXPECT_EQ(ft.second.funcs.size(), 3);
        }
        simulateIMU(); simulateFeat();
        simulateIMU(); simulateGNSS();
        simulateIMU(); simulateFeat();

        simulateIMU(); createNewKeyframe();
        EXPECT_EQ(salsa.current_node_, salsa.node_window_+2);
        EXPECT_EQ(salsa.xfeat_.size(), salsa.node_window_/2 + 3);
        EXPECT_EQ(salsa.prange_.size(), salsa.node_window_/2);
        EXPECT_EQ(salsa.xbuf_tail_, 2);
        for (const std::pair<const int,Feat>& ft: salsa.xfeat_)
        {
            EXPECT_GT(ft.first, 2);
            if (ft.first == 13 )
                EXPECT_EQ(ft.second.funcs.size(), 0);
            else if (ft.first == 3 || ft.first == 12)
                EXPECT_EQ(ft.second.funcs.size(), 1);
            else if (ft.first == 4 || ft.first == 11)
                EXPECT_EQ(ft.second.funcs.size(), 2);
            else
                EXPECT_EQ(ft.second.funcs.size(), 3);
            EXPECT_GT(ft.second.kf0, 0);
        }
    }

    void runSubsequentKF()
    {
        // i    0                         1                         2                       3
        // K    |  0     1                |  2     3                | 4      5              |  6
        // N    |  0     1     2     2    |  3     4     5     5    | 6      7    8    8    |  9
        // head |  0     1     2     2    |  3     4     5     5    | 6      7    8    8    |  9
        //      | KF -- KF --  F --  G -- | KF -- KF --  F --  G -- | KF -- KF -- F -- G -- | KF ...
        for (int i = 0; i < salsa.node_window_; i++)
        {
            simulateIMU(); createNewKeyframe();
            EXPECT_EQ(salsa.xbuf_head_, i*3);
            EXPECT_EQ(salsa.current_node_, i*3);
            EXPECT_EQ(salsa.current_kf_, 2*i);
            EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, 2*i);
            simulateIMU(); createNewKeyframe();
            EXPECT_EQ(salsa.xbuf_head_, 3*i+1);
            EXPECT_EQ(salsa.current_node_, 3*i+1);
            EXPECT_EQ(salsa.current_kf_, 2*i+1);
            EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, 2*i+1);
            simulateIMU(); simulateFeat();
            EXPECT_EQ(salsa.xbuf_head_, 3*i+2);
            EXPECT_EQ(salsa.current_node_, 3*i+2);
            EXPECT_EQ(salsa.current_kf_, 2*i+1);
            EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, -1);
            simulateIMU(); simulateGNSS();
            EXPECT_EQ(salsa.xbuf_head_, 3*i+2);
            EXPECT_EQ(salsa.current_node_, 3*i+2);
            EXPECT_EQ(salsa.current_kf_, 2*i+1);
            EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, -1);
        }
    }

    void runKFSPlitGNSS()
    {
        // i    0            1          2
        // K    |  0         |  1       |  2
        // N    |  0    1    |  2     3 |  4
        // head |  0    1    |  2     3 |  4
        //      | KF -- G -- | KF --  G | KF ...
        for (int i = 0; i < salsa.node_window_; i++)
        {
            simulateIMU(); createNewKeyframe();
            EXPECT_EQ(salsa.xbuf_head_, i*2);
            EXPECT_EQ(salsa.current_node_, i*2);
            EXPECT_EQ(salsa.current_kf_, i);
            EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, i);
            simulateIMU(); simulateGNSS();
            EXPECT_EQ(salsa.xbuf_head_, i*2+1);
            EXPECT_EQ(salsa.current_node_, i*2+1);
            EXPECT_EQ(salsa.current_kf_, i);
            EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, -1);
        }
    }


    std::vector<Vector6d, aligned_allocator<Vector6d>> imu;
    std::vector<Vector6d, aligned_allocator<Vector6d>>::iterator imuit;
    Matrix6d R_imu;

    GTime log_start;
    Salsa salsa;
    Matrix<double, 4, 3> l;
    Features feat;
    Matrix2d R_pix;
    Matrix1d R_depth;
    SatVec sats;
    int kf_cb_id, kf_cb_cond;
    bool expected_kf = false;
    double t = 0.0;
    double dt = 0.002;

    int keyframe_;
};

TEST_F (SalsaFeatGNSSTest, GNSSFirst)
{
    runGNSSFirst();
}

TEST_F (SalsaFeatGNSSTest, KfFirst)
{
    runFeatFirst();
}

TEST_F (SalsaFeatGNSSTest, GNSSThenKF)
{
    runGNSSThenKF();
}

TEST_F (SalsaFeatGNSSTest, KFThenGNSS)
{
    runKFThenGNSS();
}

TEST_F (SalsaFeatGNSSTest, MixedCleanup)
{
    runMixedCleanup();
}

TEST_F (SalsaFeatGNSSTest, SubsequentKF)
{
    runSubsequentKF();
}

TEST_F (SalsaFeatGNSSTest, KFSPlitGNSS)
{
    runKFSPlitGNSS();
}
