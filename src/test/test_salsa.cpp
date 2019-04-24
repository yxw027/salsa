#include <gtest/gtest.h>

#include "gnss_utils/satellite.h"

#include "salsa/salsa.h"

using namespace gnss_utils;
using namespace salsa;
using namespace multirotor_sim;

class SalsaFeatGNSSTest : public ::testing::Test
{
public:
    void initGNSS()
    {
        std::vector<int> sat_ids = {3, 8, 10, 11, 14, 18, 22, 31, 32};
        log_start = GTime::fromUTC(1541454646,  0.993);
        log_start += 200;

        for (int i = 1; i < sat_ids.size(); i++)
        {
            sats.emplace_back(sat_ids[i], i);
            sats.back().readFromRawFile(GNSS_UTILS_DIR"/sample/eph.dat");
        }
    }

    void initFeat()
    {
        l << 1,  0, 1,
                0,  1, 1,
                0, -1, 1,
                -1,  0, 1;


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

        R_pix = Matrix2d::Identity();
        R_depth = Matrix1d::Identity();
    }

    void initSalsa()
    {
        salsa.init(small_feat_test("/tmp/Salsa/ManualKFTest/"));
        salsa.x_u2c_.q() = quat::Quatd::Identity();
        salsa.x_u2c_.t() = Vector3d::Zero();
        Vector3d provo_lla{40.246184 * M_PI/180.0 , -111.647769 * M_PI/180.0, 1387.997511};
        Vector3d provo_ecef = WGS84::lla2ecef(provo_lla);
        salsa.x_e2n_ = WGS84::x_ecef2ned(provo_ecef);
        salsa.new_kf_cb_ = [this](int kf_id, int kf_cond) {this->new_kf_cb(kf_id, kf_cond);};
    }

    void initIMU()
    {
        imu.resize(3);
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

    void simFeatMeas()
    {
        feat.id += 1;
        feat.t = t;
        for (int j = 0; j < 4; j++)
        {
            feat.zetas[j] = salsa.current_state_.x.transformp(l.row(j).transpose()).normalized();
            feat.depths[j] = (l.row(j).transpose() - salsa.current_state_.x.t()).norm();
        }
        salsa.imageCallback(t, feat, R_pix, salsa.calcNewKeyframeCondition(feat));
    }

    void createNewKeyframe()
    {
        feat.feat_ids[(2*keyframe_) % feat.feat_ids.size()] += feat.feat_ids.size();
        feat.feat_ids[(2*keyframe_+1) % feat.feat_ids.size()] += feat.feat_ids.size();
        simFeatMeas();
    }

    void incrementImuIt()
    {
        imuit++;
        if (imuit == imu.end())
            imuit = imu.begin();
    }

    void simulateIMU()
    {
        t += dt;
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
    }

    void run()
    {
        simulateIMU(); // initialize everything
        simulateGNSS(); // initialize with GNSS pntpos
        simulateIMU(); // space between GNSS and vision

        EXPECT_EQ(salsa.xbuf_head_, 0);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, -1);
        EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].node, 0);

        keyframe_ = 0;
        while (keyframe_ < salsa.STATE_BUF_SIZE*1.2)
        {


            keyframe_++;
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

TEST_F (SalsaFeatGNSSTest, GNSS_First)
{
    run();
}

//    for (int k = 0; k < salsa.node_window_*2; k++)
//    {
//        for (int i = 0; i < 10; i++)
//        {
//            if (i == 0)
//                expected_kf = true;
//            salsa.imageCallback(t, feat, R_pix, salsa.calcNewKeyframeCondition(feat));
//            EXPECT_FALSE(expected_kf);
//            EXPECT_LT(salsa.summary_.initial_cost, 1e-8);
//            if (i == 0)
//            {
//                EXPECT_EQ(salsa.xbuf_tail_, k < salsa.node_window_ ? 0 : k - salsa.node_window_);
//                EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].node, k);
//                EXPECT_EQ(salsa.current_node_, k);
//                EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, k);
//                EXPECT_EQ(salsa.current_kf_, k);
//                EXPECT_EQ(salsa.xbuf_head_, k);
//            }
//            else
//            {
//                EXPECT_EQ(salsa.xbuf_tail_, k < salsa.node_window_ ? 0 : k - salsa.node_window_ + 1);
//                EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].node, k+1);
//                EXPECT_EQ(salsa.current_node_, k+1);
//                EXPECT_EQ(salsa.xbuf_[salsa.xbuf_head_].kf, -1);
//                EXPECT_EQ(salsa.current_kf_, k);
//                EXPECT_EQ(salsa.xbuf_head_, k+1);
//            }

//            for (int ii = 0; ii < 3; ii++)
//            {
//                t += dt;
//                salsa.imuCallback(t, *imuit, R_imu);

//                if (ii == 2)
//                {
//                    VecVec3 z;
//                    VecMat3 R;
//                    GTime gime = log_start + t;
//                    for (auto& sat: sats)
//                    {
//                        sat.computeMeasurement(gtime, );
//                        z.push_back(sat);
//                    }
//                    salsa.rawGnssCallback(log_start + t, );
//                }
//            }


//        }

//        EXPECT_EQ(kf_cb_id, k);
//        if (k == 0)
//            EXPECT_EQ(kf_cb_cond, Salsa::FIRST_KEYFRAME);
//        else
//            EXPECT_EQ(kf_cb_cond, Salsa::INSUFFICIENT_MATCHES);

//        if (k > salsa.node_window_)
//        {
//            for (auto feat : salsa.xfeat_)
//            {
//                Feat& ft(feat.second);
//                if (feat.first < 2*(k - salsa.node_window_))
//                {
//                    EXPECT_EQ(ft.kf0, k - salsa.node_window_);
//                }
//            }
//        }


//        imuit++;
//        if (imuit == imu.end())
//            imuit = imu.begin();

//        feat.feat_ids[(2*k) % feat.feat_ids.size()] += feat.feat_ids.size();
//        feat.feat_ids[(2*k+1) % feat.feat_ids.size()] += feat.feat_ids.size();
//    }
//}
