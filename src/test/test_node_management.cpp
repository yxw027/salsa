#include <gtest/gtest.h>

#include "salsa/salsa.h"
#include "salsa/test_common.h"
#include "multirotor_sim/simulator.h"
#include "gnss_utils/gtime.h"

using namespace salsa;
using namespace multirotor_sim;
using namespace Eigen;
using namespace xform;
using namespace gnss_utils;

class Management : public ::testing::Test, public Salsa
{
public:
    Vector6d z_imu;
    Matrix6d R_imu;
    ObsVec z_gnss;
    Xformd z_mocap;
    Matrix6d R_mocap;
    Features z_img;
    Matrix2d R_pix;
    Matrix<double, 4, 3> l;

    GTime log_start;
    SatVec sats;

    double _dt;
    double _last_t;

    void SetUp() override
    {
        init(default_params("/tmp/Salsa/ManualKFTest/"));
        disable_solver_ = true;

        _dt = 0.01;
        _last_t = -_dt;

        initIMU();
        initFeat();
        initGNSS();
        initMocap();

        initialize(0, Xformd::Identity(), Vector3d::Zero(), Vector2d::Zero());
    }

    void initIMU()
    {
        z_imu << 0.00, 0.0, -9.80665, 0.0, 0, 0;
        R_imu = Matrix6d::Identity() * 1e-3;
    }

    void initFeat()
    {
        l << 1,  0, 1,
                0,  1, 1,
                0, -1, 1,
                -1,  0, 1;


        z_img.resize(4);
        for (int j = 0; j < 4; j++)
        {
            z_img.zetas[j] = l.row(j).transpose().normalized();
            Vector2d pix = cam_.proj(z_img.zetas[j]);
            z_img.pix[j].x = pix.x();
            z_img.pix[j].y = pix.y();
            z_img.depths[j] = l.row(j).transpose().norm();
            z_img.feat_ids[j] = j;
        }
        z_img.id = 0;
        z_img.t = 0.0;

        R_pix = Matrix2d::Identity();
    }

    void initMocap()
    {
        z_mocap = Xformd::Identity();
        R_mocap = Matrix6d::Identity()*1e-3;
    }


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

    void createIMUString(double t)
    {
        while (_last_t < t)
        {
            double t_new = _last_t + _dt;
            imu_meas_buf_.push_back(meas::Imu(t_new, z_imu, R_imu));
            _last_t = t_new;
        }
    }
};

TEST_F (Management, RoundOffHelpers)
{
    double a = 1.0;
    double b = -1.0;

    double ap = a + eps - std::numeric_limits<double>::epsilon();
    double am = a - eps + std::numeric_limits<double>::epsilon();
    double bp = b + eps - std::numeric_limits<double>::epsilon();
    double bm = b - eps + std::numeric_limits<double>::epsilon();

    EXPECT_TRUE(eq(ap, am));
    EXPECT_TRUE(eq(bp, bm));
    EXPECT_TRUE(le(ap, am));
    EXPECT_TRUE(le(bp, bm));
    EXPECT_TRUE(ge(ap, am));
    EXPECT_TRUE(ge(bp, bm));
    EXPECT_FALSE(gt(ap, am));
    EXPECT_FALSE(lt(am, ap));
    EXPECT_FALSE(gt(bp, bm));
    EXPECT_FALSE(lt(bm, bp));
    EXPECT_TRUE(gt(am, bm));
    EXPECT_TRUE(lt(bm, am));
}

TEST_F (Management, NewNodeEndOfString)
{
    createIMUString(0.2);
    newNode(0.2);
    EXPECT_TRUE(checkIMUOrder());
    EXPECT_EQ(xbuf_head_, 1);
    EXPECT_EQ(xbuf_tail_, 0);
    EXPECT_FLOAT_EQ(xhead().t, 0.2);
    EXPECT_EQ(xhead().node, 1);
    EXPECT_EQ(imu_.size(), 1);
}

TEST_F (Management, NodeBeforeEndOfString)
{
    createIMUString(0.2);
    newNode(0.1);
    EXPECT_EQ(imu_meas_buf_.size(), 10);
    EXPECT_EQ(xbuf_head_, 1);
    EXPECT_EQ(xbuf_tail_, 0);
    EXPECT_FLOAT_EQ(xhead().t, 0.1);
    EXPECT_EQ(xhead().node, 1);
    EXPECT_EQ(imu_.size(), 1);
}


TEST_F (Management, NodeBarelyAfterOtherNode)
{
    createIMUString(0.2);
    EXPECT_EQ(newNode(0.2), 1);
    createIMUString(0.22);
    EXPECT_EQ(imu_meas_buf_.size(), 2);
    EXPECT_EQ(2, newNode(0.205)); // create a new halfway to the next node
    EXPECT_EQ(xbuf_head_, 2);
    EXPECT_EQ(xbuf_tail_, 0);
    EXPECT_FLOAT_EQ(xbuf_[1].t, 0.2);
    EXPECT_FLOAT_EQ(xhead().t, 0.205);
    EXPECT_EQ(xhead().node, 2);
    EXPECT_EQ(imu_.size(), 2);
    EXPECT_EQ(imu_meas_buf_.size(), 2);
}

TEST_F (Management, MoveNodeToEndOfString)
{
    createIMUString(0.2);
    EXPECT_EQ(newNode(0.1), 1);
    EXPECT_EQ(moveNode(0.2), 1);

    EXPECT_FLOAT_EQ(imu_.back().delta_t_, 0.2);
    EXPECT_EQ(imu_.size(), 1);
    EXPECT_EQ(xhead().node, 1);
    EXPECT_EQ(xbuf_head_, 1);
    EXPECT_EQ(imu_meas_buf_.size(), 0);
}

TEST_F (Management, InsertNode)
{
    createIMUString(0.2);
    EXPECT_EQ(newNode(0.2), 2);
    EXPECT_EQ(insertNode(0.1), 1);

    EXPECT_EQ(imu_.size(), 2);
    EXPECT_EQ(xbuf_head_, 2);
    EXPECT_EQ(current_node_, 2);
    EXPECT_EQ(xhead().node, 2);
    EXPECT_FLOAT_EQ(xhead().t, 0.2);
    EXPECT_EQ(xbuf_[1].node, 1);
    EXPECT_FLOAT_EQ(xbuf_[1].t, 0.1);
}

TEST_F (Management, InsertNodeOnTopOfPrevious)
{
    createIMUString(0.2);
    EXPECT_EQ(newNode(0.1), 1);
    EXPECT_EQ(newNode(0.2), 2);
    EXPECT_EQ(insertNode(0.1), 1);

    EXPECT_EQ(xbuf_head_, 2);
    EXPECT_FLOAT_EQ(xhead().t, 0.2);
    EXPECT_EQ(xhead().node, 2);
    EXPECT_FLOAT_EQ(xbuf_[1].t, 0.1);
    EXPECT_EQ(xbuf_[1].node, 1);
    EXPECT_EQ(imu_.size(), 2);

}

TEST_F (Management, InsertNodeBarelyBehindCurrent)
{
    EXPECT_TRUE(0);
}

TEST_F (Management, InsertNodeOnTopOfCurrent)
{
    EXPECT_TRUE(0);
}

TEST_F (Management, InsertNodeExactlyOneImuIntervalAfterPrevious)
{

}

TEST_F (Management, InsertNodeExactlyOneImuIntervalBeforePrevious)
{

}

TEST_F (Management, InsertNodeHalfOfOneImuIntervalAfterPrevious)
{

}

TEST_F (Management, InsertNodeHalfOfOneImuIntervalBeforePrevious)
{

}

