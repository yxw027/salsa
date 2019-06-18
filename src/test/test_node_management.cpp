#include <gtest/gtest.h>

#include "salsa/salsa.h"
#include "salsa/test_common.h"
#include "multirotor_sim/simulator.h"
#include "gnss_utils/gtime.h"

using namespace salsa;
using namespace Eigen;
using namespace xform;

class NodeManagement : public ::testing::Test, public Salsa
{
public:
    Vector6d z_imu;
    Matrix6d R_imu;

    double _dt;
    double _last_t;

    void SetUp() override
    {
        init(default_params("/tmp/Salsa/ManualKFTest/"));
        disable_solver_ = true;

        _dt = 0.01;
        _last_t = -_dt;

        initIMU();
    }

    void initIMU()
    {
        z_imu << 0.00, 0.0, -9.80665, 0.0, 0, 0;
        R_imu = Matrix6d::Identity() * 1e-3;
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

class MeasManagement : public NodeManagement
{
public:
    ObsVec z_gnss;
    Xformd z_mocap;
    Matrix6d R_mocap;
    Features z_img;
    Matrix2d R_img;
    Matrix<double, 4, 3> l;

    gnss_utils::GTime log_start;

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

        R_img = Matrix2d::Identity();
    }

    void initMocap()
    {
        z_mocap = Xformd::Identity();
        R_mocap = Matrix6d::Identity()*1e-3;
    }


    void initGNSS()
    {
        std::vector<int> sat_ids = {3, 8, 10, 11, 14, 18, 22, 31, 32};
        log_start = gnss_utils::GTime(2026, 165029);
        log_start += 200;

        for (int i = 1; i < sat_ids.size(); i++)
        {
            sats_.emplace_back(sat_ids[i], i);
            sats_.back().readFromRawFile(SALSA_DIR"/sample/eph.dat");
        }
    }
};

TEST (RoundOff, RoundOffHelpers)
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

TEST_F (NodeManagement, NewNodeEndOfString)
{
    initialize(0, Xformd::Identity(), Vector3d::Zero(), Vector2d::Zero());
    createIMUString(0.2);
    newNode(0.2);
    EXPECT_TRUE(checkIMUString());
    EXPECT_EQ(xbuf_head_, 1);
    EXPECT_EQ(xbuf_tail_, 0);
    EXPECT_FLOAT_EQ(xhead().t, 0.2);
    EXPECT_EQ(xhead().node, 1);
    EXPECT_EQ(imu_.size(), 1);
    EXPECT_EQ(clk_.size(), 1);
}

TEST_F (NodeManagement, NodeBeforeEndOfString)
{
    initialize(0, Xformd::Identity(), Vector3d::Zero(), Vector2d::Zero());
    createIMUString(0.2);
    newNode(0.1);
    EXPECT_EQ(imu_meas_buf_.size(), 10);
    EXPECT_EQ(xbuf_head_, 1);
    EXPECT_EQ(xbuf_tail_, 0);
    EXPECT_FLOAT_EQ(xhead().t, 0.1);
    EXPECT_EQ(xhead().node, 1);
    EXPECT_EQ(imu_.size(), 1);
    EXPECT_EQ(clk_.size(), 1);
}


TEST_F (NodeManagement, NodeBarelyAfterOtherNode)
{
    initialize(0, Xformd::Identity(), Vector3d::Zero(), Vector2d::Zero());
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
    EXPECT_EQ(clk_.size(), 2);
    EXPECT_EQ(imu_meas_buf_.size(), 2);
}

TEST_F (NodeManagement, MoveNodeToEndOfString)
{
    initialize(0, Xformd::Identity(), Vector3d::Zero(), Vector2d::Zero());
    createIMUString(0.2);
    EXPECT_EQ(newNode(0.1), 1);
    EXPECT_EQ(moveNode(0.2), 1);

    EXPECT_FLOAT_EQ(imu_.back().delta_t_, 0.2);
    EXPECT_EQ(imu_.size(), 1);
    EXPECT_EQ(clk_.size(), 1);
    EXPECT_EQ(xhead().node, 1);
    EXPECT_EQ(xbuf_head_, 1);
    EXPECT_EQ(imu_meas_buf_.size(), 0);
}

TEST_F (NodeManagement, InsertNodeIntoBuffer)
{
    initialize(0, Xformd::Identity(), Vector3d::Zero(), Vector2d::Zero());
    for (int i = 0; i < 10; i ++)
    {
        xbuf_[i].t = i*0.1;
        xbuf_[i].node = i;
    }
    xbuf_head_ = 9;
    xbuf_tail_ = 0;

    salsa::State& new_node(insertNodeIntoBuffer(5));
    new_node.t = 0.55;

    for (int i = 0; i < 11; i++)
        EXPECT_EQ(xbuf_[i].node, i);
    for (int i = 0; i < 5; i++)
        EXPECT_FLOAT_EQ(xbuf_[i].t, i*0.1);
    xbuf_[5].t = 0.55;
    for (int i = 6; i < 11; i++)
        EXPECT_FLOAT_EQ(xbuf_[i].t, (i-1)*0.1);

}

TEST_F (NodeManagement, InsertNode)
{
    initialize(0, Xformd::Identity(), Vector3d::Zero(), Vector2d::Zero());
    createIMUString(0.2);
    EXPECT_EQ(newNode(0.2), 1);
    EXPECT_EQ(insertNode(0.1), 1);

    EXPECT_EQ(imu_.size(), 2);
    EXPECT_EQ(clk_.size(), 2);
    EXPECT_EQ(xbuf_head_, 2);
    EXPECT_EQ(current_node_, 2);
    EXPECT_EQ(xhead().node, 2);
    EXPECT_FLOAT_EQ(xhead().t, 0.2);
    EXPECT_EQ(xbuf_[1].node, 1);
    EXPECT_FLOAT_EQ(xbuf_[1].t, 0.1);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());

}

TEST_F (NodeManagement, InsertNodeOnTopOfPrevious)
{
    initialize(0, Xformd::Identity(), Vector3d::Zero(), Vector2d::Zero());
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
    EXPECT_EQ(clk_.size(), 2);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());
}

TEST_F (NodeManagement, InsertNodeOnTopOfCurrent)
{
    initialize(0, Xformd::Identity(), Vector3d::Zero(), Vector2d::Zero());
    createIMUString(0.2);
    EXPECT_EQ(newNode(0.1), 1);
    EXPECT_EQ(newNode(0.2), 2);
    EXPECT_EQ(insertNode(0.2), 2);

    EXPECT_EQ(xbuf_head_, 2);
    EXPECT_FLOAT_EQ(xhead().t, 0.2);
    EXPECT_EQ(xhead().node, 2);
    EXPECT_FLOAT_EQ(xbuf_[1].t, 0.1);
    EXPECT_EQ(xbuf_[1].node, 1);
    EXPECT_EQ(imu_.size(), 2);
    EXPECT_EQ(clk_.size(), 2);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());
}

TEST_F (NodeManagement, InsertNodeBarelyBehindCurrent)
{
    initialize(0, Xformd::Identity(), Vector3d::Zero(), Vector2d::Zero());
    createIMUString(0.2);
    EXPECT_EQ(newNode(0.1), 1);
    EXPECT_EQ(newNode(0.2), 2);
    EXPECT_EQ(insertNode(0.195), 2);

    EXPECT_EQ(xbuf_head_, 3);
    EXPECT_FLOAT_EQ(xhead().t, 0.2);
    EXPECT_EQ(xhead().node, 3);
    EXPECT_FLOAT_EQ(xbuf_[2].t, 0.195);
    EXPECT_EQ(xbuf_[2].node, 2);
    EXPECT_EQ(imu_.size(), 3);
    EXPECT_EQ(clk_.size(), 3);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());
}



TEST_F (NodeManagement, InsertOneIMUAfterPrev)
{
    initialize(0, Xformd::Identity(), Vector3d::Zero(), Vector2d::Zero());
    createIMUString(0.2);
    EXPECT_EQ(newNode(0.1), 1);
    EXPECT_EQ(newNode(0.2), 2);
    EXPECT_EQ(insertNode(0.11), 2);

    EXPECT_EQ(xbuf_head_, 3);
    EXPECT_FLOAT_EQ(xhead().t, 0.2);
    EXPECT_EQ(xhead().node, 3);
    EXPECT_FLOAT_EQ(xbuf_[2].t, 0.11);
    EXPECT_EQ(xbuf_[2].node, 2);
    EXPECT_FLOAT_EQ(xbuf_[1].t, 0.1);
    EXPECT_EQ(xbuf_[1].node, 1);
    EXPECT_EQ(imu_.size(), 3);
    EXPECT_EQ(clk_.size(), 3);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());
}

TEST_F (NodeManagement, InsertOntIMUBeforePrev)
{
    initialize(0, Xformd::Identity(), Vector3d::Zero(), Vector2d::Zero());
    createIMUString(0.2);
    EXPECT_EQ(newNode(0.1), 1);
    EXPECT_EQ(newNode(0.2), 2);
    EXPECT_EQ(insertNode(0.09), 1);

    EXPECT_EQ(xbuf_head_, 3);
    EXPECT_FLOAT_EQ(xhead().t, 0.2);
    EXPECT_EQ(xhead().node, 3);
    EXPECT_FLOAT_EQ(xbuf_[2].t, 0.1);
    EXPECT_EQ(xbuf_[2].node, 2);
    EXPECT_FLOAT_EQ(xbuf_[1].t, 0.09);
    EXPECT_EQ(xbuf_[1].node, 1);
    EXPECT_EQ(imu_.size(), 3);
    EXPECT_EQ(clk_.size(), 3);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());
}

TEST_F (NodeManagement, InsertNHalfIMUAfterPrev)
{
    initialize(0, Xformd::Identity(), Vector3d::Zero(), Vector2d::Zero());
    createIMUString(0.2);
    EXPECT_EQ(newNode(0.1), 1);
    EXPECT_EQ(newNode(0.2), 2);
    EXPECT_EQ(insertNode(0.105), 2);

    EXPECT_EQ(xbuf_head_, 3);
    EXPECT_FLOAT_EQ(xhead().t, 0.2);
    EXPECT_EQ(xhead().node, 3);
    EXPECT_FLOAT_EQ(xbuf_[2].t, 0.105);
    EXPECT_EQ(xbuf_[2].node, 2);
    EXPECT_FLOAT_EQ(xbuf_[1].t, 0.1);
    EXPECT_EQ(xbuf_[1].node, 1);
    EXPECT_EQ(imu_.size(), 3);
    EXPECT_EQ(clk_.size(), 3);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());
}

TEST_F (NodeManagement, InsertHalfImuBeforePrev)
{
    initialize(0, Xformd::Identity(), Vector3d::Zero(), Vector2d::Zero());
    createIMUString(0.2);
    EXPECT_EQ(newNode(0.1), 1);
    EXPECT_EQ(newNode(0.2), 2);
    EXPECT_EQ(insertNode(0.095), 1);

    EXPECT_EQ(xbuf_head_, 3);
    EXPECT_FLOAT_EQ(xhead().t, 0.2);
    EXPECT_EQ(xhead().node, 3);
    EXPECT_FLOAT_EQ(xbuf_[2].t, 0.1);
    EXPECT_EQ(xbuf_[2].node, 2);
    EXPECT_FLOAT_EQ(xbuf_[1].t, 0.095);
    EXPECT_EQ(xbuf_[1].node, 1);
    EXPECT_EQ(imu_.size(), 3);
    EXPECT_EQ(clk_.size(), 3);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());
}

TEST_F (MeasManagement, NominalIMUMocap)
{
    update_on_mocap_ = true;
    createIMUString(0.1);
    addMeas(meas::Mocap(0.1, z_mocap, R_mocap));
    createIMUString(0.2);
    addMeas(meas::Mocap(0.2, z_mocap, R_mocap));
    createIMUString(0.3);
    addMeas(meas::Mocap(0.3, z_mocap, R_mocap));

    EXPECT_EQ(xbuf_head_, 2);
    EXPECT_EQ(imu_.size(), 2);
    EXPECT_EQ(clk_.size(), 2);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());

    EXPECT_EQ(xhead().t, 0.3);
    EXPECT_EQ(xhead().type, State::Mocap);
    EXPECT_EQ(xbuf_[1].t, 0.2);
    EXPECT_EQ(xbuf_[1].type, State::Mocap);
    EXPECT_EQ(xbuf_[0].t, 0.1);
    EXPECT_EQ(xbuf_[0].type, State::Mocap);
}

TEST_F (MeasManagement, NominalIMUGnss)
{
    update_on_gnss_ = true;
    createIMUString(0.1);
    addMeas(meas::Gnss(0.1, z_gnss));
    createIMUString(0.2);
    addMeas(meas::Gnss(0.2, z_gnss));
    createIMUString(0.3);
    addMeas(meas::Gnss(0.3, z_gnss));

    EXPECT_EQ(xbuf_head_, 2);
    EXPECT_EQ(imu_.size(), 2);
    EXPECT_EQ(clk_.size(), 2);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());

    EXPECT_EQ(xhead().t, 0.3);
    EXPECT_EQ(xhead().type, State::Gnss);
    EXPECT_EQ(xbuf_[1].t, 0.2);
    EXPECT_EQ(xbuf_[1].type, State::Gnss);
    EXPECT_EQ(xbuf_[0].t, 0.1);
    EXPECT_EQ(xbuf_[0].type, State::Gnss);
}

TEST_F (MeasManagement, NominalIMUVision)
{
    update_on_gnss_ = true;
    createIMUString(0.1);
    addMeas(meas::Img(0.1, z_img, R_img, true)); // first image is always a keyframe
    createIMUString(0.2);
    addMeas(meas::Img(0.2, z_img, R_img, true)); // in nominal case, every frame is a keyframe!
    createIMUString(0.3);
    addMeas(meas::Img(0.3, z_img, R_img, true));

    EXPECT_EQ(xbuf_head_, 2);
    EXPECT_EQ(imu_.size(), 2);
    EXPECT_EQ(clk_.size(), 2);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());

    EXPECT_EQ(xhead().t, 0.3);
    EXPECT_EQ(xhead().type, State::Camera);
    EXPECT_EQ(xbuf_[1].t, 0.2);
    EXPECT_EQ(xbuf_[1].type, State::Camera);
    EXPECT_EQ(xbuf_[0].t, 0.1);
    EXPECT_EQ(xbuf_[0].type, State::Camera);
}

TEST_F (MeasManagement, IMUVisionMoveNode)
{
    update_on_gnss_ = true;
    createIMUString(0.1);
    addMeas(meas::Img(0.1, z_img, R_img, true)); // first image is always a keyframe
    createIMUString(0.2);
    addMeas(meas::Img(0.2, z_img, R_img, false));
    createIMUString(0.3);
    addMeas(meas::Img(0.3, z_img, R_img, false));

    EXPECT_EQ(xbuf_head_, 1); // moved node means extended IMU
    EXPECT_EQ(imu_.size(), 1);
    EXPECT_FLOAT_EQ(imu_.back().delta_t_, 0.2);
    EXPECT_EQ(clk_.size(), 1);
    EXPECT_FLOAT_EQ(clk_.back().dt_, 0.2);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());

    EXPECT_EQ(xhead().t, 0.3);
    EXPECT_EQ(xhead().type, State::Camera);
    EXPECT_EQ(xbuf_[1].t, 0.3);
    EXPECT_EQ(xbuf_[1].type, State::Camera);
    EXPECT_EQ(xbuf_[0].t, 0.1);
    EXPECT_EQ(xbuf_[0].type, State::Camera);
}

TEST_F (MeasManagement, RewindIMUMocap)
{
    update_on_mocap_ = true;
    createIMUString(0.35);
    addMeas(meas::Mocap(0.1, z_mocap, R_mocap));
    addMeas(meas::Mocap(0.2, z_mocap, R_mocap));
    addMeas(meas::Mocap(0.3, z_mocap, R_mocap));

    EXPECT_EQ(xbuf_head_, 2);
    EXPECT_EQ(imu_.size(), 2);
    EXPECT_EQ(clk_.size(), 2);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());

    EXPECT_EQ(xhead().t, 0.3);
    EXPECT_EQ(xhead().type, State::Mocap);
    EXPECT_EQ(xbuf_[1].t, 0.2);
    EXPECT_EQ(xbuf_[1].type, State::Mocap);
    EXPECT_EQ(xbuf_[0].t, 0.1);
    EXPECT_EQ(xbuf_[0].type, State::Mocap);
}

TEST_F (MeasManagement, RewindIMUGnss)
{
    update_on_gnss_ = true;
    createIMUString(0.35);
    addMeas(meas::Gnss(0.1, z_gnss));
    addMeas(meas::Gnss(0.2, z_gnss));
    addMeas(meas::Gnss(0.3, z_gnss));

    EXPECT_EQ(xbuf_head_, 2);
    EXPECT_EQ(imu_.size(), 2);
    EXPECT_EQ(clk_.size(), 2);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());

    EXPECT_EQ(xhead().t, 0.3);
    EXPECT_EQ(xhead().type, State::Gnss);
    EXPECT_EQ(xbuf_[1].t, 0.2);
    EXPECT_EQ(xbuf_[1].type, State::Gnss);
    EXPECT_EQ(xbuf_[0].t, 0.1);
    EXPECT_EQ(xbuf_[0].type, State::Gnss);
}

TEST_F (MeasManagement, RewindIMUVision)
{
    update_on_gnss_ = true;
    createIMUString(0.35);
    addMeas(meas::Img(0.1, z_img, R_img, true)); // first image is always a keyframe
    addMeas(meas::Img(0.2, z_img, R_img, true)); // in nominal case, every frame is a keyframe!
    addMeas(meas::Img(0.3, z_img, R_img, true));

    EXPECT_EQ(xbuf_head_, 2);
    EXPECT_EQ(imu_.size(), 2);
    EXPECT_EQ(clk_.size(), 2);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());

    EXPECT_EQ(xhead().t, 0.3);
    EXPECT_EQ(xhead().type, State::Camera);
    EXPECT_EQ(xbuf_[1].t, 0.2);
    EXPECT_EQ(xbuf_[1].type, State::Camera);
    EXPECT_EQ(xbuf_[0].t, 0.1);
    EXPECT_EQ(xbuf_[0].type, State::Camera);
}

TEST_F (MeasManagement, MocapOnTopOfSlidImg)
{
    update_on_gnss_ = true;
    createIMUString(0.35);
    addMeas(meas::Img(0.1, z_img, R_img, true));
    addMeas(meas::Gnss(0.1, z_gnss));
    addMeas(meas::Mocap(0.1, z_mocap, R_mocap));
    addMeas(meas::Img(0.2, z_img, R_img, false));
    addMeas(meas::Img(0.3, z_img, R_img, false));
    addMeas(meas::Gnss(0.2, z_gnss));

    EXPECT_EQ(xbuf_head_, 2);
    EXPECT_EQ(imu_.size(), 2);
    EXPECT_EQ(clk_.size(), 2);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());

    EXPECT_EQ(xbuf_[0].t, 0.1);
    EXPECT_EQ(xbuf_[0].type, State::Camera | State::Gnss | State::Mocap);
    EXPECT_EQ(xbuf_[1].t, 0.2);
    EXPECT_EQ(xbuf_[1].type, State::Gnss);
    EXPECT_EQ(xhead().t, 0.3);
    EXPECT_EQ(xhead().type, State::Camera);
}

TEST_F (MeasManagement, MocapOnTopOfSlidingContinued)
{
    update_on_gnss_ = true;
    createIMUString(0.85);
    addMeas(meas::Img(0.1, z_img, R_img, true));
    addMeas(meas::Gnss(0.1, z_gnss));
    addMeas(meas::Mocap(0.1, z_mocap, R_mocap));
    addMeas(meas::Img(0.2, z_img, R_img, false));
    addMeas(meas::Img(0.3, z_img, R_img, false));
    addMeas(meas::Gnss(0.2, z_gnss));
    addMeas(meas::Img(0.4, z_img, R_img, false));
    addMeas(meas::Img(0.5, z_img, R_img, false));
    addMeas(meas::Img(0.6, z_img, R_img, false));
    addMeas(meas::Gnss(0.5, z_gnss));
    addMeas(meas::Img(0.7, z_img, R_img, false));

    EXPECT_EQ(xbuf_head_, 3);
    EXPECT_EQ(imu_.size(), 3);
    EXPECT_EQ(clk_.size(), 3);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());

    EXPECT_EQ(xbuf_[0].t, 0.1);
    EXPECT_EQ(xbuf_[0].type, State::Camera | State::Gnss | State::Mocap);
    EXPECT_EQ(xbuf_[1].t, 0.2);
    EXPECT_EQ(xbuf_[1].type, State::Gnss);
    EXPECT_EQ(xbuf_[2].t, 0.5);
    EXPECT_EQ(xbuf_[2].type, State::Gnss);
    EXPECT_EQ(xbuf_[3].t, 0.7);
    EXPECT_EQ(xbuf_[3].type, State::Camera);
}

TEST_F (MeasManagement, BetweenSliding)
{
    update_on_gnss_ = true;
    createIMUString(0.35);
    addMeas(meas::Img(0.1, z_img, R_img, true));
    addMeas(meas::Mocap(0.1, z_mocap, R_mocap));
    addMeas(meas::Img(0.2, z_img, R_img, false));
    addMeas(meas::Gnss(0.21, z_gnss));
    addMeas(meas::Img(0.3, z_img, R_img, false));

    EXPECT_EQ(xbuf_head_, 2);
    EXPECT_EQ(imu_.size(), 2);
    EXPECT_EQ(clk_.size(), 2);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());

    EXPECT_EQ(xbuf_[0].t, 0.1);
    EXPECT_EQ(xbuf_[0].type, State::Camera | State::Mocap);
    EXPECT_EQ(xbuf_[1].t, 0.21);
    EXPECT_EQ(xbuf_[1].type, State::Gnss);
    EXPECT_EQ(xbuf_[2].t, 0.3);
    EXPECT_EQ(xbuf_[2].type, State::Camera);
}

TEST_F (MeasManagement, RewindBetweenSliding)
{
    update_on_gnss_ = true;
    createIMUString(0.35);
    addMeas(meas::Img(0.1, z_img, R_img, true));
    addMeas(meas::Mocap(0.1, z_mocap, R_mocap));
    addMeas(meas::Img(0.2, z_img, R_img, false));
    addMeas(meas::Img(0.3, z_img, R_img, false));
    addMeas(meas::Gnss(0.21, z_gnss));

    EXPECT_EQ(xbuf_head_, 2);
    EXPECT_EQ(imu_.size(), 2);
    EXPECT_EQ(clk_.size(), 2);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());

    EXPECT_EQ(xbuf_[0].t, 0.1);
    EXPECT_EQ(xbuf_[0].type, State::Camera | State::Mocap);
    EXPECT_EQ(xbuf_[1].t, 0.21);
    EXPECT_EQ(xbuf_[1].type, State::Gnss);
    EXPECT_EQ(xbuf_[2].t, 0.3);
    EXPECT_EQ(xbuf_[2].type, State::Camera);
}

TEST_F (MeasManagement, RewindBeforeKeyframe)
{
    update_on_gnss_ = true;
    createIMUString(0.55);
    addMeas(meas::Img(0.1, z_img, R_img, true));
    addMeas(meas::Mocap(0.1, z_mocap, R_mocap));
    addMeas(meas::Img(0.2, z_img, R_img, false));
    addMeas(meas::Img(0.3, z_img, R_img, true));
    addMeas(meas::Gnss(0.205, z_gnss));
    addMeas(meas::Img(0.4, z_img, R_img, false));
    addMeas(meas::Img(0.5, z_img, R_img, false));

    EXPECT_EQ(xbuf_head_, 3);
    EXPECT_EQ(imu_.size(), 3);
    EXPECT_EQ(clk_.size(), 3);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());


    EXPECT_EQ(xbuf_[0].t, 0.1);
    EXPECT_EQ(xbuf_[0].type, State::Camera | State::Mocap);
    EXPECT_EQ(xbuf_[1].t, 0.205);
    EXPECT_EQ(xbuf_[1].type, State::Gnss);
    EXPECT_EQ(xbuf_[2].t, 0.3);
    EXPECT_EQ(xbuf_[2].type, State::Camera);
    EXPECT_EQ(xbuf_[3].t, 0.5);
    EXPECT_EQ(xbuf_[3].type, State::Camera);
}

TEST_F (MeasManagement, JustAfterKeyframe)
{
    update_on_gnss_ = true;
    createIMUString(0.55);
    addMeas(meas::Img(0.1, z_img, R_img, true));
    addMeas(meas::Mocap(0.1, z_mocap, R_mocap));
    addMeas(meas::Img(0.2, z_img, R_img, false));
    addMeas(meas::Img(0.3, z_img, R_img, true));
    addMeas(meas::Gnss(0.305, z_gnss));
    addMeas(meas::Img(0.4, z_img, R_img, false));
    addMeas(meas::Img(0.5, z_img, R_img, false));

    EXPECT_EQ(xbuf_head_, 3);
    EXPECT_EQ(imu_.size(), 3);
    EXPECT_EQ(clk_.size(), 3);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());


    EXPECT_EQ(xbuf_[0].t, 0.1);
    EXPECT_EQ(xbuf_[0].type, State::Camera | State::Mocap);
    EXPECT_EQ(xbuf_[1].t, 0.3);
    EXPECT_EQ(xbuf_[1].type, State::Camera);
    EXPECT_EQ(xbuf_[2].t, 0.305);
    EXPECT_EQ(xbuf_[2].type, State::Gnss);
    EXPECT_EQ(xbuf_[3].t, 0.5);
    EXPECT_EQ(xbuf_[3].type, State::Camera);
}


TEST_F (MeasManagement, DelayedIMUMocap)
{
    update_on_mocap_ = true;
    addMeas(meas::Mocap(0.1, z_mocap, R_mocap));
    addMeas(meas::Mocap(0.2, z_mocap, R_mocap));
    createIMUString(0.3);
    addMeas(meas::Mocap(0.3, z_mocap, R_mocap));

    EXPECT_EQ(xbuf_head_, 2);
    EXPECT_EQ(imu_.size(), 2);
    EXPECT_EQ(clk_.size(), 2);
    EXPECT_TRUE(checkIMUString());
    EXPECT_TRUE(checkClkString());

    EXPECT_EQ(xhead().t, 0.3);
    EXPECT_EQ(xhead().type, State::Mocap);
    EXPECT_EQ(xbuf_[1].t, 0.2);
    EXPECT_EQ(xbuf_[1].type, State::Mocap);
    EXPECT_EQ(xbuf_[0].t, 0.1);
    EXPECT_EQ(xbuf_[0].type, State::Mocap);
}

