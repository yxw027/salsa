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

    void SetUp() override
    {
        init(default_params("/tmp/Salsa/ManualKFTest/"));
        disable_solver_ = true;

        initIMU();
        initFeat();
        initGNSS();
        initMocap();
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
        if (imu_meas_buf_.size() == 0)
        {
            imu_meas_buf_.push_back(meas::Imu(0.0, z_imu, R_imu));
        }
        else
        {
            while (imu_meas_buf_.back().t < t)
            {
                double t_new = imu_meas_buf_.back().t + 0.01;
                imu_meas_buf_.push_back(meas::Imu(t_new, z_imu, R_imu));
            }
        }
    }
};

TEST_F (Management, NewNodeEndOfString)
{
    initialize(0, Xformd::Identity(), Vector3d::Zero(), Vector2d::Zero());
    createIMUString(0.2);
    newNode(0.2);
    ASSERT_TRUE(checkIMUOrder());
}
