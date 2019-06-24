#pragma once

#include <Eigen/Core>

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <inertial_sense/GNSSObsVec.h>
#include <inertial_sense/GNSSEphemeris.h>
#include <nav_msgs/Odometry.h>

#include "salsa/salsa.h"
#include "multirotor_sim/utils.h"

namespace salsa
{

class SalsaRosbag
{
public:
    SalsaRosbag(int argc, char **argv);

    void getArgs(int argc, char** argv);
    void loadParams();
    void displayHelp();
    void openBag();
    void parseBag();
    void loadEph();

//    void getMocapOffset();

    void imuCB(const rosbag::MessageInstance& m);
    void poseCB(const rosbag::MessageInstance& m);
    void obsCB(const rosbag::MessageInstance& m);
    void ephCB(const rosbag::MessageInstance& m);
    void odomCB(const rosbag::MessageInstance& m);
    void imgCB(const rosbag::MessageInstance& m);
    void compressedImgCB(const rosbag::MessageInstance& m);
    void imgCB(double tc, const cv_bridge::CvImageConstPtr &img);

    rosbag::Bag bag_;
    rosbag::View* view_;
    std::string bag_filename_;
    std::string param_filename_;
    std::string log_prefix_;
    double start_;
    double duration_;
    double end_;
    bool got_imu_;
    int imu_count_;


    quat::Quatd q_mocap_to_NED_pos_, q_mocap_to_NED_att_;
    ros::Time bag_start_;
    ros::Time bag_duration_;
    ros::Time bag_end_;
    ros::Duration mocap_offset_;
    double mocap_rate_;
    ros::Time prev_mocap_, prev_mocap_run_;
    xform::Xformd x_I2m_prev_;
    Salsa salsa_;

    std::string imu_topic_;
    std::string mocap_topic_;
    std::string image_topic_;

    Matrix6d imu_R_;
    Matrix6d mocap_R_;
    Eigen::Matrix2d pix_R_;

    Logger truth_log_;
    Logger imu_log_;

    std::vector<rosbag::MessageInstance> eph_;

    xform::Xformd INS_ref_;
};

}
