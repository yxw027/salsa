#pragma once

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>

#include "salsa/salsa.h"
#include "multirotor_sim/utils.h"

class SalsaRosbag
{
public:
    SalsaRosbag(int argc, char **argv);

    void getArgs(int argc, char** argv);
    void loadParams();
    void displayHelp();
    void openBag();
    double getEndTime();
    void parseBag();
    void imuCB(const rosbag::MessageInstance& m);
    void poseCB(const rosbag::MessageInstance& m);

    rosbag::Bag bag_;
    rosbag::View* view_;
    string bag_filename_;
    string param_filename_;
    string log_prefix_;
    double start_;
    double duration_;
    double end_;

    int imu_count_between_nodes_;

    Quatd q_mocap_to_NED_pos_, q_mocap_to_NED_att_;
    ros::Time bag_start_;
    ros::Time bag_duration_;
    ros::Time bag_end_;
    Salsa salsa_;

    Matrix6d imu_R_;
    string seen_imu_topic_;

    Matrix6d mocap_R_;
    Logger truth_log_;
    Logger imu_log_;
};
