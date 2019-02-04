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
    SalsaRosbag(string& bagfile, string& param, double start, double duration);

    void displayHelp();
    void openBag();
    void getEndTime();
    void parseBag();
    void imuCB(rosbag::MessageInstance* m);
    void poseCB(rosbag::MessageInstance* m);

    rosbag::Bag bag;
    rosbag::View view;
    double start;
    double duration;
    double end;

    ros::Time bag_start;
    ros::Time bag_duration;
    ros::Time bag_end;
    Salsa salsa;
};
