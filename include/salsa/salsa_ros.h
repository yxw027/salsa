#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>

#include <Eigen/Core>

#include "salsa/salsa.h"

namespace salsa
{

class SalsaROS
{
public:
    SalsaROS();

    void imuCB(const sensor_msgs::ImuConstPtr& msg);
    void odomCB(const nav_msgs::OdometryConstPtr& msg);
    void poseCB(const geometry_msgs::PoseStampedPtr& msg);
    void tformCB(const geometry_msgs::TransformStampedPtr& msg);

    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    ros::Subscriber imu_sub_;
    ros::Subscriber odom_sub_;
    ros::Subscriber pose_sub_;
    ros::Subscriber tform_sub_;

    ros::Publisher odom_pub_;

    Salsa salsa_;
    ros::Time start_time_;

    Matrix6d imu_R_;
    Matrix6d mocap_R_;
};

}
