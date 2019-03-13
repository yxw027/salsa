#include "salsa/salsa_ros.h"

SalsaROS::SalsaROS() :
  nh_(), nh_private_("~")
{
  start_time_ = ros::Time(0.0);
  pose_sub_ = nh_.subscribe("pose", 10, &SalsaROS::poseCB, this);
  tform_sub_ = nh_.subscribe("tform", 10, &SalsaROS::tformCB, this);
  odom_sub_ = nh_.subscribe("odom", 10, &SalsaROS::odomCB, this);
  imu_sub_ = nh_.subscribe("imu", 10, &SalsaROS::imuCB, this);

  odom_pub_ = nh_.advertise<nav_msgs::Odometry>("state", 20);

  double acc_stdev, gyro_stdev;
  nh_private_.param<double>("acc_stdev", acc_stdev, 1.0);
  nh_private_.param<double>("gyro_stdev", gyro_stdev, 0.2);
}


void SalsaROS::imuCB(const sensor_msgs::ImuConstPtr& msg)
{
  if (start_time_.sec == 0)
    start_time_ = msg->header.stamp;

  double t = (msg->header.stamp - start_time_).toSec();
  Vector6d z;
  z << msg->linear_acceleration.x,
       msg->linear_acceleration.y,
       msg->linear_acceleration.z,
       msg->angular_velocity.x,
       msg->angular_velocity.y,
       msg->angular_velocity.z;
  salsa_.imuCallback(t, z, imu_R_);

  if (odom_pub_.getNumSubscribers() > 0)
  {
    nav_msgs::Odometry odom;
    odom.header.stamp = msg->header.stamp;
    odom.pose.pose.position.x = salsa_.current_state_.x.t().x();
    odom.pose.pose.position.y = salsa_.current_state_.x.t().y();
    odom.pose.pose.position.z = salsa_.current_state_.x.t().z();
    odom.pose.pose.orientation.w = salsa_.current_state_.x.q().w();
    odom.pose.pose.orientation.x = salsa_.current_state_.x.q().x();
    odom.pose.pose.orientation.y = salsa_.current_state_.x.q().y();
    odom.pose.pose.orientation.z = salsa_.current_state_.x.q().z();
    odom.twist.twist.linear.x = salsa_.current_state_.v.x();
    odom.twist.twist.linear.y = salsa_.current_state_.v.y();
    odom.twist.twist.linear.z = salsa_.current_state_.v.z();
    odom.twist.twist.angular.x = msg->angular_velocity.x;
    odom.twist.twist.angular.y = msg->angular_velocity.y;
    odom.twist.twist.angular.z = msg->angular_velocity.z;
    odom_pub_.publish(odom);
  }
}


void SalsaROS::odomCB(const nav_msgs::OdometryConstPtr& msg)
{
  double t = (msg->header.stamp - start_time_).toSec();
  Xformd z;
  z.arr() << msg->pose.pose.position.x,
      msg->pose.pose.position.y,
      msg->pose.pose.position.z,
      msg->pose.pose.orientation.w,
      msg->pose.pose.orientation.x,
      msg->pose.pose.orientation.y,
      msg->pose.pose.orientation.z;
  salsa_.mocapCallback(t, z, mocap_R_);
}


void SalsaROS::poseCB(const geometry_msgs::PoseStampedPtr& msg)
{
  double t = (msg->header.stamp - start_time_).toSec();
  Xformd z;
  z.arr() << msg->pose.position.x,
      msg->pose.position.y,
      msg->pose.position.z,
      msg->pose.orientation.w,
      msg->pose.orientation.x,
      msg->pose.orientation.y,
      msg->pose.orientation.z;
  salsa_.mocapCallback(t, z, mocap_R_);
}


void SalsaROS::tformCB(const geometry_msgs::TransformStampedPtr& msg)
{
  double t = (msg->header.stamp - start_time_).toSec();
  Xformd z;
  z.arr() << msg->transform.translation.x,
      msg->transform.translation.y,
      msg->transform.translation.z,
      msg->transform.rotation.w,
      msg->transform.rotation.x,
      msg->transform.rotation.y,
      msg->transform.rotation.z;
  salsa_.mocapCallback(t, z, mocap_R_);
}




