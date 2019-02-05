#include "salsa/salsa_rosbag.h"

using namespace std;

SalsaRosbag::SalsaRosbag(int argc, char** argv)
{
  start_ = 0;
  duration_ = INFINITY;
  seen_imu_topic_ = "";
  getArgs(argc, argv);

  loadParams();
  openBag();
  getEndTime();
}

void SalsaRosbag::loadParams()
{
  double acc_stdev, gyro_stdev;
  get_yaml_node("accel_noise_stdev", param_filename_, acc_stdev);
  get_yaml_node("gyro_noise_stdev", param_filename_, gyro_stdev);
  imu_R_.setZero();
  imu_R_.topLeftCorner<3,3>() = acc_stdev * acc_stdev * I_3x3;
  imu_R_.bottomRightCorner<3,3>() = gyro_stdev * gyro_stdev * I_3x3;


  double pos_stdev, att_stdev;
  get_yaml_node("position_noise_stdev", param_filename_, pos_stdev);
  get_yaml_node("attitude_noise_stdev", param_filename_, att_stdev);
  mocap_R_ << pos_stdev * pos_stdev * I_3x3,   Matrix3d::Zero(),
              Matrix3d::Zero(),   att_stdev * att_stdev * I_3x3;
}

void SalsaRosbag::displayHelp()
{
  cout << "ROS bag parser" <<endl;
  cout << "-h            display this message" << endl;
  cout << "-f <filename> bagfile to parse" << endl;
  cout << "-y <filename> configuration yaml file" << endl;
  cout << "-s <seconds>  start time" << endl;
  cout << "-d <seconds>  duration" << endl;
  cout << "-i <topic>    IMU Topic" << endl;
  exit(0);
}

void SalsaRosbag::getArgs(int argc, char** argv)
{
  InputParser argparse(argc, argv);

  if (argparse.cmdOptionExists("-h"))
    displayHelp();
  if (!argparse.getCmdOption("-f", bag_filename_))
    displayHelp();
  if (!argparse.getCmdOption("-y", param_filename_))
    displayHelp();
  argparse.getCmdOption("-s", start_);
  argparse.getCmdOption("-d", duration_);
  argparse.getCmdOption("-i", seen_imu_topic_);
}

void SalsaRosbag::openBag()
{
  try
  {
    bag_.open(bag_filename_.c_str(), rosbag::bagmode::Read);
  }
  catch(rosbag::BagIOException e)
  {
    ROS_ERROR("unable to load rosbag %s, %s", bag_filename_.c_str(), e.what());
    exit(-1);
  }
}

double SalsaRosbag::getEndTime()
{
  end_ = start_ + duration_;
  end_ = (end_ < view_.getEndTime().toSec() - view_.getBeginTime().toSec()) ? end_ : view_.getEndTime().toSec() - view_.getBeginTime().toSec();
  cout << "Playing bag from: = " << start_ << "s to: " << end_ << "s" << endl;
  return end_;
}

void SalsaRosbag::parseBag()
{
  ProgressBar prog(view_.size(), 100);
  bag_start_ = view_.getBeginTime() + ros::Duration(start_);
  bag_end_ = view_.getBeginTime() + ros::Duration(end_);
  rosbag::View::iterator it;
  int i = 0;
  for(rosbag::MessageInstance const m  : view_)
  {
//    rosbag::MessageInstance* m = it;
    if (m.getTime() < bag_start_)
      continue;

    if (m.getTime() > bag_end_)
      break;

    prog.print(i++);
    string datatype = m.getDataType();

    if (datatype.compare("sensor_msgs/Imu") == 0)
      imuCB(m);
  }
}

void SalsaRosbag::imuCB(const rosbag::MessageInstance& m)
{
  sensor_msgs::ImuConstPtr imu = m.instantiate<sensor_msgs::Imu>();
  double t = (imu->header.stamp - bag_start_).toSec();
  Vector6d z;
  z << imu->linear_acceleration.x,
       imu->linear_acceleration.y,
       imu->linear_acceleration.z,
       imu->angular_velocity.x,
       imu->angular_velocity.y,
       imu->angular_velocity.z;
  salsa_.imuCallback(t, z, imu_R_);
  if (!seen_imu_topic_.empty() && seen_imu_topic_.compare(m.getTopic()))
    ROS_WARN_ONCE("Subscribed to Two IMU messages, use the -i argument to specify IMU topic");

  seen_imu_topic_ = m.getTopic();
}

void SalsaRosbag::poseCB(const rosbag::MessageInstance& m)
{
  geometry_msgs::PoseStampedConstPtr pose = m.instantiate<geometry_msgs::PoseStamped>();
  double t = (pose->header.stamp - bag_start_).toSec();
  Xformd z;
  z.arr() << pose->pose.position.x,
             pose->pose.position.y,
             pose->pose.position.z,
             pose->pose.orientation.w,
             pose->pose.orientation.x,
             pose->pose.orientation.y,
             pose->pose.orientation.z;
  salsa_.mocapCallback(t, z, mocap_R_);
}

int main(int argc, char** argv)
{
  SalsaRosbag thing(argc, argv);

  thing.parseBag();
}
