#include "salsa/salsa_rosbag.h"

using namespace std;

SalsaRosbag::SalsaRosbag(int argc, char** argv)
{
  start_ = 0;
  duration_ = 1e3;
  seen_imu_topic_ = "";
  getArgs(argc, argv);

  loadParams();
  salsa_.init(param_filename_);
  openBag();
  getEndTime();
  truth_log_.open(salsa_.log_prefix_ + "Truth.log");
  imu_log_.open(salsa_.log_prefix_ + "Imu.log");
  imu_count_between_nodes_ = 0;
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

  get_yaml_eigen("q_mocap_to_NED_pos", param_filename_, q_mocap_to_NED_pos_.arr_);
  get_yaml_eigen("q_mocap_to_NED_att", param_filename_, q_mocap_to_NED_att_.arr_);
}

void SalsaRosbag::displayHelp()
{
  cout << "ROS bag parser" <<endl;
  cout << "-h            display this message" << endl;
  cout << "-f <filename> bagfile to parse REQUIRED" << endl;
  cout << "-y <filename> configuration yaml file REQUIRED" << endl;
  cout << "-s <seconds>  start time" << endl;
  cout << "-d <seconds>  duration" << endl;
  cout << "-i <topic>    IMU Topic" << endl;
  cout << "-p <prefix>   Log Prefix" << endl;
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
  argparse.getCmdOption("-p", log_prefix_);
}

void SalsaRosbag::openBag()
{
  try
  {
    bag_.open(bag_filename_.c_str(), rosbag::bagmode::Read);
    view_ = new rosbag::View(bag_);
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
  end_ = (end_ < view_->getEndTime().toSec() - view_->getBeginTime().toSec()) ? end_ : view_->getEndTime().toSec() - view_->getBeginTime().toSec();
  cout << "Playing bag from: = " << start_ << "s to: " << end_ << "s" << endl;
  return end_;
}

void SalsaRosbag::parseBag()
{
  ProgressBar prog(view_->size(), 80);
  prog.set_theme_braille();
  bag_start_ = view_->getBeginTime() + ros::Duration(start_);
  bag_end_ = view_->getBeginTime() + ros::Duration(end_);
  int i = 0;
  for(rosbag::MessageInstance const m  : (*view_))
  {
    if (m.getTime() < bag_start_)
      continue;

    if (m.getTime() > bag_end_)
      break;

    prog.print(i++);
    string datatype = m.getDataType();

    if (datatype.compare("sensor_msgs/Imu") == 0)
      imuCB(m);
    else if (datatype.compare("geometry_msgs/PoseStamped") == 0)
      poseCB(m);
  }
  prog.finished();
  cout << endl;
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

  if ((z.array() != z.array()).any())
    return;

  imu_log_.log(t);
  imu_log_.logVectors(z);

  imu_count_between_nodes_ ++;
  salsa_.imuCallback(t, z, imu_R_);
  if (!seen_imu_topic_.empty() && seen_imu_topic_.compare(m.getTopic()))
    ROS_WARN_ONCE("Subscribed to Two IMU messages, use the -i argument to specify IMU topic");

  seen_imu_topic_ = m.getTopic();
}

void SalsaRosbag::poseCB(const rosbag::MessageInstance& m)
{
  geometry_msgs::PoseStampedConstPtr pose = m.instantiate<geometry_msgs::PoseStamped>();
  double t = (m.getTime() - bag_start_).toSec();
  Xformd z;
  z.arr() << pose->pose.position.x,
             pose->pose.position.y,
             pose->pose.position.z,
             pose->pose.orientation.w,
             pose->pose.orientation.x,
             pose->pose.orientation.y,
             pose->pose.orientation.z;

  // The mocap is a North, Up, East (NUE) reference frame, so we have to rotate the quaternion's
  // axis of rotation to NED by 90 deg. roll. Then we rotate that resulting quaternion by -90 deg.
  // in yaw because Leo thinks zero attitude is facing East, instead of North.
  z.t_ = q_mocap_to_NED_pos_.rotp(z.t_);
  z.q_.arr_.segment<3>(1) = q_mocap_to_NED_pos_.rotp(z.q_.arr_.segment<3>(1));
  z.q_ = z.q_ * q_mocap_to_NED_att_;


  if (imu_count_between_nodes_ > 2)
  {
    salsa_.mocapCallback(t, z, mocap_R_);
    imu_count_between_nodes_ = 0;
  }
  Vector3d v = Vector3d::Zero();
  Vector6d b = Vector6d::Zero();
  truth_log_.log(t);
  truth_log_.logVectors(z.arr(), v, b);
}

int main(int argc, char** argv)
{
  SalsaRosbag thing(argc, argv);
  thing.parseBag();
}
