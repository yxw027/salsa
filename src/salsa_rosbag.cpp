#include "salsa/salsa_rosbag.h"

using namespace std;

void SalsaRosbag::displayHelp()
{
  cout << "ROS bag parser" <<endl;
  cout << "-h            display this message" << endl;
  cout << "-f <filename> bagfile to parse" << endl;
  cout << "-y <filename> configuration yaml file" << endl;
  cout << "-s <seconds>  start time" << endl;
  cout << "-d <seconds>  duration" << endl;
}

double getArgs(string& filename, double& start, double& duration)
{
  InputParser argparse(argc, argv);

  if (argparse.cmdOptionExists("-h"))
    displayHelp();
  if (!argparse.getCmdOption("-f", filename))
    displayHelp();
  argparse.getCmdOption("-s", start);
  argparse.getCmdOption("-d", duration);
}

SalsaRosbag::openBag()
{
  try
  {
    bag.open(filename.c_str(), rosbag::bagmode::Read);
  }
  catch(rosbag::BagIOException e)
  {
    ROS_ERROR("unable to load rosbag %s, %s", filename.c_str(), e.what());
    exit(-1);
  }
}

void SalsaRosbag::getEndTime()
{
  double end_time = start_time + duration;
  end_time = (end_time < view.getEndTime().toSec() - view.getBeginTime().toSec()) ? end_time : view.getEndTime().toSec() - view.getBeginTime().toSec();
  if (verbose)
    cout << "Playing bag from: = " << start_time << "s to: " << end_time << "s" << endl;
  return end_time;
}

void SalsaRosbag::parseBag()
{
  ProgressBar prog(view.size(), 100);
  ros::Time bag_start = view.getBeginTime() + ros::Duration(start);
  ros::Time bag_end = view.getBeginTime() + ros::Duration(end);
  rosbag::View::iterator it;
  int i = 0;
  for (it = view.begin(); it != view.end(); it++)
  {
    rosbag::MessageInstance* m = it;
    if (m->getTime() < bag_start)
      continue;

    if (m->getTime() > bag_end)
      break;

    prog.print(i++);
    string datatype = m->getDataType();

    if (datatype.compare("sensor_msgs/Imu") == 0)
      imuCB(m);
  }
}

void imuCB(rosbag::MessageInstance* m)
{
  sensor_msgs::Imu* imu(m.instantiate<sensor_msgs::Imu>());
  double t = imu->header.stamp - bag_start;
  Vector6d z;
  z << imu->linear_acceleration.x,
       imu->linear_acceleration.y,
       imu->linear_acceleration.z,
       imu->angular_velocity.x,
       imu->angular_velocity.y,
       imu->angular_velocity.z;
  salsa.imageCallback();
  if (!seen_imu_topic.empty() && seen_imu_topic.compare(m.getTopic()))
    ROS_WARN_ONCE("Subscribed to Two IMU messages, use the -imu argument to specify IMU topic");

  seen_imu_topic = m.getTopic();
}

int main(int argc, char** argv)
{
  string filename;
  double start, duration;
  getArgs(filename, start, duration);

  rosbag::Bag bag = openBag(filename);
  rosbag::View view(bag);

  // Figure out the end time of the bag
  double end = getEndTime(view, start, druation);

  parse_bag(start, end, view);
}
