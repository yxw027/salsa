#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <opencv2/highgui.hpp>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>


#include "multirotor_sim/utils.h"

using namespace std;

void displayHelp()
{
    cout << "ROSbag Decompressor" <<endl;
    cout << "-h            display this message" << endl;
    cout << "-f <filename> ROSbag to decompress" << endl;
    cout << "-o <filename> Output Filename" << endl;
    cout << "-s            Start time (s) [0]" << endl;
    cout << "-u            Duration (s) [Inf]" << endl;
    exit(0);
}

int main(int argc, char**argv)
{
  InputParser argparse(argc, argv);
\
  std::string input_filename;
  std::string output_filename;
  double start = 0;
  double duration = INFINITY;

  if (argparse.cmdOptionExists("-h"))
    displayHelp();
  if (!argparse.getCmdOption("-f", input_filename))
    displayHelp();
  if (!argparse.getCmdOption("-o", output_filename))
    displayHelp();
  argparse.getCmdOption("-s", start);
  argparse.getCmdOption("-u", duration);

  cout << "ROSbag Decompressor" <<endl;
  cout << "Source " << input_filename << endl;
  cout << "Destination " << output_filename << endl;

  rosbag::Bag inbag(input_filename);
  rosbag::Bag outbag(output_filename, rosbag::bagmode::Write);
  rosbag::View* view = new rosbag::View(inbag);

  ros::Time bag_start = view->getBeginTime() + ros::Duration(start);
  ros::Time bag_end = view->getEndTime();
  if (std::isfinite(duration))
    bag_end = bag_start + ros::Duration(duration);
  delete view;

  view = new rosbag::View(inbag, bag_start, bag_end);

  ProgressBar prog(view->size(), 80);

  int i = 0;
  for (rosbag::MessageInstance const m : (*view))
  {
    if (m.getTime() < bag_start)
      continue;
    if (m.getTime() > bag_end)
      break;

    prog.print(i++);

    if (m.isType<sensor_msgs::CompressedImage>())
    {
      sensor_msgs::CompressedImageConstPtr img = m.instantiate<sensor_msgs::CompressedImage>();
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(img);
      static int skip = 0;
      skip = (skip + 1) % 99;

      std::string topic = m.getTopic().substr(0, m.getTopic().find_last_of("\\/"));
      if ( skip == 0)
      {
        cv::imshow(topic, cv_ptr->image);
        cv::waitKey(1);
      }
      sensor_msgs::ImagePtr uncompressed = cv_ptr->toImageMsg();
      outbag.write(topic, m.getTime(), uncompressed);
    }
    else
    {
      outbag.write(m.getTopic(), m.getTime(), m);
    }
  }
  prog.finished();
}
