#include "salsa/salsa_rosbag.h"

using namespace std;
using namespace gnss_utils;
using namespace Eigen;
using namespace xform;

namespace salsa
{

SalsaRosbag::SalsaRosbag(int argc, char** argv)
{
    start_ = 0;
    duration_ = 1e3;

    param_filename_ = SALSA_DIR"/params/salsa.yaml";
    getArgs(argc, argv);

    loadParams();
    openBag();
//    getMocapOffset();

    got_imu_ = false;
    salsa_.init(param_filename_);
    truth_log_.open(salsa_.log_prefix_ + "/../Truth.log");
    imu_log_.open(salsa_.log_prefix_ + "/Imu.log");
}

void SalsaRosbag::loadParams()
{
    get_yaml_node("bag_name", param_filename_, bag_filename_);
    get_yaml_node("imu_topic", param_filename_, imu_topic_);
    get_yaml_node("mocap_topic", param_filename_, mocap_topic_);
    get_yaml_node("image_topic", param_filename_, image_topic_);
    get_yaml_node("start_time", param_filename_, start_);
    get_yaml_node("duration", param_filename_, duration_);

    // Load Sensor Noise Parameters
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

    // Configure Motion Capture Frame
    get_yaml_node("mocap_rate", param_filename_, mocap_rate_);
}

void SalsaRosbag::displayHelp()
{
    cout << "ROS bag parser" <<endl;
    cout << "-h            display this message" << endl;
    cout << "-f <filename> configuration yaml file to parse, [" SALSA_DIR "/params/salsa.yaml]" << endl;
    exit(0);
}

void SalsaRosbag::getArgs(int argc, char** argv)
{
    InputParser argparse(argc, argv);

    if (argparse.cmdOptionExists("-h"))
        displayHelp();
    argparse.getCmdOption("-f", param_filename_);
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
        fprintf(stderr, "unable to load rosbag %s, %s", bag_filename_.c_str(), e.what());
        throw e;
    }

    bag_start_ = view_->getBeginTime() + ros::Duration(start_);
    bag_end_ = bag_start_ + ros::Duration(duration_);

    if (bag_end_ > view_->getEndTime())
        bag_end_ = view_->getEndTime();

    delete view_;
    view_ = new rosbag::View(bag_, bag_start_, bag_end_);
}


void SalsaRosbag::parseBag()
{
    ProgressBar prog(view_->size(), 80);
    int i = 0;
    for(rosbag::MessageInstance const m  : (*view_))
    {
        if (m.getTime() < bag_start_)
            continue;

        if (m.getTime() > bag_end_)
            break;


        if (m.isType<sensor_msgs::Imu>() && m.getTopic().compare(imu_topic_) == 0)
        {
            prog.print(i++, (m.getTime() - bag_start_).toSec());
            imuCB(m);
        }
        else if (m.isType<geometry_msgs::PoseStamped>() && m.getTopic().compare(mocap_topic_) == 0)
            poseCB(m);
        else if (m.isType<nav_msgs::Odometry>())
            odomCB(m);
        else if (m.isType<sensor_msgs::Image>() && m.getTopic().compare(image_topic_) == 0)
            imgCB(m);
        else if (m.isType<sensor_msgs::CompressedImage>() && m.getTopic().compare(image_topic_) == 0)
            compressedImgCB(m);
        else if (m.isType<inertial_sense::GNSSObsVec>())
            obsCB(m);
        else if (m.isType<inertial_sense::GNSSEphemeris>())
            ephCB(m);
    }
    prog.finished();
    cout << endl;
    cout.flush();
}

void SalsaRosbag::imuCB(const rosbag::MessageInstance& m)
{
    got_imu_ = true;
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
    {
        std::cout << "Found NaNs in IMU data, skipping measurement" << std::endl;
        return;
    }

    salsa_.imuCallback(t, z, imu_R_);

    imu_log_.log(t);
    imu_log_.logVectors(z);
}

void SalsaRosbag::obsCB(const rosbag::MessageInstance& m)
{
    inertial_sense::GNSSObsVecConstPtr obsvec = m.instantiate<inertial_sense::GNSSObsVec>();
    salsa::ObsVec z;
    z.reserve(obsvec->obs.size());

    for (const auto& o : obsvec->obs)
    {
        salsa::Obs new_obs;
        new_obs.t = GTime::fromTime(o.time.time, o.time.sec);
        new_obs.sat = o.sat;
        new_obs.rcv = o.rcv;
        new_obs.SNR = o.SNR;
        new_obs.LLI = o.LLI;
        new_obs.code = o.code;
        new_obs.qualP = o.qualP;
        new_obs.qualL = o.qualL;
        new_obs.z << o.P, -o.D*Satellite::LAMBDA_L1, o.L;
        z.push_back(new_obs);
    }
    salsa_.obsCallback(z);
}

void SalsaRosbag::ephCB(const rosbag::MessageInstance &m)
{
    inertial_sense::GNSSEphemerisConstPtr eph = m.instantiate<inertial_sense::GNSSEphemeris>();
    eph_t new_eph;
    if (new_eph.sat > 90)
        return; // Don't use Beidou or SBAS
    new_eph.sat = eph->sat;
    new_eph.iode = eph->iode;
    new_eph.iodc = eph->iodc;
    new_eph.sva = eph->sva;
    new_eph.svh = eph->svh;
    new_eph.week = eph->week;
    new_eph.code = eph->code;
    new_eph.flag = eph->flag;
    new_eph.toe = GTime::fromTime(eph->toe.time, eph->toe.sec);
    new_eph.toc = GTime::fromTime(eph->toc.time, eph->toc.sec);
    new_eph.ttr = GTime::fromTime(eph->ttr.time, eph->ttr.sec);
    new_eph.A = eph->A;
    new_eph.e = eph->e;
    new_eph.i0 = eph->i0;
    new_eph.OMG0 = eph->OMG0;
    new_eph.omg = eph->omg;
    new_eph.M0 = eph->M0;
    new_eph.deln = eph->deln;
    new_eph.OMGd = eph->OMGd;
    new_eph.idot = eph->idot;
    new_eph.crc = eph->crc;
    new_eph.crs = eph->crs;
    new_eph.cuc = eph->cuc;
    new_eph.cus = eph->cus;
    new_eph.cic = eph->cic;
    new_eph.cis = eph->cis;
    new_eph.toes = eph->toes;
    new_eph.fit = eph->fit;
    new_eph.f0 = eph->f0;
    new_eph.f1 = eph->f1;
    new_eph.f2 = eph->f2;
    new_eph.tgd[0] = eph->tgd[0];
    new_eph.tgd[1] = eph->tgd[1];
    new_eph.tgd[2] = eph->tgd[2];
    new_eph.tgd[3] = eph->tgd[3];
    new_eph.Adot = eph->Adot;
    new_eph.ndot = eph->ndot;
    salsa_.ephCallback(GTime::fromUTC(m.getTime().sec, m.getTime().nsec /1e9), new_eph);
}

void SalsaRosbag::poseCB(const rosbag::MessageInstance& m)
{
    if (!got_imu_)
        return;

    geometry_msgs::PoseStampedConstPtr pose = m.instantiate<geometry_msgs::PoseStamped>();

    if ((m.getTime() - prev_mocap_).toSec() < 1.0/mocap_rate_)
        return;

    double t = (m.getTime() - bag_start_).toSec();
    Xformd z;
    z.arr() << pose->pose.position.x,
               pose->pose.position.y,
               pose->pose.position.z,
               pose->pose.orientation.w,
               pose->pose.orientation.x,
               pose->pose.orientation.y,
               pose->pose.orientation.z;
    z.q().normalize(); // I am a little worried that I have to do this.

    salsa_.mocapCallback(t, z, mocap_R_);

    ros::Duration dt = m.getTime() - prev_mocap_;
    Vector3d v = z.q_.rotp(z.t() - x_I2m_prev_.t())/dt.toSec();
    Vector6d b = Vector6d::Ones() * NAN;
    Vector2d tau = Vector2d::Ones() * NAN;
    Vector7d x_e2n = Vector7d::Ones() * NAN;
    Vector7d x_b2c = Vector7d::Ones() * NAN;
    truth_log_.log(t);
    truth_log_.logVectors(z.arr(), z.q().euler(), v, b, tau, x_e2n, x_b2c);
    int32_t multipath(0), denied(0);
    truth_log_.log(multipath, denied);
    for (int i = 0; i < salsa_.ns_; i++)
    {
        double sw = NAN;
        truth_log_.log(sw);
    }


    x_I2m_prev_ = z;
    prev_mocap_ = m.getTime();
}

void SalsaRosbag::odomCB(const rosbag::MessageInstance &m)
{
    nav_msgs::OdometryConstPtr odom = m.instantiate<nav_msgs::Odometry>();
//    GTime gtime = GTime::fromUTC(odom->header.stamp.sec, odom->header.stamp.nsec/1e9);
    if (salsa_.start_time_.tow_sec < 0)
        return;

    Xformd z;
    z.arr() << odom->pose.pose.position.x,
               odom->pose.pose.position.y,
               odom->pose.pose.position.z,
               odom->pose.pose.orientation.w,
               odom->pose.pose.orientation.x,
               odom->pose.pose.orientation.y,
               odom->pose.pose.orientation.z;
    if (salsa_.current_node_ < 0)
        salsa_.setInitialState(z);

//    double t = (m.getTime() - bag_start_).toSec();
//    if (imu_count_between_nodes_ > 20)
//    {
//        salsa_.mocapCallback(t, z, mocap_R_);
//        imu_count_between_nodes_ = 0;
//    }


//    truth_log_.log((m.getTime() - bag_start_).toSec());
    Vector3d v;
    v << odom->twist.twist.linear.x,
         odom->twist.twist.linear.y,
         odom->twist.twist.linear.z;
    Vector6d b = Vector6d::Ones() * NAN;
    Vector2d tau = Vector2d::Ones() * NAN;
    Vector7d x_e2n = Vector7d::Ones() * NAN;
    Vector7d x_b2c = Vector7d::Ones() * NAN;
    truth_log_.log((odom->header.stamp - bag_start_).toSec());
    truth_log_.logVectors(z.arr(), v, b, tau, x_e2n, x_b2c);
}

void SalsaRosbag::imgCB(const rosbag::MessageInstance &m)
{
    if (!got_imu_)
        return;

    sensor_msgs::ImageConstPtr img = m.instantiate<sensor_msgs::Image>();



}

void SalsaRosbag::compressedImgCB(const rosbag::MessageInstance &m)
{

}

void SalsaRosbag::imgCB(const cv_bridge::CvImagePtr &img)
{

}

//void SalsaRosbag::getMocapOffset()
//{
//    ros::Duration biggest_dt = ros::DURATION_MIN;
//    for(rosbag::MessageInstance const m  : (*view_))
//    {
//        if (m.getTime() < bag_start_) continue;
//        if (m.getTime() > bag_end_) break;

//        if (m.isType<geometry_msgs::PoseStamped>() && m.getTopic().compare(mocap_topic_) == 0)
//        {
//            ros::Time header = m.instantiate<geometry_msgs::PoseStamped>()->header.stamp;
//            ros::Duration dt = header - m.getTime();
//            if (dt > biggest_dt)
//                biggest_dt = dt;
//        }
//    }
//    mocap_offset_ = biggest_dt;
//    prev_mocap_ = ros::Time(0,0);
//}

}

int main(int argc, char** argv)
{
    salsa::SalsaRosbag thing(argc, argv);
    thing.parseBag();
}
