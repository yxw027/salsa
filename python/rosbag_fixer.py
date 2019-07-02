import rosbag
import rospy
import sys
from tqdm import tqdm
from inertial_sense.msg import GNSSObsVec
from geometry_msgs.msg import PoseStamped

def adjustTime(inputfile, outputfile, gps_topics, ignored_topics):
    dt = gpsOffset(inputfile, gps_topics)


    inbag = rosbag.Bag(inputfile)
    outbag = rosbag.Bag(outputfile, 'w')

    topics = [x for x in inbag.get_type_and_topic_info()[1].keys() if x not in ignored_topics]

    last = rospy.Time(0)
    last_topic = ""
    for topic, msg, t in tqdm(inbag.read_messages(topics), total=inbag.get_message_count()):
        if hasattr(msg, "header") and msg.header.stamp.to_sec() < 1.0:
            msg.header.stamp = t

        new_time = []
        if topic in gps_topics and hasattr(msg , "header"):
            new_time = msg.header.stamp
        else:
            if hasattr(msg , "header"):
                msg.header.stamp += dt
                new_time = msg.header.stamp
            else:
                new_time = t + dt
        outbag.write(topic, msg, new_time)
        if (last - new_time).to_sec() > 1.0:
            print((last - new_time).to_sec(), topic, last_topic)
        last = new_time
        last_topic = topic
    outbag.reindex()
    outbag.close()
    inbag.close()
    

def gpsOffset(inputfile, gps_topics):
    bag = rosbag.Bag(inputfile)

    biggest_dt = rospy.Duration(0)
    last_topic= ""
    for topic, msg, t in tqdm(bag.read_messages(topics=gps_topics), total=bag.get_message_count()):
        if not hasattr(msg , "header"):
            continue
        dt = msg.header.stamp - t
        if abs((dt - biggest_dt).to_sec()) > 1.0:
            print ((dt - biggest_dt).to_sec(), topic, last_topic)
        if dt > biggest_dt:
            biggest_dt = dt
        last_topic = topic
    print biggest_dt.secs
    return biggest_dt


def gtime2rostime(gtime):
    rostime = rospy.Time()
    rostime.secs = gtime.time
    rostime.nsecs = gtime.sec*1e9
    return rostime


def convertObsType(inputfile, outputfile):
    print("input file: %s" % inputfile)
    print('output file: %s' % outputfile)

    outbag = rosbag.Bag(outputfile, 'w')

    try:
        bag = rosbag.Bag(inputfile)
    except:
        print("No bag file found at %s" % inputfile)
        quit(1)

    obs = GNSSObsVec()
    for topic, msg, t in tqdm(bag.read_messages(), total=bag.get_message_count()):
        if topic == "/gps/obs":
            if len(obs.obs) > 0 and (obs.obs[0].time.sec != msg.time.sec or obs.obs[0].time.time != msg.time.time):
                obs.header = gtime2rostime(obs.obs[0].time)
                outbag.write(topic, obs, gtime2rostime(obs.obs[0].time))
                obs.obs = []
            obs.obs.append(msg)
        else:
            outbag.write(topic, msg, msg.header.stamp)


if __name__ == '__main__':
    adjustTime("/home/superjax/rosbag/gps_carry/bright_small1.bag",
               "/home/superjax/rosbag/gps_carry/bright_small1_adjust.bag",
               ["/gps", "/ins", "/gps/geph", "gps/obs", "/uins/imu"],
               ["/output_raw", "/rc_raw"])


