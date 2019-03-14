import rosbag
import rospy
import sys
from tqdm import tqdm
from inertial_sense.msg import GNSSObsVec

def adjustTime(inputfile, outputfile):
    print("input file: %s" % inputfile)
    print('output file: %s' % outputfile)

    outbag = rosbag.Bag(outputfile, 'w')

    try:
        bag = rosbag.Bag(inputfile)
    except:
        print("No bag file found at %s" % inputfile)
        quit(1)

    for topic, msg, t in tqdm(bag.read_messages(), total=bag.get_message_count()):
        if hasattr(msg,"header") and topic != "/output_raw":
            outbag.write(topic, msg, msg.header.stamp)

    sys.stdout.flush()
    outbag.close()
    print ("done")

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
    # adjustTime("/home/superjax/rosbag/MocapCalCollect2.bag",
               # "/home/superjax/rosbag/MocapCalCollect2Adjust.bag")
    convertObsType("/home/superjax/rosbag/thor_outdoors_walk2.bag",
               "/home/superjax/rosbag/thor_outdoors_walk2_adjust.bag")


