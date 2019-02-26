import rosbag
import rospy
import sys
from tqdm import tqdm

def adjustTime(inputfile, outputfile):

    # print console header
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

if __name__ == '__main__':
    adjustTime("/home/superjax/rosbag/MocapCalCollect2.bag",
               "/home/superjax/rosbag/MocapCalCollect2Adjust.bag")
Vector6d::Zero()

