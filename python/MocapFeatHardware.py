from plotResults import plotResults
import subprocess
import yaml
import os

# bagfile = "/home/superjax/rosbag/mynt_mocap/mocap2.bag"
# bagfile = "/home/superjax/rosbag/mynt_mocap/mocap2_adjust.bag"
directory = "/tmp/Salsa/MocapFeatHardware/"
prefix = "Est/"

if not os.path.exists(os.path.join(directory, prefix)):
    os.makedirs(os.path.join(directory, prefix))

params = yaml.load(file("../params/salsa.yaml"))
params['bag_name'] = "/home/superjax/rosbag/mynt_mocap_ned/uncompressed/mocap3.bag"
# params['bag_name'] = "/home/superjax/rosbag/mynt_mocap_ned/uncompressed/mocap4.bag"
# params['bag_name'] = "/home/superjax/rosbag/mynt_mocap_ned/uncompressed/mocap5.bag"
# params['bag_name'] = "/home/superjax/rosbag/mynt_mocap_ned/uncompressed/mocap6.bag"
params['log_prefix'] = os.path.join(directory, prefix)
params['update_on_mocap'] = True
params['disable_mocap'] = False
params['disable_vision'] = True
params['update_on_vision'] = True
params['static_start_imu_thresh'] = 12
params['start_time'] = 0
params['duration'] = 100
params['enable_static_start'] = True
param_filename = os.path.join(directory, "tmp.yaml")
yaml.dump(params, file(param_filename, 'w'))


process = subprocess.call(("cmake", "..", "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-GNinja", "-DBUILD_ROS=ON"), cwd="../build")
process = subprocess.call(("ninja", "salsa_rosbag"), cwd="../build")
process = subprocess.call(("./salsa_rosbag", "-f", param_filename), cwd="../build")


plotResults(directory, False)