from plotResults import plotResults
import subprocess
import yaml
import os

# bagfile = "/home/superjax/rosbag/mynt_mocap/mocap2.bag"
# bagfile = "/home/superjax/rosbag/mynt_mocap/mocap2_adjust.bag"
directory = "/tmp/Salsa/GNSSHardware"
prefix = "Est/"

if not os.path.exists(os.path.join(directory, prefix)):
    os.makedirs(os.path.join(directory, prefix))

params = yaml.load(file("../params/salsa.yaml"))
# params['bag_name'] = "/home/superjax/rosbag/outdoor_raw_gps/gps_raw1.bag"
# params['bag_name'] = "/home/superjax/rosbag/outdoor_raw_gps/gps_raw2.bag"
# params['bag_name'] = "/home/superjax/rosbag/outdoor_raw_gps/gps_raw3.bag"
params['bag_name'] = "/home/superjax/rosbag/outdoor_raw_gps/gps_raw4_degraded.bag"
params['log_prefix'] = os.path.join(directory, prefix)
params['update_on_mocap'] = False
params['disable_mocap'] = True
params['disable_vision'] = True
params['disable_gnss'] = False
params['static_start_imu_thresh'] = 30
params['start_time'] = 0
params['duration'] = 200
params['x_b2o'] = [0, 0, 0, 1, 0, 0, 0]
params['x_b2m'] = [0, 0, 0, 1, 0, 0, 0]
param_filename = os.path.join(directory, "tmp.yaml")
yaml.dump(params, file(param_filename, 'w'))


process = subprocess.call(("cmake", "..", "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-GNinja", "-DBUILD_ROS=ON"), cwd="../build")
process = subprocess.call(("ninja", "salsa_rosbag"), cwd="../build")
process = subprocess.call(("./salsa_rosbag", "-f", param_filename), cwd="../build")


plotResults(directory, False)