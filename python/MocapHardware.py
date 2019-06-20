from plotResults import plotResults
import subprocess
import yaml
import os

# bagfile = "/home/superjax/rosbag/mynt_mocap/mocap2.bag"
# bagfile = "/home/superjax/rosbag/mynt_mocap/mocap2_adjust.bag"
directory = "/tmp/Salsa/MocapHardware"
prefix = "Mocap/"

if not os.path.exists(os.path.join(directory, prefix)):
    os.makedirs(os.path.join(directory, prefix))

params = yaml.load(file("../params/salsa.yaml"))
params['log_prefix'] = os.path.join(directory, prefix)
params['update_on_mocap'] = False
params['disable_mocap'] = True
params['disable_vision'] = False
param_filename = os.path.join(directory, "tmp.yaml")
yaml.dump(params, file(param_filename, 'w'))


process = subprocess.call(("cmake", "..", "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-GNinja", "-DBUILD_ROS=ON"), cwd="../build")
process = subprocess.call(("ninja", "salsa_rosbag"), cwd="../build")
process = subprocess.call(("./salsa_rosbag", "-f", param_filename), cwd="../build")


plotResults(directory, False)