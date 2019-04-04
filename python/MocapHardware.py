from plotResults import plotResults
import subprocess
import yaml
import os

directory = "/tmp/Salsa/MocapHardware/"
if not os.path.exists(directory):
    os.makedirs(directory)

params = yaml.load(file("../params/salsa.yaml"))
params['log_prefix'] = directory
yaml.dump(params, file(os.path.join(directory, "tmp.yaml"), 'w'))


process = subprocess.call(("cmake", "..", "-DCMAKE_BUILD_TYPE=RelWithDebInfo"), cwd="../build")
process = subprocess.call(("make", "-j12", "-l12"), cwd="../build")
process = subprocess.call(("./salsa_rosbag", "-f", "/home/superjax/rosbag/leo_mocap/MocapCalCollect2.bag", "-y", directory + "tmp.yaml"), cwd="../build")


plotResults("/tmp/Salsa/MocapHardware/")
