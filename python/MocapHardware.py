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


process = subprocess.Popen(("./salsa_rosbag", "-f", "/home/superjax/rosbag/MocapCalCollect2.bag", "-y", "/tmp/Salsa/MocapHardware/tmp.yaml"), cwd="../build")
process.wait()


plotResults("/tmp/Salsa/MocapHardware/")
