from plotResults import plotResults
import subprocess

if not subprocess.check_output(("cmake", "..", "-DCMAKE_BUILD_TYPE=RelWithDebInfo"), cwd="../build"): quit()
if not subprocess.check_output(("make", "-j12", "-l12"), cwd="../build"): quit()
subprocess.call(("./salsa_rosbag",
                 "-f", "/home/superjax/rosbag/outdoor_raw_gps/gps_raw1.bag",
                 "-y", "../params/salsa.yaml",
                 "-p", "/tmp/Salsa/GNSSHardware"), cwd="../build")
plotResults("/tmp/Salsa/")
