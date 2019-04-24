from plotResults import plotResults
import subprocess

# subprocess.call(("cmake", "..", "-GNinja", "-DCMAKE_BUILD_TYPE=RelWithDebInfo"), cwd="../build")
# subprocess.call(("ninja"), cwd="../build")
subprocess.call(("./salsa_rosbag",
                 "-f", "/home/superjax/rosbag/outdoor_raw_gps/gps_raw1.bag",
                 "-y", "../params/salsa.yaml",
                 "-p", "/tmp/Salsa/GNSSHardware/"), cwd="../build")
plotResults("/tmp/Salsa/GNSSHardware")
    