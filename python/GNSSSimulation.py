from plotResults import plotResults
import subprocess
import os

directory = "/tmp/Salsa/RawGNSSSimulation/"
if not os.path.exists(directory):
    os.makedirs(directory)
    
subprocess.call(("cmake", "..", "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-GNinja"), cwd="../build")
subprocess.call("ninja", cwd="../build")
subprocess.call(("./gnssSim"), cwd="../build")


plotResults(directory)
