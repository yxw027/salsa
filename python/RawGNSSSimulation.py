from plotResults import plotResults
import subprocess
import os

directory = "/tmp/Salsa/RawGNSSSimulation/"
if not os.path.exists(directory):
    os.makedirs(directory)
    
process = subprocess.call(("cmake", "..", "-DCMAKE_BUILD_TYPE=RelWithDebInfo"), cwd="../build")
process = subprocess.call(("make", "-j12", "-l12"), cwd="../build")
process = subprocess.call(("./test_salsa", "--gtest_filter=Salsa.RawGNSSSimulation"), cwd="../build")


plotResults(directory)
