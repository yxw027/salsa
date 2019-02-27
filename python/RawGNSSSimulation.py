from plotResults import plotResults
import subprocess
import os

directory = "/tmp/Salsa/RawGNSSSimulation/"
if not os.path.exists(directory):
    os.makedirs(directory)

process = subprocess.Popen(("./test_salsa", "--gtest_filter=Salsa.RawGNSSSimulation"), cwd="../build")
process.wait()


plotResults(directory)
