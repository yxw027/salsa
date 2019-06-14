from plotResults import plotResults
import subprocess
import os

directory = "/tmp/Salsa/mixedSimulation/"

subprocess.call(("cmake", "..", "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-GNinja"), cwd="../build")
subprocess.call(("ninja", "mixedSim"), cwd="../build")
subprocess.call(("./mixedSim"), cwd="../build")

plotResults(directory, False)