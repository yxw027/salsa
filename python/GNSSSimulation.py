from plotResults import plotResults
import subprocess
import os

directory = "/tmp/Salsa/RawGNSSSimulation/"

subprocess.call(("cmake", "..", "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-GNinja"), cwd="../build")
subprocess.call(("ninja", "gnssSim"), cwd="../build")
subprocess.call(("./gnssSim"), cwd="../build")

plotResults(directory)
