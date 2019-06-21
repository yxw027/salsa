from plotResults import plotResults
import subprocess
import os

directory = "/tmp/Salsa/compareSimulation/"

# subprocess.call(("cmake", "..", "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-GNinja"), cwd="../build")
# subprocess.call(("ninja", "compareSim"), cwd="../build")
# subprocess.call(("./compareSim"), cwd="../build")

plotResults(directory, False, saveFig=True)
