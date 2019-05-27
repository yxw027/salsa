from plotResults import plotResults
import subprocess

subprocess.call(("cmake", "..", "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-GNinja"), cwd="../build")
subprocess.call(("ninja", "mixedSim"), cwd="../build")
subprocess.call(("./mixedSim"), cwd="../build")

plotResults("/tmp/Salsa/MixedSimulation/")
