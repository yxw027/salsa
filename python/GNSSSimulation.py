from plotResults import plotResults
import subprocess

subprocess.call(("cmake", "..", "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-GNinja"), cwd="../build")
subprocess.call(("ninja", "gnssSim"), cwd="../build")
subprocess.call(("./gnssSim"), cwd="../build")

plotResults("/tmp/Salsa/gnssSimulation/")