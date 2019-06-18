from plotResults import plotResults
import subprocess

subprocess.call(("cmake", "..", "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-GNinja"), cwd="../build")
subprocess.call(("ninja", "featSim"), cwd="../build")
subprocess.call(("./featSim"), cwd="../build")

plotResults("/tmp/Salsa/featSimulation/")
