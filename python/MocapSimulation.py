from plotResults import plotResults
import subprocess

process = subprocess.call(("make", "-j12", "-l12"), cwd="../build")
subprocess.call(("./mocapSim"), cwd="../build")

plotResults("/tmp/Salsa/MocapSimulation/")
    