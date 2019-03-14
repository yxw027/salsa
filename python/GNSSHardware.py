from plotResults import plotResults
import subprocess

process = subprocess.call(("make", "-j12", "-l12"), cwd="../build")
process = subprocess.call(("./test_salsa", "--gtest_filter=Salsa.MocapSimulation"), cwd="../build")


plotResults("/tmp/Salsa/MocapSimulation/")
