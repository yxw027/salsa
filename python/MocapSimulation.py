from plotResults import plotResults
import subprocess

process = subprocess.Popen(("./test_salsa", "--gtest_filter=Salsa.MocapSimulation"), cwd="../build")
process.wait()


plotResults("/tmp/Salsa/MocapSimulation/")
