from plotResults import plotResults
import subprocess
# #
# process = subprocess.call(("cmake", "..", "-DCMAKE_BUILD_TYPE=RelWithDebInfo"), cwd="../build")
# process = subprocess.call(("make", "-j12", "-l12"), cwd="../build")
# process = subprocess.call(("./test_salsa", "--gtest_filter=Salsa.FeatSimulation"), cwd="../build")
#

plotResults("/tmp/Salsa/FeatSimulation/")
