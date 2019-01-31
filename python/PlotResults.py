import numpy as np
from typedefs import StateType, OptType
from plotWindow import plotWindow
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=150)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

state_data = np.fromfile("/tmp/Salsa.State.log", dtype=StateType)
opt_data = np.fromfile("/tmp/Salsa.State.log", dtype=OptType)

debug = 1
pw = plotWindow()
