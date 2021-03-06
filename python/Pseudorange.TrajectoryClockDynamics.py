import matplotlib.pyplot as plt
import numpy as np
from plotWindow import plotWindow

np.set_printoptions(linewidth=150)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
pw = plotWindow()

LOG_WIDTH = 1 + 7 + 3 + 7 + 3 + 7 + 3 + 2 + 2 + 2

data = np.reshape(np.fromfile('/tmp/Pseudorange.TrajectoryClockDynamics.log', dtype=np.float64), (-1, LOG_WIDTH))

t = data[:,0]
xhat0 = data[:,1:8]
vhat0 = data[:,8:11]
xhatf = data[:, 11:18]
vhatf = data[:, 18:21]
x = data[:, 21:28]
v = data[:, 28:31]
tauhat0 = data[:,31:33]
tauhatf = data[:,33:35]
tau = data[:,35:37]

xtitles = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
vtitles = ['vx', 'vy', 'vz']
tautitles = [r'$\tau$', r'$\dot{\tau}$']



f = plt.figure(dpi=150)
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.title(xtitles[i])
    plt.plot(t, xhat0[:,i], label="$\hat{x}_0$")
    plt.plot(t, xhatf[:,i], '--', linewidth=3, label="$\hat{x}_f$")
    plt.plot(t, x[:,i], label="$x$")
    plt.legend()
pw.addPlot("Position", f)

f = plt.figure(dpi=150)
for i in range(4):
    plt.subplot(4, 1, i+1)
    plt.title(xtitles[i+3])
    plt.plot(t, xhat0[:,i+3], label="$\hat{x}_0$")
    plt.plot(t, xhatf[:,i+3], '--', linewidth=3, label="$\hat{x}_f$")
    plt.plot(t, x[:,i+3], label="$x$")
    plt.legend()
pw.addPlot("Attitude", f)

f = plt.figure(dpi=150)
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.title(vtitles[i])
    plt.plot(t, vhat0[:,i], label="$\hat{x}_0$")
    plt.plot(t, vhatf[:,i], '--', linewidth=3, label="$\hat{x}_f$")
    plt.plot(t, v[:,i], label="$x$")
    plt.legend()
pw.addPlot("Velocity", f)

f = plt.figure(dpi=150)
for i in range(2):
    plt.subplot(2, 1, i+1)
    plt.title(tautitles[i])
    plt.plot(t, tauhat0[:,i], label="$\hat{x}_0$")
    plt.plot(t, tauhatf[:,i], '--', linewidth=3, label="$\hat{x}_f$")
    plt.plot(t, tau[:,i], label="$x$")
    plt.legend()
pw.addPlot("Clock Bias", f)

pw.show()