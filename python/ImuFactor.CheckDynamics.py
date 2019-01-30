import matplotlib.pyplot as plt
import numpy as np

from plotWindow import plotWindow

np.set_printoptions(linewidth=150)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
pw = plotWindow()

LOG_WIDTH = 1 + 10 + 10 + 9 + 10 + 6

data = np.reshape(np.fromfile('/tmp/ImuFactor.CheckDynamics.log', dtype=np.float64), (-1,LOG_WIDTH))

t = data[:, 0]
y = data[:, 1:11]
yhat = data[:, 11:21]
dy = data[:, 21:30]
ycheck = data[:, 30:40]
u = data[:, 40:46]

error = ycheck - yhat

f = plt.figure()
plt.suptitle('Position')
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(t, y[:,i], label='y', linewidth=2)
    plt.plot(t, yhat[:,i], label=r'$\hat{y}$')
    plt.plot(t, ycheck[:,i], '--', label=r'$y + \tilde{y}$')
    if i == 0:
        plt.legend()
pw.addPlot("Position", f)

f = plt.figure()
plt.suptitle('Velocity')
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(t, y[:,i+7], label='y', linewidth=2)
    plt.plot(t, yhat[:,i+7], label=r'$\hat{y}$')
    plt.plot(t, ycheck[:,i+7], '--', label=r'$y + \tilde{y}$')
    if i == 0:
        plt.legend()
pw.addPlot("Velocity", f)

f = plt.figure()
plt.suptitle('Attitude')
for i in range(4):
    plt.subplot(4, 1, i+1)
    plt.plot(t, y[:,i+3], label='y', linewidth=2)
    plt.plot(t, yhat[:,i+3], label=r'$\hat{y}$')
    plt.plot(t, ycheck[:,i+3], '--', label=r'$y + \tilde{y}$')
    if i == 0:
        plt.legend()
pw.addPlot("Attitude", f)

f = plt.figure()
plt.suptitle('Input')
labels=['a_x', 'a_y', 'a_z', r'$\omega_x$', r'$\omega_{y}$', r'$\omega_{z}$']
for j in range(2):
    for i in range(3):
        plt.subplot(3, 2, i*2+1 + j)
        plt.plot(t, u[:,i+j*3], label=labels[i+j*3])
        plt.legend()
pw.addPlot("Input", f)

f = plt.figure()
plt.suptitle('Error')
labels=[r'$p_x$', r'$p_y$', r'$p_z$',
        r'$v_x$', r'$v_y$', r'$v_z$',
        r'$q_x$', r'$q_y$', r'$q_z$']
for j in range(3):
    for i in range(3):
        plt.subplot(3, 3, i*3+1+j)
        plt.plot(t, error[:,i+j*3], label=labels[i+j*3])
        plt.legend()

pw.addPlot("Error", f)

pw.show()
