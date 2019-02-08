import numpy as np
from typedefs import StateType, OptType
from plotWindow import plotWindow
import matplotlib.pyplot as plt


def plotResults(prefix):
    np.set_printoptions(linewidth=150)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    state = np.fromfile(prefix + ".State.log", dtype=StateType)
    truth = np.fromfile(prefix + ".Truth.log", dtype=StateType)
    opt_windows = np.fromfile(prefix + ".Opt.log", dtype=OptType)

    pw = plotWindow()

    xtitles = ['$p_x$', '$p_y$', '$p_z$', '$q_w$', '$q_x$', '$q_y$', '$q_z$']
    vtitles = ['$v_x$', '$v_y$', '$v_z$']
    tautitles = [r'$\tau$', r'$\dot{\tau}$']

    f = plt.figure()
    plt.suptitle('Position')
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.title(xtitles[i])
        plt.plot(truth['t'], truth['x'][:,i], label='x')
        plt.plot(state['t'], state['x'][:,i], label=r'\hat{x}')
        # plt.plot(state[:,0], state[:,i], label=r'\hat{x}')
        if i == 0:
            plt.legend()
    pw.addPlot("Position", f)

    f = plt.figure()
    plt.suptitle('Attitude')
    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.title(xtitles[i+3])
        plt.plot(truth['t'], truth['x'][:,i+3], label='x')
        plt.plot(state['t'], state['x'][:,i+3], label=r'\hat{x}')
        # plt.plot(state[:,0], state[:,i+3], label=r'\hat{x}')
        if i == 0:
            plt.legend()
    pw.addPlot("Attitude", f)

    f = plt.figure()
    plt.suptitle('Velocity')
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.title(xtitles[i])
        plt.plot(truth['t'], truth['v'][:,i], label='x')
        plt.plot(state['t'], state['v'][:,i], label=r'\hat{x}')
        # plt.plot(state[:,0], state[:,i+7], label=r'\hat{x}')
        if i == 0:
            plt.legend()
    pw.addPlot("Velocity", f)

    pw.show()

if __name__ == '__main__':
    # plotResults("/tmp/Salsa.MocapSimulation")
    plotResults("/tmp/Salsa.Pseudorange.ImuTrajectory")
