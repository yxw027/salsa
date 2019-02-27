import numpy as np
from typedefs import StateType, OptType
from plotWindow import plotWindow
import matplotlib.pyplot as plt
import os


def plotResults(prefix):
    np.set_printoptions(linewidth=150)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    state = np.fromfile(os.path.join(prefix,"State.log"), dtype=StateType)
    truth = np.fromfile(os.path.join(prefix,"Truth.log"), dtype=StateType)
    opt = np.fromfile(os.path.join(prefix,"Opt.log"), dtype=OptType)

    imu_titles = [r"$acc_x$", r"$acc_y$", r"$acc_z$",
                  r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]
    pw = plotWindow()
    
    f = plt.figure()
    plt.suptitle('Bias')
    for i in range(3):
        for j in range(2):
            plt.subplot(3,2,i*2+j+1)
            plt.plot(state['t'], state['b'][:, j*3+i])
            plt.title(imu_titles[j*3+i])
    pw.addPlot("Bias", f)

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


    # f = plt.figure()
    # plt.suptitle('Acc')
    # for i in range(3):
    #     plt.subplot(3, 1, i+1)
    #     plt.title(imu_titles[i])
    #     plt.plot(imu[:,0], imu[:,i+1])
    # pw.addPlot("Acc", f)


    # f = plt.figure()
    # plt.suptitle('Omega')
    # for i in range(3):
    #     plt.subplot(3, 1, i+1)
    #     plt.title(imu_titles[i+3])
    #     plt.plot(imu[:,0], imu[:,i+4])
    # pw.addPlot("Omega", f)






    pw.show()

if __name__ == '__main__':
    # plotResults("/tmp/Salsa.MocapSimulation")
    plotResults("/tmp/Salsa/")
