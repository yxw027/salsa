import numpy as np
from typedefs import *
from plotWindow import plotWindow
import matplotlib.pyplot as plt
import os

def plotRawRes(rawGPSRes):
    f = plt.figure()
    for id in np.unique(rawGPSRes['id']):
        if id == -1: continue

        pass
    return f

def plotClockBias(state, truth, opt):
    f = plt.figure()
    plt.suptitle('Clock Bias')
    tautitles = [r'$\tau$', r'$\dot{\tau}$']
    for i in range(2):
        plt.subplot(2, 1, i + 1)
        plt.title(tautitles[i])
        plt.plot(truth['t'], truth['tau'][:, i], label='x')
        plt.plot(state['t'], state['tau'][:, i], label=r'\hat{x}')
        # plt.plot(state[:,0], state[:,i], label=r'\hat{x}')
        if i == 0:
            plt.legend()
    return f

def plotImuBias(state, truth, opt):
    f = plt.figure()
    plt.suptitle('Bias')
    imu_titles = [r"$acc_x$", r"$acc_y$", r"$acc_z$",
                  r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]
    for i in range(3):
        for j in range(2):
            plt.subplot(3, 2, i * 2 + j + 1)
            plt.plot(truth['t'], truth['b'][:, j * 3 + i], label='x')
            plt.plot(state['t'], state['b'][:, j * 3 + i], label=r'\hat{x}')
            plt.title(imu_titles[j * 3 + i])
        if i == 0:
            plt.legend()
    return f


def fixQuat(x):
    x['x'][:,3:] *= np.sign(x['x'][:,3,None])
    return x



def plotResults(prefix):
    np.set_printoptions(linewidth=150)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    state = fixQuat(np.fromfile(os.path.join(prefix,"State.log"), dtype=StateType))
    truth = fixQuat(np.fromfile(os.path.join(prefix,"Truth.log"), dtype=StateType))
    opt = np.fromfile(os.path.join(prefix,"Opt.log"), dtype=OptType)
    rawGPSRes = np.fromfile(os.path.join(prefix, "RawRes.log"), dtype=RawGNSSResType)

    imu_titles = [r"$acc_x$", r"$acc_y$", r"$acc_z$",
                  r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]
    pw = plotWindow()
    
    xtitles = ['$p_x$', '$p_y$', '$p_z$', '$q_w$', '$q_x$', '$q_y$', '$q_z$']
    vtitles = ['$v_x$', '$v_y$', '$v_z$']


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



    pw.addPlot("IMU Bias", plotImuBias(state, truth, opt))
    pw.addPlot("Clock Bias", plotClockBias(state, truth, opt))

    f = plotRawRes(rawGPSRes)

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
