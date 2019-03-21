import numpy as np
from typedefs import *
from plotWindow import plotWindow
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def norm(v, axis=None):
    return np.sqrt(np.sum(v*v, axis=axis))

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

def plot3DMap(x, true, featPos, trueFeat):
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    plt.suptitle("Feature Positions")

    k = [x['kf'] != -1][0]
    ax.plot(x['x'][:,1],x['x'][:,0], -x['x'][:,2], label=r'$\hat{x}$')
    ax.plot(x['x'][k,1],x['x'][k,0], -x['x'][k,2], '*')
    ax.plot(true['x'][:,1],true['x'][:,0], -true['x'][:,2], label=r'$x$')
    ax.legend()

    for id, arr in featPos.iteritems():
        if id == 't': continue
        ax.plot(arr[:,1], arr[:,0], -arr[:,2], linewidth=0.5, alpha=0.5)
    ax.plot(trueFeat[:,1], trueFeat[:,0], -trueFeat[:,2], 'x')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-2, 4)
    return f



def fixState(x):
    x['x'][:,3:] *= np.sign(x['x'][:,3,None])
    return x



def plotResults(prefix):
    np.set_printoptions(linewidth=150)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    x = fixState(np.fromfile(os.path.join(prefix, "State.log"), dtype=StateType))
    state = fixState(np.fromfile(os.path.join(prefix,"CurrentState.log"), dtype=CurrentStateType))
    truth = fixState(np.fromfile(os.path.join(prefix,"Truth.log"), dtype=CurrentStateType))
    opt = np.fromfile(os.path.join(prefix,"Opt.log"), dtype=OptType)
    rawGPSRes = np.fromfile(os.path.join(prefix, "RawRes.log"), dtype=RawGNSSResType)
    featRes = ReadFeatRes(os.path.join(prefix, "FeatRes.log"))
    featPos = ReadFeat(os.path.join(prefix, "Feat.log"))
    trueFeatPos = np.fromfile(os.path.join(prefix, "TrueFeat.log"), dtype=(np.float64, 3)) - truth['x'][0,0:3]
    truth['x'][:,:3] -= truth['x'][0,0:3]

    imu_titles = [r"$acc_x$", r"$acc_y$", r"$acc_z$",
                  r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]
    pw = plotWindow()
    
    xtitles = ['$p_x$', '$p_y$', '$p_z$', '$q_w$', '$q_x$', '$q_y$', '$q_z$']
    vtitles = ['$v_x$', '$v_y$', '$v_z$']

    pw.addPlot("3D Map",plot3DMap(x, truth, featPos, trueFeatPos), threeD=True)

    f = plt.figure()
    plt.suptitle('Position')
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.title(xtitles[i])
        plt.plot(truth['t'], truth['x'][:,i], label='x')
        plt.plot(state['t'], state['x'][:,i], label=r'$\hat{x}$')
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
        plt.subplot(4, 1, i+1)
        plt.title(xtitles[i])
        plt.plot(truth['t'], truth['v'][:,i], label='x')
        plt.plot(state['t'], state['v'][:,i], label=r'\hat{x}')
        # plt.plot(state[:,0], state[:,i+7], label=r'\hat{x}')
        if i == 0:
            plt.legend()
    plt.subplot(4,1,4)
    plt.plot(state['t'], norm(state['v'], axis=1), label=r'\hat{x}')
    plt.plot(truth['t'], norm(truth['v'], axis=1), label=r'x')
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
