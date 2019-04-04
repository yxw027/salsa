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

def plotClockBias():
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
    pw.addPlot("Clock Bias", f)

def plotImuBias():
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
    pw.addPlot("IMU Bias", f)

def plot3DMap():
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')

    k = [x['kf'] != -1][0]
    ax.plot(state['x'][:,1],state['x'][:,0], -state['x'][:,2], label=r'$\hat{x}$')
    ax.plot(x['x'][k,1],x['x'][k,0], -x['x'][k,2], '*')
    ax.plot(truth['x'][:,1],truth['x'][:,0], -truth['x'][:,2], label=r'$x$')
    ax.legend()

    pw.addPlot("3D", f, True)

def init3DMap():
    pass

def animate3DMap(frame):
    pass

def plotMocapRes():
    f = plt.figure()
    for i in range(2):
        for j in range(3):
            plt.subplot(3,2, i*3+j+1)
            plt.plot(mocapRes['r']['t'].T, mocapRes['r']['res'][:,:,i*3+j].T)
    pw.addPlot("MocapRes", f)


def plotFeatRes(allFeat=False):
    ncols = 8
    nrows = 6
    id = -1
    page = -1
    while id <= np.max(featRes['f']['id']):
        page += 1
        f = plt.figure()
        plt.suptitle('Feat Res')
        for c in range(ncols):
            for r in range(nrows):
                id += 1
                while id not in featRes['f']['id'] or np.sum([featRes['f']['id'] == id]) < 20:
                    id += 1
                    if id > np.max(featRes['f']['id']):
                        break
                plt.subplot(nrows, ncols, c*nrows+r+1)
                # t = featRes['f'][featRes['f']['id'] == 27]['to']['t']
                res = featRes['f'][featRes['f']['id'] == id]['to']['res']
                plt.title((str(id)))
                plt.plot(res[:,:,0], res[:,:,1])
        pw.addPlot("Feat Res" + str(page), f)
        if not allFeat:
            break

def plotFeatDepths():
    f = plt.figure()
    plt.suptitle("Feat Pos")
    p = plt.plot(featPos['t'], featPos['ft']['rho'], alpha=0.7)
    for i in range(len(p)):
        plt.plot(featPos['t'], featPos['ft']['rho_true'][:,i], linestyle='--', color=p[i].get_color(), alpha=0.5)
        slide_idx = np.hstack((np.diff(featPos['ft']['slide_count'][:,i]) > 0, False))
        plt.plot(featPos['t'][slide_idx], featPos['ft']['rho_true'][slide_idx, i], marker='x', linestyle=' ', color=p[i].get_color(), alpha=0.5)
    plt.ylim([0, 2])
    pw.addPlot("Depth", f)

def plotPosition():
    f = plt.figure()
    plt.suptitle('Position')
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.title(xtitles[i])
        plt.plot(truth['t'], truth['x'][:,i], label='x')
        plt.plot(state['t'], state['x'][:,i], label=r'$\hat{x}$')
        if i == 0:
            plt.legend()
    pw.addPlot("Position", f)

def plotAttitude():
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

def fixState(x):
    x['x'][:,3:] *= np.sign(x['x'][:,3,None])
    return x

def plotVelocity():
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
    plt.ylabel("Magnitude")
    plt.plot(state['t'], norm(state['v'], axis=1), label=r'\hat{x}')
    plt.plot(truth['t'], norm(truth['v'], axis=1), label=r'x')
    pw.addPlot("Velocity", f)

def plotGnssRes():
    pass
    # f = plt.figure()
    # for i in range(2):
    #     plt.subplot(2,1,i+1)
    #     plt.plot(GnssRes['r']['t'].T, GnssRes['r']['res'][:,:,i].T)

def plotResults(prefix):
    np.set_printoptions(linewidth=150)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    global x, state, truth, opt, GnssRes, featRes, featPos, cb, xtitles, vtitles, trueFeatPos
    global offset, pw, mocapRes

    offset = 10
    x = fixState(np.fromfile(os.path.join(prefix, "State.log"), dtype=StateType))
    state = fixState(np.fromfile(os.path.join(prefix,"CurrentState.log"), dtype=CurrentStateType))
    truth = fixState(np.fromfile(os.path.join(prefix,"Truth.log"), dtype=CurrentStateType))
    opt = np.fromfile(os.path.join(prefix,"Opt.log"), dtype=OptType)
    GnssRes = np.fromfile(os.path.join(prefix, "RawRes.log"), dtype=GnssResType)
    featRes = np.fromfile(os.path.join(prefix, "FeatRes.log"), dtype=FeatResType)
    featPos = np.fromfile(os.path.join(prefix, "Feat.log"), dtype=FeatType)
    trueFeatPos = np.fromfile(os.path.join(prefix, "TrueFeat.log"), dtype=(np.float64, 3))
    cb = np.fromfile(os.path.join(prefix, "CB.log"), dtype=[('t' ,np.float64), ('cb', np.int32)])
    mocapRes = np.fromfile(os.path.join(prefix, "MocapRes.log"), dtype=MocapResType)
    # trueFeatPos -= truth['x'][0,0:3]
    # truth['x'][:,:3] -= truth['x'][0,0:3]

    imu_titles = [r"$acc_x$", r"$acc_y$", r"$acc_z$",
                  r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]
    pw = plotWindow()

    xtitles = ['$p_x$', '$p_y$', '$p_z$', '$q_w$', '$q_x$', '$q_y$', '$q_z$']
    vtitles = ['$v_x$', '$v_y$', '$v_z$']

    plot3DMap()
    plotPosition()
    plotAttitude()
    plotVelocity()
    plotImuBias()

    if max(GnssRes['size']) > 0:
        plotClockBias()
        plotGnssRes()

    if max(featPos['size']) > 0:
        plotFeatRes()
        plotFeatDepths()

    if max(mocapRes['size']) > 0:
        plotMocapRes()






    pw.show()

if __name__ == '__main__':
    # plotResults("/tmp/Salsa.MocapSimulation")
    plotResults("/tmp/Salsa/")
