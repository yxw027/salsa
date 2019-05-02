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
    ax.set_aspect('equal')

    k = [x['node'] != -1][0]
    ax.plot(state['x']['p'][:,1],state['x']['p'][:,0], -state['x']['p'][:,2], label=r'$\hat{x}$')
    ax.plot(x['x']['p'][k,1],x['x']['p'][k,0], -x['x']['p'][k,2], '*')
    ax.plot(truth['x']['p'][:,1],truth['x']['p'][:,0], -truth['x']['p'][:,2], label=r'$x$')
    ax.legend()
    plt.grid()

    pw.addPlot("3D", f, True)

def plotSatPos():
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    for sat in np.unique(satPos['sats']['id']):
        if sat < 0: continue
        idx = satPos['sats']['id'] == sat
        ax.plot(satPos['sats'][idx]['p'][:,0], satPos['sats']['p'][idx][:,1], satPos['sats']['p'][idx][:,2], label=str(sat))
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 6371e3 * np.outer(np.cos(u), np.sin(v))
    y = 6371e3 * np.outer(np.sin(u), np.sin(v))
    z = 6371e3 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.7)
    ax.plot([-1798780.451], [-4532177.657], [4099857.983], 'x')
    ax.legend()
    plt.grid()
    pw.addPlot("SatPos", f, True)

def plotAzel():
    f = plt.figure()
    for sat in np.unique(satPos['sats']['id']):
        if sat < 0: continue
        idx = satPos['sats']['id'] == sat
        labels = ["az", "el"]
        for i in range(2):
            plt.subplot(2,1,i+1)
            plt.plot(satPos['t'][np.sum(idx, axis=1).astype(np.bool)], 180.0/np.pi * satPos['sats']['azel'][idx][:,i], label=str(sat))
            plt.ylabel(labels[i])
            if i == 0:
                plt.legend()
    azel = satPos['sats']['azel'][-1]*180.0/np.pi
    dist = np.sqrt(np.sum(np.square(satPos['sats']['p'][-1]), axis=1))/1000
    ids = satPos['sats']['id'][-1]
    for i in range(len(ids)):
        print ids[i], ", ", dist[i], ", ", azel[i,:]

    pw.addPlot("AzEl", f)


def plotPRangeRes():
    f = plt.figure()
    for sat in np.unique(prangeRes['sats']['id']):
        if sat < 0: continue
        idx = prangeRes['sats']['id'] == sat
        for i in range(2):
            plt.subplot(2,1,i+1)
            plt.plot(prangeRes['t'][np.sum(idx, axis=1).astype(np.bool)], prangeRes['sats']['res'][idx][:,i], label=str(sat))
            if i == 0:
                plt.legend()
    plt.grid()
    pw.addPlot("PRangeRes", f)
    f = plt.figure()
    for sat in np.unique(prangeRes['sats']['id']):
        if sat < 0: continue
        idx = prangeRes['sats']['id'] == sat
        for i in range(2):
            plt.subplot(2, 1, i + 1)
            p = plt.plot(prangeRes['t'][np.sum(idx, axis=1).astype(np.bool)], prangeRes['sats']['z'][idx][:, i], label='z')
            plt.plot(prangeRes['t'][np.sum(idx, axis=1).astype(np.bool)], prangeRes['sats']['zhat'][idx][:, i], '--', color=p[0].get_color(), label='zhat')
    plt.legend()
    plt.grid()
    pw.addPlot("PRangeResDebug", f)


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
        plt.plot(truth['t'], truth['x']['p'][:,i], label='x')
        plt.plot(x['t'], x['x']['p'][:,i], label=r'$\hat{x}$')
        if i == 0:
            plt.legend()
    pw.addPlot("Position", f)

def plotAttitude():
    f = plt.figure()
    plt.suptitle('Attitude')
    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.title(xtitles[i+3])
        plt.plot(truth['t'], truth['x']['q'][:,i], label='x')
        plt.plot(x['t'], x['x']['q'][:,i], label=r'\hat{x}')
        # plt.plot(state[:,0], state[:,i+3], label=r'\hat{x}')
        if i == 0:
            plt.legend()
    pw.addPlot("Attitude", f)

def fixState(x):
    x['x'][:,3:] *= np.sign(x['x'][:,3,None])
    return x

def plotImu():
    f = plt.figure()
    plt.suptitle('Imu')
    imu_titles = [r"$acc_x$", r"$acc_y$", r"$acc_z$",
                  r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]
    for i in range(3):
        plt.subplot(3, 2, 2*i+1)
        plt.plot(Imu['t'], Imu['acc'][:, i], label=imu_titles[i])
        plt.legend()
        plt.subplot(3, 2, 2*i+2)
        plt.plot(Imu['t'], Imu['omega'][:, i], label=imu_titles[i+3])
        plt.legend()
    pw.addPlot("IMU", f)

def plotXe2n():
    f = plt.figure()
    # plt.suptitle(r"$T_{e}^{n}$")
    plt.suptitle(r"$T_{e}^{n}$")
    # titles = ["p_{x}", "p_{y}", "p_{z}", "~", "q_{w}", "q_{x}", "q_{y}", "q_{z}"]
    titles = ["x", "y", "z", "~", "w", "x", "y", "z"]
    for i in range(4):
        t = np.nanmax(opt['x']['t'], axis=1)
        if i < 3:
            plt.subplot(4,2,2*i+1)
            plt.plot(t, opt['x_e2n']['p'][:,i], label=r"$\hat{p}_{"+titles[i]+"}$")
            plt.plot(truth['t'], truth['x_e2n']['p'][:,i], label=r"$p_{"+titles[i]+"}$")
            plt.legend()
        plt.subplot(4,2, 2*i+2)
        plt.plot(t, opt['x_e2n']['q'][:, i], label=r"$\hat{q}_{" + titles[i] + "}$")
        plt.plot(truth['t'], truth['x_e2n']['q'][:, i], label=r"$q_{" + titles[i] + "}$")
        plt.legend()
    pw.addPlot("X_e2n", f)

def plotXb2c():
    f = plt.figure()
    # plt.suptitle(r"$T_{e}^{n}$")
    plt.suptitle(r"$T_{b}^{c}$")
    # titles = ["p_{x}", "p_{y}", "p_{z}", "~", "q_{w}", "q_{x}", "q_{y}", "q_{z}"]
    titles = ["x", "y", "z", "~", "w", "x", "y", "z"]
    for i in range(4):
        t = np.nanmax(opt['x']['t'], axis=1)
        if i < 3:
            plt.subplot(4,2,2*i+1)
            plt.plot(t, opt['x_b2c']['p'][:,i], label=r"$\hat{p}_{"+titles[i]+"}$")
            plt.plot(truth['t'], truth['x_b2c']['p'][:,i], label=r"$p_{"+titles[i]+"}$")
            plt.legend()
        plt.subplot(4,2, 2*i+2)
        plt.plot(t, opt['x_b2c']['q'][:, i], label=r"$\hat{q}_{" + titles[i] + "}$")
        plt.plot(truth['t'], truth['x_b2c']['q'][:, i], label=r"$q_{" + titles[i] + "}$")
        plt.legend()
    pw.addPlot("X_b2c", f)


def plotVelocity():
    f = plt.figure()
    plt.suptitle('Velocity')
    for i in range(3):
        plt.subplot(4, 1, i+1)
        plt.title(xtitles[i])
        plt.plot(truth['t'], truth['v'][:,i], label='x')
        plt.plot(x['t'], x['v'][:,i], label=r'\hat{x}')
        # plt.plot(state[:,0], state[:,i+7], label=r'\hat{x}')
        if i == 0:
            plt.legend()
    plt.subplot(4,1,4)
    plt.ylabel("Magnitude")
    plt.plot(truth['t'], norm(truth['v'], axis=1), label=r'x')
    plt.plot(x['t'], norm(x['v'], axis=1), label=r'\hat{x}')
    pw.addPlot("Velocity", f)

def plotResults(prefix):
    np.set_printoptions(linewidth=150)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    global x, state, truth, opt, GnssRes, featRes, featPos, cb, xtitles, vtitles, trueFeatPos
    global offset, pw, mocapRes, satPos, prangeRes, Imu

    offset = 10
    x = np.fromfile(os.path.join(prefix, "State.log"), dtype=StateType)
    state = np.fromfile(os.path.join(prefix,"CurrentState.log"), dtype=CurrentStateType)
    truth = np.fromfile(os.path.join(prefix,"Truth.log"), dtype=SimStateType)
    opt = np.fromfile(os.path.join(prefix,"Opt.log"), dtype=OptType)
    GnssRes = np.fromfile(os.path.join(prefix, "RawRes.log"), dtype=GnssResType)
    featRes = np.fromfile(os.path.join(prefix, "FeatRes.log"), dtype=FeatResType)
    featPos = np.fromfile(os.path.join(prefix, "Feat.log"), dtype=FeatType)
    # trueFeatPos = np.fromfile(os.path.join(prefix, "TrueFeat.log"), dtype=(np.float64, 3))
    cb = np.fromfile(os.path.join(prefix, "CB.log"), dtype=[('t' ,np.float64), ('cb', np.int32)])
    mocapRes = np.fromfile(os.path.join(prefix, "MocapRes.log"), dtype=MocapResType)
    satPos = np.fromfile(os.path.join(prefix, "SatPos.log"), dtype=SatPosType)
    prangeRes = np.fromfile(os.path.join(prefix, "PRangeRes.log"), dtype=PRangeResType)
    Imu = np.fromfile(os.path.join(prefix, "Imu.log"), dtype=ImuType)
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
    plotImu()
    plotXe2n()
    plotXb2c()

    if len(prangeRes) > 0 and max(prangeRes['size']) > 0:
        plotPRangeRes()
        plotClockBias()
        plotAzel()

    if len(featPos) > 0 and max(featPos['size']) > 0:
        plotFeatRes()
        plotFeatDepths()

    if len(mocapRes) > 0 and max(mocapRes['size']) > 0:
        plotMocapRes()

    if len(satPos) > 0 and max(satPos['size']) > 0:
        plotSatPos()








    pw.show()

if __name__ == '__main__':
    # plotResults("/tmp/Salsa.MocapSimulation")
    # plotResults("/tmp/Salsa/")
    plotResults("/tmp/Salsa/FeatSimulation/")
