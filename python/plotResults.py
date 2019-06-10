import numpy as np
from typedefs import *
from plotWindow import plotWindow
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import os

sim_params = yaml.load(open("../params/sim_params.yaml"))

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
        for log in data:
            plt.plot(log.state['t'], log.state['tau'][:, i], label=log.label)
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
            for log in data:
                plt.plot(log.state['t'], log.state['b'][:, j * 3 + i], label=log.label)
            plt.title(imu_titles[j * 3 + i])
        if i == 0:
            plt.legend()
    pw.addPlot("IMU Bias", f)

def plot3DMap():
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    for log in data:
        k = [log.x['node'] != -1][0]
        x = log.x
        state = log.state
        ax.plot(x['x']['p'][k,1],x['x']['p'][k,0], -x['x']['p'][k,2], '*')
        ax.plot(state['x']['p'][:,1],state['x']['p'][:,0], -state['x']['p'][:,2], label=log.label)
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
    # for i in range(len(ids)):
    #     print ids[i], ", ", dist[i], ", ", azel[i,:]

    pw.addPlot("AzEl", f)


def plotPRangeRes():
    f = plt.figure()
    for s, sat in enumerate(np.unique(prangeRes['sats']['id'])):
        if sat < 0: continue
        idx = prangeRes['sats']['id'] == sat
        for i in range(2):
            plt.subplot(2,1,i+1)
            try:
                plt.plot(prangeRes['t'][np.sum(idx, axis=1).astype(bool)], prangeRes['sats']['res'][idx][:,i], label=str(sat))
            except:
                debug = 1
            if i == 0:
                plt.legend()
            if s == 1:
                plotMultipathTime()
    # plt.grid()
    pw.addPlot("PRangeRes", f)
    f = plt.figure()
    for s, sat in enumerate(np.unique(prangeRes['sats']['id'])):
        if sat < 0: continue
        idx = prangeRes['sats']['id'] == sat
        for i in range(2):
            plt.subplot(2, 1, i + 1)
            p = plt.plot(prangeRes['t'][np.sum(idx, axis=1).astype(np.bool)], prangeRes['sats']['z'][idx][:, i], label='z')
            plt.plot(prangeRes['t'][np.sum(idx, axis=1).astype(np.bool)], prangeRes['sats']['zhat'][idx][:, i], '--', color=p[0].get_color(), label='zhat')
            if s == 1:
                plotMultipathTime()
    plt.legend()
    # plt.grid()
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
        for log in data:
            state = log.state
            x = log.x;
            plt.plot(state['t'], state['x']['p'][:,i], label=log.label)
            plt.plot(x['t'], x['x']['p'][:,i], 'x')
        if i == 0:
            plt.legend()
        plotMultipathTime()
    pw.addPlot("Position", f)

def plotAttitude():
    f = plt.figure()
    plt.suptitle('Attitude')
    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.title(xtitles[i+3])
        plt.plot(truth['t'], truth['x']['q'][:,i], label='x')
        for log in data:
            state = log.state
            x = log.x
            plt.plot(state['t'], state['x']['q'][:,i]*np.sign(state['x']['q'][:,0]), label=log.label)
            plt.plot(x['t'], x['x']['q'][:,i]*np.sign(x['x']['q'][:,0]), 'x')
        if i == 0:
            plt.legend()
        plotMultipathTime()
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
        plt.plot(data[0].Imu['t'], data[0].Imu['acc'][:, i], label=imu_titles[i])
        plt.legend()
        plt.subplot(3, 2, 2*i+2)
        plt.plot(data[0].Imu['t'], data[0].Imu['omega'][:, i], label=imu_titles[i+3])
        plt.legend()
    pw.addPlot("IMU", f)

def plotXe2n():
    f = plt.figure()
    # plt.suptitle(r"$T_{e}^{n}$")
    plt.suptitle(r"$T_{e}^{n}$")
    # titles = ["p_{x}", "p_{y}", "p_{z}", "~", "q_{w}", "q_{x}", "q_{y}", "q_{z}"]
    titles = ["x", "y", "z", "~", "w", "x", "y", "z"]
    for i in range(4):
        if i < 3:
            plt.subplot(4,2,2*i+1)
            plt.ylabel(r"$p_{"+titles[i]+"}$")
            plt.plot(truth['t'], truth['x_e2n']['p'][:,i], label="x")
            for log in data:
                plt.plot(np.nanmax(log.opt['x']['t'], axis=1), log.opt['x_e2n']['p'][:,i], label=log.label)
            plt.legend()
        plt.subplot(4,2, 2*i+2)
        plt.plot(truth['t'], truth['x_e2n']['q'][:, i], label="x")
        plt.ylabel("$q_{" + titles[i] + "}$")
        for log in data:
            plt.plot(np.nanmax(log.opt['x']['t'], axis=1), log.opt['x_e2n']['q'][:, i], label=log.label)
        plt.legend()
        plotMultipathTime()
    pw.addPlot("X_e2n", f)

def plotXb2c():
    f = plt.figure()
    # plt.suptitle(r"$T_{e}^{n}$")
    plt.suptitle(r"$T_{b}^{c}$")
    # titles = ["p_{x}", "p_{y}", "p_{z}", "~", "q_{w}", "q_{x}", "q_{y}", "q_{z}"]
    titles = ["x", "y", "z", "~", "w", "x", "y", "z"]
    for i in range(4):

        if i < 3:
            plt.subplot(4,2,2*i+1)
            plt.ylabel(r"$p_{"+titles[i]+"}$")
            plt.plot(truth['t'], truth['x_b2c']['p'][:,i], label="x")
            for log in data:
                plt.plot(np.nanmax(log.opt['x']['t'], axis=1), log.opt['x_b2c']['p'][:,i], label=log.label)
            plt.legend()
        plt.subplot(4,2, 2*i+2)
        plt.plot(truth['t'], truth['x_b2c']['q'][:, i], label="x")
        plt.ylabel(r"$q_{" + titles[i] + "}$")
        for log in data:
            plt.plot(np.nanmax(log.opt['x']['t'], axis=1), log.opt['x_b2c']['q'][:, i], label=log.label)
        plt.legend()
    pw.addPlot("X_b2c", f)


def plotVelocity():
    f = plt.figure()
    plt.suptitle('Velocity')
    for i in range(3):
        plt.subplot(4, 1, i+1)
        plt.title(xtitles[i])
        plt.plot(truth['t'], truth['v'][:,i], label='x')
        for log in data:
            state = log.state
            x = log.x
            plt.plot(state['t'], state['v'][:,i], label=log.label)
            plt.plot(x['t'], x['v'][:,i], 'x')
        if i == 0:
            plt.legend()
        plotMultipathTime()
    plt.subplot(4,1,4)
    plt.ylabel("Magnitude")
    plt.plot(truth['t'], norm(truth['v'], axis=1), label=r'x')
    plt.plot(x['t'], norm(x['v'], axis=1), label=r'\hat{x}')
    pw.addPlot("Velocity", f)

def getMultipathTime():
    global multipathTime
    switch_on = truth['t'][np.where(truth['multipath'][:-1] < truth['multipath'][1:])[0]]
    switch_off = truth['t'][np.where(truth['multipath'][:-1] > truth['multipath'][1:])[0]]

    if switch_on.size > switch_off.size:
        switch_off = np.append(switch_off, np.max(truth['t']))
    multipathTime =np.vstack((switch_on, switch_off)).T

def getDeniedTime():
    global deniedTime
    switch_on = truth['t'][np.where(truth['denied'][:-1] < truth['denied'][1:])[0]]
    switch_off = truth['t'][np.where(truth['denied'][:-1] > truth['denied'][1:])[0]]

    if switch_on.size > switch_off.size:
        switch_off = np.append(switch_off, np.max(truth['t']))
    deniedTime = np.vstack((switch_on, switch_off)).T

def plotMultipathTime():
    for row in multipathTime:
        plt.axvspan(row[0], row[1], alpha=0.2, color='black')
    for row in deniedTime:
        plt.axvspan(row[0], row[1], alpha=0.4, color='red')

def plotMultipath():
    nsat = truth["mp"][0].size
    f = plt.figure()
    cmap = plt.cm.get_cmap('Paired', len(data)+1)
    for i in range(nsat):
        plt.subplot(nsat, 1, i+1)
        plt.plot(truth["t"], truth["mp"][:,i], color=cmap(0), label=r'$x$')
        for l, log in enumerate(data):
            plt.plot(log.swParams['p']['t'][:,:,i], log.swParams['p']['s'][:,:,i], alpha=0.3, color=cmap(l+1), label=log.label)
        if i == 0:
            plt.legend()
        plt.ylim([-0.05, 1.05])
    pw.addPlot("Multipath", f)

class Log:
    def __init__(self, prefix):
        self.prefix = prefix
        self.load(prefix)

    def load(self, prefix):
        setattr(self, "x", np.fromfile(os.path.join(prefix, "State.log"), dtype=StateType))
        setattr(self, "state", np.fromfile(os.path.join(prefix,"CurrentState.log"), dtype=CurrentStateType))
        setattr(self, "opt", np.fromfile(os.path.join(prefix,"Opt.log"), dtype=OptType))
        setattr(self, "GnssRes", np.fromfile(os.path.join(prefix, "RawRes.log"), dtype=GnssResType))
        setattr(self, "featRes", np.fromfile(os.path.join(prefix, "FeatRes.log"), dtype=FeatResType))
        setattr(self, "featPos", np.fromfile(os.path.join(prefix, "Feat.log"), dtype=FeatType))
        setattr(self, "cb", np.fromfile(os.path.join(prefix, "CB.log"), dtype=[('t' ,np.float64), ('cb', np.int32)]))
        setattr(self, "mocapRes", np.fromfile(os.path.join(prefix, "MocapRes.log"), dtype=MocapResType))
        setattr(self, "satPos", np.fromfile(os.path.join(prefix, "SatPos.log"), dtype=SatPosType))
        setattr(self, "prangeRes", np.fromfile(os.path.join(prefix, "PRangeRes.log"), dtype=PRangeResType))
        setattr(self, "Imu", np.fromfile(os.path.join(prefix, "Imu.log"), dtype=ImuType))
        setattr(self, "swParams", np.fromfile(os.path.join(prefix, "SwParams.log"), dtype=SwParamsType))
        setattr(self, "label", open(os.path.join(prefix, "label.txt"), "r").read().splitlines()[0])
        self.label.replace(r"//", r"/")


def plotResults(directory):
    np.set_printoptions(linewidth=150)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    global data, truth, pw, xtitles, imu_titles, vtitles
    xtitles = ['$p_x$', '$p_y$', '$p_z$', '$q_w$', '$q_x$', '$q_y$', '$q_z$']
    vtitles = ['$v_x$', '$v_y$', '$v_z$']
    imu_titles = [r"$acc_x$", r"$acc_y$", r"$acc_z$",
                  r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]

    subdirs = [os.path.join(directory, o) for o in os.listdir(directory) if os.path.isdir(os.path.join(directory,o))]
    truth = np.fromfile(os.path.join(directory,"Truth.log"), dtype=SimStateType)
    # trueFeatPos = np.fromfile(os.path.join(prefix, "TrueFeat.log"), dtype=(np.float64, 3))

    data = [Log(subdir) for subdir in subdirs]

    getMultipathTime()
    getDeniedTime()

    pw = plotWindow()

    plot3DMap()
    plotPosition()
    plotAttitude()
    plotVelocity()
    plotImuBias()
    plotImu()
    plotXe2n()
    # plotXb2c()
    #
    if len(data[0].prangeRes) > 0 and max(data[0].prangeRes['size']) > 0:
        # plotPRangeRes()
        plotClockBias()
        plotMultipath()
        # plotAzel()
    #
    # if len(featPos) > 0 and max(featPos['size']) > 0:
    #     plotFeatRes()
    #     plotFeatDepths()
    #
    # if len(mocapRes) > 0 and max(mocapRes['size']) > 0:
    #     plotMocapRes()
    #
    # # if len(satPos) > 0 and max(satPos['size']) > 0:
    # #     plotSatPos()
    pw.show()

if __name__ == '__main__':
    # plotResults("/tmp/Salsa.MocapSimulation")
    # plotResults("/tmp/Salsa/")
    plotResults("/tmp/Salsa/FeatSimulation/")
