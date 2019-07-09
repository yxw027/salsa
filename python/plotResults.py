import numpy as np
from typedefs import *
from plotWindow import plotWindow
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import os
from scipy import interpolate
import scipy

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
        for l, log in enumerate(data):
            plt.plot(log.state['t'], log.state['tau'][:, i], label=log.label)
            plt.plot(log.opt['x']['t'], log.opt['x']['tau'][:,:,i], color=colors[l+1], label=log.label, alpha=0.3)
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
            for k, log in enumerate(data):
                plt.plot(log.opt['x']['t'], log.opt['x']['imu'][:, :, j * 3 + i], color=colors[k+1], label=log.label, alpha=0.3)
            plt.title(imu_titles[j * 3 + i])
        if i == 0:
            plt.legend()
    pw.addPlot("IMU Bias", f)

def plot3DMap():
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    ax.plot(truth['x']['p'][:,1],truth['x']['p'][:,0], -truth['x']['p'][:,2], label=r'$x$')
    for log in data:
        if plotKF:
            k = [log.x['node'] != -1][0]
            ax.plot(log.x['x']['p'][k,1],log.x['x']['p'][k,0], -log.x['x']['p'][k,2], '*')
        ax.plot(log.state['x']['p'][:,1],log.state['x']['p'][:,0], -log.state['x']['p'][:,2], label=log.label)
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
    for sat in np.unique(data[0].satPos['sats']['id']):
        if sat < 0: continue
        idx = data[0].satPos['sats']['id'] == sat
        labels = ["az", "el"]
        for i in range(2):
            plt.subplot(2,1,i+1)
            plt.plot(data[0].satPos['t'][np.sum(idx, axis=1).astype(np.bool)], 180.0/np.pi * data[0].satPos['sats']['azel'][idx][:,i], label=str(sat))
            plt.ylabel(labels[i])
            if i == 0:
                plt.legend()
    azel = data[0].satPos['sats']['azel'][-1]*180.0/np.pi
    dist = np.sqrt(np.sum(np.square(data[0].satPos['sats']['p'][-1]), axis=1))/1000
    ids = data[0].satPos['sats']['id'][-1]
    # for i in range(len(ids)):
    #     print ids[i], ", ", dist[i], ", ", azel[i,:]

    pw.addPlot("AzEl", f)


def plotPRangeRes():
    f = plt.figure()
    for s, sat in enumerate(np.unique(data[0].prangeRes['sats']['id'])):
        if sat < 0: continue
        idx = data[0].prangeRes['sats']['id'] == sat
        for i in range(2):
            plt.subplot(2,1,i+1)
            try:
                plt.plot(data[0].prangeRes['t'][np.sum(idx, axis=1).astype(bool)], data[0].prangeRes['sats']['res'][idx][:,i], label=str(sat))
            except:
                debug = 1
            if i == 0:
                plt.legend()
            if s == 1:
                plotMultipathTime()
    # plt.grid()
    pw.addPlot("PRangeRes", f)
    f = plt.figure()
    for s, sat in enumerate(np.unique(data[0].prangeRes['sats']['id'])):
        if sat < 0: continue
        idx = data[0].prangeRes['sats']['id'] == sat
        for i in range(2):
            plt.subplot(2, 1, i + 1)
            p = plt.plot(data[0].prangeRes['t'][np.sum(idx, axis=1).astype(np.bool)], data[0].prangeRes['sats']['z'][idx][:, i], label='z')
            plt.plot(data[0].prangeRes['t'][np.sum(idx, axis=1).astype(np.bool)], data[0].prangeRes['sats']['zhat'][idx][:, i], '--', color=p[0].get_color(), label='zhat')
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
            for log in data:
                plt.plot(log.mocapRes['r']['t'].T, log.mocapRes['r']['res'][:,:,i*3+j].T, label=log.label)
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
    p = plt.plot(data[0].featPos['t'], data[0].featPos['ft']['rho'], alpha=0.7)
    for i in range(len(p)):
        plt.plot(data[0].featPos['t'], data[0].featPos['ft']['rho_true'][:,i], linestyle='--', color=p[i].get_color(), alpha=0.5)
        slide_idx = np.hstack((np.diff(data[0].featPos['ft']['slide_count'][:,i]) > 0, False))
        plt.plot(data[0].featPos['t'][slide_idx], data[0].featPos['ft']['rho_true'][slide_idx, i], marker='x', linestyle=' ', color=p[i].get_color(), alpha=0.5)
    plt.ylim([0, 2])
    pw.addPlot("Depth", f)

def plotPosition(): 
    f = plt.figure()
    plt.suptitle('Position')
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.title(xtitles[i])
        plt.plot(truth['t'], truth['x']['p'][:,i], label='x')
        for k, log in enumerate(data):
            plt.plot(log.state['t'], log.state['x']['p'][:,i], label=log.label)
            plt.plot(log.opt['x']['t'], log.opt['x']['p'][:,:,i], label=log.label, alpha=0.3, color=colors[k+2])
            if plotKF:
                plt.plot(log.x['t'], log.x['x']['p'][:,i], 'x')
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
        plt.plot(truth['t'], truth['x']['q'][:,i]*np.sign(truth['x']['q'][:,0]), label='x')
        for l, log in enumerate(data):
            plt.plot(log.state['t'], log.state['x']['q'][:,i]*np.sign(log.state['xu']['q'][:,0]), label=log.label)
            plt.plot(log.opt['x']['t'], log.opt['x']['q'][:,:,i]*np.sign(log.opt['x']['q'][:,:,0]), label=log.label, color=colors[l+2], alpha=0.3)
            if plotKF:
                plt.plot(log.x['t'], log.x['x']['q'][:,i]*np.sign(log.x['x']['q'][:,0]), 'x')
        if i == 0:
            plt.legend()
        plotMultipathTime()
    pw.addPlot("Attitude", f)

def plotEuler():
    f = plt.figure()
    plt.suptitle("Euler")
    labels=[r"$\phi$", r"$\theta$", r"$\psi$"]
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.title(labels[i])
        plt.plot(truth['t'], 180.0/np.pi * truth['euler'][:,i], label='x')
        for log in data:
            plt.plot(log.state['t'], 180.0/np.pi * log.state['euler'][:,i], label=log.label)
        if i == 0:
            plt.legend()
        plotMultipathTime()
    pw.addPlot("Euler", f)


def fixState(x):
    x['x'][:,3:] *= np.sign(x['x'][:,3,None])
    return x

def plotImu():
    f = plt.figure()
    plt.suptitle('Imu')
    imu_titles = [r"$acc_x$", r"$acc_y$", r"$acc_z$",
                  r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]
    for i in range(3):
        plt.subplot(4, 2, 2*i+1)
        plt.plot(data[0].Imu['t'], data[0].Imu['acc'][:, i], label=imu_titles[i])
        plt.legend()
        plt.subplot(4, 2, 2*i+2)
        plt.plot(data[0].Imu['t'], data[0].Imu['omega'][:, i], label=imu_titles[i+3])
        plt.legend()
    plt.subplot(4,2,7)
    plt.plot(data[0].Imu['t'], scipy.linalg.norm(data[0].Imu['acc'], axis=1))
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
        for l, log in enumerate(data):
            plt.plot(log.state['t'], log.state['v'][:,i], label=log.label)
            plt.plot(log.opt['x']['t'], log.opt['x']['v'][:,:,i], label=log.label, alpha=0.3, color=colors[l+2])
            if plotKF:
                plt.plot(log.x['t'], log.x['v'][:,i], 'x')
        if i == 0:
            plt.legend()
        plotMultipathTime()
    plt.subplot(4,1,4)
    plt.ylabel("Magnitude")
    plt.plot(truth['t'], norm(truth['v'], axis=1), label=r'x')
    for log in data:
        plt.plot(log.state['t'], norm(log.state['v'], axis=1), label=r'\hat{x}')
    pw.addPlot("Velocity", f)

def getMultipathTime():
    global multipathTime
    switch_on = truth['t'][np.where(truth['multipath'][:-1] < truth['multipath'][1:])[0]]
    switch_off = truth['t'][np.where(truth['multipath'][:-1] > truth['multipath'][1:])[0]]

    if len(switch_on) == 0 and len(switch_off) == 0:
        multipathTime = None

    if switch_on.size > switch_off.size:
        switch_off = np.append(switch_off, np.max(truth['t']))
    elif switch_off.size > switch_on.size:
        switch_on = np.insert(switch_on, 0, np.min(truth['t']))
    multipathTime =np.vstack((switch_on, switch_off)).T

def getDeniedTime():
    global deniedTime
    switch_on = truth['t'][np.where(truth['denied'][:-1] < truth['denied'][1:])[0]]
    switch_off = truth['t'][np.where(truth['denied'][:-1] > truth['denied'][1:])[0]]

    if switch_on.size > switch_off.size:
        switch_off = np.append(switch_off, np.max(truth['t']))
    elif switch_off.size > switch_on.size:
        switch_on = np.insert(switch_on, 0, np.min(truth['t']))
    deniedTime = np.vstack((switch_on, switch_off)).T

def plotMultipathTime():
    for row in multipathTime:
        plt.axvspan(row[0], row[1], alpha=0.2, color='black', label="multipath")
    for row in deniedTime:
        plt.axvspan(row[0], row[1], alpha=0.2, color='red', label="denied")

def plotMultipath():
    nsat = params["num_sat"]
    f = plt.figure()
    plt.suptitle("$\kappa$ estimation")
    legend_entries = list()
    legend_entries.append(plt.Line2D([np.nan, np.nan], [np.nan, np.nan], color=colors[0], label=r'$x$'))
    for i in range(nsat):
        plt.subplot(nsat, 1, i+1)
        plt.plot(truth["t"], truth["mp"][:,i], color=colors[0], label=r'$x$')
        for l, log in enumerate(data):
            plt.plot(log.swParams['p']['t'][:,:,i], log.swParams['p']['s'][:,:,i], linewidth=1, alpha=0.2, color=colors[l+1])
            if i == 0:
                legend_entries.append(plt.Line2D([np.nan, np.nan], [np.nan, np.nan], color=colors[l+1], label=log.label))
        if i == 0:
            plt.legend(handles=legend_entries, ncol=7)
        if i != nsat-1:
            plt.xticks([])
        plt.ylabel(str(data[0].satPos['sats']['id'][-1, i]))
        plt.ylim([-0.05, 1.05])
        # plotMultipathTime()
    pw.addPlot("Multipath", f)

def saveMultipath():
    nsat = truth["mp"][0].size
    f = plt.figure(figsize=[8, 6], dpi=600)
    plt.suptitle("$\kappa$ estimation")
    # cmap = plt.cm.get_cmap('tab10', len(data)+1)
    legend_entries = []
    legend_entries.append(plt.Line2D([np.nan, np.nan], [np.nan, np.nan], color=colors[0], label=r'$x$'))
    nplot = 8
    for i in range(nsat):
        if i >= nplot: continue
        plt.subplot(nplot, 1, i + 1)
        # plt.subplots_adjust(right=0.8)
        plt.plot(truth["t"], truth["mp"][:, i], color=colors[0], label=r'$x$')
        for l, log in enumerate(data):
            if "kappa" not in log.label or "GV" not in log.label: continue
            plt.plot(log.swParams['p']['t'][:, :, i], log.swParams['p']['s'][:, :, i], linewidth=1, alpha=0.2,
                     color=colors[l + 1])
            if i == 0:
                legend_entries.append(
                    plt.Line2D([np.nan, np.nan], [np.nan, np.nan], color=colors[l + 1], label=log.label))
        if i == 0:
            plt.legend(handles=legend_entries, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.04, 1, 0.7))
        if i != nplot - 1:
            plt.xticks([])
        plt.ylim([-0.05, 1.05])
        # plotMultipathTime()

    if savePlots:
        plt.savefig(filePrefix + "multipath.png", dpi=600, facecolor='w', bbox_inches='tight')

def plotPosError():
    f = plt.figure(figsize=[9,6])
    plt.suptitle('Position Error')
    # for i in range(3):
    #     plt.subplot(4, 1, i + 1)
    #     plt.title(xtitles[i])
    #     plt.plot([np.nan, np.nan], [np.nan, np.nan]) # empty plot so the colors match
    #     for log in data:
    #         plt.plot(log.state['t'], np.abs(log.state['x']['p'][:, i] - truth_pos_interp(log.state['t'])[i,:]), label=log.label)
    #     if i == 0:
    #         plt.legend()
    # plt.subplot(4,1,4)
    plt.xlabel('s')
    plt.ylabel('m')
    plt.plot([np.nan, np.nan], [np.nan, np.nan])  # empty plot so the colors match
    for log in data:
        plt.plot(log.state['t'], scipy.linalg.norm(log.state['x']['p'][:, :] - truth_pos_interp(log.state['t'])[:, :].T, axis=1), label=log.label)
    plotMultipathTime()
    plt.legend(ncol=2, framealpha=1)
    if savePlots:
        plt.savefig(filePrefix + "pos_error.png", dpi=600, facecolor='w', bbox_inches='tight') 
    pw.addPlot("Position Error", f)

def plotVelError():
    f = plt.figure(figsize=[9,6])
    plt.suptitle('Velocity Error')
    # for i in range(3):
    #     plt.subplot(3, 1, i + 1)
    #     plt.title(xtitles[i])
    #     plt.plot([np.nan, np.nan], [np.nan, np.nan]) # empty plot so the colors match
    #     for log in data:
    #         plt.plot(log.state['t'], np.abs(log.state['v'][:, i] - truth_vel_interp(log.state['t'])[i,:]), label=log.label)
    #     if i == 0:
    #         plt.legend()
    #     plotMultipathTime()
    plt.xlabel("s")
    plt.ylabel("m/s")
    plt.plot([np.nan, np.nan], [np.nan, np.nan])  # empty plot so the colors match
    for log in data:
        plt.plot(log.state['t'], scipy.linalg.norm(log.state['v'][:, :] - truth_vel_interp(log.state['t'])[:, :].T, axis=1), label=log.label)
    plotMultipathTime()
    plt.legend(ncol=2, framealpha=1)
    if savePlots:
        plt.savefig(filePrefix + "vel_error.png", dpi=600, facecolor='w', bbox_inches='tight')
    pw.addPlot("Velocity Error", f)

def plotPosLla():
    f = plt.figure()
    plt.suptitle("PP LLA")
    for i in range(3):
        plt.subplot(3, 2,2*i+1)
        for log in data:
            if i == 2:
                plt.plot(log.Lla['t'], log.Lla['pp_lla'][:,i], label=log.label+"pp")
                plt.plot(log.state['t'], log.state['lla'][:,i], label=log.label + "est")
            else:
                plt.plot(log.Lla['t'], log.Lla['pp_lla'][:,i]*180.0/np.pi, label=log.label + "pp")
                plt.plot(log.state['t'], log.state['lla'][:,i]*180.0/np.pi, label=log.label)
        if i == 0:
            plt.legend()
    plt.subplot(1,2,2)
    for log in data:
        plt.plot(log.Lla['pp_lla'][:,1]*180/np.pi, log.Lla['pp_lla'][:,0]*180.0/np.pi, label=log.label + "pp")
        plt.plot(log.state['lla'][:,1]*180/np.pi, log.state['lla'][:,0]*180.0/np.pi, label=log.label + "est")
    pw.addPlot("Lla", f);

def plotColoredPosLla():
    f = plt.figure()
    plt.suptitle("Lat Lon Alt")
    cmap = plt.get_cmap("plasma", 8)
    for log in data:
        counts = np.sum(np.greater(log.swParams['p']['s'], 0.5), axis=2)
        print counts.shape
        print log.opt['x']['lla'].shape
        while counts.shape[0] < log.opt['x']['lla'].shape[0]:
            counts = np.vstack((counts, np.zeros((1,20), dtype=np.bool)))
        while counts.shape[0] > log.opt['x']['lla'].shape[0]:
            counts = counts[:-1,:].copy()
        print counts.shape
        print log.opt['x']['lla'].shape

        for i in range(np.max(counts)):
            tmp = np.ones(log.opt['x']['lla'].shape)*np.nan
            tmp[counts > i, :] = log.opt['x']['lla'][counts > i, :]
            print tmp.shape
            plt.plot(tmp[:,:,1]*180/np.pi, tmp[:,:,0]*180.0/np.pi, color=cmap(float(i)/float(np.max(counts))), alpha=0.3)
            plt.plot(np.nan,np.nan, color=cmap(float(i)/float(np.max(counts))), label=str(i))
    plt.axis("equal")
    plt.legend()
    pw.addPlot("Colored Lla", f)

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
        setattr(self, "Lla", np.fromfile(os.path.join(prefix, "PP_LLA.log"), dtype=LlaType))
        setattr(self, "label", open(os.path.join(prefix, "label.txt"), "r").read().splitlines()[0])
        self.label.replace(r"//", r"/")


def interpolateTruth():
    global truth_pos_interp, truth_vel_interp
    truth_pos_interp = interpolate.interp1d(truth['t'], truth['x']['p'].T)
    truth_vel_interp = interpolate.interp1d(truth['t'], truth['v'].T)


def plotResults(directory, plotKeyframes=True, saveFig=False, prefix=""):
    np.set_printoptions(linewidth=150)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    global data, truth, pw, xtitles, imu_titles, vtitles, plotKF, colors
    global savePlots, filePrefix, params
    savePlots = saveFig
    filePrefix = prefix
    params = yaml.load(open("../params/salsa.yaml"))
    xtitles = ['$p_x$', '$p_y$', '$p_z$', '$q_w$', '$q_x$', '$q_y$', '$q_z$']
    vtitles = ['$v_x$', '$v_y$', '$v_z$']
    imu_titles = [r"$acc_x$", r"$acc_y$", r"$acc_z$",
                  r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plotKF = plotKeyframes

    subdirs = [os.path.join(directory, o) for o in os.listdir(directory) if os.path.isdir(os.path.join(directory,o))]
    truth = np.fromfile(os.path.join(directory,"Truth.log"), dtype=SimStateType)
    # trueFeatPos = np.fromfile(os.path.join(prefix, "TrueFeat.log"), dtype=(np.float64, 3))

    data = [Log(subdir) for subdir in subdirs]

    # interpolateTruth()
    getMultipathTime()
    getDeniedTime()

    pw = plotWindow()

    plot3DMap()
    plotPosition()
    plotAttitude()
    plotEuler()
    plotVelocity()
    plotPosLla()
    plotColoredPosLla()
    # plotPosError()
    # plotVelError()
    plotImuBias()
    plotImu()
    plotXe2n()
    plotXb2c()
    #
    if len(data[0].prangeRes) > 0 and max(data[0].prangeRes['size']) > 0:
    #     plotPRangeRes()
    #     plotClockBias()
        plotMultipath()
    #     saveMultipath()
    #     plotAzel()  
    # # #
    # if len(data[0].featPos) > 0 and max(data[0].featPos['size']) > 0:
    #     # plotFeatRes()
    #     plotFeatDepths()
    # #
    # if len(data[0].mocapRes) > 0 and max(data[0].mocapRes['size']) > 0:
    #     plotMocapRes()
    #
    # if len(satPos) > 0 and max(satPos['size']) > 0:
    #     plotSatPos()
    pw.show()

if __name__ == '__main__':
    # plotResults("/tmp/Salsa.MocapSimulation")
    # plotResults("/tmp/Salsa/MocapFeatHardware", False)
    plotResults("/tmp/Salsa/GNSSHardware", False)

    # plotResults("/tmp/Salsa/FeatSimulation/")

