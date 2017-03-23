"""
signale.place_cells
===================

A module for place cells.
"""
__author__ = ("KT", "Moritz Dittmeyer", "Olivia Haas")
__version__ = "7.2.2, November 2014"


# python modules
import os
import sys
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)-2s: %(message)s')
logger = logging.getLogger('signale.place_cells')

# add additional custom paths
extraPaths = [os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../scripts')]
for p in extraPaths:
    if not sys.path.count(p):
        sys.path.insert(1, p)

# other modules
import numpy
import inspect
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pl
# pl.ioff()
import matplotlib.path as mpl_path
import NeuroTools.signals as NTsig
import scipy.optimize
import scipy.special
import copy


# custom made modules
import custom_plot

# package modules
from signals import signal
import io
import tools
import spikes


# colors
transred = '#FFA1A1'
FR_binsize = 0.03  # 3 cm orig 2cm
FR_bin_smooth = 3  # orig. 3 bins wide


###################################################### FUNCTIONS

def phasePlot(spikes, phases, places, traj, mutterTraj=True, gc=False, GetPlaceField=False, wHists=True, wTraj=True, fig=None,
              ax=None, labelx=True, labely=True, limx=None, limy=None, color=[0, 0, 0], labelsize=None, title=''):
    """ Phase plot.
    """

    if not fig:
        fig = pl.figure(figsize=(4.5, 3.5))
    if not ax:
        pos = [.25, .21, .65, .625]
        ax = fig.add_axes(pos)
    pos = ax.get_position().bounds

    color = numpy.array(color)

    if places.size:
        # print places[:, 1]
        ax.plot(places[:, 0], phases, 'o', markerfacecolor=[2./3., 2./3., 2./3.], markeredgecolor=color) #color=color, fillstyle='none')
        ax.plot(places[:, 0], phases+360, 'o', markerfacecolor=[2./3., 2./3., 2./3.], markeredgecolor=color) #, fillstyle='none')
    else:
        print 'spike places and phases array is empty!'


    if limx is None or not limx:
        #limx = [0, traj.trackLength]
        limx = traj.xlim
    elif limx is True and places.size:
        dx = numpy.nanmax(places[:, 0])-numpy.nanmin(places[:, 0])
        dx *= .2
        limx = [max(0, numpy.nanmin(places[:, 0])-dx), numpy.nanmin(traj.trackLength, numpy.nanmax(places[:, 0])+dx)]

    if limy is None:
        limy = traj.ylim

    xticks = numpy.linspace(numpy.round(limx[0], 1), numpy.round(limx[1], 1), 3)

    custom_plot.huebschMachen(ax)
    if labelx:
        ax.set_xlabel('Position ('+traj.spaceUnit+')')

    ax.set_xticks(xticks)
    if labelsize:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(labelsize)
    
    if labely:
        ax.set_ylabel('Spike phase (deg)')
    ax.set_yticks(range(0, 720+1, 180))
    if labelsize:
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(labelsize)
    #else:
    #    ax.set_yticks(range(0, 720+1, 180))
    #    ax.set_yticklabels([])
    ax.set_xlim(limx)
    ax.set_ylim(0, 720)

    if wHists:
        histCol = custom_plot.grau
        if color.any():
            histCol = color

        posAbove = list(pos)
        posAbove[1] = pos[1]+pos[3]
        posAbove[3] = pos[3]*.08
        posRight = list(pos)
        posRight[0] = pos[0]+pos[2]
        posRight[2] = posAbove[3]
        axAbove = fig.add_axes(posAbove)
        if places.size:
            axAbove.hist(places[:, 0], numpy.arange(limx[0], limx[1], .03),
                         facecolor=histCol, edgecolor=custom_plot.grau*2./3, linewidth=0.5)
        custom_plot.allOff(axAbove)
        axAbove.set_xlim(limx)

        axRight = fig.add_axes(posRight)
        if phases.size:
            axRight.hist(numpy.append(phases, phases+360), numpy.arange(0, 720, 11.25),
                         orientation='horizontal', facecolor=histCol, edgecolor=custom_plot.grau*2./3, linewidth=0.5)
        axRight.set_ylim(0, 720)
        custom_plot.allOff(axRight)
        ax = axAbove

    traj_ylim = limy+numpy.diff(limy)*.1*[-1, 1]

    # plot trajectory with spikes - in red - above phase plots
    if wTraj:
        posUppermost = list(posAbove)
        posUppermost[1] += posUppermost[3]
        posUppermost[3] = pos[3]*.11
        axUppermost = fig.add_axes(posUppermost)

        if wTraj == 'full':
            axUppermost.plot(traj.places[:, 0], traj.places[:, 1], linewidth=1, color=numpy.ones(3)*.5) # original/full trajectory
            if places.size:
                axUppermost.plot(places[:, 0], places[:, 1], '.', color=transred)
        elif wTraj == 'noSpeck':
            axUppermost.plot(traj.places[:, 0], traj.places[:, 1], linewidth=1, color=numpy.ones(3)*.5) # original/full trajectory
            if places.size:
                axUppermost.plot(places[:, 0], places[:, 1], 'o', color='r', markeredgecolor='r')
        elif places.size:
            axUppermost.plot(places[:, 0], places[:, 1], 'o', color='r', markeredgecolor='r')

        axUppermost.set_xlim(limx)
        axUppermost.set_ylim(traj_ylim)
        custom_plot.allOff(axUppermost)

        # if the place field heatmap plot is suppose to be plotted:
        if GetPlaceField:
            pos_ganz_oben = list(posUppermost)
            pos_ganz_oben[1] += pos_ganz_oben[3]+.01
            pos_ganz_oben[3] = pos_ganz_oben[3]
            ax_ganz_oben = fig.add_axes(pos_ganz_oben)

            ax_ganz_oben.set_xlabel('', visible=False)
            ax_ganz_oben.set_ylabel('', visible=False)

            ax_ganz_oben.set_xlim(limx)
            ax_ganz_oben.set_ylim(traj_ylim)
            custom_plot.allOff(ax_ganz_oben)

            pos_ganz_ganz_oben = list(pos_ganz_oben)

        else:
            ax_ganz_oben = 0
            pos_ganz_ganz_oben = list(posUppermost)

        pos_ganz_ganz_oben[1] += pos_ganz_ganz_oben[3]+.01
        pos_ganz_ganz_oben[3] = pos_ganz_ganz_oben[3]*3
        ax_ganz_ganz_oben = fig.add_axes(pos_ganz_ganz_oben)

        # numpy.set_printoptions(threshold=numpy.inf)
        # print 'SPIKE times: ', spikes
        if wTraj == 'full':
            ax_ganz_ganz_oben.plot(traj.times, traj.places[:, 0], linewidth=1, color=numpy.ones(3)*.5)             # original/full trajectory
            if places.size:
                ax_ganz_ganz_oben.plot(spikes, places[:, 0], '.', color=transred)  # all spiketimes
        elif wTraj == 'noSpeck':
            ax_ganz_ganz_oben.plot(traj.times, traj.places[:, 0], linewidth=1, color=numpy.ones(3)*.5)             # original/full trajectory
            if places.size:
                ax_ganz_ganz_oben.plot(spikes, places[:, 0], 'r.')                  # only running-spikes
        elif places.size:
            ax_ganz_ganz_oben.plot(spikes, places[:, 0], 'r.')                  # only running-spikes
        custom_plot.allOff(ax_ganz_ganz_oben)
        #problem: traj.times[0], traj.times[-1] waeren nur die des ersten runs des gain changes!
        if mutterTraj is False:
            ax_ganz_ganz_oben.set_xlim(traj.times[0], traj.times[-1])
        else:
            ax_ganz_ganz_oben.set_xlim(mutterTraj.times[0], mutterTraj.times[-1])
        ax_ganz_ganz_oben.set_ylim(limx)

        #if gc is False:
        duration = traj.t_stop-traj.t_start
        if not title == '':
            fig.text(pos_ganz_ganz_oben[0]+pos_ganz_ganz_oben[2]+.01, pos_ganz_ganz_oben[1]+.01,
                     str(numpy.round(duration/60, 1))+' min', fontsize=10)
        ax = ax_ganz_ganz_oben

    ax.set_title(title, fontsize=10)

    if wTraj:
        return fig, ax_ganz_ganz_oben, ax_ganz_oben, axUppermost
    else:
        return fig, ax, None, None


def rcc(lindat0, phidat0, abound=[-2., 2.], display=False, PF=False):
    """
    Calculate slope and correlation value on circular data (phidat).

    Parameters
    ----------
    lindat0 : list
    phidat0 : list
        Phases. Expected in radiants.
    abound : list, optional
        Bounds of slope. Default [-2., 2.].
    display : bool, optional
        Display figure with fitting parameters. Default False.

    Returns
    -------
    results_dict : dict
        Dictionary containing the keys: 'rho', 'p', 'R', 'aopt', 'phi0'
    """

    # copy input array because we will do changes to them below
    lindat = numpy.copy(lindat0)   # lindat are the spike places
    phidat = numpy.copy(phidat0)   # phidat are the spike phases

    phidat = numpy.float_(phidat)   # make phidat to floats
    lindat = numpy.float_(lindat)   # make lindat to floats

    if phidat.max > 2*numpy.pi:
        print 'phases, i.e data in phidat, seem to be degrees. Converting to radiants.'
        phidat /= 180./numpy.pi

    global aopts, Rs
    Rs = []
    aopts = []

    print 'T-----------------------------------------------------------------------------------------------------'

    if PF:
        print 'TAKING GIVEN PLACE FIELD BOUNDARIES! ', PF
        if numpy.nanmin(lindat0) < min(PF) or numpy.nanmax(lindat0) > max(PF):
            print 'spike_places min max', numpy.nanmin(lindat0), numpy.nanmax(lindat0), ' not within PF boundaries of '+\
                                                                                        min(PF), max(PF)
            sys.exit()
        lindat_min = min(PF)
        lindat_max = max(PF)
        lindat_width = abs(lindat_max - lindat_min)
    else:
        lindat_min = numpy.nanmin(lindat)
        lindat_max = numpy.nanmax(lindat)
        lindat_width = abs(lindat_max - lindat_min)

    print 'slope bound = ', abound

    lindat = (lindat - lindat_min)/lindat_width  # subtract place field minimum and normalise place field width

    # print 'new place field min and max = ', numpy.nanmin(lindat), numpy.nanmax(lindat)

    # starting points of maximization
    Nrep = 20
    da = abs(abound[1]-abound[0])/Nrep
    astart = min(abound)

    Rfunc = lambda a: -numpy.absolute(numpy.mean(numpy.exp(1j*(phidat-2.*numpy.pi*a*lindat))))

    aopt = numpy.nan
    R = -10**10
    for n in range(Nrep):
        a0 = astart+n*da  # lower bound to find the minimum of Rfunc
        a1 = astart+(n+1)*da  # upper bound to find the minimum of Rfunc

        returnValues = scipy.optimize.fminbound(Rfunc, a0, a1, full_output=True, disp=1)  # xtol=1e-10

        atmp = returnValues[0]  # Parameters (over given interval) which minimize the objective function
        rtmp = returnValues[1]  # The function value at the minimum point

        if display:
            aopts.append(atmp)
            Rs.append(-rtmp)

        if -rtmp > R and max(abound) >= atmp >= min(abound):
         #   print rtmp, R, aopt, atmp
            R = -rtmp
            aopt = atmp

    print 'Optimal slope is ', aopt
    print '_L____________________________________________________________________________________________________'

    # phase offset
    v = numpy.mean(numpy.exp(1j*(phidat-2*numpy.pi*aopt*lindat)))
    phi0 = numpy.mod(numpy.angle(v), 2*numpy.pi)

    # lindat are the normalised spike places
    # phidat are the spike phases
    theta = numpy.angle(numpy.exp(2*numpy.pi*1j*numpy.abs(aopt)*lindat))

    thmean = numpy.angle(numpy.sum(numpy.exp(1j*theta)))
    phmean = numpy.angle(numpy.sum(numpy.exp(1j*phidat)))

    sthe = numpy.sin(theta-thmean)
    sphi = numpy.sin(phidat-phmean)

    c12 = numpy.sum(sthe*sphi)
    c11 = numpy.sum(sthe*sthe)
    c22 = numpy.sum(sphi*sphi)

    rho = c12/numpy.sqrt(c11*c22)

    lam22 = numpy.mean(sthe**2.*sphi**2)
    lam20 = numpy.mean(sphi**2)
    lam02 = numpy.mean(sthe**2)
    tval = rho*numpy.sqrt(lindat.size*lam20*lam02/lam22)

    p = 1 - scipy.special.erf(abs(tval)/numpy.sqrt(2))  # as in Kempter paper right after formula (A.17)

    print 'p-Value is ', p, ' ----------------------------------------------------------------------'

    if p > 1 or p < 0:
        'p-Value not possible! Calculation stopped!'
        sys.exit()

    if display:
        fig = pl.figure()
        ax = fig.add_subplot(221)
        ax.plot(lindat, phidat, '.k')
        ax.plot(lindat, phidat-2*numpy.pi, '.k')
        ax.plot(lindat, phi0+2*numpy.pi*aopt*lindat, 'r-')
        custom_plot.huebschMachen(ax)
        ax.set_yticks(numpy.arange(-3*numpy.pi, 3*numpy.pi, numpy.pi))


        ax = fig.add_subplot(222)
        # ax.plot(lindat, theta, '.k')
        ax.plot(lindat*lindat_width+lindat_min, phidat/(numpy.pi/180), '.k')
        ax.plot(lindat*lindat_width+lindat_min, phidat/(numpy.pi/180)-360, '.k')
        ax.plot(lindat*lindat_width+lindat_min, (phi0/(numpy.pi/180))+360*aopt*lindat, 'r-')
        custom_plot.huebschMachen(ax)

        ax = fig.add_subplot(234)
        ax.plot(sthe, sphi, '.b')
        custom_plot.huebschMachen(ax)

        ax = fig.add_subplot(235)
        try:
            ax.hist(sthe*sphi, 20)
        except IndexError:
            print 'place_cells.rcc: Could not make some plot.'
        custom_plot.huebschMachen(ax)

        ax = fig.add_subplot(236)
        ax.plot(aopts, Rs, '.-')
        ax.set_xlabel('a')
        ax.set_ylabel('R')
        custom_plot.huebschMachen(ax)

    results_dict = {}

    for i in ('rho', 'p', 'R', 'aopt', 'phi0'):
        results_dict[i] = locals()[i]

    return results_dict


###################################################### CLASSES

class placeCellList(spikes.spikezugList):

    def __init__(self, fileName=None, timeUnit='ms', **kwargs):
        spikes.spikezugList.__init__(self, fileName=None, timeUnit='ms', **kwargs)

    def __setitem__(self, id, spktrain):
        assert isinstance(spktrain, NTsig.SpikeList), "A SpikeList object can only contain SpikeTrain objects"
        self.spiketrains[id] = spktrain
        self._SpikeList__calc_startstop()

    def __recalc_startstop(self):
        if len(self.spiketrains):
            for id in self.spiketrains.keys():
                self.spiketrains[id]._spikezugList__recalc_startstop()

    def changeTimeUnit(self, newUnit='ms'):
        if len(self.spiketrains):
            for id in self.spiketrains.keys():
                self.spiketrains[id].changeTimeUnit(newUnit=newUnit)


class placeCell(spikes.spikezugList):
    """ A place cell.

    It is a spikezugList object but should contain data from one place cell only.
    """

    def __init__(self, fileName=None, timeUnit='ms', **kwargs):
        spikes.spikezugList.__init__(self, fileName=None, timeUnit='ms', **kwargs)
        self.traj = None
        self.thetaRange = (0, 20)
        self.standard_spiketrain_types = {0: ''}
        self.gain = None

    def get_runSpikes(self, threshspeed=None):
        """ Get spikes that occurred above certain running speed.

        Parameters
        ----------
        threshspeed : float, optional
            Minimum running speed for accepting spikes. Defaults to None. If None self.traj.threshspeed is used.
            If not None self.traj.threshspeed is taken.

        Returns
        -------
        run_pc : placeCell object
            placeCell object that contains only the spikes during running.
        """

        if threshspeed is not None and threshspeed != self.traj.threshspeed:
            logger.info('Reset threshspeed from '+str(self.traj.threshspeed)+' to ' +
                str(threshspeed)+' '+self.traj.spaceUnit+'/'+self.traj.timeUnit)
            self.traj.threshspeed = threshspeed
            logger.info('NOTE: Now calculating running traj >= '+str(threshspeed)+' '+self.traj.spaceUnit+'/'+self.traj.timeUnit)

        if not hasattr(self.traj, 'run_traj'):
            logger.info('NOTE: Now calculating running traj from self.traj!')
            self.traj.get_runTraj()

        # if not hasattr(self.traj, 'run_times'):
        #     logger.info('NOTE: Now calculating running traj >= '+str(threshspeed)+' '+self.traj.spaceUnit+'/'+self.traj.timeUnit)
        #     self.traj.getRunningTraj(threshspeed=threshspeed)

        #--- prepare new place cell object
        pc_type = type(self)
        run_pc = pc_type(timeUnit=self.timeUnit, t_start=self.t_start, t_stop=self.t_stop, dims=[2])
        run_pc.meta = self.meta
        run_pc.tags = self.tags

        #--- put running spike trains into the new object
        for id, st in self.spiketrains.iteritems():
            zug = st.get_runSpikes(self.traj)
            run_pc.__setitem__(id, zug)
            run_pc._SpikeList__calc_startstop()

        run_pc.traj = self.traj.run_traj
        #run_pc.getPlaceField()

        self.run_pc = run_pc

        if hasattr(run_pc, 'gain') and hasattr(self, 'gain'):
            run_pc.gain = self.gain

        return run_pc

    def getSpikePhases(self, lfp):
        """ Determine phases of spikes for all spike trains.

        Parameters
        ----------
        lfp : CSCsignal object
        """

        if not hasattr(lfp, 'hilbertPhase'):
            lfp.filter(self.thetaRange[0], self.thetaRange[1])
            lfp.hilbertTransform()

        for id, st in self.spiketrains.iteritems():
            st.getSpikePhases(lfp)


    def getSpikePlaces(self, interp=True, Traj=None):
        """ Determine places of spikes for all spike trains.

        Parameters
        ----------
        interp : bool, optional
            Interpolate places if true. Default True.
        Traj : trajectory, optional
            Default None. In that case self.traj is used.
        """

        if not Traj:
            Traj = self.traj
        for id, st in self.spiketrains.iteritems():
            st.getSpikePlaces(Traj, interp=interp)

    def get_inFieldSpikes(self, pf_xlim=False, xbin_only=False):
        """ Get spikes that occurred in the place field.

        Returns
        -------
        infield_pc : placeCell object
            placeCell object that contains only the spikes in the place field.
        """

        pc_type = type(self)
        infield_pc = pc_type(timeUnit=self.timeUnit, t_start=self.t_start, t_stop=self.t_stop, dims=[2])
        infield_pc.meta = self.meta
        infield_pc.tags = self.tags

        for id, st in self.spiketrains.iteritems():
            pc = st.get_inFieldSpikes(pf_xlim=pf_xlim, xbin_only=xbin_only)\

            infield_pc.__setitem__(id, pc)
            infield_pc._SpikeList__calc_startstop()

        infield_pc.traj = self.traj

        if hasattr(infield_pc, 'gain') and hasattr(self, 'gain'):
            infield_pc.gain = self.gain

        if pf_xlim is False:
            infield_pc.getPlaceField(xbin_only=xbin_only)

        self.inField_pc = infield_pc

        return infield_pc

    def getPlaceField(self, bin_size=FR_binsize, bin_smooth=FR_bin_smooth, xbin_only=False):
        """ Calculate place field for all spikes trains in the obejct.

        Parameters
        ----------
        bin_size : float, optional
            Width of a spatial bin. Default 0.02.
        bin_smooth : int, optional
            Bins for smoothing. Default 3.
        """

        for id, st in self.spiketrains.iteritems():
            st.getPlaceField(self.traj, bin_size=bin_size, bin_smooth=bin_smooth, xbin_only=xbin_only, gain=self.gain)

    def plotPlaceField(self, spiketrain_type='', bin_size=FR_binsize, bin_smooth=FR_bin_smooth, fig=None, ax=None,
                  show_contour=True, show_limits=True, show_colorbar=True, xbin_only=False, shading='gouraud'):
        """ Pseudo color plot of firing field.
        """

        for id, st in self.spiketrains.iteritems():
            if st.spiketrain_type == spiketrain_type:
                logger.info('Plotting '+spiketrain_type+' firing field.')
                st.plotPlaceField(self.traj, bin_size=bin_size, bin_smooth=bin_smooth,
                             fig=fig, ax=ax, show_contour=show_contour, show_limits=show_limits,
                             show_colorbar=show_colorbar, xbin_only=xbin_only, shading=shading, gain=self.gain)

    def spiketrain_types(self):

        st_types = {}
        for id, st in self.spiketrains.iteritems():
            st_types[id] = st.spiketrain_type
        return st_types

    def check_spiketrain_types(self):

        st_types = self.spiketrain_types()
        for i, st_type in st_types:
            if not st_type in self.standard_spiketrain_types:
                logger.warning(st_type+' not in standard_spiketrain_types for class '+str(self.__class__))
            elif self.standard_spiketrain_types[i] != st_type:
                logger.warning('Sorting of '+st_type+' not as in standard_spiketrain_types for class '+str(self.__class__))

    def plotSpikesvsPlace(self, spiketrain_type='', onlyRunning=False, showHeadDir=False, fig=None, ax=None):
        """ Plot spikes vs. trajectory.
        """
        for id, st in self.spiketrains.iteritems():
            if st.spiketrain_type == spiketrain_type:
                logger.info('Plotting '+spiketrain_type+' spikes.')
                if not onlyRunning:
                    st.plotSpikesvsPlace(traj=self.traj, showHeadDir=showHeadDir, color=transred, fig=fig, ax=ax)
                    self.run_pc[id].plotSpikesvsPlace(showHeadDir=showHeadDir, color='r', fig=fig, ax=ax)
                else:
                    self.run_pc[id].plotSpikesvsPlace(traj=self.traj, showHeadDir=showHeadDir, color='r', fig=fig, ax=ax)
                        # alternatively provide self.traj.run_traj here if you only want running trajectories to be shown

    def time_slice(self, t_start, t_stop, meta=None, gain=None):
        """ Return a new place cell obtained by slicing between t_start and t_stop

        Parameters
        ----------
        t_start : float
            Beginning of the new object, in ms.
        t_stop  : float
            End of the new object, in ms.

        Returns
        -------
        new_pc : placeCell object
        """
        if not meta:
            meta = self.meta

        dims = 1
        if meta.has_key('dimensions'):
           dims = meta['dimensions']

        pc_type = type(self)
        new_pc = pc_type(timeUnit=self.timeUnit, t_start=t_start, t_stop=t_stop, dims=dims)
        new_pc.timeUnit = self.timeUnit

        new_pc.meta = meta

        if inspect.ismethod(self.id_list):
            IDlist = self.id_list()
        else:
            IDlist = self.id_list

        for id in IDlist:
            new_pc.append(id, self.spiketrains[id].time_slice(t_start, t_stop))
            new_pc._SpikeList__calc_startstop()

        for id in self.tags:
            new_pc.tags = self.tags.copy()

        if self.traj:
            new_pc.traj = self.traj.time_slice(t_start, t_stop)

        if hasattr(self, 'spiketrains') and hasattr(new_pc, 'spiketrains'):
            new_pc.spiketrains[0].timeUnit = self.spiketrains[0].timeUnit

        # take over thetaRange:
        new_pc.thetaRange = self.thetaRange

        if gain:
            new_pc.gain = gain

        return new_pc

    def concatenate(self, placeCell, meta=None, nan=True):
        """ Return a new place cell with its attached run_pc, obtained by concatenating given placeCell object

        Parameters
        ----------
        placeCell : placeCell object

        Returns
        -------
        new_pc : placeCell object
        new_pc.run_pc : placeCell object
        """
        if not meta:
            meta = self.meta

        dims = 1
        if meta.has_key('dimensions'):
           dims = meta['dimensions']

        # create new place cells, 1 - for concatenated place cell and 2 - for its concatenated run place cell
        pc_type = type(self)
        new_pc = pc_type(timeUnit=self.timeUnit, t_start=self.t_start, t_stop=placeCell.t_stop, dims=dims)
        new_run_pc = pc_type(timeUnit=self.timeUnit, t_start=self.t_start, t_stop=placeCell.t_stop, dims=dims)
        new_pc.timeUnit = self.timeUnit
        new_run_pc.timeUnit = self.timeUnit

        # update metadata
        new_pc.meta = meta
        new_run_pc.meta = meta

        # update tags
        for id in self.tags:
            new_pc.tags = self.tags.copy()
            new_run_pc.tags = self.tags.copy()

        # getting always all, rightwards and leftwards runs and their spike_times
        for pc in [self, placeCell]:
            pc.get_runSpikes(pc.traj.threshspeed)
        for p in [self, self.run_pc, placeCell, placeCell.run_pc]:
            p.traj.getLeftAndRightwardRuns()
            p.getLeftAndRightwardSpikes()

        # concatenate complete, rightward and leftward spiketrains of both place cells and their run place cells.
        for id in [0, 1, 2]:
            new_pc.append(id, self.spiketrains[id].concatenate(placeCell.spiketrains[id], t_start=self.t_start,
                                                               t_stop=placeCell.t_stop))
            new_pc._SpikeList__calc_startstop()

            new_run_pc.append(id, self.run_pc.spiketrains[id].concatenate(placeCell.run_pc.spiketrains[id],
                                                                          t_start=self.t_start, t_stop=placeCell.t_stop))
            new_run_pc._SpikeList__calc_startstop()


        # concatenating trajs with nans to avoid jumps in the trajectory. Nan can be set to False if the trajs follow
        # up without jump.
        new_pc.traj = self.traj.concatenate(Traj=placeCell.traj, t_start=self.t_start, t_stop=placeCell.t_stop, nan=nan)
        new_run_pc.traj = self.run_pc.traj.concatenate(Traj=placeCell.run_pc.traj, t_start=self.t_start, t_stop=placeCell.t_stop, nan=nan)

        # if hasattr(self.traj, 'rightward_traj'):
        new_pc.traj.rightward_traj = self.traj.rightward_traj.concatenate(Traj=placeCell.traj.rightward_traj, t_start=self.t_start,
                                                                          t_stop=placeCell.t_stop, nan=nan)
        new_pc.traj.leftward_traj = self.traj.leftward_traj.concatenate(Traj=placeCell.traj.leftward_traj, t_start=self.t_start,
                                                                        t_stop=placeCell.t_stop, nan=nan)

        new_run_pc.traj.rightward_traj = self.run_pc.traj.rightward_traj.concatenate(Traj=placeCell.run_pc.traj.rightward_traj,
                                                                                     t_start=self.t_start, t_stop=placeCell.t_stop, nan=nan)
        new_run_pc.traj.leftward_traj = self.run_pc.traj.leftward_traj.concatenate(Traj=placeCell.run_pc.traj.leftward_traj,
                                                                                     t_start=self.t_start, t_stop=placeCell.t_stop, nan=nan)

        # update timeUnits
        if hasattr(self, 'spiketrains') and hasattr(new_pc, 'spiketrains'):
            new_pc.spiketrains[0].timeUnit = self.spiketrains[0].timeUnit
            new_run_pc.spiketrains[0].timeUnit = self.spiketrains[0].timeUnit

        # attach new_run_pc to new_pc
        new_pc.run_pc = new_run_pc

        # take over thetaRange:
        new_pc.thetaRange = self.thetaRange
        new_pc.run_pc.thetaRange = self.thetaRange

        if hasattr(self, 'gain') and hasattr(placeCell, 'gain'):
            if self.gain == placeCell.gain:
                new_pc.gain = self.gain
                new_pc.run_pc.gain = self.gain
            else:
                print '_______________ Two place cells which should be concatenated have different gains! ' \
                      'New place cell will not have a gain attribute _______________'
        else:
            print '_______________ Two place cells which should be concatenated have NO gain attribute! ' \
                  'New place cell will not have a gain attribute _______________'

        return new_pc

    def subtract_time_offset(self, time_offset):

        """ Return a new place cell obtained by subtracting time_offset from spikes and trajectory.

        Parameters
        ----------
        time_offset : float
            Beginning of the new object, in ms.

        Returns
        -------
        new_pc : placeCell object
        """

        # getting all running times and therefore self.run_pc
        if not hasattr(self, 'run_pc'):
            self.get_runSpikes(self.traj.threshspeed)

        if not hasattr(self.traj, 'rightward_traj'):
            for p in [self, self.run_pc]:
                p.traj.getLeftAndRightwardRuns()
                p.getLeftAndRightwardSpikes()

        new_pc = copy.deepcopy(self)

        pc = [new_pc, new_pc.run_pc]

        for i in numpy.arange(len(pc)):

            for c in [0, 1, 2]:
                if len(pc[i].spiketrains[c].spike_times):
                    pc[i].spiketrains[c].spike_times -= time_offset

            pc[i].traj.times -= time_offset
            pc[i].traj.rightward_traj.times -= time_offset
            pc[i].traj.leftward_traj.times -= time_offset

        new_pc.t_start = numpy.min(new_pc.traj.times)
        new_pc.t_stop = numpy.max(new_pc.traj.times)

        return new_pc

    def subtract_xtraj_offset(self, xtraj_offset, normalisation_gain=1.0, changeSign=False):

        """ Return a new place cell obtained by subtracting xtraj_offset from trajectory.

        Parameters
        ----------
        xtraj_offset : float
            Beginning of the new object, in ms.

        Returns
        -------
        new_pc : placeCell object
        """

        # getting all running times and therefore self.run_pc
        if not hasattr(self, 'run_pc'):
            self.get_runSpikes(self.traj.threshspeed)

        # getting left and rightward trajs and spikes if they dont exist
        if not hasattr(self.traj, 'rightward_traj'):
            for p in [self, self.run_pc]:
                p.traj.getLeftAndRightwardRuns()
                p.getLeftAndRightwardSpikes()

        # get all possible spike places if they dont exist
        for p in [self, self.run_pc]:
            for c in [0, 1, 2]:
                if not hasattr(p.spiketrains[c], 'spike_places'):
                    p.spiketrains[c].getSpikePlaces(traj=self.traj)

        # copy old placeCel (= self) into new_pc, so that old place cell does not get modified!
        new_pc = copy.deepcopy(self)

        pc = [new_pc, new_pc.run_pc]

        for i in numpy.arange(len(pc)):

            for c in [0, 1, 2]:
                if len(pc[i].spiketrains[c].spike_places):  # if not empty
                    pc[i].spiketrains[c].spike_places[:, 0] -= xtraj_offset
                    pc[i].spiketrains[c].spike_places[:, 0] /= normalisation_gain
                    if changeSign:
                        pc[i].spiketrains[c].spike_places[:, 0] *= -1

            for trajs in [pc[i].traj, pc[i].traj.rightward_traj, pc[i].traj.leftward_traj]:
                trajs.places[:, 0] -= xtraj_offset
                trajs.places[:, 0] /= normalisation_gain

                if changeSign:
                    trajs.places[:, 0] *= -1

        return new_pc


class placeCell1d(placeCell):
    """ A place cell in a 1D environment.

    Is a spikezugList object but should contain data from one place cell only.
    Made up of up to three different placeCell_spikezug objects:
    (0) all data,
    (1) rightward runs,
    (2) leftward runs.
    """

    def __init__(self, fileName=None, timeUnit='ms', **kwargs):
        placeCell.__init__(self, fileName=None, timeUnit='ms', **kwargs)
        self.standard_spiketrain_types = {0: '', 1: 'rightward', 2: 'leftward'}

    def getLeftAndRightwardSpikes(self):

        st = numpy.copy(self.spiketrains[0].spike_times)
        rechts_spikeTimes = numpy.copy(self.spiketrains[0].spike_times)
        links_spikeTimes = numpy.copy(self.spiketrains[0].spike_times)

        # calculate running directions
        if not hasattr(self.traj, 'rightward_traj'):
            rightward_traj, leftward_traj = self.traj.getLeftAndRightwardRuns()
        else:
            rightward_traj = self.traj.rightward_traj
            leftward_traj = self.traj.leftward_traj

        for j, time in enumerate(st):
            if numpy.ma.count_masked(numpy.ma.masked_values(rightward_traj.times, time, atol=rightward_traj.dt)) == 0:
                rechts_spikeTimes[j] = numpy.nan
            if numpy.ma.count_masked(numpy.ma.masked_values(leftward_traj.times, time, atol=leftward_traj.dt)) == 0:
                links_spikeTimes[j] = numpy.nan

        #remove nan's from Spike-arrays!!
        if rechts_spikeTimes.size > 0:    # sonst fehler, wenn array empty
            rechts_spikeTimes = rechts_spikeTimes[~numpy.isnan(rechts_spikeTimes)]
        if links_spikeTimes.size > 0:     # sonst fehler, wenn array empty
            links_spikeTimes = links_spikeTimes[~numpy.isnan(links_spikeTimes)]

        # put as spike trains into list

        # HAD TO CHANGE NeuroTools/signals/spikes.py (line 120 and 128)!!

        # It captures the case when several spikes occur at the same time, but not any others occur, such that the
        # spiketrain calculated t_start (e.g. numpy.min(self.spike_times)) would be identical with t_stop. In this case
        # NeuroTools/signals/spikes.py gives out the 'Exception: Incompatible time interval : t_start = 92593.4,
        # t_stop = 92593.4'. In NeuroTools/signals/spikes.py (old line 127) ONLY the case is considered when only one
        # spike occurs (same problem). The described case is now identical with the protocol for one spike in
        # NeuroTools/signals/spikes.py and just resets t_stop by adding 0.1

        # 1) Add the new line 120 (it is just the discription):
        # # several with identical times: t_start = time, t_stop = time + 0.1
        # 2) Change line 128 (was line 127 before) to:
        # elif size == 1 or size > 1 and numpy.min(self.spike_times) == numpy.max(self.spike_times): # spike list may
        # be empty or may contain only a few spike at the same time point
        # 3) delete NeuroTools/signals/spikes.pyc so that the script NeuroTools/signals/spikes.py will be automatically
        # recompiled!

        if len(rechts_spikeTimes) > 1 and numpy.nanmin(rechts_spikeTimes) == numpy.nanmax(rechts_spikeTimes) or \
            len(links_spikeTimes) > 1 and numpy.nanmin(links_spikeTimes) == numpy.nanmax(links_spikeTimes):
            print 'WARNING: if you get the - Exception: Incompatible time interval - you will have to adjust ' \
                  'NeuroTools/signals/spikes.py. For that read the information in place_cells.py at the end of ' \
                  'getLeftAndRightwardSpikes() !'

        self.__setitem__(1, placeCell_spikezug(rechts_spikeTimes, 'rightward'))
        self.__setitem__(2, placeCell_spikezug(links_spikeTimes, 'leftward'))


    def plotLeftAndRightwardSpikes(self, fig=None, ax=None):

        if not fig:
            fig = pl.figure()
        if not ax:
            ax = fig.add_subplot(111)

        st = self.spiketrains[0]

        # # plot all spikes in transred color.
        # if st.spike_times.size:
        #     ax.plot(st.spike_times, st.spike_places[:, 0], color=transred, marker='.', linestyle='None')

        #--- left/right trajs
        ax.plot(self.traj.rightward_traj.times, self.traj.rightward_traj.places[:, 0], linewidth=1.0, color='m', label="right")
        ax.plot(self.traj.leftward_traj.times, self.traj.leftward_traj.places[:, 0], linewidth=1.0, color='b', label="left")

        #--- left/right spikes
        for id, st in self.spiketrains.iteritems():
            if st.spiketrain_type == 'rightward' and st.spike_times.size:
                ax.plot(st.spike_times, st.spike_places[:, 0], color='r', marker='o', markersize=4, linestyle='None')
            elif st.spiketrain_type == 'leftward' and st.spike_times.size:
                ax.plot(st.spike_times, st.spike_places[:, 0], color='g', marker='o', markersize=4, linestyle='None')

        ax.set_xlabel('Time ('+self.traj.timeUnit+')')
        ax.set_xlim(self.traj.times[0], self.traj.times[-1])

        custom_plot.huebschMachen(ax)


class placeCell_linear(placeCell1d):
    """ A place cell on a linear track.

    It is a spikezugList object but should contain data from one place cell only.
    """

    def __init__(self, fileName=None, timeUnit='ms', **kwargs):
        placeCell1d.__init__(self, fileName=None, timeUnit='ms', **kwargs)

    def fit_theta_phases(self, display_fitting_window=False):

        for id, st in self.spiketrains.iteritems():
            print 'fitting for spiketrains'
            fit_dictio = st.fit_theta_phases(display_fitting_window=display_fitting_window)

    def plotPhases(self, spiketrain_type='', fit=False):

        for id, st in self.spiketrains.iteritems():
                if st.spiketrain_type == spiketrain_type:
                    st.plotPhases(self.traj, fit=fit)

    def plotSpikesvsXComponent(self, spiketrain_type='', onlyRunning=False, fig=None, ax=None):

        if not fig:
            fig = pl.figure()
        if not ax:
            ax = fig.add_subplot(111)

        ax.plot(self.traj.run_traj.times, self.traj.run_traj.places[:, 0], linewidth=2.5, color=numpy.ones(3)*.25)    #only run trajectory

        if not onlyRunning:
            ax.plot(self.traj.times, self.traj.places[:, 0], linewidth=.5, color='k')             # original/full trajectory

        for id, st in self.spiketrains.iteritems():
            if st.spiketrain_type == spiketrain_type:
                logger.info('Plotting '+spiketrain_type+' spikes.')
                if not onlyRunning and len(st.spike_places):
                    ax.plot(st.spike_times, st.spike_places[:, 0], color=transred, marker='.', linestyle='None') # all spiketimes
                if len(self.run_pc[id].spike_places):
                    ax.plot(self.run_pc[id].spike_times, self.run_pc[id].spike_places[:, 0], 'r.')               # only running-spikes

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('x position ('+self.traj.spaceUnit+')')
        ax.set_xlim(self.traj.times[0], self.traj.times[-1])
        custom_plot.huebschMachen(ax)

    def save(self, inField=True, xbin_only=False):
        """ Save data in placeCell.

        Does not work with whole object. Instead we store important arrays after they were collected into a dictionary.

        Parameters
        ----------
        inField : bool, optional
            If true only spikes inside the place field are saved. Defaults to True.
        """

        if inField:
            pc = self.inField_pc
        else:
            pc = self

        dictio = {}
        for id, st in pc.spiketrains.iteritems():
            d = st.save(direct_save=False, xbin_only=xbin_only)

            if st.spiketrain_type != '':
                keys = d.keys()
                for key in keys:
                    d[st.spiketrain_type+'_'+key] = d.pop(key)
            dictio.update(d)

        # for k, v in dictio.iteritems():
        #     print k, type(v)

        folder = os.path.normpath(self.tags['dir'])+'/ana/'
        if io.save_analysis(folder+'pc'+self.tags['file'].split('.')[0]+'.hkl', dictio):
            logger.info('saved analysis for '+self.tags['file']+' ...')


class placeCell_spikezug(spikes.spikezug):
    """ Spike train class for a place modulated cell.

    It extends the NeuroTools SpikeTrain class.
    """

    def __init__(self, spike_times, spiketrain_type='', t_start=None, t_stop=None, timeUnit='ms'):

        spikes.spikezug.__init__(self, spike_times, spiketrain_type=spiketrain_type, t_start=t_start, t_stop=t_stop)

    def get_runSpikes(self, traj):
        """ Get spikes that occurred above certain running speed.

        Parameters
        ----------
        traj : trajectory object
            The spatial trajectory.

        Returns
        -------
        new_zug : placeCell_spikezug object
        """

        #threshspeed = traj.threshspeed


        #--- parameters
        inclPhases = True
        if not hasattr(self, 'spike_phases'):
            # logger.warning('Calculating without spike phases! If spike phases are needed caluculate them in advance!')
            inclPhases = False
        inclHeadDir = True
        if not hasattr(self, 'spike_headDirections'):
            # logger.warning('Calculating without spike head directions! If spike head directions are needed caluculate them in advance!')
            inclHeadDir = False
        inclPlaces = True
        if not hasattr(self, 'spike_places'):
            # logger.warning('Calculating without spike places! If spike places are needed caluculate them in advance!')
            inclPlaces = False

        #--- initialization
        run_spikeTimes = numpy.copy(self.spike_times)
        if inclPhases:
            run_spikePhases = numpy.copy(self.spike_phases)
        if inclHeadDir:
            run_spikeHeadDirections = numpy.copy(self.spike_headDirections)
        # if not hasattr(self, 'spike_places'):
        #     logger.info('Now calculating places of spikes.')
        #     self.getSpikePlaces(traj)
        # run_spikePlaces = numpy.copy(self.spike_places)

        if inclPlaces:
            run_spikePlaces = numpy.copy(self.spike_places)

        #--- calculate spikes during running
        for j, time in enumerate(self.spike_times):
            if numpy.ma.count_masked(numpy.ma.masked_values(traj.run_traj.times, time, atol=traj.dt)) == 0:
                run_spikeTimes[j] = numpy.nan
                if inclPlaces:
                    run_spikePlaces[j] = numpy.nan
                if inclPhases:
                    run_spikePhases[j] = numpy.nan
                if inclHeadDir:
                    run_spikeHeadDirections[j] = numpy.nan

        # remove nans from arrays
        if run_spikeTimes.size > 0: # sonst fehler, wenn array empty
            run_spikeTimes = run_spikeTimes[~numpy.isnan(run_spikeTimes)]
            if inclPlaces:
                run_spikePlaces = run_spikePlaces[~numpy.isnan(run_spikePlaces).any(1)]
            if inclPhases:
                run_spikePhases = run_spikePhases[~numpy.isnan(run_spikePhases)]
            if inclHeadDir:
                run_spikeHeadDirections = run_spikeHeadDirections[~numpy.isnan(run_spikeHeadDirections)]

        #--- dump everything into new placeCell_spikezug object
        new_zug = placeCell_spikezug(run_spikeTimes, spiketrain_type=self.spiketrain_type, timeUnit=self.timeUnit)
        if inclPlaces:
            new_zug.spike_places = run_spikePlaces
        if inclPhases:
            new_zug.spike_phases = run_spikePhases
        if inclHeadDir:
            new_zug.spike_headDirections = run_spikeHeadDirections

        return new_zug

    def getSpikePhases(self, lfp):
        """ Get firing phase with respect to LFP.

        Parameters
        ----------
        lfp : CSCsignal object

        Returns
        -------
        spike_phases : ndarray
            Spike phases with respect to theta oscillation. In addition written into self.spike_phases.
        """

        spikes = self.spike_times

        if not hasattr(lfp, 'hilbertPhase'):
            sys.exit('ERROR: lfp should be bandpass-filtered in advance!')

        phases = []
        for spike in spikes:
            #phases.append(lfp.hilbertPhase[(lfp.timeAxis).searchsorted(spike)])
            if len(lfp.hilbertPhase) != len(lfp.times):
                times = lfp.times[0:len(lfp.hilbertPhase)]
            else:
                times = lfp.times
            phases.append(lfp.hilbertPhase[tools.findNearest(times, spike)[0]])

        self.spike_phases = numpy.array(phases)

        return self.spike_phases

    def getSpikePlaces(self, traj, interp=True):
        """ Determine the places at which the spikes in the train occurred.
        """

        if interp:
            logger.info('Using linear interpolation to get the places.')

        spike_places = []

        for spike in self.spike_times:
            if spike > traj.times[-1]:
                logger.warning('At time '+str(spike)+' there was a spike after the end of the trajectory '+str(traj.times[-1])+' !')
                spike_places.append(numpy.ones(3)*0)
            else:
                place = traj.getPlaceFromTime(spike, interp=interp)
                if place is None:
                    spike_places.append(numpy.ones(3)*0)
                else:
                    spike_places.extend(place)
                #spike_places.extend(traj.getPlaceFromTime(spike, interp=interp))

        self.spike_places = numpy.array(spike_places)

    def getSpikeHeadDirection(self, interp=True):
        """ Calculate the head direction at which a certain spike occurred.

        The results are not directly returned but save into self.spike_headDirections.

        Parameters
        ----------
        interp : bool, optional
            Interpolate places.
        """

        if interp:
            print "NOTE: Using interpolation to get head directions."

        spike_headDirections = []
        for spike in self.spike_times:
            if spike > self.traj.times[-1]:
                print 'WARNING: at time', spike, 'there was a spike after the trajectory was over!'
                spike_headDirections.append(0)
            else:
                spike_headDirections.extend(self.traj.getHeadDirectionFromTime(spike, interp=interp))
        self.spike_headDirections = numpy.array(spike_headDirections)

    def plotSpikesvsPlace(self, traj=None, color='r', showHeadDir=False, fig=None, ax=None):
        """ Plot spikes vs. trajectory.

        Parameters
        ----------
        traj : trajectory-like object
        """

        if not fig:
            fig = pl.figure()
        if not ax:
            ax = fig.add_subplot(111)

        if not hasattr(self, 'spike_places'):
            self.getSpikePlaces(traj)

        # plot trajectory
        if traj is not None:
            ax.plot(traj.places[:, 0], traj.places[:, 1], linewidth=1, color=numpy.ones(3)*.5) # original/full trajectory

        # plot spikes
        if len(self.spike_places):
            ax.plot(self.spike_places[:, 0], self.spike_places[:, 1], color=color, marker='.', linestyle='None') #, label='all spikes')  # all spiketimes

        # plot head direction
        if showHeadDir:
            hd = traj.getHeadDirection()
            ax.quiver(traj.places[:, 0], traj.places[:, 1], numpy.cos(hd),
                      numpy.sin(hd), pivot='mid', color='g', units='y')
            ax.quiver(self.spike_places[:, 0], self.spike_places[:, 1], numpy.cos(self.spike_headDirections),
                      numpy.sin(self.spike_headDirections), pivot='mid', color=[.1, .1, 1], units='dots')

        # huebsch machen
        if traj is not None:
            ax.set_xlim(traj.xlim+numpy.diff(traj.xlim)*.05*[-1, 1])
            ax.set_ylim(traj.ylim+numpy.diff(traj.ylim)*.1*[-1, 1])

            ax.set_xlabel('x position ('+traj.spaceUnit+')')
            ax.set_ylabel('y position ('+traj.spaceUnit+')')

        return fig, ax

    def get_inFieldSpikes(self, pf_xlim=False, xbin_only=False):
        """Get spikes in place field.

        Returns
        -------
        new_zug : placeCell_spikezug object

        Notes
        -----
        Run getPlaceField() before!
        """

        if not pf_xlim and not xbin_only:
            if not self.place_field.border.size:
                print 'No', self.spiketrain_type, 'field.'
                return placeCell_spikezug([], spiketrain_type=self.spiketrain_type, timeUnit=self.timeUnit)

        spike_places = self.spike_places
        spikes = self.spike_times
        if hasattr(self, 'spike_phases'):
            spike_phases = self.spike_phases

        if not pf_xlim:
            if not xbin_only:
                pol = mpl_path.Path(self.place_field.border)
                s = pol.contains_points(spike_places[:, :2])
            else:
                if spike_places.size:
                    s = numpy.where(numpy.logical_and(spike_places[:, 0] >= self.place_field_xbin.x0,
                                                      spike_places[:, 0] <= self.place_field_xbin.x1))[0]
                else:
                    s = []
        else:
            if spike_places.size:
                s = numpy.where(numpy.logical_and(spike_places[:, 0] >= pf_xlim[0],
                                                  spike_places[:, 0] <= pf_xlim[1]))[0]
            else:
                s = []

        pf_spikes = spikes[s]
        pf_places = spike_places[s]
        if hasattr(self, 'spike_phases'):
            pf_phases = spike_phases[s]

        new_zug = placeCell_spikezug(pf_spikes, spiketrain_type=self.spiketrain_type, timeUnit=self.timeUnit)
        new_zug.spike_places = pf_places
        if hasattr(self, 'spike_phases'):
            new_zug.spike_phases = pf_phases

        return new_zug

    def fit_theta_phases(self, display_fitting_window=False):

        if not self.spike_times.size:
            self.phase_fit_dictio = {'phi0': numpy.nan, 'aopt': numpy.nan}
            return self.phase_fit_dictio

        spike_places = self.spike_places[:, 0]
        spike_phases = self.spike_phases

        # if hasattr(self, 'place_field_xbin'):
        #     print 'xbin PLACE FIELD FOUND!'
        #     field = [self.place_field_xbin.x0, self.place_field_xbin.x1]
        #     field_begin = min(field)
        if hasattr(self, 'place_field'):
            field = [self.place_field.x0, self.place_field.x1]
            field_begin = min(field)
            field_end = max(field)
            x_width = abs(field_end - field_begin)
        else:
            field = False
            field_begin = numpy.nanmin(spike_places)
            x_width = abs(numpy.nanmax(spike_places) - numpy.nanmin(spike_places))

        self.phase_fit_dictio = rcc(spike_places, spike_phases, abound=[-2.5, 2.5], display=display_fitting_window,
                                    PF=field)

        # if places where normalised my its maximum => slope is without unit
        # => numltiplication by 360 converts it to degrees
        self.phase_fit_dictio['aopt'] *= 360

        # undo normalisation with place field width which happens in the rcc fitting function
        self.phase_fit_dictio['aopt'] /= x_width
        # self.phase_fit_dictio['aopt'] *= (spike_places - numpy.nanmin(spike_places)) / (spike_places*x_width)

        # convert offset phase from radiants into degrees
        self.phase_fit_dictio['phi0'] /= numpy.pi/180
        # move offset according to place field beginning
        self.phase_fit_dictio['phi0'] -= self.phase_fit_dictio['aopt']*field_begin

        return self.phase_fit_dictio

    def getPlaceField(self, traj, bin_size=FR_binsize, bin_smooth=FR_bin_smooth, xbin_only=False, gain=False, occu_perc=0.05):
        """ Get spatial firing field, i.e. place field.

        Parameters
        ----------
        traj : trajectory object
        bin_size : float, optional
            Width of a spatial bin. Default 0.02.
        bin_smooth : int, optional
            Bins for smoothing. Default 3.

        Returns
        -------
        self.place_field : place_field object
        """
        # xbin_only = True
        if not self.spike_times.size:
            self.place_field = placeField([[0]], [[0]], [[0]])
            self.place_field_xbin = placeField([[0]], [[0]], [[0]])
            self.occupancy_probability_ysum = 0
            self.spike_count_xbin = 0
            if xbin_only:
                return self.place_field, self.place_field_xbin, self.occupancy_probability_ysum, self.spike_count_xbin
            else:
                return self.place_field

        spikes = self.spike_places[:, :2]

        # bin track!

        # adjust binsize for gain 1.5:

        if gain == 1.5:
            bin_size = 1.5*(FR_binsize/0.5)

        # if tracklegth is not 2m then normalise bin size:
        # length1 = 4./3
        # length2 = 4.
        # length_vis = 2.
        # closest_lim = numpy.argmin([abs(traj.xlim[1]-length1), abs(traj.xlim[1]-length_vis), abs(traj.xlim[1]-length2)])

        # to make normalised plots in visual binning
        # if closest_lim == 0:
        #     bin_size = length1/(length_vis/bin_size)
        # elif closest_lim == 2:
        #     bin_size = length2/(length_vis/bin_size)

        # to make visual plots in normalised binning
        # if closest_lim == 1:
        #     if gain == 0.5:
        #         bin_size = length_vis/(length2/bin_size)
        #     elif gain == 1.5:
        #         bin_size = length_vis/(length1/bin_size)
        #     else:
        #         print 'ERROR: No gain information in place_cells.getPlaceField() given !'
        #         sys.exit()

        track_x1 = numpy.arange(traj.xlim[0], traj.xlim[1], bin_size)
        track_x2 = numpy.arange(traj.xlim[0]+bin_size, traj.xlim[1]+bin_size, bin_size)

        track_y1 = numpy.arange(traj.ylim[0], traj.ylim[1], bin_size)
        track_y2 = numpy.arange(traj.ylim[0]+bin_size, traj.ylim[1]+bin_size, bin_size)

        if not track_x1.size:
            track_x1 = numpy.array([traj.xlim[0]])
        if not track_x2.size:
            track_x2 = numpy.array([traj.xlim[0]]) + bin_size
        if not track_y1.size:
            track_y1 = numpy.array([traj.ylim[0]])
        if not track_y2.size:
            track_y2 = numpy.array([traj.ylim[0]]) + bin_size

        if xbin_only:
            track_y1a = numpy.array([traj.ylim[0]])
            track_y2a = numpy.array([traj.ylim[1]])

            if not track_y1a.size:
                track_y1a = numpy.array([traj.ylim[0]])
            if not track_y2a.size:
                track_y2a = numpy.array([traj.ylim[0]]) + bin_size

        # create polygones & count spikes
        spike_number = []
        traj_number = []
        for l, ySP in enumerate(track_y1):
            for j, xSP in enumerate(track_x1):
                pol = mpl_path.Path([[track_x1[j], track_y1[l]], [track_x1[j], track_y2[l]],
                                    [track_x2[j], track_y2[l]], [track_x2[j], track_y1[l]]])    # Polygon erzeugen
                # pol = mpl_path.Path([[track_x1[j], track_y1[l]], [track_x2[j], track_y1[l]],
                #                     [track_x1[j], track_y2[l]], [track_x2[j], track_y2[l]]])    # Polygon erzeugen
                spike_number.append(numpy.sum(pol.contains_points(spikes)))     # count number of spikes in polygon
                traj_number.append(numpy.sum(pol.contains_points(traj.places[:, :2])))


        spike_number = numpy.array(spike_number)
        traj_number = numpy.array(traj_number)

        bin_occupancy = numpy.array(traj.dt * traj_number)

        # min_occu = occu_perc*numpy.nanmax(bin_occupancy)
        # bin_occupancy[bin_occupancy <= min_occu] = numpy.nan
        #
        f = spike_number/bin_occupancy

        # occu_add = occu_perc*numpy.nanmax(bin_occupancy)
        # f = spike_number/(bin_occupancy+occu_add)

        f[numpy.isnan(f)] = 0
        f[numpy.isinf(f)] = 0

        f = f.reshape(len(track_y1), len(track_x1))  # reshape & array spike_number list

        if xbin_only:
            # create polygones & count spikes
            spike_number_a = []
            traj_number_a = []

            for l, ySP in enumerate(track_y1a):
                for j, xSP in enumerate(track_x1):
                    pol_a = mpl_path.Path([[track_x1[j], track_y1a[l]], [track_x1[j], track_y2a[l]],
                                           [track_x2[j], track_y2a[l]], [track_x2[j], track_y1a[l]]])    # Polygon erzeugen
                    # pol_a = mpl_path.Path([[track_x1[j], track_y1a[l]], [track_x2[j], track_y1a[l]],
                    #                       [track_x1[j], track_y2a[l]], [track_x2[j], track_y2a[l]]])    # Polygon erzeugen (old version!)
                    spike_number_a.append(numpy.sum(pol_a.contains_points(spikes)))     # count number of spikes in polygon
                    traj_number_a.append(numpy.sum(pol_a.contains_points(traj.places[:, :2])))

            spike_number_a = numpy.array(spike_number_a)
            traj_number_a = numpy.array(traj_number_a)
            bin_occupancy_ysum = numpy.array(traj.dt * traj_number_a)

            # min_occu_y = occu_perc*numpy.nanmax(bin_occupancy_ysum)
            # bin_occupancy_ysum[bin_occupancy_ysum <= min_occu_y] = numpy.nan

            f_a = spike_number_a/bin_occupancy_ysum
            # occu_addy = occu_perc*numpy.nanmax(bin_occupancy_ysum)
            # f_a = spike_number_a/(bin_occupancy_ysum+occu_addy)

            f_a[numpy.isnan(f_a)] = 0
            f_a[numpy.isinf(f_a)] = 0

            self.occupancy_probability_ysum = numpy.array(bin_occupancy_ysum/sum(bin_occupancy_ysum))

        # pad field a bit to get rid of boundary effects (e.g. open contour lines)
        pads = 2
        for pas in range(pads):
            track_x1 = numpy.concatenate([[track_x1[0]-bin_size], track_x1, [track_x1[-1]+bin_size]])
            track_y1 = numpy.concatenate([[track_y1[0]-bin_size], track_y1, [track_y1[-1]+bin_size]])

        f = numpy.pad(f, (pads, pads), 'constant', constant_values=(0, 0))

        # smooth firing rate a bit
        smoothed_f = tools.blur_image(f, bin_smooth)
        # smoothed_spike_number = tools.blur_image(spike_number, bin_smooth)

        x, y = numpy.meshgrid(track_x1, track_y1)
        self.place_field = placeField(x, y, smoothed_f, gain=gain)

        if xbin_only:
            # padding only for xvalues of the firing rate, so that the arrays of x and f have the same length
            f_a = numpy.pad(f_a, (pads, pads), 'constant', constant_values=(0, 0))
            f_a = f_a.reshape(len(track_y1a), len(track_x1))

            smoothed_f_a = tools.blur_image(f_a, bin_smooth)
            x_a, y_a = numpy.meshgrid(track_x1, track_y1a)

            self.place_field_xbin = placeField(x_a, y_a, smoothed_f_a, xbin_only=True)
            self.spike_count_xbin = spike_number_a
            self.occupancy_xbin = bin_occupancy_ysum

            return self.place_field, self.place_field_xbin, self.occupancy_probability_ysum, self.spike_count_xbin

        else:
            return self.place_field

    def plotPlaceField(self, traj, bin_size=FR_binsize, bin_smooth=FR_bin_smooth, fig=None, ax=None, show_contour=True, show_limits=True,
                       limx=None, limy=None, show_colorbar=True, xbin_only=False, shading='gouraud', gain=None):
        """ Pseudo color plot of firing field.
        """
        # xbin_only = False
        if not fig:
            fig = pl.figure()
        if not ax:
            ax = fig.add_subplot(111)

        # check if place field exists and if default values are given. If not recalculate.
        if bin_size != .01 or bin_smooth != 5 or not hasattr(self, 'place_field'):
            logger.info('Recalculating place field.')
            print 'USING plotPlaceField!'
            self.getPlaceField(traj, bin_size=bin_size, bin_smooth=bin_smooth, xbin_only=xbin_only, gain=gain)

        if not xbin_only:
            place_field = self.place_field
        else:
            place_field = self.place_field_xbin

        # plot trajectory
        ax.plot(traj.places[:, 0], traj.places[:, 1], linewidth=1, color=numpy.ones(3)*.5)  # original/full trajectory

        x = self.place_field.x
        y = self.place_field.y
        field = self.place_field.f  # 2d firing rate of entire track space, not just place field!

        if xbin_only:
            # calculating maximal 2d firing rate within 1d place field x boundaries
            x0_idx = tools.findNearest(self.place_field.x[0], self.place_field_xbin.x0)[0]
            x1_idx = tools.findNearest(self.place_field.x[0], self.place_field_xbin.x1)[0]

            TwoD_PFmax = numpy.nanmax(self.place_field.f[:, x0_idx:x1_idx+1])

            pf_thresh = TwoD_PFmax*self.place_field_xbin.thresh_percent
        else:
            # in class placeField() self.place_field.thresh is defined as
            # = numpy.nanmax(self.place_field.f)*self.place_field.thresh_percent
            pf_thresh = self.place_field.thresh

        # mask (not show heatmap) for all firing rates (=field) which are below the pf_thresh:

        Hmasked = numpy.ma.masked_where(field <= pf_thresh*.5, field)       # Mask pixels with a value below 1

        # use 2d place field data only for color plot!__________________________

        plot_handle = ax.pcolormesh(x, y, Hmasked, zorder=3, shading=shading)
        # plot_handle = ax.pcolormesh(x, y, field, zorder=3, shading='gouraud')

        if show_colorbar:
            pos = ax.get_position().bounds
            colorbar_handle = pl.colorbar(plot_handle)
            ax.set_position(pos)
            pos = list(pos)
            pos[0] += pos[2]+.01
            colorbar_handle.ax.set_position(pos)

        # add maximum and 5% contour line
        if show_contour:

            ax.plot(place_field.x_max, place_field.y_max, 'mo', zorder=4)
            for cntr in place_field.contours.collections:
                for p in cntr.get_paths():
                    ax.plot(p.vertices[:, 0], p.vertices[:, 1], 'm-', linewidth=1, zorder=4)
            if place_field.border.size:
                ax.plot(place_field.border[:, 0], place_field.border[:, 1], 'm:', linewidth=3, zorder=4)

        if show_limits:

            ax.axvline(place_field.x0, linestyle=':', linewidth=0.5)
            ax.axvline(place_field.x1, linestyle=':', linewidth=0.5)
            ax.text(place_field.x0 + place_field.width*.5, traj.ylim[0],
                    str(numpy.round(place_field.width, 2))+' '+traj.spaceUnit,
                    ha='center')
            ax.axhline(place_field.y0, linestyle=':', linewidth=0.5)
            ax.axhline(place_field.y1, linestyle=':', linewidth=0.5)
            ax.text(traj.xlim[0], place_field.y0 + place_field.height*.5,
                    str(numpy.round(place_field.height, 2))+' '+traj.spaceUnit,
                    va='center', rotation=90)

        if limx:
            ax.set_xlim(limx)
        else:
            ax.set_xlim(traj.xlim[0], traj.xlim[1]-bin_size)
        if limy:
            ax.set_ylim(limy+numpy.diff(limy)*.1*[-1, 1])
        else:
            ax.set_ylim(traj.ylim[0], traj.ylim[1]-bin_size)

        ax.set_xlabel('x position ('+traj.spaceUnit+')')
        ax.set_ylabel('y position ('+traj.spaceUnit+')')

        return fig, ax

    def plotPhases(self, traj, fit=False, wHists=True, wTraj=False, fig=None, ax=None, labelx=True, labely=True, limx=False,
                   color=[0, 0, 0], lfp=None, labelsize=None):
                
        if not hasattr(self, 'spike_phases'):
            if not lfp:
                print 'Error: No lfp trace given to calculate spike phases!'
            else:
                self.getSpikePhases(lfp)

        if not fig:
            fig = pl.figure(figsize=(3.5, 3.5))
        if not ax:
            pos = [.2, .21, .65, .625]
            ax = fig.add_axes(pos)
        
        phasePlot(self.spike_times, self.spike_phases, self.spike_places, traj,
                            wHists=wHists, wTraj=wTraj, fig=fig, ax=ax, labelx=labelx, labely=labely,
                            limx=limx, color=color, labelsize=labelsize)
        if fit:
            dictio = self.fit_theta_phases()
            x = numpy.arange(numpy.nanmin(self.spike_places[:, 0]), numpy.nanmax(self.spike_places[:, 0]), .01)
            phi0 = dictio['phi0']
            aopt = dictio['aopt']
            for offset in [-360, 0, 360]:
                ax.plot(x, phi0+offset+aopt*x, 'r-')

        return fig, ax

    def save(self, direct_save=True, xbin_only=False):
        """ Save data in placeCell_spikezug.

        Does not work with whole object. Instead we store important arrays after they were collected into a dictionary.
        """
        # xbin_only = False
        if not xbin_only:
            place_field = self.place_field
        else:
            place_field = self.place_field_xbin

        # spike data
        params = ['run_spikePhases', 'run_spikePlaces', 'run_spikeTimes',
                  'spike_places', 'spike_phases', 'spike_times']
        values = [getattr(self, p) for p in params if hasattr(self, p)]

        # place field parameters

        for p in ['border', 'bounds', 'f_max']:
            params.append('place_field_'+p)
            values.append(getattr(place_field, p))
            if p == 'f_max':
                values[-1] = float(values[-1])          # hickle 0.2 does not like numpy.floats

        # theta phase fit parameters
        for p in ['phi0', 'aopt']:
            params.append('theta_phase_'+p)
            values.append(float(self.phase_fit_dictio[p]))

        # put into dictionary
        dictio = dict(zip(params, values))

        if direct_save:
            folder = os.path.normpath(self.tags['dir'])+'/ana/'
            if io.save_analysis(folder+'pc'+self.tags['file'].split('.')[0]+'.hkl', dictio):
                logger.info('saved analysis for '+self.tags['file']+' ...')

        return dictio

    def time_slice(self, t_start, t_stop):

        if hasattr(t_start, '__len__'):
            if len(t_start) != len(t_stop):
                raise ValueError("t_start has %d values and t_stop %d. They must be of the same length." % (len(t_start), len(t_stop)))
            mask = False
            for t0, t1 in zip(t_start, t_stop):
                mask = mask | ((self.spike_times >= t0) & (self.spike_times <= t1))
            t_start = t_start[0]
            t_stop = t_stop[-1]
        else:
            mask = (self.spike_times >= t_start) & (self.spike_times <= t_stop)
        spikes = numpy.extract(mask, self.spike_times)

        pc = placeCell_spikezug(spikes, t_start=t_start, t_stop=t_stop)

        return pc

    def concatenate(self, spiketrain, t_start, t_stop):

        spikes = numpy.concatenate((self.spike_times, spiketrain.spike_times))

        pc = placeCell_spikezug(spikes, t_start=t_start, t_stop=t_stop)

        return pc

    def subtract_time_offset(self, time_offset, t_start, t_stop):

        new_spike_times = self.spike_times.copy()

        pc = placeCell_spikezug(new_spike_times - time_offset, t_start=t_start, t_stop=t_stop)

        return pc


class placeField(object):
    """ Class for place fields.
    """

    def __init__(self, x, y, f, xbin_only=False, gain=False, index_maxx=False, index_maxy=False):

        self.xbin_only = xbin_only
        self.x = numpy.array(x)
        self.y = numpy.array(y)
        self.f = numpy.array(f)
        self.border = numpy.array([])

        if not index_maxy:
            self.f_max = numpy.nanmax(self.f)
            self.indices_max = numpy.where(self.f == self.f_max)
        else:
            self.indices_max = index_maxy
            if index_maxx:
                self.f_max = self.f[index_maxy][index_maxx]

        self.x_max = self.x[self.indices_max]
        self.y_max = self.y[self.indices_max]
        # check if more than one point was found. If more than one, take first one.
        if self.x_max.size > 1:
            logger.warning('Found '+str(self.x_max.size)+' points for maximum firing rate! Choosing first occurence.')
            self.x_max = self.x_max[0]
            self.y_max = self.y_max[0]

        # if gain == 1.5:
        #     self.thresh_percent = 0.15
        # else:
        self.thresh_percent = 0.1  # 05

        self.thresh = self.f_max * self.thresh_percent
        self.get_properties()

    def get_contours(self):

        self.contours = pl.contour(self.x, self.y, self.f, [self.thresh])
        for cntr in self.contours.collections:
            cntr.set_visible(False)
            for p in cntr.get_paths():
                if p.contains_point([self.x_max, self.y_max]):
                    self.border = p.vertices

    # puts subarrays into oneDarray when there is a value skipped in the array
    # (array values have to have the given stepsize!) stepsize=1 when an index 1d-array is looked at!
    def consecutive(self, oneDarray, stepsize=1):
        return numpy.split(oneDarray, numpy.where(numpy.diff(oneDarray) != stepsize)[0]+1)

    def get_properties(self):

        if not self.xbin_only:
            # maybe border was not determined
            if self.border.size == 0:
                self.get_contours()

            if self.border.size > 0:    # there is a place field
                self.x0 = numpy.nanmin(self.border[:, 0])
                self.x1 = numpy.nanmax(self.border[:, 0])
                self.y0 = numpy.nanmin(self.border[:, 1])
                self.y1 = numpy.nanmax(self.border[:, 1])
            else:                       # no place field. Put traj dimensions
                logger.warning('Found no place field! Set place field bounds to default values for linear track = ')
                self.x0 = 0
                self.x1 = 2.0
                self.y0 = 0
                self.y1 = 0.3
                print '[', self.x0, self.x1, ']'
        else:
            # define place field borders etc. from 1d firing rate, x, and y values
            f_argmax = numpy.nanargmax(self.f)
            # for x0: for the last consecutive index array [-1] (up to f_max, which is larger or equal to the FR thresh)
            #         take the first (lowest) x-value [0]
            self.x0 = self.x[0][self.consecutive(numpy.where(self.f[0][:f_argmax+1] >= self.thresh)[0])[-1][0]]

            # for x1: for the first consecutive index array [0] (from f_max, which is larger or equal to the FR thresh)
            #         take the last (highest) x-value [-1]
            self.x1 = self.x[0][self.consecutive(numpy.where(self.f[0][f_argmax:] >= self.thresh)[0]+f_argmax)[0][-1]]
            # in 1d use default linTrack values for place field height.
            self.y0 = 0
            self.y1 = 0.1
            # self.contours =
            # self.border =

        self.width = self.x1 - self.x0
        self.height = self.y1 - self.y0
        self.bounds = [self.x0, self.y0, self.width, self.height]