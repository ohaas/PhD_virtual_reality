"""
Overview plot including spike phases. Extended version.
"""
__author__ = ("KT", "Moritz Dittmeyer", "Olivia Haas")
__version__ = "8.1, September 2014"


# python modules
import sys
import os

# add additional custom paths
extraPaths=[
    "/home/thurley/data/",
    "/home/haas/packages/lib/python2.6/site-packages",
    os.path.join(os.path.abspath(os.path.dirname(__file__)), '../scripts')]
for p in extraPaths:
    if not sys.path.count(p):
        sys.path.insert(1, p)

# other modules
import numpy
import math
import inspect

# custom made modules
import signale
import trajectory
import custom_plot

# plotting modules
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pl
import matplotlib.lines as mlines
# import hickle


###################################################### functions

def getData(folderName):
    global cscID, cscList, loadedSomething, item
    global ID, traj, HDtraj, eventData, cscName

    if os.path.isdir(folderName):
        dirList=os.listdir(folderName)
        os.chdir(folderName)
    else:
        dirList = [folderName]

    for item in dirList:
        if os.path.isfile(item):
            if item.endswith(cscName): # not any([item.find(str(s))+1 for s in excludeCSCs]):
                print 'loading', item , 'from folder: '+folderName
                loadedSomething = True
                csc = signale.load_ncsFile(item, showHeader=False)
                cscID += 1
                cscList.append(cscID, csc)
                cscList.addTags(cscID, file=item, dir=folderName)
            elif (TTName.__class__ == list and item in TTName) or \
                    (TTName.__class__ == str and item.endswith('.t')):
                print 'loading', item , 'from folder: '+folderName
                spikes = signale.load_tFile_place_cell(item, showHeader=False)
                ID += 1
                stList.__setitem__(ID, spikes)
                stList.addTags(ID, file=item, dir=folderName)
            # real
            elif expType == 'real':
                if item.endswith('.nvt'):# or item.endswith('2.ncs'):
                    print 'loading', item, 'from folder: '+folderName
                    loadedSomething = True
                    traj = trajectory.load_nvtFile(item, 'linearMaze', showHeader=False)
                    HDtraj = traj[1]        # head direction
                    traj = traj[0]          # trajectory
            # vr
            elif expType == 'vr':
                if item.endswith('.nev'):
                    print 'loading', item , 'from folder: '+folderName
                    eventData = signale.load_nevFile(item, showHeader=False)
                elif item.endswith('.traj') and item.find('position')+1\
                 and not item.find('collisions_position')+1 and not item.find('rewardsVisited_position')+1:
                    print 'loading', item , 'from folder: '+folderName
                    traj = trajectory.load_trajectory(item, showHeader=False)
        elif os.path.isdir(item):
            getData(item)
    os.chdir('..')


def prepareData():

    for csc in cscList:
        csc.filter(thetaRange[0], thetaRange[1])
        csc.hilbertTransform()


# prepare list with data that should be plotted and plot the data
def prepareAndPlotPooledPhases(place_cell, lfp, cscName, fig, axes, limx, fontsize, limy=None, wfit=True, noSpeck=False,
                               text=True, mutterTraj=False, GetPlaceField=True, gc=False, smooth=False,
                               normalisation_gain=None, pf_xlim=False, direction=None, title=None, axNum=None,
                               extraPlots=True, xbin_only=False):
    """Phase plot and corresponding preparation steps.
    """

    grau2 = numpy.array([1, 1, 1])*.85
    print '--------------------------------------------------------------------------'

    if not hasattr(place_cell, 'run_pc'):
        # get running spikes, stored as a placeCell object into place_cell.run_pc
        print 'Getting run_pc and run_pc.spike_places!'
        place_cell.get_runSpikes(place_cell.traj.threshspeed)

    #--- calculate parameters
    for pc in [place_cell, place_cell.run_pc]:
        if smooth and not gc:
            pc.traj.smooth(1.0)  # value 1.0 is in seconds and stands for the size for the smoothing kernel
        elif smooth and gc:
            print 'WARNING: due to boundary effects single runs are not smoothed. If they should be smoothed, smooth ' \
                  'the entire trajectory beforehand!'

        if not hasattr(pc.traj, 'rightward_traj'):
            print 'calculating Left and Rightwards Runs'
            pc.traj.getLeftAndRightwardRuns()

        if inspect.ismethod(pc.id_list):
            pc_IDlist = pc.id_list()
        else:
            pc_IDlist = pc.id_list

        if len(pc_IDlist) != 3:
            print 'calculating Left and Rightward Spikes'
            pc.getLeftAndRightwardSpikes()

        pc_places = numpy.sum([hasattr(pc[0], 'spike_places'), hasattr(pc[1], 'spike_places'), hasattr(pc[2], 'spike_places')])
        pc_phases = numpy.sum([hasattr(pc[0], 'spike_phases'), hasattr(pc[1], 'spike_phases'), hasattr(pc[2], 'spike_phases')])

        if not gc and not pc_places == 3:
            pc.getSpikePlaces()
        elif not pc_places == 3:
            print 'INFO: using muttertraj to calculate spike places'
            pc.getSpikePlaces(Traj=mutterTraj)

        if not pc_phases == 3:
            print 'INFO: using mutterCSC to calculate spike phases'
            pc.getSpikePhases(lfp)

        # Normalising spike places after they where calculated with the unnormalised muttertraj and the phases
        # were calculated
        if normalisation_gain != 1.0:
            for i in numpy.arange(len(pc)):
                print 'INFO: Normalising spike places'
                pc[i].spike_places /= normalisation_gain

    if noSpeck:
        dummy = place_cell.traj
        place_cell = place_cell.run_pc
        place_cell.traj = dummy         # a bit weird but we want to keep the original traj for plotting
                                        # the traj of run_pc is only the run_traj
        place_cell.run_pc = place_cell

    trajs = [place_cell.traj, place_cell.traj.rightward_traj, place_cell.traj.leftward_traj]
    dummy_titles = ['all directions', 'rightward runs', 'leftward runs']
    if direction is not None and title is not None:
        dummy_titles[direction] = title
    firing_rate_distribution = []
    firing_rate_distribution_ySum = []
    pf_limits = []
    wfit_xy = []
    pf_phases_left_right = []
    occupancy_probability_ySum = []
    spike_count_ySum = []
    fit_dictionary = []

    for i, pc in enumerate(place_cell):

        if direction is None or direction == i:
            if direction is None and axes is not False:
                axe = axes[i]
            elif direction is not None:
                axe = axes[axNum]
            run_pc = place_cell.run_pc[i]

            # if limx, plot is getting restricted to placeField size -/+ 1/2 place field width to have a better view!
            if limx is True:
                if pf_xlim is False:
                    if not hasattr(pc, 'place_field'):
                        pc.getPlaceField(xbin_only=xbin_only)
                    if xbin_only:
                        pl_field = pc.place_field_xbin
                        # pl_field = pc.place_field
                    else:
                        pl_field = pc.place_field
                    limx = numpy.array([pl_field.x0, pl_field.x1]) + numpy.array([-1, 1])*pl_field.width*.5
                else:
                    limx = numpy.array(pf_xlim) + numpy.array([-1, 1])*(pf_xlim[1]-pf_xlim[0])*.5
                limx = limx.tolist()

            if not limy:
                limy = None

            if mutterTraj is False:
                mutterTraj = trajs[0]

            if extraPlots is True:
                w_traj = 'noSpeck'
                w_hists = True
            else:
                w_traj = False
                w_hists = False

            # Phase plots
            if fig is not False:
                if noSpeck and run_pc.spike_places.size:
                    if i == 1:  # rightwards run
                        spike_places = run_pc.spike_places[:, 0]
                        spike_phases = run_pc.spike_phases
                        spike_times = run_pc.spike_times
                        print 'place_cell.gain = ', place_cell.gain
                        # if axNum > 3:
                        #     hickle.dump(spike_places, '/Users/haasolivia/Desktop/plots/10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_spike_placesSR'+str(axNum-4)+'_gain'+str(place_cell.gain)+'.hkl', mode='w')
                        #     hickle.dump(spike_phases, '/Users/haasolivia/Desktop/plots/10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_spike_phasesSR'+str(axNum-4)+'_gain'+str(place_cell.gain)+'.hkl', mode='w')
                        #     hickle.dump(spike_times, '/Users/haasolivia/Desktop/plots/10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_spike_timesSR'+str(axNum-4)+'_gain'+str(place_cell.gain)+'.hkl', mode='w')
                        # elif axNum == 0:
                        #     hickle.dump(spike_places, '/Users/haasolivia/Desktop/plots/10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_spike_places_gain'+str(place_cell.gain)+'.hkl', mode='w')
                        #     hickle.dump(spike_phases, '/Users/haasolivia/Desktop/plots/10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_spike_phases_gain'+str(place_cell.gain)+'.hkl', mode='w')
                        #     hickle.dump(spike_times, '/Users/haasolivia/Desktop/plots/10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_spike_times_gain'+str(place_cell.gain)+'.hkl', mode='w')
                    fig, ax_comp, ax_pf, ax_traj = signale.place_cells.phasePlot(run_pc.spike_times, run_pc.spike_phases,
                        run_pc.spike_places, trajs[i], mutterTraj=mutterTraj, gc=gc, GetPlaceField=GetPlaceField,
                        wHists=w_hists, wTraj=w_traj, fig=fig, ax=axe, labely=True, limx=limx, limy=limy, title=dummy_titles[i])
                else:
                    fig, ax_comp, ax_pf, ax_traj = signale.place_cells.phasePlot(pc.spike_times, pc.spike_phases,
                        pc.spike_places, trajs[i], mutterTraj=mutterTraj, gc=gc, GetPlaceField=GetPlaceField, fig=fig,
                        ax=axe, labely=True, limx=limx, limy=limy, color=grau2, wTraj='full')

                    fig, ax_comp, ax_pf, ax_traj = signale.place_cells.phasePlot(run_pc.spike_times, run_pc.spike_phases,
                        run_pc.spike_places, trajs[i], mutterTraj=mutterTraj, gc=gc, GetPlaceField=GetPlaceField, fig=fig,
                        ax=axe, labely=True, limx=limx, limy=limy, title=dummy_titles[i])

                if i and direction is None:
                    axe.set_ylabel('')
                    axe.set_yticklabels([])

                if gc is True:
                    axes.append(ax_comp)

            # getting place field boundaries and in field spikes and plot spikes as heat plot
            if GetPlaceField:
                if pf_xlim is False:

                    print 'INFO: getting place field'
                    if i == 0:
                        place_cell.run_pc.getPlaceField(xbin_only=xbin_only)

                    # important to calculate the place field borders for the spectific spiketrain:
                    run_pc.getPlaceField(trajs[i], xbin_only=xbin_only, gain=place_cell.gain)

                print 'INFO: getting infield spikes'
                place_cell.run_pc.get_inFieldSpikes(pf_xlim=pf_xlim, xbin_only=xbin_only)

                if fig is not False and ax_pf is not None:
                    run_pc.plotPlaceField(trajs[i], fig=fig, ax=ax_pf, show_contour=False, show_limits=False, limx=limx,
                                          limy=limy, show_colorbar=False, xbin_only=xbin_only, shading='flat', gain=place_cell.gain)
                    pos_pf = ax_pf.get_position().bounds

                    # maximal FR in plot shows the maximal FR of the entire space, not only the place field!
                    if dummy_titles[i] is not '':
                        # fig.text(pos_pf[0]+pos_pf[2]+.01, pos_pf[1]+.01, str(numpy.round(run_pc.place_field.f_max, 1))+
                        #          ' Hz', fontsize=10)

                        # Get indices for pf limits within FR_distribution:

                        pf_limit_args = [signale.tools.findNearest(run_pc.place_field.x[0], run_pc.place_field_xbin.x0)[0],
                                         signale.tools.findNearest(run_pc.place_field.x[0], run_pc.place_field_xbin.x1)[0]]
                        # FR_distribution within pf:
                        FR_2d_max_idxx = numpy.nanargmax(numpy.array([numpy.nanmax(run_pc.place_field.f[:, xpf]) for
                                         xpf in numpy.arange(pf_limit_args[0], pf_limit_args[1]+1)])) + pf_limit_args[0]

                        FR_2d_max_idxy = numpy.nanargmax(run_pc.place_field.f[:, FR_2d_max_idxx])
                        FR_2d_max = run_pc.place_field.f[FR_2d_max_idxy][FR_2d_max_idxx]

                        fig.text(pos_pf[0]+pos_pf[2]+.01, pos_pf[1]+.01, str(numpy.round(FR_2d_max, 1))+
                                 ' Hz', fontsize=10)

                # getting all (not just in place field!) x values and their firing rates
                if gc is True:
                    firing_rate_maxima = []
                    # firing_rate_local_ysum = []

                    if xbin_only:
                        place_field, place_field_ysum, occupancy_probability_ysum, spike_count_xbin = \
                            run_pc.getPlaceField(trajs[i], xbin_only=xbin_only, gain=place_cell.gain)
                    else:
                        place_field = run_pc.getPlaceField(trajs[i], gain=place_cell.gain)

                    # getting the 2d FR maximum for each x value _______________________
                    x_places = place_field.x[0]

                    for p in numpy.arange(len(x_places)):
                        firing_rate_maxima.append(numpy.nanmax(place_field.f[:, p]))
                    firing_rate_distribution.append([x_places, firing_rate_maxima])
                    # __________________________________________________________________

                    if xbin_only:
                        firing_rate_local_ysum = place_field_ysum.f[0]  #  numpy.nansum(place_field.f[:, p])/len(place_field.f[:, p]))
                        firing_rate_distribution_ySum.append([x_places, firing_rate_local_ysum])
                        occupancy_probability_ySum.append(occupancy_probability_ysum)
                        # x_places where padded with 2 extra bins in the beginning and end!
                        if isinstance(spike_count_xbin, int):
                            spike_count_xbin = numpy.array(numpy.zeros(len(x_places[2:-2])))
                        spike_count_ySum.append([x_places[2:-2], spike_count_xbin])

            #--- add position-phase fit (wfit)?
            if wfit:

                i_pc = place_cell.run_pc.inField_pc[i]

                # run_pc.place field is calculated with all spikes and i_pc.place_field is calculated within the
                # run_pc.place field and thus makes it a bit smaller. We want to use the place field found from the
                # full data set and therefore replace i_pc.place_field.x0 and x1 with the run_pc.place_field values!

                if not hasattr(i_pc, 'place_field'):
                    i_pc.getPlaceField(trajs[i], xbin_only=xbin_only, gain=place_cell.gain)
                #
                # print 'Using run_pc place field: ', run_pc.place_field.x0, run_pc.place_field.x1

                if not pf_xlim:
                    if not hasattr(run_pc, 'place_field'):
                        run_pc.getPlaceField(trajs[i], xbin_only=xbin_only, gain=place_cell.gain)  #, index_maxx=FR_2d_max_idxx, index_maxy=FR_2d_max_idxy)

                    if xbin_only:
                        i_pc.place_field.x0 = run_pc.place_field_xbin.x0
                        i_pc.place_field.x1 = run_pc.place_field_xbin.x1
                    else:
                        i_pc.place_field.x0 = run_pc.place_field.x0
                        i_pc.place_field.x1 = run_pc.place_field.x1
                else:
                    i_pc.place_field.x0 = pf_xlim[0]
                    i_pc.place_field.x1 = pf_xlim[1]

                phase_fit_dictio = i_pc.fit_theta_phases()

                if i_pc.spike_times.size:
                    fit_dictionary.append(phase_fit_dictio)
                    phi0 = i_pc.phase_fit_dictio['phi0']
                    aopt = i_pc.phase_fit_dictio['aopt']
                    delta_places = numpy.nanmax(i_pc.spike_places[:, 0]) - numpy.nanmin(i_pc.spike_places[:, 0])
                    delta_places *= .1
                    x = numpy.arange(numpy.nanmin(i_pc.spike_places[:, 0]) - delta_places,
                                     numpy.nanmax(i_pc.spike_places[:, 0]) + delta_places, .01)
                    if pf_xlim:
                        x_pfBound = numpy.arange(pf_xlim[0], pf_xlim[1] + .01, .01)
                        y_left_right = [phi0+aopt*pf_xlim[0], phi0+aopt*pf_xlim[1]]
                    else:
                        if xbin_only:
                            # pl_field = pc.place_field
                            pl_field = pc.place_field_xbin
                        else:
                            pl_field = pc.place_field
                        x_pfBound = numpy.arange(pl_field.x0, pl_field.x1 + .01, .01)
                        y_left_right = [phi0+aopt*pl_field.x0, phi0+aopt*pl_field.x1]

                    # wfit_xy = [ [allRuns_wfitX, allRuns_wfitY], [rightRuns_wfitX, rightRuns_wfitY], [leftRuns_wfitX,
                    # leftRuns_wfitY] ]
                    y = phi0+aopt*x
                    y_pfBound = phi0+aopt*x_pfBound
                    # y = phi0+offset+aopt*x

                    wfitXy = [x, y]
                    wfit_xy.append(wfitXy)

                    if fig is not False:
                        for offset in numpy.arange(-3600, 3960, 360):
                            if axNum is None or axNum in [0, 1, 2]:
                                c = 1
                            else:
                                c = axNum-2
                            Num = len(custom_plot.pretty_colors_set2)
                            if c >= Num:
                                c -= Num*math.floor(c/Num)
                                c = int(c)

                            if axNum == 0 and 0 < y_left_right[0]+offset < 720 and 0 < y_left_right[1]+offset < 720:
                                pf_phases_left_right.append(numpy.array(y_left_right)+offset)

                            axe.plot(x_pfBound, y_pfBound+offset, '-', color=custom_plot.pretty_colors_set2[c], alpha=.75)
                            if axNum is not None and axNum not in [0, 1, 2]:
                                axes[2].plot(x_pfBound, y_pfBound+offset, '-', color=custom_plot.pretty_colors_set2[c], alpha=.75)

                                if 0 < y_left_right[0]+offset < 720 and 0 < y_left_right[1]+offset < 720:
                                    pf_phases_left_right.append(numpy.array(y_left_right)+offset)

            #--- plot grey lines to mark the place field outlines, which is used for creating the position-phase fit (wfit)
            #if not noSpeck:

            if GetPlaceField and fig is not False:
                grau = numpy.array([1, 1, 1])*.6

                if pf_xlim is False:
                    if xbin_only:
                        # runpl_field = run_pc.place_field  # calculated with 2d binning
                        runpl_field = run_pc.place_field_xbin  # calculated with 1d binning
                    else:
                        runpl_field = run_pc.place_field

                    x_limits = [runpl_field.x0, runpl_field.x1]
                else:
                    x_limits = [pf_xlim[0], pf_xlim[1]]

                pf_limits.append(x_limits)

                #--- show place field borders --------------------------------------

                axe.axvline(x_limits[0], linestyle='--', color=grau, linewidth=1)
                axe.axvline(x_limits[1], linestyle='--', color=grau, linewidth=1)
                if limx:
                    # dont plot xtick borders of the place field, if they would overlap with the limits of the x-axis
                    x_limits = numpy.round(x_limits, 2)
                    limx = list(numpy.round(limx, 2))
                    dummy = numpy.array(numpy.unique(numpy.concatenate((numpy.array(x_limits), numpy.array(limx)))))
                    new_xlimits = list(numpy.delete(dummy, (numpy.where(dummy == limx[0])[0][0], numpy.where(dummy == limx[1])[0][0])))
                    # setting the xticks for the place field borders with a rotation so that the numbers are also visible
                    # for small place fields
                    axe.set_xticks(new_xlimits, minor=True)
                    axe.set_xticklabels(new_xlimits, rotation=60, minor=True)
                    # plotting the limits of the x-axis
                    axe.set_xticks(limx)
                    # making the fontsize of the angles ticks smaller
                    for labels in axe.xaxis.get_minorticklabels():
                        labels.set_fontsize(6)

                if ax_pf is not None:
                    ax_pf.axvline(x_limits[0], linestyle='--', color=grau, linewidth=1)
                    ax_pf.axvline(x_limits[1], linestyle='--', color=grau, linewidth=1)
                    ax_comp.axhline(x_limits[0], linestyle='--', color=grau, linewidth=1)
                    ax_comp.axhline(x_limits[1], linestyle='--', color=grau, linewidth=1)
                if ax_traj is not None:
                    ax_traj.axvline(x_limits[0], linestyle='--', color=grau, linewidth=1)
                    ax_traj.axvline(x_limits[1], linestyle='--', color=grau, linewidth=1)
                grey_line = mlines.Line2D([], [], color=grau, linestyle='--')
                if xbin_only:
                    # runpl_field = run_pc.place_field
                    runpl_field = run_pc.place_field_xbin
                else:
                    runpl_field = run_pc.place_field
                fig.legend(handles=[grey_line], labels=[str(runpl_field.thresh_percent*100)+'% of max FR'], loc=1, prop={'size': 13})

             #--- plot overlaps between right and left run spikes?
            if not noSpeck and i and fig is not False:
                indices = numpy.in1d(place_cell.spiketrains[1].spike_times, place_cell.spiketrains[2].spike_times)
                axe.plot(place_cell.spiketrains[1].spike_places[indices, 0], place_cell.spiketrains[1].spike_phases[indices], 'ro')
                axe.plot(place_cell.spiketrains[1].spike_places[indices, 0], place_cell.spiketrains[1].spike_phases[indices]+360, 'ro')

            #--- text for data set
            if text and fig is not False:
                #if gc is not True:
                pos0 = [.05, .21, .15, .525]
                pos1 = list(pos0)
                pos1[0] += pos1[2]+.09
                pos2 = list(pos1)
                pos2[0] += pos2[2]+.09

                pos = [pos0, pos1, pos2]
                if direction is None:
                    fig.text(pos[i][0], .04,  str(pc.spike_times.size) + ' (' + str(run_pc.spike_times.size)+') (run) spikes',
                        fontsize=fontsize-6,  horizontalalignment='left')

                if i == 0 or text:
                    fig.text(.995, .01, 'v_run >= '+str(place_cell.traj.threshspeed)+' '+place_cell.traj.spaceUnit+'/'+place_cell.traj.timeUnit+', '\
                            +cscName+', '+place_cell.tags['dir'] + ', ' + place_cell.tags['file'],
                            fontsize=fontsize-6,  horizontalalignment='right')

    if gc is True:
        return fig, axes, firing_rate_distribution, firing_rate_distribution_ySum, pf_limits, wfit_xy, \
               pf_phases_left_right, occupancy_probability_ySum, spike_count_ySum, fit_dictionary
    else:
        return fig, axes


def plotSingleRuns(num_singleRuns, st, traj, ax1, ax2):
    for run in range(num_singleRuns):
        # right runs
        i0 = 0
        i1 = 0
        while i0+1 >= i1:
            indices = numpy.where(numpy.diff(traj.rechts_times)>traj.dt*2)[0]
            index = numpy.random.randint(indices.size-1)    # never get the last run (just that I avoid programming for that special case)
            time0 = traj.rechts_times[indices[index]+1]
            time1 = traj.rechts_times[indices[index+1]]

            i0 = signale.findNearest(st.spiketrains[1].spike_times, time0)[0]
            i1 = signale.findNearest(st.spiketrains[1].spike_times, time1)[0]

        places1 = st.spiketrains[1].spike_places[i0:i1, 0]
        if ~(numpy.diff(places1) > 0).all():
            print "WARNING: Something went wrong with the place array for the single run!"
        phases1 = st.spiketrains[1].spike_phases[i0:i1]      # get phase of single run
        for i, p in enumerate(phases1[:-1]):        # make it smooth
            p1 = phases1[i+1]
            indeks = numpy.argmin(numpy.abs([p1-p, p1-360-p, p1+360-p]))
            if indeks == 1:
                phases1[i+1] -= 360
            elif indeks == 2:
                phases1[i+1] += 360
        col = ([.4, .7])[run]*numpy.ones(3)
        ax1.plot(places1, phases1, '-o',\
            color=col, markersize=5, linewidth=2)
        ax1.plot(places1, phases1+360, '-o',\
            color=col, markersize=5, linewidth=2)

        # left runs
        i0 = 0
        i1 = 0
        while i0+1 >= i1:
            indices = numpy.where(numpy.diff(traj.links_times)>traj.dt*2)[0]
            index = numpy.random.randint(indices.size-1)    # never get the last run (just that I avoid programming for that special case)
            time0 = traj.links_times[indices[index]+1]
            time1 = traj.links_times[indices[index+1]]

            i0 = signale.findNearest(st.spiketrains[2].spike_times, time0)[0]
            i1 = signale.findNearest(st.spiketrains[2].spike_times, time1)[0]

        places1 = st.spiketrains[2].spike_places[i0:i1, 0]
        if ~(-numpy.diff(places1)>0).all():
            print "WARNING: Something went wrong with the place array for the single run!"
        phases1 = st.spiketrains[2].spike_phases[i0:i1]      # get phase of single run
        for i in range(phases1.size-1):        # make it smooth
            p = phases1[i]
            p1 = phases1[i+1]
            indeks = numpy.argmin(numpy.abs([p1-p, p1-360-p, p1+360-p]))
            if indeks == 1:
                phases1[i+1] -= 360
            elif indeks == 2:
                phases1[i+1] += 360

        col = ([.4, .7])[run]*numpy.ones(3)
        ax2.plot(places1, phases1, '-o',\
            color=col, markersize=5, linewidth=2)
        ax2.plot(places1, phases1+360, '-o',\
            color=col, markersize=5, linewidth=2)


def isiHist(fig, st):
    pos = [.8, .73, .175, .2]
    ax = fig.add_axes(pos)
    if st.spiketrains[1].spike_times.size > 2:
        col = numpy.ones(3)*.3
        ax.hist(numpy.diff(st.spiketrains[1].spike_times), 30, range=(0, .2),
                facecolor=col, edgecolor=col, histtype='step', linewidth=2)
    if st.spiketrains[2].spike_times.size > 2:
        col = numpy.ones(3)*.7
        ax.hist(numpy.diff(st.spiketrains[2].spike_times), 30, range=(0, .2),
                facecolor=col, edgecolor=col, histtype='step', linewidth=2)
    ax.set_xticks(numpy.arange(0, .21, .1))
    ax.set_xlabel('Interspike interval (s)')
    ax.yaxis.set_visible(False)
    custom_plot.turnOffAxes(ax, spines=['left', 'right', 'top'])
    ax.legend(['right', 'left'], bbox_to_anchor=(0., 1., 1., .102), loc=3,
       ncol=2, mode="expand", borderaxespad=0., frameon=False, fontsize=fontsize-4, markerscale=.05)
    return pos


def cscFft(fig, ax, csc, thetaRange):

    csc.fft_plot(fig, ax)

    if not hasattr(csc, 'sp_filtered'):
        csc.filter(thetaRange[0], thetaRange[1])
        csc.hilbertTransform()

    # hickle.dump(csc.freq, '/Users/haasolivia/Desktop/plots/10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_csc_freq.hkl', mode='w')
    # hickle.dump(csc.spPower, '/Users/haasolivia/Desktop/plots/10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_csc_power.hkl', mode='w')
    ax.plot(csc.freq, 2*csc.sp_filtered*csc.sp_filtered.conj(), 'k', alpha=.75)
    ax.set_xlabel('LFP frequency (Hz)')

    # mark maximum
    i1 = numpy.where(csc.freq >= thetaRange[0])[0]
    i2 = numpy.where(csc.freq <= thetaRange[1])[0]
    indices = numpy.intersect1d(i1, i2)
    i = numpy.argmax(csc.spPower[indices]) + indices.min()
    ax.plot([csc.freq[i], csc.freq[i]], [0, csc.spPower.max()], 'b--', alpha=.25, linewidth=2) #csc.freq[i] = freq with maximal power in chosen thetaRange
    ax.text(csc.freq[i], csc.spPower.max(), numpy.round(csc.freq[i], 1), ha='center')


    #ax.set_xticks(numpy.arange(0, 31, 10))
    ax.set_xticks([0]+thetaRange+[15])
    ax.set_xlim(0, 15)
    ax.yaxis.set_visible(False)
    custom_plot.turnOffAxes(ax, spines=['left', 'right', 'top'])


def setFigTitles(fig, metadata):
    if expType == 'vr':
        fig.suptitle('Animal:' + str(metadata.get('animal')) + ', Tetrode depth:' + str(metadata.get('depth')) +
                     ' $\mu m$' + ', VR gain:' + str(gain))
    else:
        fig.suptitle('Animal:' + str(metadata.get('animal')) + ', Tetrode depth:' + str(metadata.get('depth')) +
                     ' $\mu m$')
    return fig

###################################################### main

if __name__ == "__main__":

    ###################################################### initialize logging

    try:
        logging
    except NameError:
        import logging
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-2s -- %(levelname)-2s: %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename='spikesPhase.log',
                            filemode='w')
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-2s -- %(levelname)-2s: %(message)s')   # set a format which is simpler for console use
        console.setFormatter(formatter)                                             # tell the handler to use this format
        logging.getLogger('').addHandler(console)                                   # add the handler to the root logger
        logger1 = logging.getLogger('spikesPhase.py')
    else:
        print 'Logger already loaded'

    ###################################################### initialize plotting

    colors = custom_plot.colors
    grau = numpy.array([1, 1, 1])*.6
    grau2 = numpy.array([1, 1, 1])*.85

    Bildformat='pdf'

    fontsize=14.0

    mpl.rcParams['lines.markersize'] = 3
    mpl.rcParams['font.size'] = fontsize

    ###################################################### commandline paramters

    dummy = sys.argv[1]				# second argument should be the name of the folder to load
    folderName = dummy.split('\\')[0]
    for d in dummy.split('\\')[1:]:
        folderName += '/'+d

    # parameters
    limx = False
    noSpeck = False
    wfit = False
    showFigs = True
    saveFigs = True
    useRecommended = False
    expType = 'real'
    num_singleRuns = 0
    saveAna = False
    fullblown = False
    xbin_only = False
    for argv in sys.argv[2:]:
        if argv == 'noShow':
            showFigs = False			# show pics
        if argv == 'saveFigs':
            saveFigs = True				# save pics
        if argv == 'saveAna':
            saveAna = True				# save analysis
        if argv.startswith('csc:'):
            csc = argv.split(':')[-1]   # csc file to load
        if argv.startswith('tt:'):                      # Tetrode files to load, write e.g. as TT:['TT2_01.t']
            tt = argv.split('tt:')[1].strip('[').strip(']')
            tt = [s for s in tt.split(',')]
        if argv == 'noSpeck':
            noSpeck = True
        if argv == 'wfit':
            wfit = True
        if argv=='fullblown':
            fullblown = True
        if argv == 'useRecommended':
            useRecommended = True
        if argv == 'limx':
            limx = True				                    # limit x axis to place field
        if argv.startswith('showSingleRuns:'):
            num_singleRuns = int(argv.split(':')[-1])
        if argv.startswith('expType:'):
            expType = argv.split(':')[-1]
        if argv.startswith('thetaRange:'):
            thetaRange = argv.split('thetaRange:')[1].strip('[').strip(']')   #write into terminal e.g. as thetaRange:'[6, 10]'
            thetaRange = [float(thetaRange.split(',')[0]), float(thetaRange.split(',')[1])]
        if argv.startswith('threshspeed:'):
            threshspeed = float(argv.split('threshspeed:')[-1])

    ###################################################### initialization

    # initialize in order to make them available globally
    spikes = []
    ID = -1
    stList = signale.placeCellList(t_start=None, t_stop=None, dims=[2])
    eventData = None
    traj = None
    gain = None

    cscID = -1
    cscList = signale.NeuralynxCSCList()

    loadedSomething = False
    cwd = os.getcwd()

    if expType == 'vr':
        Subfolders = [x[0] for x in os.walk(folderName)]
        if os.path.isfile(Subfolders[1]+'/'+'linearMaze_position.traj'):
            traj_metadata=signale.io._read_metadata(Subfolders[1]+'/'+'linearMaze_position.traj', showHeader=False) # get metadata from linearMaze_position.traj
            if traj_metadata.has_key('gain'):
                gain = traj_metadata['gain']

    # get parameters

    parameters = {'csc': 'CSC1.ncs', 'tt': '', 'thetaRange': [6, 10], 'threshspeed': 0.1, 'animal': '', 'depth': 0}
    if useRecommended:
        fileName = os.path.normpath(folderName)+'/metadata.dat'
    else:
        fileName = ''
    dictio, metadata = signale.get_metadata(fileName, parameters, locals())
    locals().update(dictio)

    # file endings
    cscName = csc
    if not cscName.endswith('.ncs'):
        cscName += '.ncs'
    TTName = tt
    if TTName == ['']:
        TTName = ''
    else:
        for i, tt in enumerate(TTName):
            if not tt.endswith('.t'):
                TTName[i] += '.t'

    ###################################################### load data

    if os.path.isdir(folderName):
        getData(folderName)
    elif os.path.isfile(folderName):
        sys.exit('Point to a folder not a single file.')
    else:
        sys.exit('Folder or data name does not exist.')
    os.chdir(cwd)

    if not ID+1:
        sys.exit('The folders do not contain the tetrode data (t files)!')

    # real
    elif expType == 'real':
        # set begining to zero time
        #for st in stList:
        #    st.spike_times -= st.t_start
        #stList._spikezugList__recalc_startstop()

        # cut away all earlier spikes
        #stList = stList.time_slice(0., stList.t_stop)

        # change to seconds since trajectories are also stored in seconds
        stList._placeCellList__recalc_startstop()
        stList.changeTimeUnit('s')
        cscList.changeTimeUnit('s')

        #stList = stList.time_slice(0, 20000)      # reduce data size a bit

    # vr
    elif expType == 'vr':

        # set beginning to zero time
    ##    eventData.times -= eventData.t_start
    ##    for st in stList:
    ##        st.spike_times -= eventData.t_start
    ##    stList._spikezugList__recalc_startstop()
    ##
    ##    for csc in cscList:
    ##        csc.times -= eventData.t_start


        # cut away all earlier spikes
        #stList = stList.time_slice(0., stList.t_stop)

        # change to seconds since trajectories are also stored in seconds
        eventData.changeTimeUnit('s')
        stList.changeTimeUnit('s')
        cscList.changeTimeUnit('s')

        eventData, traj = trajectory.align(eventData, traj, display=True)

        traj.times = eventData.times

        #stList = stList.time_slice(0, 20000)      # reduce data size a bit


    ###################################################### execute main functions

    traj.getTrajDimensions()
    prepareData()
    
    for zaehler, st in enumerate(stList):

        # prepare figure
        fig = pl.figure(figsize=(12, 7.5))
        pos0 = [.06, .15, .15, .45]
        pos1 = list(pos0)
        pos1[0] += pos1[2]+.09
        pos2 = list(pos1)
        pos2[0] += pos2[2]+.09
        ax0 = fig.add_axes(pos0)
        ax1 = fig.add_axes(pos1)
        ax2 = fig.add_axes(pos2)

        # prepare data
        csc = cscList[0]
        traj.threshspeed = threshspeed
        st.traj = traj

        # prepare list with data that should be plotted and plot the data
        fig, axes = prepareAndPlotPooledPhases(st, csc, cscName, fig, [ax0, ax1, ax2], limx, fontsize, wfit=wfit, noSpeck=noSpeck)

        #--- single run
        plotSingleRuns(num_singleRuns, st, traj, ax1, ax2)

        #--- isi hist
        pos = isiHist(fig, st)

        #--- spectrogram
        if fullblown:
            Pxx, freqs, t0 = csc.spectrogram(minFreq=0, maxFreq=15, windowSize=2**16) #8192
            Pxx = signale.blur_image(Pxx, 10)       # smooth Pxx a bit

            posSpect = list(pos)
            posSpect[1] = .4
            ax = fig.add_axes(posSpect)

            ax.pcolormesh(t0, freqs, Pxx, vmin=Pxx.min(), vmax=numpy.percentile(Pxx, 99))
            ax.set_yticks([0]+thetaRange+[15])
            ax.set_xlim(t0.min(), t0.max())
            ax.set_ylim(freqs.min(), freqs.max())
            ax.set_ylabel('LFP frequency (Hz)')
            custom_plot.turnOffAxes(ax, spines=['bottom', 'right', 'top'])

        #--- csc fft
        pos[1] = pos0[1]
        ax = fig.add_axes(pos)
        cscFft(fig=fig, ax=ax, csc=csc, thetaRange=thetaRange)

        #set main title as animal name and depth
        fig = setFigTitles(fig, metadata)

        if saveFigs:
            if fullblown:
                name = '_spikePhase.png'
            else:
                name = '_spikePhase_reduced.png'
            if limx:
                name = '_spikePhase_limx.png'
            fig.savefig(stList.tags[zaehler]['dir']+'/'+stList.tags[zaehler]['file'].split('.')[0]+name,\
                format='png')  #save figure , dpi=300
    ##        fig.savefig(stList.tags[zaehler]['dir']+stList.tags[zaehler]['file'].split('.')[0]+'_spikePhase.pdf',\
    ##            format='pdf')  #save figure
        if not showFigs:
            pl.ioff()
            pl.close(fig)
        else:
            pl.show()

        #--- save data
        if saveAna:
            st.save(xbin_only=xbin_only)
        
    

    ###################################################### finishing

    if not loadedSomething:
        sys.exit('The folders do not contain csc data!')
    
    if showFigs:
        pl.show()
