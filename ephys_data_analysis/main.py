"""
For plotting spike data from a gainChanger maze.
"""

__author__ = "Olivia Haas"
__version__ = "2.0, April 2015"

# python modules
import sys
import os
import logging
import re
import ast

# add additional custom paths
extraPaths = [os.path.join(os.path.abspath(os.path.dirname(__file__)), '../scripts'),
              os.path.join(os.path.abspath(os.path.dirname(__file__)), '/opt/anaconda/bin/python'),
              os.path.join(os.path.abspath(os.path.dirname(__file__)), '/opt/anaconda/pkgs')]

for p in extraPaths:
    if not sys.path.count(p):
        sys.path.insert(1, p)

# plotting modules
import matplotlib as mpl
# mpl.use('Agg')
# mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as pl

# other modules
import numpy
import math
import copy
import hickle
import scipy.ndimage.measurements as image_m

# custom made modules
import trajectory
import custom_plot
import signale
import spikesPhase
import spikesPlace

###################################################### functions


def getData(folderName, cscName, TTName):
    global cscID, cscList, stList, loadedSomething, item
    global ID, traj, rewards_traj, events, eventData, events_fileName, stimuli, stimuli_folder, \
        stimuli_fileName, main_folder, loaded_cleaned_events, loaded_cleaned_stimuli, hs, rz

    spikes = []
    ID = -1
    cscID = -1
    cscList = signale.NeuralynxCSCList()
    stList = signale.placeCellList(t_start=None, t_stop=None, dims=[2])
    eventData = []
    traj = None
    rewards_traj = None
    events = None
    events_fileName = None
    stimuli = None
    stimuli_fileName = None
    stimuli_folder = None
    main_folder = None
    loaded_cleaned_events = False
    loaded_cleaned_stimuli = False
    hs = []
    rz = []
    loadedSomething = False
    cwd = os.getcwd()
    L = 0

    if os.path.isdir(folderName):
        dirList = os.listdir(folderName)
        os.chdir(folderName)
    else:
        dirList = [folderName]

    for item in dirList:
        if os.path.isfile(item):
            if cscName and item.endswith(cscName):  # not any([item.find(str(s))+1 for s in excludeCSCs]):
                print 'loading', item, 'from folder: '+folderName
                main_folder = folderName
                loadedSomething = True
                csc = signale.load_ncsFile(item, showHeader=False)
                cscID += 1
                cscList.append(cscID, csc)
                cscList.addTags(cscID, file=item, dir=folderName)
            elif (TTName.__class__ == list and item in TTName) or \
                    (TTName.__class__ == str and item.endswith('.t')):
                print 'loading', item, 'from folder: '+folderName
                spikes = signale.load_tFile_place_cell(item, showHeader=False)
                ID += 1
                stList.__setitem__(ID, spikes)
                stList.addTags(ID, file=item, dir=folderName)
            elif item.endswith('.nev'):
                print 'loading eventData: ', item, 'from folder: '+folderName
                eventData = signale.load_nevFile(item, showHeader=False)

        elif os.path.isdir(item):
            dirList1 = os.listdir(item)
            os.chdir(item)
            for item1 in dirList1:
                if item1.endswith('events_position_cleaned.traj'):
                        print 'loading cleaned events: ', item1, 'from folder: '+folderName+item
                        events_fileName = item1
                        events = trajectory.load_rewardTrajectory(item1)
                        loaded_cleaned_events = True
                elif item1.endswith('stimuli_cleaned.tsv'):
                    print 'loading cleaned stimuli: ', item1, 'from folder: '+folderName+item
                    stimuli_folder = folderName
                    stimuli_fileName = item1
                    stimuli = trajectory.load_params(item1)
                    loaded_cleaned_stimuli = True
            for item1 in dirList1:
                if os.path.isfile(item1):
                    if item1.endswith('properties.tsv'):
                        print 'loading properties: ', item1, 'from folder: '+folderName+item
                        properties_file = open(item1)
                        properties_lines = properties_file.readlines()
                        for i in numpy.arange(len(properties_lines)):
                            if 'hotspot' in properties_lines[i]:
                                s = properties_lines[i].split('pos:')[1].strip('[').strip(']')[:-3]
                                hs.append([float(s.split(',')[0]), float(s.split(',')[1])])
                            elif 'rewardspot' in properties_lines[i]:
                                s = properties_lines[i].split('pos:')[1].strip('[').strip(']')[:-3]
                                rz.append([float(s.split(',')[0]), float(s.split(',')[1])])
                    elif item1.endswith('.traj') and item1.find('position')+1 \
                            and not item1.find('collisions_position')+1 \
                            and not item1.find('rewardsVisited_position')+1 and not item1.find('events_position')+1:
                        print 'loading trajectory: ', item1, 'from folder: '+folderName+item
                        traj = trajectory.load_trajectory(item1, showHeader=False)
                    elif item1.endswith('rewardsVisited_position.traj'):
                        print 'loading rewards trajectory: ', item1, 'from folder: '+folderName+item
                        rewards_traj = trajectory.load_rewardTrajectory(item1)
                    elif not loaded_cleaned_events and item1.endswith('events_position.traj'):
                        print 'loading uncleaned events: ', item1, 'from folder: '+folderName+item
                        events_fileName = item1
                        events = trajectory.load_rewardTrajectory(item1)
                    elif not loaded_cleaned_stimuli and item1.endswith('.tsv') and item1.find('stimuli')+1:
                        print 'loading uncleaned stimuli: ', item1, 'from folder: '+folderName+item
                        stimuli_folder = item
                        stimuli_fileName = item1
                        stimuli = trajectory.load_params(item1)
            os.chdir('..')

    return spikes, ID, cscID, cscList, stList, eventData, traj, rewards_traj, events, events_fileName, stimuli, \
           stimuli_fileName, stimuli_folder, main_folder, loaded_cleaned_events, loaded_cleaned_stimuli, hs, rz, \
           loadedSomething, cwd


def prepareData():

    for csc in cscList:
        csc.filter(thetaRange[0], thetaRange[1])
        csc.hilbertTransform()


def findCycleFrequencies(lfp, threshMin=0.0, threshMax=100.0, GC_element=None):
    if not hasattr(lfp, 'hilbertPhase'):
        lfp.filter(thetaRange[0], thetaRange[1])
        lfp.hilbertTransform()
    Maxima = signale.findMaximaMinima(lfp.hilbertPhase, findMinima=0)
    MaximaIndices = Maxima['maxima_indices']
    MaximaTimes = lfp.times[MaximaIndices]
    MaximaSignals = lfp.signal[MaximaIndices]
    diffTimes = numpy.diff(MaximaTimes)

    # delta
    # csc.filter is using csc.sp -> one-dimensional discrete Fourier Transform for real input signal
    deltaRange = [2, 4]
    lfp.filter(deltaRange[0], deltaRange[1])
    lfp.hilbertTransform()
    deltaAmp = lfp.hilbertAbsolute.mean()   # float

    # theta
    lfp.filter(thetaRange[0], thetaRange[1])
    lfp.hilbertTransform()
    thetaAmp = lfp.hilbertAbsolute.mean()   # float

    # theta_delta_ratio = numpy.round(thetaAmp/deltaAmp, 2)

    if lfp.timeUnit == 'ms':
        print 'Calculating with csc times in s'
        diffTimes = diffTimes/1000.0
    freq = 1.0/diffTimes

    hilbert_maxIdx = signale.findMaximaMinima(lfp.analyticalTrain, findMinima=0)
    hilbert_max = lfp.analyticalTrain[hilbert_maxIdx['maxima_indices']][1:]

    # calculate where hilbert amplitude drops to 10 percent of the mean and delete all corresponding frequencies
    hil_drop = numpy.where(hilbert_max < numpy.mean(hilbert_max)/10)[0]

    discard = numpy.concatenate((numpy.where(freq < threshMin)[0], numpy.where(freq > threshMax)[0],
                                 hil_drop))

    filteredFreq = numpy.delete(freq, discard)

    if GC_element:
        MaximaTraj_x = []

        for time in MaximaTimes:
            (time_idx, traj_time) = signale.tools.findNearest(array=GC_element.placeCell.traj.times, value=time)
            if GC_element.placeCell.traj.spaceUnit != 'm':
                print 'placeCell.traj.spaceUnit is in ', GC_element.placeCell.traj.spaceUnit
                print 'Program aborted!'
                sys.exit()
            MaximaTraj_x.append(GC_element.placeCell.traj.places[:, 0][time_idx])

        diffPlaces = abs(numpy.diff(MaximaTraj_x))
        speeds_real = diffPlaces/diffTimes
        speeds_virtual = speeds_real*GC_element.gain_in

        # if gain_normalised:
        #     real_speed_thresh = GC_element.pc_gain_in.traj.threshspeed
        # else:
        #     real_speed_thresh = GC_element.pc_gain_in.traj.threshspeed/GC_element.gain_in
        real_speed_thresh = GC_element.pc_gain_in.traj.threshspeed

        # delete speed values which were deleted from frequencies and when running speed was under threshold
        speeds_real = numpy.delete(speeds_real, discard)
        below_speed_thresh_idx = numpy.where(speeds_real < real_speed_thresh)[0]
        speeds_real = numpy.delete(speeds_real, below_speed_thresh_idx)

        speeds_virtual = numpy.delete(speeds_virtual, discard)
        speeds_virtual = numpy.delete(speeds_virtual, below_speed_thresh_idx)

        # deleting frequencies where the running speed was under the threshold
        filteredFreq = numpy.delete(filteredFreq, below_speed_thresh_idx)

        return filteredFreq, speeds_real, speeds_virtual, MaximaSignals, deltaAmp, thetaAmp

    return filteredFreq


def plotCycleFrequencies(fig, ax, xvalues, plotNum=None, frequencies=None, lfp=None, threshMin=None, threshMax=None, labelx=None, labely=None):
    if not frequencies:
        frequencies = findCycleFrequencies(lfp, threshMin, threshMax)
    if not plotNum:
        color = [0.6, 0.6, 0.6]
    else:
        color = list(numpy.array([1.0,1.0,1.0])-(plotNum/6.))
    xvalues = numpy.ones(len(frequencies))*xvalues
    ax.plot(xvalues, frequencies, color=color, ls='*', marker='.', markersize=4)
    if labelx:    
        ax.set_xlabel(labelx)
    if labely:
        ax.set_ylabel(labely, fontsize=14)
    return fig, ax


def plotCscSlices(GC, fig, ax):
    for id, gc in enumerate(GC):
        s_in = gc.csc_gain_in.signal.copy()
        s_out = gc.csc_gain_middle.signal.copy()
        s = gc.csc.signal.copy()
        s_in -= s.mean()
        s_out -= s.mean()
        #min = numpy.min(1e100, csc.signal.min())
        s_max = numpy.max(-1e-100, csc.signal.max())
        s_in /= s_max*.8
        s_out /= s_max*.8
        #normalize time axis
        s_in_time_axis = s_in.time_axis()-numpy.min(s_in.time_axis())
        gainChangeTime = numpy.max(s_in_time_axis)
        s_out_time_axis = s_out.time_axis()-numpy.min(s_out.time_axis())+gainChangeTime
        pl.plot(s_in_time_axis, s_in+id, '-', linewidth=1,
                color=custom_plot.pretty_colors_set2[id % custom_plot.pretty_colors_set2.__len__()])
        pl.plot(s_out_time_axis, s_out+id, '-', linewidth=1,
                color=custom_plot.pretty_colors_set2[id+1 % custom_plot.pretty_colors_set2.__len__()])
        ax.set_ylabel(cscName)
        ax.set_xlabel('Normalized time (', s.timeUnit, ')')


def getGainsAndTheirCscs(gain_in, gain_middle, GC):
    # find gains and their lfp which fit the input gain or output gain of the considered gain change
    # for the second frequency plot in pooled plot
    gains_in = []
    gains_middle = []
    cscs_in = []
    cscs_middle = []
    for s in numpy.arange(len(stimuli.times)-1):
        if GC[s].gain_in == float(gain_in): 
            gains_in.append(GC[s].gain_in)
            cscs_in.append(GC[s].csc_gain_in)
        if GC[s].gain_in == float(gain_middle):
            gains_middle.append(GC[s].gain_in)
            cscs_middle.append(GC[s].csc_gain_in)
        # for the last stimulus pair look at gain_in and gain_middle, otherwise the last gain would not be included
        if s == len(stimuli.times)-2:
            if GC[s].gain_middle == float(gain_in):
                gains_in.append(GC[s].gain_middle)
                cscs_in.append(GC[s].csc_gain_middle)
            if GC[s].gain_middle == float(gain_middle):                       
                gains_middle.append(GC[s].gain_middle)
                cscs_middle.append(GC[s].csc_gain_middle)
    return gains_in, gains_middle, cscs_in, cscs_middle


def getSingleRunAxNum(inField_pc, run_traj, minSpikeNum_perRun=4, minThetaCycles=4,
                      plot=False, gc_indexes=False, pf_x=False):

    singleTrialNum = []
    traj_in_pf_and_gain = []
    for runs in [0, 1, 2]:
        singleTrials = []

        inField_times = inField_pc[runs].spike_times

        if hasattr(inField_pc[runs], 'spike_places'):
            if len(inField_pc[runs].spike_places):
                inField_places = inField_pc[runs].spike_places[:, 0]
            else:
                inField_places = []
        if runs == 0:
            direction_trajTimes = run_traj.times
            direction_trajPlaces = run_traj.places[:, 0]
        elif runs == 1:
            direction_trajTimes = run_traj.rightward_traj.times
            direction_trajPlaces = run_traj.rightward_traj.places[:, 0]
        elif runs == 2:
            direction_trajTimes = run_traj.leftward_traj.times
            direction_trajPlaces = run_traj.leftward_traj.places[:, 0]
        else:
            direction_trajTimes = []
            direction_trajPlaces = []
            if runs is None:
                print 'ERROR: run direction for single run plots has to be specified in the terminal!'
                sys.exit()

        if plot:
            fiig, axx = pl.subplots()
            axx.plot(direction_trajTimes, direction_trajPlaces, color=numpy.ones(3)*.5)
            axx.set_title('Place field spikes within running trajectory')
            axx.set_xlabel('Time (s)')
            axx.set_ylabel('X position (m)')


        # find out to how many trajectories these inField_times belong

        cut = numpy.append(numpy.append(-1, numpy.where(numpy.isnan(direction_trajTimes))[0]), len(direction_trajTimes))

        if pf_x:
            if gain_normalised and Grun == 1.5 and pf_x[runs][0] == 0 and pf_x[runs][1] == 1.5:
                pf_x[runs] = [0.5, 0.5]
            direction_trajPlaces_nonan = direction_trajPlaces[numpy.logical_not(numpy.isnan(direction_trajPlaces))]

            if runs == 0:
                border = 0
                devide = 1.
            else:
                border = runs-1
                devide = 2.
            diff = direction_trajPlaces_nonan-pf_x[runs][border]
            crosses = numpy.array((numpy.diff((numpy.sign(diff) == 0)*1+numpy.sign(diff)) != 0)*1)

            pfEntry_num_runs = numpy.ceil(sum(crosses)/devide)
            traj_in_pf_and_gain.append(pfEntry_num_runs)

        hilbert_zero_times = numpy.append(numpy.append(cscList[0].times[0], numpy.repeat(cscList[0].times[
                                     numpy.where(abs(numpy.diff(cscList[0].hilbertPhase)) > 300)[0]+1], 2)),
                                                  cscList[0].times[-1])

        hilbert_cycles = zip(hilbert_zero_times[::2], hilbert_zero_times[1::2])

        for i in numpy.arange(len(cut)-1):
            if direction_trajTimes[cut[i]+1: cut[i+1]].size and not numpy.isnan(direction_trajTimes[cut[i]+1: cut[i+1]][0]):
                time_range = direction_trajTimes[cut[i]+1:cut[i+1]]
                # spikes within traj piece

                traj_PFspikes = numpy.logical_and(inField_times >= time_range[0], inField_times <= time_range[-1])

                PFspike_times = inField_times[traj_PFspikes]

                cycle_nums = []

                for st in PFspike_times:
                    cycle_index = numpy.where([st >= hilbert_cycles[h][0] and st <= hilbert_cycles[h][1] for h in
                                               numpy.arange(len(hilbert_cycles))])[0]
                    # if spike is exactly on border of two hilbert cycles and has therefore two cycle indexes,
                    # take the first one! when there is one cycle index take the zero entrance of the array:
                    if len(cycle_index) >= 1:
                        cycle_nums.append(cycle_index[0])

                if numpy.sum(traj_PFspikes) >= minSpikeNum_perRun and \
                        len(numpy.unique(cycle_nums)) >= minThetaCycles:
                    if gc_indexes:
                        for g in gc_indexes:
                            times = GC[g].placeCell.traj.times
                            if numpy.sum(numpy.logical_and(times > time_range[0], times < time_range[-1])) and g not in singleTrials:
                                singleTrials.append(g)
                    id = numpy.where(traj_PFspikes)[0]
                    if plot:
                        if hasattr(inField_places[runs], 'spike_places') and len(inField_places):
                            axx.plot(inField_times[id], inField_places[id], 'o', markerfacecolor='r', markeredgecolor='r')
                        else:
                            print 'WARNING: There are no place field spikes!'
        singleTrialNum.append(sorted(singleTrials))  # [[GainIndexes_allRuns], [GI_rightRuns], [GI_leftRuns]]

    return singleTrialNum, traj_in_pf_and_gain


def getFigure(AxesNum=None):

    if plotPlace:
        fig = pl.figure(figsize=(12, 7.5))  # figsize = (width, height)
    # figure for pooled phase plot
    elif not AxesNum:
        fig = pl.figure(figsize=(12, 6))
    elif AxesNum <= 5:
        fig = pl.figure(figsize=(17, 10))   # 2 rows = 1-summary plots, 2-single runs (max 5 per row), without plots above phase plot
    elif 5 < AxesNum <= 10:
        fig = pl.figure(figsize=(17, 14))
    elif 10 < AxesNum <= 15:
        fig = pl.figure(figsize=(17, 18))
    elif 15 < AxesNum <= 20:
        fig = pl.figure(figsize=(17, 22))
    else:
        fig = pl.figure(figsize=(17, 30))
    return fig


def addAxesPositions(fig, Num=None, pooled=False):

    # calculate number of single trial plot rows (one row has 5 plots)

    axes = []
    all_runs = [.06, .15, .15, .45]  # [left, bottom, width, height]
    right = list(all_runs)
    right[0] += right[2]+.09
    left = list(right)
    left[0] += left[2]+.09
    fft = [0.82, 0.21, 0.15, 0.25]
    pos = [all_runs, right, left, fft]
    if pooled or Num == 0:
        isi = [.84, .675, .13, .25]
        isi2 = [.82, .675, .15, .25]
        pos = tuple(pos)
        for p in numpy.arange(len(pos)):
                axes.append(fig.add_axes(pos[p]))
    else:
        rows = math.ceil(Num/5.)
        d = [1.]
        if int(rows) > 0:
            for r in numpy.arange(int(rows)+1):
                if r == 0:
                    pass
                else:
                    d.append(d[-1]+0.5)
        #d = [1., 1.5, 2., 2.5, 3.]
        devider = d[int(rows)]

        first = [.05, .16/devider, .15, .4/devider] #rows]  # [left, bottom, width, height]
        second = list(first)
        second[0] += first[2]+.045
        third = list(second)
        third[0] += second[2]+.045
        fourth = list(third)
        fourth[0] += fourth[2]+.045
        fifth = list(fourth)
        fifth[0] += fifth[2]+.045

        middle_row_plots = [first, second, third, fourth, fifth]

        fft = [fourth[0], .1, fourth[2], 0.25/devider]  # rows]
        first_row_plots = [copy.deepcopy(first), copy.deepcopy(second), copy.deepcopy(third), fft]

        # make single trial axes
        # if Num == 0:
        #     pl.close(fig)
        #     print 'Figure closed because there are no gain changes of type ', Title

        # first row of single trial plots (directly under the top summary plots)
        # else:
        plot_height = .5/devider      #rows
        if int(rows) == 0:
            row_plots = []
            for row in numpy.arange(len(first_row_plots)):
                row_plots.append(copy.deepcopy(first_row_plots[row]))
                # if Num == 0:
                #     axes.append(fig.add_axes(first_row_plots[row]))
        else:
            for r in numpy.arange(0, rows+1):
                if r == 0:
                    row_plots = first_row_plots
                elif r == int(rows):
                    last_num = Num - (5*math.floor(Num/5.))
                    row_plots = []
                    if int(last_num) == 0:
                        row_plots = copy.deepcopy(middle_row_plots)
                    else:
                        for plot in numpy.arange(int(last_num)):
                            row_plots.append(copy.deepcopy(middle_row_plots[plot]))
                else:
                    row_plots = copy.deepcopy(middle_row_plots)
                if len(row_plots) != 0:
                    for i in numpy.arange(len(row_plots)):
                        row_plots[i][1] += (rows-r)*plot_height
                        axes.append(fig.add_axes(row_plots[i]))

    return axes


# def plotSingleRunPhases(GC_object, object_num, fig, axes):
#
#     GC_object.plotPhase(fig, axes[object_num])
#     if object_num == len(axes)-1:
#             row = (numpy.ceil((object_num+1)/6.)-1)
#             axes[int(row*6)].set_ylabel('Spike phase (deg)')
#             fig.text(.5, .01, 'Position ('+traj.spaceUnit+')')

#def plotSingleRunPhases(placeCell, fig, axes):


def plotPlaces(GC):

    fig = spikesPlace.plotSpikePlaces(st, noSpeck=noSpeck, text=True, zaehler=zaehler)
    return fig


def gainChangeFreq(fig, ax, GC_object, Title):

    fig, ax = plotCycleFrequencies(fig, ax, xvalues=0, lfp=GC_object.csc, threshMin=thetaRange[0],
                                       threshMax=thetaRange[1], labelx='Gain change',
                                       labely=r'$f_{\theta}$ $(Hz)$')

    ax.hist(findCycleFrequencies(lfp=GC_object.csc, threshMin=thetaRange[0], threshMax=thetaRange[1]),
            orientation='horizontal', normed=0.5, bottom=0.05, color=[0.6, 0.6, 0.6])
    ax.set_xticks([-0.2, 0.0, 0.2, 0.6])
    ax.set_xticklabels(['', '', Title])
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(numpy.around(numpy.arange(start, end, 1), decimals=1))


def individualGainFreq(fig, ax):
    if transitions:  # and z == AxesNum[PT]-1:
        gains_in, gains_middle, cscs_in, cscs_middle = getGainsAndTheirCscs(float(GC_in), float(GC_middle), GC)
        for i in numpy.arange(len(cscs_in)):
            if i == 0:
                fig, ax = plotCycleFrequencies(fig, ax, xvalues=0, lfp=cscs_in[i], threshMin=thetaRange[0],
                                                threshMax=thetaRange[1], labelx='Gain',
                                                labely=r'$f_{\theta}$ $(Hz)$')
                ax.hist(findCycleFrequencies(lfp=cscs_in[i], threshMin=thetaRange[0], threshMax=thetaRange[1]),
                        orientation='horizontal', normed=0.3, bottom=0.2, color=[0.6, 0.6, 0.6])
                ax.set_xlim([-0.2, 5.0])
                ax.set_xticks([-0.2, 0.0, 0.3, 2.5, 5.0])
                ax.set_xticklabels([str(gains_in[0])+' ('+str(len(gains_in))+'x)', '', '', str(gains_middle[0])+' ('+str(len(gains_middle))+'x)', ''])
                start, end = ax.get_ylim()
                ax.yaxis.set_ticks(numpy.around(numpy.arange(start, end+1, 1),decimals=1))
            else:
                fig, ax = plotCycleFrequencies(fig, ax, xvalues=0, lfp=cscs_in[i], threshMin=thetaRange[0],
                                                threshMax=thetaRange[1])
                ax.hist(findCycleFrequencies(lfp=cscs_in[i], threshMin=thetaRange[0], threshMax=thetaRange[1]),
                         orientation='horizontal', normed=0.3, bottom=0.2, color=[0.6, 0.6, 0.6])
        if transitions:
            for o in numpy.arange(len(cscs_middle)):
                fig, ax = plotCycleFrequencies(fig, ax, xvalues=2.5, lfp=cscs_middle[o], threshMin=thetaRange[0],
                                                   threshMax=thetaRange[1])
                ax.hist(findCycleFrequencies(lfp=cscs_middle[o], threshMin=thetaRange[0], threshMax=thetaRange[1]),
                            orientation='horizontal', normed=0.3, bottom=2.7, color=[0.6, 0.6, 0.6])


def rawCscTraces(gc_indexes):

    set_csc_length = 4.0

    fig1 = pl.figure(figsize=(12, 7))
    ax = fig1.add_subplot(111)

    if transitions:
        xlabel = 'relative time to gain change (s)'
    else:
        xlabel = 'time since last gain change (s)'
    # GC[gc_index].csc is for transitions cut from start over middle to stop and for individual gains from start to middle

    # first get longest trace of all GC[gc_index].csc to set limx=max_csc_length
    csc_length_in = []
    csc_length_out = []
    csc_length = []

    print 'number of gain change incidents: ', len(gc_indexes)

    for gc_index in gc_indexes:
        if GC[gc_index].csc.timeUnit == 'ms':
            print 'Changing raw csc trace time Unit from ms in s!'
            GC[gc_index].csc.changeTimeUnit('s')

        normalised_stop = GC[gc_index].csc.times_stop-GC[gc_index].csc.times_start
        # getting csc gain change time and length of different parts of csc traces
        if transitions:
            csc_time_gain = signale.tools.findNearest(GC[gc_index].csc.times, GC[gc_index].t_middle)[1]
            csc_time_gain -= GC[gc_index].csc.times_start  # = normalised gain time
            csc_length_in.append(csc_time_gain)
            csc_length_out.append(normalised_stop-csc_time_gain)
        else:
            csc_length.append(normalised_stop)

    if transitions:
        max_csc_length_in = numpy.max(csc_length_in)
        max_csc_length_out = numpy.max(csc_length_out)
        max_csc_length = max_csc_length_in+max_csc_length_out
        pos_gc = max_csc_length_in
    else:
        max_csc_length = numpy.max(csc_length)

    print 'max_csc_length: ', max_csc_length, ' in ', GC[gc_index].csc.timeUnit

    for id, gc_index in enumerate(gc_indexes):
        # aligning gain change positions in case of transitions = True

        # copy GC[gc_index].csc.times so that the original array doesnt get modified!
        t = GC[gc_index].csc.times.copy()
        # normalising time Axes
        t -= GC[gc_index].csc.times_start
        if transitions:
            # setting starting time of individual csc signals (s) to its gain change time
            t -= csc_length_in[id]
            # moving gain change time to max_csc_length/2
            # t += pos_gc
            t += set_csc_length/2.0

        # subtract mean and make signal smaller for plotting
        s = GC[gc_index].csc.signal.copy()
        s -= s.mean()
        for csc in cscList:
            #s /= max(-1e-100, csc.signal.max())*.8
            s /= max(-1e-100, csc.signal.max())*1.0

        pl.plot(t, s+(2*id), '-', linewidth=1,
                color=custom_plot.pretty_colors_set2[id%custom_plot.pretty_colors_set2.__len__()])

    # gain change (events.times[GC[gc_index].index_middle]) should be marked by vertical red line and therefore aligned
    #ax.axvline(x=pos_gc, linewidth=2, color='r')
    ax.axvline(x=set_csc_length/2.0, linewidth=2, color='r')

    # finish off plotting
    custom_plot.huebschMachen(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(cscName)
    ax.set_xlim(0, set_csc_length)
    ax.set_ylim(-1, 2*len(gc_indexes)-1)
    ax.set_yticks(numpy.arange(2*len(gc_indexes)))
    ax.yaxis.set_ticklabels([])
    return fig1


def getTransitionIndexes(GC, GC_in, GC_middle):
    gains = []
    run_direction = []
    transition_indexes = []
    for g in numpy.arange(len(GC)):
        if GC[g].gain_in == GC_in and GC[g].gain_middle == GC_middle and \
                            GC[g].pc_gain_in.traj.places[-7, 0] < GC[g].pc_gain_out.traj.places[7, 0]:
            run_direction.append('rightwards')
            gains.append((GC_in, GC_middle))
            transition_indexes.append(g)
        elif GC[g].gain_middle == GC_in and GC[g].gain_in == GC_middle and \
                        GC[g].pc_gain_in.traj.places[-7, 0] > GC[g].pc_gain_out.traj.places[7, 0]:
            run_direction.append('leftwards')
            gains.append((GC_middle, GC_in))
            transition_indexes.append(g)
    return numpy.array(gains), numpy.array(run_direction), numpy.array(transition_indexes)


def plotPooledPhases(placeCell, fig, axes, text, Ax = None, singleRuns=False, gc_indexes=False, pf_xlim=False,
                     traj_xlim=0, xbin_only=False):

    singleTrialNum = [[0], [0], [0]]
    fig_SR = []
    axes_SR = []
    fit_phases_pf_left_right = [[], [], []]  # for pooled fits [[allRuns_Data], [rightRuns_Data], [leftRuns_Data]]
    fit_phases_pf_left_right_SR = [[], [], []]  # for singleRun fits [[allRuns_Data], [rightRuns_Data], [leftRuns_Data]]
    fit_dictio = [[], [], []]  # for pooled fits [[allRuns_Data], [rightRuns_Data], [leftRuns_Data]]
    fit_dictio_SR = [[], [], []]  # for singleRun fits [[allRuns_Data], [rightRuns_Data], [leftRuns_Data]]
    spike_count_SR = [[], [], []]  # for singleRun spike counts [[allRuns_Data], [rightRuns_Data], [leftRuns_Data]]

    # xlim is the rounded array of the [xMin, xMax] of the entire (unsliced) trajectory
    xlim = (numpy.around(traj.xlim, decimals=1)/normalisation_gain).tolist()
    ylim = list([numpy.floor((traj.ylim[0]/normalisation_gain)*100)/100, numpy.ceil((traj.ylim[1]/normalisation_gain)*100)/100])

    if gain_normalised:
        print 'INFO: normalising place cell trajectory places, their x,y limits and trackLength with the ' \
              'gain ', normalisation_gain
        #placeCell.traj.places = placeCell.traj.places/normalisation_gain
        placeCell.traj.xlim = xlim
        placeCell.traj.ylim = ylim
        placeCell.traj.trackLength = xlim[1]-xlim[0]

    # main phase plot: adds 12 new axes to the top of phase plots (3 histograms on top, 3 histograms on the right,
    # 3 traj plots on top and 3 single run traj plots on top)

    # If fig = False and axes = False no plots are generated. If limx = True the plot gets restricted to pf area
    # if pf_xlim = [x0, x1], the place field limits are set to these values and the wfit is done within that window.

    fig, axes, FR_distribution, FR_distribution_ySum, pf_limits, wfit_xy, pf_phases_left_right, \
    occupancy_probability_ysum, spike_count_ySum, fit_dictionary = spikesPhase.prepareAndPlotPooledPhases(
        placeCell, cscList[0], cscName, fig=fig, axes=axes, limx=xlim, limy=ylim, fontsize=fontsize, wfit=wfit,
        noSpeck=noSpeck, text=text, mutterTraj=traj, GetPlaceField=True, gc=True,
        normalisation_gain=normalisation_gain, pf_xlim=pf_xlim, xbin_only=xbin_only)

    # For SINGLE RUN PLOTS: Asses how many single run plots are needed
    # For that: get all, right or left run number, where there were spikes within the place field.

    if singleRuns:
        # singleTrialNum = [[singleTrialNum_allRuns], [singleTrialNum_rightRuns], [singleTrialNum_leftRuns]]

        singleTrialNum, traj_in_pf_and_gain = getSingleRunAxNum(inField_pc=placeCell.run_pc.inField_pc,
                                                                run_traj=placeCell.run_pc.traj, minSpikeNum_perRun=4,
                                                                plot=False, gc_indexes=gc_indexes, pf_x=pf_limits)

        # get single Run figures and axes
        for r in [0, 1, 2]:

            fig_sr = getFigure(AxesNum=len(singleTrialNum[r]))
            axes_sr = addAxesPositions(fig_sr, Num=len(singleTrialNum[r]))

            if not axes_sr == []:

                # plot pooled non zoomed run direction data into the first summary plot
                fig_sr, axes_sr, FR_dist_sr, FR_dist_ySum_sr, pf_limits_sr, wfit_xy_sr, pf_phases_left_right_sr, \
                occupancy_probability_ysum_sr, spike_count_ySum_sr, fit_dictionary_sr = \
                    spikesPhase.prepareAndPlotPooledPhases(placeCell, cscList[0], cscName, fig=fig_sr, axes=axes_sr,
                                                           limx=xlim, limy=ylim, fontsize=fontsize, wfit=wfit,
                                                           noSpeck=noSpeck, text=text, mutterTraj=traj,
                                                           GetPlaceField=True, gc=True, normalisation_gain=1.0,
                                                           pf_xlim=pf_xlim, direction=r, title='Not Zoomed', axNum=0,
                                                           xbin_only=xbin_only)

                if len(pf_phases_left_right_sr):
                    fit_phases_pf_left_right[r].append(pf_phases_left_right_sr[0])
                    fit_dictio[r].append(fit_dictionary_sr[0])

                # plot zoomed run direction data into the second and third summary plot
                fig_sr, axes_sr, FR_dist_sr, FR_dist_ySum_sr, pf_limits_sr, wfit_xy_sr, pf_phases_left_right_sr, \
                occupancy_probability_ysum_sr, spike_count_ySum_sr, fit_dictionary_sr = \
                    spikesPhase.prepareAndPlotPooledPhases(placeCell, cscList[0], cscName, fig=fig_sr, axes=axes_sr,
                                                           limx=True, limy=ylim, fontsize=fontsize, wfit=wfit,
                                                           noSpeck=noSpeck, text=False, mutterTraj=traj,
                                                           GetPlaceField=True, gc=True, normalisation_gain=1.0,
                                                           pf_xlim=pf_xlim, direction=r, title='', axNum=1,
                                                           xbin_only=xbin_only)

                fig_sr, axes_sr, FR_dist_sr, FR_dist_ySum_sr, pf_limits_sr, wfit_xy_sr, pf_phases_left_right_sr, \
                occupancy_probability_ysum_sr, spike_count_ySum_sr, fit_dictionary_sr = \
                    spikesPhase.prepareAndPlotPooledPhases(placeCell, cscList[0], cscName, fig=fig_sr, axes=axes_sr,
                                                           limx=True, limy=ylim, fontsize=fontsize, wfit=False,
                                                           noSpeck=noSpeck, text=False, mutterTraj=traj,
                                                           GetPlaceField=True, gc=True, normalisation_gain=1.0,
                                                           pf_xlim=pf_xlim, direction=r, title='', axNum=2,
                                                           xbin_only=xbin_only)

                for a in [1, 2]:
                    axes_sr[a].set_ylabel('')
                    axes_sr[a].set_yticklabels([])
                    if len(singleTrialNum[r]) > 2:
                        axes_sr[a].set_xlabel('')

                axes_sr[0].set_xlabel('')
                if len(singleTrialNum[r]) == 2:
                    axes_sr[1].set_xlabel('')

                # PLOT SINGLE RUNS
                for num, gain_index in enumerate(singleTrialNum[r]):
                    number = num+4

                    # GC[gain_index].placeCell is not normalised. When r = 0 (all runs), rightward and leftward runs are
                    # normalised as well. For r = 1 (rightwards runs) and r = 2 (leftwards runs) normalisation is then
                    # already done. Thats why for r = 1 , 2 the normalisation has to be 1.0!
                    if r == 0:
                        Normalisation = normalisation_gain
                    else:
                        Normalisation = 1.0

                    fig_sr, axes_sr, FR_dist_sr, FR_dist_ySum_sr, pf_limits_sr, wfit_xy_sr, pf_phases_left_right_sr, \
                    occupancy_probability_ysum_sr, spike_count_ySum_sr, fit_dictionary_sr = \
                        spikesPhase.prepareAndPlotPooledPhases(GC[gain_index].placeCell, cscList[0], cscName,
                                                               fig=fig_sr, axes=axes_sr, limx=True, limy=ylim,
                                                               fontsize=fontsize, wfit=True, noSpeck=noSpeck,
                                                               text=False, mutterTraj=traj, GetPlaceField=True,
                                                               gc=True, normalisation_gain=Normalisation,
                                                               pf_xlim=pf_limits[r], direction=r, title='',
                                                               axNum=number, extraPlots=False, xbin_only=xbin_only)
                    if len(pf_phases_left_right_sr):
                        fit_phases_pf_left_right_SR[r].append(pf_phases_left_right_sr[0])
                        fit_dictio_SR[r].append(fit_dictionary_sr[0])

                    if len(spike_count_ySum_sr):
                        if len(spike_count_ySum_sr[0]) == 2:
                            if len(spike_count_ySum_sr[0][1]):
                                spike_count_SR[r].append(numpy.nansum(spike_count_ySum_sr[0][1]))
                            else:
                                spike_count_SR[r].append(0)
                        else:
                                spike_count_SR[r].append(0)
                    else:
                        spike_count_SR[r].append(0)

                    if number not in [4, 9, 14, 19, 24, 29]:
                        axes_sr[number].set_ylabel('')
                        axes_sr[number].set_yticklabels([])
                    if len(singleTrialNum[r]) > 5:
                        if num in numpy.arange(len(singleTrialNum[r])-5):
                            axes_sr[number].set_xlabel('')

                fig_SR.append(fig_sr)
                axes_SR.append(axes_sr)

    all_FR_ySum = []
    all_FR_2dim = []
    FR = []
    FR_inPF = []
    FR_ySum = []
    FR_ySum_inPF = []
    spike_count_perRun_pf_sum = []
    spike_count_pf_sum = []

    FR_dis = FR_distribution_ySum  # FR_distribution or FR_distribution_ySum used only for the FR plot!
    FR_dis_2d = FR_distribution

    if Ax:
        for i in numpy.arange(len(FR_dis)):

            vis_track_length = 2.
            if gain_normalised:
                start = vis_track_length/Grun
            else:
                start = vis_track_length

            x_idx = numpy.array([num for num in numpy.arange(len(FR_dis[i][0])) if 0 <= FR_dis[i][0][num] <= start])
            FR_dis[i][0] = numpy.array(FR_dis[i][0])[x_idx]
            FR_dis[i][1] = numpy.array(FR_dis[i][1])[x_idx]

            if i == 2:  # for leftward runs plot abolute x-value from start position
                FR_x = abs(FR_dis[i][0]-start)
            else:
                FR_x = FR_dis[i][0]
            if Grun == 0.5:
                co = custom_plot.pretty_colors_set2[0]
            else:
                co = custom_plot.pretty_colors_set2[1]

            line = Ax[i].plot(FR_x, FR_dis[i][1], color=co)
                       # custom_plot.pretty_colors_set2[numpy.where(Titles == Grun)[0][0]%custom_plot.pretty_colors_set2.__len__()])

            if xlim > traj_xlim:
                Ax[i].set_xlim(xlim)
                Ax[i].set_xticks([numpy.around(xlim[0], decimals=2), numpy.around(xlim[1]/2., decimals=2),
                                numpy.around(xlim[1], decimals=2)])
            if i == 0:
                lines.append(line)
                labels.append('Gain: '+str(Grun))

    for i in numpy.arange(len(FR_dis)):

        # Get indices for pf limits within spike_count_ySum:

        if len(spike_count_ySum[i][0]):
            pf_limit_args_sc = [signale.tools.findNearest(spike_count_ySum[i][0], pf_limits[i][0])[0],
                                signale.tools.findNearest(spike_count_ySum[i][0], pf_limits[i][1])[0]]
            # spike_count_ySum within pf:
            spike_count_pfSum = sum(spike_count_ySum[i][1][pf_limit_args_sc[0]: pf_limit_args_sc[1]+1])
            spike_count_pf_sum.append(spike_count_pfSum)
            if not 'traj_in_pf_and_gain' in locals():
                spike_count_perRun_pf_sum.append(['no traj_in_pf_and_gain value!'])
            else:
                spike_count_perRun_pf_sum.append(spike_count_pfSum/traj_in_pf_and_gain[i])

        else:
            spike_count_perRun_pf_sum.append(0)
            spike_count_pf_sum.append(0)

        # Get indices for pf limits within FR_distribution:
        pf_limit_args = [signale.tools.findNearest(FR_dis_2d[i][0], pf_limits[i][0])[0],
                         signale.tools.findNearest(FR_dis_2d[i][0], pf_limits[i][1])[0]]
        # FR_distribution within pf:
        FR_distribution_pf = FR_dis_2d[i][1][pf_limit_args[0]: pf_limit_args[1]+1]

        # FR_distribution_ySum within pf:
        FR_distribution_ySum_pf = FR_dis[i][1][pf_limit_args[0]: pf_limit_args[1]+1]

        # find center of mass x value index within firing rate plot:
        # For y FR maxima for all x values:
        if math.isnan(numpy.round(image_m.center_of_mass(numpy.array(FR_dis_2d[i][1]))[0])):  # if all FR are nan, take the track center
            r_CM = int(numpy.round(len(FR_dis_2d[i][0])/2.0))
        else:
            r_CM = int(numpy.round(image_m.center_of_mass(numpy.array(FR_dis_2d[i][1]))[0]))

        # For y FR maxima for x values within place field:
        if math.isnan(numpy.round(image_m.center_of_mass(numpy.array(FR_distribution_pf))[0])):
            r_CM_pf = int(numpy.round(len(FR_distribution_pf)/2.0))+pf_limit_args[0]  # set max to pf middle
        else:
            r_CM_pf = int(numpy.round(image_m.center_of_mass(numpy.array(FR_distribution_pf))[0]))+pf_limit_args[0]

        # For y-summed FR for all x values:
        if math.isnan(numpy.round(image_m.center_of_mass(numpy.array(FR_dis[i][1]))[0])):
            r_CM_ySum = int(numpy.round(len(FR_dis[i][0])/2.0))
        else:
            r_CM_ySum = int(numpy.round(image_m.center_of_mass(numpy.array(FR_dis[i][1]))[0]))

        # For y-summed FR for x values within place field:
        if math.isnan(numpy.round(image_m.center_of_mass(numpy.array(FR_distribution_ySum_pf))[0])):
            r_CM_ySum_pf = int(numpy.round(len(FR_distribution_ySum_pf)/2.0))+pf_limit_args[0]  # set max to pf middle
        else:
            r_CM_ySum_pf = int(numpy.round(image_m.center_of_mass(numpy.array(FR_distribution_ySum_pf))[0]))+pf_limit_args[0]

        all_FR_ySum.append([FR_dis[i][0], FR_dis[i][1]])
        all_FR_2dim.append([FR_dis_2d[i][0], FR_dis_2d[i][1]])
        FR.append([FR_dis_2d[i][0][numpy.argmax(FR_dis_2d[i][1])], max(FR_dis_2d[i][1]), FR_dis_2d[i][0][r_CM]])
        FR_inPF.append([FR_dis_2d[i][0][numpy.argmax(FR_distribution_pf)+pf_limit_args[0]],
                       max(FR_distribution_pf), FR_dis_2d[i][0][r_CM_pf]])

        FR_ySum.append([FR_dis[i][0][numpy.argmax(FR_dis[i][1])], max(FR_dis[i][1]),
                        FR_dis[i][0][r_CM_ySum]])

        FR_ySum_inPF.append([FR_dis[i][0][numpy.argmax(FR_distribution_ySum_pf)+pf_limit_args[0]],
                             max(FR_distribution_ySum_pf), FR_dis[i][0][r_CM_ySum_pf]])

    max_FR = {'xMaxFR_MaxFR_xCM_allRuns_gain_'+str(Grun): FR[0], 'xMaxFR_MaxFR_xCM_rightRuns_gain_'+str(Grun): FR[1],
              'xMaxFR_MaxFR_xCM_leftRuns_gain_'+str(Grun): FR[2]}
    max_FR_in_pf = {'xMaxFRinPF_MaxFRinPF_xCMinPF_allRuns_gain_'+str(Grun): FR_inPF[0],
                   'xMaxFRinPF_MaxFRinPF_xCMinPF_rightRuns_gain_'+str(Grun): FR_inPF[1],
                   'xMaxFRinPF_MaxFRinPF_xCMinPF_leftRuns_gain_'+str(Grun): FR_inPF[2]}
    max_FR_ySum = {'xMaxFRySum_MaxFRySum_xCMySum_allRuns_gain_'+str(Grun): FR_ySum[0],
                   'xMaxFRySum_MaxFRySum_xCMySum_rightRuns_gain_'+str(Grun): FR_ySum[1],
                   'xMaxFRySum_MaxFRySum_xCMySum_leftRuns_gain_'+str(Grun): FR_ySum[2]}
    max_FR_ySum_in_pf = {'xMaxFRySuminPF_MaxFRySuminPF_xCMySuminPF_allRuns_gain_'+str(Grun): FR_ySum_inPF[0],
                         'xMaxFRySuminPF_MaxFRySuminPF_xCMySuminPF_rightRuns_gain_'+str(Grun): FR_ySum_inPF[1],
                         'xMaxFRySuminPF_MaxFRySuminPF_xCMySuminPF_leftRuns_gain_'+str(Grun): FR_ySum_inPF[2]}
    all_FR = {'allFR_x_y_gain_'+str(Grun): all_FR_ySum[0],
              'rightFR_x_y_gain_'+str(Grun): all_FR_ySum[1],
              'leftFR_x_y_gain_'+str(Grun): all_FR_ySum[2]}
    all_FR_2d = {'allFR_2d_x_y_gain_'+str(Grun): all_FR_2dim[0],
              'rightFR_2d_x_y_gain_'+str(Grun): all_FR_2dim[1],
              'leftFR_2d_x_y_gain_'+str(Grun): all_FR_2dim[2]}

    phaseFit_pfLeft_pfRight_pooled = {'Pooled_phases_pfLeft_pfRight_allRuns_gain_'+str(Grun): fit_phases_pf_left_right[0],
                                      'Pooled_phases_pfLeft_pfRight_rightRuns_gain_'+str(Grun): fit_phases_pf_left_right[1],
                                      'Pooled_phases_pfLeft_pfRight_leftRuns_gain_'+str(Grun): fit_phases_pf_left_right[2]}
    d_R = [[], [], []]
    d_aopt = [[], [], []]
    d_p = [[], [], []]
    d_phi0 = [[], [], []]
    d_rho = [[], [], []]
    for d in [0, 1, 2]:
        if len(fit_dictio[d]):
            d_R[d].append(fit_dictio[d][0]['R'])
            d_aopt[d].append(fit_dictio[d][0]['aopt'])
            d_p[d].append(fit_dictio[d][0]['p'])
            d_phi0[d].append(fit_dictio[d][0]['phi0'])
            d_rho[d].append(fit_dictio[d][0]['rho'])

    phaseFit_dictio = {'Pooled_phaseFit_R_allRuns_gain_'+str(Grun): d_R[0],
                       'Pooled_phaseFit_aopt_allRuns_gain_'+str(Grun): d_aopt[0],
                       'Pooled_phaseFit_p_allRuns_gain_'+str(Grun): d_p[0],
                       'Pooled_phaseFit_phi0_allRuns_gain_'+str(Grun): d_phi0[0],
                       'Pooled_phaseFit_rho_allRuns_gain_'+str(Grun): d_rho[0],
                       'Pooled_phaseFit_R_rightRuns_gain_'+str(Grun): d_R[1],
                       'Pooled_phaseFit_aopt_rightRuns_gain_'+str(Grun): d_aopt[1],
                       'Pooled_phaseFit_p_rightRuns_gain_'+str(Grun): d_p[1],
                       'Pooled_phaseFit_phi0_rightRuns_gain_'+str(Grun): d_phi0[1],
                       'Pooled_phaseFit_rho_rightRuns_gain_'+str(Grun): d_rho[1],
                       'Pooled_phaseFit_R_leftRuns_gain_'+str(Grun): d_R[2],
                       'Pooled_phaseFit_aopt_leftRuns_gain_'+str(Grun): d_aopt[2],
                       'Pooled_phaseFit_p_leftRuns_gain_'+str(Grun): d_p[2],
                       'Pooled_phaseFit_phi0_leftRuns_gain_'+str(Grun): d_phi0[2],
                       'Pooled_phaseFit_rho_leftRuns_gain_'+str(Grun): d_rho[2]}

    phaseFit_pfLeft_pfRight_SR = {'SR_phases_pfLeft_pfRight_allRuns_gain_'+str(Grun): fit_phases_pf_left_right_SR[0],
                                  'SR_phases_pfLeft_pfRight_rightRuns_gain_'+str(Grun): fit_phases_pf_left_right_SR[1],
                                  'SR_phases_pfLeft_pfRight_leftRuns_gain_'+str(Grun): fit_phases_pf_left_right_SR[2]}

    R_SR = [[], [], []]
    aopt_SR = [[], [], []]
    p_SR = [[], [], []]
    phi0_SR = [[], [], []]
    rho_SR = [[], [], []]
    for d in [0, 1, 2]:
        if len(fit_dictio_SR[d]):
            for idx in numpy.arange(len(fit_dictio_SR[d])):
                R_SR[d].append(fit_dictio_SR[d][idx]['R'])
                aopt_SR[d].append(fit_dictio_SR[d][idx]['aopt'])
                p_SR[d].append(fit_dictio_SR[d][idx]['p'])
                phi0_SR[d].append(fit_dictio_SR[d][idx]['phi0'])
                rho_SR[d].append(fit_dictio_SR[d][idx]['rho'])

    phaseFit_dictio_SR = {'SR_phaseFit_R_allRuns_gain_'+str(Grun): R_SR[0],
                          'SR_phaseFit_aopt_allRuns_gain_'+str(Grun): aopt_SR[0],
                          'SR_phaseFit_p_allRuns_gain_'+str(Grun): p_SR[0],
                          'SR_phaseFit_phi0_allRuns_gain_'+str(Grun): phi0_SR[0],
                          'SR_phaseFit_rho_allRuns_gain_'+str(Grun): rho_SR[0],
                          'SR_phaseFit_R_rightRuns_gain_'+str(Grun): R_SR[1],
                          'SR_phaseFit_aopt_rightRuns_gain_'+str(Grun): aopt_SR[1],
                          'SR_phaseFit_p_rightRuns_gain_'+str(Grun): p_SR[1],
                          'SR_phaseFit_phi0_rightRuns_gain_'+str(Grun): phi0_SR[1],
                          'SR_phaseFit_rho_rightRuns_gain_'+str(Grun): rho_SR[1],
                          'SR_phaseFit_R_leftRuns_gain_'+str(Grun): R_SR[2],
                          'SR_phaseFit_aopt_leftRuns_gain_'+str(Grun): aopt_SR[2],
                          'SR_phaseFit_p_leftRuns_gain_'+str(Grun): p_SR[2],
                          'SR_phaseFit_phi0_leftRuns_gain_'+str(Grun): phi0_SR[2],
                          'SR_phaseFit_rho_leftRuns_gain_'+str(Grun): rho_SR[2]}

    # draw red vertical line at x position of the gain change postition
    # axes[0].axvline(events.places[GC_object.index_middle, 0], linewidth=1, color='r')
    # #for top left single runs plot:
    # axes[len(axes)-3].axhline(events.places[GC_object.index_middle, 0], linewidth=1, color='r')
    # draw reward zones in all three top single runs plots:

    if pooled:
        zones = numpy.arange(3)
    else:
        zones = [0]

    if text and axes is not False and not axes == []:
        b = [3, 2, 1]
        for a in zones:
            if hs == []:
                for r in numpy.arange(len(rz)):
                    axes[a].axvline(rz[r][0]/normalisation_gain, linewidth=1, color='r')
                    if pooled:
                        axes[len(axes)-b[a]].axhline(rz[r][0]/normalisation_gain, linewidth=1, color='r')
            else:
                for r in numpy.arange(len(rz)):
                    axes[a].axvline(rz[r][0]/normalisation_gain, linewidth=1, color='g')
                    axes[len(axes)-b[a]].axhline(rz[r][0]/normalisation_gain, linewidth=1, color='g')
                for h in numpy.arange(len(hs)):
                    axes[a].axvline(hs[h][0]/normalisation_gain, linewidth=1, color='r')
                    if pooled:
                        axes[len(axes)-b[a]].axhline(hs[h][0]/normalisation_gain, linewidth=1, color='r')

        # if singleRuns:
        #     y = [.025, .025]
        #     x = .85
        # else:
        x = .995
        y = [.04, .07]
        if smooth:
            fig.text(.995, y[0], 'Trajectory smoothing window: '+str(smooth)+' s', fontsize=fontsize-6,
                     horizontalalignment='right')
        else:
            fig.text(.995, y[0], 'Not smoothed trajectory', fontsize=fontsize-6,  horizontalalignment='right')

        if gain_normalised:
            fig.text(x, y[1], 'Trajectory is normalised with gain: '+str(normalisation_gain), fontsize=fontsize-6,
                     horizontalalignment='right')

    # draw red horizontal line at x position of the gain change postition
    if fig is not False:
        fig.canvas.draw()

    return fig, axes, fig_SR, axes_SR, pf_limits, wfit_xy, singleTrialNum, xlim, max_FR, max_FR_in_pf, max_FR_ySum, \
           max_FR_ySum_in_pf, all_FR, all_FR_2d, occupancy_probability_ysum, spike_count_ySum, spike_count_pf_sum, \
           spike_count_perRun_pf_sum, phaseFit_pfLeft_pfRight_pooled, phaseFit_dictio, phaseFit_pfLeft_pfRight_SR, \
           phaseFit_dictio_SR, spike_count_SR


def plotPhasesOrPlaces(GC, Ax=None, concatenated=True, z=0, traj_xlim=0, pf_xlim=False, xbin_only=False):

    transition_indexes = []
    gc_number = 0
    gc_indexes = []
    max_FR = []
    max_FR_ySum = []
    max_FR_ySum_in_pf = []
    max_FR_in_pf = []
    all_FR = []
    all_FR_2d = []
    pf_limits = []
    occupancy_probability_ysum = []
    spike_count_ySum = []
    spike_count_in_pf = []
    phaseFit_pfLeft_pfRight_pooled = []
    phaseFit_dictio = []
    phaseFit_pfLeft_pfRight_SR = []
    phaseFit_dictio_SR = []
    lfp_freq_gain05 = []
    lfp_freq_gain15 = []
    speeds_real_gain05 = []
    speeds_real_gain15 = []
    speeds_virtual_gain05 = []
    speeds_virtual_gain15 = []
    theta_power_gain05 = []
    theta_power_gain15 = []
    delta_power_gain05 = []
    delta_power_gain15 = []

    if transitions:
        # get transition indexes, which have GC_in on the left and GC_out on the right of the GC
        gains, run_direction, transition_indexes = getTransitionIndexes(GC, GC_in, GC_middle)

    # print info on gain change occurrence
    if len(transition_indexes) == 1 and transitions or len(transition_indexes) == 0 and transitions:
        print 'Gain change ', GC_in, ' -> ', GC_middle, ': occurred', len(transition_indexes), 'time'
    elif transitions:
        print 'Gain change ', GC_in, ' -> ', GC_middle, ': occurred', len(transition_indexes), 'times'
    else:
        print 'Gain ', Grun, ' occurred ', AxNum, ' times'

    # get figure with appropriate axes, depending what is suppose to be plotted
    if pooled:
        fig = getFigure()
        if not plotPlace:
            axes = addAxesPositions(fig, Title, pooled=True)
    #     singleRuns = False
    # elif not pooled and concatenated:
    #     fig = False
    #     axes = False
    #     singleRuns = True

    # start the main part of this function -> to execute the phase plot, depending on the command line settings

    for s in numpy.arange(len(GC)):

        if z == 0:
            text = True
        else:
            text = False

        if GC[s].gain_in == Grun and GC[s].ol_in == 1:
            print 'OPEN LOOP ON!!'

        freq, speeds_real, speeds_virtual, MaximaSignals, deltaAmp, thetaAmp = \
            findCycleFrequencies(lfp=GC[s].csc_gain_in, GC_element=GC[s])

        if GC[s].gain_in == 0.5:
            lfp_freq_gain05.append(numpy.array(freq))
            speeds_real_gain05.append(numpy.array(speeds_real))
            speeds_virtual_gain05.append(numpy.array(speeds_virtual))
            # theta_power_gain05.append(numpy.array(MaximaSignals))
            theta_power_gain05.append(thetaAmp)
            delta_power_gain05.append(deltaAmp)
        elif GC[s].gain_in == 1.5:
            lfp_freq_gain15.append(numpy.array(freq))
            speeds_real_gain15.append(numpy.array(speeds_real))
            speeds_virtual_gain15.append(numpy.array(speeds_virtual))
            # theta_power_gain15.append(numpy.array(MaximaSignals))
            theta_power_gain15.append(thetaAmp)
            delta_power_gain15.append(deltaAmp)

        if transitions and s in transition_indexes or not transitions and GC[s].gain_in == Grun:  # \
                # and GC[s].visible_in == 1 and GC[s].ol_in == 0:

            gc_number += 1

            print 'GC index is: ', s
            if not transitions:
                print 'Gain run is: ', Grun

            gc_indexes.append(s)

            if not transitions:

                #placeCell = None

                GC[s].placeCell = GC[s].pc_gain_in
                GC[s].csc = GC[s].csc_gain_in

                if gain_normalised:
                    print 'INFO: normalising place cell trajectory places with the gain ', normalisation_gain
                    GC[s].placeCell.traj.places = GC[s].placeCell.traj.places/normalisation_gain

                if pooled and concatenated or concatenated:
                    # concatenate all seperate place cells with the same gain to one place cell
                    print 'conatenate place cells!'
                    if z == 0:
                        placeCell = GC[s].placeCell
                    else:
                        placeCell = placeCell.concatenate(GC[s].placeCell) #, csc=Csc, placeCell_csc=GC[s].csc)

                    # when all place cells are concatenated (z == AxNum-1) and not the places are suppose to be plotted,
                    # plot the phase plot
                    if not plotPlace and z == AxNum-1:

                        # placeCell.traj has trajectory where all points with running speed under threshold are removed!
                        # if the entire turajectory is suppose to be illustrated, overwrite it with original traj:
                        # placeCell.traj = traj  # careful! changes mean spikecount per run etc.!!

                        fig, axes, fig_SR, axes_SR, pf_limits, wfit_xy, singleTrialNum, traj_xlim, max_FR, \
                        max_FR_in_pf, max_FR_ySum, max_FR_ySum_in_pf, all_FR, all_FR_2d, occupancy_probability_ysum, \
                        spike_count_ySum, spike_count_pf_sum, spike_count_perRun_pf_sum, phaseFit_pfLeft_pfRight_pooled, \
                        phaseFit_dictio, phaseFit_pfLeft_pfRight_SR, phaseFit_dictio_SR, spike_count_SR = \
                             plotPooledPhases(placeCell=placeCell, fig=fig, axes=axes, text=True, Ax=Ax,
                                              singleRuns=singleRuns, gc_indexes=gc_indexes, pf_xlim=pf_xlim,
                                              traj_xlim=traj_xlim, xbin_only=xbin_only)

                elif pooled and not plotPlace:

                    # if separate place cells are not suppose to be concatenated and not the places are suppose to be
                    # plotted, plot the phase plot for every individual place cell (GC[s].placeCell) into one plot.
                    fig, axes, fig_SR, axes_SR, pf_limits, wfit_xy, singleTrialNum, traj_xlim, max_FR, max_FR_in_pf, \
                    max_FR_ySum, max_FR_ySum_in_pf, all_FR, all_FR_2d, occupancy_probability_ysum, spike_count_ySum, \
                    spike_count_pf_sum, spike_count_perRun_pf_sum, phaseFit_pfLeft_pfRight_pooled, phaseFit_dictio, \
                    phaseFit_pfLeft_pfRight_SR, phaseFit_dictio_SR, spike_count_SR = \
                        plotPooledPhases(placeCell=GC[s].placeCell, fig=fig, axes=axes, text=text, pf_xlim=pf_xlim,
                                         traj_xlim=traj_xlim, xbin_only=xbin_only)

                    # if runs is not None:
                    #     print 'run limits: ', pf_limits[runs]

            elif transitions:
                if gain_normalised:
                    print 'INFO: normalising place cell trajectory places with the gains ', GC[s].gain_in, ' and ', GC[s].gain_middle
                    GC[s].pc_gain_in.traj.places = GC[s].pc_gain_in.traj.places/GC[s].gain_in
                    GC[s].pc_gain_out.traj.places = GC[s].pc_gain_out.traj.places/GC[s].gain_middle
                    GC[s].placeCell = GC[s].pc_gain_in.concatenate(GC[s].pc_gain_out, nan=False)
                if z == 0:
                    placeCell = GC[s].placeCell
                    Csc = GC[s].csc
                else:
                    placeCell = placeCell.concatenate(GC[s].placeCell)
                    Csc = Csc.concatenate(GC[s].csc)

                if pooled and not plotPlace and z == len(transition_indexes)-1:
                    fig, axes, fig_SR, axes_SR, pf_limits, wfit_xy, singleTrialNum, traj_xlim, max_FR, max_FR_in_pf, \
                    max_FR_ySum, max_FR_ySum_in_pf, all_FR, all_FR_2d, occupancy_probability_ysum, spike_count_ySum, \
                    spike_count_pf_sum, spike_count_perRun_pf_sum, phaseFit_pfLeft_pfRight_pooled, phaseFit_dictio, \
                    phaseFit_pfLeft_pfRight_SR, phaseFit_dictio_SR, spike_count_SR = \
                        plotPooledPhases(placeCell=placeCell, fig=fig, axes=axes, text=True, Ax=Ax,
                                         traj_xlim=traj_xlim, xbin_only=xbin_only)

                # Top right plot in axes[3]:
                # 1) draw frequencies and histogram for gain change theta frequencies
                # gainChangeFreq(fig=fig, ax=axes[3], GC_object=GC[s], Title=Titles[PT])

                # Bottom right plot in axes[4]:
                # 1) draw frequencies and histogram for individual gains
                #individualGainFreq(fig=fig, ax=axes[4])

                # or 2) LFP of the entire session / traj

            # MAKE FFT PLOT
            if pooled and text:  # or not pooled and z == AxNum-1 and not axes == []:
                spikesPhase.cscFft(fig=fig, ax=axes[3], csc=cscList[0], thetaRange=thetaRange)

            if singleRuns and z == AxNum-1:  # only plots the fft for the gain of the last run!!
                for n, figure in enumerate(fig_SR):
                    if not axes_SR[n] == []:
                        spikesPhase.cscFft(fig=figure, ax=axes_SR[n][3], csc=cscList[0], thetaRange=thetaRange)

            # if not pooled and not plotPlace:
            #     if gain_normalised:
            #         print 'INFO: gain normalisation is not implemented yet for plotSingleRunPhases function!'
            #         sys.exit()
                #plotSingleRunPhases(GC_object=GC[s], object_num=z, fig=fig, axes=axes)
            elif plotPlace:
                if gain_normalised:
                    print 'INFO: gain normalisation is not implemented yet for plotPlace function!'
                    sys.exit()
                spikesPlace.plotSpikePlaces(traj=traj, st=GC[s].placeCell, fig=fig, ttName=ttName, noSpeck=noSpeck, text=text, zaehler=zaehler)
            z += 1

    if not 'singleTrialNum' in locals():  # plots for the gain which was not at the last run (AxNum-1) !
        # placeCell.traj has trajectory where all points with running speed under threshold are removed!
        # if the entire turajectory is suppose to be illustrated, overwrite it with original traj:
        # placeCell.traj = traj  # careful! changes mean spikecount per run etc.!!

        fig, axes, fig_SR, axes_SR, pf_limits, wfit_xy, singleTrialNum, traj_xlim, max_FR, max_FR_in_pf, max_FR_ySum, \
        max_FR_ySum_in_pf, all_FR, all_FR_2d, occupancy_probability_ysum, spike_count_ySum, spike_count_pf_sum, \
        spike_count_perRun_pf_sum, phaseFit_pfLeft_pfRight_pooled, phaseFit_dictio, phaseFit_pfLeft_pfRight_SR, \
        phaseFit_dictio_SR, spike_count_SR = \
            plotPooledPhases(placeCell=placeCell, fig=fig,
                             axes=axes, text=True, Ax=Ax, singleRuns=singleRuns, gc_indexes=gc_indexes, pf_xlim=pf_xlim,
                             traj_xlim=traj_xlim, xbin_only=xbin_only)
        if singleRuns:                   # plots the FFT for the gain which was not at the last run!
            for n, figure in enumerate(fig_SR):
                if not axes_SR[n] == []:
                    spikesPhase.cscFft(fig=figure, ax=axes_SR[n][3], csc=cscList[0], thetaRange=thetaRange)

    # within each dictionary parameter all arrays have to have the length, for hickle to save it!
    # For that the longest array has to be determined and all othe ones have to be filled up with numpy.nan!

    len05 = []
    len15 = []
    for lf in numpy.arange(len(lfp_freq_gain05)):
        len05.append(len(lfp_freq_gain05[lf]))
        # len05.append(len(theta_power_gain05[lf]))
    for lf in numpy.arange(len(lfp_freq_gain15)):
        len15.append(len(lfp_freq_gain15[lf]))
        # len15.append(len(theta_power_gain15[lf]))

    for lf in numpy.arange(len(lfp_freq_gain05)):
        if len(lfp_freq_gain05[lf]) != max(len05):
            lfp_freq_gain05[lf] = numpy.append(lfp_freq_gain05[lf],
                                               numpy.repeat(numpy.nan, max(len05) - len(lfp_freq_gain05[lf])))
            speeds_virtual_gain05[lf] = numpy.append(speeds_virtual_gain05[lf],
                                                     numpy.repeat(numpy.nan, max(len05) - len(speeds_virtual_gain05[lf])))
            speeds_real_gain05[lf] = numpy.append(speeds_real_gain05[lf],
                                                  numpy.repeat(numpy.nan, max(len05) - len(speeds_real_gain05[lf])))
            # theta_power_gain05[lf] = numpy.append(theta_power_gain05[lf],
            #                                       numpy.repeat(numpy.nan, max(len05) - len(theta_power_gain05[lf])))
    for lf in numpy.arange(len(lfp_freq_gain15)):
        if len(lfp_freq_gain15[lf]) != max(len15):
            lfp_freq_gain15[lf] = numpy.append(lfp_freq_gain15[lf],
                                               numpy.repeat(numpy.nan, max(len15) - len(lfp_freq_gain15[lf])))
            speeds_virtual_gain15[lf] = numpy.append(speeds_virtual_gain15[lf],
                                                     numpy.repeat(numpy.nan, max(len15) - len(speeds_virtual_gain15[lf])))
            speeds_real_gain15[lf] = numpy.append(speeds_real_gain15[lf],
                                                  numpy.repeat(numpy.nan, max(len15) - len(speeds_real_gain15[lf])))
            # theta_power_gain15[lf] = numpy.append(theta_power_gain15[lf],
            #                                       numpy.repeat(numpy.nan, max(len15) - len(theta_power_gain15[lf])))

    lfp_freq = {'lfp_freq_gain05': lfp_freq_gain05, 'lfp_freq_gain15': lfp_freq_gain15}
    speeds = {'speeds_virtual_gain05': speeds_virtual_gain05, 'speeds_virtual_gain15': speeds_virtual_gain15,
              'speeds_real_gain05': speeds_real_gain05, 'speeds_real_gain15': speeds_real_gain15,
              'theta_power_gain05': theta_power_gain05, 'theta_power_gain15': theta_power_gain15,
              'delta_power_gain05': delta_power_gain05, 'delta_power_gain15': delta_power_gain15}

    return fig, fig_SR, gc_indexes, placeCell, transition_indexes, singleTrialNum, traj_xlim, max_FR, max_FR_in_pf, \
           max_FR_ySum, max_FR_ySum_in_pf, all_FR, all_FR_2d, pf_limits, gc_number, occupancy_probability_ysum, \
           spike_count_ySum, spike_count_pf_sum, spike_count_perRun_pf_sum, phaseFit_pfLeft_pfRight_pooled, \
           phaseFit_dictio, phaseFit_pfLeft_pfRight_SR, phaseFit_dictio_SR, lfp_freq, speeds, spike_count_SR


def getAllgainChanges(spiketrain, transitions):

    GC = []
    # last stimulus is len(stimuli.times)-1 and that should be a gain_middle, so the last gain_in parameter
    # is at len(stimuli.times)-2 and numpy.arange(x) starts with 0 and goes to x-1: here len(stimuli.times)-2!
    if transitions:
        stimulus_length = len(stimuli.times)-1
    # for non transitions each gain should be gain_in, which is the case except the last row. Here the last gain would
    # only the gain_middle. So in this case, the gainChange class has to go to the last stimulus row:
    else:
        stimulus_length = len(stimuli.times)

    for n in numpy.arange(stimulus_length):
        GC.append(gainChange(n, spiketrain))

    return GC


def getAllgainChangePCs(GC, mother_spiketrain):

    delete_elements = []
    for n in numpy.arange(len(GC)):
        if GC[n].t_middle != GC[n].t_stop:
            GC[n].cut_placeCells(mother_spiketrain)
            if not hasattr(GC[n], 'pc_gain_out'):
                print 'WARNING: Gain change object deleted because trajectory.py time slice was not possible!'
                delete_elements.append(n)
        else:
            print 'WARNING: Gain change object deleted because trajectory.py time slice was not possible!'
            delete_elements.append(n)
        if n == len(GC)-1:
            GC = numpy.delete(GC, delete_elements)
            # for d in numpy.arange(len(delete_elements)):
            #     del GC[delete_elements[d]]
    return GC


def getAllgainChangeCSCs(GC):

    delete_elements = []
    for n in numpy.arange(len(GC)):
        if GC[n].t_middle != GC[n].t_stop:
            GC[n].cut_csc()
        if n == len(GC)-1:
            GC = numpy.delete(GC, delete_elements)
    return GC


def ini_plotting():
    colors = custom_plot.colors
    grau = numpy.array([1, 1, 1])*.6
    grau2 = numpy.array([1, 1, 1])*.85

    Bildformat='pdf'

    fontsize=14.0

    mpl.rcParams['lines.markersize'] = 3
    mpl.rcParams['font.size'] = fontsize

    return colors, grau, grau2, Bildformat, fontsize


def commandline_params():
    grau = numpy.array([1, 1, 1])*.6
    dummy = sys.argv[1]				# second argument should be the name of the folder to load
    folderName = dummy.split('\\')[0]
    for d in dummy.split('\\')[1:]:
        folderName += '/'+d

    #parameters
    mazeName = 'gainChanger'           # maze type
    lang = 'e'
    transitions = False
    offset = 2
    gain_in = False
    gain_middle = False
    allRunns = False
    pooled = False
    plotPlace = False
    chic = False
    showFigs = True
    saveFigs = True
    saveAna = True
    update = False
    color1 = grau
    color2 = 'k'
    noSpeck = False
    useRecommended = False
    expType = 'vr'
    num_singleRuns = 0
    smooth = False

    return mazeName, lang, transitions, offset, gain_in, gain_middle, allRunns, pooled, plotPlace, chic, showFigs, \
        saveFigs, saveAna, update, color1, color2, noSpeck, useRecommended, expType, num_singleRuns, folderName, smooth


def get_params(tt, csc):
    # file endings
    cscName = csc
    if not cscName.endswith('.ncs'):
        cscName += '.ncs'
    TTName = tt
    if TTName == ['']:
        TTName = ''
    else:
        if type(TTName) == str:
            TTName = ast.literal_eval(tt)

        for i, tt in enumerate(TTName):
            if not tt.endswith('.t'):
                TTName[i] += '.t'

    return TTName, cscName


def getDataCheck(folderName):
    if os.path.isfile(folderName):
        sys.exit('Point to a folder not a single file.')
    elif not os.path.isdir(folderName):
        sys.exit('Folder or data name does not exist.')
    os.chdir(os.getcwd())


def cropData(eventData, traj, threshspeed):

    eventData.changeTimeUnit('s')
    stList.changeTimeUnit('s')
    cscList.changeTimeUnit('s')

    print 'changed eventData has timeUnit: ', eventData, eventData.timeUnit
    print 'changed traj has timeUnit', traj, traj.timeUnit

    eventData, traj = trajectory.align(eventData, traj, display=True)
    time_shift = eventData.t_start - traj.t_start  # time shift has to be added onto stimuli and events_position!
    traj.times = eventData.times  # eventData und csc have same time axis

    stimuli.times = stimuli.times+time_shift  # add time_shift to stimuli times and save changes in variable
    events.times = events.times+time_shift  # add time_shift to stimuli times and save changes in variable
    rewards_traj.times = rewards_traj.times + time_shift

    traj.threshspeed = threshspeed
    # getting e.g. x limits of the entire trajectory:
    traj.getTrajDimensions()


def getGainTransitionsAndOccurance(transitions, showFigs=True, saveFig=False, saveFolder=''):
    if transitions:
        paramsTransDict = stimuli.parameterTransitionsDict('gain', labelsize=9, showFigs=showFigs, saveFig=saveFig, saveFolder=saveFolder)
    else:
        paramsTransDict = stimuli.parameterDict('gain', labelsize=9, showFigs=showFigs, saveFig=saveFig, saveFolder=saveFolder)

    # Titles gives back an array of all gain transitions (or gains) and AxesNum gives the transition (gain) occurrence
    Titles = numpy.array(paramsTransDict.keys())
    AxesNum = numpy.array(paramsTransDict.values())

    return Titles, AxesNum, paramsTransDict


###################################################### classes
   
class gainChange:
    """Common base class for all gain changes"""

    def __init__(self, row, sp):
        
        global traj, stimuli, csc

        self.row = row

        gains = numpy.array(stimuli.getParameter('gain'))  # gains from stimuli file
        visible = numpy.array(stimuli.getParameter('visible'))  # visible (light=1 or dark=0) from stimuli file
        ol = numpy.array(stimuli.getParameter('open_loop'))  # open_loop (fake walk displayed = 1 or normal = 0) from stimuli file

        if traj.timeUnit == 'ms':
            traj.changeTimeUnit('s')
        elif stimuli.timeUnit == 'ms':
            stimuli.changeTimeUnit('s')
        elif events.timeUnit == 'ms':
            events.changeTimeUnit('s')

        self.gain_in = gains[row]
        self.t_start = signale.tools.findNearest(traj.times, stimuli.times[row])[1]
        self.t_RZ_exit = signale.tools.findNearest(traj.times, stimuli.times[row])[1]
        if len(visible) == len(gains):
            self.visible_in = visible[row]
        else:
            self.visible_in = 1
        if len(ol) == len(gains):
            self.ol_in = ol[row]
        else:
            self.ol_in = 0

        if row == len(stimuli.times)-2:  # last row is second last in stimuli file and ends where the traj ends
            self.gain_middle = gains[row+1]
            if len(visible) == len(gains):
                self.visible_middle = visible[row+1]
            else:
                self.visible_middle = 1
            if len(ol) == len(gains):
                self.ol_middle = ol[row+1]
            else:
                self.ol_middle = 1
            self.t_middle = signale.tools.findNearest(traj.times, stimuli.times[row+1])[1]
            self.index_middle = signale.tools.findNearest(events.times, stimuli.times[row+1])[0]
            self.t_stop = traj.times[len(traj.times)-1]
            self.index_start = signale.tools.findNearest(events.times, stimuli.times[row])[0]
            # trajectory doesnt stop with a new gain change, so index_stop doesnt exist for the last gain change. The
            # stop index is therefore set to a number, which can be filtered out easily later (e.g. in plotPhase)
            self.index_stop = -2  # len(events.times)-1

        elif row < len(stimuli.times)-2:
            self.gain_middle = gains[row+1]
            self.gain_stop = gains[row+2]
            if len(visible) == len(gains):
                self.visible_middle = visible[row+1]
                self.visible_stop = visible[row+2]
            else:
                self.visible_middle = 1
                self.visible_stop = 1
            if len(ol) == len(gains):
                self.ol_middle = ol[row+1]
                self.ol_stop = ol[row+2]
            else:
                self.ol_middle = 1
                self.ol_stop = 1
            self.t_middle = signale.tools.findNearest(traj.times, stimuli.times[row+1])[1]
            self.index_middle = signale.tools.findNearest(events.times, stimuli.times[row+1])[0]
            self.t_stop = signale.tools.findNearest(traj.times, stimuli.times[row+2])[1]
            self.index_stop = signale.tools.findNearest(events.times, stimuli.times[row+2])[0]
            # first gain starts actually with the trajectory, and no event it written for that. Thats why no index for
            # the event file exists, and the start index is set to an arbitrary number which can be filtered out easily
            # later (e.g. in plotPhase)
            if row == 0:
                self.index_start = -1
            else:
                self.index_start = signale.tools.findNearest(events.times, stimuli.times[row])[0]

        # only if no gain changes are looked at, but single gains instead and the last gain is suppose to be used for
        # self.placeCell and self.pc_gain_in
        elif row == len(stimuli.times)-1:
            self.t_stop = traj.times[len(traj.times)-1]
            self.t_middle = self.t_stop

    # the given spiketrain sp is a signale.place_cells.placeCell_linear and the time_sliced self.placeCell
    # is a signale.place_cells.placeCell. The place cells cut traj and its threshspeed is available.
    # The threshspeed is taken from the mother placeCell_linear

    def cut_placeCells(self, sp):

        self.placeCell = sp.time_slice(self.t_start, self.t_stop)
        self.pc_gain_in = sp.time_slice(self.t_start, self.t_middle, gain=self.gain_in)

        # if self.pc_gain_in.traj != 0 and gain_normalised:
        #     print 'max x place_______________________________________', max(self.pc_gain_in.traj.places[:, 0])
        #     print 'threshspeed used_________________________________________', self.pc_gain_in.traj.threshspeed
        #     self.pc_gain_in.traj.threshspeed *= self.gain_in

        # GET PLACECELLS WHICH ARE TIME OR PLACE ALIGNED FOR RZ EXIT!

        # get RZ x coordinates
        rz_xValues = numpy.array(rz)[:, 0]
        min_max = [numpy.min, numpy.max]
        # find if the crossed RZ is closest to the min or max of the track:
        self.rz_index = signale.tools.findNearest(rz_xValues, self.pc_gain_in.traj.places[:, 0][0])[0]
        # get turning point of trajectory piece:
        self.pc_gain_in.x_turn_index = numpy.where(self.pc_gain_in.traj.places[:, 0] == min_max[self.rz_index](
            self.pc_gain_in.traj.places[:, 0]))[0][0]
        # get trajectory index of the RZ exit:
        self.pc_gain_in.rz_exit_index = signale.tools.findNearest(self.pc_gain_in.traj.places[:, 0]
                                                                  [self.pc_gain_in.x_turn_index:],
                                                                  rz_xValues[self.rz_index])[0]+\
                                        self.pc_gain_in.x_turn_index
        # get RZ exit time to use for traj and spike alignement:
        self.pc_gain_in.rz_exit_time = self.pc_gain_in.traj.times[self.pc_gain_in.rz_exit_index]
        self.pc_gain_in.rz_exit_xplace = self.pc_gain_in.traj.places[:, 0][self.pc_gain_in.rz_exit_index]

        # Create new placeCell with time_offest (is time zero) as rz_exit_time, is when the RZ is left!!
        pc_gain_in_RZexit_time_aligned = self.pc_gain_in.subtract_time_offset(time_offset=self.pc_gain_in.rz_exit_time)

        # Create new placeCell with xtraj_offset (is new xtraj zero) as rz_exit_xplace (where the RZ is left)!!
        # For leftwards runs the right rz place (close to maximum x value) will be subtracted. The x values for the
        # run will therefore be negative and their sign has to be switched!
        signSwitch = [False, True]
        self.pc_gain_in_RZexit_aligned = pc_gain_in_RZexit_time_aligned.subtract_xtraj_offset(
            xtraj_offset=self.pc_gain_in.rz_exit_xplace, normalisation_gain=self.gain_in, changeSign=signSwitch[self.rz_index])

        # delete all spike_times and spike_places before RZ exit:
        st = self.pc_gain_in_RZexit_aligned.spiketrains

        for sk in numpy.arange(len(st)):  # s should be spikestrains for all, rightward and leftward runs
            arr = numpy.where(st[sk].spike_times < 0)[0]
            st[sk].spike_times = numpy.delete(st[sk].spike_times, arr)
            st[sk].spike_places = numpy.delete(st[sk].spike_places, arr, axis=0)

            if len(st[sk].spike_places):
                arr1 = numpy.where(st[sk].spike_places[:, 0] < 0)[0]
                st[sk].spike_times = numpy.delete(st[sk].spike_times, arr1)
                st[sk].spike_places = numpy.delete(st[sk].spike_places, arr1, axis=0)

        # after RZ substraction, normalise threshspeed:
        if not 'gain_normalised' in locals():
            gain_normalised = False
            'DANGER: NOT GAIN NORMALISED !'
        if self.pc_gain_in.traj != 0 and gain_normalised:
            self.pc_gain_in.traj.threshspeed /= self.gain_in

        # is always true if gain changes are looked at because then row end at len(stimuli.times)-2
        if self.row < len(stimuli.times)-1:
            self.pc_gain_out = sp.time_slice(self.t_middle, self.t_stop)
            if self.pc_gain_out.traj != 0 and gain_normalised:
                self.pc_gain_out.traj.threshspeed /= self.gain_middle

    def cut_csc(self):

        # cscList has cscID and csc
        for csc in cscList:
            self.csc = csc.time_slice(self.t_start, self.t_stop)
            self.csc_gain_in = csc.time_slice(self.t_start, self.t_middle)
            if self.row < len(stimuli.times)-1:
                self.csc_gain_out = csc.time_slice(self.t_middle, self.t_stop)

    def plotPhase(self, fig, ax):
        self.placeCell.phasePlot(traj=self.placeCell.traj, fig=fig, labelx=False, labely=False, ax=ax, lfp=self.csc, labelsize=8)

        # draw vertical lines in positions where gain_in, gain_middle and gain_stop was initiated
        if self.index_start == -1:
            ax.axvline(x=traj.places[0, 0], linewidth=1, color='g')
        else:
            ax.axvline(x=events.places[self.index_start, 0], linewidth=1, color='g')

        ax.axvline(x=events.places[self.index_middle, 0], linewidth=1, color='r')

        if self.index_stop == -2:
            ax.axvline(x=traj.places[len(traj.places)-1, 0], linewidth=1, color='k')
        else:
            ax.axvline(x=events.places[self.index_stop, 0], linewidth=1, color='k')

        fig.canvas.draw()
        custom_plot.huebschMachen(ax)

    def plotPlace(self, row, fig):
        pos = [(row+1)*.08, (row+1)*.21, .05, .13]  # [left, bottom, width, height]
        ax = fig.add_axes(pos)
        self.placeCell.plotSpikesvsPlace(fig=fig, ax=ax)
        custom_plot.huebschMachen(ax)


if __name__ == "__main__":


    ###################################################### plotting initialization

    colors, grau, grau2, Bildformat, fontsize = ini_plotting()

    ###################################################### commandline paramters

    mazeName, lang, transitions, offset, gain_in, gain_middle, allRunns, pooled, plotPlace, chic, showFigs, \
    saveFigs, saveAna, update, color1, color2, noSpeck, useRecommended, expType, num_singleRuns, \
    folderName, smooth = commandline_params()

    Gain = None
    gain_normalised = False
    normalisation_gain = 1.0
    rate = False
    Ax = None
    wfit = False
    placeCell = None
    # runs = None
    pf_xlim = False
    pooled = True
    singleRuns = False
    rausmap = False
    hickle_data = False

    for argv in sys.argv[2:]:
        if argv == 'rausmap':
            rausmap = True
        if argv == 'noShow':
            showFigs = False			# show pics
        if argv == 'saveFigs':
            saveFigs = True				# save pics
        if argv.startswith('csc:'):
            csc = argv.split(':')[-1]   # csc file to load
        if argv.startswith('tt:'):                      # Tetrode files to load, write e.g. as TT:['TT2_01.t']
            tt = argv.split('tt:')[1].strip('[').strip(']')
            tt = [s for s in tt.split(',')]
        if argv == 'noSpeck':
            noSpeck = True				            # only running spikes
        if argv == 'useRecommended':
            useRecommended = True                       # use recommendations from metadata.dat
        if argv.startswith('thetaRange:'):
            thetaRange = argv.split('thetaRange:')[1].strip('[').strip(']')   #write into terminal e.g. as thetaRange:'[6, 10]'
            thetaRange = [float(thetaRange.split(',')[0]), float(thetaRange.split(',')[1])]
        if argv.startswith('pf_xlim'):
            pf_xlim = argv.split('pf_xlim:')[1].strip('[').strip(']')   #write into terminal e.g. as pf_xlim:'[1.5, 1.6]'
            pf_xlim = [float(pf_xlim.split(',')[0]), float(pf_xlim.split(',')[1])]
        if argv.startswith('gainChange:'):
            gainChange = argv.split('gainChange:')[1].strip('[').strip(']')
            gain_in = float(gainChange.split(',')[0])
            gain_middle = float(gainChange.split(',')[1])
            print 'Gain change phases are plotted for gains ', gain_in, ' -> ', gain_middle
        if argv == 'allRunns':
            allRunns = True
            print 'Gain change phases are plotted for ALL GAIN CHANGES !!'
        if argv == 'pooled':
            pooled = True
            print 'Gain change phases are being POOLED for individual gain changes!'
        if argv.startswith('threshspeed:'):
            threshspeed = float(argv.split('threshspeed:')[-1])
        if argv == 'plotPlace':
            plotPlace = True
        if argv == 'transitions':
            transitions = True
        if argv.startswith('gain_in:'):
            gain_in = argv.split('gain_in:')[1].strip('[').strip(']')   #write into terminal e.g. as gain_in:'[1.0]'
            gain_in = float(gain_in.split(',')[0])
            if transitions:
                print 'Gain changes are plotted for gain in: ', gain_in
            else:
                print 'individual gains are plotted for gain: ', gain_in
        if argv.startswith('singleRuns'):  #'runs:'):
            # runs = argv.split('runs:')[1]   #write into terminal e.g. as runs:'all', runs:'right' or runs:'left'
            print 'Single runs are plotted!'  # for: ', runs, 'runs'
            singleRuns = True
            # if runs == 'all':
            #     runs = 0
            # elif runs == 'right':
            #     runs = 1
            # elif runs == 'left':
            #     runs = 2
            # else:
            #     print '''ERROR: runs has to be either runs:'all', runs:'right' or runs:'left' !'''
            #     sys.exit()
        if argv.startswith('smooth:'):
            smooth = argv.split('smooth:')[1].strip('[').strip(']')   #write into terminal e.g. as smooth:'[1.0]'
            smooth = float(smooth.split(',')[0])
        if argv == 'gain_normalised':
            gain_normalised = True
        if argv == 'rate':
            rate = True
        if argv == 'wfit':
            wfit = True
        if argv == 'hickle_data':
            hickle_data = True


    ###################################################### initialization

    if transitions:
        print 'WARNING: threshspeed is set to zero because there would have to be two different threshspeeds for both ' \
              'gains (before and after the gain change). This is not implemented yet!'
        threshspeed = 0.0

    # get parameters
    parameters = {'csc': 'CSC1.ncs', 'tt': '', 'thetaRange': [6, 10], 'threshspeed': 0.1, 'animal': '', 'depth': 0}
    if useRecommended:
        fileName = os.path.normpath(folderName)+'/metadata.dat'
    else:
        fileName = ''


    dictio, metadata = signale.get_metadata(fileName, parameters, locals())

    locals().update(dictio)
    if tt:
        TTName, cscName = get_params(tt=tt, csc=csc)
    else:
        TTName = None
        cscName = None


    ###################################################### get data

    getDataCheck(folderName=folderName)

    spikes, ID, cscID, cscList, stList, eventData, traj, rewards_traj, events, events_fileName, stimuli, \
    stimuli_fileName, stimuli_folder, main_folder, loaded_cleaned_events, loaded_cleaned_stimuli, hs, rz, \
    loadedSomething, cwd = getData(folderName=folderName, cscName=cscName, TTName=TTName)

    ###################################################### crop data

    if hasattr(eventData, 'changeTimeUnit'):
        cropData(eventData=eventData, traj=traj, threshspeed=threshspeed)
    else:
        print 'WARNING: eventData object has no attribute changeTimeUnit'

    ###################################################### clean stimuli and event files

    if stimuli_fileName.endswith('stimuli.tsv'):
        print 'CLEANING EVENTS AND STIMULI NOW!'
        events, stimuli = signale.cleaner.clean(events, stimuli, traj, rewards_traj, main_folder, stimuli_folder,
                                                stimuli_fileName, events_fileName, hs, rz)
    else:
        print 'NOT CLEANING EVENTS AND STIMULI, THEY HAVE ALREADY BEEN CLEANED!'

    ###################################################### only proceed if tetrode files are available

    if ID == -1:
        sys.exit('The folders do not contain tetrode data (t files)!')

    ###################################################### set transitions to False if there is more than one hotspot

    # if len(hs) == 0 there are no set hotspots and the reward zones were used as the gain changing 'hotspots'

    if transitions and len(hs) > 1 or transitions and len(hs) == 0 and len(rz) > 1:
        print 'INFO: there is more then one Hotspot and plotting transitions in this case doesnt make sense'
        print 'SETTING TRANSITIONS TO FALSE!'
        transitions = False

    ###################################################### get gain changes and do the execute the MAIN FUNCTIONS

    Titles, AxesNum, paramsTransDict = getGainTransitionsAndOccurance(transitions=transitions, showFigs=True,
                                                                      saveFig=True, saveFolder=folderName)

    for zaehler, pc in enumerate(stList):
        print 'T-file:', pc.tags['file']

        # in spikesPhase.prepareAndPlotPooledPhases the noSpeck version plots only the pc running spikes depending on
        # its pc.threshspeed and the normal version plots all spikes in grey and the running spikes in red on top
        pc.traj = traj
        if smooth:
            print 'INFO: trajectory is smoothed with the smoothing kernel width of ', smooth, ' s'
            pc.traj.smooth(smooth)
        pc.thetaRange = (thetaRange[0], thetaRange[1])

        GC = getAllgainChanges(pc, transitions=transitions)
        GC = getAllgainChangePCs(GC=GC, mother_spiketrain=pc)
        GC = getAllgainChangeCSCs(GC=GC)
        ttName = stList.tags[zaehler]['file'].split('.')[0]

        old_start_gains = []
        traj_xlim = 0
        max_FRates = {}
        all_FRates = {}
        all_FRates2d = {}
        pf_Limits = {}
        occupancy = {}
        spike_count = {}
        phases = {}

        if not transitions and rate:
            # prepare firerate over position plot for all gains
            f, (axA, axB, axC) = pl.subplots(1, 3, sharey=True, figsize=(12, 6))
            Ax = [axA, axB, axC]
            Ax[0].set_ylabel('Firing rate in Hz')
            dummy_titles = ['All directions', 'Rightward runs', 'Leftward runs']
            for i in [0, 1, 2]:
                if i == 0:
                    xlab = 'Position (m)'
                else:
                    xlab = 'Position from start point (m)'
                Ax[i].set_xlabel(xlab)
                Ax[i].set_title(dummy_titles[i])
        lines = []
        labels = []

        # main loop for plotting data!
        for PT in numpy.arange(len(paramsTransDict)):  # go through available gainchange events calulated by the Dict
            Title = Titles[PT]
            AxNum = AxesNum[PT]
            if transitions:
                if gain_normalised:
                    print 'INFO: for gain transitions, the gain normalisation is not implemented yet!'
                    sys.exit()
                # find the gain_in and gain_middle from dictionary string
                Grun = 0
                GC_in = float(re.findall(r"[-+]?\d*\.\d+|\d+", Title)[0])
                GC_middle = float(re.findall(r"[-+]?\d*\.\d+|\d+", Title)[1])
                # define running direction around gain change:
                # if bla:
                #   run_direction = 'rightwards'
                # else:
                #   run_direction = 'leftwards'
                # TODO Gain needs to be in the order of the plot!
                # if run_direction == 'rightwards':
                Gain = (GC_in, GC_middle)
                # elif run_direction == 'leftwards'
                #   Gain = (GC_middle, GC_in)
                # if only a specific gain pair is wanted, they will have to equal the Dict pair
                if gain_in and gain_middle:
                    if float(gain_in) == Gain[0] and float(gain_middle) == Gain[1]:
                        pass
                    else:
                        continue
            else:
                Grun = Title
                if gain_in:
                    if float(gain_in) == Grun:
                        Gain = Grun
                    elif float(gain_in) != Grun:
                        continue
                else:
                    Gain = Grun
                if gain_normalised:
                    normalisation_gain = Grun

            if not Gain in old_start_gains:
                fig, fig_SR, gc_indexes, placeCell, transition_indexes, singleTrialNum, traj_xlim, max_FR, \
                max_FR_in_pf, max_FR_ySum, max_FR_ySum_in_pf, all_FR, all_FR_2d, pf_limits, gc_number, \
                occupancy_probability_ysum, spike_count_ySum, spike_count_pf_sum, spike_count_perRun_pf_sum, \
                phaseFit_pfLeft_pfRight_pooled, phaseFit_dictio, phaseFit_pfLeft_pfRight_SR, phaseFit_dictio_SR, \
                lfp_freq, speeds, spike_count_SR = \
                    plotPhasesOrPlaces(GC=GC, Ax=Ax, traj_xlim=traj_xlim, pf_xlim=pf_xlim, xbin_only=True)

                old_start_gains.append(Gain)
                max_FRates.update(max_FR)
                max_FRates.update(max_FR_in_pf)
                max_FRates.update(max_FR_ySum)
                max_FRates.update(max_FR_ySum_in_pf)
                all_FRates.update(all_FR)
                all_FRates2d.update(all_FR_2d)
                pf_Limits.update({'pf_limits_allRuns_gain_'+str(Grun): pf_limits[0],
                                  'pf_limits_rightRuns_gain_'+str(Grun): pf_limits[1],
                                  'pf_limits_leftRuns_gain_'+str(Grun): pf_limits[2],
                                  'pf_width_allRuns_gain_'+str(Grun): numpy.diff(pf_limits[0])[0],
                                  'pf_width_rightRuns_gain_'+str(Grun): numpy.diff(pf_limits[1])[0],
                                  'pf_width_leftRuns_gain_'+str(Grun): numpy.diff(pf_limits[2])[0],
                                  'traj_xlim_'+str(Grun): traj_xlim})
                occupancy.update({'occupancy_probability_ysum_allRuns_gain_'+str(Grun): occupancy_probability_ysum[0],
                                  'occupancy_probability_ysum_rightRuns_gain_'+str(Grun): occupancy_probability_ysum[1],
                                  'occupancy_probability_ysum_leftRuns_gain_'+str(Grun): occupancy_probability_ysum[2]})
                spike_count.update({'spike_count_ysum_allRuns_gain_'+str(Grun): spike_count_ySum[0],
                                    'spike_count_ysum_rightRuns_gain_'+str(Grun): spike_count_ySum[1],
                                    'spike_count_ysum_leftRuns_gain_'+str(Grun): spike_count_ySum[2],
                                    'spike_count_pf_sum_allRuns_gain_'+str(Grun): spike_count_pf_sum[0],
                                    'spike_count_pf_sum_rightRuns_gain_'+str(Grun): spike_count_pf_sum[1],
                                    'spike_count_pf_sum_leftRuns_gain_'+str(Grun): spike_count_pf_sum[2],
                                    'spike_count_perRun_pf_sum_allRuns_gain_'+str(Grun): spike_count_perRun_pf_sum[0],
                                    'spike_count_perRun_pf_sum_rightRuns_gain_'+str(Grun): spike_count_perRun_pf_sum[1],
                                    'spike_count_perRun_pf_sum_leftRuns_gain_'+str(Grun): spike_count_perRun_pf_sum[2],
                                    'spike_count_SR_allRuns_gain_'+str(Grun): spike_count_SR[0],
                                    'spike_count_SR_rightRuns_gain_'+str(Grun): spike_count_SR[1],
                                    'spike_count_SR_leftRuns_gain_'+str(Grun): spike_count_SR[2]})

                phases.update(phaseFit_pfLeft_pfRight_pooled)
                phases.update(phaseFit_dictio)
                phases.update(phaseFit_pfLeft_pfRight_SR)
                phases.update(phaseFit_dictio_SR)

                # raw csc traces for gain changes or individual gain is done after this loop, when all plotted GC
                # indexes are saved in the variable gc_indexes
                # fig1 = rawCscTraces(gc_indexes=gc_indexes)

                # set figure Title

                run_direction = ''
                if transitions:
                    AxNum = len(transition_indexes)
                elif singleRuns:  #not pooled:
                    AxNum_SR = [len(singleTrialNum[0]), len(singleTrialNum[1]), len(singleTrialNum[2])]
                    run_directions = ['All run directions, ', 'Rightward runs, ', 'Leftward runs, ']

                    # direction = ['All run directions, ', 'Rightward runs, ', 'Leftward runs, ']
                    # run_direction = direction[runs]

                if animal and depth:

                    fig.suptitle(run_direction + 'Animal:' + str(animal) + ', Tetrode depth:' + str(depth) +
                                 ' $\mu m$, ' + 'VR gain:' + str(Title) + ' (' + str(gc_number) + 'x)', fontsize=12)

                    if singleRuns:
                        for num, fi in enumerate(fig_SR):
                            fi.suptitle(run_directions[num] + 'Animal:' + str(animal) + ', Tetrode depth:' + str(depth) +
                                 ' $\mu m$, ' + 'VR gain:' + str(Title) + ' (' + str(gc_number) + 'x)', fontsize=12)

                else:
                    fig.suptitle('Gain: '+str(Title))
                    if singleRuns:
                        for num, fi in enumerate(fig_SR):
                            fi.suptitle('Gain: '+str(Title))

                if not transitions and rate and animal and depth:
                    f.suptitle('Animal:' + str(animal) + ', Tetrode depth:'+str(depth)+' $\mu m$', fontsize=12)

                # save the plotted figures

                if not rausmap:

                    name = folderName.split('/olivia/')[0]+'/olivia/hickle/Plots/'+folderName.split('/')[-4]+'_'+\
                           folderName.split('/')[-3]+'_'+folderName.split('/')[-2]+'_'
                if rausmap:

                    name = folderName.split('/olivia/')[0]+'/olivia/hickle/cells_not_used_79/Plots/'+\
                           folderName.split('/')[-4]+'_'+\
                           folderName.split('/')[-3]+'_'+folderName.split('/')[-2]+'_'

                if hickle_data:
                    FolderNames = [folderName, name]
                else:
                    FolderNames = [folderName]
                for i in numpy.arange(len(FolderNames)):
                    if saveFigs and transitions:
                        print 'Phase plot saved to:', FolderNames[i]+ttName+'_GC_'+str(GC_in)+'_'+str(GC_middle)+'_spikePhase.pdf'
                        fig.savefig(FolderNames[i]+ttName+'_GC_'+str(GC_in)+'_'+str(GC_middle)+'_spikePhase.pdf', format='pdf')
                        #print 'csc gain change plot saved to:', FolderNames[i]+ttName+'_GC_'+str(GC_in)+'_'+str(GC_middle)+'_csc.pdf'
                        #fig1.savefig(FolderNames[i]+ttName+'_GC_'+str(GC_in)+'_'+str(GC_middle)+'_csc.pdf', format='pdf')
                    elif saveFigs and not transitions:
                        if not gain_normalised:  # and pooled:
                            print 'Picture saved to:', FolderNames[i]+ttName+'_Gain_'+str(Grun)+'_spikePhase.pdf'
                            fig.savefig(FolderNames[i]+ttName+'_Gain_'+str(Grun)+'_spikePhase.pdf', format='pdf')
                            #print 'csc gain plot saved to:', FolderNames[i]+ttName+'_Gain_'+str(Grun)+'_csc.pdf'
                            #fig1.savefig(FolderNames[i]+ttName+'_Gain_'+str(Grun)+'_csc.pdf', format='pdf')
                        if not gain_normalised and singleRuns:  # not pooled:
                            direction = ['all', 'right', 'left']
                            for num, fi in enumerate(fig_SR):
                                print 'Picture saved to:', FolderNames[i]+ttName+'_Gain_'+str(Grun)+'_spikePhase_singleRuns_'+direction[num]+'.pdf'
                                fi.savefig(FolderNames[i]+ttName+'_Gain_'+str(Grun)+'_spikePhase_singleRuns_'+direction[num]+'.pdf', format='pdf')

                        if gain_normalised and singleRuns:  # not pooled:
                            direction = ['all', 'right', 'left']
                            for num, fi in enumerate(fig_SR):
                                print 'Picture saved to:', FolderNames[i]+ttName+'_Gain_'+str(Grun)+'_spikePhase_singleRuns_'+\
                                                           direction[num]+'_gain_normalised.pdf'
                                fi.savefig(FolderNames[i]+ttName+'_Gain_'+str(Grun)+'_spikePhase_singleRuns_'+direction[num]+
                                            '_gain_normalised.pdf', format='pdf')

                        if gain_normalised:
                            print 'Picture saved to:', FolderNames[i]+ttName+'_Gain_'+str(Grun)+'_spikePhase_gain_normalised.pdf'
                            fig.savefig(FolderNames[i]+ttName+'_Gain_'+str(Grun)+'_spikePhase_gain_normalised.pdf', format='pdf')

                #show plotted figures, except if noShow (sets showFigs=False) was stated in command line
                if not showFigs:
                    #pl.ioff()
                    #for f in [fig, fig1]:
                    #pl.close(fig)
                    pl.close('all')
                else:
                    pl.show()

        if not transitions and rate:
            # fix the legend for the firerate over position plot for all gains
            l = []
            for i in numpy.arange(len(lines)):
                l.append(lines[i][0])
            f.legend(tuple(l), tuple(labels), loc=1, prop={'size': 13})
            f.subplots_adjust(bottom=0.15, right=0.8)
            if gain_normalised:
                f.text(.995, .01, folderName+ttName+'_FR-Position_gain_normalised.pdf', fontsize=fontsize-6,
                       horizontalalignment='right')
            else:
                f.text(.995, .01, folderName+ttName+'_FR-Position.pdf', fontsize=fontsize-6, horizontalalignment='right')
            if smooth:
                f.text(.995, .04, 'Trajectory smoothing window: '+str(smooth)+' s', fontsize=fontsize-6,
                         horizontalalignment='right')

            if not gain_normalised:
                print 'Picture saved to:', folderName+ttName+'_FR-Position.pdf'
                f.savefig(folderName+ttName+'_FR-Position.pdf', format='pdf')

                if hickle_data:
                    print 'And to: ', name+ttName+'_FR-Position.pdf'
                    f.savefig(name+ttName+'_FR-Position.pdf', format='pdf')

            else:
                print 'Picture saved to:', folderName+ttName+'_FR-Position_gain_normalised.pdf'
                f.savefig(folderName+ttName+'_FR-Position_gain_normalised.pdf', format='pdf')

                if hickle_data:
                    print 'And to: ', name+ttName+'_FR-Position_gain_normalised.pdf'
                    f.savefig(name+ttName+'_FR-Position_gain_normalised.pdf', format='pdf')

        RZ_exit_aligned_sTimes = [[], [], []]
        lengths_times = [[], [], []]
        RZ_exit_aligned_sPlaces = [[], [], []]

        for g in numpy.arange(len(GC)):
            for i in [0, 1, 2]:
                if len(GC[g].pc_gain_in_RZexit_aligned.spiketrains[i].spike_times):
                    time_arr = numpy.array(GC[g].pc_gain_in_RZexit_aligned.spiketrains[i].spike_times)
                    RZ_exit_aligned_sTimes[i].append(time_arr)
                    lengths_times[i].append(len(time_arr))
                    RZ_exit_aligned_sPlaces[i].append(numpy.array(GC[g].pc_gain_in_RZexit_aligned.spiketrains[i].spike_places[:, 0]))

        if len(lengths_times[0]):
            max_1 = max(lengths_times[0])
        else:
            max_1 = 0

        if len(lengths_times[1]):
            max_2 = max(lengths_times[1])
        else:
            max_2 = 0

        if len(lengths_times[2]):
            max_3 = max(lengths_times[2])
        else:
            max_3 = 0

        max_len = [max_1, max_2, max_3]

        # within each dictionary parameter all arrays have to have the length, for hickle to save it!
        # For that the longest array has to be determined and all othe ones have to be filled up with numpy.nan!

        for i in [0, 1, 2]:
            for l in numpy.arange(len(RZ_exit_aligned_sTimes[i])):
                if len(RZ_exit_aligned_sTimes[i][l]) != max_len[i]:
                    RZ_exit_aligned_sTimes[i][l] = numpy.append(RZ_exit_aligned_sTimes[i][l],
                                        numpy.repeat(numpy.nan, max_len[i] - len(RZ_exit_aligned_sTimes[i][l])))
                    RZ_exit_aligned_sPlaces[i][l] = numpy.append(RZ_exit_aligned_sPlaces[i][l],
                                        numpy.repeat(numpy.nan, max_len[i] - len(RZ_exit_aligned_sPlaces[i][l])))

        RZ_exit_aligned_sTimes_sPlaces = {'RZ_exit_aligned_all_spikeTimes': RZ_exit_aligned_sTimes[0],
                                          'RZ_exit_aligned_right_spikeTimes': RZ_exit_aligned_sTimes[1],
                                          'RZ_exit_aligned_left_spikeTimes': RZ_exit_aligned_sTimes[2],
                                          'RZ_exit_aligned_all_spikeXplaces': RZ_exit_aligned_sPlaces[0],
                                          'RZ_exit_aligned_right_spikeXplaces': RZ_exit_aligned_sPlaces[1],
                                          'RZ_exit_aligned_left_spikeXplaces': RZ_exit_aligned_sPlaces[2]}

        # pf_info = dict(max_FRates.items() + pf_Limits.items() + all_FRates.items() + all_FRates2d.items() +
        #                occupancy.items() + spike_count.items() + RZ_exit_aligned_sTimes_sPlaces.items() +
        #                lfp_freq.items() + speeds.items())

        pf_info = dict(max_FRates.items() + pf_Limits.items() + all_FRates.items() + all_FRates2d.items() +
                       occupancy.items() + spike_count.items() + phases.items() +
                       RZ_exit_aligned_sTimes_sPlaces.items() + lfp_freq.items() + speeds.items())

        if hickle_data:

            if gain_normalised:
                hkl_name = '_PF_info_normalised.hkl'
            else:
                hkl_name = '_PF_info.hkl'

            print 'dumping info.hkl to ', folderName+ttName+hkl_name

            if not rausmap:
                hickle.dump(pf_info, folderName+ttName+hkl_name, mode='w')
                print 'dumping info.hkl to ', folderName.split('/olivia/')[0]+'/olivia/hickle/'+\
                                              folderName.split('/')[-4]+'_'+\
                                              folderName.split('/')[-3]+'_'+folderName.split('/')[-2]+'_'+ttName+hkl_name

                hickle.dump(pf_info, folderName.split('/olivia/')[0]+'/olivia/hickle/'+folderName.split('/')[-4]+'_'+
                            folderName.split('/')[-3]+'_'+folderName.split('/')[-2]+'_'+ttName+hkl_name, mode='w')

            if rausmap:

                print 'dumping info.hkl to ', folderName.split('/olivia/')[0]+'/olivia/hickle/cells_not_used_79/'+\
                                              folderName.split('/')[-4]+'_'+\
                                              folderName.split('/')[-3]+'_'+folderName.split('/')[-2]+'_'+ttName+hkl_name

                hickle.dump(pf_info, folderName.split('/olivia/')[0]+'/olivia/hickle/cells_not_used_79/'+
                            folderName.split('/')[-4]+'_'+
                            folderName.split('/')[-3]+'_'+folderName.split('/')[-2]+'_'+ttName+hkl_name, mode='w')

    os.chdir(cwd)

