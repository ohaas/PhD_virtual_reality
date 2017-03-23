"""
For looking at LFP and running speed for real environments.
"""
__author__ = ("KT")
__version__ = "2.0.1, August 2014"



import sys, os, inspect, struct

# add additional custom paths
extraPaths=["/home/thurley/python/lib/python2.5/site-packages/", \
    "/home/thurley/python/lib/python2.6/site-packages/", \
    "/home/thurley/python/lib/python2.7/dist-packages/", \
    os.path.join(os.path.abspath(os.path.dirname(__file__)), '../scripts')]
for p in extraPaths:
    if not sys.path.count(p):
        sys.path.append(p)

import numpy, scipy

import NeuroTools.signals as NTsig
import signale, trajectory, custom_plot



###################################################### plotting initialization

import matplotlib as mpl
import matplotlib.pyplot as pl


fontsize=14.0

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'


colors=['#FF0000','#0000FF','#008000','#00FFFF','#FF00FF','#EE82EE',
        '#808000','#800080','#FF6347','#FFFF00','#9ACD32','#4B0082',
        '#FFFACD','#C0C0C0','#A0522D','#FA8072','#FFEFD5','#E6E6FA',
        '#F1FAC1','#C5C5C5','#A152ED','#FADD72','#F0EFD0','#EEE6FF',
        '#01FAC1','#F5F5F5','#A152FF','#FAFD72','#F0EFDF','#EEEFFF',
        '#F1FA99','#C9C9C9','#A152DD','#FA5572','#FFFFD0','#EDD6FF']


###################################################### commandline parameters


dummy = sys.argv[1]				# second argument should be the name of the folder to load
folderName = dummy.split('\\')[0]
for d in dummy.split('\\')[1:]:
    folderName+='/'+d

# parameters
useRecommended = False
showHilbertPhase=True
onlyRunning = False
TTName = '.t'
time_closeUp = None
expType = 'real'
for argv in sys.argv[2:]:
    if argv.startswith('csc:'):
        cscName=argv.split(':')[-1] + '.ncs'       # csc file to load
    if argv == 'useRecommended':
        useRecommended = True                       # use recommendations from metadata.dat
    if argv.startswith('thetaRange:'):
        thetaRange = argv.split('thetaRange:')[1].strip('[').strip(']')   #write into terminal e.g. as thetaRange:'[6, 10]'
        thetaRange = thetaRange.split(',')
        thetaRange = [int(thetaRange[0]), int(thetaRange[1])]
    if argv.startswith('TT:'):                      # Tetrode files to load, write e.g. as TT:['TT2_01.t']
        TTName = argv.split('TT:')[1].strip('[').strip(']')
        TTName = [s for s in TTName.split(',')]
    if argv == 'noPhases':
        showHilbertPhase=False
    if argv.startswith('expType:'):
        expType = argv.split(':')[-1]
    if argv == 'onlyRunning':
        onlyRunning = True
    if argv.startswith('time_closeUp:'):
        time_closeUp = argv.split('time_closeUp:')[1].strip('[').strip(']')   #write into terminal e.g. as thetaRange:'[6, 10]'
        time_closeUp = time_closeUp.split(',')
        time_closeUp = [int(time_closeUp[0]), int(time_closeUp[1])]				

###################################################### initialization


# initialize in order to make them available globally
eventData = None
eventDataList = []
cscID=-1
cscList = signale.NeuralynxCSCList()
traj = None
spikes=[]
ID=-1
nvtID=-1
stList=signale.spikezugList(t_start=None, t_stop=None, dims=[2])



cwd=os.getcwd()

if useRecommended:
    if os.path.isfile(folderName+'metadata.dat'):
        print 'Loading metadata:'
        metadata = signale.io._read_metadata(folderName+'metadata.dat', showHeader=True)
        try:
            cscName
            print 'Taking given csc data:', cscName
        except NameError:
            if metadata.has_key('csc'):
                cscName = metadata['csc']
                print 'Taking csc data listed in metadata.dat! csc:', cscName
            else:
                print 'No csc name given!'
        try:
            thetaRange
            print 'Taking given thetaRange data:', thetaRange
        except NameError:
            if metadata.has_key('thetaRange'):
                thetaRange = metadata['thetaRange']
                print 'Taking thetaRange data listed in metadata.dat:', thetaRange

        if TTName == '.t' and metadata.has_key('tt'):
            exec 'TTName =' + metadata['tt']
            print 'Taking tetrode data listed in metadata.dat! TT:', TTName
        if metadata.has_key('animal'):
            animal=metadata['animal']
        if metadata.has_key('depth'):
            depth=metadata['depth']
    else:
        print 'NOTE: There is no metadata.dat. Proceeding without instead.'


try:
   thetaRange
except NameError:
    thetaRange = [6, 12]


###################################################### functions

def getData(folderName):
    global cscID, eventData, cscList, traj, ID

    if os.path.isdir(folderName):
        dirList=os.listdir(folderName)
        os.chdir(folderName)
    else:
        dirList = [folderName]

    for item in dirList:
        if os.path.isfile(item):
            if item.endswith('.nev'):
                print 'loading', item , 'from folder: '+folderName
                eventData = signale.load_nevFile(item, showHeader=True)
            elif item.endswith(cscName):# or item.endswith('2.ncs'):
                print 'loading', item , 'from folder: '+folderName
                csc = signale.load_ncsFile(item, showHeader=True)
                cscID += 1
                cscList.append(cscID, csc)
                cscList.addTags(cscID, file=item, dir=folderName)
            elif (TTName.__class__ == list and item in TTName):
                print 'loading', item , 'from folder: '+folderName
                spikes = signale.load_tFile(item, showHeader=True)
                ID += 1
                stList.__setitem__(ID, spikes)
                stList.addTags(ID, file=item, dir=folderName)
            # real
            elif expType == 'real':
                if item.endswith('.nvt'):# or item.endswith('2.ncs'):
                    print 'loading', item , 'from folder: '+folderName
                    loadedSomething = True
                    traj = trajectory.load_nvtFile(item, 'linearMaze', showHeader=False)
                    # HDtraj = traj[1]        # head direction
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


def reorderEventData():
    global eventData, eventDataList

    indices = findall(eventData.eventStrings, 'Starting Recording')     # find all recording occasions in the file
    if not eventData.eventStrings.__len__()-1 in indices:               # in case there are more events than just the start etc.
        indices.append(eventData.eventStrings.__len__()-1)
    indices.append(0)
    indices.sort()

    for i, index1 in enumerate(indices[:-1]):
        index2 = indices[i+1]+1
        print index1, index2, eventData.times, eventData.eventStrings
        nev = signale.NeuralynxEvents(eventData.times[index1:index2], eventData.eventStrings[index1:index2])
        nev.tags['file'] = eventData.tags['file']
        nev.tags['path'] = eventData.tags['path']
        nev.purge(2)     # Remove the first two entries, when the data was recorded in the VR setup
                         #       since they are just initialization.
        eventDataList.append(nev)


def findall(list, value, start=0):
    indices = []
    i = start - 1
    try:
        while 1:
            i = list.index(value, i+1)
            indices.append(i)
    except ValueError:
        pass

    return indices



def prepareData():

    for csc in cscList:
        csc.filter(thetaRange[0], thetaRange[1])
        csc.hilbertTransform()


###################################################### main

getData(folderName)                   # load data files
os.chdir(cwd)
prepareData()


if not cscID+1:
    sys.exit('The given folder does not contain CSC data!')


eventData.changeTimeUnit('s')
cscList.changeTimeUnit('s')
stList.changeTimeUnit('s')


# real
if expType == 'real':
    # set begining to zero time
    #for st in stList:
    #    st.spike_times -= st.t_start
    #stList._spikezugList__recalc_startstop()

    # cut away all earlier spikes
    #stList = stList.time_slice(0., stList.t_stop)

    # change to seconds since trajectories are also stored in seconds
    stList._spikezugList__recalc_startstop()
    traj.purge(1)   # noch aktuell? mal testen, denn in spikesPhase wird das nicht benutzt,
                    # wird es das in spikesPlace?
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
    eventData, traj = trajectory.align(eventData, traj, display=True)
    traj.times = eventData.times

    #stList = stList.time_slice(0, 20000)      # reduce data size a bit




###################################################### csc and speed


csc = cscList[0]
#traj.time_offset(6.5)


t, s = traj.getSpeed()
t -= traj.times[0]

fig = pl.figure()
ax = fig.add_subplot(111)
ax.plot(csc.time_axis(), csc.signal, 'r')
ax.set_xlim(t[0], t[-1])

ax2 = ax.twinx()
for tl in ax2.get_yticklabels():
    tl.set_color('k')
ax2.plot(t, s, color='k', alpha=0.5)
ax2.set_xlabel('Time ('+traj.timeUnit+')')
ax2.set_ylabel('Speed [('+traj.spaceUnit+'/'+traj.timeUnit+')')
ax2.set_xlim(t[0], t[-1])
ax2.set_ylim(0, s.mean()+4*s.std())



#--- spectrogram
yes=1
minFreq = thetaRange[0]
maxFreq = thetaRange[1]
if yes:
    csc.changeTimeUnit('ms')
    stList.changeTimeUnit('ms')


    # determine times of cycle beginnings
    # NOTE: be careful, depends on a time_offset!
    cycles=numpy.array([])
    phaseDiff=numpy.diff(csc.hilbertPhase)
    for i, p in enumerate(phaseDiff):
        if p<0:
            cycles=numpy.append(cycles, csc.time_axis()[i])

    Pxx, freqs, t0 = csc.spectrogram(minFreq=0, maxFreq=30, windowSize=8192) #12288

    if not time_closeUp:
        time_closeUp = [t0.min(), t0.max()]

        fig = pl.figure(102, figsize = (8, 7))
        pos = [.09, .1, .8, .55]
        pos3 = list(pos)
        pos3[1] = pos[1] + pos[3] + .05
        pos3[3] = .25
    else:
        fig = pl.figure(102, figsize = (10, 10))
        pos = [.1, .06, .8, .35]
        pos2 = list(pos)
        pos2[1] = pos[1] + pos[3] + .05
        pos3 = list(pos2)
        pos3[1] = pos2[1] + pos2[3] + .025
        pos3[3] = .15
    ax = fig.add_axes(pos3)

    if showHilbertPhase:
        for i in range(0, cycles.shape[-1]-1, 2):
            ax.axvspan(cycles[i], cycles[i+1], facecolor=[.7, .7, .7], edgecolor=[.6, .6, .6], alpha=0.6, zorder=.1)


    # plot csc trace
    ax.plot(csc.time_axis(), csc.signal, 'k', linewidth=1)
    ax.plot(csc.time_axis(), csc.signal_filtered, color=[0, .75, 1], linewidth=2)


    ax.set_xlim(time_closeUp[0], time_closeUp[1])
    custom_plot.turnOffAxes(ax, spines=['left'])
    ax.set_ylim(numpy.percentile(csc.signal, .5), numpy.percentile(csc.signal, 99.5))
    ax.set_xticklabels([])
    custom_plot.huebschMachen(ax)
#    custom_plot.turnOffAxes(ax, spines=['top'])

    ax2 = ax.twinx()
    for tl in ax2.get_yticklabels():
        tl.set_color('k')
    ax2.plot(t*1000, s/(s.mean()+4*s.std()), color='k', alpha=0.5)
    ax2.plot((traj.times-traj.times[0])*1000, traj.getXComponents()/traj.getXComponents().max(), color='g', alpha=0.5)
    ax2.set_xlim(time_closeUp[0], time_closeUp[1])
    custom_plot.turnOffAxes(ax2, spines=['right'])

#    ax2.plot(csc.time_axis(), csc.hilbertPhase,'g', alpha=.3, zorder=.1)
#    ax2.set_ylim(0, 360)
#    ax2.set_yticks(numpy.arange(0, 361, 90))
#    ax2.set_ylabel('Theta phase (deg)')


    if stList.__len__():
        # add spikes of example cell(s)
        ax2 = ax.twinx()
        for tl in ax2.get_yticklabels():
            tl.set_color('k')
        ax.set_xlim(time_closeUp[0], time_closeUp[1])
        ax2.set_ylim(0, stList.__len__()+10)
        ax2.set_yticklabels([])
        custom_plot.huebschMachen(ax2)

        stList.changeTimeUnit('s')
        for i, st in enumerate(stList):
            st.traj = traj
            st.getRunningSpikes(threshspeed = .2)

            if not onlyRunning:
                spikes = st.spike_times.copy()*1000.
                spikes -= csc.times_start
                ax2.plot(spikes, numpy.ones(spikes.shape)+i, '|', color=colors[i], alpha=.5, markeredgewidth=3, markersize=10)

            spikes = st.run_spikeTimes*1000.
            spikes -= csc.times_start
            ax2.plot(spikes, numpy.ones(spikes.shape)+i, '|', color=colors[i], markeredgewidth=3, markersize=10)


    if time_closeUp[0] != t0.min and time_closeUp[1] != t0.max():

        # closeUp spectrogram
        ax = fig.add_axes(pos2)
        ax.pcolormesh(t0, freqs, Pxx, vmin=Pxx.min(), vmax=numpy.percentile(Pxx,99))
        ax.set_xlim(time_closeUp[0], time_closeUp[1])
        ax.set_ylim(freqs.min(), freqs.max())
        ax.set_xticklabels(ax.get_xticks()/1000.)
        ax.set_ylabel('Frequency (Hz)')

        ax2 = ax.twinx()
        for tl in ax2.get_yticklabels():
            tl.set_color('k')
        ax2.plot(t*1000., s, color='k', alpha=0.6)
        ax2.set_ylabel('Speed ('+traj.spaceUnit+'/'+traj.timeUnit+')')
        ax2.set_xlim(time_closeUp[0], time_closeUp[1])
        ax2.set_ylim(0, s.mean()+4*s.std())
        custom_plot.turnOffAxes(ax, spines=['top'])



    # full spectrogram
    ax = fig.add_axes(pos)
    ax.pcolormesh(t0, freqs, Pxx, vmin=Pxx.min(), vmax=numpy.percentile(Pxx,99))
    ax.set_xlim(t0.min(), t0.max())
    ax.set_ylim(freqs.min(), freqs.max())
    ax.set_xticklabels(ax.get_xticks()/1000.)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')


    ax2 = ax.twinx()
    for tl in ax2.get_yticklabels():
        tl.set_color('k')
    ax2.plot(t*1000., s, color='k', alpha=0.6)
    ax2.set_ylabel('Speed ('+traj.spaceUnit+'/'+traj.timeUnit+')')
    ax2.set_xlim(t0.min(), t0.max())
    # ax2.set_ylim(0, s.max()*1.5)
    ax2.set_ylim(0, s.mean()+4*s.std())
    custom_plot.turnOffAxes(ax, spines=['top'])

    # Inset with spectrum
    posInset = [.65, pos[1]+pos[3]-.175, .2, .15]
    axInset = fig.add_axes(posInset)
    axInset.patch.set_alpha(.5)
    csc.fft_plot(fig, axInset)

    axInset.plot(csc.freq, 2*csc.sp_filtered*csc.sp_filtered.conj(), 'k', alpha=.75)
    axInset.set_xlabel('LFP frequency (Hz)')

    # mark maximum
    i1 = numpy.where(csc.freq >= thetaRange[0])[0]
    i2 = numpy.where(csc.freq <= thetaRange[1])[0]
    indices = numpy.intersect1d(i1, i2)
    i = numpy.argmax(csc.spPower[indices]) + indices.min()
    axInset.plot([csc.freq[i], csc.freq[i]], [0, csc.spPower.max()], 'b--', alpha=.25, linewidth=2)
    axInset.text(csc.freq[i], csc.spPower.max(), numpy.round(csc.freq[i], 1), ha='center')

    #ax.set_xticks(numpy.arange(0, 31, 10))
    axInset.set_xticks([0]+thetaRange+[20])
    axInset.set_xlim(0, 20)
    axInset.set_ylim(0, csc.spPower.max()*1.2)
    axInset.yaxis.set_visible(False)
    axInset.spines['bottom'].set_color('white')
    axInset.xaxis.label.set_color('white')
    axInset.tick_params(axis='x', colors='white')
    custom_plot.turnOffAxes(axInset, spines=['left', 'right', 'top'])

    #--- plot together with speed
    # integrate Pxx at freqs
    minFreqIndex = numpy.where(freqs[:, 0] >= minFreq)[0][0]
    maxFreqIndex = numpy.where(freqs[:, 0] <= maxFreq)[0][-1]
    integratedPxx = numpy.zeros(t0.shape[1])
    for i, time in enumerate(t0[0, :]):
        integratedPxx[i] = Pxx[minFreqIndex:maxFreqIndex, i].sum()

    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.plot(t0[0,:], integratedPxx, 'r')
    ax.set_xlim(time_closeUp[0], time_closeUp[1])
    #ax.set_xlim(traj.t_start, traj.t_stop)

    ax2 = ax.twinx()
    for tl in ax2.get_yticklabels():
        tl.set_color('k')
    ax2.plot(t*1000., s, color='k', alpha=0.5)
    ax2.set_xlabel('Time ('+traj.timeUnit+')')
    ax2.set_ylabel('Speed ('+traj.spaceUnit+'/'+traj.timeUnit+')')
    ax2.set_xlim(time_closeUp[0], time_closeUp[1])
    ax2.set_ylim(0, s.mean()+4*s.std())

    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.plot(scipy.signal.resample(s, integratedPxx.size), integratedPxx, 'b.')
    ax.set_xlabel('Speed ('+traj.spaceUnit+'/'+traj.timeUnit+')')
    ax.set_ylabel('Integrated theta power')



# print traj.times.shape, eventData.times.shape, cscList[0].times.shape
# print traj.times[-1], eventData.times[-1], cscList[0].times[-1]


pl.show()
