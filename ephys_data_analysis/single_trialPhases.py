"""
For plotting spike data from a gainChanger maze.
"""

__author__ = ("Olivia Haas")
__version__ = "1.0, January 2014"

# python modules
import os, sys, re

# add additional custom paths
extraPaths = [os.path.join(os.path.abspath(os.path.dirname(__file__)),'../scripts'),
              '/Library/Python/2.7/site-packages']
for p in extraPaths:
    if not sys.path.count(p):
        sys.path.insert(1, p)

# other modules
import numpy

# custom made modules
import trajectory, custom_plot, signale, spikesPhase
from signale import tools

###################################################### plotting initialization

import matplotlib.pyplot as pl

grau=numpy.array([1,1,1])*.6
grau2=numpy.array([1,1,1])*.95

Bildformat = 'pdf'
Bildformat2 = 'png'

fontsize=20.0

###################################################### other initialization


for argv in sys.argv[2:]:
    if argv.startswith('maze:'):
        mazeName=argv.split(':')[-1]  # load u and uCircle etc. by calling with maze:u ...
    if argv=='e' or argv=='d':      # get language
        lang=argv
    if argv.startswith('offset:'):
        offset = int(argv.split(':')[-1])  # get offset by calling offset:NUMBER
    if argv=='chic':
        chic = True				# update analysis arrays
    if argv=='noShow':
        showFigs = False			# show pics
    if argv=='saveFigs':
        saveFigs = True				# save pics
    if argv=='save':
        saveAna = True				# save analysis arrays
    if argv=='update':
        update = True				# update analysis arrays


###################################################### commandline paramters

dummy = sys.argv[1]				# second argument should be the name of the folder to load
folderName = dummy.split('\\')[0]
for d in dummy.split('\\')[1:]:
    folderName+='/'+d

###################################################### parameters
mazeName='gainChanger'           # maze type
lang='e'
transitions= False
offset=2
gainIn = False
gainOut = False
allRunns = False
pooled = False
chic=False
showFigs=True
saveFigs=True
saveAna=True
update=False
lang='e'
color1=grau
color2='k'
noSpeck = False
useRecommended = False
expType = 'vr'
cscName1 = '.ncs'
TTName = '.t'
num_singleRuns = 0
thetaRange = [6, 10]
for argv in sys.argv[2:]:
    if argv=='noShow':
        showFigs = False			# show pics
    if argv=='saveFigs':
        saveFigs = True				# save pics
    if argv.startswith('csc:'):
        cscName = argv.split(':')[-1] + cscName1     # csc file to load
    if argv.startswith('TT:'):                      # Tetrode files to load, write e.g. as TT:['TT2_01.t']
        TTName = argv.split('TT:')[1].strip('[').strip(']')
        TTName = [s for s in TTName.split(',')]
    if argv=='noSpeck':
        noSpeck = True				            # only running spikes
    if argv=='useRecommended':
        useRecommended = True                       # use recommendations from metadata.dat
    if argv.startswith('thetaRange:'):
        thetaRange = argv.split('thetaRange:')[1].strip('[').strip(']')   #write into terminal e.g. as thetaRange:'[6, 10]'
        thetaRange = [float(thetaRange.split(',')[0]), float(thetaRange.split(',')[1])]
    if argv.startswith('gainChange:'):
        gainChange = argv.split('gainChange:')[1].strip('[').strip(']')
        gainIn = float(gainChange.split(',')[0])
        gainOut = float(gainChange.split(',')[1])
        print 'Gain change phases are plotted for gains ', gainIn, ' -> ', gainOut
    if argv=='allRunns':
        allRunns=True
        print 'Gain change phases are plotted for ALL GAIN CHANGES !!'
    if argv=='pooled':
        pooled=True
        print 'Gain change phases are being POOLED for individual gain changes!'
    if argv=='threshspeed':
        threshspeed = argv.split(':')[-1]

###################################################### initialization

# initialize in order to make them available globally

spikes=[]

ID = -1
stList = signale.spikezugList(t_start=None, t_stop=None, dims=[2])
eventData = None
traj = None
events = None
stimuli = None
threshspeed = .1

cscID = -1
cscList = signale.NeuralynxCSCList()

cwd = os.getcwd()


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
        if metadata.has_key('tt'):
            exec 'TTName =' + metadata['tt']
            print 'Taking tetrode data listed in metadata.dat! TT:', TTName
        if metadata.has_key('animal'):
            animal=metadata['animal']
        if metadata.has_key('depth'):
            depth=metadata['depth']
        if metadata.has_key('thetaRange'):
            thetaRange=metadata['thetaRange']
            print 'Taking thetaRange data listed in metadata.dat! thetaRange:', thetaRange
        if metadata.has_key('threshspeed'):
            threshspeed=metadata['threshspeed']
            print 'Taking threshspeed listed in metadata.dat! threshspeed:', threshspeed
    else:
        print 'NOTE: There is no metadata.dat. Proceeding without instead.'
else:
    try:
        cscName
    except NameError:
        print 'Error, no cscName given!'
        cscName = ''


###################################################### functions

def getData(folderName):
    global cscID, cscList
    global ID, traj, HDtraj, eventData, cscName, events, stimuli

    if os.path.isdir(folderName):
        dirList=os.listdir(folderName)
        os.chdir(folderName)
    else:
        dirList = [folderName]

    
    for item in dirList:
        if os.path.isfile(item):
            if item.endswith(cscName): # not any([item.find(str(s))+1 for s in excludeCSCs]):
                print 'loading', item , 'from folder: '+folderName
                csc = signale.load_ncsFile(item, showHeader=False)
                cscID += 1
                cscList.append(cscID, csc)
                cscList.addTags(cscID, file=item, dir=folderName)
            elif (TTName.__class__ == list and item in TTName) or \
                    (TTName.__class__ == str and item.endswith('.t')):
                print 'loading', item , 'from folder: '+folderName
                spikes = signale.load_tFile(item, showHeader=False)
                ID += 1
                stList.__setitem__(ID, spikes)
                stList.addTags(ID, file=item, dir=folderName)
            elif item.endswith('.nev'):
                print 'loading', item , 'from folder: '+folderName
                eventData = signale.load_nevFile(item, showHeader=False)
            elif item.endswith('.traj') and item.find('position')+1\
            and not item.find('collisions_position')+1 and not item.find('rewardsVisited_position')+1\
            and not item.find('events_position')+1:
                print 'loading', item , 'from folder: '+folderName
                traj = trajectory.load_trajectory(item, showHeader=False)
            elif item.endswith('.traj') and item.find('events_position')+1:
                 print 'loading', item, 'from folder: '+folderName
                 events = trajectory.load_rewardTrajectory(item)
            elif item.endswith('.traj') and item.find('rewardsVisited_position')+1:
                 print 'loading', item, 'from folder: '+folderName
                 stimuli = trajectory.load_rewardTrajectory(item)
        elif os.path.isdir(item):
            getData(item)
    os.chdir('..')



def prepareData():

    for csc in cscList:
        csc.filter(thetaRange[0], thetaRange[1])
        csc.hilbertTransform()

def findCycleFrequencies(lfp, threshMin, threshMax):
    if not hasattr(lfp , 'hilbertPhase'):
        lfp.filter(thetaRange[0], thetaRange[1])
        lfp.hilbertTransform()
    Maxima = signale.findMaximaMinima(lfp.hilbertPhase, findMinima=0)
    MaximaIndices = Maxima['maxima_indices']
    MaximaTimes = lfp.time_axis()[MaximaIndices]
    diffTimes = numpy.diff(MaximaTimes)
    if lfp.timeUnit != 'ms':
        print 'Assuming csc time axis -time_axis- is given in ms'
    diffTimesSec = diffTimes/1000.0
    freq = 1.0/diffTimesSec
    filteredFreq = numpy.delete(freq, numpy.where(freq < threshMin))
    filteredFreq = numpy.delete(filteredFreq, numpy.where(filteredFreq > threshMax))
    return filteredFreq
    
def plotCycleFrequencies(fig, ax, xvalues, plotNum=None, frequencies=None, lfp=None, threshMin=None, threshMax=None, labelx=None, labely=None):
    if not frequencies:
        frequencies = findCycleFrequencies(lfp, threshMin, threshMax)
    if not plotNum:
        color=[0.6,0.6,0.6]
    else:
        color=list(numpy.array([1.0,1.0,1.0])-(plotNum/6.))
    xvalues = numpy.ones(len(frequencies))*xvalues
    ax.plot(xvalues, frequencies, color=color, ls='*', marker='.', markersize=4)
    if labelx:    
        ax.set_xlabel(labelx)
    if labely:
        ax.set_ylabel(labely, fontsize=14)
    return fig, ax
    
def getGainsAndTheirCscs(gainIn, gainOut, GC):
    # find gains and their lfp which fit the input gain or output gain of the considered gain change
    # for the second frequency plot in pooled plot
    gainsIn = []
    gainsOut = []
    cscsIn = []
    cscsOut = []
    for s in numpy.arange(len(stimuli.times)-1):
        if GC[s].gainIn == float(gainIn): 
            gainsIn.append(GC[s].gainIn)
            cscsIn.append(GC[s].cscGainIn)
        if GC[s].gainIn == float(gainOut):
            gainsOut.append(GC[s].gainIn)
            cscsOut.append(GC[s].cscGainIn)
        # for the last stimulus pair look at gainIn and gainOut, otherwise the last gain would not be included
        if s == len(stimuli.times)-2:
            if GC[s].gainOut == float(gainIn):
                gainsIn.append(GC[s].gainOut)
                cscsIn.append(GC[s].cscGainOut)
            if GC[s].gainOut == float(gainOut):                       
                gainsOut.append(GC[s].gainOut)
                cscsOut.append(GC[s].cscGainOut)
    return gainsIn , gainsOut, cscsIn, cscsOut

###################################################### load and crop data


if os.path.isdir(folderName):
    getData(folderName)
    prepareData()
    
elif os.path.isfile(folderName):
    sys.exit('Point to a folder not a single file.')
else:
    sys.exit('Folder or data name does not exist.')
os.chdir(cwd)

if not ID+1:
    sys.exit('The folders do not contain tetrode data (t files)!')

eventData.changeTimeUnit('s')
stList.changeTimeUnit('s')
cscList.changeTimeUnit('s')

eventData, traj = trajectory.align(eventData, traj, display=True)
time_shift = eventData.t_start - traj.t_start # time shift has to be added onto stimuli and events_position!
traj.times = eventData.times # eventData und csc have same time axis

if stimuli:
    stimuli.times = stimuli.times+time_shift # add time_shift to stimuli times and save changes in variable
    print 'stimuli times adjusted'
if events:
    events.times = events.times+time_shift # add time_shift to stimuli times and save changes in variable
    print 'event times adjusted'


#for zaehler, st in enumerate(stList):
#    st.traj = traj
#    st.traj.threshspeed = threshspeed
#    st.thetaRange = (thetaRange[0], thetaRange[1])
#    st.getSpikePhases(cscList[0])
#    st.getRunningSpikes()
#
# 
#    traj.getLeftAndRightwardRuns(onlyRunning=False)
#    st.getLeftAndRightwardSpikes(onlyRunning=False)
# 
#    rechtsTraj=st.traj.rechts_places
#    linksTraj=st.traj.links_places
#    rechtsTimes=st.traj.rechts_times
#    linksTimes=st.traj.links_times
#    print 'times'
#    print rechtsTimes
#    print 'traj'
#    print rechtsTraj
# 
# #stimuli and event times croppen, sodass Hotspotausloesezeiten dem jeweilig letzten HotspotID entspricht
#    i=0
#    while i in numpy.arange(len(events.times)-1):
#     # find closest in rechts and links times
#     indexA=tools.findNearest(rechtsTimes,events.times[i])# array index and value of closest traj.time value to event time
#     indexB=tools.findNearest(linksTimes,events.times[i])
#     indexAB=tools.findNearest(numpy.array([indexA[1], indexB[1]]),events.times[i])
#     indexC=tools.findNearest(rechtsTimes,events.times[i+1])
#     indexD=tools.findNearest(linksTimes,events.times[i+1])
#     indexCD=tools.findNearest(numpy.array([indexC[1],indexD[1]]),events.times[i+1])
#     # if hotspot IDs of two consecutive events and the trajectory direction are identical it is a false hotspot and deleted
#     if events.IDs[i]==events.IDs[i+1] and indexAB[0]==indexCD[0] or events.IDs[i]==events.IDs[i+1] and (events.times[i+1]-events.times[i])<1.0:
#             events.times = numpy.delete(events.times, (i)) 
#             events.IDs = numpy.delete(events.IDs, (i)) 
#             events.places = numpy.delete(events.places, (i))
#             stimuli.times = numpy.delete(stimuli.times, (i+1)) # delete row i+1 , because stimuli file has the start up stimulus as first one (extra)
#             stimuli.parameters = numpy.delete(stimuli.parameters, (i+1))
#     else:
#       i=i+1
            

                   

###################################################### classes
   
class gainChange:
    'Common base class for all gain changes'

    def __init__(self, row, spikes):
        global traj, stimuli, csc
        #gains = numpy.array(stimuli.getParameter('gain'))  # gains from stimuli file  
        # last row is second last in stimuli file and ends where the traj ends with gainOut
        if row == len(stimuli.times)-3: 
            #self.gainIn = gains[row]
            #self.gainOut = gains[row+1]
            self.t_start = tools.findNearest(traj.times,stimuli.times[row])[1]
            self.t_middle = tools.findNearest(traj.times,stimuli.times[row+1])[1]    
            self.t_stop = traj.times[len(traj.times)-1]
            self.index_start = tools.findNearest(traj.times,stimuli.times[row])[0]          
            self.index_gain = tools.findNearest(traj.times,stimuli.times[row+1])[0]
            self.index_stop  = len(traj.times)-1
#        else:
#            self.gainIn = gains[row]
#            self.gainOut = gains[row+1]    
#            self.gainStop = gains[row+2]
            # first gain starts actually with the trajectory but time in gain stimulus is recorded a bit later
        elif row == 0:
            self.t_start = traj.times[0]#only up to one decimal place accurate in comparison of stimuli, events and traj.times
            self.t_middle = tools.findNearest(traj.times,stimuli.times[1])[1]
            self.t_stop = tools.findNearest(traj.times,stimuli.times[2])[1]
            self.index_start = 0           
            self.index_gain = tools.findNearest(traj.times,stimuli.times[1])[0]
            self.index_stop  = tools.findNearest(traj.times,stimuli.times[2])[0]
        else:
            self.t_start = tools.findNearest(traj.times,stimuli.times[row])[1]
            self.t_middle = tools.findNearest(traj.times,stimuli.times[row+1])[1]        
            self.t_stop  = tools.findNearest(traj.times,stimuli.times[row+2])[1]         
            self.index_start = tools.findNearest(traj.times,stimuli.times[row])[0]
            self.index_gain  = tools.findNearest(traj.times,stimuli.times[row+1])[0]
            self.index_stop  = tools.findNearest(traj.times,stimuli.times[row+2])[0]
        #for ID, spikes in enumerate(stList):
        self.placeCell = spikes.time_slice(self.t_start,self.t_stop)  #self.traj=self.placeCell.traj
        self.pCgainIn = spikes.time_slice(self.t_start,self.t_middle)
        self.pCgainOut = spikes.time_slice(self.t_middle,self.t_stop)
        for csc in cscList:        
            self.csc = csc.time_slice(self.t_start, self.t_stop) 
            self.cscGainIn = csc.time_slice(self.t_start, self.t_middle) 
            self.cscGainOut = csc.time_slice(self.t_middle, self.t_stop) 
            self.placeCell.getSpikePlaces()
            self.placeCell.traj.threshspeed = threshspeed
            self.pCgainIn.getSpikePlaces()
            self.pCgainIn.traj.threshspeed = threshspeed
            self.pCgainOut.getSpikePlaces()
            self.pCgainOut.traj.threshspeed = threshspeed
           
            
    def plotPhase(self, fig, ax, pooled=False):
        self.placeCell.phasePlot(fig=fig,labelx=False, labely=False, ax=ax, lfp=self.csc, labelsize=8)
        ax.axvline(x=traj.places[self.index_start,0],linewidth=1, color='g')
        ax.axvline(x=traj.places[self.index_stop-2,0],linewidth=1, color='k')
        ax.axvline(x=traj.places[self.index_gain,0],linewidth=1, color='r')        
        fig.canvas.draw()
        custom_plot.huebschMachen(ax)

    def plotPlace(self, row, fig):
        pos = [(row+1)*.08, (row+1)*.21, .05, .13] #  [left, bottom, width, height]
        ax = fig.add_axes(pos)
        self.placeCell.plotSpikesvsPlace(fig=fig, ax=ax)
        custom_plot.huebschMachen(ax)

def getAllgainChanges(st):
    GC = []
    for n in numpy.arange(len(stimuli.times)-2):
        GC.append(gainChange(n, st))
    return GC

def getFigure(AxesNum=None):
    if not AxesNum:
        fig = pl.figure(figsize=(12, 4.5))
    elif AxesNum <= 1:
        fig = pl.figure(figsize=(4, 4))
    elif 1 < AxesNum <= 6:
        fig = pl.figure(figsize=(3*(AxesNum), 4))
    elif 6 < AxesNum <= 12:
        fig = pl.figure(figsize=(18, 8))
    else:
        fig = pl.figure(figsize=(18, 13))
    return fig

def addAxesPositions(fig, Title, Num=None, pooled=False):
    axes = []
    pos = 0
    if pooled:
        pos = ([.08, .21, .15, .525], [0.32, 0.21, 0.15, 0.525], [0.56, 0.21, 0.15, 0.525], [.84, .675, .13, .25],
               [0.82, 0.21, 0.15, 0.25])
        for p in numpy.arange(len(pos)):
                axes.append(fig.add_axes(pos[p]))
    else:
        for a in numpy.arange(Num):
            if Num == 0:
                pl.close(fig)
                print 'Figure closed because there are no gain changes of type ', Title
            elif Num == 1:
                pos = [.3, .1, .4, .55] #  [left, bottom, width, height]
            elif 1 < Num <= 6:
                pos = [.09+(float(0.9*a)/float(Num)), .1, .66/(1.1*float(Num)), .6] #  [left, bottom, width, height]
            elif 6 < Num <= 12:
                if a <= 5:
                    pos = [.06+float(.16*a), .54, .11, .3] #  [left, bottom, width, height]
                else:
                    pos = [.06+float(.16*(a-6)), .05, .11, .3] #  [left, bottom, width, height]
            else:
                if a <= 5:
                    pos = [.06+float(.16*a), .7, .11, .185] #  [left, bottom, width, height]
                elif 5 < a < 11:
                    pos = [.06+float(.16*(a-6)), .36, .11, .185] #  [left, bottom, width, height]
                else:
                    pos = [.06+float(.16*(a-12)), .03, .11, .185] #  [left, bottom, width, height]
            axes.append(fig.add_axes(pos))
    return axes


def plotSingleRunPhases(GC, gainIn=None, gainOut=None, pooled=False, ttName=''):
    #if transitions:
    #    paramsTransDict = stimuli.parameterTransitionsDict('gain', labelsize=9)
    #else:
    #    paramsTransDict = stimuli.parameterDict('gain', labelsize=9)
    #Titles=numpy.array(paramsTransDict.keys())
    #AxesNum=numpy.array(paramsTransDict.values())
    z=0
    for PT in numpy.arange(len(stimuli.times)-3): # go through available gainchange events calulated by the Dict
        print 'run number ', PT, 'of ', len(stimuli.times)-4
        #GCin = float(re.findall(r"[-+]?\d*\.\d+|\d+",Titles[PT])[0]) # find the Dict gainIn and gainOut from string
        #GCout = float(re.findall(r"[-+]?\d*\.\d+|\d+",Titles[PT])[1])
        #z = 0
        # if only a specific gain pair is wanted, they will have to equal the Dict pair
#        if gainIn and gainOut:
#            if float(gainIn) == GCin and float(gainOut) == GCout:
#                if AxesNum[PT] == 1:
#                    print 'Gain change ', gainIn, ' -> ', gainOut, ': occurred', AxesNum[PT], 'time'
#                else:
#                    print 'Gain change ', gainIn, ' -> ', gainOut, ': occurred', AxesNum[PT], 'times'
#                # get figure sizes and axes
#                if pooled:
#                    fig = getFigure()
#                    axes = addAxesPositions(fig, Titles[PT], pooled=True)
#                else:
#                    fig = getFigure(AxesNum=AxesNum[PT])
#                    axes = addAxesPositions(fig, Titles[PT], Num=AxesNum[PT])
#                # get figure title
#                if animal and depth:
#                    fig.suptitle('Animal:' + str(animal) + ', Tetrode depth:' + str(depth) + ' $\mu m$\n' + 'VR gain change:' 
#                    + Titles[PT] + ' (' + str(AxesNum[PT]) +'x)', fontsize=12) 
#                else:
#                    fig.suptitle('Gain change from: '+Titles[PT])
#                # go through all stimuli and plot the data which fits the Dict gain change pair
#                for s in numpy.arange(len(stimuli.times)-1):
#                    # for all gain pairs that fit the given gain change plot the data   
#                    if GC[s].gainIn == float(gainIn) and GC[s].gainOut == float(gainOut):
#                        if pooled:
#                            if z == 0:
#                                text = True
#                            else:
#                                text = False
#                            # plot first 3 pooled data phase plots
#                            spikesPhase.prepareAndPlotPooledPhases(GC[s].placeCell, GC[s].csc, cscName, fig, axes[0], axes[1], 
#                                                                   axes[2], traj.xlim, noSpeck=True, text=text)
#                            # draw gain change as vertical red line in all 3 phase plots
#                            axes[0].axvline(x=traj.places[GC[s].index_gain,0],linewidth=1, color='r')
#                            if traj.places[GC[s].index_gain,0] in rechtsTraj:
#                                axes[1].axvline(x=traj.places[GC[s].index_gain,0],linewidth=1, color='r')  
#                            if traj.places[GC[s].index_gain,0] in linksTraj:  
#                                axes[2].axvline(x=traj.places[GC[s].index_gain,0],linewidth=1, color='r')  
#                            fig.canvas.draw()
#                            # calculate theta frequencies and plot them for one gain change
#                            fig, ax = plotCycleFrequencies(fig, axes[3], plotNum=z+1, xvalues=0, lfp=GC[s].csc, threshMin=thetaRange[0], 
#                                                           threshMax=thetaRange[1], labelx='Gain change',
#                                                           labely=r'$f_{\theta}$ $(Hz)$')
#                            # draw frequencies and histogram for gain change theta frequencies
#                            ax.hist(findCycleFrequencies(lfp=GC[s].csc, threshMin=thetaRange[0], threshMax=thetaRange[1]),
#                                    orientation='horizontal', normed=0.5, bottom=0.05, color=[0.6,0.6,0.6])
#                            ax.set_xticks([-0.2, 0.0, 0.2, 0.6])
#                            ax.set_xticklabels(['', '', Titles[PT]])
#                            start, end = ax.get_ylim()
#                            ax.yaxis.set_ticks(numpy.around(numpy.arange(start, end, 1),decimals=1))
#                            # draw frequencies and histogram for individual gains
#                            if z == AxesNum[PT]-1:
#                                gainsIn , gainsOut, cscsIn, cscsOut = getGainsAndTheirCscs(float(gainIn), float(gainOut), GC)
#                                for i in numpy.arange(len(cscsIn)):
#                                    if i == 0:
#                                        fig, ax1 = plotCycleFrequencies(fig, axes[4], xvalues=0, lfp=cscsIn[i], threshMin=thetaRange[0],
#                                                                        threshMax=thetaRange[1], labelx='Gain',
#                                                                        labely=r'$f_{\theta}$ $(Hz)$')
#                                        ax1.hist(findCycleFrequencies(lfp=cscsIn[i], threshMin=thetaRange[0], threshMax=thetaRange[1]),
#                                                orientation='horizontal', normed=0.4, bottom=0.2, color=[0.6,0.6,0.6])
#                                        ax1.set_xlim([-0.2, 5.0])
#                                        ax1.set_xticks([-0.2, 0.0, 0.3, 2.5, 5.0])
#                                        ax1.set_xticklabels([str(gainsIn[0])+' ('+str(len(gainsIn))+'x)','', '', str(gainsOut[0])+' ('+str(len(gainsOut))+'x)', ''])
#                                        start, end = ax1.get_ylim()
#                                        ax1.yaxis.set_ticks(numpy.around(numpy.arange(start, end+1, 1),decimals=1))
#                                    else:
#                                        fig, ax1 = plotCycleFrequencies(fig, axes[4], xvalues=0, lfp=cscsIn[i], threshMin=thetaRange[0],
#                                                                        threshMax=thetaRange[1])
#                                        ax1.hist(findCycleFrequencies(lfp=cscsIn[i], threshMin=thetaRange[0], threshMax=thetaRange[1]),
#                                                 orientation='horizontal', normed=0.3, bottom=0.2, color=[0.6,0.6,0.6])
#                                for o in numpy.arange(len(cscsOut)):
#                                    fig, ax1 = plotCycleFrequencies(fig, axes[4], xvalues=2.5, lfp=cscsOut[o], threshMin=thetaRange[0],
#                                                                       threshMax=thetaRange[1])
#                                    ax1.hist(findCycleFrequencies(lfp=cscsOut[o], threshMin=thetaRange[0], threshMax=thetaRange[1]),
#                                                orientation='horizontal', normed=0.3, bottom=2.7, color=[0.6,0.6,0.6])
#                            z += 1
#                        else:
#                            GC[s].plotPhase(fig, axes[z])
#                            if z == len(axes)-1:
#                                row=(numpy.ceil((z+1)/6.)-1)
#                                axes[int(row*6)].set_ylabel('Spike phase (deg)')
#                                fig.text(.5, .01, 'Position ('+traj.spaceUnit+')')
#                            z += 1
#        else:
#        if AxesNum[PT] == 1:
#            print 'Gain change ', GCin, ' -> ', GCout, ': occurred', AxesNum[PT], 'time'
#        else:
#            print 'Gain change ', GCin, ' -> ', GCout, ': occurred', AxesNum[PT], 'times'
#        if pooled:
#            fig = getFigure()
#            axes = addAxesPositions(fig, Titles[PT], pooled=True)
#        else:
        fig = pl.figure(figsize=(12, 4.5))
        pos0 = [.08, .21, .15, .525]
        pos1 = list(pos0)
        pos1[0] += pos1[2]+.09
        pos2 = list(pos1)
        pos2[0] += pos2[2]+.09
        ax0 = fig.add_axes(pos0)
        ax1 = fig.add_axes(pos1)
        ax2 = fig.add_axes(pos2)
        if animal and depth:
            fig.suptitle('Animal:' + str(animal) + ', Tetrode depth:' + str(depth) + ' $\mu m$\n' + 'run number ' +  str(PT)\
                        + ' of ' + str(len(stimuli.times)-4), fontsize=12) 
        #else:
        #    fig.suptitle('Gain: 1.0')
#        for s in numpy.arange(len(stimuli.times)-1):
#            if GC[s].gainIn == GCin and GC[s].gainOut == GCout:
#                if pooled:
#                    if z == 0:
#                        text = True
#                    else:
#                        text = False
#                    spikesPhase.prepareAndPlotPooledPhases(GC[s].placeCell, GC[s].csc, cscName, fig, axes[0], axes[1], 
#                                                           axes[2], traj.xlim,  noSpeck=True, text=text)
#                    axes[0].axvline(x=traj.places[GC[s].index_gain,0],linewidth=1, color='r')  
#                    if traj.places[GC[s].index_gain,0] in rechtsTraj:
#                            axes[1].axvline(x=traj.places[GC[s].index_gain,0],linewidth=1, color='r')  
#                    if traj.places[GC[s].index_gain,0] in linksTraj:  
#                            axes[2].axvline(x=traj.places[GC[s].index_gain,0],linewidth=1, color='r')  
#                    fig.canvas.draw()
#                    fig, ax = plotCycleFrequencies(fig, axes[3], xvalues=0, lfp=GC[s].csc, threshMin=thetaRange[0], 
#                                                       threshMax=thetaRange[1], labelx='Gain change',
#                                                       labely=r'$f_{\theta}$ $(Hz)$')
#                    # draw frequencies and histogram for gain change theta frequencies
#                    ax.hist(findCycleFrequencies(lfp=GC[s].csc, threshMin=thetaRange[0], threshMax=thetaRange[1]),
#                            orientation='horizontal', normed=0.5, bottom=0.05, color=[0.6,0.6,0.6])
#                    ax.set_xticks([-0.2, 0.0, 0.2, 0.6])
#                    ax.set_xticklabels(['', '', Titles[PT]])
#                    start, end = ax.get_ylim()
#                    ax.yaxis.set_ticks(numpy.around(numpy.arange(start, end, 1),decimals=1))
#                    # draw frequencies and histogram for individual gains
#                    if z == AxesNum[PT]-1:
#                        gainsIn , gainsOut, cscsIn, cscsOut = getGainsAndTheirCscs(float(GCin), float(GCout), GC)
#                        for i in numpy.arange(len(cscsIn)):
#                            if i == 0:
#                                fig, ax1 = plotCycleFrequencies(fig, axes[4], xvalues=0, lfp=cscsIn[i], threshMin=thetaRange[0],
#                                                                threshMax=thetaRange[1], labelx='Gain',
#                                                                labely=r'$f_{\theta}$ $(Hz)$')
#                                ax1.hist(findCycleFrequencies(lfp=cscsIn[i], threshMin=thetaRange[0], threshMax=thetaRange[1]),
#                                        orientation='horizontal', normed=0.3, bottom=0.2, color=[0.6,0.6,0.6])
#                                ax1.set_xlim([-0.2, 5.0])
#                                ax1.set_xticks([-0.2, 0.0, 0.3, 2.5, 5.0])
#                                ax1.set_xticklabels([str(gainsIn[0])+' ('+str(len(gainsIn))+'x)','', '', str(gainsOut[0])+' ('+str(len(gainsOut))+'x)', ''])
#                                start, end = ax1.get_ylim()
#                                ax1.yaxis.set_ticks(numpy.around(numpy.arange(start, end+1, 1),decimals=1))
#                            else:
#                                fig, ax1 = plotCycleFrequencies(fig, axes[4], xvalues=0, lfp=cscsIn[i], threshMin=thetaRange[0],
#                                                                threshMax=thetaRange[1])
#                                ax1.hist(findCycleFrequencies(lfp=cscsIn[i], threshMin=thetaRange[0], threshMax=thetaRange[1]),
#                                         orientation='horizontal', normed=0.3, bottom=0.2, color=[0.6,0.6,0.6])
#                        for o in numpy.arange(len(cscsOut)):
#                            fig, ax1 = plotCycleFrequencies(fig, axes[4], xvalues=2.5, lfp=cscsOut[o], threshMin=thetaRange[0],
#                                                               threshMax=thetaRange[1])
#                            ax1.hist(findCycleFrequencies(lfp=cscsOut[o], threshMin=thetaRange[0], threshMax=thetaRange[1]),
#                                        orientation='horizontal', normed=0.3, bottom=2.7, color=[0.6,0.6,0.6])
#                    z += 1
#                else:
        if GC[PT].pCgainIn.spike_times.size:
            spikesPhase.prepareAndPlotPooledPhases(GC[PT].pCgainIn, GC[PT].cscGainIn, cscName, fig, ax0, ax1, ax2, traj.xlim, noSpeck=True, text=True, zaehler=0)               
            #GC[PT].plotPhase(fig, axes[z])
            #row=(numpy.ceil((z+1)/6.)-1)
            #axes[int(row*6)].set_ylabel('Spike phase (deg)')
            #fig.text(.5, .01, 'Position ('+traj.spaceUnit+')')
            if saveFigs:
                print 'Picture saved to:', folderName+ttName+'_singleTrial_'+str(z)+'_spikePhase.pdf'
                fig.savefig(folderName+ttName+'_singleTrial_'+str(z)+'_spikePhase.pdf',\
                            format='pdf')
        z += 1
        






###################################################### main

print 'stList_length:', len(stList)
for zaehler, st in enumerate(stList):
    print zaehler
    st.traj = traj
    st.traj.threshspeed = threshspeed
    st.thetaRange = (thetaRange[0], thetaRange[1])
    st.getSpikePhases(cscList[0])
    st.getRunningSpikes()

    traj.getLeftAndRightwardRuns(onlyRunning=False)
    st.getLeftAndRightwardSpikes(onlyRunning=False)

    rechtsTraj=st.traj.rechts_places
    linksTraj=st.traj.links_places
#    rechtsTimes=st.traj.rechts_times
#    linksTimes=st.traj.links_times


#stimuli and event times croppen, sodass Hotspotausloesezeiten dem jeweilig letzten HotspotID entspricht
#    i=0
#==============================================================================
#     while i in numpy.arange(len(events.times)-1):
#         # find closest in rechts and links times
#         indexA=tools.findNearest(rechtsTimes,events.times[i])# array index and value of closest traj.time value to event time
#         indexB=tools.findNearest(linksTimes,events.times[i])
#         indexAB=tools.findNearest(numpy.array([indexA[1], indexB[1]]),events.times[i])
#         indexC=tools.findNearest(rechtsTimes,events.times[i+1])
#         indexD=tools.findNearest(linksTimes,events.times[i+1])
#         indexCD=tools.findNearest(numpy.array([indexC[1],indexD[1]]),events.times[i+1])
#         # if hotspot IDs of two consecutive events and the trajectory direction are identical it is a false hotspot and deleted
#         if events.IDs[i]==events.IDs[i+1] and indexAB[0]==indexCD[0] or events.IDs[i]==events.IDs[i+1] and (events.times[i+1]-events.times[i])<1.0:
#                 events.times = numpy.delete(events.times, (i)) 
#                 events.IDs = numpy.delete(events.IDs, (i)) 
#                 events.places = numpy.delete(events.places, (i))
#                 stimuli.times = numpy.delete(stimuli.times, (i+1)) # delete row i+1 , because stimuli file has the start up stimulus as first one (extra)
#                 stimuli.parameters = numpy.delete(stimuli.parameters, (i+1))
#         else:
#==============================================================================
#            i=i+1


    GC=getAllgainChanges(st)
    ttName=stList.tags[zaehler]['file'].split('.')[0]
    if gainIn and gainOut and not pooled:
        plotSingleRunPhases(GC, gainIn=gainIn, gainOut=gainOut, ttName=ttName)
    if gainIn and gainOut and pooled:
        plotSingleRunPhases(GC, gainIn=gainIn, gainOut=gainOut, pooled=True, ttName=ttName)
    if allRunns and not pooled:
        plotSingleRunPhases(GC, ttName=ttName)
    if allRunns and pooled:
        plotSingleRunPhases(GC, pooled=True, ttName=ttName)
    
#plotSingleRunPhases(GC, gainIn=1.0, gainOut=1.5, pooled=False)

if showFigs:
        pl.show()
        
os.chdir(cwd)

    

