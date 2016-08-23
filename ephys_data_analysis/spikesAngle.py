"""
Head direction plotting for spikes.
"""
__author__ = ("Moritz Dittmeyer", "KT")
__version__ = "3.2, July 2013"



import sys, os, inspect, struct

# add additional custom paths
extraPaths=["/home/thurley/python/lib/python2.5/site-packages/",\
    "/home/thurley/python/lib/python2.6/site-packages/",\
    "/home/thurley/python/lib/python2.7/dist-packages/",\
    "/home/thurley/data/py/",\
    os.path.join(os.path.abspath(os.path.dirname(__file__)),'../scripts')]
for p in extraPaths:
    if not sys.path.count(p):
        sys.path.append(p)

import numpy

import NeuroTools.signals as NTsig
import signale, trajectory



###################################################### plotting initialization

import matplotlib as mpl
import matplotlib.pyplot as pl

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection, CircleCollection


fontsize=12.0
markersize = 6

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = markersize
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'


colors=['#FF0000','#0000FF','#008000','#00FFFF','#FF00FF','#EE82EE',
        '#808000','#800080','#FF6347','#FFFF00','#9ACD32','#4B0082',
        '#FFFACD','#C0C0C0','#A0522D','#FA8072','#FFEFD5','#E6E6FA',
        '#F1FAC1','#C5C5C5','#A152ED','#FADD72','#F0EFD0','#EEE6FF',
        '#01FAC1','#F5F5F5','#A152FF','#FAFD72','#F0EFDF','#EEEFFF',
        '#F1FA99','#C9C9C9','#A152DD','#FA5572','#FFFFD0','#EDD6FF']

grey=numpy.ones(3)*.25
transred='#FFA1A1'



###################################################### commandline paramters


dummy = sys.argv[1]				# second argument should be the name of the folder to load
folderName = dummy.split('\\')[0]
for d in dummy.split('\\')[1:]:
    folderName+='/'+d


# parameters
prefix = ''
suffix = ''
onlyRunning = False
polar = False
useRecommended = False
saveFigs = True
TTName = '.t'
for argv in sys.argv[2:]:
    if argv.startswith('prefix:'):
        prefix=argv.split(':')[-1]  # name prefix
    if argv.startswith('suffix:'):
        suffix=argv.split(':')[-1]  # name suffix
    if argv.startswith('TT:'):                      # Tetrode files to load
        TTName = argv.split('TT:')[1].strip('[').strip(']')
        TTName = [s for s in TTName.split(',')]
    if argv == 'onlyRunning':       # display only running spikes
        onlyRunning = True
    if argv == 'polar':             # display in polar plot
        polar = True
    if argv=='saveFigs':
        saveFigs = True				# save pics
    if argv=='useRecommended':
        useRecommended = True       # use recommendations from metadata.dat


###################################################### initialization


# initialize in order to make them available globally
spikes=[]
ID=-1
nvtID=-1
stList=signale.spikezugList(t_start=None, t_stop=None, dims=[2])
traj = None
#HDtraj = None
trajList = []

cwd=os.getcwd()


if useRecommended:
    if os.path.isfile(folderName+'metadata.dat'):
        print 'Loading metadata:'
        metadata = signale.io._read_metadata(folderName+'metadata.dat', showHeader=True)
        if metadata.has_key('csc'):
            cscName = metadata['csc']
            print 'Taking csc data listed in metadata.dat! csc:', cscName
        if metadata.has_key('tt'):
            exec 'TTName =' + metadata['tt']
            print 'Taking tetrode data listed in metadata.dat! TT:', TTName
        print
    else:
        print 'NOTE: There is no metadata.dat. Proceeding without instead.'


###################################################### functions

def getData(folderName):
    global spikes, ID, nvtID, traj, trajList#, HDtraj

    if os.path.isdir(folderName):
        dirList=os.listdir(folderName)
        os.chdir(folderName)
    else:
        dirList = [folderName]

    if not any([item.endswith('.t') for item in dirList]):
        os.chdir(cwd)
        sys.exit('The folders do not contain tetrode data (t files)!')
    for item in dirList:
        if os.path.isfile(item):
            if (TTName.__class__ == list and item in TTName) or\
             (TTName.__class__ == str and item.endswith(suffix+'.t') and item.startswith(prefix)):
                print 'loading', item , 'from folder: '+folderName
                spikes = signale.load_tFile(item, showHeader=True)
                ID += 1
                stList.__setitem__(ID, spikes)
                stList.addTags(ID, file=item, dir=folderName)
            elif item.endswith('.nvt'):# or item.endswith('2.ncs'):
                print 'loading', item , 'from folder: '+folderName
                loadedSomething = True
                traj = trajectory.load_nvtFile(item, 'linearMaze', showHeader=True)
                HDtraj = traj[1]        # head direction
                traj = traj[0]          # trajectory
                nvtID += 1
        elif os.path.isdir(item):
            getData(item)
    os.chdir('..')








###################################################### load data


if os.path.isdir(folderName):
    getData(folderName)
elif os.path.isfile(folderName):
    sys.exit('Point to a folder not a single file.')
else:
    sys.exit('Folder or data name does not exist.')
os.chdir(cwd)

if not ID+1:
    sys.exit('The folders do not contain tetrode data (t files)!')
else:
    # set begining to zero time
    #for st in stList:
    #    st.spike_times -= st.t_start
    #stList._spikezugList__recalc_startstop()

    # cut away all earlier spikes
    #stList = stList.time_slice(0., stList.t_stop)

    # change to seconds since trajectories are also stored in seconds
    stList._spikezugList__recalc_startstop()
    stList.changeTimeUnit('s')

    #stList = stList.time_slice(0, 20000)      # reduce data size a bit


###################################################### main

###################################################### clean data


traj.getTrajDimensions()
traj.getHeadDirection()

# remove from traj.times and traj.places, times at which animal didn't run <threshspeed
traj.getRunningTraj(threshspeed=.15, window_len=71)



###################################################### spikes against place

#########

for zaehler, st in enumerate(stList):
    st.traj = traj
    st.getSpikeHeadDirection()

 #shorten to test
    #traj.times=traj.times[0:len(traj.times)/10]
    #traj.places=traj.places[0:len(traj.places)/10]


    st.getRunningSpikes(traj) # clean spikeTimes, remove spikes
                                    # that occurred when the animal was at rest



###################################################### plots




    fig = pl.figure(20+zaehler, figsize=(5, 7))


##    if noSpeck:
##        numRows = 2
##    else:
##        numRows = 4


########## Plot Head Direction ###############

    bins = numpy.arange(-numpy.pi, numpy.pi/2., 0.1)

    if onlyRunning:
        ax = fig.add_subplot(2, 1, 1, polar=polar)
        ax.hist(traj.run_headDirections, bins, color='blue', label='onlyRunning', histtype='step', linewidth=2)

        ax.set_title('Head direction', fontsize=fontsize+1)
        if polar:
            ax.set_xticklabels([])
        else:
            ax.set_xticks(numpy.arange(-numpy.pi, 2*numpy.pi, numpy.pi/2.))
            ax.set_xticklabels([])
            ax.set_xlim(-numpy.pi, numpy.pi)
        ax.set_yticklabels([])
    else:
        ax = fig.add_subplot(2, 2, 1, polar=polar)
        ax.hist(traj.headDirections, bins, color='green', label='all', histtype='step', linewidth=2)
        ax.hist(traj.run_headDirections, bins, color='blue', label='onlyRunning', histtype='step', linewidth=2)

        ax.set_title('Head direction', fontsize=fontsize+1)
        if polar:
            ax.set_xticklabels([])
        else:
            ax.set_xticks(numpy.arange(-numpy.pi, 2*numpy.pi, numpy.pi/2.))
            ax.set_xticklabels([])
            ax.set_xlim(-numpy.pi, numpy.pi)
        ax.set_yticklabels([])


        ax = fig.add_subplot(2, 2, 2, polar=polar)
        ax.hist(traj.run_headDirections, bins, color='blue', label='onlyRunning', histtype='step', linewidth=2)

        ax.set_title('Head direction', fontsize=fontsize+1)
        if polar:
            ax.set_xticklabels([])
        else:
            ax.set_xticks(numpy.arange(-numpy.pi, 2*numpy.pi, numpy.pi/2.))
            ax.set_xticklabels([])
            ax.set_xlim(-numpy.pi, numpy.pi)
        ax.set_yticklabels([])




############################Plot Head Direction of Spike Times #################



    if onlyRunning:
        ax = fig.add_subplot(2, 1, 2, polar=polar)
        ax.hist(st.run_spikeHeadDirections, bins, color='blue', label='onlyRunning', histtype='step', linewidth=2)

        ax.set_title('Spike head direction', fontsize=fontsize+1)
        #ax.grid(False)
        leg = ax.legend(bbox_to_anchor=(0.5, 0.02))
        if polar:
            ax.set_xticklabels([])
        else:
            ax.set_xticks(numpy.arange(-numpy.pi, 2*numpy.pi, numpy.pi/2.))
            ax.set_xticklabels(numpy.arange(-360, 450, 180))
            ax.set_xlim(-numpy.pi, numpy.pi)
        ax.set_yticklabels([])

    else:
        ax = fig.add_subplot(2, 2, 3, polar=polar)
        ax.hist(st.spike_headDirections, bins, color='green', label='all', histtype='step', linewidth=2)
        ax.hist(st.run_spikeHeadDirections, bins, color='blue', label='onlyRunning', histtype='step', linewidth=2)

        ax.set_title('Spike head direction', fontsize=fontsize+1)
        leg = ax.legend(bbox_to_anchor=(1., 0.02))
        if polar:
            ax.set_xticklabels([])
        else:
            ax.set_xticks(numpy.arange(-numpy.pi, 2*numpy.pi, numpy.pi/2.))
            ax.set_xticklabels(numpy.arange(-360, 450, 180))
            ax.set_xlim(-numpy.pi, numpy.pi)
        ax.set_yticklabels([])

        ax = fig.add_subplot(2, 2, 4, polar=polar)
        ax.hist(st.run_spikeHeadDirections, bins, color='blue', label='onlyRunning', histtype='step', linewidth=2)

        ax.set_title('Spike head direction', fontsize=fontsize+1)
        if polar:
            ax.set_xticklabels([])
        else:
            ax.set_xticks(numpy.arange(-numpy.pi, 2*numpy.pi, numpy.pi/2.))
            ax.set_xticklabels(numpy.arange(-360, 450, 180))
            ax.set_xlim(-numpy.pi, numpy.pi)
        ax.set_yticklabels([])




##################

    #--- text for data set
    # .995, .98
    fig.text(.995, .01, stList.tags[zaehler]['dir'] + ', ' + stList.tags[zaehler]['file'],\
        fontsize=fontsize-10,  horizontalalignment='right')

    if saveFigs:
        fig.savefig(stList.tags[zaehler]['dir']+stList.tags[zaehler]['file'].split('.')[0]+'_HD.png',\
            format='png')  #save figure




###################################################### finishing


pl.show()



