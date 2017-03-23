"""
LFP spectrogram
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
time_closeUp = None
saveFigs = True
for argv in sys.argv[2:]:
    if argv.startswith('yMax:'):
        yMax=float(argv.split(':')[-1])   # value expected as string, e.g. ymax:'16'
    if argv=='noShow':
        showFigs = False
    if argv.startswith('csc:'):
        CSC=argv.split(':')[-1]
        cscName=argv.split(':')[-1] + '.ncs'       # csc file to load
    if argv == 'useRecommended':
        useRecommended = True                       # use recommendations from metadata.dat
    if argv.startswith('thetaRange:'):
        thetaRange = argv.split('thetaRange:')[1].strip('[').strip(']')   #write into terminal e.g. as thetaRange:'[6, 10]'
        thetaRange = thetaRange.split(',')
        thetaRange = [int(thetaRange[0]), int(thetaRange[1])]
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
                CSC=cscName.split('.')[0]
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
            if item.endswith(cscName):# or item.endswith('2.ncs'):
                print 'loading', item , 'from folder: '+folderName
                csc = signale.load_ncsFile(item, showHeader=True)
                cscID += 1
                cscList.append(cscID, csc)
                cscList.addTags(cscID, file=item, dir=folderName)
        elif os.path.isdir(item):
            getData(item)
    os.chdir('..')


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

cscList.changeTimeUnit('s')


###################################################### csc and speed


csc = cscList[0]
#traj.time_offset(6.5)


#--- spectrogram
minFreq = thetaRange[0]
maxFreq = thetaRange[1]
csc.changeTimeUnit('ms')
stList.changeTimeUnit('ms')


Pxx, freqs, t0 = csc.spectrogram(minFreq=0, maxFreq=20, windowSize=2**15) #8192
# smooth Pxx a bit
Pxx = signale.blur_image(Pxx, 10)

if not time_closeUp:
    time_closeUp = [t0.min(), t0.max()]


fig = pl.figure()
pos = [.1, .12, .8, .8]
ax = fig.add_axes(pos)

ax.pcolormesh(t0, freqs, Pxx, vmin=Pxx.min(), vmax=numpy.percentile(Pxx, 99))
ax.set_xlim(time_closeUp[0], time_closeUp[1])
try:
   yMax
except NameError:
    yMax=freqs.max()

ax.set_ylim(freqs.min(), yMax)
#ax.set_xticklabels([])
#ax.set_yticklabels([])
ax.set_xticklabels(ax.get_xticks()/1000.)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')
custom_plot.huebschMachen(ax)


# Inset with spectrum
posInset = [pos[1]+pos[3]*.66, pos[1]+pos[3]*.75, .2, .15]
#posInset = [pos[1]+pos[3]*.66, pos[1]+pos[3]*.55, .2, .15]
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
#axInset.plot([csc.freq[i], csc.freq[i]], [0, csc.spPower.max()], 'w--', alpha=.25, linewidth=2)
axInset.plot([csc.freq[i], csc.freq[i]], [0, csc.spPower.max()], 'b--', alpha=.25, linewidth=2)
axInset.text(csc.freq[i], csc.spPower.max(), numpy.round(csc.freq[i], 1), ha='center')

#ax.set_xticks(numpy.arange(0, 31, 10))
axInset.set_xticks([0]+thetaRange+[20])
axInset.set_xlim(0, 20)
axInset.set_ylim(0, csc.spPower.max()*1.2)
axInset.yaxis.set_visible(False)
axInset.spines['bottom'].set_color('white')
axInset.xaxis.label.set_color('white')
#axInset.xaxis.set_ticklabels([])
#axInset.set_xlabel('')
axInset.tick_params(axis='x', colors='white')
custom_plot.turnOffAxes(axInset, spines=['left', 'right', 'top'])

if saveFigs:
    print 'Figure saved to: ', folderName+'lfp_spectrogram_'+CSC+'.png'
    fig.savefig(folderName+'lfp_spectrogram_'+CSC+'.png',\
                format='png')  #save figure , dpi=300
    #fig.savefig(folderName+'lfp_spectrogram_'+CSC+'_noLabels.pdf',\
     #           format='pdf')

pl.show()
