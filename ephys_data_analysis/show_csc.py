"""
Show neuralynx ncs-files with python/numpy.
"""
__author__ = ("KT", "Olivia Haas")
__version__ = "3.0, August 2014"



import sys, os, inspect, struct

# add additional custom paths
extraPaths=["/home/thurley/python/lib/python2.5/site-packages/",\
    "/home/thurley/python/lib/python2.6/site-packages/",\
    "/home/thurley/python/lib/python2.7/dist-packages/",\
    "/home/thurley/data/",\
    "/home/haas/packages/lib/python2.6/site-packages",\
    os.path.join(os.path.abspath(os.path.dirname(__file__)), '../scripts')]
for p in extraPaths:
    if not sys.path.count(p):
        sys.path.insert(1, p)


# other modules
import numpy

import custom_plot, signale

###################################################### plotting initialization

import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
import matplotlib as mpl
mpl.matplotlib_fname()				# KT: Ali, what is this for?


colors = custom_plot.colors
grau = numpy.array([1, 1, 1]) * .6
grau2 = numpy.array([1, 1, 1]) * .95

Bildformat = 'pdf'

fontsize = 8



###################################################### initialization




dummy = sys.argv[1]				# second argument should be the name of the file to load
folderName = dummy.split('\\')[0]
for d in dummy.split('\\')[1:]:
    folderName+='/'+d
folderName += '/'





# parameters
lang='e'
color1=grau
color2='k'
showFigs=True
saveFigs=True
saveAna=True
onlyWithTTData=False
useRecommended = False
full_ana = False
for argv in sys.argv[2:]:
    if argv=='noShow':
        showFigs = False			# show pics
    if argv=='saveFigs':
        saveFigs = True				# save pics
    if argv=='full_ana':
        full_ana = True				# run full analysis
    if argv=='e' or argv=='d':      # get language
        lang=argv
    if argv=='colored':             # colored plots?
        color1='b'
        color2='g'
    if argv=='onlyWithTTData':             # colored plots?
        onlyWithTTData=True
    if argv=='useRecommended':
        useRecommended = True                       # use recommendations from metadata.dat
    if argv.startswith('exclude:'):
        excludeCSCs = argv.split('exclude:')[1]
        exec 'excludeCSCs ='+excludeCSCs
    if argv.startswith('thetaRange:'):
        thetaRange = argv.split('thetaRange:')[1].strip('[').strip(']')   #write into terminal e.g. as thetaRange:'[6, 10]'
        thetaRange = [float(thetaRange.split(',')[0]), float(thetaRange.split(',')[1])]



#################Make a help flag for the script!!!
#
####################################################

# initialize in order to make them globally available
cscID = -1
cscList = signale.NeuralynxCSCList()

loadedSomething = False
cwd = os.getcwd()


# get parameters
parameters = {'excludeCSCs': [], 'thetaRange': [6, 10], 'deltaRange': [2, 4]}
if useRecommended:
    fileName = os.path.normpath(folderName)+'/metadata.dat'
else:
    fileName = ''
dictio, metadata = signale.get_metadata(fileName, parameters, locals())
locals().update(dictio)


###################################################### functions

def getData(folderName):
    global cscID, cscList, loadedSomething

    if os.path.isdir(folderName):
        dirList=os.listdir(folderName)
        os.chdir(folderName)
    else:
        dirList = [folderName]

    if onlyWithTTData and not any([item.endswith('.t') for item in dirList]):
        os.chdir(cwd)
        sys.exit('The folders do not contain tetrode data (t files)! Therefore skipping folder!')
    for item in dirList:
        if os.path.isfile(item):
            if item.endswith('.ncs') and not any([item.find(str(s))+1 for s in excludeCSCs]):
                print 'loading Neuralynx data', item , 'from folder: '+folderName
                loadedSomething = True
                csc = signale.load_ncsFile(item, showHeader=False)
                cscID += 1
                cscList.append(cscID, csc)
                cscList.addTags(cscID, file=item, dir=folderName)
            elif item.endswith('.raw'):# or item.endswith('2.ncs'):
                print 'loading RAW data', item , 'from folder: '+folderName
                loadedSomething = True
                cscList = []
                cscList = signale.load_rawFile(item, exclude=excludeCSCs, showHeader=False)
        #elif os.path.isdir(item):
        #    getData(item)
    os.chdir('..')


def plotData(csc):
    csc.fft()

    fig = pl.figure(figsize=(12, 7))
    pos1 = [.1, .7, .8, .22]
    ax1 = fig.add_axes(pos1)
    pos2 = pos1
    pos2[1] = .39
    ax2 = fig.add_axes(pos2)
    pos3 = pos1
    pos3[1] = .1
    ax3 = fig.add_axes(pos3)

    ax1.set_title(csc.tags['file'])
    csc.plot(fig, ax1)
    custom_plot.huebschMachen(ax1)

    csc.fft_plot(fig, ax2)
    custom_plot.huebschMachen(ax2)
    ax2.set_xlabel('')
    ax2.set_xticklabels([])

    csc.fft_plot(fig, ax3)
    ax3.set_xlim(0, 15.)
    custom_plot.huebschMachen(ax3)


###################################################### analyse and plot


slow_gammaRange = [25, 60]
fast_gammaRange = [60, 140]
rippleRange = [100, 220]


if os.path.isfile(folderName):
    # get path name of the file to load
    index = folderName.find(folderName.split('/')[-1])
    path = ''
    if index:
        path = folderName[0:index-1]+'/'

    # load csc data
    csc = signale.load_ncsFile(folderName, showHeader=True)
    loadedSomething = True

    plotData(csc)

elif os.path.isdir(folderName):

    getData(folderName)
    cscList.changeTimeUnit('s')

    cscList.removeMean()

    fig, ax = cscList.plot()
    fig_fft, ax = cscList.fft_plot(0, 30)
    fig.suptitle('Data from ' + folderName, fontsize=fontsize)
    fig_fft.suptitle('Data from ' + folderName, fontsize=fontsize)

    if full_ana:
        pos = list(ax.get_position().bounds)
        pos[0] = 0.07
        pos[2] /= 2.5
        ax.set_position(pos)
        ylim = ax.get_ylim()

        # add axis for delta
        pos = list(pos)
        pos[0] += pos[2] + 0.03
        pos[2] /= 4.5
        ax = fig_fft.add_axes(pos)
        cscList.fft_plot(deltaRange[0], deltaRange[1], fig=fig_fft, ax=ax)
        ax.set_ylim(ylim)
        ax.set_yticklabels([])
        ax.set_title('delta\n'+str(deltaRange), fontsize=10)
        ax.set_xticklabels([])
        ax.set_xlabel('')

        # add axis for theta
        pos = list(pos)
        pos[0] += pos[2] + 0.03
        ax = fig_fft.add_axes(pos)
        cscList.fft_plot(thetaRange[0], thetaRange[1], fig=fig_fft, ax=ax)
        ax.set_ylim(ylim)
        ax.set_yticklabels([])
        ax.set_title('theta\n'+str(thetaRange), fontsize=10)
        ax.set_xticklabels([])
        ax.set_xlabel('')

        for id, csc in enumerate(cscList):
            # # 1/f correction
            # if not hasattr(csc, 'spPower'):
            #     csc.fft()
            # freqz = numpy.power(csc.freq, 1.25)
            # csc.sp = numpy.multiply(csc.sp, freqz)

            # delta
            # csc.filter is using csc.sp -> one-dimensional discrete Fourier Transform for real input signal
            csc.filter(deltaRange[0], deltaRange[1])
            csc.hilbertTransform()
            deltaAmp = csc.hilbertAbsolute.mean()

            # theta
            csc.filter(thetaRange[0], thetaRange[1])
            csc.hilbertTransform()
            thetaAmp = csc.hilbertAbsolute.mean()

            theta_delta_ratio = numpy.round(thetaAmp/deltaAmp, 2)

            fig_fft.text(pos[0]+.1, pos[1]+id*pos[3]/cscList.__len__(), str(theta_delta_ratio), fontsize=10)
        fig_fft.text(pos[0]+.1, pos[1]+(id+1)*pos[3]/cscList.__len__(), 'theta/delta', fontsize=10)

        # add axis slow gamma
        pos = list(pos)
        pos[0] += pos[2] + 0.1
        ax = fig_fft.add_axes(pos)
        cscList.fft_plot(slow_gammaRange[0], slow_gammaRange[1], fig=fig_fft, ax=ax)
        ax.set_ylim(ylim)
        ax.set_yticklabels([])
        ax.set_title('slow gamma\n'+str(slow_gammaRange), fontsize=10)
        ax.set_xticklabels([])
        ax.set_xlabel('')

        # add axis fast gamma
        pos = list(pos)
        pos[0] += pos[2] + 0.03
        ax = fig_fft.add_axes(pos)
        cscList.fft_plot(fast_gammaRange[0], fast_gammaRange[1], fig=fig_fft, ax=ax)
        ax.set_ylim(ylim)
        ax.set_yticklabels([])
        ax.set_title('fast gamma\n'+str(fast_gammaRange), fontsize=10)
        ax.set_xticklabels([])
        ax.set_xlabel('')

        # add axis ripple
        pos = list(pos)
        pos[0] += pos[2] + 0.03
        ax = fig_fft.add_axes(pos)
        cscList.fft_plot(rippleRange[0], rippleRange[1], fig=fig_fft, ax=ax)
        ax.set_ylim(ylim)
        ax.set_yticklabels([])
        ax.set_title('ripple\n'+str(rippleRange), fontsize=10)
        ax.set_xticklabels([])
        ax.set_xlabel('')



else:
    sys.exit('Folder or data name does not exist.')







###################################################### finishing

os.chdir(cwd)

if saveFigs:
    fig.savefig(folderName+'csc.png', format='png')
    fig_fft.savefig(folderName+'csc_fft.png', format='png')
    print '... saved figures to', os.path.relpath(folderName)


if not loadedSomething:
    sys.exit('The folders do not contain csc data!')

if showFigs:
    pl.show()
