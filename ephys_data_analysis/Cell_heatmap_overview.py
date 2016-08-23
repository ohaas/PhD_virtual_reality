__author__ = 'haasolivia'

# python modules
import sys
import os

# add additional custom paths
extraPaths = [os.path.join(os.path.abspath(os.path.dirname(__file__)), '../scripts')]
# ,
#           os.path.join(os.path.abspath(os.path.dirname(__file__)), '/opt/anaconda/bin/python'),
#           os.path.join(os.path.abspath(os.path.dirname(__file__)), '/opt/anaconda/pkgs')]

for p in extraPaths:
    if not sys.path.count(p):
        sys.path.insert(1, p)

# further modules

import numpy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pl
import seaborn as sns
import hickle
from sklearn import mixture
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import csv
from matplotlib import animation
from matplotlib.lines import Line2D
import scipy
from scipy import stats
import pandas as pd
import custom_plot
import signale
from matplotlib.colors import LinearSegmentedColormap


# _______________________________________ SET OUTPUT OPTIONS ________________________________________

# mixed = 1 double and single field cells are not sorted into separate categories
mixed = 1
# cmaps_on = 1 each line is plotted with the heatmap according to its category
cmaps_on = 1
# 0 is for all cells together + the gains plotted separately next to each other
# 1 for all cells together
# 2 for cells separately into prop, vis and rem cathegories
all_together = 0
# if heatmaps should only show the FR maximum
maxima_heatmap = 0
# if fits should be plotted
fits = 0
# Double cells should be plotted (= 1) or not (= 0)
double_on = 0
# Only double cells should be plotted (= 1) or not (= 0)
only_double = 0
# example linewidth
linew = 8  # 2 or 8
# rausmap = 0 rausmapping cells not included
rausmap = 1
# 0 = False, 1 = True
only_rausmap = 0
# show example visual and prop cells (0 = False, 1 = True)
bsp = 0

# _______________________________________ LOAD DATA ________________________________________

server = 'saw'
path = '/Users/haasolivia/Documents/'+server+'/dataWork/olivia/hickle/'

hkl_files_pre = numpy.repeat(hickle.load(path+'Summary/used_filenames.hkl'), 2)
# hkl_files = [hkl_files_pre[i].split('_normalised')[0]+'.hkl' if (i%2 != 0) else hkl_files_pre[i] for i in
#              range(len(hkl_files_pre))]
hkl_files = [hkl_files_pre[i] if (i%2 != 0) else hkl_files_pre[i] for i in range(len(hkl_files_pre))]
run_direc = numpy.repeat(hickle.load(path+'Summary/running_directions.hkl'), 2)

double_info = hickle.load(path+'Summary/delta_and_weight_info.hkl')
f = numpy.array(double_info['used_files'])
double_cells = numpy.array(double_info['double_cell_files'])
double_direc = double_info['double_cell_direc']
no_double_cells = double_info['no_double_cell_files']
no_double_direc = double_info['no_double_cell_direc']
category = hickle.load(path+'Summary/prop_vis_rem_filenames.hkl')

double_color = '#31a354'
# file_p = []
# direc_p = []
# file_v = []
# direc_v = []
# file_r = []
# direc_r = []
# for p in numpy.arange(len(category['prop_files'])):
#     file_p.append(category['prop_files'][p].split('_info')[0])
#     file_p.append(category['prop_files'][p].split('_info')[1].split('_')[-1])
# for v in numpy.arange(len(category['vis_files'])):
#     file_v.append(category['vis_files'][v].split('_info')[0])
#     file_v.append(category['vis_files'][v].split('_info')[1].split('_')[-1])
# for r in numpy.arange(len(category['rem_files'])):
#     file_r.append(category['rem_files'][r].split('_info')[0])
#     file_r.append(category['rem_files'][r].split('_info')[1].split('_')[-1])

# treadmill_data = hkl_files[::2]
# treadmill_run_direc = run_direc[::2]
#
# virtual_data = hkl_files[1::2]
# virtual_data = run_direc[1::2]

x_maxFR_05 = []
x05 = []
y05 = []
f05 = []
x_maxFR_15 = []
x15 = []
y15 = []
f15 = []

x_maxFR_05pm = []
x05pm = []
y05pm = []
f05pm = []
x_maxFR_15pm = []
x15pm = []
y15pm = []
f15pm = []

x_maxFR_05vm = []
x05vm = []
y05vm = []
f05vm = []
x_maxFR_15vm = []
x15vm = []
y15vm = []
f15vm = []

x_maxFR_05rm = []
x05rm = []
y05rm = []
f05rm = []
x_maxFR_15rm = []
x15rm = []
y15rm = []
f15rm = []

x_maxFR_05d = []
x05d = []
y05d = []
f05d = []
x_maxFR_15d = []
x15d = []
y15d = []
f15d = []

x_maxFR_05v = []
x05v = []
y05v = []
f05v = []
x_maxFR_15v = []
x15v = []
y15v = []
f15v = []

x_maxFR_05vpm = []
x05vpm = []
y05vpm = []
f05vpm = []
x_maxFR_15vpm = []
x15vpm = []
y15vpm = []
f15vpm = []

x_maxFR_05vvm = []
x05vvm = []
y05vvm = []
f05vvm = []
x_maxFR_15vvm = []
x15vvm = []
y15vvm = []
f15vvm = []

x_maxFR_05vrm = []
x05vrm = []
y05vrm = []
f05vrm = []
x_maxFR_15vrm = []
x15vrm = []
y15vrm = []
f15vrm = []

x_maxFR_05vd = []
x05vd = []
y05vd = []
f05vd = []
x_maxFR_15vd = []
x15vd = []
y15vd = []
f15vd = []

counter = 0

bad = numpy.array(['10353_2014-06-19_VR_GCend_linTrack1_GC_TT1_SS_07_PF_info_left',
                   '10353_2014-06-19_VR_GCend_linTrack1_GC_TT1_SS_07_PF_info_normalised_left',
                   '10528_2015-04-13_VR_GCend_ol_linTrack1_TT2_SS_13_PF_info_right',
                   '10528_2015-04-13_VR_GCend_ol_linTrack1_TT2_SS_13_PF_info_normalised_right',
                   '10528_2015-04-14_VR_GCend_Dark_linTrack1_TT3_SS_05_PF_info_right',
                   '10528_2015-04-14_VR_GCend_Dark_linTrack1_TT3_SS_05_PF_info_normalised_right',
                   '10528_2015-04-14_VR_GCend_Dark_linTrack1_TT3_SS_05_PF_info_left',
                   '10528_2015-04-14_VR_GCend_Dark_linTrack1_TT3_SS_05_PF_info_normalised_left',
                   '10823_2015-07-20_VR_GCendDark_linTrack1_TT3_SS_01_PF_info_right',
                   '10823_2015-07-20_VR_GCendDark_linTrack1_TT3_SS_01_PF_info_normalised_right',
                   '10823_2015-07-31_VR_GCend_linTrack1_TT4_SS_06_PF_info_right',
                   '10823_2015-07-31_VR_GCend_linTrack1_TT4_SS_06_PF_info_normalised_right',
                   '10823_2015-08-04_VR_GCend_linTrack1_TT2_SS_06_PF_info_right',
                   '10823_2015-08-04_VR_GCend_linTrack1_TT2_SS_06_PF_info_normalised_right',
                   '10823_2015-08-12_VR_GCend_linTrack1_TT4_SS_07_PF_info_left',
                   '10823_2015-08-12_VR_GCend_linTrack1_TT4_SS_07_PF_info_normalised_left',
                   '10823_2015-08-19_VR_GCend_linTrack1_TT2_SS_16_PF_info_left',
                   '10823_2015-08-19_VR_GCend_linTrack1_TT2_SS_16_PF_info_normalised_left',
                   '10823_2015-08-19_VR_GCend_linTrack1_TT3_SS_07_PF_info_right',
                   '10823_2015-08-19_VR_GCend_linTrack1_TT3_SS_07_PF_info_normalised_right',
                   '10823_2015-08-19_VR_GCend_linTrack1_TT3_SS_07_PF_info_left',
                   '10823_2015-08-19_VR_GCend_linTrack1_TT3_SS_07_PF_info_normalised_left',
                   '10823_2015-08-19_VR_GCend_linTrack1_TT3_SS_19_PF_info_right',
                   '10823_2015-08-19_VR_GCend_linTrack1_TT3_SS_19_PF_info_normalised_right',
                   '10823_2015-08-20_VR_GCend_linTrack1_TT4_SS_07_PF_info_left',
                   '10823_2015-08-20_VR_GCend_linTrack1_TT4_SS_07_PF_info_normalised_left',
                   '10529_2015-03-27_VR_linTrack1_TT8_SS_14_PF_info_right',     # rausmappened
                   '10529_2015-03-27_VR_linTrack1_TT8_SS_14_PF_info_normalised_right',
                   '10823_2015-07-24_VR_GCend_linTrack1_TT3_SS_04_PF_info_right',   # doppel: prop + rausmappened
                   '10823_2015-07-24_VR_GCend_linTrack1_TT3_SS_04_PF_info_normalised_right',
                   '10823_2015-08-18_VR_GCend_linTrack1_TT3_SS_10_PF_info_left',  # bad activity
                   '10823_2015-08-18_VR_GCend_linTrack1_TT3_SS_10_PF_info_normalised_left',
                   '10823_2015-08-03_VR_GCend_linTrack1_TT2_SS_04_PF_info_right',  # only active in one gain
                   '10823_2015-08-03_VR_GCend_linTrack1_TT2_SS_04_PF_info_normalised_right',
                   '10528_2015-03-11_VR_GCend_linTrack1_TT1_SS_01_PF_info_left',   # bad activity
                   '10528_2015-03-11_VR_GCend_linTrack1_TT1_SS_01_PF_info_normalised_left',
                   '10823_2015-07-27_VR_GCend_linTrack1_TT3_SS_06_PF_info_right',  # only active in one gain
                   '10823_2015-07-27_VR_GCend_linTrack1_TT3_SS_06_PF_info_normalised_right',
                   '10823_2015-07-27_VR_GCend_linTrack1_TT3_SS_06_PF_info_left',  # only active in one gain
                   '10823_2015-07-27_VR_GCend_linTrack1_TT3_SS_06_PF_info_normalised_left',
                   '10823_2015-06-30_VR_GCend_linTrack1_TT4_SS_14_PF_info_left',  # only active in one gain
                   '10823_2015-06-30_VR_GCend_linTrack1_TT4_SS_14_PF_info_normalised_left',
                   '10528_2015-04-21_GCend_Dark_linTrack1_TT2_SS_06_PF_info_left',  # only active in one gain
                   '10528_2015-04-21_GCend_Dark_linTrack1_TT2_SS_06_PF_info_normalised_left',
                   '10529_2015-03-25_VR_nami_linTrack2_TT5_SS_03_PF_info_left',     # only active in one gain
                   '10529_2015-03-25_VR_nami_linTrack2_TT5_SS_03_PF_info_normalised_left',
                   '10823_2015-07-03_VR_GCend_linTrack1_TT3_SS_05_PF_info_left',    # bad activity
                   '10823_2015-07-03_VR_GCend_linTrack1_TT3_SS_05_PF_info_normalised_left',
                   '10823_2015-07-24_VR_GCendDark_linTrack1_TT2_SS_11_PF_info_left',  # only active in one gain
                   '10823_2015-07-24_VR_GCendDark_linTrack1_TT2_SS_11_PF_info_normalised_left',
                   '10823_2015-08-03_VR_GCend_linTrack1_TT4_SS_01_PF_info_right',     # only active in one gain
                   '10823_2015-08-03_VR_GCend_linTrack1_TT4_SS_01_PF_info_normalised_right',
                   '10823_2015-08-18_VR_GCend_nami_linTrack1_TT2_SS_03_PF_info_right',  # bad activity
                   '10823_2015-08-18_VR_GCend_nami_linTrack1_TT2_SS_03_PF_info_normalised_right',
                   '10823_2015-08-26_VR_GCend_nami_linTrack1_TT2_SS_07_PF_info_right',  # only active in one gain
                   '10823_2015-08-26_VR_GCend_nami_linTrack1_TT2_SS_07_PF_info_normalised_right',
                   '10823_2015-08-27_VR_GCend_linTrack1_TT2_SS_11_PF_info_left',        # bad activity
                   '10823_2015-08-27_VR_GCend_linTrack1_TT2_SS_11_PF_info_normalised_left',
                   '10529_2015-03-02_VR_GCend_abig_linTrack1_TT1_SS_13_PF_info_right',  # CA1
                   '10529_2015-03-02_VR_GCend_abig_linTrack1_TT1_SS_13_PF_info_normalised_right',
                   '10529_2015-03-02_VR_GCend_abig_linTrack1_TT8_SS_08_PF_info_left',   # CA1
                   '10529_2015-03-02_VR_GCend_abig_linTrack1_TT8_SS_08_PF_info_normalised_left',
                   '10529_2015-03-04_VR_GCend_linTrack1_TT8_SS_07_PF_info_left',   # CA1
                   '10529_2015-03-04_VR_GCend_linTrack1_TT8_SS_07_PF_info_normalised_left',
                   '10535_2015-09-30_VR_GCend_linTrack1_TT3_SS_10_PF_info_right',   # CA1
                   '10535_2015-09-30_VR_GCend_linTrack1_TT3_SS_10_PF_info_normalised_right',
                   '10535_2015-09-30_VR_GCend_linTrack1_TT3_SS_10_PF_info_left',   # CA1
                   '10535_2015-09-30_VR_GCend_linTrack1_TT3_SS_10_PF_info_normalised_left',
                   '10535_2015-10-06_VR_GCend_linTrack1_TT4_SS_07_PF_info_right',   # CA1
                   '10535_2015-10-06_VR_GCend_linTrack1_TT4_SS_07_PF_info_normalised_right',
                   '10535_2015-10-06_VR_GCend_linTrack1_TT4_SS_07_PF_info_left',   # CA1
                   '10535_2015-10-06_VR_GCend_linTrack1_TT4_SS_07_PF_info_normalised_left',
                   '10535_2015-10-06_VR_GCend_linTrack1_TT4_SS_12_PF_info_right',   # CA1
                   '10535_2015-10-06_VR_GCend_linTrack1_TT4_SS_12_PF_info_normalised_right',
                   '10535_2015-10-06_VR_GCend_linTrack1_TT4_SS_12_PF_info_left',   # CA1
                   '10535_2015-10-06_VR_GCend_linTrack1_TT4_SS_12_PF_info_normalised_left',
                   '10537_2015-10-07_VR_GCend_linTrack1_TT8_SS_13_PF_info_right',   # CA1
                   '10537_2015-10-07_VR_GCend_linTrack1_TT8_SS_13_PF_info_normalised_right',
                   '10537_2015-10-07_VR_GCend_linTrack1_TT8_SS_13_PF_info_left',   # CA1
                   '10537_2015-10-07_VR_GCend_linTrack1_TT8_SS_13_PF_info_normalised_left',
                   '10537_2015-10-16_VR_GCend_linTrack1_TT1_SS_01_PF_info_right',   # CA1
                   '10537_2015-10-16_VR_GCend_linTrack1_TT1_SS_01_PF_info_normalised_right',
                   '10537_2015-10-16_VR_GCend_linTrack1_TT1_SS_01_PF_info_left',   # CA1
                   '10537_2015-10-16_VR_GCend_linTrack1_TT1_SS_01_PF_info_normalised_left',
                   '10537_2015-10-16_VR_GCend_linTrack1_TT6_SS_08_PF_info_right',   # CA1
                   '10537_2015-10-16_VR_GCend_linTrack1_TT6_SS_08_PF_info_normalised_right',
                   '10537_2015-10-16_VR_GCend_linTrack1_TT6_SS_08_PF_info_left',   # CA1
                   '10537_2015-10-16_VR_GCend_linTrack1_TT6_SS_08_PF_info_normalised_left',
                   '10537_2015-10-20_VR_GCend_linTrack1_TT8_SS_08_PF_info_right',   # CA1
                   '10537_2015-10-20_VR_GCend_linTrack1_TT8_SS_08_PF_info_normalised_right',
                   '10537_2015-10-20_VR_GCend_linTrack1_TT8_SS_08_PF_info_left',   # CA1
                   '10537_2015-10-20_VR_GCend_linTrack1_TT8_SS_08_PF_info_normalised_left',
                   '10537_2015-10-22_VR_GCend_linTrack1_TT1_SS_09_PF_info_right',   # CA1
                   '10537_2015-10-22_VR_GCend_linTrack1_TT1_SS_09_PF_info_normalised_right',
                   '10537_2015-10-22_VR_GCend_linTrack1_TT1_SS_09_PF_info_left',   # CA1
                   '10537_2015-10-22_VR_GCend_linTrack1_TT1_SS_09_PF_info_normalised_left',
                   '10537_2015-10-22_VR_GCend_linTrack1_TT5_SS_01_PF_info_right',   # CA1
                   '10537_2015-10-22_VR_GCend_linTrack1_TT5_SS_01_PF_info_normalised_right',
                   '10537_2015-10-22_VR_GCend_linTrack1_TT5_SS_08_PF_info_left',   # CA1
                   '10537_2015-10-22_VR_GCend_linTrack1_TT5_SS_08_PF_info_normalised_left'])

# remove 'bad' data

# print 'len(f) before', len(f)
vis_idx = []
for index, v in enumerate(f):

    if v not in bad:
        vis_idx.append(index)
    else:
        print v

f = f[vis_idx]
# print 'len(f) after', len(f)

# add rausmap Cells to file array ----------------------------------------------------------------

rausmapping_names = []

if rausmap:

    with open('/Users/haasolivia/Documents/'+server+'/dataWork/olivia/hickle/Summary/Cell_overview.csv', 'rb') as fi:
        reader = csv.reader(fi)
        cell_overview = numpy.array(list(reader))

    right_1gain = [i for i in numpy.arange(len(cell_overview[:, 1])) if cell_overview[:, 1][i] == 'None \xe2\x80\x93 one gain(0.5 raus)']
    left_1gain = [i for i in numpy.arange(len(cell_overview[:, 2])) if cell_overview[:, 2][i] == 'None \xe2\x80\x93 one gain(0.5 raus)']

    names_right = cell_overview[:, 0][right_1gain]
    names_left = cell_overview[:, 0][left_1gain]
    direc_r = ['right' for j in numpy.arange(len(names_right))]
    direc_l = ['left' for j in numpy.arange(len(names_left))]

    rausmapping_files = numpy.array(list(names_right) + list(names_left))
    rausmapping_direc = numpy.array(direc_r + direc_l)

    # info = {'rausmapping_files': rausmapping_files, 'rausmapping_direc': rausmapping_direc}
    # hickle.dump(info, '/Users/haasolivia/Documents/'+server+'/dataWork/olivia/hickle/Summary/rausmap_info.hkl', mode='w')
    # sys.exit()
    rausmapping_names = [(rausmapping_files[r]).split('_info')[0]+'_info_'+rausmapping_direc[r] for
                         r in numpy.arange(len(rausmapping_files))]

    rausmap_pre = numpy.repeat(rausmapping_names, 2)

    rausmap_f = [rausmap_pre[i].split('_info')[0]+'_info_normalised'+rausmap_pre[i].split('_info')[1] if (i%2 != 0)
                 else rausmap_pre[i] for i in numpy.arange(len(rausmap_pre))]

    rausmapping_names = rausmap_f

    if not only_rausmap:
        f = numpy.array(list(f)+rausmap_f)
        run_direc = numpy.array(list(run_direc)+list(numpy.repeat(rausmapping_direc, 2)))
    else:
        f = numpy.array(rausmap_f)
        run_direc = numpy.array(numpy.repeat(rausmapping_direc, 2))

# ------------------------------------------------------------------------------------------------

for i, file1 in enumerate(f):  # hkl_files):
    file = file1.split('info')[0]+'info.hkl'
    run_direc[i] = (file1.split('info')[1]).split('_')[-1]
    # print file
    # all_file = file.split('.hkl')[0]+'_'+run_direc[i]
    all_file = file1

    try:
        a = hickle.load(path+file)
    except IOError:
        a = hickle.load(path+'cells_not_used_79/'+file)
          #'10353_2014-06-17_VR_GCend_linTrack1_GC_TT3_SS_07_PF_info_normalised.hkl')  #file
    double = 0
    # if file.endswith('normalised.hkl'):
    #     fileB = file.split('_normalised.hkl')[0]+'.hkl'
    #     if fileB in double_cells and \
    #                     run_direc[i] in numpy.array(double_direc)[numpy.where(numpy.array(double_cells) == fileB)[0]]:
    #         double = 1
    #     elif fileB in no_double_cells and \
    #                     run_direc[i] in numpy.array(no_double_direc)[numpy.where(numpy.array(no_double_cells) == fileB)[0]]:
    #         double = 0
    #     else:
    #         print 'ERROR FILE NOT FOUND IN EITHER DOUBLE OR NO_DOUBLE CELL NAMES!'
    #         print 'file = ', fileB
    #         print 'running direction = ', run_direc[i]
    #         sys.exit()
    # else:
    if file in double_cells and \
                    run_direc[i] in numpy.array(double_direc)[numpy.where(numpy.array(double_cells) == file)[0]]:
        double = 1
    elif file in no_double_cells and \
                    run_direc[i] in numpy.array(no_double_direc)[numpy.where(numpy.array(no_double_cells) == file)[0]]:
        double = 0
    elif file1 in rausmapping_names:
        double = 0
    else:
        print 'ERROR FILE AND RUNNING DIRECTION NOT FOUND IN EITHER DOUBLE OR NO_DOUBLE CELL NAMES!'
        print 'file = ', file1
        # print 'running direction = ', run_direc[i]
        sys.exit()

    # if file is from double cell then
    # if double_on:
    #     double = 1

    for gain in ['0.5', '1.5']:
        print 'counter: ', counter, ' out of ', 2*len(f)-1  #len(hkl_files)*2-9
        counter += 1
        # run_direc[i] = 'right'
        fr = a[run_direc[i]+'FR_x_y_gain_'+gain]
        x = fr[0]
        y = fr[1]

        # correct x-axis for leftwards runs -- they were always saved as a FR array from traj.xlim[0] to
        # traj.xlim[1], which goes for x from [0 .. e.g. 2] no matter what the running direction of the animal was!
        # For leftward runs spikes at e.g. x=2 would be at the beginning of the run for the animal, therefore need
        # to be corrected to be x=0.
        if run_direc[i] == 'left':  # for leftward runs plot abolute x-value from start position

            # sys.exit()
            vis_track_length = 2.

            if file.endswith('normalised.hkl'):
                start = vis_track_length/float(gain)
            else:
                start = vis_track_length
            x = abs(x-start)

        file_i = file.split('_info')[0]+'_info_'+run_direc[i]
        file_inorm = file.split('_info')[0]+'_info_normalised_'+run_direc[i]

        if i % 2 == 0:  # if i is an even number, then it is a normalised.hkl file
            x /= float(gain)  # calculating normalised places
            if gain == '0.5':
                if double == 1:
                    x_maxFR_05d.append(x[numpy.nanargmax(y)])
                    x05d.append(x)
                    y05d.append(y)
                    f05d.append(file)
                elif not only_double:
                    x_maxFR_05.append(x[numpy.nanargmax(y)])
                    x05.append(x)
                    y05.append(y)
                    f05.append(file)

                    if file_i in category['prop_files'] or file_inorm in category['prop_files']:  #or file_i in rausmapping_names:
                        x_maxFR_05pm.append(x[numpy.nanargmax(y)])
                        x05pm.append(x)
                        y05pm.append(y)
                        if file_i in category['prop_files']:  # or file_i in rausmapping_names:
                            f05pm.append(file_i)
                        else:
                            f05pm.append(file_inorm)
                    elif file_i in category['vis_files'] or file_inorm in category['vis_files'] or \
                         file_i in category['rem_files'] or file_inorm in category['rem_files']:
                        x_maxFR_05vm.append(x[numpy.nanargmax(y)])
                        x05vm.append(x)
                        y05vm.append(y)
                        if file_i in category['vis_files'] or file_i in category['rem_files']:
                            f05vm.append(file_i)
                        else:
                            f05vm.append(file_inorm)
                    elif file_i in category['rem_files'] or file_inorm in category['rem_files'] or file_i in rausmapping_names:
                        x_maxFR_05rm.append(x[numpy.nanargmax(y)])
                        x05rm.append(x)
                        y05rm.append(y)
                        if file_i in category['rem_files'] or file_i in rausmapping_names:
                            f05rm.append(file_i)
                        else:
                            f05rm.append(file_inorm)
                    else:
                        print 'file ', file, run_direc[i], ' not found in prop_files, vis_files or rem_files !'
                        # sys.exit()
            else:
                if double == 1:
                    x_maxFR_15d.append(x[numpy.nanargmax(y)])
                    x15d.append(x)
                    y15d.append(y)
                    f15d.append(file)
                elif not only_double:
                    x15.append(x)
                    f15.append(file)
                    if file_i in rausmapping_names:
                        x_maxFR_15.append(0)
                        y15.append(y * 0)
                    else:
                        x_maxFR_15.append(x[numpy.nanargmax(y)])
                        y15.append(y)

                    if file_i in category['prop_files'] or file_inorm in category['prop_files']:  # or file_i in rausmapping_names:
                        x15pm.append(x)
                        # if file_i in rausmapping_names:
                        #     x_maxFR_15pm.append(0)
                        #     y15pm.append(y * 0)
                        # else:
                        x_maxFR_15pm.append(x[numpy.nanargmax(y)])
                        y15pm.append(y)
                        if file_i in category['prop_files']:  # or file_i in rausmapping_names:
                            f15pm.append(file_i)
                        else:
                            f15pm.append(file_inorm)
                    elif file_i in category['vis_files'] or file_inorm in category['vis_files'] or \
                         file_i in category['rem_files'] or file_inorm in category['rem_files']:
                        x_maxFR_15vm.append(x[numpy.nanargmax(y)])
                        x15vm.append(x)
                        y15vm.append(y)
                        if file_i in category['vis_files'] or file_i in category['rem_files']:
                            f15vm.append(file_i)
                        else:
                            f15vm.append(file_inorm)
                    elif file_i in category['rem_files'] or file_inorm in category['rem_files'] or file_i in rausmapping_names:
                        x15rm.append(x)
                        if file_i in rausmapping_names:
                            x_maxFR_15rm.append(0)
                            y15rm.append(y * 0)
                        else:
                            x_maxFR_15rm.append(x[numpy.nanargmax(y)])
                            y15rm.append(y)

                        if file_i in category['rem_files'] or file_i in rausmapping_names:
                            f15rm.append(file_i)
                        else:
                            f15rm.append(file_inorm)
                    else:
                        print 'file ', file, run_direc[i], ' not found in prop_files, vis_files or rem_files !'
                        # sys.exit()

        else:  # for visual data
            if gain == '0.5':
                if double == 1:
                    x_maxFR_05vd.append(x[numpy.nanargmax(y)])
                    x05vd.append(x)
                    y05vd.append(y)
                    f05vd.append(file)
                elif not only_double:
                    x_maxFR_05v.append(x[numpy.nanargmax(y)])
                    x05v.append(x)
                    y05v.append(y)
                    f05v.append(file)

                    if file_i in category['prop_files'] or file_inorm in category['prop_files']:  # or file_i in rausmapping_names:

                        x05vpm.append(x)
                        x_maxFR_05vpm.append(x[numpy.nanargmax(y)])
                        y05vpm.append(y)
                        if file_i in category['prop_files']:  # or file_i in rausmapping_names:
                            f05vpm.append(file_i)
                        else:
                            f05vpm.append(file_inorm)
                    elif file_i in category['vis_files'] or file_inorm in category['vis_files'] or \
                         file_i in category['rem_files'] or file_inorm in category['rem_files']:
                        x_maxFR_05vvm.append(x[numpy.nanargmax(y)])
                        x05vvm.append(x)
                        y05vvm.append(y)
                        if file_i in category['vis_files'] or file_i in category['rem_files']:
                            f05vvm.append(file_i)
                        else:
                            f05vvm.append(file_inorm)
                    elif file_i in category['rem_files'] or file_inorm in category['rem_files'] or file_i in rausmapping_names:
                        x_maxFR_05vrm.append(x[numpy.nanargmax(y)])
                        x05vrm.append(x)
                        y05vrm.append(y)
                        if file_i in category['rem_files'] or file_i in rausmapping_names:
                            f05vrm.append(file_i)
                        else:
                            f05vrm.append(file_inorm)
                    else:
                        print 'file ', file, run_direc[i], ' not found in prop_files, vis_files or rem_files !'
                        # sys.exit()
            else:
                if double == 1:
                    x_maxFR_15vd.append(x[numpy.nanargmax(y)])
                    x15vd.append(x)
                    y15vd.append(y)
                    f15vd.append(file)
                elif not only_double:
                    x15v.append(x)
                    f15v.append(file)
                    if file_i in rausmapping_names:
                        x_maxFR_15v.append(0)
                        y15v.append(y * 0)
                    else:
                        x_maxFR_15v.append(x[numpy.nanargmax(y)])
                        y15v.append(y)

                    if file_i in category['prop_files'] or file_inorm in category['prop_files']:  # or file_i in rausmapping_names:

                        x15vpm.append(x)
                        # if file_i in rausmapping_names:
                        #     x_maxFR_15vpm.append(0)
                        #     y15vpm.append(y * 0)
                        # else:
                        x_maxFR_15vpm.append(x[numpy.nanargmax(y)])
                        y15vpm.append(y)
                        if file_i in category['prop_files']:  # or file_i in rausmapping_names:
                            f15vpm.append(file_i)
                        else:
                            f15vpm.append(file_inorm)
                    elif file_i in category['vis_files'] or file_inorm in category['vis_files'] or \
                         file_i in category['rem_files'] or file_inorm in category['rem_files']:
                        x_maxFR_15vvm.append(x[numpy.nanargmax(y)])
                        x15vvm.append(x)
                        y15vvm.append(y)
                        if file_i in category['vis_files'] or file_i in category['rem_files']:
                            f15vvm.append(file_i)
                        else:
                            f15vvm.append(file_inorm)
                    elif file_i in category['rem_files'] or file_inorm in category['rem_files'] or file_i in rausmapping_names:
                        x15vrm.append(x)
                        if file_i in rausmapping_names:
                            x_maxFR_15vrm.append(0)
                            y15vrm.append(y * 0)
                        else:
                            x_maxFR_15vrm.append(x[numpy.nanargmax(y)])
                            y15vrm.append(y)
                        if file_i in category['rem_files'] or file_i in rausmapping_names:
                            f15vrm.append(file_i)
                        else:
                            f15vrm.append(file_inorm)
                    else:
                        print 'file ', file, run_direc[i], ' not found in prop_files, vis_files or rem_files !'
                        # sys.exit()

# ___________________________________ DEFINITIONS ___________________________________


def sortX_and_Y(xarray, yarray, norm=True):  # if norm=True every row will go from firing rate 0 to 1
    # xbin_width is 0.03m = 3 cm
    y_max = []
    max_arrays = []
    idx_max = []

    for a in numpy.arange(len(xarray)):
        # when xarray is starts with large values, turn x and y array around
        xarray[a] = numpy.array(xarray[a])
        yarray[a] = numpy.array(yarray[a])

        if len(xarray[a]) > 1:
            if numpy.diff(xarray[a])[0] < 0:
                xarray[a] = numpy.flipud(xarray[a])
                yarray[a] = numpy.flipud(yarray[a])
        if norm:
            m = numpy.nanmax(yarray[a])
            if numpy.isnan(m):
                m_idx = 0
            else:
                m_idx = numpy.nanargmax(yarray[a])
            idx_max.append(m_idx)
            y_max.append(numpy.around(m, 1))
            yarray[a] = yarray[a]/m
            max_array = numpy.zeros(len(yarray[a]))
            max_array[m_idx] = 1
            max_arrays.append(max_array)

    xarray = list(xarray)
    yarray = list(yarray)

    return xarray, yarray, numpy.array(y_max), numpy.array(max_arrays), numpy.array(idx_max)


def interpolate(new_x, x, y):
    if len(x) == len(new_x):
        for i in numpy.arange(len(y)):
            y[i] = numpy.interp(new_x[i], x[i], y[i])
            x[i] = new_x[i]
    else:
        print 'len(x) != len(new_x)'
        sys.exit()
    return x, y


def sorting(x_maxFR, x_05, y_05, f_05, x_15, y_15, f_15, normalisedFR = True, inter=False):
    idx_s = numpy.argsort(x_maxFR)
    x_maxFR = numpy.array(x_maxFR)[idx_s]
    x_05 = numpy.array(x_05)[idx_s]
    y_05 = numpy.array(y_05)[idx_s]
    f_05 = numpy.array(f_05)[idx_s]
    x_05, y_05, y_max05, max_arrays05, idx_max05 = sortX_and_Y(x_05, y_05, norm=normalisedFR)
    x_15 = numpy.array(x_15)[idx_s]
    y_15 = numpy.array(y_15)[idx_s]
    f_15 = numpy.array(f_15)[idx_s]
    x_15, y_15, y_max15, max_arrays15, idx_max15 = sortX_and_Y(x_15, y_15, norm=normalisedFR)

    if inter:
        x_15, y_15 = interpolate(new_x=x_05, x=x_15, y=y_15)
        x_15, y_15, y_max15, max_arrays15, idx_max15 = sortX_and_Y(x_15, y_15, norm=normalisedFR)

    return x_maxFR, x_05, y_05, y_max05, max_arrays05, idx_max05, f_05, x_15, y_15, y_max15, max_arrays15, idx_max15, f_15


# begin_array = [a1, a2, a3, a4], insert_array = [i1, i2, i3, i4] => result = [a1, i1, a2, i2, a3, i3, a4, i4]
def merge(begin_array, insert_array):
    a = []
    if len(begin_array) == len(insert_array):
        for i in numpy.arange(len(begin_array)):
            a.append(begin_array[i])
            a.append(insert_array[i])
    else:
        print 'len(begin_array) != len(insert_array)'
        sys.exit()
    return a

# ___________________________________ COLORMAPS FOR PROP, VISUAL AND REMAPPING CELL TYPES _____________________________

colors_bl = ['white', '#4575b4']
Blues = LinearSegmentedColormap.from_list('Blues_ol', colors_bl)  # FOR PROP CELLS
colors_bl1 = ['white', 'k']   # #6682ac
Blues1 = LinearSegmentedColormap.from_list('Blues_ol1', colors_bl1)  # FOR RAUSMAP PROP CELLS
colors_red = ['white', '#d73027']
Reds = LinearSegmentedColormap.from_list('Reds_ol', colors_red)  # FOR VISUAL CELLS
colors_br = ['white', '#8e6701']
Browns = LinearSegmentedColormap.from_list('Browns', colors_br)  # FOR REMAP CELLS

co_05 = custom_plot.pretty_colors_set2[0]
co_15 = custom_plot.pretty_colors_set2[1]

colors_gain05 = ['white', co_05]
colors_05 = LinearSegmentedColormap.from_list('gain05_ol', colors_gain05)  # FOR GAIN 0.5 CELLS
colors_gain15 = ['white', co_15]
colors_15 = LinearSegmentedColormap.from_list('gain15_ol', colors_gain15)  # FOR GAIN 0.5 CELLS

# ___________________________________ SORTING ___________________________________


x_maxFR_05, x05, y05, ymax05, max_arrays05, idx_max05, f05, x15, y15, ymax15, max_arrays15, idx_max15, f15 = \
    sorting(x_maxFR=x_maxFR_05, x_05=x05, y_05=y05, f_05=f05, x_15=x15, y_15=y15, f_15=f15)
# interpolate(new_x=x05, x=x15, y=y15)

x_maxFR_05pm, x05pm, y05pm, ymax05pm, max_arrays05pm, idx_max05pm, f05pm, x15pm, y15pm, ymax15pm, max_arrays15pm, \
idx_max15pm, f15pm = \
    sorting(x_maxFR=x_maxFR_05pm, x_05=x05pm, y_05=y05pm, f_05=f05pm, x_15=x15pm, y_15=y15pm, f_15=f15pm)
# interpolate(new_x=x05pm, x=x15pm, y=y15pm)

x_maxFR_05vm, x05vm, y05vm, ymax05vm, max_arrays05vm, idx_max05vm, f05vm, x15vm, y15vm, ymax15vm, max_arrays15vm, \
idx_max15vm, f15vm = \
    sorting(x_maxFR=x_maxFR_05vm, x_05=x05vm, y_05=y05vm, f_05=f05vm, x_15=x15vm, y_15=y15vm, f_15=f15vm)
# interpolate(new_x=x05vm, x=x15vm, y=y15vm)

x_maxFR_05rm, x05rm, y05rm, ymax05rm, max_arrays05rm, idx_max05rm, f05rm, x15rm, y15rm, ymax15rm, max_arrays15rm, \
idx_max15rm, f15rm = \
    sorting(x_maxFR=x_maxFR_05rm, x_05=x05rm, y_05=y05rm, f_05=f05rm, x_15=x15rm, y_15=y15rm, f_15=f15rm)
# interpolate(new_x=x05rm, x=x15rm, y=y15rm)

x_maxFR_05d, x05d, y05d, ymax05d, max_arrays05d, idx_max05d, f05d, x15d, y15d, ymax15d, max_arrays15d, idx_max15d, f15d = \
    sorting(x_maxFR=x_maxFR_05d, x_05=x05d, y_05=y05d, f_05=f05d, x_15=x15d, y_15=y15d, f_15=f15d)
# interpolate(new_x=x05d, x=x15d, y=y15d)


#  _________________ visual data _______________


x_maxFR_05v, x05v, y05v, ymax05v, max_arrays05v, idx_max05v, f05v, x15v, y15v, ymax15v, max_arrays15v, idx_max15v, f15v = \
    sorting(x_maxFR=x_maxFR_05v, x_05=x05v, y_05=y05v, f_05=f05v, x_15=x15v, y_15=y15v, f_15=f15v, inter=True)
# interpolate(new_x=x05v, x=x15v, y=y15v)  # interpolate for gain 1.5 in order to have the same x values as in gain 0.5!

x_maxFR_05vpm, x05vpm, y05vpm, ymax05vpm, max_arrays05vpm, idx_max05vpm, f05vpm, x15vpm, y15vpm, ymax15vpm, \
max_arrays15vpm, idx_max15vpm, f15vpm = \
    sorting(x_maxFR=x_maxFR_05vpm, x_05=x05vpm, y_05=y05vpm, f_05=f05vpm, x_15=x15vpm, y_15=y15vpm, f_15=f15vpm, inter=True)
# interpolate(new_x=x05vpm, x=x15vpm, y=y15vpm)

x_maxFR_05vvm, x05vvm, y05vvm, ymax05vvm, max_arrays05vvm, idx_max05vvm, f05vvm, x15vvm, y15vvm, ymax15vvm, \
max_arrays15vvm, idx_max15vvm, f15vvm = \
    sorting(x_maxFR=x_maxFR_05vvm, x_05=x05vvm, y_05=y05vvm, f_05=f05vvm, x_15=x15vvm, y_15=y15vvm, f_15=f15vvm, inter=True)
# interpolate(new_x=x05vvm, x=x15vvm, y=y15vvm)

x_maxFR_05vrm, x05vrm, y05vrm, ymax05vrm, max_arrays05vrm, idx_max05vrm, f05vrm, x15vrm, y15vrm, ymax15vrm, \
max_arrays15vrm, idx_max15vrm, f15vrm = \
    sorting(x_maxFR=x_maxFR_05vrm, x_05=x05vrm, y_05=y05vrm, f_05=f05vrm, x_15=x15vrm, y_15=y15vrm, f_15=f15vrm, inter=True)
# interpolate(new_x=x05vrm, x=x15vrm, y=y15vrm)

x_maxFR_05vd, x05vd, y05vd, ymax05vd, max_arrays05vd, idx_max05vd, f05vd, x15vd, y15vd, ymax15vd, max_arrays15vd, \
idx_max15vd, f15vd = \
    sorting(x_maxFR=x_maxFR_05vd, x_05=x05vd, y_05=y05vd, f_05=f05vd, x_15=x15vd, y_15=y15vd, f_15=f15vd, inter=True)
# interpolate(new_x=x05vd, x=x15vd, y=y15vd)


# ___________________________________ MERGING ___________________________________

# y0515 = numpy.insert(y15, numpy.arange(len(y05)), y05)  # arrays in y05 and y15 have to have the same length!
y0515 = merge(begin_array=y05, insert_array=y15)
# y15d.append(numpy.array([numpy.nan]))
# y0515d = numpy.insert(y15d, numpy.arange(len(y05d)), y05d)  # arrays in y05 and y15 have to have the same length!
y0515d = merge(begin_array=y05d, insert_array=y15d)
# y0515d = y0515d[0:-1]
# y15v.append(numpy.array([numpy.nan]))
# y0515v = numpy.insert(y15v, numpy.arange(len(y05v)), y05v)  # arrays in y05 and y15 have to have the same length!

y0515v = merge(begin_array=y05v, insert_array=y15v)
# y0515v = y0515v[0:-1]
# y15vd.append(numpy.array([numpy.nan]))
# y0515vd = numpy.insert(y15vd, numpy.arange(len(y05vd)), y05vd)  # arrays in y05 and y15 have to have the same length!

y0515vd = merge(begin_array=y05vd, insert_array=y15vd)
# y0515vd = y0515vd[0:-1]

ymax0515 = numpy.insert(ymax15, numpy.arange(len(ymax05)), ymax05)
ymax0515d = numpy.insert(ymax15d, numpy.arange(len(ymax05d)), ymax05d)
ymax0515v = numpy.insert(ymax15v, numpy.arange(len(ymax05v)), ymax05v)
ymax0515vd = numpy.insert(ymax15vd, numpy.arange(len(ymax05vd)), ymax05vd)

# ___________________________________ APPENDING ___________________________________

treadmill_y = numpy.concatenate((y0515, y0515d))
treadmill_ymax = numpy.concatenate((ymax0515, ymax0515d))

if only_double:  #not cmaps_on == 1:

    treadmill_y05 = y05d  #numpy.concatenate((y05, y05d))
    treadmill_f05 = f05d  #numpy.concatenate((f05, f05d))
    treadmill_y05max = ymax05d  #numpy.concatenate((ymax05, ymax05d))
    treadmill_y15 = y15d  #numpy.concatenate((y15, y15d))
    treadmill_f15 = f15d  #numpy.concatenate((f15, f15d))
    treadmill_y15max = ymax15d  #numpy.concatenate((ymax15, ymax15d))

    idx_sorted_t05 = numpy.argsort(x_maxFR_05d)  #numpy.concatenate((x_maxFR_05, x_maxFR_05d)))

# ______ only for single cells ___________

else:
    treadmill_y05 = list(y05pm) + list(y05vm) + list(y05rm)
    if not rausmap:
        treadmill_y05_cmaps = list(numpy.repeat(Blues, len(y05pm))) + list(numpy.repeat(Reds, len(y05vm))) + \
                              list(numpy.repeat(Browns, len(y05rm)))
    else:
        treadmill_y05_cmaps = list(numpy.repeat(Blues, len(y05pm))) + list(numpy.repeat(Reds, len(y05vm))) + \
                              list(numpy.repeat(Blues1, len(y05rm)))
    treadmill_x05max = list(x_maxFR_05pm) + list(x_maxFR_05vm) + list(x_maxFR_05rm)
    treadmill_f05 = numpy.concatenate((f05pm, f05vm, f05rm))
    treadmill_y05max = list(ymax05pm) + list(ymax05vm) + list(ymax05rm)

    treadmill_y15 = list(y15pm) + list(y15vm) + list(y15rm)
    if not rausmap:
        treadmill_y15_cmaps = list(numpy.repeat(Blues, len(y15pm))) + list(numpy.repeat(Reds, len(y15vm))) + \
                              list(numpy.repeat(Browns, len(y15rm)))
    else:
        treadmill_y15_cmaps = list(numpy.repeat(Blues, len(y15pm))) + list(numpy.repeat(Reds, len(y15vm))) + \
                              list(numpy.repeat(Blues1, len(y15rm)))
    treadmill_f15 = numpy.concatenate((f15pm, f15vm, f15rm))
    treadmill_y15max = list(ymax15pm) + list(ymax15vm) + list(ymax15rm)

    idx_sorted_t05 = numpy.argsort(numpy.array(treadmill_x05max))

# _______________________________________

visual_y = numpy.concatenate((y0515v, y0515vd))
visual_ymax = numpy.concatenate((ymax0515v, ymax0515vd))

if only_double:  #not cmaps_on == 1:

    visual_y05 = y05vd  #numpy.concatenate((y05v, y05vd))
    visual_f05 = f05vd  #numpy.concatenate((f05v, f05vd))
    visual_y05max = ymax05vd  #numpy.concatenate((ymax05v, ymax05vd))
    visual_y15 = y15vd  #numpy.concatenate((y15v, y15vd))
    visual_f15 = f15vd  #numpy.concatenate((f15v, f15vd))
    visual_y15max = ymax15vd  #numpy.concatenate((ymax15v, ymax15vd))

    idx_sorted_v05 = numpy.argsort(x_maxFR_05vd)  #numpy.concatenate((x_maxFR_05v, x_maxFR_05vd)))


# ______ only for single cells ___________

else:
    visual_y05 = list(y05vpm) + list(y05vvm) + list(y05vrm)
    if not rausmap:
        visual_y05_cmaps = list(numpy.repeat(Blues, len(y05vpm))) + list(numpy.repeat(Reds, len(y05vvm))) + \
                           list(numpy.repeat(Browns, len(y05vrm)))
    else:
        visual_y05_cmaps = list(numpy.repeat(Blues, len(y05vpm))) + list(numpy.repeat(Reds, len(y05vvm))) + \
                           list(numpy.repeat(Blues1, len(y05vrm)))
    visual_x05max = list(x_maxFR_05vpm) + list(x_maxFR_05vvm) + list(x_maxFR_05vrm)
    visual_f05 = numpy.concatenate((f05vpm, f05vvm, f05vrm))
    visual_y05max = list(ymax05vpm) + list(ymax05vvm) + list(ymax05vrm)

    visual_y15 = list(y15vpm) + list(y15vvm) + list(y15vrm)
    if not rausmap:
        visual_y15_cmaps = list(numpy.repeat(Blues, len(y15vpm))) + list(numpy.repeat(Reds, len(y15vvm))) + \
                           list(numpy.repeat(Browns, len(y15vrm)))
    else:
        visual_y15_cmaps = list(numpy.repeat(Blues, len(y15vpm))) + list(numpy.repeat(Reds, len(y15vvm))) + \
                           list(numpy.repeat(Blues1, len(y15vrm)))
    visual_f15 = numpy.concatenate((f15vpm, f15vvm, f15vrm))
    visual_y15max = list(ymax15vpm) + list(ymax15vvm) + list(ymax15vrm)

    idx_sorted_v05 = numpy.argsort(numpy.array(visual_x05max))


if mixed == 1:
    # idx_sorted_t = numpy.argsort(treadmill_ymax)
    # treadmill_y = treadmill_y[idx_sorted_t]
    # treadmill_ymax = treadmill_ymax[idx_sorted_t]


    treadmill_y05 = numpy.array(treadmill_y05)[idx_sorted_t05]
    if cmaps_on == 1:
        treadmill_y05_cmaps = numpy.array(treadmill_y05_cmaps)[idx_sorted_t05]
    treadmill_f05 = numpy.array(treadmill_f05)[idx_sorted_t05]
    treadmill_y05max = numpy.array(treadmill_y05max)[idx_sorted_t05]

    treadmill_y15 = numpy.array(treadmill_y15)[idx_sorted_t05]
    if cmaps_on == 1:
        treadmill_y15_cmaps = numpy.array(treadmill_y15_cmaps)[idx_sorted_t05]
    treadmill_f15 = numpy.array(treadmill_f15)[idx_sorted_t05]
    treadmill_y15max = numpy.array(treadmill_y15max)[idx_sorted_t05]

    # idx_sorted_v = numpy.argsort(visual_ymax)
    # visual_y = visual_y[idx_sorted_v]
    # visual_ymax = visual_ymax[idx_sorted_v]

    visual_y05 = numpy.array(visual_y05)[idx_sorted_v05]
    if cmaps_on == 1:
        visual_y05_cmaps = numpy.array(visual_y05_cmaps)[idx_sorted_v05]
    visual_f05 = numpy.array(visual_f05)[idx_sorted_v05]
    visual_y05max = numpy.array(visual_y05max)[idx_sorted_v05]

    visual_y15 = numpy.array(visual_y15)[idx_sorted_v05]
    if cmaps_on == 1:
        visual_y15_cmaps = numpy.array(visual_y15_cmaps)[idx_sorted_v05]
    visual_f15 = numpy.array(visual_f15)[idx_sorted_v05]
    visual_y15max = numpy.array(visual_y15max)[idx_sorted_v05]


# ___________________________________ PLOTTING ___________________________________

sns.set_style("whitegrid", {'axes.grid': False})
fonts = 60

yaxis_off = 1

if all_together == 1:

    gains = numpy.array(['0.5', '1.5'])

    gainsT = numpy.tile(gains, len(treadmill_y)/2.)
    df_t = pd.DataFrame(list(treadmill_y))

    visual_exp_idx = numpy.where(numpy.array(treadmill_f05) == '10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_PF_info_right')[0][0]
    prop_exp_idx = numpy.where(numpy.array(treadmill_f05) == '10529_2015-03-26_VR_linTrack2_TT3_SS_12_PF_info_right')[0][0]

    figT, axT = pl.subplots(1, 1, figsize=(15, 45))
    hT = sns.heatmap(df_t, cmap='Reds', linewidths=0.0, rasterized=True, ax=axT, cbar=False)  #, cbar_kws={"orientation": "horizontal"})

    axT.axvline((2./1.5)*((df_t.shape[1]-1)/4.)+.5, color='k')
    # if not mixed == 1:
    axT.axhline(len(df_t)-len(ymax0515), color='k')
    axT.xaxis.set_ticks([0, (2./1.5)*((df_t.shape[1]-1)/4.), df_t.shape[1]-1])
    axT.set_xticklabels([0.0, 1.3, 4.0], rotation=0, fontsize=fonts)
    axT.set_yticklabels(numpy.flipud(gainsT), rotation=0)
    axT2 = axT.twinx()
    T2_ylim = axT2.get_ylim()
    Tticks = numpy.arange(1./(len(treadmill_ymax)*2.), T2_ylim[1], T2_ylim[1]/float(len(treadmill_ymax)))
    axT2.set_yticks(Tticks)
    axT2.set_yticklabels(numpy.flipud(treadmill_ymax), rotation=0)
    axT2.set_ylabel('Maximum firing rate (Hz)', fontsize=fonts)
    axT.xaxis.set_ticks_position('none')
    axT.set_xlabel('Treadmill position (m)', fontsize=fonts)
    axT.set_ylabel('Gain', fontsize=fonts)

    y1 = [len(treadmill_f05)-prop_exp_idx-1, len(treadmill_f05)-prop_exp_idx]
    x1 = list(numpy.array(axT.get_xticks().tolist())[numpy.array([0, 2])])
    x2 = list(numpy.array(axT2.get_xticks().tolist())[numpy.array([0, 2])])
    x1 = [.5, x1[1]+.5]
    x2 = [.5, x2[1]+.5]
    axT.plot([x1[0], x1[-1], x1[-1], x1[0], x1[0]], [y1[0], y1[0], y1[-1], y1[-1], y1[0]], '-', color=colors_bl[1])
    axT2.plot([x2[0], x2[-1], x2[-1], x2[0], x2[0]], [y1[0], y1[0], y1[-1], y1[-1], y1[0]], '-', color=colors_bl[1])

    y2 = [len(treadmill_f05)-visual_exp_idx-1, len(treadmill_f05)-visual_exp_idx]
    axT.plot([x1[0], x1[-1], x1[-1], x1[0], x1[0]], [y2[0], y2[0], y2[-1], y2[-1], y2[0]], '-', color=colors_red[1])
    axT2.plot([x2[0], x2[-1], x2[-1], x2[0], x2[0]], [y2[0], y2[0], y2[-1], y2[-1], y2[0]], '-', color=colors_red[1])

    figT.tight_layout()

    print 'Saving figure under: /Users/haasolivia/Desktop/plots/Treadmill_heatmap_allGains.pdf'
    if not rausmap:
        figT.savefig('/Users/haasolivia/Desktop/plots/Treadmill_heatmap_allGains.pdf', format='pdf')
    else:
        figT.savefig('/Users/haasolivia/Desktop/plots/Treadmill_heatmap_allGains_rausmap.pdf', format='pdf')

    gainsV = numpy.tile(gains, len(visual_y)/2.)
    df_v = pd.DataFrame(list(visual_y))

    visual_exp_idx = numpy.where(numpy.array(visual_f05) == '10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_PF_info_right')[0][0]
    prop_exp_idx = numpy.where(numpy.array(visual_f05) == '10529_2015-03-26_VR_linTrack2_TT3_SS_12_PF_info_right')[0][0]

    figV, axV = pl.subplots(1, 1, figsize=(15, 45))
    sns.heatmap(df_v, cmap='Reds', linewidths=0.0, rasterized=True, ax=axV, cbar=False)

    # if not mixed == 1:
    axV.axhline(len(df_v)-len(ymax0515v), color='k')
    axV.xaxis.set_ticks([0, df_v.shape[1]-1])
    axV.set_xticklabels([0.0, 2.0], rotation=0, fontsize=fonts)
    axV.set_yticklabels(numpy.flipud(gainsV), rotation=0)
    axV2 = axV.twinx()
    V2_ylim = axV2.get_ylim()
    Vticks = numpy.arange(1./(len(visual_ymax)*2.), V2_ylim[1], V2_ylim[1]/float(len(visual_ymax)))
    axV2.set_yticks(Vticks)
    axV2.set_yticklabels(numpy.flipud(visual_ymax), rotation=0)
    axV.xaxis.set_ticks_position('none')
    axV.set_xlabel('Virtual position (m)', fontsize=fonts)
    axV.set_ylabel('Gain', fontsize=fonts)
    axV2.set_ylabel('Maximum firing rate (Hz)', fontsize=fonts)

    y1 = [len(visual_f05)-prop_exp_idx-1, len(visual_f05)-prop_exp_idx]
    x1 = list(numpy.array(axV.get_xticks().tolist())[numpy.array([0, 2])])
    x2 = list(numpy.array(axV2.get_xticks().tolist())[numpy.array([0, 2])])
    x1 = [.5, x1[1]+.5]
    x2 = [.5, x2[1]+.5]
    axV.plot([x1[0], x1[-1], x1[-1], x1[0], x1[0]], [y1[0], y1[0], y1[-1], y1[-1], y1[0]], '-', color=colors_bl[1])
    axV2.plot([x2[0], x2[-1], x2[-1], x2[0], x2[0]], [y1[0], y1[0], y1[-1], y1[-1], y1[0]], '-', color=colors_bl[1])

    y2 = [len(visual_f05)-visual_exp_idx-1, len(visual_f05)-visual_exp_idx]
    axV.plot([x1[0], x1[-1], x1[-1], x1[0], x1[0]], [y2[0], y2[0], y2[-1], y2[-1], y2[0]], '-', color=colors_red[1])
    axV2.plot([x2[0], x2[-1], x2[-1], x2[0], x2[0]], [y2[0], y2[0], y2[-1], y2[-1], y2[0]], '-', color=colors_red[1])

    figV.tight_layout()

    print 'Saving figure under: /Users/haasolivia/Desktop/plots/Visual_heatmap_allGains.pdf'
    if not rausmap:
        figV.savefig('/Users/haasolivia/Desktop/plots/Visual_heatmap_allGains.pdf', format='pdf')
    else:
        figV.savefig('/Users/haasolivia/Desktop/plots/Visual_heatmap_allGains_rausmap.pdf', format='pdf')

elif all_together == 0:

    df_t05 = pd.DataFrame(list(treadmill_y05))

    if not only_double:
        visual_exp_idx = numpy.where(numpy.array(treadmill_f05) == '10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_PF_info_right')[0][0]
        prop_exp_idx = numpy.where(numpy.array(treadmill_f05) == '10529_2015-03-26_VR_linTrack2_TT3_SS_12_PF_info_right')[0][0]
        # rem_exp_idx = numpy.where(numpy.array(treadmill_f05) == '10823_2015-08-27_VR_GCend_linTrack1_TT2_SS_11_PF_info_normalised_right')[0][0]
    else:
       double_exp_idx = numpy.where(numpy.array(treadmill_f05) == '10823_2015-07-03_VR_GCendOL_linTrack1_TT3_SS_18_PF_info.hkl')[0][0]

    if not cmaps_on == 1:
        df_t15 = pd.DataFrame(list(treadmill_y15)) #[0:-1]))
        treadmill_y05_cmaps = colors_05  #'Greys'
        treadmill_y15_cmaps = colors_15  #'Greys'
        name = 'Treadmill_heatmap_separateGains.pdf'
        if only_double:
            name = 'Treadmill_heatmap_separateGains_double.pdf'
        if rausmap:
            name = 'Treadmill_heatmap_separateGains_rausmap.pdf'

        figT, axT = pl.subplots(1, 2, figsize=(30, 30))
        axT05, axT15 = axT.flatten()
    else:
        df_t15 = pd.DataFrame(list(treadmill_y15))
        name = 'Treadmill_heatmap_separateGains_noDoubleCells.pdf'
        if rausmap:
            name = 'Treadmill_heatmap_separateGains_noDoubleCells_rausmap.pdf'

        # figT, axT = pl.subplots(2, 2, figsize=(30, 30), sharex='col')
        figT, axT = pl.subplots(len(treadmill_y05_cmaps), 2, figsize=(30, 30), sharex='col')
        # figT.tight_layout()
        figT.subplots_adjust(hspace=-0.001)
        figT.patch.set_visible(False)
        axT05 = axT[:, 0]
        axT15 = axT[:, 1]

    if not cmaps_on == 1:
        sns.heatmap(df_t05, cmap=treadmill_y05_cmaps, linewidths=0.0, rasterized=True, ax=axT05, cbar=False)
        sns.heatmap(df_t15, cmap=treadmill_y15_cmaps, linewidths=0.0, rasterized=True, ax=axT15, cbar=False)
    else:
        for dF in numpy.arange(len(df_t05)):
            print dF, ' out of ', len(df_t05)-1
            sns.heatmap(df_t05._slice(slice(dF, dF+1)).fillna(0), cmap=treadmill_y05_cmaps[dF], linewidths=0.0,
                        rasterized=True, ax=axT05[dF], cbar=False)
            sns.heatmap(df_t15._slice(slice(dF, dF+1)).fillna(0), cmap=treadmill_y15_cmaps[dF], linewidths=0.0,
                        rasterized=True, ax=axT15[dF], cbar=False)

            if dF == 0:
                axT05[dF].set_title('Gain 0.5', fontsize=fonts)
                axT15[dF].set_title('Gain 1.5', fontsize=fonts)
            if dF == len(df_t05)-1:  # for the last row
                axT05 = axT05[dF]
                axT15 = axT15[dF]
            else:
                axT05[dF].axis('off')
                axT15[dF].axis('off')
                axT15[dF].set_xlim([0.0, df_t05.shape[1]])
                axT05[dF].axvline(df_t15.shape[1], color='k')
                axT15[dF].axvline(df_t15.shape[1], color='k')

    x1 = list(numpy.array(axT05.get_xticks().tolist())[numpy.array([0, -1])])
    x2 = list(numpy.array(axT15.get_xticks().tolist())[numpy.array([0, -1])])
    x1 = [.5, x1[1]+.5]
    x2 = [.5, x2[1]+.5]
    if not only_double:
        y1 = [len(df_t05)-prop_exp_idx-1, len(df_t05)-prop_exp_idx]
        axT05.plot([x1[0], x1[-1], x1[-1], x1[0], x1[0]], [y1[0], y1[0], y1[-1], y1[-1], y1[0]], '-', color=colors_bl[1], linewidth=linew)
        axT15.plot([x2[0], x2[-1], x2[-1], x2[0], x2[0]], [y1[0], y1[0], y1[-1], y1[-1], y1[0]], '-', color=colors_bl[1], linewidth=linew)

        y2 = [len(df_t05)-visual_exp_idx-1, len(df_t05)-visual_exp_idx]
        axT05.plot([x1[0], x1[-1], x1[-1], x1[0], x1[0]], [y2[0], y2[0], y2[-1], y2[-1], y2[0]], '-', color=colors_red[1], linewidth=linew)
        axT15.plot([x2[0], x2[-1], x2[-1], x2[0], x2[0]], [y2[0], y2[0], y2[-1], y2[-1], y2[0]], '-', color=colors_red[1], linewidth=linew)

        # y3 = [len(df_t05)-rem_exp_idx-1, len(df_t05)-rem_exp_idx]
        # axT05.plot([x1[0], x1[-1], x1[-1], x1[0], x1[0]], [y3[0], y3[0], y3[-1], y3[-1], y3[0]], '-', color=colors_br[1])
        # axT15.plot([x2[0], x2[-1], x2[-1], x2[0], x2[0]], [y3[0], y3[0], y3[-1], y3[-1], y3[0]], '-', color=colors_br[1])
    else:
        y1 = [len(df_t05)-double_exp_idx-1, len(df_t05)-double_exp_idx]
        axT05.plot([x1[0], x1[-1], x1[-1], x1[0], x1[0]], [y1[0], y1[0], y1[-1], y1[-1], y1[0]], '-', color=double_color, linewidth=linew)
        axT15.plot([x2[0], x2[-1], x2[-1], x2[0], x2[0]], [y1[0], y1[0], y1[-1], y1[-1], y1[0]], '-', color=double_color, linewidth=linew)

    if not mixed == 1:
        axT05.axhline(len(df_t05)-len(ymax05), color='k')
        axT15.axhline(len(df_t15)-len(ymax15), color='k')
    axT05.xaxis.set_ticks([0, df_t15.shape[1]-1, df_t05.shape[1]-1])
    axT15.xaxis.set_ticks([0, df_t15.shape[1]-1, df_t05.shape[1]-1])
    axT05.set_xticklabels([0.0, 1.3, 4.0], rotation=0, fontsize=fonts)
    axT15.set_xlim([0.0, df_t05.shape[1]])
    axT05.axvline(df_t15.shape[1], color='k')
    axT15.axvline(df_t15.shape[1], color='k')
    axT15.set_xticklabels([0.0, 1.3, 4.0], rotation=0, fontsize=fonts)
    if not yaxis_off == 1:
        axT05.set_yticklabels(numpy.flipud(treadmill_y05max), rotation=0)
        axT15.yaxis.tick_right()
        axT15.set_yticklabels(numpy.flipud(treadmill_y15max), rotation=0)
        if not cmaps_on == 1:
            axT05.set_ylabel('Maximum firing rate (Hz)', fontsize=fonts)
        else:
            figT.text(0.03, 0.45, 'Maximum firing rate (Hz)', rotation='vertical', fontsize=fonts)
    else:
        axT05.set_yticklabels('', rotation=0)
        axT15.set_yticklabels('', rotation=0)
    # axT15.set_ylabel('Maximum firing rate (Hz)', fontsize=fonts)
    axT05.xaxis.set_ticks_position('none')
    axT15.xaxis.set_ticks_position('none')
    axT05.set_xlabel('Treadmill position (m)', fontsize=fonts)
    axT15.set_xlabel('Treadmill position (m)', fontsize=fonts)
    if not cmaps_on == 1:
        axT05.set_title('Gain 0.5', fontsize=fonts)
        axT15.set_title('Gain 1.5', fontsize=fonts)
        figT.tight_layout()

    print 'Saving figure under: /Users/haasolivia/Desktop/plots/'+name
    figT.savefig('/Users/haasolivia/Desktop/plots/'+name, format='pdf')

    # visual cells  _______________________________________________________________________________

    df_v05 = pd.DataFrame(list(visual_y05))
    df_v15 = pd.DataFrame(list(visual_y15))

    if not only_double:
        visual_exp_idx = numpy.where(numpy.array(visual_f05) == '10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_PF_info_right')[0][0]
        prop_exp_idx = numpy.where(numpy.array(visual_f05) == '10529_2015-03-26_VR_linTrack2_TT3_SS_12_PF_info_right')[0][0]
    else:
        double_exp_idx = numpy.where(numpy.array(visual_f05) == '10823_2015-07-03_VR_GCendOL_linTrack1_TT3_SS_18_PF_info.hkl')[0][0]

    if not cmaps_on == 1:
        visual_y05_cmaps = colors_05  #'Greys'
        visual_y15_cmaps = colors_15  #'Greys'
        nameV = 'Visual_heatmap_separateGains.pdf'
        if only_double:
            nameV = 'Visual_heatmap_separateGains_double.pdf'
        if rausmap:
            nameV = 'Visual_heatmap_separateGains_rausmap.pdf'

        figV, axV = pl.subplots(1, 2, figsize=(30, 30))
        axV05, axV15 = axV.flatten()
    else:
        nameV = 'Visual_heatmap_separateGains_noDoubleCells.pdf'
        if rausmap:
            nameV = 'Visual_heatmap_separateGains_noDoubleCells_rausmap.pdf'

        figV, axV = pl.subplots(len(visual_y05_cmaps), 2, figsize=(30, 30), sharex='col')
        # figV.tight_layout()
        pl.subplots_adjust(hspace=-0.001)
        figV.patch.set_visible(False)
        axV05 = axV[:, 0]
        axV15 = axV[:, 1]

    if not cmaps_on == 1:
        sns.heatmap(df_v05, cmap=visual_y05_cmaps, linewidths=0.0, rasterized=True, ax=axV05, cbar=False)
        sns.heatmap(df_v15, cmap=visual_y15_cmaps, linewidths=0.0, rasterized=True, ax=axV15, cbar=False)
    else:
        for dFv in numpy.arange(len(df_v05)):
            print dFv, ' out of ', len(df_v05)-1
            sns.heatmap(df_v05._slice(slice(dFv, dFv+1)).fillna(0), cmap=visual_y05_cmaps[dFv], linewidths=0.0,
                        rasterized=True, ax=axV05[dFv], cbar=False)
            sns.heatmap(df_v15._slice(slice(dFv, dFv+1)).fillna(0), cmap=visual_y05_cmaps[dFv], linewidths=0.0,
                        rasterized=True, ax=axV15[dFv], cbar=False)

            # if dFv == 0:
                # axV05[dFv].set_title('Gain 0.5', fontsize=fonts)
                # axV15[dFv].set_title('Gain 1.5', fontsize=fonts)
            if dFv == len(df_v05)-1:  # for the last row
                axV05 = axV05[dFv]
                axV15 = axV15[dFv]
            else:
                axV05[dFv].axis('off')
                axV15[dFv].axis('off')

    x1 = axV05.get_xticks().tolist()
    x2 = axV15.get_xticks().tolist()
    x1[1] = df_v05.shape[1]
    x2[1] = df_v15.shape[1]
    x1 = [.05, x1[1]-.05]
    x2 = [.05, x2[1]-.05]
    if not only_double:
        y1 = [len(df_v05)-visual_exp_idx-1, len(df_v05)-visual_exp_idx]
        axV05.plot([x1[0], x1[-1], x1[-1], x1[0], x1[0]], [y1[0], y1[0], y1[-1], y1[-1], y1[0]], '-', color=colors_red[1], linewidth=linew)
        axV15.plot([x2[0], x2[-1], x2[-1], x2[0], x2[0]], [y1[0], y1[0], y1[-1], y1[-1], y1[0]], '-', color=colors_red[1], linewidth=linew)

        y2 = [len(df_v05)-prop_exp_idx-1, len(df_v05)-prop_exp_idx]
        axV05.plot([x2[0], x2[-1], x2[-1], x2[0], x2[0]], [y2[0], y2[0], y2[-1], y2[-1], y2[0]], '-', color=colors_bl[1], linewidth=linew)
        axV15.plot([x2[0], x2[-1], x2[-1], x2[0], x2[0]], [y2[0], y2[0], y2[-1], y2[-1], y2[0]], '-', color=colors_bl[1], linewidth=linew)

        # y3 = [len(df_v05)-rem_exp_idx-1, len(df_v05)-rem_exp_idx]
        # axV05.plot([x2[0], x2[-1], x2[-1], x2[0], x2[0]], [y3[0], y3[0], y3[-1], y3[-1], y3[0]], '-', color=colors_br[1])
        # axV15.plot([x2[0], x2[-1], x2[-1], x2[0], x2[0]], [y3[0], y3[0], y3[-1], y3[-1], y3[0]], '-', color=colors_br[1])
    else:
        y1 = [len(df_v05)-double_exp_idx-1, len(df_v05)-double_exp_idx]
        axV05.plot([x1[0], x1[-1], x1[-1], x1[0], x1[0]], [y1[0], y1[0], y1[-1], y1[-1], y1[0]], '-', color=double_color, linewidth=linew)
        axV15.plot([x2[0], x2[-1], x2[-1], x2[0], x2[0]], [y1[0], y1[0], y1[-1], y1[-1], y1[0]], '-', color=double_color, linewidth=linew)

    if not mixed == 1:
        axV05.axhline(len(df_v05)-len(ymax05v), color='k')
        axV15.axhline(len(df_v15)-len(ymax15v), color='k')
    axV05.xaxis.set_ticks([0, df_v05.shape[1]-1])
    axV15.xaxis.set_ticks([0, df_v15.shape[1]-1])
    axV05.set_xticklabels(['', ''], rotation=0, fontsize=fonts)
    axV15.set_xticklabels(['', ''], rotation=0, fontsize=fonts)
    # axV05.set_xticklabels([0.0, 2.0], rotation=0, fontsize=fonts)
    # axV15.set_xticklabels([0.0, 2.0], rotation=0, fontsize=fonts)
    if not yaxis_off == 1:
        axV05.set_yticklabels(numpy.flipud(visual_y05max), rotation=0)
        axV15.yaxis.tick_right()
        axV15.set_yticklabels(numpy.flipud(visual_y15max), rotation=0)
        if not cmaps_on == 1:
            axV05.set_ylabel('Maximum firing rate (Hz)', fontsize=fonts)
        else:
            figV.text(0.03, 0.45, 'Maximum firing rate (Hz)', rotation='vertical', fontsize=fonts)
    else:
        axV05.set_yticklabels('', rotation=0)
        axV15.set_yticklabels('', rotation=0)
    # axV15.set_ylabel('Maximum firing rate (Hz)', fontsize=fonts)
    axV05.xaxis.set_ticks_position('none')
    axV15.xaxis.set_ticks_position('none')
    # axV05.set_xlabel('Virtual position (m)', fontsize=fonts)
    # axV15.set_xlabel('Virtual position (m)', fontsize=fonts)
    if not cmaps_on == 1:
        # axV05.set_title('Gain 0.5', fontsize=fonts)
        # axV15.set_title('Gain 1.5', fontsize=fonts)
        figV.tight_layout()

    print 'Saving figure under: /Users/haasolivia/Desktop/plots/'+nameV
    figV.savefig('/Users/haasolivia/Desktop/plots/'+nameV, format='pdf')

else:

    if not maxima_heatmap == 1:
        df_t05p = pd.DataFrame(list(y05pm))
        df_t15p = pd.DataFrame(list(y15pm))
        df_t05v = pd.DataFrame(list(y05vm))
        df_t15v = pd.DataFrame(list(y15vm))
        df_t05r = pd.DataFrame(list(y05rm))
        df_t15r = pd.DataFrame(list(y15rm))

        df_v05p = pd.DataFrame(list(y05vpm))
        df_v15p = pd.DataFrame(list(y15vpm))
        df_v05v = pd.DataFrame(list(y05vvm))
        df_v15v = pd.DataFrame(list(y15vvm))
        df_v05r = pd.DataFrame(list(y05vrm))
        df_v15r = pd.DataFrame(list(y15vrm))

    # or for maxima heatmap
    else:
        df_t05p = pd.DataFrame(list(max_arrays05pm))
        df_t15p = pd.DataFrame(list(max_arrays15pm))
        df_t05v = pd.DataFrame(list(max_arrays05vm))
        df_t15v = pd.DataFrame(list(max_arrays15vm))
        df_t05r = pd.DataFrame(list(max_arrays05rm))
        df_t15r = pd.DataFrame(list(max_arrays15rm))

        df_v05p = pd.DataFrame(list(max_arrays05vpm))
        df_v15p = pd.DataFrame(list(max_arrays15vpm))
        df_v05v = pd.DataFrame(list(max_arrays05vvm))
        df_v15v = pd.DataFrame(list(max_arrays15vvm))
        df_v05r = pd.DataFrame(list(max_arrays05vrm))
        df_v15r = pd.DataFrame(list(max_arrays15vrm))

    if fits == 1:
        xfit = []
        yfit = []
        xfitv = []
        yfitv = []

        #  ___________ gain normalised data fits ___________

        for ind, x1 in enumerate([idx_max05pm, idx_max15pm, idx_max05vm, idx_max15vm, idx_max05rm, idx_max15rm]):
            if len(x1):
                y1 = numpy.array(numpy.repeat(len(x1)-.5, len(x1)) - numpy.array(numpy.arange(len(x1))))
                # if ind in [0, 1]:
                #     slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=x1[:-5], y=y1[:-5])
                # else:
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=x1, y=y1)
                # A = x1[:-11][:, numpy.newaxis]
                # slope, d, d1, d2 = numpy.linalg.lstsq(A, y1[:-11][::-1])
                # intercept = len(x1)-.5
                fit = (y1-intercept)/slope #(slope*x1)+intercept
                xfit.append(fit)
                yfit.append(y1)

        #  ____________ visual data fits __________________

        for x2 in [idx_max05vpm, idx_max15vpm, idx_max05vvm, idx_max15vvm, idx_max05vrm, idx_max15vrm]:
            if len(x2):
                y2 = numpy.repeat(len(x2)-.5, len(x2)) - numpy.array(numpy.arange(len(x2)))
                slope2, intercept2, r_value2, p_value2, std_err2 = scipy.stats.linregress(x=x2, y=y2)
                # A2 = x2[:, numpy.newaxis]
                # slope2, d, d1, d2 = numpy.linalg.lstsq(A2, y2[::-1])
                # intercept2 = len(x2)-.5
                fit2 = (y2-intercept2)/slope2  #(slope2*x2)+intercept2
                xfitv.append(fit2)
                yfitv.append(y2)

        splitnum = 2  # 3
        xfit = numpy.split(numpy.array(xfit), splitnum)
        yfit = numpy.split(numpy.array(yfit), splitnum)
        xfitv = numpy.split(numpy.array(xfitv), splitnum)
        yfitv = numpy.split(numpy.array(yfitv), splitnum)

    co_05 = custom_plot.pretty_colors_set2[0]
    co_15 = custom_plot.pretty_colors_set2[1]

    df_t = [[df_t05p, df_t15p], [df_t05v, df_t15v], [df_t05r, df_t15r]]
    df_v = [[df_v05p, df_v15p], [df_v05v, df_v15v], [df_v05r, df_v15r]]

    if bsp:
        vis_cell = '10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_PF_info_right'
        prop_cell = '10529_2015-03-26_VR_linTrack2_TT3_SS_12_PF_info_right'

        idx_vv = numpy.where(numpy.array(f05vvm) == vis_cell)[0][0]
        idx_tt = numpy.where(numpy.array(f05pm) == prop_cell)[0][0]

    if not maxima_heatmap == 1:
        df_t_name = ['Treadmill_heatmap_prop_singleCells', 'Treadmill_heatmap_vis_singleCells', 'Treadmill_heatmap_remap_singleCells']
        df_v_name = ['Visual_heatmap_prop_singleCells', 'Visual_heatmap_vis_singleCells', 'Visual_heatmap_remap_singleCells']
    else:
        df_t_name = ['Treadmill_heatmap_prop_singleCellsMax', 'Treadmill_heatmap_vis_singleCellsMax', 'Treadmill_heatmap_remap_singleCellsMax']
        df_v_name = ['Visual_heatmap_prop_singleCellsMax', 'Visual_heatmap_vis_singleCellsMax', 'Visual_heatmap_remap_singleCellsMax']

    if not rausmap:
        colormaps = [Blues, Reds, Browns]
    else:
        colormaps = [Blues, Reds, Blues1]

    if not rausmap:
        areas = [0, 1]  # prop, vis, remapping = [0, 1, 2]
    else:
        areas = [2]     # rausmapping

    for cell in areas:

        figT, axT = pl.subplots(1, 2, figsize=(30, 30))
        axT05, axT15 = axT.flatten()

        sns.heatmap(df_t[cell][0].fillna(0), cmap=colormaps[cell], linewidths=0.0, rasterized=True, ax=axT05, cbar=False)
        sns.heatmap(df_t[cell][1].fillna(0), cmap=colormaps[cell], linewidths=0.0, rasterized=True, ax=axT15, cbar=False)

        axT05.plot([df_t[cell][1].shape[1], df_t[cell][1].shape[1]], [0, df_t[cell][1].shape[0]], color='k')
        axT15.plot([df_t[cell][1].shape[1], df_t[cell][1].shape[1]], [0, df_t[cell][1].shape[0]], color='k')

        x1 = list(numpy.array(axT05.get_xticks().tolist())[numpy.array([0, -1])])
        x2 = list(numpy.array(axT15.get_xticks().tolist())[numpy.array([0, -1])])
        x1 = [.5, x1[1]+.5]
        x2 = [.5, x2[1]+.5]

        if cell == 0 and bsp:
            y1 = [df_t[cell][0].shape[0]-idx_tt-1, df_t[cell][0].shape[0]-idx_tt]
            c = colors_bl[1]
        elif cell == 1 and bsp:
            y2 = [df_t[cell][0].shape[0]-idx_vv-1, df_t[cell][0].shape[0]-idx_vv]
            c = colors_red[1]
        if cell in [0, 1] and bsp:
            axT05.plot([x1[0], x1[-1], x1[-1], x1[0], x1[0]], [y1[0], y1[0], y1[-1], y1[-1], y1[0]], '-', color=c, linewidth=linew)
            axT15.plot([x2[0], x2[-1], x2[-1], x2[0], x2[0]], [y1[0], y1[0], y1[-1], y1[-1], y1[0]], '-', color=c, linewidth=linew)

        if fits == 1:
            w = 20
            axT05.plot(xfit[cell][0], yfit[cell][0], color=co_05, linewidth=w)  # gain 0.5 fit
            axT15.plot(xfit[cell][0], yfit[cell][0], color=co_05, linewidth=w)  # gain 0.5 fit
            # axT05.plot(xfit[cell][1], yfit[cell][1], color=co_15, linewidth=w)  # gain 1.5 fit
            axT15.plot(xfit[cell][1], yfit[cell][1], color=co_15, linewidth=w)  # gain 1.5 fit

        axT05.xaxis.set_ticks([0, df_t[cell][1].shape[1]-1, df_t[cell][0].shape[1]-1])
        axT15.xaxis.set_ticks([0, df_t[cell][1].shape[1]-1, df_t[cell][0].shape[1]-1])

        axT15.set_xlim([0.0, df_t[cell][0].shape[1]])

        axT05.set_xticklabels(['', '', ''], rotation=0, fontsize=fonts)
        axT15.set_xticklabels(['', '', ''], rotation=0, fontsize=fonts)
        # axT05.set_xticklabels([0.0, 1.3, 4.0], rotation=0, fontsize=fonts)
        # axT15.set_xticklabels([0.0, 1.3, 4.0], rotation=0, fontsize=fonts)

        axT05.set_yticklabels('', rotation=0)
        axT15.set_yticklabels('', rotation=0)
        axT05.xaxis.set_ticks_position('none')
        axT15.xaxis.set_ticks_position('none')
        # axT05.set_xlabel('Treadmill position (m)', fontsize=fonts)
        # axT15.set_xlabel('Treadmill position (m)', fontsize=fonts)
        axT05.set_title('n = '+str(df_t[cell][1].shape[0]), fontsize=fonts)
        # axT05.set_title('Gain 0.5', fontsize=fonts)
        # axT15.set_title('Gain 1.5', fontsize=fonts)
        figT.tight_layout()

        axT05.set_ylim(-1, df_t[cell][1].shape[0]+1)
        axT15.set_ylim(-1, df_t[cell][1].shape[0]+1)

        if cell == 0:
            axT05.set_ylim(-4, df_t[cell][1].shape[0]+1)
            axT15.set_ylim(-4, df_t[cell][1].shape[0]+1)
            custom_plot.add_scalebar(ax=axT05, matchx=False, matchy=False, hidex=False, hidey=False,
                                     sizex=df_t[cell][1].shape[1]/2., sizey=0, loc=3, thickness=0.5)

        print 'Saving figure under: /Users/haasolivia/Desktop/plots/'+df_t_name[cell]+'.pdf'
        if not rausmap:
            figT.savefig('/Users/haasolivia/Desktop/plots/'+df_t_name[cell]+'.pdf', format='pdf')
        else:
            figT.savefig('/Users/haasolivia/Desktop/plots/'+df_t_name[cell]+'_rausmap.pdf', format='pdf')

        figV, axV = pl.subplots(1, 2, figsize=(30, 30))
        axV05, axV15 = axV.flatten()

        sns.heatmap(df_v[cell][0].fillna(0), cmap=colormaps[cell], linewidths=0.0, rasterized=True, ax=axV05, cbar=False)
        sns.heatmap(df_v[cell][1].fillna(0), cmap=colormaps[cell], linewidths=0.0, rasterized=True, ax=axV15, cbar=False)

        x1 = list(numpy.array(axT05.get_xticks().tolist())[numpy.array([0, -1])])
        x2 = list(numpy.array(axT15.get_xticks().tolist())[numpy.array([0, -1])])
        x1 = [.5, x1[1]+.5]
        x2 = [.5, x2[1]+.5]

        if cell == 0 and bsp:
            y2 = [df_v[cell][0].shape[0]-idx_tt-1, df_v[cell][0].shape[0]-idx_tt]
            c = colors_bl[1]
        elif cell == 1 and bsp:
            y2 = [df_v[cell][0].shape[0]-idx_vv-1, df_v[cell][0].shape[0]-idx_vv]
            c = colors_red[1]
        if cell in [0, 1] and bsp:
            axV05.plot([x1[0], x1[-1], x1[-1], x1[0], x1[0]], [y2[0], y2[0], y2[-1], y2[-1], y2[0]], '-', color=c, linewidth=linew)
            axV15.plot([x2[0], x2[-1], x2[-1], x2[0], x2[0]], [y2[0], y2[0], y2[-1], y2[-1], y2[0]], '-', color=c, linewidth=linew)

        if fits == 1:
            axV05.plot(xfitv[cell][0], yfitv[cell][0], color=co_05, linewidth=w)  # gain 0.5 fit
            axV15.plot(xfitv[cell][0], yfitv[cell][0], color=co_05, linewidth=w)  # gain 0.5 fit
            # axV05.plot(xfitv[cell][1], yfitv[cell][1], color=co_15, linewidth=w)  # gain 1.5 fit
            axV15.plot(xfitv[cell][1], yfitv[cell][1], color=co_15, linewidth=w)  # gain 1.5 fit

        axV05.xaxis.set_ticks([0, df_v[cell][0].shape[1]-1])
        axV15.xaxis.set_ticks([0, df_v[cell][1].shape[1]-1])
        axV05.set_xticklabels(['', ''], rotation=0, fontsize=fonts)
        axV15.set_xticklabels(['', ''], rotation=0, fontsize=fonts)
        # axV05.set_xticklabels([0.0, 2.0], rotation=0, fontsize=fonts)
        # axV15.set_xticklabels([0.0, 2.0], rotation=0, fontsize=fonts)

        axV05.set_yticklabels('', rotation=0)
        axV15.set_yticklabels('', rotation=0)
        # axV15.set_ylabel('Maximum firing rate (Hz)', fontsize=fonts)
        axV05.xaxis.set_ticks_position('none')
        axV15.xaxis.set_ticks_position('none')
        # axV05.set_xlabel('Virtual position (m)', fontsize=fonts)
        # axV15.set_xlabel('Virtual position (m)', fontsize=fonts)

        axV05.set_title('n = '+str(df_v[cell][0].shape[0]), fontsize=fonts)
        # axV05.set_title('Gain 0.5', fontsize=fonts)
        # axV15.set_title('Gain 1.5', fontsize=fonts)
        figV.tight_layout()

        axV05.set_ylim(-1, df_v[cell][0].shape[0]+1)
        axV15.set_ylim(-1, df_v[cell][0].shape[0]+1)

        if cell == 0:
            axV05.set_ylim(-4, df_v[cell][0].shape[0]+1)
            axV15.set_ylim(-4, df_v[cell][0].shape[0]+1)
            custom_plot.add_scalebar(ax=axV05, matchx=False, matchy=False, hidex=False, hidey=False,
                                     sizex=df_v[cell][0].shape[1]/2., sizey=0, loc=3, thickness=0.5)

        print 'Saving figure under: /Users/haasolivia/Desktop/plots/'+df_v_name[cell]+'.pdf'
        if not rausmap:
            figV.savefig('/Users/haasolivia/Desktop/plots/'+df_v_name[cell]+'.pdf', format='pdf')
        else:
            figV.savefig('/Users/haasolivia/Desktop/plots/'+df_v_name[cell]+'_rausmap.pdf', format='pdf')