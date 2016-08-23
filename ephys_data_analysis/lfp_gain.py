__author__ = "Olivia Haas"

# python modules
import sys
import os

import numpy
import matplotlib as mpl
import matplotlib.pyplot as pl
import hickle
from sklearn import mixture
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import csv
from matplotlib import animation
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
import pandas as pd

server = 'saw'
hkl_files = []
path = '/Users/haasolivia/Documents/'+server+'/dataWork/olivia/'


def sliding_mean(x_orig, y_orig, x_win=0.2):

    x_orig1 = x_orig[numpy.logical_not(numpy.isnan(x_orig))]
    y_orig1 = y_orig[numpy.logical_not(numpy.isnan(x_orig))]
    x_orig1 = x_orig1[numpy.logical_not(numpy.isnan(y_orig1))]
    y_orig1 = y_orig1[numpy.logical_not(numpy.isnan(y_orig1))]

    x_and_y = numpy.array(zip(x_orig1, y_orig1))
    x = numpy.unique(x_orig1)
    print 'len(x) = ', len(x)

    y = [x_and_y[:, 1][numpy.where(x_and_y[:, 0] == i)[0]] for i in x]
    print 'y done'
    x = numpy.array(x)
    y = numpy.array(y)


    # window = int(numpy.nanargmin(abs(x - (x[0]+window))))
    # window = numpy.rint(len(x)/70.0)

    # win = [i+int(numpy.nanargmin(abs(x[i:] - (x[i]+x_win)))) for i in numpy.arange(len(x))]
    # print 'window = ', window
    if not len(x) == 0:

        if len(x) != len(y):
            print 'WARNING: SLIDING MEAN ABORTED BECAUSE X AND Y ARRAYS HAVE DIFFERENT LENGTH!'
            sys.exit()

        n = len(numpy.concatenate(y))  # sample size = number of points
        w = numpy.array([int(numpy.rint(numpy.nanargmin(abs(x[i:] - (x[i]+x_win))))) for i in numpy.arange(len(x))])
        w[w == 0] = 1
        w[w > 20000] = 20000
        window = 1000
        print numpy.array(w)

        dummy = [[numpy.mean(numpy.concatenate(y[numpy.arange(max(i - w[i] + 1, 0), min(i + w[i] + 1, len(x)))])),
                  numpy.std(numpy.concatenate(y[numpy.arange(max(i - w[i] + 1, 0), min(i + w[i] + 1, len(x)))])),
                  numpy.percentile(numpy.concatenate(y[numpy.arange(max(i - w[i] + 1, 0), min(i + w[i] + 1, len(x)))]), 25),
                  numpy.percentile(numpy.concatenate(y[numpy.arange(max(i - w[i] + 1, 0), min(i + w[i] + 1, len(x)))]), 75),
                  numpy.median(numpy.concatenate(y[numpy.arange(max(i - w[i] + 1, 0), min(i + w[i] + 1, len(x)))]))]
                 for i in numpy.arange(len(x))]

        avg_y = numpy.array(dummy)[:, 0]
        std_y = numpy.array(dummy)[:, 1]
        perc_25 = numpy.array(dummy)[:, 2]
        perc_75 = numpy.array(dummy)[:, 3]
        median_y = numpy.array(dummy)[:, 4]

        return x, y, avg_y, std_y, n, perc_25, perc_75, median_y

    else:
        return [], [], [], [], [], [], [], []


def nan_pad(array):
    l = []
    for a in numpy.arange(len(array)):
        l.append(len(array[a]))

    for a in numpy.arange(len(array)):
        if len(array[a]) != max(l):
            array[a] = numpy.append(array[a], numpy.repeat(numpy.nan, max(l) - len(array[a])))


#
# speeds_real05 = []  #numpy.array([])
# speeds_real15 = []
# speeds_vir05 = []
# speeds_vir15 = []
# lfp05 = []
# lfp15 = []
#
# for f in os.listdir(path+'hickle/'):
#     if f.endswith('.hkl'):
#         hkl_files.append(f)
#
# visual_hkl_files = []
# normalised_hkl_files = []
# animal_nums = []
#
# for p in numpy.arange(len(hkl_files)):
#     animal_nums.append(hkl_files[p].split('_')[0])
#     if hkl_files[p].endswith('normalised.hkl'):
#         normalised_hkl_files.append(hkl_files[p])
#     else:
#         visual_hkl_files.append(hkl_files[p])
#
# os.chdir('/Users/haasolivia/Documents/'+server+'/dataWork/olivia/hickle/')
#
# animal_nums = numpy.unique(animal_nums)
# length = numpy.zeros(len(animal_nums))
# len_SR = []
# len_z = []
#
# for i in numpy.arange(len(animal_nums)):
#     for j in [speeds_real05, speeds_real15, speeds_vir05, speeds_vir15, lfp05, lfp15]:
#         j.append([])
#
# for title_num, file in enumerate(normalised_hkl_files):
#     print title_num, ' out of ', len(normalised_hkl_files)-1
#     dic = hickle.load(file)
#     animal_loc = numpy.where(numpy.array(animal_nums) == file.split('_')[0])[0][0]
#     speeds_real05[animal_loc].append(dic['speeds_real_gain05'])
#     len_SR.append(len(dic['speeds_real_gain05'][0]))
#     len_z.append(len(dic['speeds_real_gain05']))
#     speeds_real15[animal_loc].append(dic['speeds_real_gain15'])
#     len_SR.append(len(dic['speeds_real_gain15'][0]))
#     len_z.append(len(dic['speeds_real_gain15']))
#     speeds_vir05[animal_loc].append(dic['speeds_virtual_gain05'])
#     len_SR.append(len(dic['speeds_virtual_gain05'][0]))
#     len_z.append(len(dic['speeds_virtual_gain05']))
#     speeds_vir15[animal_loc].append(dic['speeds_virtual_gain15'])
#     len_SR.append(len(dic['speeds_virtual_gain15'][0]))
#     len_z.append(len(dic['speeds_virtual_gain15']))
#     lfp05[animal_loc].append(dic['lfp_freq_gain05'])
#     len_SR.append(len(dic['lfp_freq_gain05'][0]))
#     len_z.append(len(dic['lfp_freq_gain05']))
#     lfp15[animal_loc].append(dic['lfp_freq_gain15'])
#     len_SR.append(len(dic['lfp_freq_gain15'][0]))
#     len_z.append(len(dic['lfp_freq_gain15']))
#     length[animal_loc] += 1
#
#
# for j in [speeds_real05, speeds_real15, speeds_vir05, speeds_vir15, lfp05, lfp15]:
#     for l in numpy.arange(len(animal_nums)):
#         if len(j[l]) != max(length):
#             for k in numpy.arange(max(length) - len(j[l])):
#                 j[l].append([numpy.repeat(numpy.nan, 1)])  #numpy.repeat(numpy.nan, max(length) - len(j[l])))
#         for z in numpy.arange(len(j[l])):
#             if len(j[l][z]) != max(len_z):
#                 for n in numpy.arange(max(len_z) - len(j[l][z])):
#                     j[l][z].append([numpy.repeat(numpy.nan, 1)])
#             # for sr in numpy.arange(len(j[l][z])):
#             #     if len(j[l][z][sr]) != max(len_SR):
#             #         j[l][z][sr] = numpy.append(j[l][z][sr], numpy.repeat(numpy.nan, max(len_SR) - len(j[l][z][sr])))
#
# for j in [speeds_real05, speeds_real15, speeds_vir05, speeds_vir15, lfp05, lfp15]:
#     for l in numpy.arange(len(animal_nums)):
#         for z in numpy.arange(len(j[l])):
#             for sr in numpy.arange(len(j[l][z])):
#                 if len(j[l][z][sr]) != max(len_SR):
#                     j[l][z][sr] = numpy.append(j[l][z][sr], numpy.repeat(numpy.nan, max(len_SR) - len(j[l][z][sr])))
#
# for j in [speeds_real05, speeds_real15, speeds_vir05, speeds_vir15, lfp05, lfp15]:
#     j = numpy.array(j)

    # speeds_real05 = numpy.concatenate((speeds_real05, numpy.concatenate(dic['speeds_real_gain05'])))
    # speeds_real15 = numpy.concatenate((speeds_real15, numpy.concatenate(dic['speeds_real_gain15'])))
    # speeds_vir05 = numpy.concatenate((speeds_vir05, numpy.concatenate(dic['speeds_virtual_gain05'])))
    # speeds_vir15 = numpy.concatenate((speeds_vir15, numpy.concatenate(dic['speeds_virtual_gain15'])))
    # lfp05 = numpy.concatenate((lfp05, numpy.concatenate(dic['lfp_freq_gain05'])))
    # lfp15 = numpy.concatenate((lfp15, numpy.concatenate(dic['lfp_freq_gain15'])))


# speeds_real05 = [[[[animal1_cell1a], [animal1_cell1b]], [[animal1_cell2a], [animal1_call2b]]],
#                  [[[animal2_cell1a], [animal2_cell1b]], [[animal2_cell2a], [animal2_call2b]]]]
dic = hickle.load(path+'Summary/freq_over_speed.hkl')
speeds_real05o = dic['speeds_real05']
speeds_real15o = dic['speeds_real15']
speeds_vir05o = dic['speeds_virtual05']
speeds_vir15o = dic['speeds_virtual15']
lfp05o = dic['lfp05']
lfp15o = dic['lfp15']
animal_nums = dic['animal_nums']

fig, ax = pl.subplots(1, 1, figsize=(10, 10))
fig1, ax1 = pl.subplots(1, 1, figsize=(10, 10))

ax.set_xlabel('Treadmill speed in m/s')
ax.set_ylabel('lfp period in s')
ax.set_xlim(0, 3)
ax.set_ylim(0, 20)

ax1.set_xlabel('Virtual speed in m/s', fontsize=16)
ax1.set_ylabel('lfp period in s', fontsize=16)
ax1.set_xlim(0, 3)
ax1.set_ylim(0, 20)

cl1_line = Line2D((0, 1), (0, 0), color='b', lw=7)
cl2_line = Line2D((0, 1), (0, 0), color='r', lw=7)
fig.legend((cl1_line, cl2_line), ("Gain = 0.5", "Gain = 1.5"), numpoints=1, loc="best", fontsize=15)
fig1.legend((cl1_line, cl2_line), ("Gain = 0.5", "Gain = 1.5"), numpoints=1, loc="best", fontsize=15)

s_real05 = []
s_real15 = []
s_vir05 = []
s_vir15 = []
lfp_05 = []
lfp_15 = []
X05 = []
X15 = []
Xv05 = []
Xv15 = []
M05 = []
M15 = []
Mv05 = []
Mv15 = []
S05 = []
S15 = []
Sv05 = []
Sv15 = []
N05 = []
N15 = []
Nv05 = []
Nv15 = []
P25_05 = []
P25_15 = []
P25_05v = []
P25_15v = []
P75_05 = []
P75_15 = []
P75_05v = []
P75_15v = []
Median05 = []
Median15 = []
Median05v = []
Median15v = []


# for ani in numpy.arange(len(animal_nums)):
#     print 'animal ', ani, ' of ', len(animal_nums)-1
#     for cell in numpy.arange(len(speeds_real05o[ani])):
#         print 'cell ', cell, ' of ', len(speeds_real05o[ani])-1
#         for sr in numpy.arange(len(speeds_real05o[ani][cell])):

print 'begin concatenate'
speeds_real05 = numpy.concatenate(numpy.concatenate(numpy.concatenate(speeds_real05o)))  #[ani][cell][sr]
speeds_real15 = numpy.concatenate(numpy.concatenate(numpy.concatenate(speeds_real15o)))  #[ani][cell][sr]
speeds_vir05 = numpy.concatenate(numpy.concatenate(numpy.concatenate(speeds_vir05o)))  #[ani][cell][sr]
speeds_vir15 = numpy.concatenate(numpy.concatenate(numpy.concatenate(speeds_vir15o)))  #[ani][cell][sr]
lfp05 = numpy.concatenate(numpy.concatenate(numpy.concatenate(lfp05o)))  #[ani][cell][sr]
lfp15 = numpy.concatenate(numpy.concatenate(numpy.concatenate(lfp15o)))  #[ani][cell][sr]
print 'end concatenate'
small_lfp05 = numpy.where(lfp05 < 17)[0]
small_lfp15 = numpy.where(lfp15 < 17)[0]
speeds_real05 = speeds_real05[small_lfp05]
speeds_real15 = speeds_real15[small_lfp15]
speeds_vir05 = speeds_vir05[small_lfp05]
speeds_vir15 = speeds_vir15[small_lfp15]
lfp05 = lfp05[small_lfp05]
lfp15 = lfp15[small_lfp15]
print 'end pre sort'
win = 0.2

# make lpf frequency to period:
lfp05 = 1./lfp05
lfp15 = 1./lfp15

# if not len(numpy.array(speeds_real05)[numpy.logical_not(numpy.isnan(numpy.array(speeds_real05)))]) == 0 \
#         or not len(numpy.array(speeds_real15)[numpy.logical_not(
#                 numpy.isnan(numpy.array(numpy.array(speeds_real15))))]) == 0\
#         or not len(numpy.array(speeds_vir05)[numpy.logical_not(
#                 numpy.isnan(numpy.array(numpy.array(speeds_vir05))))]) == 0\
#         or not len(numpy.array(speeds_vir15)[numpy.logical_not(
#                 numpy.isnan(numpy.array(numpy.array(speeds_vir15))))]) == 0:

x05, y05, mean05, std05, n05, perc25_05, perc75_05, median05 = \
    sliding_mean(x_orig=numpy.array(speeds_real05), y_orig=numpy.array(lfp05), x_win=win)

x15, y15, mean15, std15, n15, perc25_15, perc75_15, median15 = \
    sliding_mean(x_orig=numpy.array(speeds_real15), y_orig=numpy.array(lfp15), x_win=win)

ax.scatter(speeds_real05, lfp05, facecolors='b', edgecolors='none', linewidth='1')
ax.scatter(speeds_real15, lfp15, facecolors='r', edgecolors='none', linewidth='1')

if not x05 == []:
    # ax.plot(x05, mean05, color='b', lw=2, alpha=0.5)
    # ax.fill_between(x05, mean05-std05, mean05+std05, color='#00008B', alpha=0.5)
    ax.plot(x05, median05, color='#00008B', lw=2)
    ax.fill_between(x05, perc25_05, perc75_05, color='#00008B', alpha=0.5)
if not x15 == []:
    # ax.plot(x15, mean15, color='r', lw=2, alpha=0.5)
    # ax.fill_between(x15, mean15-std15, mean15+std15, color='#800000', alpha=0.5)
    ax.plot(x15, median15, color='#800000', lw=2)
    ax.fill_between(x15, perc25_15, perc75_15, color='#800000', alpha=0.5)

x05v, y05v, mean05v, std05v, n05v, perc25_05v, perc75_05v, median05v = \
    sliding_mean(x_orig=numpy.array(speeds_vir05), y_orig=numpy.array(lfp05), x_win=win/0.5)

x15v, y15v, mean15v, std15v, n15v, perc25_15v, perc75_15v, median15v = \
    sliding_mean(x_orig=numpy.array(speeds_vir15), y_orig=numpy.array(lfp15), x_win=win/1.5)

ax1.scatter(speeds_vir15, lfp15, facecolors='r', edgecolors='none', linewidth='1')
ax1.scatter(speeds_vir05, lfp05, facecolors='b', edgecolors='none', linewidth='1')

if not x15v == []:
    # ax1.plot(x15v, mean15v, color='r', lw=2, alpha=0.5)
    # ax1.fill_between(x15v, mean15v-std15v, mean15v+std15v, color='#800000', alpha=0.5)
    ax1.plot(x15v, median15v, color='#800000', lw=2)
    ax1.fill_between(x15v, perc25_15v, perc75_15v, color='#800000', alpha=0.5)
if not x05v == []:
    # ax1.plot(x05v, mean05v, color='b', lw=2, alpha=0.5)
    # ax1.fill_between(x05v, mean05v-std05v, mean05v+std05v, color='#00008B', alpha=0.5)
    ax1.plot(x05v, median05v, color='#00008B', lw=2)
    ax1.fill_between(x05v, perc25_05v, perc75_05v, color='#00008B', alpha=0.5)

if not x05 == []:
    s_real05.append(speeds_real05)
    lfp_05.append(lfp05)
    X05.append(x05)
    M05.append(mean05)
    S05.append(std05)
    N05.append(n05)
    P25_05.append(perc25_05)
    P75_05.append(perc75_05)
    Median05.append(mean05)
if not x15 == []:
    s_real15.append(speeds_real15)
    lfp_15.append(lfp15)
    X15.append(x15)
    M15.append(mean15)
    S15.append(std15)
    N15.append(n15)
    P25_15.append(perc25_15)
    P75_15.append(perc75_15)
    Median15.append(mean15)
if not x05v == []:
    s_vir05.append(speeds_vir05)
    Xv05.append(x05v)
    Mv05.append(mean05v)
    Sv05.append(std05v)
    Nv05.append(n05v)
    P25_05v.append(perc25_05v)
    P75_05v.append(perc75_05v)
    Median05v.append(mean05v)
if not x15v == []:
    s_vir15.append(speeds_vir15)
    Xv15.append(x15v)
    Mv15.append(mean15v)
    Sv15.append(std15v)
    Nv15.append(n15v)
    P25_15v.append(perc25_15v)
    P75_15v.append(perc75_15v)
    Median15v.append(mean15v)


print 'Saving figure under: /Users/haasolivia/Desktop/period_over_speed_median.pdf'
fig.savefig('/Users/haasolivia/Desktop/period_over_speed_median.pdf', format='pdf')
print 'Saving figure under: /Users/haasolivia/Desktop/period_over_speed_virtual_median.pdf'
fig1.savefig('/Users/haasolivia/Desktop/period_over_speed_virtual_median.pdf', format='pdf')

# freq_over_speed_plot_info = {'speeds_real05': speeds_real05, 'speeds_real15': speeds_real15,
#                              'speeds_vir05': speeds_vir05, 'speeds_vir15': speeds_vir15,
#                              'lfp05': lfp05, 'lfp15': lfp15,
#                              'x05': x05, 'mean05': mean05, 'std05': std05, 'n05': n05,
#                              'x15': x15, 'mean15': mean15, 'std15': std15, 'n15': n15,
#                              'x05v': x05v, 'mean05v': mean05v, 'std05v': std05v, 'n05v': n05v,
#                              'x15v': x15v, 'mean15v': mean15v, 'std15v': std15v, 'n15v': n15v}

for array in [s_real05, s_real15, s_vir05, s_vir15, lfp_05, lfp_15, X05, M05, S05, X15, M15, S15, Xv05,
              Mv05, Sv05, Xv15, Mv15, Sv15, P25_05, P25_15, P25_05v, P25_15v, P75_05, P75_15, P75_05v, P75_15v,
              Median05, Median15, Median05v, Median15v]:
    nan_pad(array)

freq_over_speed_plot_info = {'speeds_real05': s_real05, 'speeds_real15': s_real15,
                             'speeds_vir05': s_vir05, 'speeds_vir15': s_vir15,
                             'period05': lfp_05, 'period15': lfp_15,
                             'x05': X05, 'mean05': M05, 'std05': S05, 'n05': N05, 'Perc25_05': P25_05,
                             'Perc75_05': P75_05, 'Median05': Median05,
                             'x15': X15, 'mean15': M15, 'std15': S15, 'n15': N15, 'Perc25_15': P25_15,
                             'Perc75_15': P75_15, 'Median15': Median15,
                             'x05v': Xv05, 'mean05v': Mv05, 'std05v': Sv05, 'n05v': Nv05, 'Perc25_05v': P25_05v,
                             'Perc75_05v': P75_05v, 'Median05v': Median05v,
                             'x15v': Xv15, 'mean15v': Mv15, 'std15v': Sv15, 'n15v': Nv15, 'Perc25_15v': P25_15v,
                             'Perc75_15v': P75_15v, 'Median15v': Median15v}

print 'saving hickle under: '+path+'hickle/Summary/period_over_speed_plot_info.hkl'
hickle.dump(freq_over_speed_plot_info, path+'hickle/Summary/period_over_speed_plot_info.hkl', mode='w')
#
# hz_speed_info = {'speeds_real05': speeds_real05, 'speeds_real15': speeds_real15,
#                  'speeds_virtual05': speeds_vir05, 'speeds_virtual15': speeds_vir15, 'lfp05': lfp05, 'lfp15': lfp15,
#                  'animal_nums': animal_nums}
#
# print 'saving hickle under: '+path+'Summary/freq_over_speed.hkl'
# hickle.dump(hz_speed_info, path+'Summary/freq_over_speed.hkl', mode='w')
