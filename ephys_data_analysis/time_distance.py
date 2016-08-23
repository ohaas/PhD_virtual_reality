__author__ = "Olivia Haas"

import numpy
import hickle
import seaborn as sns

import math
import scipy
import matplotlib.pyplot as pl


server = 'saw'
summ = '/Users/haasolivia/Documents/'+server+'/dataWork/olivia/hickle/Summary/'
addon1 = 'summary_dict'
addon2 = '_FRySum'
all_dicts = [hickle.load(summ+addon1+addon2+'.hkl'), hickle.load(summ+addon1+'_normalised'+addon2+'.hkl')]
clusters = hickle.load(summ+'cluster_indices.hkl')

aST = 'RZ_exit_aligned_spike_times in s'
aSD = 'RZ_exit_aligned_spike_distances in cm'

cluster1_indices = clusters['prop_cluster_indices']
cluster2_indices = clusters['vis_cluster_indices']

# ---------------------------- load data and get their maxima ----------------------
t = all_dicts[1][aST]
dist = all_dicts[1][aSD]
t_vis = all_dicts[0][aST]
dist_vis = all_dicts[0][aSD]

max_plot_t = 10.
max_plot_d = numpy.nanmax(dist)
max_plot_d_vis = numpy.nanmax(dist_vis)

min_t = int(math.floor(numpy.nanmin(t)))
max_t = int(math.ceil(numpy.nanmax(t)))+1
min_d = int(math.floor(numpy.nanmin(dist)))
max_d = int(math.ceil(numpy.nanmax(dist)))+1
min_t_vis = int(math.floor(numpy.nanmin(t_vis)))
max_t_vis = int(math.ceil(numpy.nanmax(t_vis)))+1
min_d_vis = int(math.floor(numpy.nanmin(dist_vis)))
max_d_vis = int(math.ceil(numpy.nanmax(dist_vis)))+1

# ---------------------------- set initial values ----------------------------------

time_bin = 1.  # sec
distance_bin = 25.  # cm  use average running speed -  in time cells for rats about 40 cm/sec

binwidth_d = 2  # cm
max_d_bins = int(math.ceil(max_plot_d/binwidth_d))+1
max_d_bins_vis = int(math.ceil(max_plot_d_vis/binwidth_d))+1

# binwidth_t = max_plot_t/max_d_bins  # sec
binwidth_t = binwidth_d/distance_bin
max_t_bins = int(math.ceil(max_plot_t/binwidth_t))+1
# max_binNum_t = max_plot_t/binwidth_t  # sec/binwidth

smoothing_sigma = 2

# ---------------------------- get figures and axes ---------------------------------

f_t, ax_t = pl.subplots(1, 2, sharey=True, figsize=(18, 5))
f_t.subplots_adjust(wspace=0)
ax_t = ax_t.flatten()
f_t_vis, ax_t_vis = pl.subplots(1, 2, sharey=True, figsize=(18, 5))
f_t_vis.subplots_adjust(wspace=0)
ax_t_vis = ax_t_vis.flatten()

f_d, ax_d = pl.subplots(1, 2, sharey=True, figsize=(18, 5))
f_d.subplots_adjust(wspace=0)
ax_d = ax_d.flatten()
f_d_vis, ax_d_vis = pl.subplots(1, 2, sharey=True, figsize=(18, 5))
f_d_vis.subplots_adjust(wspace=0)
ax_d_vis = ax_d_vis.flatten()

# max_t_bins = int(math.ceil(max_plot_t/binwidth_t))+1
# max_d_bins = int(math.ceil(max_plot_d/binwidth_d))+1

# ---------------------------- initiate histogram lists ------------------------------

ST_hist_timeTuning = []
SD_hist_timeTuning = []
ST_hist_distTuning = []
SD_hist_distTuning = []
ST_hist_equalTuning = []
SD_hist_equalTuning = []

ST_hist_timeTuning_vis = []
SD_hist_timeTuning_vis = []
ST_hist_distTuning_vis = []
SD_hist_distTuning_vis = []
ST_hist_equalTuning_vis = []
SD_hist_equalTuning_vis = []

# c_pallett_heat = sns.color_palette("RdYlBu")
# indices have to be in propropceptive cluster: cluster1_indices

# ---------------------------- definitions ------------------------------------------


def get_histData(array, binwidth_a):
    bins = numpy.arange(int(numpy.nanmin(array)), numpy.nanmax(array) + binwidth_a, binwidth_a)
    # removing nans from array
    x = numpy.array(array)
    x1 = x[numpy.logical_not(numpy.isnan(x))]
    data = numpy.histogram(x1, bins=len(bins))
    return data


def heatmap(array, ax, mini, maxi, stepsize, binMax, label='', ticks=4):
    # main plot
    sns.heatmap(array, cmap=pl.cm.jet, linewidths=0.0, rasterized=True, xticklabels=False,
                yticklabels=False, cbar=False, ax=ax)
    # set and label xticks
    ax.xaxis.set_ticks(numpy.arange(mini, binMax+stepsize, stepsize))
    a = ax.get_xticks().tolist()
    if ticks == 4:
        b = numpy.round(numpy.arange(mini, maxi+(maxi/ticks), maxi/ticks), 1)
    elif ticks == 3:
        b = numpy.concatenate((numpy.arange(mini, maxi, maxi/ticks), numpy.array([''])), axis=1)
    for i in numpy.arange(len(a)):
        if i < len(b):
            a[i] = b[i]
    ax.set_xticklabels(a)
    # label axis
    ax.set_xlabel(label)
    # limit x axis
    ax.set_xlim([0, binMax])


def consecutive(data, stepsize=1):
    return numpy.split(data, numpy.where(numpy.diff(data) != stepsize)[0]+1)


def delta(data, max_percentage=5):
    lower = data[1][numpy.where(data[0][:numpy.argmax(data[0])+1] >= max(data[0])*0.01*max_percentage)[0][0]]
    upper = data[1][consecutive(numpy.where(data[0][numpy.argmax(data[0]):] >= max(data[0])*0.01*max_percentage)[0]+
                                numpy.argmax(data[0]))[0][-1]]
    delta_data = upper - lower
    return delta_data

# ---------------------------- main loop ------------------------------------------

if __name__ == "__main__":

    for i in numpy.arange(len(t)):
        if i in cluster1_indices:  # proprioceptive cell indices
            # get histogram Data
            data_t = get_histData(array=t[i], binwidth_a=binwidth_t)
            data_d = get_histData(array=dist[i], binwidth_a=binwidth_d)

            # get values for 5% time and distance pf boundaries
            delta_t = delta(data=data_t)/time_bin
            delta_d = delta(data=data_d)/distance_bin

            # remove all time data, which is larger than max_plot_t, because it wont be plotted:
            smaller_idx = numpy.where(data_t[1][:-1] < 10.0)[0]
            data_t0 = data_t[0][smaller_idx].copy()
            data_d0 = data_d[0].copy()
            # smooth array
            data_t0 = scipy.ndimage.filters.gaussian_filter1d(data_t0, smoothing_sigma)
            data_d0 = scipy.ndimage.filters.gaussian_filter1d(data_d0, smoothing_sigma)
            if len(data_t0) != max_t_bins+1:
                 data_t0 = numpy.concatenate((data_t0, numpy.repeat(0, max_t_bins+1-len(data_t0))), axis=1)
            if len(data_d0) != max_d_bins:
                 data_d0 = numpy.concatenate((data_d0, numpy.repeat(0, max_d_bins-len(data_d0))), axis=1)
            if delta_t < delta_d:
                ST_hist_timeTuning.append(list(data_t0.astype(float)/max(data_t0)))
                SD_hist_timeTuning.append(list(data_d0.astype(float)/max(data_d0)))
            elif delta_d < delta_t:
                ST_hist_distTuning.append(list(data_t0.astype(float)/max(data_t0)))
                SD_hist_distTuning.append(list(data_d0.astype(float)/max(data_d0)))
            else:
                ST_hist_equalTuning.append(list(data_t0.astype(float)/max(data_t0)))
                SD_hist_equalTuning.append(list(data_d0.astype(float)/max(data_d0)))

    for v in numpy.arange(len(t_vis)):
        if i in cluster2_indices:  # visual cell indices
            # get histogram Data
            data_t_vis = get_histData(array=t_vis[i], binwidth_a=binwidth_t)
            data_d_vis = get_histData(array=dist_vis[i], binwidth_a=binwidth_d)

            # get values for 5% time and distance pf boundaries
            delta_t_vis = delta(data=data_t_vis)/time_bin
            delta_d_vis = delta(data=data_d_vis)/distance_bin

            # remove all time data, which is larger than max_plot_t, because it wont be plotted:
            smaller_idx_vis = numpy.where(data_t_vis[1][:-1] < 10.0)[0]
            data_t0_vis = data_t_vis[0][smaller_idx_vis].copy()
            data_d0_vis = data_d_vis[0].copy()
            # smooth array
            data_t0_vis = scipy.ndimage.filters.gaussian_filter1d(data_t0_vis, smoothing_sigma)
            data_d0_vis = scipy.ndimage.filters.gaussian_filter1d(data_d0_vis, smoothing_sigma)
            if len(data_t0_vis) != max_t_bins+1:
                 data_t0_vis = numpy.concatenate((data_t0_vis, numpy.repeat(0, max_t_bins+1-len(data_t0_vis))), axis=1)
            if len(data_d0_vis) != max_d_bins_vis:
                 data_d0_vis = numpy.concatenate((data_d0_vis, numpy.repeat(0, max_d_bins_vis-len(data_d0_vis))), axis=1)
            if delta_t_vis < delta_d_vis:
                ST_hist_timeTuning_vis.append(list(data_t0_vis.astype(float)/max(data_t0_vis)))
                SD_hist_timeTuning_vis.append(list(data_d0_vis.astype(float)/max(data_d0_vis)))
            elif delta_d_vis < delta_t_vis:
                ST_hist_distTuning_vis.append(list(data_t0_vis.astype(float)/max(data_t0_vis)))
                SD_hist_distTuning_vis.append(list(data_d0_vis.astype(float)/max(data_d0_vis)))
            else:
                ST_hist_equalTuning_vis.append(list(data_t0_vis.astype(float)/max(data_t0_vis)))
                SD_hist_equalTuning_vis.append(list(data_d0_vis.astype(float)/max(data_d0_vis)))

    for h in [SD_hist_distTuning, ST_hist_distTuning, ST_hist_timeTuning, SD_hist_timeTuning, ST_hist_equalTuning,
              SD_hist_equalTuning,
              SD_hist_distTuning_vis, ST_hist_distTuning_vis, ST_hist_timeTuning_vis, SD_hist_timeTuning_vis,
              ST_hist_equalTuning_vis, SD_hist_equalTuning_vis]:
        h = numpy.array(h)
    # SD_hist_distTuning = numpy.array(SD_hist_distTuning)
    # ST_hist_distTuning = numpy.array(ST_hist_distTuning)
    #
    # ST_hist_timeTuning = numpy.array(ST_hist_timeTuning)
    # SD_hist_timeTuning = numpy.array(SD_hist_timeTuning)
    #
    # ST_hist_equalTuning = numpy.array(ST_hist_equalTuning)
    # SD_hist_equalTuning = numpy.array(SD_hist_equalTuning)

    max_idx_t = []
    max_idx_d = []
    max_idx_t_vis = []
    max_idx_d_vis = []

    for i in numpy.arange(len(SD_hist_distTuning)):
        max_idx_d.append(numpy.where(SD_hist_distTuning[i] == max(SD_hist_distTuning[i]))[0][0])
    for i in numpy.arange(len(ST_hist_timeTuning)):
        max_idx_t.append(numpy.where(ST_hist_timeTuning[i] == max(ST_hist_timeTuning[i]))[0][0])

    for i in numpy.arange(len(SD_hist_distTuning_vis)):
        max_idx_d_vis.append(numpy.where(SD_hist_distTuning_vis[i] == max(SD_hist_distTuning_vis[i]))[0][0])
    for i in numpy.arange(len(ST_hist_timeTuning_vis)):
        max_idx_t_vis.append(numpy.where(ST_hist_timeTuning_vis[i] == max(ST_hist_timeTuning_vis[i]))[0][0])

    sorted_idx_t = numpy.argsort(max_idx_t)
    sorted_idx_d = numpy.argsort(max_idx_d)
    sorted_idx_t_vis = numpy.argsort(max_idx_t_vis)
    sorted_idx_d_vis = numpy.argsort(max_idx_d_vis)

    ST_hist_tSort = ST_hist_timeTuning[sorted_idx_t]
    SD_hist_tSort = SD_hist_timeTuning[sorted_idx_t]
    ST_hist_tSort_vis = ST_hist_timeTuning_vis[sorted_idx_t_vis]
    SD_hist_tSort_vis = SD_hist_timeTuning_vis[sorted_idx_t_vis]

    ST_hist_dSort = ST_hist_distTuning[sorted_idx_d]
    SD_hist_dSort = SD_hist_distTuning[sorted_idx_d]
    ST_hist_dSort_vis = ST_hist_distTuning_vis[sorted_idx_d_vis]
    SD_hist_dSort_vis = SD_hist_distTuning_vis[sorted_idx_d_vis]

    tickNum = 4

    f_t.suptitle(str(len(ST_hist_tSort))+' time tuned cells')
    f_d.suptitle(str(len(ST_hist_dSort))+' distance tuned cells')
    f_t_vis.suptitle(str(len(ST_hist_tSort))+' time tuned cells')
    f_d_vis.suptitle(str(len(ST_hist_dSort))+' distance tuned cells')

    heatmap(ST_hist_tSort, ax_t[0], mini=min_t, maxi=max_plot_t, stepsize=(max_t_bins+1)/tickNum, binMax=max_t_bins+1,
            label='Time from RZ exit in s', ticks=3)
    heatmap(SD_hist_tSort, ax_t[1], mini=min_d, maxi=max_plot_d, stepsize=max_d_bins/tickNum, binMax=max_d_bins,
            label='Distance from RZ exit in cm')

    heatmap(ST_hist_dSort, ax_d[0], mini=min_t, maxi=max_plot_t, stepsize=(max_t_bins+1)/tickNum, binMax=max_t_bins+1,
            label='Time from RZ exit in s', ticks=3)
    heatmap(SD_hist_dSort, ax_d[1], mini=min_d, maxi=max_plot_d, stepsize=max_d_bins/tickNum, binMax=max_d_bins,
            label='Distance from RZ exit in cm')

    heatmap(ST_hist_tSort_vis, ax_t_vis[0], mini=min_t_vis, maxi=max_plot_t, stepsize=(max_t_bins+1)/tickNum,
            binMax=max_t_bins+1, label='Time from RZ exit in s', ticks=3)
    heatmap(SD_hist_tSort_vis, ax_t_vis[1], mini=min_d_vis, maxi=max_plot_d_vis, stepsize=max_d_bins_vis/tickNum,
            binMax=max_d_bins_vis, label='Distance from RZ exit in cm')

    heatmap(ST_hist_dSort_vis, ax_d_vis[0], mini=min_t_vis, maxi=max_plot_t, stepsize=(max_t_bins+1)/tickNum,
            binMax=max_t_bins+1, label='Time from RZ exit in s', ticks=3)
    heatmap(SD_hist_dSort_vis, ax_d_vis[1], mini=min_d_vis, maxi=max_plot_d_vis, stepsize=max_d_bins_vis/tickNum,
            binMax=max_d_bins_vis, label='Distance from RZ exit in cm')


    print 'Saving figure under:'+summ+'propTimeCells_summary.pdf'
    f_t.savefig(summ+'propTimeCells_summary.pdf', format='pdf')
    print 'Saving figure under:'+summ+'propDistanceCells_summary.pdf'
    f_d.savefig(summ+'propDistanceCells_summary.pdf', format='pdf')

    print 'Saving figure under:'+summ+'visTimeCells_summary.pdf'
    f_t_vis.savefig(summ+'visTimeCells_summary.pdf', format='pdf')
    print 'Saving figure under:'+summ+'visDistanceCells_summary.pdf'
    f_d_vis.savefig(summ+'visDistanceCells_summary.pdf', format='pdf')

    if len(ST_hist_equalTuning):
        print 'WARNING: there are '+str(len(ST_hist_equalTuning))+' cells with equal tuning!'

    pl.show()
