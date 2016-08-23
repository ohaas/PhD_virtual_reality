__author__ = "Olivia Haas"

# python modules
import sys
import os

# add additional custom paths
extraPaths=[
    "/home/haas/packages/lib/python2.6/site-packages",
    os.path.join(os.path.abspath(os.path.dirname(__file__)), '../scripts')]
for p in extraPaths:
    if not sys.path.count(p):
        sys.path.insert(1, p)

# other modules
import numpy
import hickle
import seaborn as sns
from sklearn import mixture

# plotting modules
import matplotlib
import matplotlib.pyplot as pl
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

import itertools
from scipy import linalg
import matplotlib.mlab as mlab
from matplotlib.colors import LogNorm

# custom modules
import CA1_CA3
import custom_plot

server = 'saw'
summ = '/Users/haasolivia/Documents/'+server+'/dataWork/olivia/hickle/Summary/'
addon1 = 'summary_dict'  # visual binned data
# addon2 = '_FRySum'
# addon2 = '_FRySum_cm'
filenames = 'all_filenames.hkl'
# all_dicts = [hickle.load(summ+addon1+addon2+'.hkl'), hickle.load(summ+addon1+'_normalised'+addon2+'.hkl')]
all_dicts = [hickle.load(summ+addon1+'.hkl'), hickle.load(summ+addon1+'_normalised.hkl')]
# gauss_info = hickle.load(summ+'MaxFR_doublePeak_info.hkl')
gauss_info = hickle.load(summ+'Gauss_width_FRmax.hkl')

# categorized file names including double cells___________________________

prop_vis_rem_filenames = hickle.load(summ+'prop_vis_rem_filenames.hkl')
double_cell_files = prop_vis_rem_filenames['double_cell_files']
prop_files = numpy.array(prop_vis_rem_filenames['prop_files'])
vis_files = numpy.array(prop_vis_rem_filenames['vis_files'])
rem_files = numpy.array(prop_vis_rem_filenames['rem_files'])

# remove double cells from files in categories___________________________

prop_files = numpy.array([pr for pr in prop_files if not pr in double_cell_files])
vis_files = numpy.array([vi for vi in vis_files if not vi in double_cell_files])
rem_files = numpy.array([re for re in rem_files if not re in double_cell_files])

# make two filename lists: one ending in info_normalised_direction and one ending in info_direction___________

files = hickle.load(summ+'used_filenames.hkl')  # always end in PF_info.hkl
run_direc = hickle.load(summ+'running_directions.hkl')
norm_files = []
notNorm_files = []

for idx, fi in enumerate(files):
    beginning = fi.split('.hkl')[0]
    norm_files.append(beginning+'_normalised_'+run_direc[idx])
    notNorm_files.append(beginning+'_'+run_direc[idx])

# find indices of those list of files in the categories_________________________________________

prop_files_idx = numpy.array([index for index in numpy.arange(len(norm_files)) if norm_files[index] in prop_files or
                              notNorm_files[index] in prop_files])
vis_files_idx = numpy.array([index for index in numpy.arange(len(norm_files)) if norm_files[index] in vis_files or
                             notNorm_files[index] in vis_files])
rem_files_idx = numpy.array([index for index in numpy.arange(len(norm_files)) if norm_files[index] in rem_files or
                             notNorm_files[index] in rem_files])

prop_color = '#4575b4'
vis_color = '#d73027'
rem_color = '#8e6701'

pp_prop_files = []
pp_vis_files = []
pp_rem_files = []

pp_prop_pics = []
pp_vis_pics = []
pp_rem_pics = []

pp_prop = []
pp_vis = []
pp_rem = []

# GAUSSIAN MIXTURE MODEL FIT:


def fit_samples(samples, xlim=None, ylim=None, fz=15):
    gmix = mixture.GMM(n_components=2, covariance_type='spherical')  # 'full'
    gmix.fit(samples)

    if s == 0:
        cluster1_indices = numpy.where(gmix.predict(samples) == 0)[0]
        cluster2_indices = numpy.where(gmix.predict(samples) == 1)[0]

    else:
        # cluster1 = blue ----- is blue right or left of the parting?
        # if middle1 won, blue is on the right
        if x_clusterParting == middle1:
            cluster1_indices = numpy.where(samples[:, 0] > x_clusterParting)[0]  # blue cluster has to be on the right
            cluster2_indices = numpy.where(samples[:, 0] < x_clusterParting)[0]
        else:
            cluster1_indices = numpy.where(samples[:, 0] < x_clusterParting)[0]  # blue cluster has to be on the left
            cluster2_indices = numpy.where(samples[:, 0] > x_clusterParting)[0]

    #sns.set(style="white")
    J = (sns.JointGrid(samples[:, 0], samples[:, 1], size=8, ratio=9)
            .set_axis_labels(x_label, y_label[s], fontsize=fz))

    J.ax_joint.set_position([.12, .12, .7, .7])
    J.ax_marg_x.set_position([.12, .82, .7, .13])
    J.ax_marg_y.set_position([.82, .12, .13, .7])

    if xlim:
        J.ax_joint.set_xlim(xlim)
        J.ax_marg_x.set_xlim(xlim)

    if ylim:
        J.ax_joint.set_ylim(ylim)
        J.ax_marg_y.set_ylim(ylim)

    hist_alpha = 0.6


    J.ax_marg_x.hist(samples[cluster1_indices, 0], weights=numpy.ones_like(samples[cluster1_indices, 0])/len(samples[cluster1_indices, 0]), color=c_pallett[0],
                     alpha=hist_alpha, bins=numpy.arange(min(samples[:, 0]), max(samples[:, 0]), binwidth[0]), normed=0)
    J.ax_marg_x.hist(samples[cluster2_indices, 0], weights=numpy.ones_like(samples[cluster2_indices, 0])/len(samples[cluster2_indices, 0]), color=c_pallett[1],
                     alpha=hist_alpha, bins=numpy.arange(min(samples[:, 0]), max(samples[:, 0]), binwidth[0]), normed=0)

    if xlim and ylim:
        binwidth[s] = (binwidth[0]/(xlim[1]-xlim[0]))*(ylim[1]-ylim[0])

    #J.ax_marg_y.hist(samples[:, 1], color='#808080', alpha=.9, bins=numpy.arange(min(samples[:, 1]), max(samples[:, 1]), binwidth[s]), orientation="horizontal")
    J.ax_marg_y.hist(samples[cluster1_indices, 1], weights=numpy.ones_like(samples[cluster1_indices, 1])/len(samples[cluster1_indices, 1]), color=c_pallett[0],
                     alpha=hist_alpha, bins=numpy.arange(min(samples[:, 1]), max(samples[:, 1]), binwidth[s]), orientation="horizontal", normed=0)
    J.ax_marg_y.hist(samples[cluster2_indices, 1], weights=numpy.ones_like(samples[cluster2_indices, 1])/len(samples[cluster2_indices, 1]), color=c_pallett[1],
                     alpha=hist_alpha, bins=numpy.arange(min(samples[:, 1]), max(samples[:, 1]), binwidth[s]), orientation="horizontal", normed=0)

    if not CA1 and not CA3:
        CA1_idx, CA3_idx = CA1_CA3.get_CA1CA3_clusteridx()

        cluster1_and_CA1_indices = cluster1_indices[numpy.where(numpy.in1d(cluster1_indices, CA1_idx))[0]]
        cluster1_and_CA3_indices = cluster1_indices[numpy.where(numpy.in1d(cluster1_indices, CA3_idx))[0]]

        cluster2_and_CA1_indices = cluster2_indices[numpy.where(numpy.in1d(cluster2_indices, CA1_idx))[0]]
        cluster2_and_CA3_indices = cluster2_indices[numpy.where(numpy.in1d(cluster2_indices, CA3_idx))[0]]

    elif CA1:
        cluster1_and_CA1_indices = cluster1_indices
        cluster2_and_CA1_indices = cluster2_indices

    elif CA3:
        cluster1_and_CA3_indices = cluster1_indices
        cluster2_and_CA3_indices = cluster2_indices

    # J.x = samples[cluster1_indices, 0]
    # J.y = samples[cluster1_indices, 1]
    # J.plot_joint(pl.scatter, color=c_pallett[0])

    # J.x = samples[cluster2_indices, 0]
    # J.y = samples[cluster2_indices, 1]
    # J.plot_joint(pl.scatter, color=c_pallett[1])

    if CA1 or not CA1 and not CA3:
        J.x = samples[cluster1_and_CA1_indices, 0]
        J.y = samples[cluster1_and_CA1_indices, 1]
        J.plot_joint(pl.scatter, color=c_pallett[0], marker='^')

        J.x = samples[cluster2_and_CA1_indices, 0]
        J.y = samples[cluster2_and_CA1_indices, 1]
        J.plot_joint(pl.scatter, color=c_pallett[1], marker='^')

    if CA3 or not CA1 and not CA3:
        J.x = samples[cluster1_and_CA3_indices, 0]
        J.y = samples[cluster1_and_CA3_indices, 1]
        J.plot_joint(pl.scatter, color=c_pallett[0])

        J.x = samples[cluster2_and_CA3_indices, 0]
        J.y = samples[cluster2_and_CA3_indices, 1]
        J.plot_joint(pl.scatter, color=c_pallett[1])

      # size=8, ratio=9, marginal_kws={'bins': 30}, color=c_pallett[1])
    cl1_tri = Line2D([0], [0], linestyle="none", marker="^", markersize=10, markerfacecolor=c_pallett[0])
    cl1_dot = Line2D([0], [0], linestyle="none", marker="o", markersize=10, markerfacecolor=c_pallett[0])
    cl2_tri = Line2D([0], [0], linestyle="none", marker="^", markersize=10, markerfacecolor=c_pallett[1])
    cl2_dot = Line2D([0], [0], linestyle="none", marker="o", markersize=10, markerfacecolor=c_pallett[1])

    # pl.legend((cl1_dot, cl2_dot), ("n = "+str(len(cluster1_indices)), "n = "+str(len(cluster2_indices))), numpoints=1, loc="best")

    if not CA1 and not CA3:
        pl.legend((cl1_tri, cl1_dot, cl2_tri, cl2_dot), ("CA1 n = "+str(len(cluster1_and_CA1_indices)),
                                                         "CA3 n = "+str(len(cluster1_and_CA3_indices)),
                                                         "CA1 n = "+str(len(cluster2_and_CA1_indices)),
                                                         "CA3 n = "+str(len(cluster2_and_CA3_indices))),
                                                         numpoints=1, loc=4, fontsize=fz)  #"best")
    elif CA1:
        pl.legend((cl1_tri, cl2_tri), ("CA1 n = "+str(len(cluster1_and_CA1_indices)),
                                       "CA1 n = "+str(len(cluster2_and_CA1_indices))), numpoints=1, loc="best", fontsize=fz)
    elif CA3:
        pl.legend((cl1_dot, cl2_dot), ("CA3 n = "+str(len(cluster1_and_CA3_indices)),
                                       "CA3 n = "+str(len(cluster2_and_CA3_indices))), numpoints=1, loc="best", fontsize=fz)

    return gmix, J, cluster1_indices, cluster2_indices


def getClusterParting(samples, cluster1_indices, cluster2_indices):
    middle1 = max(samples[cluster1_indices, 0])+abs(min(samples[cluster1_indices, 0])-max(samples[cluster2_indices, 0]))/2
    middle2 = max(samples[cluster1_indices, 0])+abs(min(samples[cluster2_indices, 0])-max(samples[cluster1_indices, 0]))/2
    x_clusterParting = min(middle1, middle2)
    return x_clusterParting, middle1


class nf(float):
    def __repr__(self):
        str = '%.1f' % (self.__float__(),)
        if str[-1]=='0':
            return '%.0f' % self.__float__()
        else:
            return '%.1f' % self.__float__()


def contours(gauss_mix, ax, fz=15):
    # display predicted scores by the model as a contour plot
    dx = 0.01
    b_all = [numpy.arange(-1.7, 2.1, dx), numpy.arange(-1.5, 1.5, dx), numpy.arange(-45, 120, dx)]

    n = numpy.arange(-1.5, 5.0, dx)
    m = b_all[s]
    A, B = numpy.meshgrid(n, m)
    AA = numpy.array([A.ravel(), B.ravel()]).T
    Z = numpy.exp(gauss_mix.score_samples(AA)[0])
    Z = Z.reshape(A.shape)

    sigma = 68.27
    twoSigma = 95.45
    threeSigma = 99.7
    nearest_sigma = (-sigma/100)+1
    nearest_twoSigma = (-twoSigma/100)+1
    nearest_threeSigma = (-threeSigma/100)+1
    levels = [nearest_sigma, nearest_twoSigma, nearest_threeSigma]

    CS = ax.contour(A, B, Z, levels, alpha=0.5, colors='#808080')
    CS.levels = (1-CS.levels)*100
    CS.levels = [nf(val) for val in CS.levels]

    ax.clabel(CS, CS.levels, inline=True, fmt='%r %%', fontsize=fz)  # CS, inline=1, fontsize=10)


def kde_plots(samples, cluster1_indices, cluster2_indices, axes, shade=True, linestyle='-', fz=15):

    sns.kdeplot(samples[cluster1_indices], shade=shade, color=c_pallett[0], linestyle=linestyle, ax=axes)
    sns.kdeplot(samples[cluster2_indices], shade=shade, color=c_pallett[1], linestyle=linestyle, ax=axes)
    axes.set_xlabel(y_label[s], fontsize=fz)


def hist_plots(samples, cluster1_indices, cluster2_indices, axes, cluster3_indices=numpy.array([]), shade=True,
               linestyle='solid', fz=15, label=True, c1=True, c2=True, c3=True, color1=False, color2=False,
               color3=False, fig=None, xlim=[], ylim=[]):

    bin_num = 10.

    print 'samples', samples
    print 'cluster1_indices', cluster1_indices
    c1_samples = samples[cluster1_indices]
    c2_samples = samples[cluster2_indices]
    c1_samples = c1_samples[~numpy.isnan(c1_samples)]
    c2_samples = c2_samples[~numpy.isnan(c2_samples)]
    if len(cluster3_indices):
        c3_samples = samples[cluster3_indices]
        min_sam = min([min(c1_samples), min(c2_samples), min(c3_samples)])
        max_sam = max([max(c1_samples), max(c2_samples), max(c3_samples)])
    else:
        min_sam = min([min(c1_samples), min(c2_samples)])
        max_sam = max([max(c1_samples), max(c2_samples)])

    # if binwidth[s] < max(c1_samples)+1-min(c1_samples):
    #     c1_bins = numpy.arange(min(c1_samples), max(c1_samples)+1, binwidth[s])
    # else:
    #     c1_bins = numpy.arange(min(c1_samples), max(c1_samples)+1, (max(c1_samples)+1-min(c1_samples))/4)

    if len(xlim):
        min_sam = xlim[0]
        max_sam = xlim[1]

    # if binwidth[s] < max(c2_samples)+1-min(c2_samples):
    #     c2_bins = numpy.arange(min(c2_samples), max(c2_samples)+1, binwidth[s])
    # else:
    #     c2_bins = numpy.arange(min(c2_samples), max(c2_samples)+1, (max(c2_samples)+1-min(c2_samples))/4)
    if len(cluster3_indices):
        c3_samples = c3_samples[~numpy.isnan(c3_samples)]

        # if binwidth[s] < max(c3_samples)+1-min(c3_samples):
        #     c3_bins = numpy.arange(min(c3_samples), max(c3_samples)+1, binwidth[s])
        # else:
        #     c3_bins = numpy.arange(min(c3_samples), max(c3_samples)+1, (max(c3_samples)+1-min(c3_samples))/4)

    c1_bins = numpy.arange(min_sam, max_sam+1, (max_sam-min_sam)/bin_num)
    c2_bins = c1_bins
    c3_bins = c1_bins

    if not color1:
        color1 = c_pallett[0]
    if not color2:
        color2 = c_pallett[1]
    if not color3:
        color3 = c_pallett[2]

    if s % 2 == 0:  # an even number, then visual dictionary (all_dicts[0]) is used
        b = 3
    else:
        b = 0

    if shade:
        sns.distplot(c1_samples, bins=c1_bins, kde=False, color=color1, ax=axes[b],
                    hist_kws={"histtype": "stepfilled", "linewidth": 0, 'alpha': 1.0})
                     # hist_kws={"linestyle": linestyle, 'linewidth': 3})

        sns.distplot(c2_samples, bins=c2_bins, kde=False, color=color2, ax=axes[b+1],
                    hist_kws={"histtype": "stepfilled", "linewidth": 0, 'alpha': 1.0})
                     # hist_kws={"linestyle": linestyle, 'linewidth': 3})

        if len(cluster3_indices):
            sns.distplot(c3_samples, bins=c3_bins, kde=False, color=color3, ax=axes[b+2],
                    hist_kws={"histtype": "stepfilled", "linewidth": 0, 'alpha': 1.0})
                     # hist_kws={"linestyle": linestyle, 'linewidth': 3})
    else:
        lwidth = 2
        if c1:
            sns.distplot(c1_samples, bins=c1_bins, kde=False, color=color1, ax=axes[b],
                         hist_kws={"histtype": "step", "linestyle": 'solid', 'linewidth': lwidth, 'alpha': 1.0})
        if c2:
            sns.distplot(c2_samples, bins=c2_bins, kde=False, color=color2, ax=axes[b+1],
                         hist_kws={"histtype": "step", "linestyle": 'solid', 'linewidth': lwidth, 'alpha': 1.0})
        if c3 and len(cluster3_indices):
            sns.distplot(c3_samples, bins=c3_bins, kde=False, color=color3, ax=axes[b+2],
                         hist_kws={"histtype": "step", "linestyle": 'solid', 'linewidth': lwidth, 'alpha': 1.0})

    samp = [c1_samples, c2_samples, c3_samples]
    if s == 17:
        sub = [-4, -3.5, -0.5]
    elif s == 18:
        sub = [-5, -2, -2]

    if linestyle == 'solid' and s in [17, 18]:
        for num, axi in enumerate([axes[b], axes[b+1], axes[b+2]]):
            l1 = axi.get_yticks().tolist()

            if not len(xlim):
                xmax = numpy.nanmax(samp[num])
            else:
                xmax = xlim[1]

            # axi.text(xmax, numpy.round(l1[-1]+sub[num], 1), 'n = '+str(len(samp[num])), horizontalalignment='right',
            #          verticalalignment='top', zorder=10, fontsize=fz/2., color=color1)

    if linestyle == 'dashed' or shade:
        for num, axi in enumerate([axes[b], axes[b+1], axes[b+2]]):

            if len(xlim):
                axi.set_xlim(xlim)

            l1 = axi.get_yticks().tolist()
            l2 = axi.get_xticks().tolist()

            if not len(xlim):
                xmax = numpy.nanmax(samp[num])
            else:
                xmax = xlim[1]

            if s not in [17, 18]:
                color1 = 'k'

            # axi.text(xmax, numpy.round(l1[-1], 1), 'n = '+str(len(samp[num])), horizontalalignment='right',
            #          verticalalignment='top', zorder=10, fontsize=fz/2., color=color1)


            axi.set_yticks([0, l1[-1]/2., l1[-1]*1.01])
            axi.set_yticklabels(['', '', l1[-1]])

            if num == 2:

                if len(xlim):
                    l2[0] = xlim[0]
                    l2[-1] = xlim[-1]
                    if l2[0] == 0:
                        l2[0] = l2[-1]/2.

                if len(xlim) and l2[0] == -360:
                    axi.set_xticks([l2[0], l2[0]/2., 0, l2[-1]/2., l2[-1]*1.02])
                    axi.set_xticklabels([l2[0], '', 0, '', l2[-1]])
                else:
                    axi.set_xticks([l2[0], 0, l2[-1]*1.01])
                    axi.set_xticklabels([l2[0], 0, l2[-1]])
            elif len(xlim) and l2[0] == -360:
                axi.set_xticks([l2[0], l2[0]/2., 0, l2[-1]/2., l2[-1]*1.02])
                pl.setp(axi.get_xticklabels(), visible=False)
            else:
                axi.set_xticks([l2[0], 0, l2[-1]*1.01])
                pl.setp(axi.get_xticklabels(), visible=False)
            # axi.set_yticklabels(['', '', l1[-1]])

    if label and shade and not s % 2:
        axes[5].set_xlabel(y_label[s], fontsize=fz)
    elif label and s == 18 and linestyle == 'dashed':
        fig.text(.87, .007, y_label[s], fontsize=fz)
        # axes[-1].set_xlabel(y_label[s], fontsize=fz)
    if label and s % 2 and s in [1, 2, 3, 4]:
        axes[b+1].set_ylabel('Treadmill\n\n\nCount', fontsize=fz)
    if label and not s % 2 and s in [1, 2, 3, 4]:
        axes[b+1].set_ylabel('Visual\n\n\nCount', fontsize=fz)


def bar_plots(samples, cluster1_indices, cluster2_indices, axes, shade=True, add=0, fz=15):

    bar_width = 0.7

    xpos1 = .5+add
    xpos2 = 1.5+add

    axes.bar(xpos1, numpy.mean(samples[cluster1_indices]), yerr=numpy.std(samples[cluster1_indices]),
             width=bar_width, error_kw={'ecolor': custom_plot.grau3, 'elinewidth': 1},
             align='center', color=c_pallett[0], edgecolor='none')
    axes.bar(xpos2, numpy.mean(samples[cluster2_indices]), yerr=numpy.std(samples[cluster2_indices]),
             width=bar_width, error_kw={'ecolor': custom_plot.grau3, 'elinewidth': 1},
             align='center', color=c_pallett[1], edgecolor='none')

    if not shade:
        ticklabels = ['Gain 0.5', 'Gain 0.5', 'Gain 1.5', 'Gain 1.5']
        axes.xaxis.set_ticks([0.5, 1.5, 2.5, 3.5])

    else:
        ticklabels = ['Gain 0.5 - Gain 1.5', 'Gain 0.5 - Gain 1.5']

        axes.xaxis.set_ticks([0.5, 1.5])

    axes.xaxis.set_ticklabels(ticklabels, rotation=20, fontsize=fz)

    if s % 2:
        axes.set_xlabel('Real Coordinates', fontsize=fz)
        axes.set_ylabel(y_label[s+1], fontsize=fz)
    else:
        axes.set_xlabel('Virtual Coordinates', fontsize=fz)

    pos = [xpos1, xpos2]
    array = [samples[cluster1_indices], samples[cluster2_indices]]
    for p in [0, 1]:
        for i, v in enumerate(array[p]):
            col = custom_plot.dunkelgrau
            face_col = 'none'
            axes.plot(pos[p], v, 'o', markeredgewidth=1, markeredgecolor=col, markerfacecolor=face_col)
            # if i == highlight_index:
            #     col = custom_plot.tableau20[4]
            #     face_col = col
            # else:


def violin_plots(samples, cluster1_indices, cluster2_indices, axes, shade=True, add=0, fz=15):

    violin_width = 0.7

    xpos1 = .5+add
    xpos2 = 1.5+add

    A = samples[cluster1_indices]
    B = samples[cluster2_indices]
    sns.violinplot(A[~numpy.isnan(A)], widths=violin_width, positions=xpos1, color=c_pallett[0], ax=axes)
    sns.violinplot(B[~numpy.isnan(B)], widths=violin_width, positions=xpos2, color=c_pallett[1], ax=axes)

    if not shade:
        ticklabels = ['Gain 0.5', 'Gain 0.5', 'Gain 1.5', 'Gain 1.5']
        axes.xaxis.set_ticks([0.5, 1.5, 2.5, 3.5])
        axes.set_xlim(0, 4)

    else:
        ticklabels = ['Gain 0.5 - Gain 1.5', 'Gain 0.5 - Gain 1.5']
        axes.xaxis.set_ticks([0.5, 1.5])
        axes.set_xlim(0, 2)

    axes.xaxis.set_ticklabels(ticklabels, rotation=0, fontsize=fz)

    if s % 2:
        axes.set_ylabel(y_label[s+1], fontsize=fz)
        if not shade:
            axes.set_xlabel('Real Coordinates', fontsize=fz)

    elif not shade:
        axes.set_xlabel('Virtual Coordinates', fontsize=fz)


if __name__ == "__main__":

    fz = 15
    # sns.set_style("ticks")
    sns.set(style="ticks", font_scale=1.5)  #, style="white")

    CA1 = False
    CA3 = False

    fig_names = ['XcenterDiffprop_vs_XcenterDiffvis',
                 'XcenterDiffprop_vs_MaxFRdiffprop', 'XcenterDiffprop_vs_MaxFRdiffvis',
                 'XcenterDiffprop_vs_MaxFRprop', 'XcenterDiffprop_vs_MaxFRvis',
                 'XcenterDiffprop_vs_PFwidthDiffprop', 'XcenterDiffprop_vs_PFwidthDiffvis',
                 'XcenterDiffprop_vs_PFwidthprop', 'XcenterDiffprop_vs_PFwidthvis',
                 'XcenterDiffprop_vs_SIDiffprop', 'XcenterDiffprop_vs_SIDiffvis',
                 'XcenterDiffprop_vs_SIprop', 'XcenterDiffprop_vs_SIvis',
                 'XcenterDiffprop_vs_PPDiffprop', 'XcenterDiffprop_vs_PPDiffvis',
                 'XcenterDiffprop_vs_PPprop', 'XcenterDiffprop_vs_PPvis',
                 '', '',
                 'XcenterDiffprop_vs_PFscDiffprop', 'XcenterDiffprop_vs_PFscDiffvis',
                 'XcenterDiffprop_vs_PFscprop', 'XcenterDiffprop_vs_PFscvis']

    # [xcenter diff, FR_diff_prop, FR_diff_vis, FR_prop, FR_vis, width_diff_prop, width_diff_vis, width_prop, width_vis,
    # , SI_diff_prop, SI_diff_vis, SI_prop, SI_vis, PP_diff_prop, PP_diff_vis, PP_prop, PP_vis, PP_sr, PP_sr]
    binwidth = [0.25, 4.0, 4.0, 4.0, 4.0, 0.25, 0.25, 0.25, 0.25, 0.3, 0.3, 0.3, 0.3, 100.0, 100.0, 100.0, 100.0,
                100.0, 100.0, 50.0, 50.0, 50.0, 50.0]

    # like in binwidth every value (except the first one) is loaded twice for respective prop and visual plots

    # real data infos___________________________________________________

    pfc = 'pf_center_change in m (gain 0.5 - gain 1.5 center)'
    PFx05 = 'x_pf_maxFR_gain_0.5 in m'
    PFx15 = 'x_pf_maxFR_gain_1.5 in m'

    FRc = 'pf_maxFR_change in Hz (gain 0.5 - gain 1.5 FR)'
    FR05 = 'pf_maxFR_gain_0.5 in Hz'
    FR15 = 'pf_maxFR_gain_1.5 in Hz'

    wc = 'pf_width_change in m (gain 0.5 - gain 1.5 pf width)'
    w05 = 'pf_width_gain_0.5 in Hz'
    w15 = 'pf_width_gain_1.5 in Hz'

    SIc = 'spatial_info_change in bits (gain 0.5 - gain 1.5 pf width)'
    SI05 = 'spatial_info_gain_0.5 in bits'
    SI15 = 'spatial_info_gain_1.5 in bits'

    PPc = 'pooled_phase_precession_slope_change in degree per field width (gain 0.5 - gain 1.5)'
    PP05 = 'pooled_phase_precession_slopes_gain_0.5 in degree per field width'
    PP15 = 'pooled_phase_precession_slopes_gain_1.5 in degree per field width'

    PPsr05 = 'single_run_phase_precession_slopes_gain_0.5 in degree per field width'
    PPsr15 = 'single_run_phase_precession_slopes_gain_1.5 in degree per field width'
    srIx05 = 'single_run_pp_pooled_indexes_gain_0.5'
    srIx15 = 'single_run_pp_pooled_indexes_gain_1.5'

    SCc = 'pf_spike_count_change'
    SC05 = 'pf_spike_count_0.5'
    SC15 = 'pf_spike_count_1.5'

    aST = 'RZ_exit_aligned_spike_times in s'
    aSD = 'RZ_exit_aligned_spike_distances in cm'

    # gauss fits infos __________________________________________

    GwidthL = 'GaussPFwidthL'
    GwidthS = 'GaussPFwidthS'
    GwidthC = 'GaussPFwidthCombi'
    G_FR_L = 'GaussFRmaxL'
    G_FR_S = 'GaussFRmaxS'
    G_FR_C = 'GaussFRmaxCombi'
    gauss_info

    a = numpy.array([pfc, FRc, FRc, FR05, FR05, wc, wc, w05, w05, SIc, SIc, SI05, SI05, PPc, PPc, PP05, PP05,  # 16
                     PPsr05, PPsr05, SCc, SCc, SC05, SC05])
    xlimits = [None, [-20, 30], [-20, 30], [0, 30], [0, 30], [-1, 2], [-1.5, 1.5], [0, 2], [0, 2], [-1, 2], [-1, 2],
               [0, 3], [0, 3], [-360, 360], [-360, 360], [-360, 360], [-360, 360], [-360, 360], [-360, 360],
               [-20, 45], [-20, 45], [0, 40], [0, 40]]

    c_pallett = sns.color_palette("muted")

    # f_width, ax_width = pl.subplots(2, 2, figsize=(18, 10))  #, sharey='row')  #, sharex=True)
    # f_FR, ax_FR = pl.subplots(2, 2, figsize=(18, 10))  #, sharey='row')  #, sharex=True)
    # f_SI, ax_SI = pl.subplots(2, 2, figsize=(18, 10))  #, sharey='row')  #, sharex=True)
    # f_PP, ax_PP = pl.subplots(2, 2, figsize=(18, 10))  #, sharey='row')  #, sharex=True)
    # f_PPsr, ax_PPsr = pl.subplots(1, 2, figsize=(18, 10))  #, sharey='row')  #, sharex=True)
    # f_spikeC, ax_spikeC = pl.subplots(2, 2, figsize=(18, 10))  #, sharey='row')  #, sharex=True)
    # f_xPF, ax_xPF = pl.subplots(1, 2, figsize=(18, 10))  #, sharey='row')  #, sharex=True)
    #
    # ax_width = ax_width.flatten()
    # ax_FR = ax_FR.flatten()
    # ax_SI = ax_SI.flatten()
    # ax_PP = ax_PP.flatten()
    # ax_PPsr = ax_PPsr.flatten()
    # ax_spikeC = ax_spikeC.flatten()
    # ax_xPF = ax_xPF.flatten()
    #
    # for all_ax in [ax_width, ax_FR, ax_SI, ax_PP, ax_PPsr, ax_spikeC, ax_xPF]:
    #     for ax1 in all_ax:
    #         # Hide the right and top spines
    #         ax1.spines['right'].set_visible(False)
    #         ax1.spines['top'].set_visible(False)
    #         # Only show ticks on the left and bottom spines
    #         ax1.yaxis.set_ticks_position('left')
    #         ax1.xaxis.set_ticks_position('bottom')

    pl.rcParams['xtick.major.pad'] = 1
    pl.rcParams['ytick.major.pad'] = 1

    f_prop, ax_prop = pl.subplots(14, 6, figsize=(12, 10))
    for axo in [-1, -2, -3, -4, -5, -6, -7, -8]:
        ax_prop[axo, -1].axis('off')
        if axo in [-1, -2]:
            for axo1 in numpy.arange(6):
                ax_prop[axo, axo1].axis('off')

    pl.subplots_adjust(wspace=1.2, hspace=0.001)

    # for ti in [1, 3]:
    #     ax_width[ti].yaxis.tick_right()
    #     ax_FR[ti].yaxis.tick_right()
    #     ax_SI[ti].yaxis.tick_right()
    #     ax_PP[ti].yaxis.tick_right()
    #     ax_spikeC[ti].yaxis.tick_right()
    # ax_PPsr[1].yaxis.tick_right()
    # ax_xPF[1].yaxis.tick_right()

    # for figu in [f_width, f_FR, f_SI, f_PP, f_PPsr, f_spikeC]:
    #     figu.subplots_adjust(wspace=0)  #, hspace=0)

    # setting x limits for SI plots:
    # ax_SI[0].set_ylim([-2, 4])
    # ax_SI[1].set_ylim([-4, 4])
    # ax_SI[2].set_ylim([-1, 3])
    # ax_SI[3].set_ylim([-1, 4])
    # for axsi in numpy.arange(len(ax_SI)):
    #     ax_SI[axsi].set_ylim([-4, 10])
        # ax_SI[axsi].set_xlim([-3, 4])
        # ax_SI[axsi].xaxis.set_ticks(numpy.arange(-3, 4, 1))

    # setting x limits for PP plots:
    # for axpp in numpy.arange(len(ax_PP)):
    #     ax_PP[axpp].set_xlim([-1000, 1000])
    #     ax_PP[axpp].xaxis.set_ticks(numpy.arange(-1000, 1000, 250))
    #
    # # setting x limits for srPP plots:
    # for axppsr in numpy.arange(len(ax_PPsr)):
    #     ax_PPsr[axppsr].set_xlim([-1000, 1000])
    #     ax_PPsr[axppsr].xaxis.set_ticks(numpy.arange(-1000, 1000, 250))
    #
    # setting x limits for FR plots:
    # ax_FR[0].set_ylim([-30, 50])
    # ax_FR[1].set_ylim([-30, 40])
    # ax_FR[2].set_ylim([0, 50])
    # ax_FR[3].set_ylim([0, 50])
    # for axfr in [2, 3]:  #numpy.arange(len(ax_FR)):
    #     ax_FR[axfr].set_ylim([0, 80])
    #     ax_FR[axfr].set_xlim([-20, 60])
    #     ax_FR[axfr].xaxis.set_ticks(numpy.arange(-20, 60, 10))
    #
    # setting x limits for width plots:
    # ax_width[0].set_ylim([-2, 3])
    # ax_width[1].set_ylim([-1.5, 1.5])
    # ax_width[2].set_ylim([0, 4])
    # ax_width[3].set_ylim([0, 2])
    # for axw in [2, 3]:  #numpy.arange(len(ax_width)):
    #     ax_width[axw].set_ylim([0, 4])
    #     ax_width[axw].set_xlim([-2, 3])
    #     ax_width[axw].xaxis.set_ticks(numpy.arange(-2, 3, 1))
    # ax_spikeC[0].set_ylim([-2, 6])
    # ax_spikeC[1].set_ylim([-20, 20])
    # ax_spikeC[2].set_ylim([0, 50])
    # ax_spikeC[3].set_ylim([0, 50])

    for s, k in enumerate(a):
        # f_main, axis = pl.subplots(1, 1, figsize=(8, 8))

        x = numpy.asarray(all_dicts[1][a[0]])
        y1 = numpy.asarray(numpy.zeros_like(all_dicts[1][a[0]]))

        if s == 0:
            y = numpy.asarray(all_dicts[0][a[0]])
        elif s % 2 == 0:  # an even number, then visual dictionary (all_dicts[0]) is used
            y = numpy.asarray(all_dicts[0][k])  # gain 0.5 data
            if s in [4, 8, 12, 16, 18, 22]:
                y1 = numpy.asarray(all_dicts[0][k.split('0.5')[0]+'1.5'+k.split('0.5')[1]])  # gain 1.5 data
            if s == 18:  # SR PP index visual data
                PP_indexes_05 = all_dicts[0][srIx05]
                # srPP_indexes_05 = numpy.arange(len(all_dicts[0][PPsr05]))
                PP_indexes_15 = all_dicts[0][srIx15]
                # srPP_indexes_15 = numpy.arange(len(all_dicts[0][PPsr15]))
                prop_files_idx05 = numpy.hstack(numpy.array([numpy.where(PP_indexes_05 == i)[0] for i in prop_files_idx]).flat)
                prop_files_idx15 = numpy.hstack(numpy.array([numpy.where(PP_indexes_15 == i)[0] for i in prop_files_idx]).flat)
                vis_files_idx05 = numpy.hstack(numpy.array([numpy.where(PP_indexes_05 == i)[0] for i in vis_files_idx]).flat)
                vis_files_idx15 = numpy.hstack(numpy.array([numpy.where(PP_indexes_15 == i)[0] for i in vis_files_idx]).flat)
                rem_files_idx05 = numpy.hstack(numpy.array([numpy.where(PP_indexes_05 == i)[0] for i in rem_files_idx]).flat)
                rem_files_idx15 = numpy.hstack(numpy.array([numpy.where(PP_indexes_15 == i)[0] for i in rem_files_idx]).flat)
        else:  # for odd numbers, the normalised dictionary (all_dicts[1]) is used
            y = numpy.asarray(all_dicts[1][k])  # gain 0.5 data
            if s == 19:
                y_dist = numpy.asarray(all_dicts[1][aSD])
            if s in [3, 7, 11, 15, 17, 21]:
                y1 = numpy.asarray(all_dicts[1][k.split('0.5')[0]+'1.5'+k.split('0.5')[1]])  # gain 1.5 data
            if s == 17:  # SR PP index visual data
                PP_indexes_05 = all_dicts[1][srIx05]
                # srPP_indexes_05 = numpy.arange(len(all_dicts[1][PPsr05]))
                PP_indexes_15 = all_dicts[1][srIx15]
                # srPP_indexes_15 = numpy.arange(len(all_dicts[1][PPsr15]))
                prop_files_idx05 = numpy.hstack(numpy.array([numpy.where(PP_indexes_05 == i)[0] for i in prop_files_idx]).flat)
                prop_files_idx15 = numpy.hstack(numpy.array([numpy.where(PP_indexes_15 == i)[0] for i in prop_files_idx]).flat)
                vis_files_idx05 = numpy.hstack(numpy.array([numpy.where(PP_indexes_05 == i)[0] for i in vis_files_idx]).flat)
                vis_files_idx15 = numpy.hstack(numpy.array([numpy.where(PP_indexes_15 == i)[0] for i in vis_files_idx]).flat)
                rem_files_idx05 = numpy.hstack(numpy.array([numpy.where(PP_indexes_05 == i)[0] for i in rem_files_idx]).flat)
                rem_files_idx15 = numpy.hstack(numpy.array([numpy.where(PP_indexes_15 == i)[0] for i in rem_files_idx]).flat)

        if s == 15:  # normalised pp pooled
            prop_not_nans_idx = numpy.where(~numpy.isnan(y[prop_files_idx]))[0]
            pp_prop_file = numpy.array(norm_files)[prop_files_idx][prop_not_nans_idx]
            pp_prop_files.append(pp_prop_file)
            pp_prop.append(y[prop_files_idx][prop_not_nans_idx])
            for filep in pp_prop_file:
                for gain in ['0.5', '1.5']:
                    pp_prop_pics.append(filep.split('PF')[0]+'Gain_'+gain+'_spikePhase_singleRuns_'+
                                        filep.split('normalised_')[1]+'_gain_normalised.pdf')

            vis_not_nans_idx = numpy.where(~numpy.isnan(y[vis_files_idx]))[0]
            pp_vis_file = numpy.array(norm_files)[vis_files_idx][vis_not_nans_idx]
            pp_vis_files.append(pp_vis_file)
            pp_vis.append(y[vis_files_idx][vis_not_nans_idx])
            for filev in pp_vis_file:
                for gain in ['0.5', '1.5']:
                    pp_vis_pics.append(filev.split('PF')[0]+'Gain_'+gain+'_spikePhase_singleRuns_'+
                                       filev.split('normalised_')[1]+'_gain_normalised.pdf')

            rem_not_nans_idx = numpy.where(~numpy.isnan(y[rem_files_idx]))[0]
            pp_rem_file = numpy.array(norm_files)[rem_files_idx][rem_not_nans_idx]
            pp_rem_files.append(pp_rem_file)
            pp_rem.append(y[rem_files_idx][rem_not_nans_idx])
            for filer in pp_rem_file:
                for gain in ['0.5', '1.5']:
                    pp_rem_pics.append(filer.split('PF')[0]+'Gain_'+gain+'_spikePhase_singleRuns_'+
                                       filer.split('normalised_')[1]+'_gain_normalised.pdf')

        with sns.axes_style("white"):
            x_label = 'Place field center difference (real m)'
            y_label = ['Place field center difference (virtual m)',
                       'Normalised Maximal FR change (Hz) \n (gain 0.5 - gain 1.5 FR)',
                       'Firing rate\n(Hz)',  #\n (gain 0.5 - gain 1.5 FR)',
                       'Normalised 1d binned maximal FR (Hz) \n for both gains (0.5 and 1.5)',
                       '1d binned maximal FR (Hz)',  #\n for both gains (0.5 and 1.5)',
                       'Normalised place field width change (m) \n (gain 0.5 - gain 1.5 pf width)',
                       'Place field\nwidth (m)',  #\n (gain 0.5 - gain 1.5 pf width)',
                       'Normalised place field width (m) \n for both gains (0.5 and 1.5)',
                       'Place field width (m)',  #\n for both gains (0.5 and 1.5)',
                       'Normalised spatial info change (bits) \n (gain 0.5 - gain 1.5 pf width)',
                       'Spatial info\n(bits)',  # \n (gain 0.5 - gain 1.5 pf width)',
                       'Normalised spatial info (bits) \n for both gains (0.5 and 1.5)',
                       'Spatial info (bits)',  #\n for both gains (0.5 and 1.5)',
                       'Pooled phase precession slope change for normalised runs ('+unichr(176)+'/field) \n (gain 0.5 - gain 1.5)',
                       'Phase precession\nslope ('+unichr(176)+'/field)',
                       'Pooled phase precession slope for normalised runs ('+unichr(176)+'/field) \n for both gains (0.5 and 1.5)',
                       'Pooled phase precession slope ('+unichr(176)+'/field) \n for both gains (0.5 and 1.5)',
                       'Single run phase precession slope for normalised runs ('+unichr(176)+'/field) \n for both gains (0.5 and 1.5)',
                       'Single run PP\nslope ('+unichr(176)+'/field)',
                       'Normalised place field spike count change \n (gain 0.5 - gain 1.5 pf width)',
                       'Mean single run\nPF spike count',  #\n (gain 0.5 - gain 1.5 pf width)',
                       'Normalised place field spike count \n for both gains (0.5 and 1.5)',
                       'Place field spike count',  #\n for both gains (0.5 and 1.5)',
                       ]

            if k in [SCc, SC05]:
                y[numpy.isinf(y)] = numpy.nan
                y1[numpy.isinf(y1)] = numpy.nan
                y_c = numpy.copy(y)
                y1_c = numpy.copy(y1)
                y[numpy.isnan(y)] = 0
                y1[numpy.isnan(y1)] = 0

            # if s not in [17, 18, 19] and len(x) == len(y) and len(x) == len(y1):
            #     CA1_idx, CA3_idx = CA1_CA3.get_CA1CA3_clusteridx()
            #     if CA1:
            #         xca = x[CA1_idx]
            #         yca = y[CA1_idx]
            #     if CA3:
            #         xca = x[CA3_idx]
            #         yca = y[CA3_idx]
            #     if CA1 or CA3:
            #         XY = numpy.vstack(([xca.T], [yca.T])).T
            #     else:
            #         XY = numpy.vstack(([x.T], [y.T])).T
            #     XY1 = numpy.vstack(([x.T], [y1.T])).T  # for gain 1.5 in cases where both gains are plotted separately
            #
            #     if s == 0:
            #         gmix, J, cluster1_indices, cluster2_indices = fit_samples(XY, xlim=[-1, 4], ylim=[-1.7, 1.5])
            #
            #         J.ax_joint.plot([0, 8./3], [0, 0], color=custom_plot.grau3, zorder=0)
            #         J.ax_joint.plot([0, 0], [-4./3, 0], color=custom_plot.grau3, zorder=0)
            #
            #         J.ax_joint.plot([0, 4], [0, 2], color='k', zorder=0)  # y = 1/2 x
            #         J.ax_joint.plot([8./3, 4], [0, 2], color='k', zorder=0)   # y = 3/2 x - 4
            #         J.ax_joint.plot([-2, 8./3], [-7./3., 0], color='k', zorder=0)  # y = 1/2 x - 4/3
            #         J.ax_joint.plot([-2, 0], [-3, 0], color='k', zorder=0)  # y = 3/2 x
            #
            #     else:
            #         gmix, J, cluster1_indices, cluster2_indices = fit_samples(XY)
            #
            #     if s == 15:  # normalised PP data
            #         cluster1_indices_norm = cluster1_indices
            #         cluster2_indices_norm = cluster2_indices
            #     if s == 16:  # not normalised PP data
            #         cluster1_indices_vis = cluster1_indices
            #         cluster2_indices_vis = cluster2_indices
            #
            # if s == 0:
            #     x_clusterParting, middle1 = getClusterParting(XY, cluster1_indices, cluster2_indices)
            #     contours(gmix, J.ax_joint)
            #     n, m = (100, 100)
            #
            if s == 0:
                n, m = (100, 100)
            elif s in numpy.arange(1, 5):
                n = 1
                m = 2
                # ax = ax_FR
                ax = ax_prop[0:12][:, 0]
                xspace = 0
            elif s in numpy.arange(5, 9):
                n = 5
                m = 6
                # ax = ax_width
                ax = ax_prop[0:12][:, 1]
                xspace = 0.01
            elif s in numpy.arange(9, 13):
                n = 9
                m = 10
                # ax = ax_SI
                ax = ax_prop[0:12][:, 2]
                xspace = 0.02
            elif s in numpy.arange(13, 17):
                n = 13
                m = 14
                # ax = ax_PP
                ax = ax_prop[0:12][:, 4]
                xspace = 0.04
            elif s in numpy.arange(17, 19):
                n = 17
                m = 18
                # ax = ax_PPsr
                ax = ax_prop[0:12][:, 5]
                xspace = 0.05
            elif s in numpy.arange(19, 23):
                n = 19
                m = 20
                # ax = ax_spikeC
                ax = ax_prop[0:12][:, 3]
                xspace = 0.03

            if s != 0:
                for numb, axi in enumerate(ax):
                    # axi.xaxis.set_tick_params(pad=3)
                    # axi.tick_params(axis='x', which='major', pad=3)

                    axi.spines['right'].set_visible(False)
                    axi.spines['top'].set_visible(False)
                    # Only show ticks on the left and bottom spines
                    axi.yaxis.set_ticks_position('left')
                    axi.xaxis.set_ticks_position('bottom')
                    if numb in [0, 1, 3, 4, 6, 7, 9, 10]:
                        axi.get_xaxis().tick_bottom()
                        axi.axes.get_xaxis().set_visible(False)

            # if s == 22:
            #     for axi in ax_prop.flatten():
            #         pos1 = axi.get_position()  # get the original position
            #         pos2 = [pos1.x0, pos1.y0,  pos1.width, pos1.height*0.7]
            #         axi.set_position(pos2)
            #
            #     pl.subplots_adjust(wspace=0.6, hspace=0.001)
            #
            #     space = [0.085, 0.085, 0.085, 0.05, 0.05, 0.05, 0.015, 0.015, 0.015, -0.02, -0.02, -0.02]
            #
            #     for a in [ax_prop[:, 0], ax_prop[:, 1], ax_prop[:, 2], ax_prop[:, 4], ax_prop[:, 5], ax_prop[:, 3]]:
            #         for num, axi in enumerate(a):
            #             pos1 = axi.get_position()  # get the original position
            #             pos2 = [pos1.x0, pos1.y0 + space[num],  pos1.width, pos1.height]
            #             axi.set_position(pos2)

            if s in [1, 5, 9, 13, 17, 21]:
                space = [0.081, 0.081, 0.081, 0.009, 0.009, 0.009, -0.063, -0.063, -0.063, -0.135, -0.135, -0.135]

                for num, axi in enumerate(ax):
                    pos1 = axi.get_position()  # get the original position
                    pos2 = [pos1.x0 + xspace, pos1.y0 + space[num],  pos1.width, pos1.height]
                    axi.set_position(pos2)

            #
            # if s in [17, 18]:  # and len(x) == len(y) and len(x) == len(y1):  # SR_phase_precession for normalised (17) and not normalised data (18)
            #     print 'phase precession on'
            #     if s == 17:
            #         cluster1_idx = cluster1_indices_norm
            #         cluster2_idx = cluster2_indices_norm
            #     else:
            #         cluster1_idx = cluster1_indices_vis
            #         cluster2_idx = cluster2_indices_vis
            #     cluster1_indices_05 = []
            #     cluster1_indices_15 = []
            #     cluster2_indices_05 = []
            #     cluster2_indices_15 = []
            #     for pi in numpy.arange(len(PP_indexes_05)):
            #         if PP_indexes_05[pi] in cluster1_idx:
            #             cluster1_indices_05.append(srPP_indexes_05[pi])
            #         if PP_indexes_05[pi] in cluster2_idx:
            #             cluster2_indices_05.append(srPP_indexes_05[pi])
            #     for pi in numpy.arange(len(PP_indexes_15)):
            #         if PP_indexes_15[pi] in cluster1_idx:
            #             cluster1_indices_15.append(srPP_indexes_15[pi])
            #         if PP_indexes_15[pi] in cluster2_idx:
            #             cluster2_indices_15.append(srPP_indexes_15[pi])
            # else:
            #     if s in [17, 18]:
            #         print 'x = ', x
            #         print 'y = ', y
            #     cluster1_indices_05 = cluster1_indices
            #     cluster1_indices_15 = cluster1_indices
            #     cluster2_indices_05 = cluster2_indices
            #     cluster2_indices_15 = cluster2_indices
            #     # samples05 = XY[:, 1]
            #     # samples15 = XY1[:, 1]
            #
            # if k in [SCc, SC05]:
            #     # print 'the old y: ', y_c
            #     # print 'the old y1: ', y1_c
            #     y = y_c
            #     y1 = y1_c

            print 's = ', s

            gain05_color = custom_plot.pretty_colors_set2[0]
            gain15_color = custom_plot.pretty_colors_set2[1]

            if s in [n, m] and s not in [17, 18] and len(x) == len(y) and len(x) == len(y1):
                # ax[s-n].axhline(0, linestyle='-', color='k', alpha=0.8, zorder=0)  #custom_plot.grau1
                # violin_plots(samples=XY[:, 1], cluster1_indices=cluster1_indices, cluster2_indices=cluster2_indices, axes=ax[s-n])
                # bar_plots(samples=XY[:, 1], cluster1_indices=cluster1_indices, cluster2_indices=cluster2_indices, axes=ax[s-n])
                # hist_plots(samples=y, cluster1_indices=prop_files_idx, cluster2_indices=vis_files_idx, axes=ax[s-n],
                #            cluster3_indices=rem_files_idx, color1=prop_color, color2=vis_color, color3=rem_color)
                hist_plots(samples=y, cluster1_indices=prop_files_idx, cluster2_indices=vis_files_idx, axes=ax[6:],
                           cluster3_indices=rem_files_idx, color1=prop_color, color2=vis_color, color3=rem_color,
                           xlim=xlimits[s])
                # kde_plots(samples=XY[:, 1], cluster1_indices=cluster1_indices, cluster2_indices=cluster2_indices, axes=ax[s-n])
            elif s in [17, 18]:

                hist_plots(samples=y, cluster1_indices=prop_files_idx05, cluster2_indices=vis_files_idx05, axes=ax,
                           cluster3_indices=rem_files_idx05, color1=gain05_color, color2=gain05_color, color3=gain05_color,
                           shade=False, fig=f_prop, xlim=xlimits[s])  # gain 0.5 as solid line, without shading
                hist_plots(samples=y1, cluster1_indices=prop_files_idx15, cluster2_indices=vis_files_idx15, axes=ax,
                           cluster3_indices=rem_files_idx15, color1=gain15_color, color2=gain15_color, color3=gain15_color,
                           shade=False, fig=f_prop, linestyle='dashed', xlim=xlimits[s])  # gain 1.5 as dashed line, without shading

            elif s != 0:  # and len(x) == len(y) and len(x) == len(y1):  # gains have to be plotted separately!
                # ax[s-n].axhline(0, linestyle='-', color='k', alpha=0.8, zorder=0)
                # violin_plots(samples=y, cluster1_indices=cluster1_indices_05, cluster2_indices=cluster2_indices_05,
                #              axes=ax[s-n], shade=False)  # gain 0.5
                # violin_plots(samples=y1, cluster1_indices=cluster1_indices_15, cluster2_indices=cluster2_indices_15,
                #              axes=ax[s-n], shade=False, add=2)  # gain 1.5

                # bar_plots(samples=y, cluster1_indices=cluster1_indices_05, cluster2_indices=cluster2_indices_05,
                #            axes=ax[s-n], shade=False)  # gain 0.5
                # bar_plots(samples=y1, cluster1_indices=cluster1_indices_15, cluster2_indices=cluster2_indices_15,
                #            axes=ax[s-n], shade=False, add=2)  # gain 1.5

                hist_plots(samples=y, cluster1_indices=prop_files_idx, cluster2_indices=vis_files_idx, axes=ax,
                           cluster3_indices=rem_files_idx, color1=gain05_color, color2=gain05_color, color3=gain05_color,
                           shade=False, xlim=xlimits[s])  # gain 0.5 as solid line, without shading
                hist_plots(samples=y1, cluster1_indices=prop_files_idx, cluster2_indices=vis_files_idx, axes=ax,
                           cluster3_indices=rem_files_idx, color1=gain15_color, color2=gain15_color, color3=gain15_color,
                           shade=False, linestyle='dashed', xlim=xlimits[s])  # gain 1.5 as dashed line, without shading

                # kde_plots(samples=y, cluster1_indices=cluster1_indices_05, cluster2_indices=cluster2_indices_05,
                #           axes=ax[s-n], shade=False)  # gain 0.5 as solid line, without shading
                # kde_plots(samples=y1, cluster1_indices=cluster1_indices_15, cluster2_indices=cluster2_indices_15,
                #           axes=ax[s-n], shade=False, linestyle='--')  # gain 1.5 as dashed line, without shading
            # if s == 0:
            #     hist_plots(samples=numpy.asarray(all_dicts[0][PFx05]), cluster1_indices=cluster1_indices_05,
            #                cluster2_indices=cluster2_indices_05, axes=ax_xPF[0], shade=False, c1=False)  # per 0.5 gain 2 clusters
            #     hist_plots(samples=numpy.asarray(all_dicts[0][PFx15]), cluster1_indices=cluster1_indices_05,
            #                cluster2_indices=cluster2_indices_05, axes=ax_xPF[0], shade=False, linestyle='dashed', c1=False)
            #
            #     hist_plots(samples=numpy.asarray(all_dicts[0][PFx05]), cluster1_indices=cluster1_indices_05,
            #                cluster2_indices=cluster2_indices_05, axes=ax_xPF[1], shade=False, c2=False)  # per 0.5 gain 2 clusters
            #     hist_plots(samples=numpy.asarray(all_dicts[0][PFx15]), cluster1_indices=cluster1_indices_05,
            #                cluster2_indices=cluster2_indices_05, axes=ax_xPF[1], shade=False, linestyle='dashed', c2=False)
            #
            #
            #     # hist_plots(samples=numpy.asarray(all_dicts[1][PFx05]), cluster1_indices=cluster1_indices_05,
            #     #            cluster2_indices=cluster2_indices_05, axes=ax_xPF[1], shade=False)
            #     # hist_plots(samples=numpy.asarray(all_dicts[1][PFx15]), cluster1_indices=cluster1_indices_05,
            #     #            cluster2_indices=cluster2_indices_05, axes=ax_xPF[1], shade=False, linestyle='dashed')
            #
            #     ax_xPF[0].set_ylabel('Count', fontsize=fz)
            #     ax_xPF[0].set_xlabel('Place field center locations (virtual m)', fontsize=fz)
            #     ax_xPF[0].set_xlim([0, 2])
            #     ax_xPF[1].set_xlabel('Place field center locations (virtual m)', fontsize=fz)
            #     ax_xPF[1].set_xlim([0, 2])

            # if s not in [0, 17, 18]:
            #     print 'Saving figure under:'+summ+fig_names[s]+'.pdf'
            #     pl.savefig(summ+fig_names[s]+'.pdf', format='pdf')

            if s == len(a)-1:
                cl1_line = Line2D((0, 1), (0, 0), color=prop_color, lw=7)  #c_pallett[0], lw=7)
                cl2_line = Line2D((0, 1), (0, 0), color=vis_color, lw=7)  #c_pallett[1], lw=7)
                cl3_line = Line2D((0, 1), (0, 0), color=rem_color, lw=7)
                full_line = Line2D((0, 1), (0, 0), color='k', lw=7)
                dashed_line = Line2D((0, 1), (0, 0), color='k', linestyle='--', lw=7)
                # for f in [f_FR, f_width, f_SI]:
                #     # f.legend((cl1_line, cl2_line), ("proprioceptive tuning", "visual tuning"), numpoints=1, loc="best", fontsize=15)
                #     f.legend((cl1_line, cl2_line, cl3_line), ("proprioceptive tuning", "visual tuning", "remapping"),
                #              numpoints=1, loc="best", fontsize=15)
                # for f in [f_xPF]:
                #     # f.legend((cl1_line, cl2_line, full_line, dashed_line), ("proprioceptive tuning", "visual tuning",
                #     #                                                         "gain 0.5", "gain 1.5"), numpoints=1, loc="best", fontsize=15)
                #     f.legend((cl1_line, cl2_line, cl3_line, full_line, dashed_line),
                #              ("proprioceptive tuning", "visual tuning", "remapping", "gain 0.5", "gain 1.5"),
                #              numpoints=1, loc="best", fontsize=15)
                # print 'Saving figure under:'+summ+'FR_summary.pdf'
                # f_FR.savefig(summ+'FR_summary.pdf', format='pdf')
                # print 'Saving figure under:'+summ+'PFwidth_summary.pdf'
                # f_width.savefig(summ+'PFwidth_summary.pdf', format='pdf')
                # print 'Saving figure under:'+summ+'SI_summary.pdf'
                # f_SI.savefig(summ+'SI_summary.pdf', format='pdf')
                # print 'Saving figure under:'+summ+'SC_summary.pdf'
                # f_spikeC.savefig(summ+'SC_summary.pdf', format='pdf')
                # # print 'Saving figure under:'+summ+'xPF_summary.pdf'
                # # f_xPF.savefig(summ+'xPF_summary.pdf', format='pdf')
                # print 'Saving figure under:'+summ+'PP_summary.pdf'
                # f_PP.savefig(summ+'PP_summary.pdf', format='pdf')
                # print 'Saving figure under:'+summ+'PPsr_summary.pdf'
                # f_PPsr.savefig(summ+'PPsr_summary.pdf', format='pdf')
                # print 'Saving figure under:'+summ+'aTD_summary.pdf'
                # f_aTD.savefig(summ+'aTDsr_summary.pdf', format='pdf')

                print 'Saving figure under:'+summ+'properties_summary.pdf'
                f_prop.savefig(summ+'properties_summary.pdf', format='pdf')

                # cluster_dic = {'prop_cluster_indices': cluster1_indices,
                #                'vis_cluster_indices': cluster2_indices}
                cluster_dic = {'prop_cluster_indices': prop_files_idx,
                               'vis_cluster_indices': vis_files_idx,
                               'rem_cluster_indices': rem_files_idx}

                hickle.dump(cluster_dic, summ+'cluster_indices.hkl', mode='w')

    pl.show()
