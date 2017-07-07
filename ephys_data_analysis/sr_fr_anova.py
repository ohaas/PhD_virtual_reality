"""
For plotting calculating one way ANOVA for comparing single run FR for both gains.
"""

__author__ = "Olivia Haas"
__version__ = "1.0, June 2017"

# python modules
import sys
import os

# add additional custom paths
extraPaths = [os.path.join(os.path.abspath(os.path.dirname(__file__)), '../scripts'),
              os.path.join(os.path.abspath(os.path.dirname(__file__)), '/opt/anaconda/bin/python'),
              os.path.join(os.path.abspath(os.path.dirname(__file__)), '/opt/anaconda/pkgs')]

for p in extraPaths:
    if not sys.path.count(p):
        sys.path.insert(1, p)

# plotting modules
import scipy
import matplotlib as mpl
import matplotlib.pyplot as pl

# other modules
import numpy
import hickle
import scipy.stats as st

# custom made modules
import signale

######################################################

server = 'saw'
names = hickle.load('/users/haasolivia/documents/'+server+'/dataWork/olivia/hickle/Summary/all_fields.hkl')
raw_data = hickle.load('/users/haasolivia/Desktop/raw_data_info.hkl')
vis = numpy.array(raw_data['names'])[raw_data['vis_idx']]
prop = numpy.array(raw_data['names'])[raw_data['prop_idx']]
path = '/Users/haasolivia/Documents/'+server+'/dataWork/olivia/'

fr_sr = ['FR_dist_ySum_SR_allRuns_gain_0.5', 'FR_dist_ySum_SR_allRuns_gain_1.5',
         'FR_dist_ySum_SR_rightRuns_gain_0.5', 'FR_dist_ySum_SR_rightRuns_gain_1.5',
         'FR_dist_ySum_SR_leftRuns_gain_0.5', 'FR_dist_ySum_SR_leftRuns_gain_1.5']

anova_statistic = []
anova_pvalues = []
SR_maxFR = []
maxlen = []
maxlen1 = []
max_FR = []
vis_names = []
prop_names = []
vis_idx = []
prop_idx = []

# readjust the vis / prop names to have the same ending as the .hkl names:
for v in vis:
    vis_names.append(v.split('_gain')[0])  # +'_FR_ySum_SR.hkl')
for p in prop:
    prop_names.append(p.split('_gain')[0])  # +'_FR_ySum_SR.hkl')
vis_names = numpy.array(vis_names)
prop_names = numpy.array(prop_names)

counter = 0

for name_num, name in enumerate(names):

    name1 = name.split('_PF')[0]+'_FR_ySum_SR.hkl'
    file = hickle.load(path+'hickle/FR_SR/'+name1)

    name_limits = name.split('_PF')[0] + '_PF_info.hkl'
    file_limits = hickle.load(path+'hickle/'+name_limits)

    direc = name.split('info_')[1]
    if direc == 'right':
        a = 2
        b = 3
    elif direc == 'left':
        a = 4
        b = 5
    else:
        print 'direction ', direc, ' not found!'
        sys.exit()

    limits05 = file_limits['pf_limits_'+direc+'Runs_gain_0.5']
    limits15 = file_limits['pf_limits_'+direc+'Runs_gain_1.5']

    # get maximum firing rates for all single runs for allRuns, rightRuns, leftRuns, first gain 0.5 thn 1.5
    max_fr_sr = [[],[],[],[],[],[]]  # 1, 3, 5 are gain 1.5
    for c in numpy.arange(6):
        if c%2==0:  # even numbers = 0,2,4 are gain 0.5
            limit=limits05
        else:
            limit=limits15
        sr_file = file[fr_sr[c]]
        for l in numpy.arange(len(sr_file)): # l is for each available single run
            x = sr_file[l][0]
            pf_limit_args = [signale.tools.findNearest(x, limit[0])[0],
                             signale.tools.findNearest(x, limit[1])[0]]
            try:
                # m = numpy.nanmax(sr_file[l][1])
                fr_in_pf = sr_file[l][1][pf_limit_args[0]:pf_limit_args[1]+1]
                m = numpy.nanmax(fr_in_pf) # maximum FR in place field
                # median_fr = numpy.nanmedian(abs(fr_in_pf))
            except ValueError:
                m = numpy.nan
            if not numpy.isnan(m):
                max_fr_sr[c].append(m)

    # calculate one way ANOVA for the wanted running direction (a, b are single runs for both gains for that direction)
    max05 = max_fr_sr[a]
    max15 = max_fr_sr[b]
    # print ''
    # print 'sample size for gain 0.5 and 1.5 = ', len(max05), ', ', len(max15)
    # print ''
    # statistic, pvalue = st.f_oneway(numpy.array(max05), numpy.array(max15))
    statistic, pvalue = scipy.stats.mannwhitneyu(numpy.array(max05), numpy.array(max15))

    maxfr = [max05, max15]
    l_05 = len(numpy.array(max05))
    l_15 = len(numpy.array(max15))
    maxlen1.append(l_05)
    maxlen1.append(l_15)
    maxlen.append([l_05, l_15])

    SR_maxFR.append(maxfr)
    max_FR.append([numpy.nanmax(max05), numpy.nanmax(max15)])

    # append stats for all cells
    anova_statistic.append(statistic)
    anova_pvalues.append(pvalue)

    if name in vis_names:
        vis_idx.append(name_num)
    elif name in prop_names:
        prop_idx.append(name_num)
    else:
        counter+=1
        print counter, '     ', name1, ' not in vis or prop names! Should be in confidence interval.'

for i in numpy.arange(len(names)):
    for k in [0, 1]:
        if len(numpy.array(SR_maxFR[i][k])) != numpy.max(maxlen1):
            SR_maxFR[i][k] = numpy.append(SR_maxFR[i][k], numpy.repeat(numpy.nan, numpy.max(maxlen1) -
                                                                       len(numpy.array(SR_maxFR[i][k]))))

# data = {'files':names, 'anova_statistic': anova_statistic, 'anova_pvalues': anova_pvalues, 'SR_maxFR': SR_maxFR,
#         'max_FR': max_FR, 'SR_len': maxlen, 'vis_idx': vis_idx, 'prop_idx': prop_idx}
#
# print 'Saving mannwhitneyu stats in ', path+'hickle/FR_SR/anova_stats_all_pf.hkl'
#
# hickle.dump(data, path+'hickle/FR_SR/anova_stats_all_pf.hkl', mode='w')

data = {'files':names, 'mannwhitneyu_statistic': anova_statistic, 'mannwhitneyu_pvalues': anova_pvalues, 'SR_maxFR': SR_maxFR,
        'max_FR': max_FR, 'SR_len': maxlen, 'vis_idx': vis_idx, 'prop_idx': prop_idx}

print 'Saving mannwhitneyu stats in ', path+'hickle/FR_SR/mannwhitneyu_stats_all_pf.hkl'

hickle.dump(data, path+'hickle/FR_SR/mannwhitneyu_stats_all_pf.hkl', mode='w')