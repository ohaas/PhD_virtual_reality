"""
For plotting histograms from data pooled over several cells and animals.
"""

__author__ = "Olivia Haas"
__version__ = "1.0, April 2015"

# python modules
import sys
import os

# add additional custom paths
extraPaths = [os.path.join(os.path.abspath(os.path.dirname(__file__)), '../scripts')]
for p in extraPaths:
    if not sys.path.count(p):
        sys.path.insert(1, p)


# final cell count includes all cells and their right and leftward runs: one cell which is active in both directions
#  is counted as two cells!

# other modules
import numpy
import math
import copy
import hickle
import seaborn as sns

# custom made modules
import trajectory
import custom_plot
import signale
import spikesPhase
import spikesPlace

# plotting modules
import matplotlib as mpl
#matplotlib.use('TkAgg')
import matplotlib.pyplot as pl

import csv


###################################################### functions

recalc_summary_dict = 'on'  # 'off' or 'on'

FR_center_of_mass = 'off'  # 'on' or 'off'

plot_histos = 'off'  # 'on' or 'off'. If 'on' histograms are plotted.

cloud = 'off'  # 'on' or 'off'

server = 'saw'

# xMax = 'xMaxFRySum_MaxFRySum_xCMySum_'  # 1d firing rate
xMax_twoD = 'xMaxFRinPF_MaxFRinPF_xCMinPF_'  # 2d firing rate

xcenter_binwidth = 0.218  # 0.2 meters
FR_binwidth = 0.8  # meters
width_binwidth = 0.1  # meters
SI_binwidth = 0.3  # bits
SI_binwidth_vis = 1.0  # bits

FR_thresh = 2.  # in Hz for 2d firing rates  -- 5 Hz in Ravassard/Metha Science paper
FR_thresh_2nd_gain = 1.  # threshold in Hz for 2d firing rates, if other gain fulfills FR_thresh
FR_fraction = 1.0/8  # smaller FR (for FR peaks of both gains) has to be at least FR_fraction*FR_max
pf_width_thresh = 0.05  # --> 2m*0.05 = 0.1 m = 10 cm
pf_width_max = 4.0/5  # for gain 1.33 pf sizes can be around 1.05 meters! (2m*(4./5) = 1.6 meters)
factor = 1.0/20  # 1.0/3
spike_tresh = 50  # min of spikes for each gain and running direction
spike_tresh_pf = 20  # min of spikes for each gain and running direction

FR_yBins1 = 18  # for animal 10528 and 10529: in false color plot there are 18 bins in the y direction -> normalise summed FR by bin number?
FR_yBins2 = 16  # for animal 10353

# remove hkl files wich should not be used

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

if recalc_summary_dict == 'on':
    all_titles = ['Gain normalised / According to the proprioceptive pattern', 'According to the visual pattern']
else:
    all_titles = ['According to the visual pattern', 'Gain normalised / According to the proprioceptive pattern']

summ = '/Users/haasolivia/Documents/'+server+'/dataWork/olivia/hickle/Summary/summary_dict'
addon = '_FRySum'

with open('/Users/haasolivia/Documents/'+server+'/dataWork/olivia/hickle/Summary/Cell_overview.csv', 'rb') as f:
    reader = csv.reader(f)
    cell_overview = numpy.array(list(reader))

outliers = []

if recalc_summary_dict == 'off':
    if FR_center_of_mass == 'off':
        print 'Summary dictionary is taken from '+summ+addon+'.hkl and '+summ+'_normalised'+addon+'.hkl'
        all_dicts = [hickle.load(summ+addon+'.hkl'),
                     hickle.load(summ+'_normalised'+addon+'.hkl')]
    if FR_center_of_mass == 'on':
        print 'Summary dictionary is taken from '+summ+'_cm.hkl and '+summ+'_normalised_cm.hkl'
        all_dicts = [hickle.load(summ+addon+'_cm.hkl'),
                     hickle.load(summ+'_normalised_'+addon+'_cm.hkl')]
else:

    all_dicts = []
    hkl_files = []
    path = '/Users/haasolivia/Documents/'+server+'/dataWork/olivia/'

    for f in os.listdir(path+'hickle/'):
        if f.endswith('.hkl'):
            hkl_files.append(f)

    visual_hkl_files = []
    normalised_hkl_files = []

    for p in numpy.arange(len(hkl_files)):
        if hkl_files[p].endswith('normalised.hkl'):
        # if hkl_files[p].endswith('10353_2014-06-17_VR_GCend_linTrack1_GC_TT3_SS_07_PF_info_normalised.hkl') or \
        #         hkl_files[p].endswith('10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_PF_info_normalised.hkl'):
        # if hkl_files[p].endswith('10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_PF_info_normalised.hkl') or \
        #         hkl_files[p].endswith('10529_2015-03-26_VR_linTrack2_TT3_SS_12_PF_info_normalised.hkl'):  #('normalised.hkl'): # visual cell and prop cell
            normalised_hkl_files.append(hkl_files[p])
        else:
            visual_hkl_files.append(hkl_files[p])

    os.chdir('/Users/haasolivia/Documents/'+server+'/dataWork/olivia/hickle/')

    all_files = [visual_hkl_files] #[visual_hkl_files, normalised_hkl_files]
    summary_hkls = ['summary_dict', 'summary_dict_normalised']
    used_files = []
    used_files_pp = []
    not_used_files = []
    running_directions = []
    not_used_run_direc = []

    if FR_center_of_mass == 'off':
        print 'x center change calculated based on maximal firing rate!'
    if FR_center_of_mass == 'on':
        print 'x center change calculated based on center of mass firing rate!'

    for title_num, files in enumerate(all_files):  # now only: title_num=0 and files = normalised_hkl_files

        dicts = [[], [], []]
        pf_xcenter_change_right_and_left = []
        pf_xcenter_change_visual_right_and_left = []
        pf_maxFR_change_right_and_left = []
        pf_maxFR_change_visual_right_and_left = []
        pf_width_change_right_and_left = []
        pf_width_change_visual_right_and_left = []

        maxFR_gain05 = []
        maxFR_gain05_visual = []
        maxFR_gain15 = []
        maxFR_gain15_visual = []

        x_maxFR_gain05 = []
        x_maxFR_gain05_visual = []
        x_maxFR_gain15 = []
        x_maxFR_gain15_visual = []

        pf_width_gain05 = []
        pf_width_gain05_visual = []
        pf_width_gain15 = []
        pf_width_gain15_visual = []

        delta_I = []
        delta_I_vis = []
        I_gain05 = []
        I_gain15 = []
        I_gain05_vis = []
        I_gain15_vis = []

        delta_pooled_PP_slopes = []
        delta_pooled_PP_slopes_visual = []
        Pooled_PP_slopes_gain05 = []
        Pooled_PP_slopes_gain05_visual = []
        Pooled_PP_slopes_gain15 = []
        Pooled_PP_slopes_gain15_visual = []

        SR_PP_slopes_gain05 = numpy.array([])
        SR_PP_pooled_PP_indexs_gain05 = numpy.array([])
        SR_PP_slopes_gain05_visual = numpy.array([])
        SR_PP_pooled_PP_indexs_gain05_vis = numpy.array([])
        SR_PP_slopes_gain15 = numpy.array([])
        SR_PP_pooled_PP_indexs_gain15 = numpy.array([])
        SR_PP_slopes_gain15_visual = numpy.array([])
        SR_PP_pooled_PP_indexs_gain15_vis = numpy.array([])

        RZexit_aligned_spikeTimes = []
        RZexit_aligned_spikeDistance = []
        RZexit_aligned_spikeTimes_vis = []
        RZexit_aligned_spikeDistance_vis = []

        spikeCount_PF05 = []
        spikeCount_PF15 = []
        delta_spikeCount_PF = []
        spikeCount_PF05_vis = []
        spikeCount_PF15_vis = []
        delta_spikeCount_PF_vis = []

        for d in numpy.arange(len(files)):
            print '===================================================================================================='
            print 'loading file: ', files[d]
            print ''

            dic_vis = hickle.load(files[d])

            if len(normalised_hkl_files) != len(visual_hkl_files):
                sys.exit('Not an equal amount of normalised and visual hkl files!')
            # dic_visual = hickle.load(visual_hkl_files[d])
            dic = hickle.load(normalised_hkl_files[d])

            # check whether there is theta or not for that file____________

            cell_row = numpy.where(cell_overview[:, 0] == files[d].split('PF')[0]+'PF_info.hkl')[0][0]
            theta_column = numpy.where(cell_overview[0] == 'theta')[0][0]

            if cell_overview[cell_row, theta_column] == '1':
                theta = 1
            else:
                theta = 0
            # _____________________________________________________________

            for c, trial in enumerate(['right', 'left']):

                if files[d].split('.hkl')[0]+'_'+trial in bad:
                    good = 0
                else:
                    good = 1

                if numpy.around(dic_vis[xMax_twoD+trial+'Runs_gain_0.5'][1], 1) >= FR_thresh and \
                    numpy.around(dic_vis[xMax_twoD+trial+'Runs_gain_1.5'][1], 1) >= FR_thresh_2nd_gain or \
                    numpy.around(dic_vis[xMax_twoD+trial+'Runs_gain_1.5'][1], 1) >= FR_thresh and \
                        numpy.around(dic_vis[xMax_twoD+trial+'Runs_gain_0.5'][1], 1) >= FR_thresh_2nd_gain:
                    fr = 1
                else:
                    fr = 0

                if pf_width_max*dic_vis['traj_xlim_0.5'][1] > dic_vis['pf_width_'+trial+'Runs_gain_0.5'] > \
                                        pf_width_thresh*dic_vis['traj_xlim_0.5'][1] or \
                    pf_width_max*dic_vis['traj_xlim_1.5'][1] > dic_vis['pf_width_'+trial+'Runs_gain_1.5'] > \
                                        pf_width_thresh*dic_vis['traj_xlim_1.5'][1]:
                    width = 1
                else:
                    width = 0

                # FR_min = FR_fraction*max(dic_vis[xMax_twoD+trial+'Runs_gain_0.5'][1], dic_vis[xMax_twoD+trial+'Runs_gain_1.5'][1])
                #_______________________________________________________________________________________________________

                # only plot data when these criteria are fulfilled - using 2d firing rates!

                # if dic_vis[xMax_twoD+trial+'Runs_gain_0.5'][1] >= FR_thresh and \
                #     dic_vis[xMax_twoD+trial+'Runs_gain_1.5'][1] >= FR_thresh and \
                # pf_width_max*dic_vis['traj_xlim_0.5'][1] > dic_vis['pf_width_'+trial+'Runs_gain_0.5'] > \
                #                         pf_width_thresh*dic_vis['traj_xlim_0.5'][1] and \
                # pf_width_max*dic_vis['traj_xlim_1.5'][1] > dic_vis['pf_width_'+trial+'Runs_gain_1.5'] > \
                #                     pf_width_thresh*dic_vis['traj_xlim_1.5'][1] and \

                # if fr and width and \
                #     numpy.nansum(dic_vis['spike_count_ysum_'+trial+'Runs_gain_1.5'][1]) >= spike_tresh and \
                #                 numpy.nansum(dic_vis['spike_count_ysum_'+trial+'Runs_gain_0.5'][1]) >= spike_tresh and \
                #         dic_vis['spike_count_pf_sum_'+trial+'Runs_gain_1.5'] >= spike_tresh_pf and \
                #         dic_vis['spike_count_pf_sum_'+trial+'Runs_gain_0.5'] >= spike_tresh_pf:

                if good and fr and width and \
                    numpy.nansum(dic_vis['spike_count_ysum_allRuns_gain_1.5'][1]) + \
                    numpy.nansum(dic_vis['spike_count_ysum_allRuns_gain_0.5'][1]) >= 2*spike_tresh and \
                        dic_vis['spike_count_pf_sum_'+trial+'Runs_gain_1.5'] >= spike_tresh_pf and \
                        dic_vis['spike_count_pf_sum_'+trial+'Runs_gain_0.5'] >= spike_tresh_pf:
                    #and \
                    # dic_vis[xMax_twoD+trial+'Runs_gain_0.5'][1] >= FR_min and \
                    # dic_vis[xMax_twoD+trial+'Runs_gain_1.5'][1] >= FR_min and \
                    # dic_vis[xMax_twoD+trial+'Runs_gain_0.5'][1] > factor*max(dic_vis['xMaxFR_MaxFR_xCM_allRuns_gain_0.5'][1],
                    #                                                 dic_vis['xMaxFR_MaxFR_xCM_allRuns_gain_1.5'][1]) and \
                    # dic_vis[xMax_twoD+trial+'Runs_gain_1.5'][1] > factor*max(dic_vis['xMaxFR_MaxFR_xCM_allRuns_gain_0.5'][1],
                    #                                                 dic_vis['xMaxFR_MaxFR_xCM_allRuns_gain_1.5'][1]) and \
                    # len(dic['Pooled_phases_pfLeft_pfRight_'+trial+'Runs_gain_0.5']) and \
                    # len(dic['Pooled_phases_pfLeft_pfRight_'+trial+'Runs_gain_1.5']) and \
                    # len(dic_visual['Pooled_phases_pfLeft_pfRight_'+trial+'Runs_gain_0.5']) and \
                    #     len(dic_visual['Pooled_phases_pfLeft_pfRight_'+trial+'Runs_gain_1.5']):


                    # pp_column05norm = numpy.where(cell_overview[0] == 'Pooled PP gain 0.5 '+trial)[0][0]
                    # pp_column15norm = numpy.where(cell_overview[0] == 'Pooled PP gain 1.5 '+trial)[0][0]
                    # pp_column05 = numpy.where(cell_overview[0] == 'Pooled PP not normalised 0.5 '+trial)[0][0]
                    # pp_column15 = numpy.where(cell_overview[0] == 'Pooled PP not normalised 1.5 '+trial)[0][0]
                    #
                    # if cell_overview[cell_row, pp_column05norm] == '1':
                    #     pp_05norm = 1
                    # else:
                    #     pp_05norm = 0
                    #
                    # if cell_overview[cell_row, pp_column15norm] == '1':
                    #     pp_15norm = 1
                    # else:
                    #     pp_15norm = 0
                    #
                    # if cell_overview[cell_row, pp_column05] == '1':
                    #     pp_05 = 1
                    # else:
                    #     pp_05 = 0
                    #
                    # if cell_overview[cell_row, pp_column15] == '1':
                    #     pp_15 = 1
                    # else:
                    #     pp_15 = 0

                    if FR_center_of_mass == 'off' and files == visual_hkl_files:
                        index = 0
                    elif FR_center_of_mass == 'on' and files == visual_hkl_files:
                        index = 2
                    else:
                        sys.exit("FR_center_of_mass has to be defined as 'on' or 'off' !")

                    # _______ file name and trial info _____________________________

                    dicts[c].append(dic_vis)
                    used_files.append(files[d])
                    running_directions.append(trial)

                    # _______ PF center x coordinate _____________________________

                    if trial == 'left':
                        start_1_5 = 2.0/1.5
                        start_0_5 = 2.0/0.5
                        visual_1_5 = 2.0
                        visual_0_5 = 2.0
                    else:
                        start_1_5 = 0
                        start_0_5 = 0
                        visual_1_5 = 0
                        visual_0_5 = 0

                    xcenter_gain05 = abs(dic_vis[xMax_twoD+trial+'Runs_gain_0.5'][index]/0.5-start_0_5)
                    xcenter_gain15 = abs(dic_vis[xMax_twoD+trial+'Runs_gain_1.5'][index]/1.5-start_1_5)

                    pf_xcenter_change = xcenter_gain05 - xcenter_gain15

                    xcenter_gain05_visual = abs((dic_vis[xMax_twoD+trial+'Runs_gain_0.5'][index])-visual_0_5)
                    xcenter_gain15_visual = abs((dic_vis[xMax_twoD+trial+'Runs_gain_1.5'][index])-visual_1_5)

                    pf_xcenter_change_visual = xcenter_gain05_visual - xcenter_gain15_visual

                    x_maxFR_gain05.append(xcenter_gain05)
                    x_maxFR_gain15.append(xcenter_gain15)
                    pf_xcenter_change_right_and_left.append(pf_xcenter_change)

                    x_maxFR_gain05_visual.append(xcenter_gain05_visual)
                    x_maxFR_gain15_visual.append(xcenter_gain15_visual)
                    pf_xcenter_change_visual_right_and_left.append(pf_xcenter_change_visual)

                    # _______ PF max FR __________________________________________

                    maxFR05 = dic_vis[xMax_twoD+trial+'Runs_gain_0.5'][1]
                    maxFR15 = dic_vis[xMax_twoD+trial+'Runs_gain_1.5'][1]

                    pf_maxFR_change = maxFR05 - maxFR15

                    maxFR05_vis = dic_vis[xMax_twoD+trial+'Runs_gain_0.5'][1]
                    maxFR15_vis = dic_vis[xMax_twoD+trial+'Runs_gain_1.5'][1]

                    pf_maxFR_change_visual = maxFR05_vis - maxFR15_vis

                    maxFR_gain05.append(maxFR05)
                    maxFR_gain15.append(maxFR15)
                    pf_maxFR_change_right_and_left.append(pf_maxFR_change)

                    maxFR_gain05_visual.append(maxFR05_vis)
                    maxFR_gain15_visual.append(maxFR15_vis)
                    pf_maxFR_change_visual_right_and_left.append(pf_maxFR_change_visual)

                    # if xMax_twoD+trial+'Runs_gain_0.5' in dic and xMax_twoD+trial+'Runs_gain_1.5' in dic:
                    #     pf_maxFR_change = ([dic[xMax_twoD+trial+'Runs_gain_0.5'][1] - dic[xMax_twoD+trial+'Runs_gain_1.5'][1]])[0]
                    #
                    # else:
                    #     lower_idx_05 = signale.tools.findNearest(dic[trial+'FR_2d_x_y_gain_0.5'][0],
                    #                                              dic['pf_limits_'+trial+'Runs_gain_0.5'][0])[0]
                    #     upper_idx_05 = signale.tools.findNearest(dic[trial+'FR_2d_x_y_gain_0.5'][0],
                    #                                              dic['pf_limits_'+trial+'Runs_gain_0.5'][1])[0]
                    #     lower_idx_15 = signale.tools.findNearest(dic[trial+'FR_2d_x_y_gain_1.5'][0],
                    #                                              dic['pf_limits_'+trial+'Runs_gain_0.5'][0])[0]
                    #     upper_idx_15 = signale.tools.findNearest(dic[trial+'FR_2d_x_y_gain_1.5'][0],
                    #                                              dic['pf_limits_'+trial+'Runs_gain_0.5'][1])[0]
                    #     maxFR05_array = dic[trial+'FR_2d_x_y_gain_0.5'][1][lower_idx_05:upper_idx_05+1]
                    #     maxFR15_array = dic[trial+'FR_2d_x_y_gain_1.5'][1][lower_idx_15:upper_idx_15+1]
                    #     if len(maxFR05_array):
                    #         maxFR_05 = numpy.nanmax(maxFR05_array)
                    #     else:
                    #         maxFR_05 = numpy.array([])
                    #     if len(maxFR15_array):
                    #         maxFR_15 = numpy.nanmax(maxFR15_array)
                    #     else:
                    #         maxFR_15 = numpy.array([])
                    #
                    #     pf_maxFR_change = maxFR_05 - maxFR_15



                    # if xMax_twoD+trial+'Runs_gain_0.5' in dic_visual and xMax_twoD+trial+'Runs_gain_1.5' in dic_visual:
                    #     pf_maxFR_change_visual = ([dic_visual[xMax_twoD+trial+'Runs_gain_0.5'][1] -
                    #                                dic_visual[xMax_twoD+trial+'Runs_gain_1.5'][1]])[0]
                    # else:
                    #     lower_idx_05v = signale.tools.findNearest(dic_visual[trial+'FR_2d_x_y_gain_0.5'][0],
                    #                                              dic_visual['pf_limits_'+trial+'Runs_gain_0.5'][0])[0]
                    #     upper_idx_05v = signale.tools.findNearest(dic_visual[trial+'FR_2d_x_y_gain_0.5'][0],
                    #                                              dic_visual['pf_limits_'+trial+'Runs_gain_0.5'][1])[0]
                    #     lower_idx_15v = signale.tools.findNearest(dic_visual[trial+'FR_2d_x_y_gain_1.5'][0],
                    #                                              dic_visual['pf_limits_'+trial+'Runs_gain_0.5'][0])[0]
                    #     upper_idx_15v = signale.tools.findNearest(dic_visual[trial+'FR_2d_x_y_gain_1.5'][0],
                    #                                              dic_visual['pf_limits_'+trial+'Runs_gain_0.5'][1])[0]
                    #     maxFR05_arrayv = dic_visual[trial+'FR_2d_x_y_gain_0.5'][1][lower_idx_05v:upper_idx_05v+1]
                    #     maxFR15_arrayv = dic_visual[trial+'FR_2d_x_y_gain_1.5'][1][lower_idx_15v:upper_idx_15v+1]
                    #     if len(maxFR05_arrayv):
                    #         maxFR_05v = numpy.nanmax(maxFR05_arrayv)
                    #     else:
                    #         maxFR_05v = numpy.array([])
                    #     if len(maxFR15_arrayv):
                    #         maxFR_15v = numpy.nanmax(maxFR15_arrayv)
                    #     else:
                    #         maxFR_15v = numpy.array([])
                    #
                    #     pf_maxFR_change_visual = maxFR_05v - maxFR_15v

                    # _______ PF width __________________________________________

                    width_gain05 = dic_vis['pf_width_'+trial+'Runs_gain_0.5']/0.5
                    width_gain15 = dic_vis['pf_width_'+trial+'Runs_gain_1.5']/1.5

                    pf_width_change = width_gain05 - width_gain15

                    width_gain05_visual = dic_vis['pf_width_'+trial+'Runs_gain_0.5']
                    width_gain15_visual = dic_vis['pf_width_'+trial+'Runs_gain_1.5']

                    pf_width_change_visual = width_gain05_visual - width_gain15_visual

                    pf_width_gain05.append(width_gain05)
                    pf_width_gain15.append(width_gain15)
                    pf_width_change_right_and_left.append(pf_width_change)

                    pf_width_gain05_visual.append(width_gain05_visual)
                    pf_width_gain15_visual.append(width_gain15_visual)
                    pf_width_change_visual_right_and_left.append(pf_width_change_visual)

                    # _______ Spatial information parameters_______________________

                    occupancy_prob_gain05 = dic_vis['occupancy_probability_ysum_'+trial+'Runs_gain_0.5']
                    occupancy_prob_gain15 = dic_vis['occupancy_probability_ysum_'+trial+'Runs_gain_1.5']
                    occupancy_prob_gain05_vis = dic_vis['occupancy_probability_ysum_'+trial+'Runs_gain_0.5']
                    occupancy_prob_gain15_vis = dic_vis['occupancy_probability_ysum_'+trial+'Runs_gain_1.5']

                    # cut off first and last two FR values which are due to smoothing!
                    FR_gain05 = dic_vis[trial+'FR_2d_x_y_gain_0.5'][1][2:-2]
                    FR_gain15 = dic_vis[trial+'FR_2d_x_y_gain_1.5'][1][2:-2]
                    FR_gain05_vis = dic_vis[trial+'FR_2d_x_y_gain_0.5'][1][2:-2]
                    FR_gain15_vis = dic_vis[trial+'FR_2d_x_y_gain_1.5'][1][2:-2]

                    FRfrac_gain05 = FR_gain05/numpy.nansum(occupancy_prob_gain05*FR_gain05)
                    FRfrac_gain15 = FR_gain15/numpy.nansum(occupancy_prob_gain15*FR_gain15)
                    FRfrac_gain05_vis = FR_gain05_vis/numpy.nansum(occupancy_prob_gain05_vis*FR_gain05_vis)
                    FRfrac_gain15_vis = FR_gain15_vis/numpy.nansum(occupancy_prob_gain15_vis*FR_gain15_vis)

                    log2_gain05 = numpy.log2(FRfrac_gain05)
                    log2_gain15 = numpy.log2(FRfrac_gain15)
                    log2_gain05_vis = numpy.log2(FRfrac_gain05_vis)
                    log2_gain15_vis = numpy.log2(FRfrac_gain15_vis)

                    # set all nan and infinity values for all log2 arrays to zero
                    for l in [log2_gain05, log2_gain05_vis, log2_gain15, log2_gain15_vis]:
                        l[numpy.isnan(l)] = 0
                        l[numpy.isinf(l)] = 0

                    # Calculate Spatial Information
                    Info_gain05 = numpy.nansum(occupancy_prob_gain05 * FRfrac_gain05 * log2_gain05)
                    Info_gain15 = numpy.nansum(occupancy_prob_gain15 * FRfrac_gain15 * log2_gain15)

                    delta_Info = Info_gain05 - Info_gain15

                    Info_gain05_vis = numpy.nansum(occupancy_prob_gain05_vis * FRfrac_gain05_vis * log2_gain05_vis)
                    Info_gain15_vis = numpy.nansum(occupancy_prob_gain15_vis * FRfrac_gain15_vis * log2_gain15_vis)

                    delta_Info_vis = Info_gain05_vis - Info_gain15_vis

                    # add spatial information value to list of all neuron parameters
                    I_gain05.append(Info_gain05)
                    I_gain15.append(Info_gain15)
                    delta_I.append(delta_Info)

                    I_gain05_vis.append(Info_gain05_vis)
                    I_gain15_vis.append(Info_gain15_vis)
                    delta_I_vis.append(delta_Info_vis)

                    print trial + ' run for gain 0.5 with FR Max of ' + str(dic_vis[xMax_twoD+trial+'Runs_gain_0.5'][1]) + \
                          ', a pf center of x=' + str(dic_vis[xMax_twoD+trial+'Runs_gain_0.5'][0]/0.5) + \
                          ' and a pf visual center of x=' + str(dic_vis[xMax_twoD+trial+'Runs_gain_0.5'][0])

                    print trial + ' run for gain 1.5 with FR Max of ' + str(dic_vis[xMax_twoD+trial+'Runs_gain_1.5'][1]) + \
                          ' and a pf center of x=' + str(dic_vis[xMax_twoD+trial+'Runs_gain_1.5'][0]/1.5)+ \
                          ' and a pf visual center of x=' + str(dic_vis[xMax_twoD+trial+'Runs_gain_1.5'][0])

                    print trial + ' pf_xcenter_change ' + str(pf_xcenter_change) + ' and pf_xcenter_change_visual ' + \
                          str(pf_xcenter_change_visual)

                    if pf_xcenter_change_visual < -0.5:
                        outliers.append(str(files[d]))

                    # if xMax_twoD+trial+'Runs_gain_0.5' in dic and xMax_twoD+trial+'Runs_gain_0.5' in dic_visual and \
                    #     xMax_twoD+trial+'Runs_gain_1.5' in dic and xMax_twoD+trial+'Runs_gain_1.5' in dic_visual:
                    #     maxFR_gain05.append(dic[xMax_twoD+trial+'Runs_gain_0.5'][1])
                    #     x_maxFR_gain05.append(dic[xMax_twoD+trial+'Runs_gain_0.5'][0])
                    #
                    #     maxFR_gain05_visual.append(dic_visual[xMax_twoD+trial+'Runs_gain_0.5'][1])
                    #     x_maxFR_gain05_visual.append(dic_visual[xMax_twoD+trial+'Runs_gain_0.5'][0])
                    #
                    #     maxFR_gain15.append(dic[xMax_twoD+trial+'Runs_gain_1.5'][1])
                    #     x_maxFR_gain15.append(dic[xMax_twoD+trial+'Runs_gain_1.5'][0])
                    #
                    #     maxFR_gain15_visual.append(dic_visual[xMax_twoD+trial+'Runs_gain_1.5'][1])
                    #     x_maxFR_gain15_visual.append(dic_visual[xMax_twoD+trial+'Runs_gain_1.5'][0])
                    # else:
                    #     print 'xMaxFRinPF_MaxFRinPF_xCMinPF_ DICTIONARY NOT FOUND!'
                    #     sys.exit()
                        # maxFR_gain05.append(maxFR_05)
                        # maxFR_gain05_visual.append(maxFR_05v)
                        # maxFR_gain15.append(maxFR_15)
                        # maxFR_gain15_visual.append(maxFR_15v)

                    # _______ Spike counts in place field_______________________

                    sc_05 = dic_vis['spike_count_perRun_pf_sum_'+trial+'Runs_gain_0.5']
                    sc_15 = dic_vis['spike_count_perRun_pf_sum_'+trial+'Runs_gain_1.5']
                    sc_diff = sc_05 - sc_15

                    sc_05_vis = dic_vis['spike_count_perRun_pf_sum_'+trial+'Runs_gain_0.5']
                    sc_15_vis = dic_vis['spike_count_perRun_pf_sum_'+trial+'Runs_gain_1.5']
                    sc_diff_vis = sc_05_vis - sc_15_vis

                    spikeCount_PF05.append(sc_05)
                    spikeCount_PF15.append(sc_15)
                    delta_spikeCount_PF.append(sc_diff)

                    spikeCount_PF05_vis.append(sc_05_vis)
                    spikeCount_PF15_vis.append(sc_15_vis)
                    delta_spikeCount_PF_vis.append(sc_diff_vis)

                    # _______ Getting Pooled delta in degrees per field width _____

                    if theta and len(dic_vis['Pooled_phases_pfLeft_pfRight_'+trial+'Runs_gain_0.5']) and \
                            len(dic_vis['Pooled_phases_pfLeft_pfRight_'+trial+'Runs_gain_1.5']):
                        # and \
                        # len(dic['Pooled_phases_pfLeft_pfRight_'+trial+'Runs_gain_0.5']) and \
                        #     len(dic['Pooled_phases_pfLeft_pfRight_'+trial+'Runs_gain_1.5']):

                        if theta == 0:
                            print 'there is no theta !'
                            sys.exit()
                        else:
                            used_files_pp.append(files[d])

                        if c == 0:  # rightwards run
                            sub = [1, 0]
                        else:  # leftwards run
                            sub = [0, 1]

                        # proprioceptive (gain normalised data):
                        if len(dic_vis['Pooled_phases_pfLeft_pfRight_'+trial+'Runs_gain_0.5']):  # and pp_05norm:

                            p05 = dic_vis['Pooled_phases_pfLeft_pfRight_'+trial+'Runs_gain_0.5'][0]

                            deltaPP_gain05 = p05[sub[0]] - p05[sub[1]]
                            Pooled_PP_slopes_gain05.append(deltaPP_gain05)
                            SR_p05 = numpy.array(dic_vis['SR_phases_pfLeft_pfRight_'+trial+'Runs_gain_0.5'])

                        else:
                            Pooled_PP_slopes_gain05.append(numpy.nan)
                            SR_p05 = numpy.array([])

                        if len(dic_vis['Pooled_phases_pfLeft_pfRight_'+trial+'Runs_gain_1.5']):  # and pp_15norm:

                            p15 = dic_vis['Pooled_phases_pfLeft_pfRight_'+trial+'Runs_gain_1.5'][0]

                            deltaPP_gain15 = p15[sub[0]] - p15[sub[1]]
                            Pooled_PP_slopes_gain15.append(deltaPP_gain15)
                            SR_p15 = numpy.array(dic_vis['SR_phases_pfLeft_pfRight_'+trial+'Runs_gain_1.5'])

                            if len(dic_vis['Pooled_phases_pfLeft_pfRight_'+trial+'Runs_gain_0.5']):  #and pp_05norm:
                                delta_pooled_PP_slopes.append(deltaPP_gain05 - deltaPP_gain15)
                            else:
                                delta_pooled_PP_slopes.append(numpy.nan)
                        else:
                            Pooled_PP_slopes_gain15.append(numpy.nan)
                            delta_pooled_PP_slopes.append(numpy.nan)
                            SR_p15 = numpy.array([])

                        # visual (not normalised data):
                        if len(dic_vis['Pooled_phases_pfLeft_pfRight_'+trial+'Runs_gain_0.5']):

                            p05_vis = dic_vis['Pooled_phases_pfLeft_pfRight_'+trial+'Runs_gain_0.5'][0]

                            deltaPP_gain05_vis = p05_vis[sub[0]] - p05_vis[sub[1]]
                            Pooled_PP_slopes_gain05_visual.append(deltaPP_gain05_vis)
                            SR_p05_vis = numpy.array(dic_vis['SR_phases_pfLeft_pfRight_'+trial+'Runs_gain_0.5'])

                        else:
                            Pooled_PP_slopes_gain05_visual.append(numpy.nan)
                            SR_p05_vis = numpy.array([])

                        if len(dic_vis['Pooled_phases_pfLeft_pfRight_'+trial+'Runs_gain_1.5']):

                            p15_vis = dic_vis['Pooled_phases_pfLeft_pfRight_'+trial+'Runs_gain_1.5'][0]

                            deltaPP_gain15_vis = p15_vis[sub[0]] - p15_vis[sub[1]]
                            Pooled_PP_slopes_gain15_visual.append(deltaPP_gain15_vis)
                            SR_p15_vis = numpy.array(dic_vis['SR_phases_pfLeft_pfRight_'+trial+'Runs_gain_1.5'])

                            if len(dic_vis['Pooled_phases_pfLeft_pfRight_'+trial+'Runs_gain_0.5']):
                                delta_pooled_PP_slopes_visual.append(deltaPP_gain05_vis - deltaPP_gain15_vis)
                            else:
                                delta_pooled_PP_slopes_visual.append(numpy.nan)
                        else:
                            Pooled_PP_slopes_gain15_visual.append(numpy.nan)
                            delta_pooled_PP_slopes_visual.append(numpy.nan)
                            SR_p15_vis = numpy.array([])

                        # SR_p05 = numpy.array(dic['SR_phases_pfLeft_pfRight_'+trial+'Runs_gain_0.5'])
                        # SR_p15 = numpy.array(dic['SR_phases_pfLeft_pfRight_'+trial+'Runs_gain_1.5'])
                        # SR_p05_vis = numpy.array(dic_visual['SR_phases_pfLeft_pfRight_'+trial+'Runs_gain_0.5'])
                        # SR_p15_vis = numpy.array(dic_visual['SR_phases_pfLeft_pfRight_'+trial+'Runs_gain_1.5'])

                        if len(SR_p05):
                            SR_PP_slopes_gain05 = numpy.concatenate((SR_PP_slopes_gain05,
                                                  SR_p05[:, sub[0]] - SR_p05[:, sub[1]]), axis=1)
                            SR_PP_pooled_PP_indexs_gain05 = numpy.concatenate((SR_PP_pooled_PP_indexs_gain05,
                                                  numpy.repeat(len(Pooled_PP_slopes_gain05)-1, len(SR_p05))), axis=1)
                        if len(SR_p15):
                            SR_PP_slopes_gain15 = numpy.concatenate((SR_PP_slopes_gain15,
                                                  SR_p15[:, sub[0]] - SR_p15[:, sub[1]]), axis=1)
                            SR_PP_pooled_PP_indexs_gain15 = numpy.concatenate((SR_PP_pooled_PP_indexs_gain15,
                                                  numpy.repeat(len(Pooled_PP_slopes_gain15)-1, len(SR_p15))), axis=1)
                        if len(SR_p05_vis):
                            SR_PP_slopes_gain05_visual = numpy.concatenate((SR_PP_slopes_gain05_visual,
                                                  SR_p05_vis[:, sub[0]] - SR_p05_vis[:, sub[1]]), axis=1)
                            SR_PP_pooled_PP_indexs_gain05_vis = numpy.concatenate((SR_PP_pooled_PP_indexs_gain05_vis,
                                                  numpy.repeat(len(Pooled_PP_slopes_gain05_visual)-1, len(SR_p05_vis))), axis=1)
                        if len(SR_p15_vis):
                            SR_PP_slopes_gain15_visual = numpy.concatenate((SR_PP_slopes_gain15_visual,
                                                  SR_p15_vis[:, sub[0]] - SR_p15_vis[:, sub[1]]), axis=1)
                            SR_PP_pooled_PP_indexs_gain15_vis = numpy.concatenate((SR_PP_pooled_PP_indexs_gain15_vis,
                                                  numpy.repeat(len(Pooled_PP_slopes_gain15_visual)-1, len(SR_p15_vis))), axis=1)

                    else:
                        Pooled_PP_slopes_gain05.append(numpy.nan)
                        Pooled_PP_slopes_gain15.append(numpy.nan)
                        delta_pooled_PP_slopes.append(numpy.nan)
                        Pooled_PP_slopes_gain05_visual.append(numpy.nan)
                        Pooled_PP_slopes_gain15_visual.append(numpy.nan)
                        delta_pooled_PP_slopes_visual.append(numpy.nan)

                    # ____ Getting all spike times and spike distances for one cell into one array each, without nans!

                    spikeTimes = numpy.array([])
                    spikeDistance = numpy.array([])
                    spikeTimes_vis = numpy.array([])
                    spikeDistance_vis = numpy.array([])

                    t = dic['RZ_exit_aligned_'+trial+'_spikeTimes']
                    dis = dic['RZ_exit_aligned_'+trial+'_spikeXplaces']
                    t_vis = dic_vis['RZ_exit_aligned_'+trial+'_spikeTimes']
                    dis_vis = dic_vis['RZ_exit_aligned_'+trial+'_spikeXplaces']

                    for i in numpy.arange(len(t)):
                        spikeTimes = numpy.concatenate((spikeTimes, t[i]), axis=1)
                        spikeDistance = numpy.concatenate((spikeDistance, dis[i]), axis=1)

                    for i in numpy.arange(len(t_vis)):
                        spikeTimes_vis = numpy.concatenate((spikeTimes_vis, t_vis[i]), axis=1)
                        spikeDistance_vis = numpy.concatenate((spikeDistance_vis, dis_vis[i]), axis=1)

                    spikeTimes = spikeTimes[numpy.logical_not(numpy.isnan(spikeTimes))]
                    spikeDistance = spikeDistance[numpy.logical_not(numpy.isnan(spikeDistance))]
                    spikeTimes_vis = spikeTimes_vis[numpy.logical_not(numpy.isnan(spikeTimes_vis))]
                    spikeDistance_vis = spikeDistance_vis[numpy.logical_not(numpy.isnan(spikeDistance_vis))]

                    RZexit_aligned_spikeTimes.append(spikeTimes)
                    RZexit_aligned_spikeDistance.append(spikeDistance*100)  # *100 to get distance in cm as in time cell paper!
                    RZexit_aligned_spikeTimes_vis.append(spikeTimes_vis)
                    RZexit_aligned_spikeDistance_vis.append(spikeDistance_vis*100)

                    if -0.3 < pf_xcenter_change < 0.3:
                        #print 'pf_xcenter_change for '+trial+' runs: ', pf_xcenter_change
                        print ''
                        print trial+' run normalised x center change is smaller than 0.3 meters! --> pf locks ' \
                                    'on proprioceptive stimulus?'
                        #print files[d]
                    if -0.3 < pf_xcenter_change_visual < 0.3:
                        #print 'pf_xcenter_change for '+trial+' runs: ', pf_xcenter_change
                        print ''
                        print trial+' run x center change is smaller than 0.3 meters! --> pf locks ' \
                                    'on visual stimulus?'
                        #print files[d]
                else:   #if trial == 'right' or trial == 'left':
                    print '-----------------------------------------------------------------------------------------------'
                    if dic_vis[xMax_twoD+trial+'Runs_gain_0.5'][1] < FR_thresh:
                        print 'FR threshold of '+str(FR_thresh)+' Hz was not fulfilled in file '+files[d]+' for '+trial+\
                              ' runs and gain 0.5!'
                    # if dic_vis[xMax_twoD+trial+'Runs_gain_0.5'][1] < FR_min:
                    #     print 'Minimal FR of '+str(FR_min)+' Hz was not fullfilled in file '+files[d]+' for '+trial+\
                    #           ' runs and gain 0.5!'
                    # if dic_vis[xMax_twoD+trial+'Runs_gain_1.5'][1] < FR_min:
                    #     print 'Minimal FR of '+str(FR_min)+' Hz was not fullfilled in file '+files[d]+' for '+trial+\
                    #           ' runs and gain 1.5!'
                    if dic_vis[xMax_twoD+trial+'Runs_gain_1.5'][1] < FR_thresh:
                        print 'FR threshold of '+str(FR_thresh)+' Hz was not fulfilled in file '+files[d]+' for '+trial+\
                              ' runs and gain 1.5!'
                    if dic_vis['pf_width_'+trial+'Runs_gain_0.5'] < pf_width_thresh:
                        print 'PF width was smaller than threshold of '+str(pf_width_thresh)+' meter in file '+files[d]+\
                              ' for '+trial+' runs and gain 0.5!'
                    if dic_vis['pf_width_'+trial+'Runs_gain_1.5'] < pf_width_thresh:
                        print 'PF width was smaller than threshold of '+str(pf_width_thresh)+' meter in file '+files[d]+\
                              ' for '+trial+' runs and gain 1.5!'
                    # if dic_vis[xMax_twoD+trial+'Runs_gain_0.5'][1] < factor*max(dic_vis['xMaxFR_MaxFR_xCM_allRuns_gain_0.5'][1],
                    #                                                           dic_vis['xMaxFR_MaxFR_xCM_allRuns_gain_1.5'][1]):
                    #     print str(factor)+' of allRun max FR '+str(max(dic_vis['xMaxFR_MaxFR_xCM_allRuns_gain_0.5'][1],
                    #             dic_vis['xMaxFR_MaxFR_xCM_allRuns_gain_1.5'][1]))+' Hz (=' + \
                    #             str(factor*max(dic_vis['xMaxFR_MaxFR_xCM_allRuns_gain_0.5'][1],
                    #                            dic_vis['xMaxFR_MaxFR_xCM_allRuns_gain_1.5'][1])) + \
                    #           ' Hz) not fulfilled in file '+files[d]+' for '+trial+' runs and gain 0.5!'
                    # if dic_vis[xMax_twoD+trial+'Runs_gain_1.5'][1] < factor*max(dic_vis['xMaxFR_MaxFR_xCM_allRuns_gain_0.5'][1],
                    #                                                           dic_vis['xMaxFR_MaxFR_xCM_allRuns_gain_1.5'][1]):
                    #     print str(factor)+' of allRun max FR '+str(max(dic_vis['xMaxFR_MaxFR_xCM_allRuns_gain_0.5'][1],
                    #             dic_vis['xMaxFR_MaxFR_xCM_allRuns_gain_1.5'][1]))+' Hz (=' + \
                    #             str(factor*max(dic_vis['xMaxFR_MaxFR_xCM_allRuns_gain_0.5'][1],
                    #                            dic_vis['xMaxFR_MaxFR_xCM_allRuns_gain_1.5'][1])) + \
                    #           ' Hz) not fulfilled in file '+files[d]+' for '+trial+' runs and gain 1.5!'

                    # _______ file name and trial info _____________________________

                    not_used_files.append(files[d])
                    not_used_run_direc.append(trial)

        # pad arrays within RZexit_aligned_spikeTimes and RZexit_aligned_spikeDistance with nans in order to make
        # them all the same length! Otherwise hickle can not save the data!

        lenghs = []
        lenghs_vis = []

        for x in numpy.arange(len(RZexit_aligned_spikeTimes)):
            lenghs.append(len(RZexit_aligned_spikeTimes[x]))
        for xvis in numpy.arange(len(RZexit_aligned_spikeTimes_vis)):
            lenghs_vis.append(len(RZexit_aligned_spikeTimes_vis[xvis]))

        for x in numpy.arange(len(RZexit_aligned_spikeTimes)):
            if lenghs[x] != max(lenghs):
                RZexit_aligned_spikeTimes[x] = numpy.append(RZexit_aligned_spikeTimes[x],
                                                            numpy.repeat(numpy.nan, max(lenghs) - lenghs[x]))
                RZexit_aligned_spikeDistance[x] = numpy.append(RZexit_aligned_spikeDistance[x],
                                                               numpy.repeat(numpy.nan, max(lenghs) - lenghs[x]))
        for xv in numpy.arange(len(RZexit_aligned_spikeTimes_vis)):
            if lenghs_vis[xv] != max(lenghs_vis):
                RZexit_aligned_spikeTimes_vis[xv] = numpy.append(RZexit_aligned_spikeTimes_vis[xv],
                                                                 numpy.repeat(numpy.nan, max(lenghs_vis) - lenghs_vis[xv]))
                RZexit_aligned_spikeDistance_vis[xv] = numpy.append(RZexit_aligned_spikeDistance_vis[xv],
                                                                    numpy.repeat(numpy.nan, max(lenghs_vis) - lenghs_vis[xv]))

        left_and_right_dict = \
            {'pf_center_change in m (gain 0.5 - gain 1.5 center)': pf_xcenter_change_right_and_left,  # calculated with normalised dataset
             'pf_maxFR_change in Hz (gain 0.5 - gain 1.5 FR)': pf_maxFR_change_right_and_left,
             'pf_width_change in m (gain 0.5 - gain 1.5 pf width)': pf_width_change_right_and_left,
             'pf_maxFR_gain_0.5 in Hz': maxFR_gain05,
             'pf_maxFR_gain_1.5 in Hz': maxFR_gain15,
             'pf_width_gain_0.5 in Hz': pf_width_gain05,
             'pf_width_gain_1.5 in Hz': pf_width_gain15,
             'spatial_info_gain_0.5 in bits': I_gain05,
             'spatial_info_gain_1.5 in bits': I_gain15,
             'spatial_info_change in bits (gain 0.5 - gain 1.5 pf width)': delta_I,
             'pooled_phase_precession_slopes_gain_0.5 in degree per field width': Pooled_PP_slopes_gain05,
             'pooled_phase_precession_slopes_gain_1.5 in degree per field width': Pooled_PP_slopes_gain15,
             'pooled_phase_precession_slope_change in degree per field width (gain 0.5 - gain 1.5)': delta_pooled_PP_slopes,
             'single_run_phase_precession_slopes_gain_0.5 in degree per field width': SR_PP_slopes_gain05,
             'single_run_pp_pooled_indexes_gain_0.5': SR_PP_pooled_PP_indexs_gain05,
             'single_run_phase_precession_slopes_gain_1.5 in degree per field width': SR_PP_slopes_gain15,
             'single_run_pp_pooled_indexes_gain_1.5': SR_PP_pooled_PP_indexs_gain15,
             'RZ_exit_aligned_spike_times in s': RZexit_aligned_spikeTimes,
             'RZ_exit_aligned_spike_distances in cm': RZexit_aligned_spikeDistance,
             'pf_spike_count_change': delta_spikeCount_PF,
             'pf_spike_count_0.5': spikeCount_PF05,
             'pf_spike_count_1.5': spikeCount_PF15,
             'x_pf_maxFR_gain_0.5 in m': x_maxFR_gain05,
             'x_pf_maxFR_gain_1.5 in m': x_maxFR_gain15}
        left_and_right_dict_visual = \
            {'pf_center_change in m (gain 0.5 - gain 1.5 center)': pf_xcenter_change_visual_right_and_left,
             'pf_maxFR_change in Hz (gain 0.5 - gain 1.5 FR)': pf_maxFR_change_visual_right_and_left,
             'pf_width_change in m (gain 0.5 - gain 1.5 pf width)': pf_width_change_visual_right_and_left,
             'pf_maxFR_gain_0.5 in Hz': maxFR_gain05_visual,
             'pf_maxFR_gain_1.5 in Hz': maxFR_gain15_visual,
             'pf_width_gain_0.5 in Hz': pf_width_gain05_visual,
             'pf_width_gain_1.5 in Hz': pf_width_gain15_visual,
             'spatial_info_gain_0.5 in bits': I_gain05_vis,
             'spatial_info_gain_1.5 in bits': I_gain15_vis,
             'spatial_info_change in bits (gain 0.5 - gain 1.5 pf width)': delta_I_vis,
             'pooled_phase_precession_slopes_gain_0.5 in degree per field width': Pooled_PP_slopes_gain05_visual,
             'pooled_phase_precession_slopes_gain_1.5 in degree per field width': Pooled_PP_slopes_gain15_visual,
             'pooled_phase_precession_slope_change in degree per field width (gain 0.5 - gain 1.5)': delta_pooled_PP_slopes_visual,
             'single_run_phase_precession_slopes_gain_0.5 in degree per field width': SR_PP_slopes_gain05_visual,
             'single_run_pp_pooled_indexes_gain_0.5': SR_PP_pooled_PP_indexs_gain05_vis,
             'single_run_phase_precession_slopes_gain_1.5 in degree per field width': SR_PP_slopes_gain15_visual,
             'single_run_pp_pooled_indexes_gain_1.5': SR_PP_pooled_PP_indexs_gain15_vis,
             'RZ_exit_aligned_spike_times in s': RZexit_aligned_spikeTimes_vis,
             'RZ_exit_aligned_spike_distances in cm': RZexit_aligned_spikeDistance_vis,
             'pf_spike_count_change': delta_spikeCount_PF_vis,
             'pf_spike_count_0.5': spikeCount_PF05_vis,
             'pf_spike_count_1.5': spikeCount_PF15_vis,
             'x_pf_maxFR_gain_0.5 in m': x_maxFR_gain05_visual,
             'x_pf_maxFR_gain_1.5 in m': x_maxFR_gain15_visual}
        all_dicts.append(left_and_right_dict)
        all_dicts.append(left_and_right_dict_visual)

        if FR_center_of_mass == 'off':
            if xMax_twoD == 'xMaxFRySum_MaxFRySum_xCMySum_':
                hickle.dump(left_and_right_dict, path+'hickle/Summary/'+summary_hkls[1]+'_FRySum.hkl', mode='w')
                hickle.dump(left_and_right_dict_visual, path+'hickle/Summary/'+summary_hkls[0]+'_FRySum.hkl', mode='w')
            else:
                hickle.dump(left_and_right_dict, path+'hickle/Summary/'+summary_hkls[1]+'.hkl', mode='w')
                hickle.dump(left_and_right_dict_visual, path+'hickle/Summary/'+summary_hkls[0]+'.hkl', mode='w')

        if FR_center_of_mass == 'on':
            if xMax_twoD == 'xMaxFRySum_MaxFRySum_xCMySum_':
                hickle.dump(left_and_right_dict, path+'hickle/Summary/'+summary_hkls[1]+'_FRySum_cm.hkl', mode='w')
                hickle.dump(left_and_right_dict_visual, path+'hickle/Summary/'+summary_hkls[0]+'_FRySum_cm.hkl', mode='w')
            else:
                hickle.dump(left_and_right_dict, path+'hickle/Summary/'+summary_hkls[1]+'_cm.hkl', mode='w')
                hickle.dump(left_and_right_dict_visual, path+'hickle/Summary/'+summary_hkls[0]+'_cm.hkl', mode='w')

        hickle.dump(all_files, path+'hickle/Summary/all_filenames.hkl', mode='w')
        hickle.dump(used_files, path+'hickle/Summary/used_filenames.hkl', mode='w')
        hickle.dump(used_files_pp, path+'hickle/Summary/used_filenames_pp.hkl', mode='w')
        hickle.dump(running_directions, path+'hickle/Summary/running_directions.hkl', mode='w')
        hickle.dump(not_used_files, path+'hickle/Summary/not_used_filenames.hkl', mode='w')
        hickle.dump(not_used_run_direc, path+'hickle/Summary/not_used_run_directions.hkl', mode='w')

# plot histograms

if plot_histos == 'on':

    if FR_center_of_mass == 'off':
        if xMax_twoD == 'xMaxFRySum_MaxFRySum_xCMySum_':
            if recalc_summary_dict == 'on':
                fig_names = ['Gain_normalised_FRySum', 'Non_gain_normalised_FRySum']
            else:
                fig_names = ['Non_gain_normalised_FRySum', 'Gain_normalised_FRySum']
        else:
            if recalc_summary_dict == 'on':
                fig_names = ['Gain_normalised', 'Non_gain_normalised']
            else:
                fig_names = ['Non_gain_normalised', 'Gain_normalised']
    if FR_center_of_mass == 'on':
        if xMax_twoD == 'xMaxFRySum_MaxFRySum_xCMySum_':
            if recalc_summary_dict == 'on':
                fig_names = ['Gain_normalised_FRySum_cm', 'Non_gain_normalised_FRySum_cm']
            else:
                fig_names = ['Non_gain_normalised_FRySum_cm', 'Gain_normalised_FRySum_cm']
        else:
            if recalc_summary_dict == 'on':
                fig_names = ['Gain_normalised_cm', 'Non_gain_normalised_cm']
            else:
                fig_names = ['Non_gain_normalised_cm', 'Gain_normalised_cm']

    binwidth = [[xcenter_binwidth, FR_binwidth, FR_binwidth, width_binwidth, width_binwidth, SI_binwidth_vis, SI_binwidth_vis],
                [xcenter_binwidth, FR_binwidth, FR_binwidth, width_binwidth, width_binwidth, SI_binwidth, SI_binwidth]]

    if cloud == 'off':
        for i in numpy.arange(len(all_dicts)):

            fig = pl.figure(figsize=(25, 10))

            for s, k in enumerate(numpy.array([all_dicts[0].keys()[8], all_dicts[0].keys()[9], all_dicts[0].keys()[6],
                                               all_dicts[0].keys()[2], all_dicts[0].keys()[5], all_dicts[0].keys()[1],
                                               all_dicts[0].keys()[7]])):
                pl.subplot(1, 7, s+1)
                if s == 4:
                    pl.xlabel('pf_width for gains in m \n gain 0.5 (orange) and 1.5 (green)')
                elif s == 2:
                    pl.xlabel('ySum maxFR for gains in Hz \n gain 0.5 (orange) and 1.5 (green)')
                elif s == 6:
                    pl.xlabel('Spatial Info for gains in bits \n gain 0.5 (orange) and 1.5 (green)')
                else:
                    pl.xlabel(k.split('(')[0]+'\n ('+k.split('(')[1])
                if s == 0:
                    pl.ylabel('count')
                    pl.suptitle(all_titles[i]+' for '+str(len(all_dicts[i][k]))+
                                ' left/right trails out of a total of '+str(2*len(all_dicts[0][all_dicts[0].keys()[0]]))+
                                ' (for '+str(len(all_dicts[0][all_dicts[0].keys()[0]]))+' cells)')
                    fig.text(.995, 0.02, 'FR_thresh = '+str(FR_thresh)+' in Hz, FR factor of all runs FR max = '+str(factor)+
                             ', pf_width_thresh = '+str(pf_width_thresh)+' in m, pf_width_factor for traj max = '+str(pf_width_max),
                             fontsize=8,  horizontalalignment='right')
                    fig.text(.995, 0.005, 'xcenter_binwidth = '+str(xcenter_binwidth)+'m, FR_binwidth = '+str(FR_binwidth)+
                             'Hz, width_binwidth = '+str(width_binwidth)+'m', fontsize=8,  horizontalalignment='right')
                print ''
                print 'plotting histogram for ' + str(all_titles[i]) + ' with ' + str(k) + '=' + str(all_dicts[i][k])

                if s == 2 or s == 4 or s == 6:
                    if s == 2:
                        d = all_dicts[0].keys()[0]
                    elif s == 4:
                        d = all_dicts[0].keys()[3]
                    else:
                        d = all_dicts[0].keys()[4]
                    pl.hist(all_dicts[i][k], bins=numpy.arange(min(all_dicts[i][k]), max(all_dicts[i][k]) +
                                                                      binwidth[i][s], binwidth[i][s]), color=custom_plot.pretty_colors_set2[0])
                    pl.hist(all_dicts[i][d], bins=numpy.arange(min(all_dicts[i][d]), max(all_dicts[i][d]) +
                                                                      binwidth[i][s], binwidth[i][s]), color=custom_plot.pretty_colors_set2[1])
                else:
                    pl.hist(all_dicts[i][k], bins=numpy.arange(min(all_dicts[i][k]), max(all_dicts[i][k]) +
                                                                      binwidth[i][s], binwidth[i][s]), color=custom_plot.pretty_colors_set2[s+2])

            print ''
            print ''
            print 'Saving figure under: /Users/haasolivia/Documents/'+server+'/dataWork/olivia/hickle/Summary/'+fig_names[i]+'_summary.pdf'
            fig.savefig('/Users/haasolivia/Documents/'+server+'/dataWork/olivia/hickle/Summary/'+fig_names[i]+'_summary.pdf', format='pdf')

    else:
        fig = pl.figure(figsize=(20, 10))
        a = numpy.array([all_dicts[0].keys()[1], all_dicts[0].keys()[0], all_dicts[0].keys()[2]])
        for s, k in enumerate(a):
            pl.subplot(1, 3, s+1)
            pl.xlabel(k)
            pl.ylabel('count')
            pl.suptitle(all_titles[1]+' for '+str(len(all_dicts[1][a[0]]))+
                ' left/right trails out of a total of '+str(2*len(all_dicts[0][all_dicts[0].keys()[0]]))+
                ' (for '+str(len(all_dicts[0][all_dicts[0].keys()[0]]))+' cells)')
            if s == 0:
                y = numpy.asarray(all_dicts[1][a[0]])  # [k]
            else:
                y = all_dicts[1][k]

            x = numpy.asarray(all_dicts[1][a[0]])

            with sns.axes_style("white"):
                sns.jointplot(x, y, kind="hex")  # all_dicts[1] are the gain normalised ones

    pl.show()
    #pl.close('all')

os.chdir('/Users/haasolivia/Documents/Analysis_Scripts/ephys/')

