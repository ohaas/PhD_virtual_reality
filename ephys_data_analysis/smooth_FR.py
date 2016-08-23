__author__ = 'haasolivia'

import sys
import os

pos = f['xMaxFRySuminPF_MaxFRySuminPF_xCMySuminPF_'+data_rundirec[i]+'Runs_gain_'+gain][0]

extraPaths = [os.path.join(os.path.abspath(os.path.dirname(__file__)), '../scripts')]
for p in extraPaths:
    if not sys.path.count(p):
        sys.path.insert(1, p)

import numpy
import hickle
from itertools import repeat
import seaborn as sns
import matplotlib.pyplot as pl
import custom_plot
from scipy import stats
import signale

where = '/Users/haasolivia/Desktop/plots/bsp/'

server = 'saw'
hick = '/Users/haasolivia/Documents/'+server+'/dataWork/olivia/hickle/'
# hick = '/Users/haasolivia/Documents/'+server+'/dataWork/olivia/hickle_secondary/visual_and_prop/'

#

# doppelz1 = ['10823_2015-06-29_VR_GCend_linTrack1_TT3_SS_18_PF_info.hkl', '10823_2015-06-29_VR_GCend_linTrack1_TT3_SS_18_PF_info.hkl']
# doppelz2 = ['10823_2015-07-03_VR_GCendOL_linTrack1_TT3_SS_18_PF_info.hkl', '10823_2015-07-03_VR_GCendOL_linTrack1_TT3_SS_18_PF_info.hkl']
# doppelz3 = ['10823_2015-08-17_VR_GCend_linTrack1_TT2_SS_08_PF_info.hkl', '10823_2015-08-17_VR_GCend_linTrack1_TT2_SS_08_PF_info.hkl']
# doppelz4 = ['10823_2015-08-12_VR_GCend_linTrack1_TT4_SS_07_PF_info.hkl', '10823_2015-08-12_VR_GCend_linTrack1_TT4_SS_07_PF_info.hkl']
doppelz5 = ['10823_2015-06-30_VR_GCend_linTrack1_TT4_SS_14_PF_info.hkl', '10823_2015-06-30_VR_GCend_linTrack1_TT4_SS_14_PF_info.hkl']
# doppelz6 = ['10823_2015-07-20_VR_GCendDark_linTrack1_TT3_SS_09_PF_info.hkl', '10823_2015-07-20_VR_GCendDark_linTrack1_TT3_SS_09_PF_info.hkl']
doppelz7 = ['10823_2015-06-30_VR_GCend_linTrack1_TT2_SS_14_PF_info.hkl', '10823_2015-06-30_VR_GCend_linTrack1_TT2_SS_14_PF_info.hkl']

# single - bi

# doppelz1 = ['10528_2015-04-21_GCend_Dark_linTrack1_TT2_SS_06_PF_info.hkl', '10528_2015-04-21_GCend_Dark_linTrack1_TT2_SS_06_PF_info.hkl']
# doppelz2 = ['10823_2015-07-24_VR_GCendDark_linTrack1_TT2_SS_11_PF_info.hkl', '10823_2015-07-24_VR_GCendDark_linTrack1_TT2_SS_11_PF_info.hkl']
# doppelz3 = ['10823_2015-08-03_VR_GCend_linTrack1_TT4_SS_01_PF_info.hkl', '10823_2015-08-03_VR_GCend_linTrack1_TT4_SS_01_PF_info.hkl']
# doppelz4 = ['10823_2015-08-26_VR_GCend_nami_linTrack1_TT2_SS_07_PF_info.hkl', '10823_2015-08-26_VR_GCend_nami_linTrack1_TT2_SS_07_PF_info.hkl']
# doppelz5 = ['10823_2015-08-27_VR_GCend_linTrack1_TT1_SS_03_PF_info.hkl', '10823_2015-08-27_VR_GCend_linTrack1_TT1_SS_03_PF_info.hkl']

# doppelz1 = ['10353_2014-06-16_VR_GCend_linTrack1_GC_TT4_SS_02_PF_info.hkl', '10353_2014-06-16_VR_GCend_linTrack1_GC_TT4_SS_02_PF_info.hkl']
# doppelz2 = ['10353_2014-06-18_VR_GCmid_linTrack1_GC_a_TT4_SS_08_PF_info.hkl', '10353_2014-06-18_VR_GCmid_linTrack1_GC_a_TT4_SS_08_PF_info.hkl']
# doppelz3 = ['10353_2014-07-02_VR_GCend_linTrack1_GC_TT1_SS_01_PF_info.hkl', '10353_2014-07-02_VR_GCend_linTrack1_GC_TT1_SS_01_PF_info.hkl']
# doppelz4 = ['10535_2015-10-06_VR_GCend_linTrack1_TT4_SS_12_PF_info.hkl', '10535_2015-10-06_VR_GCend_linTrack1_TT4_SS_12_PF_info.hkl']
# doppelz5 = ['10537_2015-10-16_VR_GCend_linTrack1_TT1_SS_01_PF_info.hkl', '10537_2015-10-16_VR_GCend_linTrack1_TT1_SS_01_PF_info.hkl']
# doppelz6 = ['10823_2015-07-13_VR_GCend_linTrack1_TT3_SS_11_PF_info.hkl', '10823_2015-07-13_VR_GCend_linTrack1_TT3_SS_11_PF_info.hkl']
# doppelz7 = ['10823_2015-08-27_VR_GCend_linTrack1_TT1_SS_03_PF_info.hkl', '10823_2015-08-27_VR_GCend_linTrack1_TT1_SS_03_PF_info.hkl']
# doppelz8 = ['10823_2015-08-27_VR_GCend_linTrack1_TT4_SS_05_PF_info.hkl', '10823_2015-08-27_VR_GCend_linTrack1_TT4_SS_05_PF_info.hkl']

# doppelz1 = ['10823_2015-08-27_VR_GCend_linTrack1_TT1_SS_03_PF_info.hkl', '10823_2015-08-27_VR_GCend_linTrack1_TT1_SS_03_PF_info.hkl']

# rausmapping

# doppelz1 = ['10528_2015-04-04_VR_GCend_Dark_linTrack1_TT3_SS_04_PF_info.hkl', '10528_2015-04-04_VR_GCend_Dark_linTrack1_TT3_SS_04_PF_info.hkl']
# doppelz2 = ['10528_2015-04-13_VRnami_GCend_Dark_linTrack1_TT1_SS_02_PF_info.hkl', '10528_2015-04-13_VRnami_GCend_Dark_linTrack1_TT1_SS_02_PF_info.hkl']
# doppelz3 = ['10528_2015-04-18_VR_GCend_Dark_linTrack1_TT2_SS_24_PF_info.hkl', '10528_2015-04-18_VR_GCend_Dark_linTrack1_TT2_SS_24_PF_info.hkl']
# doppelz4 = ['10528_2015-04-18_VR_GCend_Dark_linTrack1_TT4_SS_10_PF_info.hkl', '10528_2015-04-18_VR_GCend_Dark_linTrack1_TT4_SS_10_PF_info.hkl']
# doppelz5 = ['10528_2015-04-19_VR_GCend_linTrack1_dark_TT2_SS_04_PF_info.hkl', '10528_2015-04-19_VR_GCend_linTrack1_dark_TT2_SS_04_PF_info.hkl']
doppelz6 = ['10528_2015-04-13_VR_GCend_ol_linTrack1_TT1_SS_07_PF_info.hkl', '10528_2015-04-13_VR_GCend_ol_linTrack1_TT1_SS_07_PF_info.hkl']
# doppelz7 = ['10537_2015-10-20_VR_GCend_linTrack1_TT8_SS_08_PF_info.hkl', '10537_2015-10-20_VR_GCend_linTrack1_TT8_SS_08_PF_info.hkl']
# doppelz8 = ['10823_2015-08-19_VR_GCend_linTrack1_TT3_SS_07_PF_info.hkl', '10823_2015-08-19_VR_GCend_linTrack1_TT3_SS_07_PF_info.hkl']
# doppelz9 = ['10823_2015-08-18_VR_GCend_nami_linTrack1_TT2_SS_13_PF_info.hkl', '10823_2015-08-18_VR_GCend_nami_linTrack1_TT2_SS_13_PF_info.hkl']
# doppelz10 = ['10823_2015-08-19_VR_GCend_linTrack1_TT3_SS_06_PF_info.hkl', '10823_2015-08-19_VR_GCend_linTrack1_TT3_SS_06_PF_info.hkl']

# single - raus

# doppelz1 = ['10353_2014-08-07_VR_GCend_linTrack1_GC_TT1_SS_02_PF_info.hkl', '10353_2014-08-07_VR_GCend_linTrack1_GC_TT1_SS_02_PF_info.hkl']
# doppelz2 = ['10528_2015-03-13_VR_GCend_linTrack1_TT2_SS_03_PF_info.hkl', '10528_2015-03-13_VR_GCend_linTrack1_TT2_SS_03_PF_info.hkl']
# doppelz3 = ['10528_2015-04-15_VR_GCend_ol_linTrack1_TT2_SS_10_PF_info.hkl', '10528_2015-04-15_VR_GCend_ol_linTrack1_TT2_SS_10_PF_info.hkl']
# doppelz4 = ['10528_2015-04-21_GCend_Dark_linTrack1_TT2_SS_06_PF_info.hkl', '10528_2015-04-21_GCend_Dark_linTrack1_TT2_SS_06_PF_info.hkl']
# doppelz5 = ['10529_2015-03-25_VR_nami_linTrack2_TT5_SS_03_PF_info.hkl', '10529_2015-03-25_VR_nami_linTrack2_TT5_SS_03_PF_info.hkl']
# doppelz6 = ['10529_2015-03-27_VR_linTrack1_TT8_SS_14_PF_info.hkl', '10529_2015-03-27_VR_linTrack1_TT8_SS_14_PF_info.hkl']
# doppelz7 = ['10535_2015-10-06_VR_GCend_linTrack1_TT4_SS_07_PF_info.hkl', '10535_2015-10-06_VR_GCend_linTrack1_TT4_SS_07_PF_info.hkl']
# doppelz8 = ['10537_2015-10-07_VR_GCend_linTrack1_TT8_SS_13_PF_info.hkl', '10537_2015-10-07_VR_GCend_linTrack1_TT8_SS_13_PF_info.hkl']
# doppelz9 = ['10537_2015-10-16_VR_GCend_linTrack1_TT6_SS_08_PF_info.hkl', '10537_2015-10-16_VR_GCend_linTrack1_TT6_SS_08_PF_info.hkl']
# doppelz10 = ['10537_2015-10-22_VR_GCend_linTrack1_TT1_SS_09_PF_info.hkl', '10537_2015-10-22_VR_GCend_linTrack1_TT1_SS_09_PF_info.hkl']
# doppelz11 = ['10823_2015-07-03_VR_GCend_linTrack1_TT3_SS_18_PF_info.hkl', '10823_2015-07-03_VR_GCend_linTrack1_TT3_SS_18_PF_info.hkl']
# doppelz12 = ['10823_2015-07-24_VR_GCendDark_linTrack1_TT2_SS_11_PF_info.hkl', '10823_2015-07-24_VR_GCendDark_linTrack1_TT2_SS_11_PF_info.hkl']
# doppelz13 = ['10823_2015-08-03_VR_GCend_linTrack1_TT4_SS_01_PF_info.hkl', '10823_2015-08-03_VR_GCend_linTrack1_TT4_SS_01_PF_info.hkl']
# doppelz14 = ['10823_2015-08-17_VR_GCend_linTrack1_TT2_SS_16_PF_info.hkl', '10823_2015-08-17_VR_GCend_linTrack1_TT2_SS_16_PF_info.hkl']

# ----------------------------------------------------------------------

# doppel1 = [hickle.load(hick+doppelz1[0]), hickle.load(hick+doppelz1[1])]
# doppel2 = [hickle.load(hick+doppelz2[0]), hickle.load(hick+doppelz2[1])]
# doppel3 = [hickle.load(hick+doppelz3[0]), hickle.load(hick+doppelz3[1])]
# doppel4 = [hickle.load(hick+doppelz4[0]), hickle.load(hick+doppelz4[1])]
doppel5 = [hickle.load(hick+doppelz5[0]), hickle.load(hick+doppelz5[1])]
doppel6 = [hickle.load(hick+'/cells_not_used_52_16single1gain_3double1gain/'+doppelz6[0]), hickle.load(hick+'/cells_not_used_52_16single1gain_3double1gain/'+doppelz6[1])]
doppel7 = [hickle.load(hick+doppelz7[0]), hickle.load(hick+doppelz7[1])]
# doppel8 = [hickle.load(hick+doppelz8[0]), hickle.load(hick+doppelz8[1])]
# doppel9 = [hickle.load(hick+doppelz9[0]), hickle.load(hick+doppelz9[1])]
# doppel10 = [hickle.load(hick+doppelz10[0]), hickle.load(hick+doppelz10[1])]
# doppel11 = [hickle.load(hick+doppelz11[0]), hickle.load(hick+doppelz11[1])]
# doppel12 = [hickle.load(hick+doppelz12[0]), hickle.load(hick+doppelz12[1])]
# doppel13 = [hickle.load(hick+doppelz13[0]), hickle.load(hick+doppelz13[1])]
# doppel14 = [hickle.load(hick+doppelz14[0]), hickle.load(hick+doppelz14[1])]

# ----------------------------------------------------------------------

# cells = [doppel1, doppel2, doppel3, doppel4, doppel5, doppel5, doppel6, doppel6, doppel7, doppel7]
# run_direc = ['left', 'right', 'left', 'right', 'left', 'right', 'left', 'right', 'left', 'right']
#

# single - bi

# cells = [doppel1, doppel1, doppel2, doppel2, doppel3, doppel3, doppel4, doppel4, doppel5, doppel5]
# run_direc = ['right', 'left', 'right', 'left', 'right', 'left', 'right', 'left', 'right', 'left']

# cells = [doppel1, doppel1, doppel2, doppel2, doppel3, doppel3, doppel4, doppel4, doppel5, doppel5, doppel6, doppel6,
#          doppel7, doppel7, doppel8, doppel8]
# run_direc = ['right', 'left', 'right', 'left', 'right', 'left', 'right', 'left', 'right', 'left', 'right', 'left',
#              'right', 'left', 'right', 'left']
cells = [doppel5, doppel5, doppel6, doppel6, doppel7, doppel7]
run_direc = ['right', 'left', 'right', 'left', 'right', 'left']

# rausmapping

# cells = [doppel1, doppel2, doppel3, doppel4, doppel5, doppel6, doppel6, doppel7, doppel7, doppel8, doppel9, doppel10]
# run_direc = ['right', 'right', 'right', 'right', 'right', 'left', 'right', 'left', 'right', 'left', 'right', 'right']

# single - raus

# cells = [doppel1, doppel1, doppel2, doppel2, doppel3, doppel3, doppel4, doppel4, doppel5, doppel5, doppel6, doppel6,
#          doppel7, doppel7, doppel8, doppel8, doppel9, doppel9, doppel10, doppel10, doppel11, doppel11, doppel12,
#          doppel12, doppel13, doppel13, doppel14, doppel14]
# run_direc = ['right', 'left', 'right', 'left', 'right', 'left', 'right', 'left', 'right', 'left', 'right', 'left',
#              'right', 'left', 'right', 'left', 'right', 'left', 'right', 'left', 'right', 'left', 'right', 'left',
#              'right', 'left', 'right', 'left']

# ----------------------------------------------------------------------

# fig_names = [[doppelz1[0].split('PF')[0]+run_direc[0], doppelz1[0].split('PF')[0]+'norm_'+run_direc[0]],
#              [doppelz2[0].split('PF')[0]+run_direc[1], doppelz2[0].split('PF')[0]+'norm_'+run_direc[1]],
#              [doppelz3[0].split('PF')[0]+run_direc[2], doppelz3[0].split('PF')[0]+'norm_'+run_direc[2]],
#              [doppelz4[0].split('PF')[0]+run_direc[3], doppelz4[0].split('PF')[0]+'norm_'+run_direc[3]],
#              [doppelz5[0].split('PF')[0]+run_direc[4], doppelz5[0].split('PF')[0]+'norm_'+run_direc[4]],
#              [doppelz5[0].split('PF')[0]+run_direc[5], doppelz5[0].split('PF')[0]+'norm_'+run_direc[5]],
#              [doppelz6[0].split('PF')[0]+run_direc[6], doppelz6[0].split('PF')[0]+'norm_'+run_direc[6]],
#              [doppelz6[0].split('PF')[0]+run_direc[7], doppelz6[0].split('PF')[0]+'norm_'+run_direc[7]],
#              [doppelz7[0].split('PF')[0]+run_direc[8], doppelz7[0].split('PF')[0]+'norm_'+run_direc[8]],
#              [doppelz7[0].split('PF')[0]+run_direc[9], doppelz7[0].split('PF')[0]+'norm_'+run_direc[9]]]

# single - bi

# fig_names = [[doppelz1[0].split('PF')[0]+run_direc[0], doppelz1[0].split('PF')[0]+'norm_'+run_direc[0]],
#              [doppelz1[0].split('PF')[0]+run_direc[1], doppelz1[0].split('PF')[0]+'norm_'+run_direc[1]],
#              [doppelz2[0].split('PF')[0]+run_direc[2], doppelz2[0].split('PF')[0]+'norm_'+run_direc[2]],
#              [doppelz2[0].split('PF')[0]+run_direc[3], doppelz2[0].split('PF')[0]+'norm_'+run_direc[3]],
#              [doppelz3[0].split('PF')[0]+run_direc[4], doppelz3[0].split('PF')[0]+'norm_'+run_direc[4]],
#              [doppelz3[0].split('PF')[0]+run_direc[5], doppelz3[0].split('PF')[0]+'norm_'+run_direc[5]],
#              [doppelz4[0].split('PF')[0]+run_direc[6], doppelz4[0].split('PF')[0]+'norm_'+run_direc[6]],
#              [doppelz4[0].split('PF')[0]+run_direc[7], doppelz4[0].split('PF')[0]+'norm_'+run_direc[7]],
#              [doppelz5[0].split('PF')[0]+run_direc[8], doppelz5[0].split('PF')[0]+'norm_'+run_direc[8]],
#              [doppelz5[0].split('PF')[0]+run_direc[9], doppelz5[0].split('PF')[0]+'norm_'+run_direc[9]],
#              [doppelz6[0].split('PF')[0]+run_direc[10], doppelz6[0].split('PF')[0]+'norm_'+run_direc[10]],
#              [doppelz6[0].split('PF')[0]+run_direc[11], doppelz6[0].split('PF')[0]+'norm_'+run_direc[11]],
#              [doppelz7[0].split('PF')[0]+run_direc[12], doppelz7[0].split('PF')[0]+'norm_'+run_direc[12]],
#              [doppelz7[0].split('PF')[0]+run_direc[13], doppelz7[0].split('PF')[0]+'norm_'+run_direc[13]],
#              [doppelz8[0].split('PF')[0]+run_direc[14], doppelz8[0].split('PF')[0]+'norm_'+run_direc[14]],
#              [doppelz8[0].split('PF')[0]+run_direc[15], doppelz8[0].split('PF')[0]+'norm_'+run_direc[15]]]

fig_names = [[doppelz5[0].split('PF')[0]+run_direc[0], doppelz5[0].split('PF')[0]+'norm_'+run_direc[0]],
             [doppelz5[0].split('PF')[0]+run_direc[1], doppelz5[0].split('PF')[0]+'norm_'+run_direc[1]],
             [doppelz6[0].split('PF')[0]+run_direc[2], doppelz6[0].split('PF')[0]+'norm_'+run_direc[2]],
             [doppelz6[0].split('PF')[0]+run_direc[3], doppelz6[0].split('PF')[0]+'norm_'+run_direc[3]],
             [doppelz7[0].split('PF')[0]+run_direc[4], doppelz7[0].split('PF')[0]+'norm_'+run_direc[4]],
             [doppelz7[0].split('PF')[0]+run_direc[5], doppelz7[0].split('PF')[0]+'norm_'+run_direc[5]]]

# rausmapping

# fig_names = [[doppelz1[0].split('PF')[0]+run_direc[0], doppelz1[0].split('PF')[0]+'norm_'+run_direc[0]],
#              [doppelz2[0].split('PF')[0]+run_direc[1], doppelz2[0].split('PF')[0]+'norm_'+run_direc[1]],
#              [doppelz3[0].split('PF')[0]+run_direc[2], doppelz3[0].split('PF')[0]+'norm_'+run_direc[2]],
#              [doppelz4[0].split('PF')[0]+run_direc[3], doppelz4[0].split('PF')[0]+'norm_'+run_direc[3]],
#              [doppelz5[0].split('PF')[0]+run_direc[4], doppelz5[0].split('PF')[0]+'norm_'+run_direc[4]],
#              [doppelz6[0].split('PF')[0]+run_direc[5], doppelz6[0].split('PF')[0]+'norm_'+run_direc[5]],
#              [doppelz6[0].split('PF')[0]+run_direc[6], doppelz6[0].split('PF')[0]+'norm_'+run_direc[6]],
#              [doppelz7[0].split('PF')[0]+run_direc[7], doppelz7[0].split('PF')[0]+'norm_'+run_direc[7]],
#              [doppelz7[0].split('PF')[0]+run_direc[8], doppelz7[0].split('PF')[0]+'norm_'+run_direc[8]],
#              [doppelz8[0].split('PF')[0]+run_direc[9], doppelz8[0].split('PF')[0]+'norm_'+run_direc[9]],
#              [doppelz9[0].split('PF')[0]+run_direc[10], doppelz9[0].split('PF')[0]+'norm_'+run_direc[10]],
#              [doppelz10[0].split('PF')[0]+run_direc[11], doppelz10[0].split('PF')[0]+'norm_'+run_direc[11]]]

# single - raus

# fig_names = [[doppelz1[0].split('PF')[0]+run_direc[0], doppelz1[0].split('PF')[0]+'norm_'+run_direc[0]],
#              [doppelz1[0].split('PF')[0]+run_direc[1], doppelz1[0].split('PF')[0]+'norm_'+run_direc[1]],
#              [doppelz2[0].split('PF')[0]+run_direc[2], doppelz2[0].split('PF')[0]+'norm_'+run_direc[2]],
#              [doppelz2[0].split('PF')[0]+run_direc[3], doppelz2[0].split('PF')[0]+'norm_'+run_direc[3]],
#              [doppelz3[0].split('PF')[0]+run_direc[4], doppelz3[0].split('PF')[0]+'norm_'+run_direc[4]],
#              [doppelz3[0].split('PF')[0]+run_direc[5], doppelz3[0].split('PF')[0]+'norm_'+run_direc[5]],
#              [doppelz4[0].split('PF')[0]+run_direc[6], doppelz4[0].split('PF')[0]+'norm_'+run_direc[6]],
#              [doppelz4[0].split('PF')[0]+run_direc[7], doppelz4[0].split('PF')[0]+'norm_'+run_direc[7]],
#              [doppelz5[0].split('PF')[0]+run_direc[8], doppelz5[0].split('PF')[0]+'norm_'+run_direc[8]],
#              [doppelz5[0].split('PF')[0]+run_direc[9], doppelz5[0].split('PF')[0]+'norm_'+run_direc[9]],
#              [doppelz6[0].split('PF')[0]+run_direc[10], doppelz6[0].split('PF')[0]+'norm_'+run_direc[10]],
#              [doppelz6[0].split('PF')[0]+run_direc[11], doppelz6[0].split('PF')[0]+'norm_'+run_direc[11]],
#              [doppelz7[0].split('PF')[0]+run_direc[12], doppelz7[0].split('PF')[0]+'norm_'+run_direc[12]],
#              [doppelz7[0].split('PF')[0]+run_direc[13], doppelz7[0].split('PF')[0]+'norm_'+run_direc[13]],
#              [doppelz8[0].split('PF')[0]+run_direc[14], doppelz8[0].split('PF')[0]+'norm_'+run_direc[14]],
#              [doppelz8[0].split('PF')[0]+run_direc[15], doppelz8[0].split('PF')[0]+'norm_'+run_direc[15]],
#              [doppelz9[0].split('PF')[0]+run_direc[16], doppelz9[0].split('PF')[0]+'norm_'+run_direc[16]],
#              [doppelz9[0].split('PF')[0]+run_direc[17], doppelz9[0].split('PF')[0]+'norm_'+run_direc[17]],
#              [doppelz10[0].split('PF')[0]+run_direc[18], doppelz10[0].split('PF')[0]+'norm_'+run_direc[18]],
#              [doppelz10[0].split('PF')[0]+run_direc[19], doppelz10[0].split('PF')[0]+'norm_'+run_direc[19]],
#              [doppelz11[0].split('PF')[0]+run_direc[20], doppelz11[0].split('PF')[0]+'norm_'+run_direc[20]],
#              [doppelz11[0].split('PF')[0]+run_direc[21], doppelz11[0].split('PF')[0]+'norm_'+run_direc[21]],
#              [doppelz12[0].split('PF')[0]+run_direc[22], doppelz12[0].split('PF')[0]+'norm_'+run_direc[22]],
#              [doppelz12[0].split('PF')[0]+run_direc[23], doppelz12[0].split('PF')[0]+'norm_'+run_direc[23]],
#              [doppelz13[0].split('PF')[0]+run_direc[24], doppelz13[0].split('PF')[0]+'norm_'+run_direc[24]],
#              [doppelz13[0].split('PF')[0]+run_direc[25], doppelz13[0].split('PF')[0]+'norm_'+run_direc[25]],
#              [doppelz14[0].split('PF')[0]+run_direc[26], doppelz14[0].split('PF')[0]+'norm_'+run_direc[26]],
#              [doppelz14[0].split('PF')[0]+run_direc[27], doppelz14[0].split('PF')[0]+'norm_'+run_direc[27]]]


def plot_smoothed_cell(axes, list_virt_real_cell, direction, win=5., fz=13):  # default win=5.

    x_label = ['Virtual position (m)', 'Real position (m)']
    x_limits = [(0, 2), (0, 4)]
    counter = 0

    for cell in numpy.arange(len(list_virt_real_cell)):
        for input in [0, 1]:

            fig, axes = pl.subplots(1, 1, figsize=(10, 7))
            sns.despine(ax=axes, top=True, right=True)
            sns.set(style="ticks", font_scale=1.5)

            c1 = custom_plot.pretty_colors_set2[0]
            c2 = custom_plot.pretty_colors_set2[1]

            data1 = list_virt_real_cell[cell][0][direction[cell]+'FR_x_y_gain_0.5']
            data2 = list_virt_real_cell[cell][0][direction[cell]+'FR_x_y_gain_1.5']

            # data1s = data1[1]
            # data2s = data2[1]
            data1s = signale.tools.smooth(data1[1], window_len=win)
            data2s = signale.tools.smooth(data2[1], window_len=win)

            vis_track_length05 = 2.
            vis_track_length15 = 2.

            if input == 1:
                data1[0] /= 0.5
                data2[0] /= 1.5
                vis_track_length05 /= 0.5
                vis_track_length15 /= 1.5

            if direction[cell] == 'left' and input == 0:
                data1[0] = abs(data1[0]-vis_track_length05)
                data2[0] = abs(data2[0]-vis_track_length15)

            axes.fill_between(data1[0], data1s, facecolor=c1, color=c1, alpha=0.5)
            axes.plot(data1[0], data1s, color=c1, linewidth=2)
            axes.fill_between(data2[0], data2s, facecolor=c2, color=c2, alpha=0.5)
            axes.plot(data2[0], data2s, color=c2, linewidth=2)

            # if counter == len(axes)-2:
            # axes.set_ylabel('Firing rate (Hz)')
            # if counter in [len(axes)-2, len(axes)-1]:
            # axes.set_xlabel(x_label[input])

            ylim = axes.get_ylim()
            axes.set_xlim(x_limits[input])
            axes.set_ylim(-ylim[1]/10., ylim[1])
            # axes.tick_params(labelsize=fz)

            # axes.get_xaxis().set_visible(False)
            # axes.get_yaxis().set_visible(False)
            pl.axis('off')

            custom_plot.add_scalebar(ax=axes, matchx=False, matchy=False, hidex=False, hidey=False,
                                     sizex=.5, sizey=0, loc=3, thickness=ylim[1]/100)

            # counter += 1

            fig.savefig(where+fig_names[cell][input]+'.pdf', format='pdf')


if __name__ == "__main__":

    # cells = [doppel3]

    # fig_rows = len(cells)
    #
    # fig, ax = pl.subplots(fig_rows, 2, figsize=(10, (2*fig_rows)), sharey=True, sharex='col')
    #
    # ax = ax.flatten()
    #
    # width = 0.42
    # height = 0.7/fig_rows
    #
    # x_begin = 0.06
    # y_begin = 0.35/fig_rows
    #
    # x_space = 0.05
    # y_space = 0.2/fig_rows
    #
    # ax_positions = []
    #
    # for r in numpy.arange(fig_rows):
    #     ax_positions.append([x_begin, y_begin+((fig_rows-1-r)*height)+((fig_rows-1-r)*y_space)])
    #     ax_positions.append([x_begin+width+x_space, y_begin+((fig_rows-1-r)*height)+((fig_rows-1-r)*y_space)])
    #
    # for i, a in enumerate(ax):
    #     pos1 = a.get_position()  # get the original position
    #     pos2 = [ax_positions[i][0], ax_positions[i][1],  width, height]  # x, y, width, height
    #     a.set_position(pos2)  # set a new position

    plot_smoothed_cell(axes=None, list_virt_real_cell=cells, direction=run_direc)
    # fig.savefig(where+fig_names[c][input]+'.pdf', format='pdf')
    # pl.close()

