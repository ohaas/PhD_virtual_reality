__author__ = "Olivia Haas"
__version__ = "1.0, June 2017"

# python modules

import numpy
import matplotlib.pyplot as pl
from matplotlib.lines import Line2D
import hickle

server = 'saw'
# a = hickle.load('/users/haasolivia/documents/'+server+'/dataWork/olivia/hickle/FR_SR/anova_stats_all_pf.hkl')
a = hickle.load('/users/haasolivia/documents/'+server+'/dataWork/olivia/hickle/FR_SR/mannwhitneyu_stats_all_pf.hkl')

min_runs = 10

# smaller = numpy.where(numpy.array(a['anova_pvalues']) < 0.05)[0]
# larger = numpy.where(numpy.array(a['anova_pvalues']) > 0.05)[0]

smaller = numpy.where(numpy.array(a['mannwhitneyu_pvalues']) < 0.05)[0]
larger = numpy.where(numpy.array(a['mannwhitneyu_pvalues']) > 0.05)[0]

la = abs(numpy.diff(numpy.array(a['max_FR'])[larger]))
sm = abs(numpy.diff(numpy.array(a['max_FR'])[smaller]))
allr = abs(numpy.diff(numpy.array(a['max_FR'])))

# find vis prop fields
vis = numpy.array(a['vis_idx'])
prop = numpy.array(a['prop_idx'])
conf = numpy.delete(numpy.arange(len(a['max_FR'])), numpy.concatenate((vis, prop)))
vp = numpy.chararray(len(a['max_FR']))  # initialise string array
vp[vis] = 'v'
vp[prop] = 'l'
vp[conf] = 'c'

# remove all sessions which have less than 5 run in each gain
l = numpy.array(a['SR_len'])[larger]
out = numpy.where(l<min_runs)[0]
nl = numpy.delete(numpy.arange(len(l)), out)
la = la[nl]

s = numpy.array(a['SR_len'])[smaller]
outs = numpy.where(s<min_runs)[0]
ns = numpy.delete(numpy.arange(len(s)), outs)
sm = sm[ns]

a1 = numpy.array(a['SR_len'])
outa = numpy.where(a1<min_runs)[0]
na = numpy.delete(numpy.arange(len(a1)), outa)
allr = allr[na]

# calculate median for all runs
median_fr = numpy.nanmedian(allr)
median_fr_l = numpy.nanmedian(la)
median_fr_s = numpy.nanmedian(sm)

# find visual, locomotor and confidence interval fields corresponding to p<0.05 and p>0.05
vp = vp[na]
smaller1 = numpy.where(numpy.array(a['mannwhitneyu_pvalues'])[na] < 0.05)[0]
larger1 = numpy.where(numpy.array(a['mannwhitneyu_pvalues'])[na] > 0.05)[0]
s_vp = vp[smaller1]
l_vp = vp[larger1]
s_count_loco = len(numpy.where(s_vp == 'l')[0])
s_count_vis = len(numpy.where(s_vp == 'v')[0])
s_count_conf = len(numpy.where(s_vp == 'c')[0])

l_count_loco = len(numpy.where(l_vp == 'l')[0])
l_count_vis = len(numpy.where(l_vp == 'v')[0])
l_count_conf = len(numpy.where(l_vp == 'c')[0])

# -----------------------------------------------------------

# plotting

min01 = min(min(la), min(sm))
max01 = max(max(la), max(sm))
binwidth = 3
pl.hist(la, fill=False, linewidth=2, bins=numpy.arange(min01, max01 + binwidth, binwidth))

width = 1.5
heights, freq = numpy.histogram(sm, bins=numpy.arange(min01, max01 + binwidth, binwidth))
#freq = [1.5, 4.25, 7.0, 9.75, 12.5, 15.25, 18.0, 20.75, 23.5, 26.25]
offset = binwidth/2. -  width/2.
freq = freq[:-1]+offset
pl.bar(freq, heights, width, color='r', edgecolor = "none")

# pl.axvline(median_fr, color='b', linestyle='dashed', linewidth=2)
pl.axvline(median_fr_l, color='k', linestyle='dashed', linewidth=2)
pl.axvline(median_fr_s, color='r', linestyle='dashed', linewidth=2)

pl.xlabel('Single run FR differences between both gains (Hz)')
pl.ylabel('Count')

line1 = Line2D([0], [0], linestyle="-", linewidth=4, color='r')
line2 = Line2D([0], [0], linestyle="-", linewidth=4, color='k')
line3 = Line2D([0], [0], linestyle="--", linewidth=4, color='b')
line4 = Line2D([0], [0], linestyle="--", linewidth=4, color='k')
line5 = Line2D([0], [0], linestyle="--", linewidth=4, color='r')

leg = pl.legend([line1, line2, line5], ['n = '+str(len(sm))+' (n$_{l}$ = '+str(s_count_loco)+
                                 ', n$_{v}$ = '+str(s_count_vis)+', n$_{c}$ = '+str(s_count_conf)+
                                 ') with p < 0.05', 'n = '+str(len(la))+' (n$_{l}$ = '+str(l_count_loco)+
                                 ', n$_{v}$ = '+str(l_count_vis)+', n$_{c}$ = '+str(l_count_conf)+') with p > 0.05',
                                        # 'median = '+str(numpy.around(median_fr, decimals=2))],
                                        'median = '+str(numpy.around(median_fr_s, decimals=2)),
                                               'median = '+str(numpy.around(median_fr_l, decimals=2))],
                numpoints=1, bbox_to_anchor=(1., 1.0), fontsize=15)
leg.get_frame().set_linewidth(0.0)
pl.show()