__author__ = "Olivia Haas"

import os
import sys

# add additional custom paths
extraPaths = [os.path.join(os.path.abspath(os.path.dirname(__file__)), '../scripts')]
for p in extraPaths:
    if not sys.path.count(p):
        sys.path.insert(1, p)

import numpy
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as pl

import hickle
from sklearn import mixture
import statsmodels
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import random
from dateutil import parser
import scipy

import signale
import seaborn as sns

from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
# from matplotlib.collections import PatchCollection
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
from matplotlib import rcParams
# from matplotlib import cm
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes

import custom_plot
from subprocess import call
import pandas
from collections import Counter
import csv

# initialte lists to be saved in hickle format:
MG_large = []
MG_small = []

MxG_large = []
MxG_small = []

MG_std_large = []
MG_std_small = []

Weights_largeGauss = []
Weights_smallGauss = []

Mx_combinedGauss = []
MG_combined = []

X = []
Y_large = []
Y_small = []
Y_comined = []

orig_data_x = []
orig_data_y = []
gauss_x = []
gauss1_y = []
gauss2_y = []
derivative_y = []
cumulative_x = []
cumulative_y = []
cumulative_95perc_index = []
real_data_in_cumulative_x = []

c_05 = custom_plot.pretty_colors_set2[0]
c_15 = custom_plot.pretty_colors_set2[1]

server = 'saw'

path = '/Users/haasolivia/Documents/'+server+'/dataWork/olivia/hickle/'
# path = '/home/ephysdata/dataWork/olivia/hickle/'
# path = '/Users/haasolivia/Desktop/plots/'

hkl_files_pre = numpy.repeat(hickle.load(path+'Summary/used_filenames.hkl'), 2)

# hkl_files_pre1 = numpy.repeat(hkl_files_pre, 2)
# hkl_files = [hkl_files_pre[i].split('_normalised')[0]+'.hkl' if (i%2 != 0) else hkl_files_pre[i] for i in range(len(hkl_files_pre))]
hkl_files = [hkl_files_pre[i] if (i%2 != 0) else hkl_files_pre[i] for i in range(len(hkl_files_pre))]

run_direc = numpy.repeat(hickle.load(path+'Summary/running_directions.hkl'), 2)

vis_hkl_files = hickle.load(path+'Summary/used_filenames.hkl')
# vis_run_direc = hickle.load(path+'Summary/running_directions.hkl')

# addon1 = 'Summary/summary_dict'
# addon2 = '_FRySum'

# vis_info = hickle.load(path+addon1+addon2+'.hkl')
# norm_info = hickle.load(path+addon1+'_normalised'+addon2+'.hkl')
# vis_info = hickle.load(path+addon1+'.hkl')
# norm_info = hickle.load(path+addon1+'_normalised.hkl')

# pfc = 'pf_center_change in m (gain 0.5 - gain 1.5 center)'
# PFx05 = 'x_pf_maxFR_gain_0.5 in m'
# PFx15 = 'x_pf_maxFR_gain_1.5 in m'

###################################################### command line parameters

multi = 100  # used to multiply FR in order to make a histogram
thresh = 0.2  # 0.2 = 20 % of 1d FR peak used as data cut off for the fit
fr_tresh = 3.  # for 2d data in Hz should be the same as FR_thresh_2nd_gain in hickle_histogram.py
smooth = 15.
pf_width_thresh = 0.05  # --> 2m*0.05 = 0.1 m = 10 cm
pf_width_max = 4.0/5  # for gain 1.33 pf sizes can be around 1.05 meters! (2m*(4./5) = 1.6 meters)
# surrogate_repeat = 1
for argv in sys.argv[1:]:
    if argv.startswith('multi:'):
        multi = float(argv.split(':')[-1])
    if argv.startswith('thresh:'):
        thresh = float(argv.split(':')[-1])
    if argv.startswith('smooth:'):
        smooth = float(argv.split(':')[-1])
    if argv.startswith('surrogate_repeat:'):
        surrogate_repeat = int(argv.split(':')[-1])


###################################################### functions


def gauss_fkt(x, *p):
    A, mu, sigma = p
    print 'm = ', mu
    print 'sigma = ', sigma
    return A*numpy.exp(-(x-mu)**2/(2.*sigma**2))


def double_gauss_fkt(x, *p):
    A1, mu1, sigma1, A2, mu2, sigma2 = p

    y = A1*numpy.exp(-(x-mu1)**2/(2.*sigma1**2))
    y += A2*numpy.exp(-(x-mu2)**2/(2.*sigma2**2))

    return y


def m_val(gauss1, gauss2, x_doublePointNum, ax=False, fig=False):

    # get values for maxima of both Gaussians
    if gauss1.max() >= gauss2.max():
        small_max = gauss2.max()
        small_max_index = gauss2.argmax()
        big_max = gauss1.max()
        big_max_index = gauss1.argmax()
    else:
        small_max = gauss1.max()
        small_max_index = gauss1.argmax()
        big_max = gauss2.max()
        big_max_index = gauss2.argmax()

    # get values for minimum inbetween Gaussians
    double_gauss = gauss1+gauss2
    indices = numpy.sort([small_max_index, big_max_index])
    #indices += numpy.array([1, -1]) * 5     # shrink by a few bins such that derivative is not returning the maxima itself

    d_double_gauss = double_gauss #numpy.abs(numpy.diff(double_gauss))

    if indices[0] == indices[1]:
        m = numpy.nan

    else:

        between_peak_min_index = d_double_gauss[indices[0]:indices[1]].argmin() + indices[0]
        between_peak_min = double_gauss[between_peak_min_index]

        # calculate m-value_______________________________________________________________________
        index_delta = small_max_index - between_peak_min_index
        if index_delta < 0:
            s_index = numpy.max([0, small_max_index + index_delta])
            l_index = numpy.min([small_max_index - index_delta, double_gauss.size-1])
        else:
            s_index = numpy.max([0, small_max_index - index_delta])
            l_index = numpy.min([small_max_index + index_delta, double_gauss.size-1])

        small_peak_mean = numpy.mean(double_gauss[s_index: l_index])
        delta_F = small_max - between_peak_min
        m = delta_F/small_peak_mean

    # plotting
    if ax:
        if indices[0] != indices[1]:
            x_between = x_doublePointNum[between_peak_min_index]
            ax.plot(x_between, between_peak_min, 'ro')
        if fig:
            ax22_pos = list(ax.get_position().bounds)
            # ax22_pos[2] *= .8
            # ax[0].set_position(ax22_pos)

            # have to add 2 degrees to intersection angle, in order to get to the middle of the 4 degree wide bin!
            fig.text(ax22_pos[0]+ax22_pos[2]*.15, ax22_pos[1]+ax22_pos[3]*.9, 'intersec. angle: '+str(numpy.round(x_between, 3)+2))
            angle = 180*numpy.arctan(2./(4./3))/numpy.pi
            fig.text(ax22_pos[0]+ax22_pos[2]*.15, ax22_pos[1]+ax22_pos[3]*.7, 'geometic angle: '+str(numpy.round(angle, 3)))
        ax.plot(x_doublePointNum[small_max_index], small_max, 'go')
        ax.plot(x_doublePointNum[big_max_index], big_max, 'bo')

        if not numpy.array_equal(d_double_gauss, double_gauss):
            ax.plot(x_doublePointNum[:d_double_gauss.size], d_double_gauss/d_double_gauss.max()*double_gauss.max(), ':')

    return m


def two_gaussians(x_in, x01, x02, sigma1, sigma2):  # x-values, amplitude a, x0 x-value of gauss peak, sigma the standart deviation
    A = (1/(sigma1*numpy.sqrt(2*numpy.pi))) * numpy.exp(-(x_in-x01)**2/(2*sigma1**2))
    B = (1/(sigma2*numpy.sqrt(2*numpy.pi))) * numpy.exp(-(x_in-x02)**2/(2*sigma2**2))
    return A + B


def two_gaussians1(x_in, x01, x02, sigma1, sigma2):  # x-values, amplitude a, x0 x-value of gauss peak, sigma the standart deviation
    A = (1/(sigma1*numpy.sqrt(2*numpy.pi))) * numpy.exp(-(x_in-x01)**2/(2*sigma1**2))
    B = (1/(sigma2*numpy.sqrt(2*numpy.pi))) * numpy.exp(-(x_in-x02)**2/(2*sigma2**2))
    return A, B


def func(x, a, x0, sigma):
    return a*numpy.exp(-(x-x0)**2/(2*sigma**2))


def gaussfunc_2g(x_in, p):
    gauss1 = p[0]*numpy.exp(-.5*((x_in-p[1])/p[2])**2)
    gauss2 = p[3]*numpy.exp(-.5*((x_in-p[4])/p[5])**2)
    return gauss1+gauss2


def fit_gaussians_etc(x, y, surrogate_repeat, gain, run_direc, file, savefig=True, fz=14, hist=False):
    M = []
    M_data = []
    good = 0
    # window length is in bins (1 FR bin = 2cm) => 3*2cm = 6cm Kernel (Chen = 8.5cm, Ravassard = 5cm)
    y = signale.tools.smooth(y, window_len=3.)

    # generate one more x-point between all existing x-points:
    x_doublePointNum = numpy.arange(x[0], x[-1]+numpy.diff(x)[0]/2., numpy.diff(x)[0]/2.)

    # data = pl.hist(x, weights=y, bins=len(numpy.arange(min(x), max(x), x[1]-x[0])))
    if max(y) < 5:
        input_y = 5. * (y/max(y))
    else:
        input_y = multi*y

    if not hist:
        data = numpy.repeat(x, numpy.around(input_y).astype(int))
    else:
        data = x
        x = numpy.unique(x)
    # bin_num = len(numpy.arange(min(x), max(x), x[1]-x[0]))
    #
    # # fit models with 1-10 components
    # # N = numpy.arange(1, 41)
    # # models = [None for i in range(len(N))]
    # #
    # # for i in range(len(N)):
    # #     models[i] = mixture.GMM(N[i]).fit(numpy.vstack((input)))
    #
    # # compute the AIC and the BIC
    # # AIC = [m.aic(numpy.vstack((input))) for m in models]
    # # BIC = [m.bic(numpy.vstack((input))) for m in models]
    #

    for su in numpy.arange(surrogate_repeat+1):

        if su != 0 and good == 1:
            if not hist:
                surrogate_data = numpy.random.choice(list(x), len(data))
                binsize = abs(x[1]-x[0])
            else:
                x_su = numpy.random.choice(list(numpy.arange(0, 2.1, .1)), len(data))
                y_su = numpy.random.choice(list(numpy.arange(0, 2.1, .1)), len(data))
                surrogate_data = numpy.array([180*numpy.arctan(y_su[idx]/x_su[idx])/numpy.pi for idx
                                              in numpy.arange(len(x_su))])
                binsize = 4.   # in degerees

            data = surrogate_data
            bin_num = len(numpy.arange(min(x), max(x), binsize))
            new_y = numpy.histogram(surrogate_data, bins=bin_num, range=(min(x), max(x)))[0]
            y = new_y

            if max(y) < 5:
                y = (new_y * max(y))/5.
            else:
                y = new_y

        gmm = mixture.GMM(n_components=2, covariance_type='full', min_covar=0.0000001)  # gmm for two components
        gmm.fit(numpy.vstack(data))  #numpy.vstack(data))  #numpy.vstack(data))  # train it!
        #
        # # m1, m2 = gmm.means_
        # # w1, w2 = gmm.weights_
        # # c1, c2 = gmm.covars_
        #
        # # histdist = pl.hist(data, bins=bin_num, normed=True, alpha=0.2)
        #
        # # linspace = numpy.linspace(min(x), max(x), len(x))
        #
        # #
        # # ax1.plot(N, AIC, '-k', label='AIC')
        # # ax1.plot(N, BIC, '--k', label='BIC')
        # # ax1.set_xlabel('n. components')
        # # ax1.set_ylabel('information criterion')
        # # ax1.legend(loc=2)
        #
        # # plot data histogram
        # # ax1.hist(data, bin_num)  # draw samples
        #
        # gauss normed to a maximum of 1, weight by the maximum of the distribution y
        # gauss1 = max(y)*(gmm.weights_[0] * matplotlib.mlab.normpdf(x, gmm.means_[0], numpy.sqrt(gmm.covars_[0])))[0]
        # gauss2 = max(y)*(gmm.weights_[1] * matplotlib.mlab.normpdf(x, gmm.means_[1], numpy.sqrt(gmm.covars_[1])))[0]

        gauss1 = (gmm.weights_[0] * matplotlib.mlab.normpdf(x_doublePointNum, gmm.means_[0], numpy.sqrt(gmm.covars_[0])))[0]
        gauss2 = (gmm.weights_[1] * matplotlib.mlab.normpdf(x_doublePointNum, gmm.means_[1], numpy.sqrt(gmm.covars_[1])))[0]
        #
        # gauss1_old = (gmm.weights_[0] * matplotlib.mlab.normpdf(x, gmm.means_[0], numpy.sqrt(gmm.covars_[0])))[0]
        # gauss2_old = (gmm.weights_[1] * matplotlib.mlab.normpdf(x, gmm.means_[1], numpy.sqrt(gmm.covars_[1])))[0]
        #
        #
        # print 'x', x
        # print 'gauss1_old', gauss1_old
        #
        # print 'x_doublePointNum', x_doublePointNum
        # print 'gauss1', gauss1

        #
        # calculate basic values for the distribution y
        std = numpy.std(y)
        mean = numpy.mean(y)
        #
        # calculate basic values for the gaussians

        mg1a = max(gauss1)
        mg2a = max(gauss2)

        # if mg1a >= mg2a:
        #     g1_maxFR = max(y)
        #     g2_maxFR = g1_maxFR / (mg1a/mg2a)
        # else:
        #     g2_maxFR = max(y)
        #     g1_maxFR = g2_maxFR / (mg2a/mg1a)
        #
        stdg1 = numpy.sqrt(gmm.covars_[0])[0][0]
        stdg2 = numpy.sqrt(gmm.covars_[1])[0][0]
        # print 'stdg1 ', stdg1
        # print 'stdg2 ', stdg2

        # large_val = 0.4

        # print 'x diff: ', abs(x[numpy.argmax(gauss1)]-x[numpy.argmax(gauss2)])

        # if abs(x[numpy.argmax(gauss1)]-x[numpy.argmax(gauss2)]) <= 1.0 and stdg1 > large_val and stdg2 > large_val:
        #     std_dev = 4.
        # else:

        xDiff = abs(x_doublePointNum[numpy.argmax(gauss1)]-x_doublePointNum[numpy.argmax(gauss2)])

        if file.endswith('normalised.hkl'):
            xDiff_cutoff = 0.2/float(gain)
        else:
            xDiff_cutoff = 0.20


        # if stdg1 < 0.3 or stdg1 > large_val and stdg2 > large_val:
            # if stdg1 > 0.4:
            #     print 'std_dev', 8.
            #     std_dev = 8.

        std_dev1 = 2.
        std_dev2 = 2.

        if xDiff > xDiff_cutoff < (1./4.)*max(x) and stdg1 > 0.3:
            std_dev1 = 8.
        if xDiff > xDiff_cutoff < (1./4.)*max(x) and stdg2 > 0.3:
            std_dev2 = 8.

        # print 'gauss1 ', gauss1
        # print 'numpy.argmax(gauss1) ', numpy.argmax(gauss1)
        x1 = numpy.argmax(gauss1)+1
        if x1 >= len(x)-1:
            x1 = len(x)-3
        g1_max0 = numpy.argmin(abs(x_doublePointNum[0:x1]-(x_doublePointNum[x1-1]-stdg1/std_dev1)))
        g1_max1 = numpy.argmin(abs(x_doublePointNum[x1:-1]-(x_doublePointNum[x1-1]+stdg1/std_dev1)))+x1

        if g1_max0 >= len(y)-1 and g1_max1+1 >= len(y)-1:
            # print 'both larger than array'
            # print 'g1_max0, g1_max1+1 ', g1_max0, g1_max1+1
            g1_max0 = len(y)-5
            g1_max1 = len(y)-2
        if g1_max0 == g1_max1+1:
            # print 'both equal'
            # print 'g1_max0, g1_max1+1 ', g1_max0, g1_max1+1
            g1_max0 -= 2
            g1_max1 += 2
        if g1_max0 < 0:
            # print 'first one smaller than zero'
            # print 'g1_max0, g1_max1+1 ', g1_max0, g1_max1+1
            g1_max0 = 0
        if g1_max1+1 > len(y)-1:
            # print 'second one larger than array'
            # print 'g1_max0, g1_max1+1 ', g1_max0, g1_max1+1
            g1_max1 = len(y)-2

        # print 'g1_max0, g1_max1+1 ', g1_max0, g1_max1+1
        # print 'len(y)-1 ', len(y)-1
        g1_maxFR = max(y[g1_max0:g1_max1+1])

        # print 'normalising window g1: ', x[g1_max0:g1_max1+1][0], ' to ', x[g1_max0:g1_max1+1][-1]

        # print 'stdg2 ', stdg2
        # if stdg2 < 0.3 or stdg1 > large_val and stdg2 > large_val:
            # if stdg2 > 0.4:
            #     print 'std_dev', 8.
            #     std_dev = 8.
        # print 'gauss2 ', gauss2
        # print 'numpy.argmax(gauss2) ', numpy.argmax(gauss2)
        x2 = numpy.argmax(gauss2)+1
        if x2 >= len(x)-1:
            x2 = len(x)-3
        g2_max0 = numpy.argmin(abs(x_doublePointNum[0:x2]-(x_doublePointNum[x2-1]-stdg2/std_dev2)))
        g2_max1 = numpy.argmin(abs(x_doublePointNum[x2:-1]-(x_doublePointNum[x2-1]+stdg2/std_dev2)))+x2

        # print 'g2_max0, g2_max1+1 ', g2_max0, g2_max1+1
        if g2_max0 >= len(y)-1 and g2_max1+1 >= len(y)-1:
            g2_max0 = len(y)-5
            g2_max1 = len(y)-2
            # print 'both larger than array'
            # print 'g2_max0, g2_max1+1 ', g2_max0, g2_max1+1
        if g2_max0 == g2_max1+1:
            g2_max0 -= 2
            g2_max1 += 2
            # print 'both equal'
            # print 'g2_max0, g2_max1+1 ', g2_max0, g2_max1+1
        if g2_max0 < 0:
            g2_max0 = 0
            # print 'first one smaller than zero'
            # print 'g2_max0, g2_max1+1 ', g2_max0, g2_max1+1
        if g2_max1+1 > len(y)-1:
            g2_max1 = len(y)-2
            # print 'second one larger than array'
            # print 'g2_max0, g2_max1+1 ', g2_max0, g2_max1+1

        # print 'g2_max0, g2_max1+1 ', g2_max0, g2_max1+1
        # print 'len(y)-1 ', len(y)-1
        g2_maxFR = max(y[g2_max0:g2_max1+1])

        # print 'normalising window g2: ', x[g2_max0:g2_max1+1][0], ' to ', x[g2_max0:g2_max1+1][-1]

        # set gauss closest to y distribution maximum to its maximum
        nearest_yMax_gauss = signale.tools.findNearest(numpy.array([x_doublePointNum[numpy.argmax(gauss1)],
                                                                    x_doublePointNum[numpy.argmax(gauss2)]]),
                                                       x[numpy.argmax(y)])[0]
        if nearest_yMax_gauss == 0:
            g1_maxFR = max(y)
        else:
            g2_maxFR = max(y)

        # normalise gaussians to FR maximum of distribution y within gausstian maximum + / - 0.5 its std:
        gauss1 = g1_maxFR*(gauss1/mg1a)
        gauss2 = g2_maxFR*(gauss2/mg2a)

        amplitude_g1 = gmm.weights_[0] * g1_maxFR/mg1a
        amplitude_g2 = gmm.weights_[1] * g2_maxFR/mg2a

        weight_g1 = amplitude_g1/(amplitude_g1 + amplitude_g2)
        weight_g2 = amplitude_g2/(amplitude_g1 + amplitude_g2)

        mg1 = max(gauss1)
        mg2 = max(gauss2)
        max_mean_diff_in_std1 = (mg1 - mean)/std
        max_mean_diff_in_std2 = (mg2 - mean)/std

        if file.endswith('normalised.hkl'):
            max_x = 2./float(gain)
        else:
            max_x = 2.

        # append gauss maxima and maxima-mean-difference in multiples of std to list

        if mg1 >= mg2:

            # print 'red stdg1 ', numpy.sqrt(gmm.covars_[0])[0][0]
            # print 'black stdg2 ', numpy.sqrt(gmm.covars_[1])[0][0]
            if su == 0:
                large_x = x_doublePointNum[numpy.argmax(gauss1)]
                small_x = x_doublePointNum[numpy.argmax(gauss2)]

                if large_x > max_x:
                    large_x = max_x
                if small_x > max_x:
                    small_x = max_x
                if large_x < 0:
                    large_x = 0
                if small_x < 0:
                    small_x = 0

                # print 'large x = ', large_x
                # print 'small x = ', small_x

                MG_large.append(mg1)
                MG_small.append(mg2)
                MxG_large.append(large_x)
                MxG_small.append(small_x)
                MG_std_large.append(max_mean_diff_in_std1)
                MG_std_small.append(max_mean_diff_in_std2)
                Weights_largeGauss.append(weight_g1)
                Weights_smallGauss.append(weight_g2)
                Mx_combinedGauss.append(x_doublePointNum[numpy.argmax(gauss1+gauss2)])
                MG_combined.append(max(gauss1+gauss2))
                Y_large.append(numpy.array(gauss1))
                Y_small.append(numpy.array(gauss2))
                X.append(numpy.array(x_doublePointNum))

            colour = ['r', 'k']
            small_max = mg2
            small_max_index = numpy.argmax(gauss2)
        else:

            # print 'red stdg2 ', numpy.sqrt(gmm.covars_[1])[0][0]
            # print 'black stdg1 ', numpy.sqrt(gmm.covars_[0])[0][0]
            if su == 0:
                large_x = x_doublePointNum[numpy.argmax(gauss2)]
                small_x = x_doublePointNum[numpy.argmax(gauss1)]

                if large_x > max_x:
                    large_x = max_x
                if small_x > max_x:
                    small_x = max_x
                if large_x < 0:
                    large_x = 0
                if small_x < 0:
                    small_x = 0

                # print 'large x = ', large_x
                # print 'small x = ', small_x

                MG_large.append(mg2)
                MG_small.append(mg1)
                MxG_large.append(large_x)
                MxG_small.append(small_x)
                MG_std_large.append(max_mean_diff_in_std2)
                MG_std_small.append(max_mean_diff_in_std1)
                Weights_largeGauss.append(weight_g2)
                Weights_smallGauss.append(weight_g1)
                Mx_combinedGauss.append(x_doublePointNum[numpy.argmax(gauss1+gauss2)])
                MG_combined.append(max(gauss1+gauss2))
                Y_large.append(numpy.array(gauss2))
                Y_small.append(numpy.array(gauss1))
                X.append(numpy.array(x_doublePointNum))

            colour = ['k', 'r']
            small_max = mg1
            small_max_index = numpy.argmax(gauss1)

        # calculate values to get m = deltaF/Fmean:
        derivative1 = numpy.diff(gauss1+gauss2) / numpy.diff(x_doublePointNum)

        # remove negative values in beginning of derivative
        if run_direc == 'left':
            # for leftwards runs the array is starting from the end of the track!
            sc = -1
            pre_sign = 1
            sign_array = numpy.arange(len(derivative1))[::-1]  # backwards array
        else:
            sc = 0
            pre_sign = -1
            sign_array = numpy.arange(len(derivative1))

        zero_crossings = numpy.where(numpy.diff(numpy.sign(derivative1)))[0]

        # print 'zero_crossings: ', zero_crossings

        if len(zero_crossings):
            first_sign_change = zero_crossings[sc]+1

            if run_direc == 'left':
                derivative1[first_sign_change:len(derivative1)][derivative1[first_sign_change:len(derivative1)] < 0] = 0.
            else:
                derivative1[0:first_sign_change][derivative1[0:first_sign_change] < 0] = 0.

        sign = numpy.sign(derivative1)

        # print 'derivative1 = ', derivative1
        # print 'sign before = ', sign
        # get rid of zeros and use sign value from the value before
        for l in sign_array:
            if sign[l] == 0.:
                if run_direc == 'right' and l == 0:
                    sign[l] = sign[l+1]
                elif run_direc == 'left' and l == len(sign)-1:
                    sign[l] = sign[l-1]
                else:
                    sign[l] = sign[l+pre_sign]
        # get rid of remaining zeros in the array edges
        for l in sign_array[::-1]:
            if sign[l] == 0.:
                if run_direc == 'left' and l == 0:
                    sign[l] = sign[l+1]
                elif run_direc == 'right' and l == len(sign)-1:
                    sign[l] = sign[l-1]
                else:
                    sign[l] = sign[l-pre_sign]

        # print 'sign = ', sign

        deri1_zero = numpy.where(numpy.diff(sign))[0]+1

        # print 'deri1_zero = ', deri1_zero
        if len(deri1_zero) == 3:
            between_peak_min_index = deri1_zero[1]

            between_peak_min = (gauss1+gauss2)[between_peak_min_index]
            index_delta = abs(between_peak_min_index-small_max_index)

            delta_F = small_max-between_peak_min

            if small_max_index-index_delta < 0:
                s_index = 0
            else:
                s_index = small_max_index-index_delta

            if small_max_index+index_delta+1 > len(x)-1:
                l_index = len(x_doublePointNum)-1
            else:
                l_index = small_max_index+index_delta+1

            small_peak_mean = numpy.mean((gauss1+gauss2)[s_index: l_index])

            m = delta_F/small_peak_mean

            if numpy.isnan(m):
                print 'delta_F = ', delta_F
                print 'small_peak_mean = ', small_peak_mean
                print 'mean for index1 to index2 : ', small_max_index-index_delta, small_max_index+index_delta+1
                print (gauss1+gauss2)[small_max_index-index_delta: small_max_index+index_delta+1]
                sys.exit()

            if su != 0:
                M.append(m)
            else:
                M_data.append(m)
                good = 1
                extra_path = 'Deriv_good/'

        else:
            if su == 0:
                M_data.append(numpy.nan)
                good = 0
                extra_path = 'Deriv_bad/'

        if su == 0:
            # plot data and gaussians from mixture model
            fig, ax11 = pl.subplots(2, 1, figsize=(18, 12))
            [ax1, ax0] = ax11.flatten()
            if good == 0:
                fig22, ax22 = pl.subplots(2, 1, figsize=(18, 12))
            else:
                fig22, ax22 = pl.subplots(3, 1, figsize=(18, 20))
            ax22 = ax22.flatten()
            # ax2 = ax1.twinx()
            # ax1.plot(x, numpy.around(input_y).astype(int), 'g')

            for axis in [ax1, ax22[0]]:
                axis.axhline(mean, linestyle='-', color=custom_plot.pretty_colors_set2[0], alpha=0.8, zorder=0)
                axis.axhspan(mean-std, mean+std, facecolor=custom_plot.pretty_colors_set2[0], alpha=0.2, linewidth=False, zorder=0)
                axis.plot(x, y, 'b')
                axis.plot(x_doublePointNum, gauss1, linewidth=2, color=colour[0])  # gauss1 = small gauss
                axis.plot(x_doublePointNum, gauss2, linewidth=2, color=colour[1])
                axis.plot(x_doublePointNum, gauss1+gauss2, linewidth=2, color='g')
                # axis.set_xlabel('Angle in degrees', fontsize=fz)
                # axis.set_ylabel('Count', fontsize=fz)
                axis.set_xlabel('Position from start point (m)', fontsize=fz)
                axis.set_ylabel('Firing rate (Hz)', fontsize=fz)

                axis.set_ylim(0, max(gauss1+gauss2)+0.01)
                axis.set_xlim(0, max(x))

            # appending data for hickle
            orig_data_x.append(x)
            orig_data_y.append(y)
            gauss_x.append(x_doublePointNum)
            gauss1_y.append(gauss1)
            gauss2_y.append(gauss2)
            derivative_y.append(derivative1)

            # pl.show()
            # sys.exit()
            if gain == '0.5':
                g = '05'
            else:
                g = '15'

            # save derivative plot
            for axe in [ax0, ax22[1]]:
                axe.plot(x_doublePointNum[:-1], derivative1, marker='o')
                axe.axhline(0, color='r', zorder=1)
                axe.set_xlabel('Position from start point (m)')
                axe.set_ylabel('dGMM/dx')
                axe.set_xlim(0, max(x))
            if savefig:
                if extra_path == 'Deriv_bad/':
                    # print 'saving figure under '+path+'Double_Peaks/'+extra_path+file.split('.hkl')[0]+'_'+run_direc+\
                    # '_gain_'+g+'_GMM_deriv.pdf'
                    fig.savefig(path+'Double_Peaks/'+extra_path+file.split('.hkl')[0]+'_'+run_direc+'_gain_'+g+\
                                 '_GMM_deriv.pdf', format='pdf')

                # save GMM + data plot
                # print 'saving figure under '+path+'Double_Peaks/'+file.split('.hkl')[0]+'_'+run_direc+'_gain_'+g+'.pdf'
                fig.savefig(path+'Double_Peaks/'+file.split('.hkl')[0]+'_'+run_direc+'_gain_'+g+'.pdf', format='pdf')

                if max_mean_diff_in_std1 < 1. or max_mean_diff_in_std2 < 1.:
                    # print 'saving figure under '+path+'Double_Peaks/Below_thresh1STD/'+file.split('.hkl')[0]+'_'+run_direc+\
                    #       '_gain_'+g+'.pdf'
                    fig.savefig(path+'Double_Peaks/Below_thresh1STD/'+file.split('.hkl')[0]+'_'+run_direc+
                                '_gain_'+g+'.pdf', format='pdf')

                else:
                    if xDiff < xDiff_cutoff:
                        # print 'saving figure under '+path+'Double_Peaks/Above_thresh1STD_xDiff_smaller_min/'+\
                        #       file.split('.hkl')[0]+'_'+run_direc+'_gain_'+g+'.pdf'
                        fig.savefig(path+'Double_Peaks/Above_thresh1STD_xDiff_smaller_min/'+file.split('.hkl')[0]+'_'+run_direc+
                                    '_gain_'+g+'.pdf', format='pdf')
                    else:
                        # print 'saving figure under '+path+'Double_Peaks/Above_thresh1STD_xDiff_larger_min/'+\
                        # file.split('.hkl')[0]+'_'+run_direc+'_gain_'+g+'.pdf'
                        fig.savefig(path+'Double_Peaks/Above_thresh1STD_xDiff_larger_min/'+file.split('.hkl')[0]+'_'+run_direc+
                                    '_gain_'+g+'.pdf', format='pdf')

        # pl.close('all')

    return M, M_data, good, fig22, ax22


def calc_peaks_bsp(file, gain, run_direc):
    gain = '1.5'
    a = hickle.load(path+file)
    fr = a[run_direc+'FR_x_y_gain_'+gain]

    x = fr[0]
    y = fr[1]

    # M, M_data, good, fig, ax = fit_gaussians_etc(x=x, y=y, surrogate_repeat=0, gain=gain, run_direc=run_direc,
    #                                              file=file, savefig=False)
    #
    # fig, ax1 = pl.subplots()
    # # print 'M_data = ', M_data
    # ax1.plot(numpy.sort(M_data), numpy.array(range(len(M_data)))/float(len(M_data)))

    if run_direc == 'left':  # for leftward runs plot abolute x-value from start position

        # sys.exit()
        vis_track_length = 2.

        if file.endswith('normalised.hkl'):
            start = vis_track_length/float(gain)
        else:
            start = vis_track_length
        x = abs(x-start)

    fig22, ax22, m = fit_gaussians(x=x, y_orig=y, plot_data=True, thresh=0.2)

    pl.show()


def make_subarrays_equal_long(mother_array):
    A = mother_array
    lenghs_A = []

    for a in numpy.arange(len(A)):
        if type(A[a]) == float or type(A[a]) == int:
            length = 1
        else:
            length = len(A[a])
        lenghs_A.append(length)

    for a in numpy.arange(len(A)):
        if lenghs_A[a] != max(lenghs_A):
            A[a] = numpy.append(A[a], numpy.repeat(numpy.nan, max(lenghs_A) - lenghs_A[a]))


def append_gauss_info(x_doublePointNum, gaussS, gaussL):

    weight_S = gaussS.max()/(gaussS.max() + gaussL.max())
    weight_L = gaussL.max()/(gaussS.max() + gaussL.max())

    large_x = x_doublePointNum[numpy.argmax(gaussL)]
    small_x = x_doublePointNum[numpy.argmax(gaussS)]

    max_x = x_doublePointNum.max()

    if large_x > max_x:
        large_x = max_x
    if small_x > max_x:
        small_x = max_x
    if large_x < 0:
        large_x = 0
    if small_x < 0:
        small_x = 0

    MG_large.append(max(gaussL))
    MG_small.append(max(gaussS))
    MG_combined.append(max(gaussS+gaussL))

    MxG_large.append(large_x)
    MxG_small.append(small_x)
    Mx_combinedGauss.append(x_doublePointNum[numpy.argmax(gaussS+gaussL)])

    Weights_largeGauss.append(weight_L)
    Weights_smallGauss.append(weight_S)

    X.append(numpy.array(x_doublePointNum))
    Y_large.append(numpy.array(gaussL))
    Y_small.append(numpy.array(gaussS))
    Y_comined.append(numpy.array(gaussS+gaussL))


def fit_gaussians(x, y_orig, surrogate=False, plot_data=False, fz=14, thresh=thresh, amp=None, hist=False):
    # global data, input_y, gmm, thresh, double_gauss, gauss1, gauss2, coeff

    factor_cope_with_gain = x.max()/2.

    if hist:
        his = numpy.histogram(x, bins=numpy.arange(min(x), max(x) + 4, 4))
        y_orig = his[0]
        factor_cope_with_gain = 1.

    # window length is in bins (1 FR bin = 2cm) => 3*2cm = 6cm Kernel (Chen = 8.5cm, Ravassard = 5cm)
    y_orig_dummy = numpy.copy(y_orig)       # store actual original data

    if not hist:
        y = signale.tools.smooth(y_orig, window_len=smooth * factor_cope_with_gain)  # , window='flat'
        y_orig = signale.tools.smooth(y_orig, window_len=5. * factor_cope_with_gain)  # , window='flat'
    else:
        y = y_orig

    # generate one more x-point between all existing x-points:________________________________________________
    # x_doublePointNum = numpy.arange(x[0], x[-1]+numpy.diff(x)[0]/2., numpy.diff(x)[0]/2.)
    # KT
    # x_doublePointNum = numpy.linspace(x[0], x[-1], 10000)


    # calculate basic values for the FR distribution y_______________________________________________________
    y_std = numpy.std(y)
    y_mean = numpy.mean(y)
    normalized_y_mean = y_mean/y.max()
    normalized_y_std = y_std/y.max()

    # KT
    if thresh is None:
        threshold = normalized_y_mean + 0*normalized_y_std
    else:
        threshold = thresh
    input_y = y/y.max() - threshold
    indices = numpy.where(input_y < 0)[0]
    input_y += threshold
    input_y[indices] = 0
    input_y *= multi
    # input_y = y
    input_y /= multi
    input_y *= y.max()

    # generate data histogram___________________________________________________________________________________
    if not hist:
        data = numpy.repeat(x, numpy.around(input_y).astype(int))
    else:
        idx = numpy.where(his[1][:-1] < 35)
        new_y = y_orig.copy()
        new_y[idx] = 0
        data = x
        x = his[1][:-1]
        input_y = y_orig
        if thresh > 0:
            y_orig = new_y

    # fit two gaussians_____________________________________________________________________________________
    gmm = mixture.GMM(n_components=2, covariance_type='full', min_covar=0.000001)  # gmm for two components
    # gmm = mixture.GMM(n_components=1, covariance_type='full', min_covar=0.000001)  # gmm for two components

    # gmm = mixture.DPGMM(n_components=2, covariance_type='diag', alpha=.1, min_covar=0.000001)
    gmm.fit(numpy.vstack(data))  #numpy.vstack(data))  #numpy.vstack(data))  # train it!

    # generate one more x-point between all existing x-points:________________________________________________
    # x_doublePointNum = numpy.arange(x[0], x[-1]+numpy.diff(x)[0]/2., numpy.diff(x)[0]/2.)
    x_doublePointNum = numpy.linspace(x[0], x[-1], 10000)


    # global mean1, mean2, std1, std2
    mean1 = gmm.means_[0][0]
    mean2 = gmm.means_[1][0]
    std1 = numpy.sqrt(gmm._get_covars()[0])[0][0]
    std2 = numpy.sqrt(gmm._get_covars()[1])[0][0]

    p0 = [gmm.weights_[0], mean1, std1, gmm.weights_[1], mean2, std2]
    # p0 = [gmm.weights_[0], mean1, std1] #, gmm.weights_[1], mean2, std2]

    if not hist:

        try:
            coeff, var_matrix = scipy.optimize.curve_fit(double_gauss_fkt, x, y_orig, p0=p0,
                                                         bounds=([0, 0, 0, 0, 0, 0],  # Lower bound for p0 parameters
                                                                 [numpy.inf, x.max()*.95, x.max()/3.,  # Upper bound for p0 parameters
                                                                  numpy.inf, x.max()*.95, x.max()/3.]),
                                                         max_nfev=100000000)
            # coeff, var_matrix = scipy.optimize.curve_fit(gauss_fkt, x, y_orig, p0=p0,
            #                                              bounds=([0, 0, 0],  # Lower bound for p0 parameters
            #                                                      [numpy.inf, x.max()*.95, x.max()/3.]),
            #                                              max_nfev=100000000)

        except ValueError:
            try:
            # print 'fitting did not work at first place. I will relieve constraints a bit and try again.'
                coeff, var_matrix = scipy.optimize.curve_fit(double_gauss_fkt, x, y_orig, p0=p0,
                                                             bounds=([0, 0, 0, 0, 0, 0],
                                                                     [numpy.inf, x.max(), x.max()/3.,
                                                                      numpy.inf, x.max(), x.max()/3.]),
                                                             max_nfev=100000000)
            except ValueError:
                try:
                    coeff, var_matrix = scipy.optimize.curve_fit(double_gauss_fkt, x, y_orig, p0=p0,
                                                                 bounds=([0, 0, 0, 0, 0, 0],
                                                                         [numpy.inf, x.max(), x.max(),
                                                                          numpy.inf, x.max(), x.max()]),
                                                                 max_nfev=100000000)
                except ValueError:
                    print 'fitting with bounds did not work! Try without bounds.'
                    coeff, var_matrix = scipy.optimize.curve_fit(double_gauss_fkt, x, y_orig, p0=p0,
                                                                 bounds=([0, 0, 0, 0, 0, 0]),
                                                                 max_nfev=100000000)
    else:
        try:
            coeff, var_matrix = scipy.optimize.curve_fit(double_gauss_fkt, x, y_orig, p0=p0,
                                                         bounds=([0, 0, 5, 0, 0, 5],
                                                                 [y.max(), x.max(), x.max(),
                                                                  y.max(), x.max(), x.max()]),
                                                         max_nfev=100000000)
            # coeff, var_matrix = scipy.optimize.curve_fit(gauss_fkt, x, y_orig, p0=p0,
            #                                              bounds=([0, 0, 5],
            #                                                      [y.max(), x.max(), x.max()]),
            #                                              max_nfev=100000000)
        except ValueError:
            try:
                coeff, var_matrix = scipy.optimize.curve_fit(double_gauss_fkt, x, y_orig, p0=p0,
                                                             bounds=([0, 0, 3, 0, 0, 3],
                                                                     [y.max(), x.max(), numpy.inf,
                                                                      y.max(), x.max(), numpy.inf]),
                                                             max_nfev=100000000)
            except ValueError:
                print 'fitting with bounds did not work! Try without bounds.'
                coeff, var_matrix = scipy.optimize.curve_fit(double_gauss_fkt, x, y_orig, p0=p0, maxfev=100000000)

    # get functions for two fitted gaussians________________________________________________________________
    # print 'Gauss 1 info: '
    gauss1 = gauss_fkt(x_doublePointNum, *coeff[:3])
    # print 'Gauss 2 info: '
    gauss2 = gauss_fkt(x_doublePointNum, *coeff[3:])

    # print 'mean1 = ', mean1, 'mean2 = ', mean2
    # print 's1 = ', std1, 's2 = ', std2

    if amp != None:
        if gauss1.max() > gauss2.max():
            gauss2 = amp*(gauss2/gauss2.max())
        else:
            gauss1 = amp*(gauss1/gauss1.max())

    # normalize for maxima
    max_gauss = numpy.max([gauss1.max(), gauss2.max()])
    index_max_gauss = numpy.argmax([gauss1.max(), gauss2.max()])
    double_gauss = gauss1 + gauss2

    # define plot colors based on which gauss is bigger______
    if index_max_gauss == 0:
        colour = ['r', 'k']
        if plot_data:
            append_gauss_info(x_doublePointNum=x_doublePointNum, gaussS=gauss2, gaussL=gauss1)
    else:
        colour = ['k', 'r']
        if plot_data:
            append_gauss_info(x_doublePointNum=x_doublePointNum, gaussS=gauss1, gaussL=gauss2)

    if plot_data:
        # plot data and gaussians from mixture model
        # sns.set(style="white")
        fig22, ax22 = pl.subplots(2, 1, figsize=(10, 8))
        ax22 = ax22.flatten()

        # move axis up
        space = [0.07, 0.02]
        for num, axi in enumerate(ax22):
            pos1 = axi.get_position()  # get the original position
            pos2 = [pos1.x0, pos1.y0 + space[num],  pos1.width, pos1.height]
            axi.set_position(pos2)

        ax22_pos = list(ax22[0].get_position().bounds)
        ax22_pos[2] *= .8
        ax22[0].set_position(ax22_pos)
        ax22a_pos = list(ax22_pos)
        ax22a_pos[0] += ax22_pos[2] + .03
        ax22a_pos[2] = .2
        ax22a = fig22.add_axes(ax22a_pos)

        ax22[0].axhline(threshold*y.max(), linestyle='-', color=custom_plot.pretty_colors_set2[0], alpha=0.8, zorder=0)
        # ax22.axhline(y_mean, linestyle='-', color=custom_plot.pretty_colors_set2[0], alpha=0.8, zorder=0)
        # ax22.axhspan(y_mean-y_std, y_mean+y_std, facecolor=custom_plot.pretty_colors_set2[0], alpha=0.2, linewidth=False, zorder=0)
        ax22[0].plot(x, y_orig, color=numpy.ones(3)*.66, linewidth=1)
        ax22a.hist(y_orig_dummy, 20, orientation='horizontal', color=numpy.ones(3)*.66, edgecolor='none')

        ax22[0].plot(x, y, 'b')
        ax22[0].plot(x_doublePointNum, gauss1, linewidth=2, color=colour[0])  # gauss1 = small gauss
        ax22[0].plot(x_doublePointNum, gauss2, linewidth=2, color=colour[1])
        ax22[0].plot(x_doublePointNum, double_gauss, linewidth=2, color='g')
        ax22[0].plot(x, input_y, '--', color=numpy.ones(3)*.75)

        ax22[0].set_xlabel('Angle in degrees', fontsize=fz)
        ax22[0].set_ylabel('Count', fontsize=fz)

        # ax22[0].set_xlabel('Position from start point (m)', fontsize=fz)
        # ax22[0].set_ylabel('Firing rate (Hz)', fontsize=fz)

        # if surrogate:
        #     ax22.set_title('Surrogate #' + str(surrogate))

        ax22[0].set_xlim(0, max(x))
        ylim = [0, y_orig_dummy.max()*1.05]
        ax22[0].set_ylim(ylim)
        ax22a.set_ylim(ylim)

        custom_plot.huebschMachen(ax22[0])
        custom_plot.allOff(ax22a)
        axis = ax22[0]
    else:
        axis = False

    # get m-value
    if plot_data:
        f = fig22
    else:
        f = None
    m = m_val(gauss1, gauss2, x_doublePointNum, ax=axis, fig=f)

    if plot_data:
        fig22.text(ax22_pos[0]+ax22_pos[2]*.75, ax22_pos[1]+ax22_pos[3]*.9, 'm: '+str(numpy.round(m, 3)))
        if surrogate == 0:
            return fig22, ax22, m

    else:
        return m

    return fig22, ax22, m


def get_fitted_pf_width_and_maxFR():
    # add pf width and filenames
    perc = 0.1
    info = hickle.load(path+'Summary/MaxFR_doublePeak_info_corrected.hkl')
    Y_large = info['y_large']
    Y_small = info['y_small']
    Y_comined = info['Y_comined']
    X = info['x']
    files = info['used_files']
    direc = info['running_direc']
    g = info['gains']

    GaussPFwidthL = []
    GaussPFwidthS = []
    GaussPFwidthCombi = []
    FR_maxS = []
    FR_maxL = []
    GaussPFboundsL = []
    GaussPFboundsS = []
    GaussPFboundsCombi = []
    namesL = []
    namesS = []
    namesC = []

    for w in numpy.arange(len(Y_large)):
        print w, ' of ', len(Y_large)-1
        if numpy.nanmax(Y_large[w]) == 0:
            pf_width1 = numpy.nan
            pf_bounds1 = numpy.nan
            namesL.append((files[w]).strip('.hkl')+'_'+direc[w]+'_'+g[w]+'_L')
        else:
            cut = numpy.nanargmax(Y_large[w])
            if cut == 0:
                cut = 1
            min_arg1 = numpy.nanargmin(abs(Y_large[w][0:cut]-(numpy.nanmax(Y_large[w])*perc)))

            cut1 = numpy.nanargmax(Y_large[w])
            if cut1 == len(Y_large[w])-1:
                cut1 = len(Y_large[w])-2
            max_arg1 = numpy.nanargmin(abs(Y_large[w][cut1:-1]-(numpy.nanmax(Y_large[w])*perc)))+\
                       numpy.nanargmax(Y_large[w])
            if max_arg1 < 0:
                max_arg1 = 0
            elif min_arg1 < 0:
                min_arg1 = 0
            elif max_arg1 > len(X[w])-1:
                max_arg1 = len(X[w])-1
            elif min_arg1 > len(X[w])-1:
                min_arg1 = len(X[w])-1
            pf_width1 = abs(X[w][max_arg1]-X[w][min_arg1])
            if X[w][max_arg1] > X[w][min_arg1]:
                pf_bounds1 = [X[w][min_arg1], X[w][max_arg1]]
            else:
                pf_bounds1 = [X[w][max_arg1], X[w][min_arg1]]
            namesL.append((files[w]).strip('.hkl')+'_'+direc[w]+'_'+g[w]+'_L')

        if numpy.nanmax(Y_small[w]) == 0:
            pf_width2 = numpy.nan
            pf_bounds2 = numpy.nan
            namesS.append((files[w]).strip('.hkl')+'_'+direc[w]+'_'+g[w]+'_S')
        else:
            cut = numpy.nanargmax(Y_small[w])
            if cut == 0:
                cut = 1
            min_arg2 = numpy.nanargmin(abs(Y_small[w][0:cut]-(numpy.nanmax(Y_small[w])*perc)))

            cut1 = numpy.nanargmax(Y_small[w])
            if cut1 == len(Y_small[w])-1:
                cut1 = len(Y_small[w])-2
            max_arg2 = numpy.nanargmin(abs(Y_small[w][cut1:-1]-(numpy.nanmax(Y_small[w])*perc)))+\
                       numpy.nanargmax(Y_small[w])
            if max_arg2 < 0:
                max_arg2 = 0
            elif min_arg2 < 0:
                min_arg2 = 0
            elif max_arg2 > len(X[w])-1:
                max_arg2 = len(X[w])-1
            elif min_arg2 > len(X[w])-1:
                min_arg2 = len(X[w])-1
            pf_width2 = abs(X[w][max_arg2]-X[w][min_arg2])
            if X[w][max_arg2] > X[w][min_arg2]:
                pf_bounds2 = [X[w][min_arg2], X[w][max_arg2]]
            else:
                pf_bounds2 = [X[w][max_arg2], X[w][min_arg2]]
            namesS.append((files[w]).strip('.hkl')+'_'+direc[w]+'_'+g[w]+'_S')

        if numpy.nanmax(Y_comined[w]) == 0:
            pf_width3 = numpy.nan
            pf_bounds3 = numpy.nan
            namesC.append((files[w]).strip('.hkl')+'_'+direc[w]+'_'+g[w]+'_C')
        else:
            cut = numpy.nanargmax(Y_comined[w])
            if cut == 0:
                cut = 1
            min_arg3 = numpy.nanargmin(abs(Y_comined[w][0:cut]-(numpy.nanmax(Y_comined[w])*perc)))

            cut1 = numpy.nanargmax(Y_comined[w])
            if cut1 == len(Y_comined[w])-1:
                cut1 = len(Y_comined[w])-2
            max_arg3 = numpy.nanargmin(abs(Y_comined[w][cut1:-1]-(numpy.nanmax(Y_comined[w])*perc)))+\
                       numpy.nanargmax(Y_comined[w])
            if max_arg3 < 0:
                max_arg3 = 0
            elif min_arg3 < 0:
                min_arg3 = 0
            elif max_arg3 > len(X[w])-1:
                max_arg3 = len(X[w])-1
            elif min_arg3 > len(X[w])-1:
                min_arg3 = len(X[w])-1
            pf_width3 = abs(X[w][max_arg3]-X[w][min_arg3])
            if X[w][max_arg3] > X[w][min_arg3]:
                pf_bounds3 = [X[w][min_arg3], X[w][max_arg3]]
            else:
                pf_bounds3 = [X[w][max_arg3], X[w][min_arg3]]
            namesC.append((files[w]).strip('.hkl')+'_'+direc[w]+'_'+g[w]+'_C')

        # if numpy.nanmax(Y_large[w]) > numpy.nanmax(Y_small[w]):
        GaussPFwidthL.append(pf_width1)
        GaussPFwidthS.append(pf_width2)
        GaussPFwidthCombi.append(pf_width3)

        GaussPFboundsL.append(pf_bounds1)
        GaussPFboundsS.append(pf_bounds2)
        GaussPFboundsCombi.append(pf_bounds3)

        # use Combi FR of the max x position as FR!
        idxS = numpy.where(numpy.array(X[w]) == info['xMaxGauss_small'][w])[0][0]
        idxL = numpy.where(numpy.array(X[w]) == info['xMaxGauss_large'][w])[0][0]
        FR_maxS.append(Y_comined[w][idxS])
        FR_maxL.append(Y_comined[w][idxL])

    width_FRmax = {'GaussPFwidthL': GaussPFwidthL, 'GaussPFwidthS': GaussPFwidthS,
                   'GaussPFwidthCombi': GaussPFwidthCombi, 'GaussFRmaxL': FR_maxL,
                   'GaussFRmaxS': FR_maxS, 'GaussFRmaxCombi': info['MaxGauss_combi'],
                   'used_files': info['used_files'], 'running_direc': info['running_direc'],
                   'gains': info['gains'], 'double_cell_names': info['double_cell_names'],
                   'GaussPFboundsL': GaussPFboundsL, 'GaussPFboundsS': GaussPFboundsS,
                   'GaussPFboundsCombi': GaussPFboundsCombi, 'namesL': namesL, 'namesS': namesS, 'namesC': namesC}

    hickle.dump(width_FRmax, path+'Summary/Gauss_width_FRmax.hkl', mode='w')


def pie_chart():
    all_cells = 165  # old value = 178
    one_perc = all_cells/100.

    dou_direc = 9
    dou_perc = dou_direc/one_perc
    dou_bidirec = 3
    dou_bi_perc = dou_bidirec/one_perc
    dou_one_gain = 3
    dou_1gain = dou_one_gain/one_perc

    dou_sin_bidirec = 3  # old value = 4
    dou_sin_perc = dou_sin_bidirec/one_perc

    sin_direc = 72  # old value 75
    sin_perc = sin_direc/one_perc
    sin_bidirec = 6  # old value 8
    sin_bi_perc = sin_bidirec/one_perc
    sin_one_gain = 12  # old value 16
    sin_1gain_perc = sin_one_gain/one_perc

    one_gain_active = 57  # old value 60 cells
    one_gain_perc = one_gain_active/one_perc

    # detailed one_gain_active:
    unidirec_1gain_remap = 9                # before 0.66 in gain 0.5 active
    unidirec_1gain_rausmap = 41             # after 0.66 in gain 0.5 active
    unidirec_1gain_rausmap_doppelfeld = 1   # both fields after 0.66 in gain 0.5 active
    bidirec_1gain_rausmap = 6               # (3+3) after 0.66 in gain 0.5 active
    bidirec_1gain_remap_rausmap = 3         # (2+1) rightwards before, leftwards after 0.66 in gain 0.5 active

    double = '#FFFF00'     # yellow
    double_bi = '#F0E68C'  # different yellow
    single = '#00FFFF'     # cyan
    single_bi = '#AFEEEE'  # different cyan
    mix = '#00FF00'  # green
    no = 'w' #custom_plot.grau  #'k'

    fig_pie, ax_pie = pl.subplots(1, 1, figsize=(10, 10))

    pos1 = ax_pie.get_position()  # get the original position
    pos2 = [pos1.x0-.15, pos1.y0-.12,  pos1.width, pos1.height]
    ax_pie.set_position(pos2)

    dec = 1

    wedges, texts = ax_pie.pie([one_gain_perc, dou_perc, dou_bi_perc, dou_sin_perc, sin_bi_perc, sin_perc,
                                sin_1gain_perc, dou_1gain],
                               colors=[no, double, double, mix, single, single, 'k', 'b'], startangle=90,
                               labels=[str(numpy.round(one_gain_active, dec)), str(numpy.round(dou_direc, dec)),
                                       str(numpy.round(dou_bidirec, dec)), str(numpy.round(dou_sin_bidirec)),
                                       str(numpy.round(sin_bidirec, dec)), str(numpy.round(sin_direc, dec))])

    for num, w in enumerate(wedges):
        if num in [2, 5]:
            w.set_linewidth(4)
            w.set_linestyle('dashed')

    lw = 3

    line_s = Line2D([0], [0], linestyle='', marker="s", markersize=10, markerfacecolor=single, markeredgecolor='k', markeredgewidth=.5)
    line_sb = Line2D([0], [0], linestyle='', marker="s", markersize=10, markerfacecolor=single, markeredgecolor='k', markeredgewidth=.5)
    line_d = Line2D([0], [0], linestyle='', marker="s", markersize=10, markerfacecolor=double, markeredgecolor='k', markeredgewidth=.5)
    line_db = Line2D([0], [0], linestyle='', marker="s", markersize=10, markerfacecolor=double, markeredgecolor='k', markeredgewidth=.5)
    line_m = Line2D([0], [0], linestyle='', marker="s", markersize=10, markerfacecolor=mix, markeredgecolor='k', markeredgewidth=.5)
    line_o = Line2D([0], [0], linestyle='', marker="s", markersize=10, markerfacecolor=no, markeredgecolor='k', markeredgewidth=.5)

    ax_pie.legend([line_sb, line_s, line_m, line_d, line_db, line_o], ['Unidirectional single field', 'Bidirectional single field',
                                                                       'Bidirectional cells with single and double field',
                                                                       'Bidirectional double field', 'Unidirectional double field',
                                                                       'Only active at one gain'],
                    numpoints=1, bbox_to_anchor=(1.35, 1.3), fontsize=20)

    print 'Saving figure under '+path+'Summary/pie.pdf'
    fig_pie.savefig(path+'Summary/pie.pdf', format='pdf', bbox_inches='tight')
    pl.show()


def cluster_quality(fz=30):

    sns.set(style="ticks")

    dw = hickle.load(path+'Summary/delta_and_weight_info.hkl')

    filenames = numpy.unique(numpy.array(dw['no_double_cell_files']))
    filenames_double = numpy.unique(numpy.array(dw['double_cell_files']))

    with open('/Users/haasolivia/Documents/'+server+'/dataWork/olivia/hickle/Summary/Cell_overview.csv', 'rb') as f:
        reader = csv.reader(f)
        cell_overview = numpy.array(list(reader))

    lratio = []
    iso_dist = []

    lratio_double = []
    iso_dist_double = []

    for file in filenames:

        if file == '10353_2014-06-16_VR_GCend_linTrack1_GC_TT4_SS_02_PF_info.hkl':
            file = '10353_2014-06-16_VR_Gcend_linTrack1_GC_TT4_SS_02_PF_info.hkl'
        cell_row = numpy.where(numpy.array(cell_overview[:, 0]) == file)[0][0]
        ratio_column = numpy.where(cell_overview[0] == 'L-Ratio')[0][0]
        isodist_column = numpy.where(cell_overview[0] == 'Isolation distance')[0][0]
        lratio.append(float(cell_overview[cell_row, ratio_column]))
        iso_dist.append(float(cell_overview[cell_row, isodist_column]))

    for file in filenames_double:

        cell_row = numpy.where(cell_overview[:, 0] == file)[0][0]
        ratio_column = numpy.where(cell_overview[0] == 'L-Ratio')[0][0]
        isodist_column = numpy.where(cell_overview[0] == 'Isolation distance')[0][0]
        lratio_double.append(float(cell_overview[cell_row, ratio_column]))
        iso_dist_double.append(float(cell_overview[cell_row, isodist_column]))

    lratio = numpy.array(lratio)
    lratio_double = numpy.array(lratio_double)
    iso_dist = numpy.array(iso_dist)
    iso_dist_double = numpy.array(iso_dist_double)

    fig_clust, ax_clust = pl.subplots(2, 1, figsize=(10, 10))
    ax_clust.flatten()
    lw = 3
    lmax = 10
    imax = 50
    binwidth = .5
    binwidth1 = 2
    ax_clust[0].hist(lratio, color='r', histtype='step', linewidth=lw, normed=True,
                              bins=numpy.arange(0, lmax, binwidth))
    ax_clust[0].hist(lratio_double, color='b', histtype='step', linewidth=lw, normed=True,
                               bins=numpy.arange(0, lmax, binwidth))
    print 'L-Ratio one fields n = ', len(lratio)
    print 'L-Ratio two fields n = ', len(lratio_double)
    # sns.kdeplot(lratio, ax=ax_clust[0], color='r')
    # sns.kdeplot(lratio_double, ax=ax_clust[0], color='b')

    ax_clust[1].hist(iso_dist, color='r', histtype='step', linewidth=lw, normed=True,
                              bins=numpy.arange(0, imax, binwidth1))
    ax_clust[1].hist(iso_dist_double, color='b', histtype='step', linewidth=lw, normed=True,
                               bins=numpy.arange(0, imax, binwidth1))
    print 'Iso-Dist one fields n = ', len(iso_dist)
    print 'Iso-Dist two fields n = ', len(iso_dist_double)
    # sns.kdeplot(iso_dist, ax=ax_clust[1], color='r')
    # sns.kdeplot(iso_dist_double, ax=ax_clust[1], color='b')

    # _____________ GETTING UN-NORMALISED HISTOGRAM COUNTS FOR STATS ______________
    hist_l = list(numpy.histogram(lratio, bins=numpy.arange(0, lmax, binwidth))[0])
    hist_l1 = list(numpy.histogram(lratio_double, bins=numpy.arange(0, lmax, binwidth))[0])
    print 'Bin lratio single n = ', len(hist_l)
    print 'Bin lratio double n = ', len(hist_l1)

    hist_i = list(numpy.histogram(iso_dist, bins=numpy.arange(0, imax, binwidth1))[0])
    hist_i1 = list(numpy.histogram(iso_dist_double, bins=numpy.arange(0, imax, binwidth1))[0])
    print 'Bin iso single n = ', len(hist_i)
    print 'Bin iso double n = ', len(hist_i1)
    # _____________________________________________________________________________

    ax_clust[0].set_xlim([0, lmax])
    ax_clust[1].set_xlim([0, imax])

    fz *= 2
    ax_clust[0].set_xlabel('L-ratio', fontsize=fz)
    ax_clust[1].set_xlabel('Isolation distance', fontsize=fz)
    ax_clust[0].set_ylabel('Proportion of cells', fontsize=fz)
    ax_clust[1].set_ylabel('Proportion of cells', fontsize=fz)

    start, end = ax_clust[0].get_ylim()
    steps = end/3.
    start1, end1 = ax_clust[1].get_ylim()
    steps1 = end1/3.

    ax_clust[0].yaxis.set_ticks(numpy.arange(start, end+steps, steps))
    ax_clust[1].yaxis.set_ticks(numpy.arange(start1, end1+steps1, steps1))
    ax_clust[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    ax_clust[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))

    pl.setp(ax_clust[0].xaxis.get_majorticklabels(), size=fz)
    pl.setp(ax_clust[1].xaxis.get_majorticklabels(), size=fz)
    pl.setp(ax_clust[0].yaxis.get_majorticklabels(), size=fz)
    pl.setp(ax_clust[1].yaxis.get_majorticklabels(), size=fz)

    line1 = Line2D([0], [0], linestyle="-", linewidth=lw, color='r')
    line2 = Line2D([0], [0], linestyle="-", linewidth=lw, color='b')

    for ax in [ax_clust[0], ax_clust[1]]:
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    ax_clust[0].legend([line1, line2], ['One field', 'Two fields'],
                        numpoints=1, bbox_to_anchor=(1., 1.0), fontsize=20)

    pos1 = ax_clust[0].get_position()  # get the original position
    pos2 = [pos1.x0, pos1.y0 + .05,  pos1.width, pos1.height]
    ax_clust[0].set_position(pos2)     # set a new position

    zeros_l = []
    zeros_i = []
    for i in numpy.arange(len(hist_l)):
        if hist_l[i] == 0 and hist_l1[i] == 0:
            zeros_l.append(i)

    for j in numpy.arange(len(hist_i)):
        if hist_i[j] == 0 and hist_i1[j] == 0:
            zeros_i.append(j)

    chi_l, p_l, dof, ex = scipy.stats.chi2_contingency(numpy.array([hist_l[:zeros_l[0]], hist_l1[:zeros_l[0]]]))
    chi_i, p_i, dof, ex = scipy.stats.chi2_contingency(numpy.array([hist_i[:zeros_i[0]], hist_i1[:zeros_i[0]]]))

    text_x = [0.7, 0.7]
    text_y = [0.75, 0.27]
    text_dy = [-0.05, -0.05]
    text_dx = [0.014, 0.014]

    frz = 20

    chi = [chi_l, chi_i]
    p = [p_l, p_i]
    alp = [1, 1]

    for c in [0, 1]:
        fig_clust.text(text_x[c], text_y[c], '$\chi^2$= '+str(numpy.round(chi[c], 2)), fontsize=frz, color='k',
                           alpha=alp[c])
        # if p[c] < 0.001:
        #     fig_clust.text(text_x[c]+text_dx[c], text_y[c]+text_dy[c], '$p$ < 0.001', fontsize=frz, color='k',
        #                        alpha=alp[c])
        # else:
        fig_clust.text(text_x[c]+text_dx[c], text_y[c]+text_dy[c], '$p$ = '+str(numpy.round(p[c], 3)),
                           fontsize=frz, color='k', alpha=alp[c])

    print 'Saving figure under '+path+'Summary/cluster_quality.pdf'
    fig_clust.savefig(path+'Summary/cluster_quality.pdf', format='pdf', bbox_inches='tight')

    info = {'lratio': lratio, 'lratio_double': lratio_double, 'iso_dist': iso_dist, 'iso_dist_double': iso_dist_double}

    hickle.dump(info, path+'Summary/lratio_isodist.hkl', mode='w')


def theta_delta_ratio():
    info = hickle.load(path+'Summary/raw_data_info.hkl')
    gains = info['gain']
    theta_mean = info['theta_mean']
    delta_mean = info['delta_mean']

    theta_delta_ratio_gain05 = []
    theta_delta_ratio_gain15 = []

    for i, g in enumerate(gains):
        if g == 0.5:
            if delta_mean[i] > 0:
                theta_delta_ratio_gain05.append(theta_mean[i]/delta_mean[i])
            else:
                theta_delta_ratio_gain05.append(numpy.nan)
        elif g == 1.5:
            if delta_mean[i] > 0:
                theta_delta_ratio_gain15.append(theta_mean[i]/delta_mean[i])
            else:
                theta_delta_ratio_gain15.append(numpy.nan)

    fig_g, ax_g = pl.subplots(2, 1, figsize=(10, 10))
    binwidth = .2
    ax_g = ax_g.flatten()
    lw = 4
    min_g = numpy.nanmin([numpy.nanmin(theta_delta_ratio_gain05), numpy.nanmin(theta_delta_ratio_gain15)])
    max_g = numpy.nanmax([numpy.nanmax(theta_delta_ratio_gain05), numpy.nanmax(theta_delta_ratio_gain15)])

    ax_g[0].hist(theta_delta_ratio_gain05, bins=numpy.arange(min_g, max_g, binwidth), histtype='step', linewidth=lw, color=c_05)
    ax_g[0].hist(theta_delta_ratio_gain15, bins=numpy.arange(min_g, max_g, binwidth), histtype='step', linewidth=lw, color=c_15)

    diff = numpy.array(theta_delta_ratio_gain05)-numpy.array(theta_delta_ratio_gain15)

    ax_g[1].hist(diff, bins=numpy.arange(numpy.nanmin(diff), numpy.nanmax(diff), .05))
                 # histtype='step', linewidth=lw, color='k')

    ax_g[0].set_xlabel('Avg. single run theta - delta ratio')
    ax_g[1].set_xlabel('Avg. single run theta - delta ratio gain difference')

    for i in [0, 1]:
        # Hide the right and top spines
        ax_g[i].spines['right'].set_visible(False)
        ax_g[i].spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax_g[i].yaxis.set_ticks_position('left')
        ax_g[i].xaxis.set_ticks_position('bottom')

    cl1_line = Line2D((0, 1), (0, 0), color=c_05, lw=7)
    cl2_line = Line2D((0, 1), (0, 0), color=c_15, lw=7)
    fig_g.legend((cl1_line, cl2_line), ("Gain 0.5", "Gain 1.5"), numpoints=1, loc='upper right', fontsize=15)

    fig_g.tight_layout()
    fig_g.savefig(path+'Summary/theta_delta_ratio.pdf')

    diff = numpy.array(diff)

    diff_t, diff_p = scipy.stats.ttest_1samp(diff[~numpy.isnan(diff)], 0)
    print 'n_diff', len(~numpy.isnan(diff))
    print 'diff_t', diff_t
    print 'diff_p', diff_p
    print 'numpy.nanmean(diff)', numpy.nanmean(diff)
    print 'numpy.std(diff[~numpy.isnan(diff)])', numpy.std(diff[~numpy.isnan(diff)])

    hickle.dump(numpy.array(theta_delta_ratio_gain05), path+'Summary/theta_delta_ratio_gain05.hkl', mode='w')
    hickle.dump(numpy.array(theta_delta_ratio_gain15), path+'Summary/theta_delta_ratio_gain15.hkl', mode='w')


def phase_precession():
    info = hickle.load(path+'Summary/raw_data_info.hkl')

    # --------------- all slopes are corrected for running direction ---------------
    pooled = info['Pooled_slope_per_m']
    pooled_p = info['Pooled_p']
    sr = info['SR_slope_per_m']
    sr_p = info['SR_p']
    width = info['width']
    names = info['names']
    gains = info['gain']
    prop_idx = info['prop_idx']
    vis_idx = info['vis_idx']
    ca = info['CA_region']
    pv = numpy.array(['' for i in numpy.arange(len(pooled))])
    pv[prop_idx] = 'p'
    pv[vis_idx] = 'v'

    sr_pp = []
    sr_pp_avg = []
    sr_pp_avg_sign = []
    for idx in numpy.arange(len(sr)):
        single_run = sr[idx]*width[idx]/360.
        sr_pp.append(single_run)  # convert to cycles/field
        sr_mean = numpy.nanmean(single_run)
        # if not numpy.isnan(sr_mean):
        sr_pp_avg.append(sr_mean)  # mean cycles/field for one all single runs
        run_sign = []
        for s in numpy.arange(len(single_run)):
            if sr_p[idx][s] < .05:   # works also for nan values! significant runs
                # print single_run[s], sr_p[idx][s]
                run_sign.append(single_run[s])  # significant runs
            elif sr_p[idx][s] > 1:
                print 'p-Value of ',  sr_p[idx][s]
                print 'filename : ', names[idx]
                sys.exit()
            else:
                run_sign.append(numpy.nan)
        if len(run_sign):
            sr_pp_avg_sign.append(numpy.nanmean(numpy.array(run_sign)))

    # sr_pp_avg = numpy.array(sr_pp_avg)
    # sr_pp_avg_sign = numpy.array(sr_pp_avg_sign)

    # --------------- remove nans ------------------------------------

    not_nan_idx = numpy.where(~numpy.isnan(pooled))[0]

    pooled = numpy.array(pooled)[not_nan_idx]
    pooled_p = numpy.array(pooled_p)[not_nan_idx]
    width = numpy.array(width)[not_nan_idx]
    pooled_pp = pooled*width/360.  # convert to cycles/field

    pooled_pp_sign = []
    for ip, p in enumerate(pooled_p):
        if p < .05:   # works also for nan values! significant runs
            pooled_pp_sign.append(pooled_pp[ip])
        else:
            pooled_pp_sign.append(numpy.nan)
    pooled_pp_sign = numpy.array(pooled_pp_sign)

    sr_pp_avg = numpy.array(sr_pp_avg)[not_nan_idx]
    sr_pp_avg_sign = numpy.array(sr_pp_avg_sign)[not_nan_idx]
    names = numpy.array(names)[not_nan_idx]
    gains = numpy.array(gains)[not_nan_idx]
    ca = numpy.array(ca)[not_nan_idx]
    pv = numpy.array(pv)[not_nan_idx]

    idx_05 = numpy.where(gains == 0.5)[0]
    idx_15 = numpy.where(gains == 1.5)[0]

    idx_ca1 = numpy.where(ca == 'CA1')[0]
    idx_ca3 = numpy.where(ca == 'CA3')[0]

    idx_prop = numpy.where(pv == 'p')[0]
    idx_vis = numpy.where(pv == 'v')[0]

    # ------------------

    idx_prop05 = numpy.where(numpy.array(pv)[idx_05] == 'p')[0]
    idx_vis05 = numpy.where(numpy.array(pv)[idx_05] == 'v')[0]
    idx_prop15 = numpy.where(numpy.array(pv)[idx_15] == 'p')[0]
    idx_vis15 = numpy.where(numpy.array(pv)[idx_15] == 'v')[0]

    # idx_all = numpy.arange(len(pooled_pp))

    # --------------- plotting ------------------------------------

    # --------------- colors and parameters ---------

    ca1_c = '#cd863c'
    ca3_c = '#682aa5'
    prop_c = '#4575b4'
    vis_c = '#d73027'

    lw = 4

    # --------------- all ---------

    fig, ax = pl.subplots(3, 1, figsize=(10, 10))
    binwidth = .2
    ax = ax.flatten()
    ax[0].hist(pooled_pp, bins=numpy.arange(numpy.nanmin(pooled_pp), numpy.nanmax(pooled_pp), binwidth))
    ax[1].hist(sr_pp_avg, bins=numpy.arange(numpy.nanmin(sr_pp_avg), numpy.nanmax(sr_pp_avg), binwidth))
    ax[2].hist(sr_pp_avg_sign, bins=numpy.arange(numpy.nanmin(sr_pp_avg_sign), numpy.nanmax(sr_pp_avg_sign), binwidth))

    # dump raw histogram data for all data mixed_______________________

    hickle.dump(pooled_pp, path+'Summary/pooled_pp.hkl', mode='w')
    hickle.dump(pooled_pp_sign, path+'Summary/pooled_pp_sign.hkl', mode='w')
    hickle.dump(sr_pp_avg, path+'Summary/sr_pp_avg.hkl', mode='w')
    hickle.dump(sr_pp_avg_sign, path+'Summary/sr_pp_avg_sign.hkl', mode='w')

    # dump raw histogram data for gain and loc/vis specific data_______

    pp05l = pooled_pp[idx_05][idx_prop05]
    pp_s05l = pooled_pp_sign[idx_05][idx_prop05]
    srpp05l = sr_pp_avg[idx_05][idx_prop05]
    srpp_s05l = sr_pp_avg_sign[idx_05][idx_prop05]

    hickle.dump(pp05l, path+'Summary/pooled_pp_05_loc.hkl', mode='w')
    hickle.dump(pp_s05l, path+'Summary/pooled_pp_sign_05_loc.hkl', mode='w')
    hickle.dump(srpp05l, path+'Summary/sr_pp_avg_05_loc.hkl', mode='w')
    hickle.dump(srpp_s05l, path+'Summary/sr_pp_avg_sign_05_loc.hkl', mode='w')

    pp15l = pooled_pp[idx_15][idx_prop15]
    pp_s15l = pooled_pp_sign[idx_15][idx_prop15]
    srpp15l = sr_pp_avg[idx_15][idx_prop15]
    srpp_s15l = sr_pp_avg_sign[idx_15][idx_prop15]

    hickle.dump(pp15l, path+'Summary/pooled_pp_15_loc.hkl', mode='w')
    hickle.dump(pp_s15l, path+'Summary/pooled_pp_sign_15_loc.hkl', mode='w')
    hickle.dump(srpp15l, path+'Summary/sr_pp_avg_15_loc.hkl', mode='w')
    hickle.dump(srpp_s15l, path+'Summary/sr_pp_avg_sign_15_loc.hkl', mode='w')

    # ---------

    pp05v = pooled_pp[idx_05][idx_vis05]
    pp_s05v = pooled_pp_sign[idx_05][idx_vis05]
    srpp05v = sr_pp_avg[idx_05][idx_vis05]
    srpp_s05v = sr_pp_avg_sign[idx_05][idx_vis05]

    hickle.dump(pp05v, path+'Summary/pooled_pp_05_vis.hkl', mode='w')
    hickle.dump(pp_s05v, path+'Summary/pooled_pp_sign_05_vis.hkl', mode='w')
    hickle.dump(srpp05v, path+'Summary/sr_pp_avg_05_vis.hkl', mode='w')
    hickle.dump(srpp_s05v, path+'Summary/sr_pp_avg_sign_05_vis.hkl', mode='w')

    pp15v = pooled_pp[idx_15][idx_vis15]
    pp_s15v = pooled_pp_sign[idx_15][idx_vis15]
    srpp15v = sr_pp_avg[idx_15][idx_vis15]
    srpp_s15v = sr_pp_avg_sign[idx_15][idx_vis15]

    hickle.dump(pp15v, path+'Summary/pooled_pp_15_vis.hkl', mode='w')
    hickle.dump(pp_s15v, path+'Summary/pooled_pp_sign_15_vis.hkl', mode='w')
    hickle.dump(srpp15v, path+'Summary/sr_pp_avg_15_vis.hkl', mode='w')
    hickle.dump(srpp_s15v, path+'Summary/sr_pp_avg_sign_15_vis.hkl', mode='w')

    # --------------- T-Tests ------------------------------------

    # ----- both gains and locomotor driven ------------

    pooled05l = pp05l[~numpy.isnan(pp05l)]
    pooled15l = pp15l[~numpy.isnan(pp15l)]
    print 'Pooled gain 0.5 loc mean', numpy.nanmean(pooled05l), ' for n = ', numpy.count_nonzero(~numpy.isnan(pooled05l))
    print 'Pooled gain 1.5 loc mean', numpy.nanmean(pooled15l), ' for n = ', numpy.count_nonzero(~numpy.isnan(pooled15l))
    pooled05l = numpy.concatenate((pooled05l, numpy.zeros(len(pooled15l) - len(pooled05l))))
    pooled_gloc_t, pooled_gloc_p = scipy.stats.ttest_rel(pooled05l, pooled15l)
    pooled05_gloc_t, pooled05_gloc_p = scipy.stats.ttest_1samp(pooled05l, 0)
    pooled15_gloc_t, pooled15_gloc_p = scipy.stats.ttest_1samp(pooled15l, 0)

    pooleds05l = pp_s05l[~numpy.isnan(pp_s05l)]
    pooleds15l = pp_s15l[~numpy.isnan(pp_s15l)]
    print 'Pooled sign. gain 0.5 loc mean', numpy.nanmean(pooleds05l), ' for n = ', numpy.count_nonzero(~numpy.isnan(pooleds05l))
    print 'Pooled sign. gain 1.5 loc mean', numpy.nanmean(pooleds15l), ' for n = ', numpy.count_nonzero(~numpy.isnan(pooleds15l))
    pooleds15l = numpy.concatenate((pooleds15l, numpy.zeros(len(pooleds05l) - len(pooleds15l))))
    pooleds_gloc_t, pooleds_gloc_p = scipy.stats.ttest_rel(pooleds05l, pooleds15l)
    pooleds05_gloc_t, pooleds05_gloc_p = scipy.stats.ttest_1samp(pooleds05l, 0)
    pooleds15_gloc_t, pooleds15_gloc_p = scipy.stats.ttest_1samp(pooleds15l, 0)

    sr05l = srpp05l[~numpy.isnan(srpp05l)]
    sr15l = srpp15l[~numpy.isnan(srpp15l)]
    print 'SR gain 0.5 loc mean', numpy.nanmean(sr05l), ' for n = ', numpy.count_nonzero(~numpy.isnan(sr05l))
    print 'SR gain 1.5 loc mean', numpy.nanmean(sr15l), ' for n = ', numpy.count_nonzero(~numpy.isnan(sr15l))
    sr15l = numpy.concatenate((sr15l, numpy.zeros(len(sr05l) - len(sr15l))))
    sr_gloc_t, sr_gloc_p = scipy.stats.ttest_rel(sr05l, sr15l)
    sr05_gloc_t, sr05_gloc_p = scipy.stats.ttest_1samp(sr05l, 0)
    sr15_gloc_t, sr15_gloc_p = scipy.stats.ttest_1samp(sr15l, 0)

    srs05l = srpp_s05l[~numpy.isnan(srpp_s05l)]
    srs15l = srpp_s15l[~numpy.isnan(srpp_s15l)]
    print 'SR sign. gain 0.5 loc mean', numpy.nanmean(srs05l), ' for n = ', numpy.count_nonzero(~numpy.isnan(srs05l))
    print 'SR sign. gain 1.5 loc mean', numpy.nanmean(srs15l), ' for n = ', numpy.count_nonzero(~numpy.isnan(srs15l))
    srs15l = numpy.concatenate((srs15l, numpy.zeros(len(srs05l) - len(srs15l))))
    srs_gloc_t, srs_gloc_p = scipy.stats.ttest_rel(srs05l, srs15l)
    srs05_gloc_t, srs05_gloc_p = scipy.stats.ttest_1samp(srs05l, 0)
    srs15_gloc_t, srs15_gloc_p = scipy.stats.ttest_1samp(srs15l, 0)

    loc = {'pooled_gloc_t': pooled_gloc_t, 'pooled_gloc_p': pooled_gloc_p, 'pooled05_loc_t': pooled05_gloc_t,
           'pooled05_loc_p': pooled05_gloc_p, 'pooled15_loc_t': pooled15_gloc_t, 'pooled15_loc_p': pooled15_gloc_p,
           'pooled_sign_gloc_t': pooleds_gloc_t, 'pooled_sign_gloc_p': pooleds_gloc_p,
           'pooled_sign_05_loc_t': pooleds05_gloc_t, 'pooled_sign_05_loc_p': pooleds05_gloc_p,
           'pooled_sign_15_loc_t': pooleds15_gloc_t, 'pooled_sign_15_loc_p': pooleds15_gloc_p, 'sr_gloc_t': sr_gloc_t,
           'sr_gloc_p': sr_gloc_p, 'sr05_loc_t': sr05_gloc_t, 'sr05_loc_p': sr05_gloc_p, 'sr15_loc_t': sr15_gloc_t,
           'sr15_loc_p': sr15_gloc_p, 'sr_sign_gloc_t': srs_gloc_t, 'sr_sign_gloc_p': srs_gloc_p,
           'sr_sign_05_loc_t': srs05_gloc_t, 'sr_sign_05_loc_p': srs05_gloc_p, 'sr_sign_15_loc_t': srs15_gloc_t,
           'sr_sign_15_loc_p': srs15_gloc_p}

    hickle.dump(loc, path+'Summary/phase_precession_gains_locomotorTuned_t_tests.hkl', mode='w')

    # ----- both gains and visually driven ------------

    pooled05v = pp05v[~numpy.isnan(pp05v)]
    pooled15v = pp15v[~numpy.isnan(pp15v)]
    print 'Pooled gain 0.5 vis mean', numpy.nanmean(pooled05v), ' for n = ', numpy.count_nonzero(~numpy.isnan(pooled05v))
    print 'Pooled gain 1.5 vis mean', numpy.nanmean(pooled15v), ' for n = ', numpy.count_nonzero(~numpy.isnan(pooled15v))
    pooled05v = numpy.concatenate((pooled05v, numpy.zeros(len(pooled15v) - len(pooled05v))))
    pooled_gvis_t, pooled_gvis_p = scipy.stats.ttest_rel(pooled05v, pooled15v)
    pooled05_gvis_t, pooled05_gvis_p = scipy.stats.ttest_1samp(pooled05v, 0)
    pooled15_gvis_t, pooled15_gvis_p = scipy.stats.ttest_1samp(pooled15v, 0)

    pooleds05v = pp_s05v[~numpy.isnan(pp_s05v)]
    pooleds15v = pp_s15v[~numpy.isnan(pp_s15v)]
    print 'Pooled sign. gain 0.5 vis mean', numpy.nanmean(pooleds05v), ' for n = ', numpy.count_nonzero(~numpy.isnan(pooleds05v))
    print 'Pooled sign. gain 1.5 vis mean', numpy.nanmean(pooleds15v), ' for n = ', numpy.count_nonzero(~numpy.isnan(pooleds15v))
    pooleds05v = numpy.concatenate((pooleds05v, numpy.zeros(len(pooleds15v) - len(pooleds05v))))
    pooleds_gvis_t, pooleds_gvis_p = scipy.stats.ttest_rel(pooleds05v, pooleds15v)
    pooleds05_gvis_t, pooleds05_gvis_p = scipy.stats.ttest_1samp(pooleds05v, 0)
    pooleds15_gvis_t, pooleds15_gvis_p = scipy.stats.ttest_1samp(pooleds15v, 0)

    sr05v = srpp05v[~numpy.isnan(srpp05v)]
    sr15v = srpp15v[~numpy.isnan(srpp15v)]
    print 'SR gain 0.5 vis mean', numpy.nanmean(sr05v), ' for n = ', numpy.count_nonzero(~numpy.isnan(sr05v))
    print 'SR gain 1.5 vis mean', numpy.nanmean(sr15v), ' for n = ', numpy.count_nonzero(~numpy.isnan(sr15v))
    sr15v = numpy.concatenate((sr15v, numpy.zeros(len(sr05v) - len(sr15v))))
    sr_gvis_t, sr_gvis_p = scipy.stats.ttest_rel(sr05v, sr15v)
    sr05_gvis_t, sr05_gvis_p = scipy.stats.ttest_1samp(sr05v, 0)
    sr15_gvis_t, sr15_gvis_p = scipy.stats.ttest_1samp(sr15v, 0)

    srs05v = srpp_s05v[~numpy.isnan(srpp_s05v)]
    srs15v = srpp_s15v[~numpy.isnan(srpp_s15v)]
    print 'SR sign. gain 0.5 vis mean', numpy.nanmean(srs05v), ' for n = ', numpy.count_nonzero(~numpy.isnan(srs05v))
    print 'SR sign. gain 1.5 vis mean', numpy.nanmean(srs15v), ' for n = ', numpy.count_nonzero(~numpy.isnan(srs15v))
    srs15v = numpy.concatenate((srs15v, numpy.zeros(len(srs05v) - len(srs15v))))
    srs_gvis_t, srs_gvis_p = scipy.stats.ttest_rel(srs05v, srs15v)
    srs05_gvis_t, srs05_gvis_p = scipy.stats.ttest_1samp(srs05v, 0)
    srs15_gvis_t, srs15_gvis_p = scipy.stats.ttest_1samp(srs15v, 0)

    vis = {'pooled_gvis_t': pooled_gvis_t, 'pooled_gvis_p': pooled_gvis_p, 'pooled05_vis_t': pooled05_gvis_t,
           'pooled05_vis_p': pooled05_gvis_p, 'pooled15_vis_t': pooled15_gvis_t, 'pooled15_vis_p': pooled15_gvis_p,
           'pooled_sign_gvis_t': pooleds_gvis_t, 'pooled_sign_gvis_p': pooleds_gvis_p,
           'pooled_sign_05_vis_t': pooleds05_gvis_t, 'pooled_sign_05_vis_p': pooleds05_gvis_p,
           'pooled_sign_15_vis_t': pooleds15_gvis_t, 'pooled_sign_15_vis_p': pooleds15_gvis_p, 'sr_gvis_t': sr_gvis_t,
           'sr_gvis_p': sr_gvis_p, 'sr05_vis_t': sr05_gvis_t, 'sr05_vis_p': sr05_gvis_p, 'sr15_vis_t': sr15_gvis_t,
           'sr15_vis_p': sr15_gvis_p, 'sr_sign_gvis_t': srs_gvis_t, 'sr_sign_gvis_p': srs_gvis_p,
           'sr_sign_05_vis_t': srs05_gvis_t, 'sr_sign_05_vis_p': srs05_gvis_p, 'sr_sign_15_vis_t': srs15_gvis_t,
           'sr_sign_15_vis_p': srs15_gvis_p}

    hickle.dump(vis, path+'Summary/phase_precession_gains_visuallyTuned_t_tests.hkl', mode='w')

    # --------------- gains ---------

    fig_g, ax_g = pl.subplots(3, 1, figsize=(10, 10))
    binwidth = .2
    ax_g = ax_g.flatten()
    min_g = numpy.nanmin([numpy.nanmin(pooled_pp[idx_05]), numpy.nanmin(pooled_pp[idx_15])])
    max_g = numpy.nanmax([numpy.nanmax(pooled_pp[idx_05]), numpy.nanmax(pooled_pp[idx_15])])
    ax_g[0].hist(pooled_pp[idx_05], bins=numpy.arange(min_g, max_g, binwidth), histtype='step', linewidth=lw, color=c_05)
    ax_g[0].hist(pooled_pp[idx_15], bins=numpy.arange(min_g, max_g, binwidth), histtype='step', linewidth=lw, color=c_15)
    min_g1 = numpy.nanmin([numpy.nanmin(sr_pp_avg[idx_05]), numpy.nanmin(sr_pp_avg[idx_15])])
    max_g1 = numpy.nanmax([numpy.nanmax(sr_pp_avg[idx_05]), numpy.nanmax(sr_pp_avg[idx_15])])
    ax_g[1].hist(sr_pp_avg[idx_05], bins=numpy.arange(min_g1, max_g1, binwidth), histtype='step', linewidth=lw, color=c_05)
    ax_g[1].hist(sr_pp_avg[idx_15], bins=numpy.arange(min_g1, max_g1, binwidth), histtype='step', linewidth=lw, color=c_15)
    min_g2 = numpy.nanmin([numpy.nanmin(sr_pp_avg_sign[idx_05]), numpy.nanmin(sr_pp_avg_sign[idx_15])])
    max_g2 = numpy.nanmax([numpy.nanmax(sr_pp_avg_sign[idx_05]), numpy.nanmax(sr_pp_avg_sign[idx_15])])
    ax_g[2].hist(sr_pp_avg_sign[idx_05], bins=numpy.arange(min_g2, max_g2, binwidth), histtype='step', linewidth=lw, color=c_05)
    ax_g[2].hist(sr_pp_avg_sign[idx_15], bins=numpy.arange(min_g2, max_g2, binwidth), histtype='step', linewidth=lw, color=c_15)

    cl1_line = Line2D((0, 1), (0, 0), color=c_05, lw=7)
    cl2_line = Line2D((0, 1), (0, 0), color=c_15, lw=7)
    fig_g.legend((cl1_line, cl2_line), ("Gain 0.5", "Gain 1.5"), numpoints=1, loc='upper right', fontsize=15)

    # --------------- CA regions ---------

    # fig_ca, ax_ca = pl.subplots(3, 1, figsize=(10, 10))
    # binwidth = .2
    # ax_ca = ax_ca.flatten()
    # min_c = numpy.nanmin([numpy.nanmin(pooled_pp[idx_ca1]), numpy.nanmin(pooled_pp[idx_ca3])])
    # max_c = numpy.nanmax([numpy.nanmax(pooled_pp[idx_ca1]), numpy.nanmax(pooled_pp[idx_ca3])])
    # ax_ca[0].hist(pooled_pp[idx_ca1], bins=numpy.arange(min_c, max_c, binwidth), histtype='step', linewidth=lw, color=ca1_c)
    # ax_ca[0].hist(pooled_pp[idx_ca3], bins=numpy.arange(min_c, max_c, binwidth), histtype='step', linewidth=lw, color=ca3_c)
    # min_c1 = numpy.nanmin([numpy.nanmin(sr_pp_avg[idx_ca1]), numpy.nanmin(sr_pp_avg[idx_ca3])])
    # max_c1 = numpy.nanmax([numpy.nanmax(sr_pp_avg[idx_ca1]), numpy.nanmax(sr_pp_avg[idx_ca3])])
    # ax_ca[1].hist(sr_pp_avg[idx_ca1], bins=numpy.arange(min_c1, max_c1, binwidth), histtype='step', linewidth=lw, color=ca1_c)
    # ax_ca[1].hist(sr_pp_avg[idx_ca3], bins=numpy.arange(min_c1, max_c1, binwidth), histtype='step', linewidth=lw, color=ca3_c)
    # min_c2 = numpy.nanmin([numpy.nanmin(sr_pp_avg_sign[idx_ca1]), numpy.nanmin(sr_pp_avg_sign[idx_ca3])])
    # max_c2 = numpy.nanmax([numpy.nanmax(sr_pp_avg_sign[idx_ca1]), numpy.nanmax(sr_pp_avg_sign[idx_ca3])])
    # ax_ca[2].hist(sr_pp_avg_sign[idx_ca1], bins=numpy.arange(min_c2, max_c2, binwidth), histtype='step', linewidth=lw, color=ca1_c)
    # ax_ca[2].hist(sr_pp_avg_sign[idx_ca3], bins=numpy.arange(min_c2, max_c2, binwidth), histtype='step', linewidth=lw, color=ca3_c)
    #
    # cl1_line = Line2D((0, 1), (0, 0), color=ca1_c, lw=7)
    # cl2_line = Line2D((0, 1), (0, 0), color=ca3_c, lw=7)
    # fig_ca.legend((cl1_line, cl2_line), ("CA1", "CA3"), numpoints=1, loc='upper right', fontsize=15)

    # --------------- prop vis ---------

    fig_pv, ax_pv = pl.subplots(3, 1, figsize=(10, 10))
    binwidth = .2
    ax_pv = ax_pv.flatten()
    min_pv = numpy.nanmin([numpy.nanmin(pooled_pp[idx_prop]), numpy.nanmin(pooled_pp[idx_vis])])
    max_pv = numpy.nanmax([numpy.nanmax(pooled_pp[idx_prop]), numpy.nanmax(pooled_pp[idx_vis])])
    ax_pv[0].hist(pooled_pp[idx_prop], bins=numpy.arange(min_pv, max_pv, binwidth), histtype='step', linewidth=lw, color=prop_c)
    ax_pv[0].hist(pooled_pp[idx_vis], bins=numpy.arange(min_pv, max_pv, binwidth), histtype='step', linewidth=lw, color=vis_c)
    min_pv1 = numpy.nanmin([numpy.nanmin(sr_pp_avg[idx_prop]), numpy.nanmin(sr_pp_avg[idx_vis])])
    max_pv1 = numpy.nanmax([numpy.nanmax(sr_pp_avg[idx_prop]), numpy.nanmax(sr_pp_avg[idx_vis])])
    ax_pv[1].hist(sr_pp_avg[idx_prop], bins=numpy.arange(min_pv1, max_pv1, binwidth), histtype='step', linewidth=lw, color=prop_c)
    ax_pv[1].hist(sr_pp_avg[idx_vis], bins=numpy.arange(min_pv1, max_pv1, binwidth), histtype='step', linewidth=lw, color=vis_c)
    min_pv2 = numpy.nanmin([numpy.nanmin(sr_pp_avg_sign[idx_prop]), numpy.nanmin(sr_pp_avg_sign[idx_vis])])
    max_pv2 = numpy.nanmax([numpy.nanmax(sr_pp_avg_sign[idx_prop]), numpy.nanmax(sr_pp_avg_sign[idx_vis])])
    ax_pv[2].hist(sr_pp_avg_sign[idx_prop], bins=numpy.arange(min_pv2, max_pv2, binwidth), histtype='step', linewidth=lw, color=prop_c)
    ax_pv[2].hist(sr_pp_avg_sign[idx_vis], bins=numpy.arange(min_pv2, max_pv2, binwidth), histtype='step', linewidth=lw, color=vis_c)

    cl1_line = Line2D((0, 1), (0, 0), color=prop_c, lw=7)
    cl2_line = Line2D((0, 1), (0, 0), color=vis_c, lw=7)
    fig_pv.legend((cl1_line, cl2_line), ("Locomotion tuned", "Visual tuned"), numpoints=1, loc='upper right', fontsize=15)

    # for a in [ax, ax_g, ax_ca, ax_pv]:
    for a in [ax, ax_g, ax_pv]:
        a[0].set_xlabel('Mean slopes (cyc/field)')
        a[1].set_xlabel('Avg. single run slopes (cyc/field)')
        a[2].set_xlabel('Avg. sign. run slopes (cyc/field)')

        for i in [0, 1, 2]:
            # Hide the right and top spines
            a[i].spines['right'].set_visible(False)
            a[i].spines['top'].set_visible(False)

            # Only show ticks on the left and bottom spines
            a[i].yaxis.set_ticks_position('left')
            a[i].xaxis.set_ticks_position('bottom')

    # --------------- Saving Figures ------------------------------------

    print 'Saving Phase Precession plots unter '+path+'Summary/phase_precession...'
    fig_names = ['phase_precession.pdf', 'phase_precession_gains.pdf', 'phase_precession_CA.pdf', 'phase_precession_vis_loc.pdf']
    # for num, f in enumerate([fig, fig_g, fig_ca, fig_pv]):
    for num, f in enumerate([fig, fig_g, fig_pv]):
        f.tight_layout()
        f.savefig(path+'Summary/'+fig_names[num])

    # --------------- T-Tests ------------------------------------

    # --------------- all ---------

    # pooled_all_t, pooled_all_p = scipy.stats.ttest_rel(pooled_pp, numpy.zeros(len(pooled_pp)))
    # samples = numpy.random.normal(loc=0.0, scale=numpy.nanstd(pooled_pp), size=len(pooled_pp))
    # pooled_all_t, pooled_all_p = scipy.stats.ttest_ind(pooled_pp, samples)

    pooled_all_t, pooled_all_p = scipy.stats.ttest_1samp(pooled_pp[~numpy.isnan(pooled_pp)], 0)
    W, P = scipy.stats.shapiro(~numpy.isnan(pooled_pp))
    print 'pooled_all mean', numpy.nanmean(pooled_pp), ' for n = ', numpy.count_nonzero(~numpy.isnan(pooled_pp)), \
        ' sign diff from normal distr p = ', P
    avSR_all_t, avSR_all_p = scipy.stats.ttest_1samp(sr_pp_avg[~numpy.isnan(sr_pp_avg)], 0)
    W, P1 = scipy.stats.shapiro(~numpy.isnan(sr_pp_avg))
    print 'avg. pooled_all mean', numpy.nanmean(sr_pp_avg), ' for n = ', numpy.count_nonzero(~numpy.isnan(sr_pp_avg)), \
        ' sign diff from normal distr p = ', P1
    avSRs_all_t, avSRs_all_p = scipy.stats.ttest_1samp(sr_pp_avg_sign[~numpy.isnan(sr_pp_avg_sign)], 0)
    W, P2 = scipy.stats.shapiro(~numpy.isnan(sr_pp_avg_sign))
    print 'avg. sign. pooled_all mean', numpy.nanmean(sr_pp_avg_sign), ' for n = ', numpy.count_nonzero(~numpy.isnan(sr_pp_avg_sign)), \
        ' sign diff from normal distr p = ', P2

    # --------------- gains ---------

    pooled_pp_05 = pooled_pp[idx_05][~numpy.isnan(pooled_pp[idx_05])]
    pooled_pp_15 = pooled_pp[idx_15][~numpy.isnan(pooled_pp[idx_15])]
    W, P3 = scipy.stats.shapiro(~numpy.isnan(pooled_pp_05))
    print 'pooled_gain 05 mean', numpy.nanmean(pooled_pp_05), ' for n = ', numpy.count_nonzero(~numpy.isnan(pooled_pp_05)), \
        ' sign diff from normal distr p = ', P3
    W, P4 = scipy.stats.shapiro(~numpy.isnan(pooled_pp_15))
    print 'pooled_gain 15 mean', numpy.nanmean(pooled_pp_15), ' for n = ', numpy.count_nonzero(~numpy.isnan(pooled_pp_15)), \
        ' sign diff from normal distr p = ', P4
    pooled_pp_05 = numpy.concatenate((pooled_pp_05, numpy.zeros(len(pooled_pp_15) - len(pooled_pp_05))))
    # lili = statsmodels.stats.lilliefors(pooled_pp_05)
    # print 'LILI = ', lili

    pooled_g_t, pooled_g_p = scipy.stats.ttest_rel(pooled_pp_05, pooled_pp_15)
    pooled05_g_t, pooled05_g_p = scipy.stats.ttest_1samp(pooled_pp_05, 0)
    pooled15_g_t, pooled15_g_p = scipy.stats.ttest_1samp(pooled_pp_15, 0)

    avSR_g_05 = sr_pp_avg[idx_05][~numpy.isnan(sr_pp_avg[idx_05])]
    avSR_g_15 = sr_pp_avg[idx_15][~numpy.isnan(sr_pp_avg[idx_15])]
    W, P5 = scipy.stats.shapiro(~numpy.isnan(avSR_g_05))
    print 'avg. gain 05 mean', numpy.nanmean(avSR_g_05), ' for n = ', numpy.count_nonzero(~numpy.isnan(avSR_g_05)), \
        ' sign diff from normal distr p = ', P5
    W, P6 = scipy.stats.shapiro(~numpy.isnan(avSR_g_15))
    print 'avg. gain 15 mean', numpy.nanmean(avSR_g_15), ' for n = ', numpy.count_nonzero(~numpy.isnan(avSR_g_15)), \
        ' sign diff from normal distr p = ', P6
    avSR_g_15 = numpy.concatenate((avSR_g_15, numpy.zeros(len(avSR_g_05) - len(avSR_g_15))))
    avSR_g_t, avSR_g_p = scipy.stats.ttest_rel(avSR_g_05, avSR_g_15)
    avSR05_g_t, avSR05_g_p = scipy.stats.ttest_1samp(avSR_g_05, 0)
    avSR15_g_t, avSR15_g_p = scipy.stats.ttest_1samp(avSR_g_15, 0)

    avSRs_g_05 = sr_pp_avg_sign[idx_05][~numpy.isnan(sr_pp_avg_sign[idx_05])]
    avSRs_g_15 = sr_pp_avg_sign[idx_15][~numpy.isnan(sr_pp_avg_sign[idx_15])]
    W, P7 = scipy.stats.shapiro(~numpy.isnan(avSRs_g_05))
    print 'avg. sign. gain 05 mean', numpy.nanmean(avSRs_g_05), ' for n = ', numpy.count_nonzero(~numpy.isnan(avSRs_g_05)), \
        ' sign diff from normal distr p = ', P7
    W, P8 = scipy.stats.shapiro(~numpy.isnan(avSRs_g_15))
    print 'avg. sign. gain 15 mean', numpy.nanmean(avSRs_g_15), ' for n = ', numpy.count_nonzero(~numpy.isnan(avSRs_g_15)), \
        ' sign diff from normal distr p = ', P8
    avSRs_g_15 = numpy.concatenate((avSRs_g_15, numpy.zeros(len(avSRs_g_05) - len(avSRs_g_15))))
    avSRs_g_t, avSRs_g_p = scipy.stats.ttest_rel(avSRs_g_05, avSRs_g_15)
    avSR05s_g_t, avSR05s_g_p = scipy.stats.ttest_1samp(avSRs_g_05, 0)
    avSR15s_g_t, avSR15s_g_p = scipy.stats.ttest_1samp(avSRs_g_15, 0)

    # --------------- CA regions ---------

    # pooled_pp_ca1 = pooled_pp[idx_ca1][~numpy.isnan(pooled_pp[idx_ca1])]
    # pooled_pp_ca3 = pooled_pp[idx_ca3][~numpy.isnan(pooled_pp[idx_ca3])]
    # print 'pooled ca1 mean', numpy.nanmean(pooled_pp_ca1)
    # print 'pooled ca3 mean', numpy.nanmean(pooled_pp_ca3)
    # pooled_pp_ca1 = numpy.concatenate((pooled_pp_ca1, numpy.zeros(len(pooled_pp_ca3) - len(pooled_pp_ca1))))
    # pooled_ca_t, pooled_ca_p = scipy.stats.ttest_rel(pooled_pp_ca1, pooled_pp_ca3)
    # pooled_ca1_t, pooled_ca1_p = scipy.stats.ttest_1samp(pooled_pp_ca1, 0)
    # pooled_ca3_t, pooled_ca3_p = scipy.stats.ttest_1samp(pooled_pp_ca3, 0)
    #
    # avSR_ca1 = sr_pp_avg[idx_ca1][~numpy.isnan(sr_pp_avg[idx_ca1])]
    # avSR_ca3 = sr_pp_avg[idx_ca3][~numpy.isnan(sr_pp_avg[idx_ca3])]
    # print 'avg. ca1 mean', numpy.nanmean(avSR_ca1)
    # print 'avg. ca3 mean', numpy.nanmean(avSR_ca3)
    # avSR_ca1 = numpy.concatenate((avSR_ca1, numpy.zeros(len(avSR_ca3) - len(avSR_ca1))))
    # avSR_ca_t, avSR_ca_p = scipy.stats.ttest_rel(avSR_ca1, avSR_ca3)
    # avSR_ca1_t, avSR_ca1_p = scipy.stats.ttest_1samp(avSR_ca1, 0)
    # avSR_ca3_t, avSR_ca3_p = scipy.stats.ttest_1samp(avSR_ca3, 0)
    #
    # avSRs_ca1 = sr_pp_avg_sign[idx_ca1][~numpy.isnan(sr_pp_avg_sign[idx_ca1])]
    # avSRs_ca3 = sr_pp_avg_sign[idx_ca3][~numpy.isnan(sr_pp_avg_sign[idx_ca3])]
    # print 'avg. sign. ca1 mean', numpy.nanmean(avSRs_ca1)
    # print 'avg. sign. ca3 mean', numpy.nanmean(avSRs_ca3)
    # avSRs_ca1 = numpy.concatenate((avSRs_ca1, numpy.zeros(len(avSRs_ca3) - len(avSRs_ca1))))
    # avSRs_ca_t, avSRs_ca_p = scipy.stats.ttest_rel(avSRs_ca1, avSRs_ca3)
    # avSRs_ca1_t, avSRs_ca1_p = scipy.stats.ttest_1samp(avSRs_ca1, 0)
    # avSRs_ca3_t, avSRs_ca3_p = scipy.stats.ttest_1samp(avSRs_ca3, 0)

    # --------------- prop vis ---------

    pooled_pp_prop = pooled_pp[idx_prop][~numpy.isnan(pooled_pp[idx_prop])]
    pooled_pp_vis = pooled_pp[idx_vis][~numpy.isnan(pooled_pp[idx_vis])]
    W, P9 = scipy.stats.shapiro(~numpy.isnan(pooled_pp_prop))
    print 'pooled loc mean', numpy.nanmean(pooled_pp_prop), ' for n = ', numpy.count_nonzero(~numpy.isnan(pooled_pp_prop)), \
        ' sign diff from normal distr p = ', P9
    W, P10 = scipy.stats.shapiro(~numpy.isnan(pooled_pp_vis))
    W, P101 = scipy.stats.mstats.normaltest(~numpy.isnan(pooled_pp_vis))
    print 'pooled vis mean', numpy.nanmean(pooled_pp_vis), ' for n = ', numpy.count_nonzero(~numpy.isnan(pooled_pp_vis)), \
        ' sign diff from normal distr p = ', P10, P101
    pooled_pp_vis = numpy.concatenate((pooled_pp_vis, numpy.zeros(len(pooled_pp_prop) - len(pooled_pp_vis))))
    pooled_pv_t, pooled_pv_p = scipy.stats.ttest_rel(pooled_pp_prop, pooled_pp_vis)
    pooled_prop_t, pooled_prop_p = scipy.stats.ttest_1samp(pooled_pp_prop, 0)
    pooled_vis_t, pooled_vis_p = scipy.stats.ttest_1samp(pooled_pp_vis, 0)

    avSR_prop = sr_pp_avg[idx_prop][~numpy.isnan(sr_pp_avg[idx_prop])]
    avSR_vis = sr_pp_avg[idx_vis][~numpy.isnan(sr_pp_avg[idx_vis])]
    W, P11 = scipy.stats.shapiro(~numpy.isnan(avSR_prop))
    print 'avg. loc mean', numpy.nanmean(avSR_prop), ' for n = ', numpy.count_nonzero(~numpy.isnan(avSR_prop)), \
        ' sign diff from normal distr p = ', P11
    W, P12 = scipy.stats.shapiro(~numpy.isnan(avSR_vis))
    print 'avg. vis mean', numpy.nanmean(avSR_vis), ' for n = ', numpy.count_nonzero(~numpy.isnan(avSR_vis)), \
        ' sign diff from normal distr p = ', P12
    avSR_vis = numpy.concatenate((avSR_vis, numpy.zeros(len(avSR_prop) - len(avSR_vis))))
    avSR_pv_t, avSR_pv_p = scipy.stats.ttest_rel(avSR_prop, avSR_vis)
    avSR_prop_t, avSR_prop_p = scipy.stats.ttest_1samp(avSR_prop, 0)
    avSR_vis_t, avSR_vis_p = scipy.stats.ttest_1samp(avSR_vis, 0)

    avSRs_prop = sr_pp_avg_sign[idx_prop][~numpy.isnan(sr_pp_avg_sign[idx_prop])]
    avSRs_vis = sr_pp_avg_sign[idx_vis][~numpy.isnan(sr_pp_avg_sign[idx_vis])]
    W, P13 = scipy.stats.shapiro(~numpy.isnan(avSRs_prop))
    print 'avg. sign. loc mean', numpy.nanmean(avSRs_prop), ' for n = ', numpy.count_nonzero(~numpy.isnan(avSRs_prop)), \
        ' sign diff from normal distr p = ', P13
    W, P14 = scipy.stats.shapiro(~numpy.isnan(avSRs_vis))
    print 'avg. sign. vis mean', numpy.nanmean(avSRs_vis), ' for n = ', numpy.count_nonzero(~numpy.isnan(avSRs_vis)), \
        ' sign diff from normal distr p = ', P14
    avSRs_vis = numpy.concatenate((avSRs_vis, numpy.zeros(len(avSRs_prop) - len(avSRs_vis))))
    avSRs_pv_t, avSRs_pv_p = scipy.stats.ttest_rel(avSRs_prop, avSRs_vis)
    avSRs_prop_t, avSRs_prop_p = scipy.stats.ttest_1samp(avSRs_prop, 0)
    avSRs_vis_t, avSRs_vis_p = scipy.stats.ttest_1samp(avSRs_vis, 0)

    info_t_test = {'all_pooled_vs_zero_t': pooled_all_t, 'all_pooled_vs_zero_p': pooled_all_p,
                   'all_avg_SR_vs_zero_t': avSR_all_t, 'all_avg_SR_vs_zero_p': avSR_all_p,
                   'all_avg_sign_SR_vs_zero_t': avSRs_all_t, 'all_avg_sign_SR_vs_zero_p': avSRs_all_p,

                   'gains_pooled_t': pooled_g_t, 'gains_pooled_p': pooled_g_p,
                   'gain0.5_pooled_vs_zero_t': pooled05_g_t, 'gain0.5_pooled_vs_zero_p': pooled05_g_p,
                   'gain1.5_pooled_vs_zero_t': pooled15_g_t, 'gain1.5_pooled_vs_zero_p': pooled15_g_p,
                   'gains_avg_SR_t': avSR_g_t, 'gains_avg_SR_p': avSR_g_p,
                   'gain0.5_avg_SR_vs_zero_t': avSR05_g_t, 'gain0.5_avg_SR_vs_zero_p': avSR05_g_p,
                   'gain1.5_avg_SR_vs_zero_t': avSR15_g_t, 'gain1.5_avg_SR_vs_zero_p': avSR15_g_p,
                   'gains_avg_sign_SR_t': avSRs_g_t, 'gains_avg_sign_SR_p': avSRs_g_p,
                   'gain0.5_avg_sign_SR_vs_zero_t': avSR05s_g_t, 'gain0.5_avg_sign_SR_vs_zero_p': avSR05s_g_p,
                   'gain1.5_avg_sign_SR_vs_zero_t': avSR15s_g_t, 'gain1.5_avg_sign_SR_vs_zero_p': avSR15s_g_p,

                   # 'CA_pooled_t': pooled_ca_t, 'CA_pooled_p': pooled_ca_p,
                   # 'CA1_pooled_vs_zero_t': pooled_ca1_t, 'CA1_pooled_vs_zero_p': pooled_ca1_p,
                   # 'CA3_pooled_vs_zero_t': pooled_ca3_t, 'CA3_pooled_vs_zero_p': pooled_ca3_p,
                   # 'CA_avg_SR_t': avSR_ca_t, 'CA_avg_SR_p': avSR_ca_p,
                   # 'CA1_avg_SR_vs_zero_t': avSR_ca1_t, 'CA1_avg_SR_vs_zero_p': avSR_ca1_p,
                   # 'CA3_avg_SR_vs_zero_t': avSR_ca3_t, 'CA3_avg_SR_vs_zero_p': avSR_ca3_p,
                   # 'CA_avg_sign_SR_t': avSRs_ca_t, 'CA_avg_sign_SR_p': avSRs_ca_p,
                   # 'CA1_avg_sign_SR_vs_zero_t': avSRs_ca1_t, 'CA1_avg_sign_SR_vs_zero_p': avSRs_ca1_p,
                   # 'CA3_avg_sign_SR_vs_zero_t': avSRs_ca3_t, 'CA3_avg_sign_SR_vs_zero_p': avSRs_ca3_p,

                   'loc_vis_pooled_t': pooled_pv_t, 'loc_vis_pooled_p': pooled_pv_p,
                   'locomotion_pooled_vs_zero_t': pooled_prop_t, 'locomotion_pooled_vs_zero_p': pooled_prop_p,
                   'visual_pooled_vs_zero_t': pooled_vis_t, 'visual_pooled_vs_zero_p': pooled_vis_p,
                   'loc_vis_avg_SR_t': avSR_pv_t, 'loc_vis_avg_SR_p': avSR_pv_p,
                   'locomotion_avg_SR_vs_zero_t': avSR_prop_t, 'locomotion_avg_SR_vs_zero_p': avSR_prop_p,
                   'visual_avg_SR_vs_zero_t': avSR_vis_t, 'visual_avg_SR_vs_zero_p': avSR_vis_p,
                   'loc_vis_avg_sign_SR_t': avSRs_pv_t, 'loc_vis_avg_sign_SR_p': avSRs_pv_p,
                   'locomotion_avg_sign_SR_vs_zero_t': avSRs_prop_t, 'locomotion_avg_sign_SR_vs_zero_p': avSRs_prop_p,
                   'visual_avg_sign_SR_vs_zero_t': avSRs_vis_t, 'visual_avg_sign_SR_vs_zero_p': avSRs_vis_p}

    # --------------- Dumping hkl with t-test values ------------------------------------

    print 'Dumping phase_precession t-test values under '+path+'Summary/phase_precession_t_tests.hkl'

    hickle.dump(info_t_test, path+'Summary/phase_precession_t_tests.hkl', mode='w')


def gauss_data_corr(xlimfr=None, xlimwidth=None, xlimpos=None, fz=20, figsize=6, scale=1.8, msize=10, double=False,
                    rausmap=False, new_slope=False):

    info = hickle.load(path+'Summary/Gauss_width_FRmax.hkl')
    # info1 = hickle.load(path+'Summary/MaxFR_doublePeak_info.hkl')
    info1 = hickle.load(path+'Summary/MaxFR_doublePeak_info_corrected.hkl')
    prop_vis = hickle.load(path+'Summary/prop_vis_rem_filenames.hkl')
    dw = hickle.load(path+'Summary/delta_and_weight_info.hkl')

    # filenames = hickle.load(path+'Summary/used_filenames.hkl')
    # data_rundirec = hickle.load(path+'Summary/running_directions.hkl')
    filenames = numpy.array(dw['no_double_cell_files'])
    data_rundirec = numpy.array(dw['no_double_cell_direc'])
    u = numpy.array(dw['used_files'])
    ca_all = numpy.array(dw['CA_region'])

    # load slope2.5_test dataset information ----------------------------------------
    if new_slope:
        files = []
        rund = []
        cas = []
        new_files = []
        for fi in os.listdir(path+'slope2.5_test/'):
            if fi.endswith('.hkl'):
                new_files.append(fi)

        for i, f in enumerate(filenames):
            if f in numpy.array(new_files):
                files.append(f)
                rund.append(data_rundirec[i])
                cas.append(ca_all[i])
        filenames = numpy.array(files)
        data_rundirec = numpy.array(rund)
        ca_all = numpy.array(ca_all)

    # load rausmapping information ----------------------------------------
    if rausmap:
        info_r = hickle.load(path+'Summary/rausmap_info.hkl')  # created in Cell_heatmap_overview.py
        filenames = numpy.array(info_r['rausmapping_files'])
        data_rundirec = numpy.array(info_r['rausmapping_direc'])

    # files = [(filenames[i]).split('PF')[0]+data_rundirec[i] for i in numpy.arange(len(filenames))]

    prop_files = prop_vis['prop_files']
    vis_files = prop_vis['vis_files']
    rem_files = prop_vis['rem_files']
    gauss_widths_pre = numpy.array(info['GaussPFwidthCombi'])
    gauss_maxi_pre = numpy.array(info['GaussFRmaxCombi'])
    gauss_xpos = numpy.array(info1['xMaxGauss_combi'])

    double_cells = numpy.array(dw['double_cell_files'])
    double_rundirec = numpy.array(dw['double_cell_direc'])
    gauss_widthsL = numpy.array(info['GaussPFwidthL'])
    gauss_widthsS = numpy.array(info['GaussPFwidthS'])
    gauss_pf_boundsL = numpy.array(info['GaussPFboundsL'])
    gauss_pf_boundsS = numpy.array(info['GaussPFboundsS'])
    gauss_maxL = numpy.array(info['GaussFRmaxL'])
    gauss_maxS = numpy.array(info['GaussFRmaxS'])
    gauss_xposL = numpy.array(info1['xMaxGauss_large'])
    gauss_xposS = numpy.array(info1['xMaxGauss_small'])
    namesL = numpy.array(info['namesL'])
    namesS = numpy.array(info['namesS'])

    # prepare double cell names

    dnames = []
    for i, d in enumerate(double_cells):
        dnames.append(d.split('.hkl')[0]+'_'+double_rundirec[i])
    dnames = numpy.array(dnames)

    gauss_files = info['used_files']
    gauss_gains = info['gains']
    gauss_rundirec = info['running_direc']
    gauss_names = [gauss_files[i].split('.hkl')[0]+'_'+gauss_rundirec[i]+'_gain_'+gauss_gains[i][0]+gauss_gains[i][2]\
                   for i in numpy.arange(len(gauss_files))]

    if double:
        filenames = double_cells
        data_rundirec = double_rundirec

    # remove 'bad' data

    good_idx = []
    files = []
    for index, v in enumerate(filenames):
        f = v.split('.hkl')[0]+'_'+data_rundirec[index]
        if f not in bad or rausmap:
            good_idx.append(index)
            files.append(f)
        else:
            print 'removing bad data: ', v

    # global files

    gauss_idx = []
    data_widths = []
    data_widths_norm = []
    data_maxi = []
    data_maxi_norm = []
    data_2dmaxi = []
    data_2dmaxi_norm = []
    data_xpos = []
    data_names = []
    prop_idx = []
    vis_idx = []
    rem_idx = []
    gains = []
    ca = []
    prop_idxL = []
    prop_idxS = []
    vis_idxL = []
    vis_idxS = []
    rem_idxL = []
    rem_idxS = []
    names = []
    spatial_info = []
    spatial_info_norm = []
    SR_avg_spike_count = []
    SR_spike_count = []
    SR_phi0 = []
    SR_p = []
    SR_p_norm = []
    SR_slope = []
    SR_slope_norm = []
    Pooled_slope = []
    Pooled_slope_norm = []
    Pooled_p = []
    Pooled_p_norm = []
    theta_mean = []
    delta_mean = []
    SR_deltax = []
    FR_x = []
    FR_y = []

    # data_files = hkl_files[::2]
    # data_rundirec = run_direc[::2]

    xremBorder = numpy.array([4./3, 8./3])
    yremBorder = numpy.array([2./3, 0])
    xpropvis = numpy.array([0, 4./3])
    ypropvis = numpy.array([0, -2./3])

    remBorder_V15_1 = (3*xremBorder[0] - 6*yremBorder[0])/4.
    remBorder_V05_1 = yremBorder[0] + remBorder_V15_1
    remBorder_V15_2 = (3*xremBorder[1] - 6*yremBorder[1])/4.
    remBorder_V05_2 = yremBorder[1] + remBorder_V15_2

    propvis_V15_1 = (3*xpropvis[0] - 6*ypropvis[0])/4.
    propvis_V05_1 = ypropvis[0] + propvis_V15_1
    propvis_V15_2 = (3*xpropvis[1] - 6*ypropvis[1])/4.
    propvis_V05_2 = ypropvis[1] + propvis_V15_2

    prop_patchV = numpy.array([[propvis_V05_1, propvis_V15_1], [0, 2], [propvis_V05_2, propvis_V15_2]])
    vis_patchV = numpy.array([[propvis_V05_1, propvis_V15_1], [propvis_V05_2, propvis_V15_2],
                              [remBorder_V05_2, remBorder_V15_2], [remBorder_V05_1, remBorder_V15_1]])
    rem_patchV = numpy.array([[remBorder_V05_1, remBorder_V15_1], [remBorder_V05_2, remBorder_V15_2], [2, 0]])

    prop = matplotlib.path.Path(prop_patchV)
    vis = matplotlib.path.Path(vis_patchV)
    rem = matplotlib.path.Path(rem_patchV)

    if not rausmap:
        num = ['0.5', '1.5']
    else:
        num = ['0.5']

    for i, file in enumerate(filenames):
        if not rausmap:
            filedirec = file.split('.hkl')[0]+'_'+data_rundirec[i]
            ca_idx = numpy.where(numpy.array(u) == filedirec)[0]
            if len(ca_idx) == 1:
                ca_name = ca_all[ca_idx[0]]
            else:
                print 'several ca_idx found!'
                sys.exit()
        if i in good_idx:
            if new_slope:
                f = hickle.load(path+'slope2.5_test/'+file)
                f_norm = hickle.load(path+file.split('.hkl')[0]+'_normalised.hkl')
            else:
                try:
                    f = hickle.load(path+file)
                except IOError:
                    f = hickle.load(path+'cells_not_used_79/'+file)
                try:
                    f_norm = hickle.load(path+file.split('.hkl')[0]+'_normalised.hkl')
                except IOError:
                    f_norm = hickle.load(path+'cells_not_used_79/'+file.split('.hkl')[0]+'_normalised.hkl')
            for gain in num:
                if gain == '0.5':
                    g = '05'
                    gain1 = '15'
                else:
                    g = '15'
                    gain1 = '05'

                name = file.split('.hkl')[0]+'_'+data_rundirec[i]+'_gain_'+g

                nameA = file.split('PF')[0]+data_rundirec[i]+'.hkl'
                name_norm = file.split('.hkl')[0]+'_normalised_'+data_rundirec[i]+'_gain_'+g
                name1 = file.split('.hkl')[0]+'_'+data_rundirec[i]+'_gain_'+gain1
                name_norm1 = file.split('.hkl')[0]+'_normalised_'+data_rundirec[i]+'_gain_'+gain1

                idx = numpy.where(numpy.array(gauss_names) == name)[0]

                if len(idx) and not rausmap:
                    gauss_idx.append(idx[0])
                    names.append(name)
                    data_names.append(name)

                elif rausmap:
                    names.append(name)

                else:
                    print 'name', name
                    print 'Problem in line 1633'
                    sys.exit()

                gains.append(float(gain))
                if not rausmap:
                    ca.append(ca_name)

                # if none of the gains in visual or prop coordinates are countes as a double cell, then the cell is
                # counted as a single field cell in that specific running direction
                # if name not in double_cells and name_norm not in double_cells and name1 not in double_cells and name_norm1 \
                #         not in double_cells:
                if not double:
                    occupancy_prob_vis = f['occupancy_probability_ysum_'+data_rundirec[i]+'Runs_gain_'+gain]
                    # cut off first and last two FR values which are due to smoothing!
                    diff = len(f[data_rundirec[i]+'FR_x_y_gain_'+gain][1]) - len(occupancy_prob_vis)
                    # print 'Before  -  len(occupancy_prob_vis) = ', len(occupancy_prob_vis), len(f[data_rundirec[i]+'FR_x_y_gain_'+gain][1])
                    # print diff
                    if not diff % 2 and diff > 0:  # even number
                        ab1 = diff/2
                        ab2 = ab1
                        FR_vis = f[data_rundirec[i]+'FR_x_y_gain_'+gain][1][ab1:-ab2]
                    elif diff == 0:
                        FR_vis = f[data_rundirec[i]+'FR_x_y_gain_'+gain][1]
                    else:
                        if diff > 0:
                            # print 'vis not an even number!'
                            ab1 = int(diff)/2
                            ab2 = ab1+1
                            FR_vis = f[data_rundirec[i]+'FR_x_y_gain_'+gain][1][ab1:-ab2]
                        else:
                            # print 'vis smaller zero!'
                            ab1 = int(-diff)/2
                            ab2 = ab1+1
                            occupancy_prob_vis = occupancy_prob_vis[ab1:-ab2]
                            FR_vis = f[data_rundirec[i]+'FR_x_y_gain_'+gain][1]
                            # sys.exit()

                    # FR_vis = f[data_rundirec[i]+'FR_2d_x_y_gain_'+gain][1][2:-2]
                    #
                    # if
                    # print 'After  -  len(occupancy_prob_vis) = ', len(occupancy_prob_vis), 'len(FR_vis) = ', len(FR_vis)
                    FRfrac_vis = FR_vis/numpy.nansum(occupancy_prob_vis*FR_vis)
                    log2_vis = numpy.log2(FRfrac_vis)
                    sinfo = numpy.nansum(occupancy_prob_vis * FRfrac_vis * log2_vis)
                    spatial_info.append(sinfo)

                    occupancy_prob = f_norm['occupancy_probability_ysum_'+data_rundirec[i]+'Runs_gain_'+gain]
                    # cut off first and last two FR values which are due to smoothing!
                    diff_n = len(f_norm[data_rundirec[i]+'FR_x_y_gain_'+gain][1]) - len(occupancy_prob)
                    # print 'Before  -  len(occupancy_prob) = ', len(occupancy_prob), len(f_norm[data_rundirec[i]+'FR_x_y_gain_'+gain][1])
                    # print diff_n
                    if not diff_n % 2 and diff_n > 0:  # even number
                        abn1 = diff_n/2
                        abn2 = abn1
                        FR = f_norm[data_rundirec[i]+'FR_x_y_gain_'+gain][1][abn1:-abn2]
                    elif diff_n == 0:
                        FR = f_norm[data_rundirec[i]+'FR_x_y_gain_'+gain][1]
                    else:
                        if diff_n > 0:
                            # print 'not an even number!'
                            abn1 = int(diff_n)/2
                            abn2 = abn1+1
                            FR = f_norm[data_rundirec[i]+'FR_x_y_gain_'+gain][1][abn1:-abn2]
                        else:
                            # print 'smaller zero!'
                            abn1 = int(-diff_n)/2
                            abn2 = abn1+1
                            occupancy_prob = occupancy_prob[abn1:-abn2]
                            FR = f_norm[data_rundirec[i]+'FR_x_y_gain_'+gain][1]
                            # sys.exit()

                    # FR = f_norm[data_rundirec[i]+'FR_2d_x_y_gain_'+gain][1][2:-2]
                    # print 'After  -  len(occupancy_prob) = ', len(occupancy_prob), 'len(FR) = ', len(FR)
                    FRfrac = FR/numpy.nansum(occupancy_prob*FR)
                    log2_n = numpy.log2(FRfrac)
                    sinfo_n = numpy.nansum(occupancy_prob * FRfrac * log2_n)
                    spatial_info_norm.append(sinfo_n)

                    SR_avg_spike_count.append(f['spike_count_perRun_pf_sum_'+data_rundirec[i]+'Runs_gain_'+gain])

                    if not rausmap:
                        SR_spike_count.append(f['spike_count_SR_'+data_rundirec[i]+'Runs_gain_'+gain])

                    phi0 = f['SR_phaseFit_phi0_'+data_rundirec[i]+'Runs_gain_'+gain]
                    SR_phi0.append(phi0)
                    slope = f['SR_phaseFit_aopt_'+data_rundirec[i]+'Runs_gain_'+gain]
                    slope_norm = f_norm['SR_phaseFit_aopt_'+data_rundirec[i]+'Runs_gain_'+gain]
                    pooled_slope = f['Pooled_phaseFit_aopt_'+data_rundirec[i]+'Runs_gain_'+gain]
                    pooled_slope_norm = f_norm['Pooled_phaseFit_aopt_'+data_rundirec[i]+'Runs_gain_'+gain]
                    pooled_p = f['Pooled_phaseFit_p_'+data_rundirec[i]+'Runs_gain_'+gain]
                    pooled_p_norm = f_norm['Pooled_phaseFit_p_'+data_rundirec[i]+'Runs_gain_'+gain]

                    if len(pooled_slope):
                        pooled_slope = pooled_slope[0]
                    else:
                        pooled_slope = numpy.nan

                    if len(pooled_slope_norm):
                        pooled_slope_norm = pooled_slope_norm[0]
                    else:
                        pooled_slope_norm = numpy.nan

                    if data_rundirec[i] == 'left':
                        slope = numpy.array(slope)*-1
                        slope_norm = numpy.array(slope_norm)*-1
                        pooled_slope = numpy.array(pooled_slope)*-1
                        pooled_slope_norm = numpy.array(pooled_slope_norm)*-1

                    SR_slope.append(slope)
                    SR_slope_norm.append(slope_norm)
                    Pooled_slope.append(pooled_slope)
                    Pooled_slope_norm.append(pooled_slope_norm)

                    if not rausmap:
                        theta_mean.append(numpy.nanmean(f['theta_power_gain'+str(gain)[0]+str(gain)[2]]))
                        delta_mean.append(numpy.nanmean(f['delta_power_gain'+str(gain)[0]+str(gain)[2]]))

                    if len(pooled_p):
                        pooled_p = pooled_p[0]
                    else:
                        pooled_p = numpy.nan
                    if len(pooled_p_norm):
                        pooled_p_norm = pooled_p_norm[0]
                    else:
                        pooled_p_norm = numpy.nan

                    Pooled_p.append(pooled_p)
                    Pooled_p_norm.append(pooled_p_norm)
                    dx = []
                    for a in numpy.arange(len(slope)):
                        dx.append(abs((phi0[a]-numpy.nanmean(phi0))/slope[a]))
                    SR_deltax.append(dx)
                    p = f['SR_phaseFit_p_'+data_rundirec[i]+'Runs_gain_'+gain]
                    p_norm = f_norm['SR_phaseFit_p_'+data_rundirec[i]+'Runs_gain_'+gain]

                    for p1 in numpy.arange(len(p)):

                        # for p2 in numpy.arange(len(p[p1])):
                        if type(p[p1]) == numpy.float64:
                            # can be larger than 1 when error function is negative (p = 1 - (-error) = 1+error)
                            if p[p1] > 1:
                                p[p1] = 2-p[p1]  # (p = 2-(1+error) = 1 - error)

                        else:
                            print type(p[p1]), p[p1]
                            print 'type(p[p1]) is not float!'
                            sys.exit()

                    for p1 in numpy.arange(len(p_norm)):

                        # for p2 in numpy.arange(len(p[p1])):
                        if type(p_norm[p1]) == numpy.float64:
                            # can be larger than 1 when error function is negative (p = 1 - (-error) = 1+error)
                            if p_norm[p1] > 1:
                                p_norm[p1] = 2-p_norm[p1]  # (p = 2-(1+error) = 1 - error)

                        else:
                            print type(p[p1]), p[p1]
                            print 'type(p[p1]) is not float!'
                            sys.exit()

                    SR_p.append(p)
                    SR_p_norm.append(p_norm)

                    FR_x.append(f[data_rundirec[i]+'FR_x_y_gain_'+gain][0])
                    FR_y.append(f[data_rundirec[i]+'FR_x_y_gain_'+gain][1])
                    data_widths.append(f['pf_width_'+data_rundirec[i]+'Runs_gain_'+gain])
                    data_widths_norm.append(f_norm['pf_width_'+data_rundirec[i]+'Runs_gain_'+gain])
                    data_maxi.append(f['xMaxFRySuminPF_MaxFRySuminPF_xCMySuminPF_'+data_rundirec[i]+'Runs_gain_'+gain][1])
                    data_maxi_norm.append(f_norm['xMaxFRySuminPF_MaxFRySuminPF_xCMySuminPF_'+data_rundirec[i]+'Runs_gain_'+gain][1])
                    data_2dmaxi.append(f['xMaxFRinPF_MaxFRinPF_xCMinPF_'+data_rundirec[i]+'Runs_gain_'+gain][1])
                    data_2dmaxi_norm.append(f_norm['xMaxFRinPF_MaxFRinPF_xCMinPF_'+data_rundirec[i]+'Runs_gain_'+gain][1])
                    pos = f['xMaxFRySuminPF_MaxFRySuminPF_xCMySuminPF_'+data_rundirec[i]+'Runs_gain_'+gain][0]

                    if data_rundirec[i] == 'left':
                        pos = abs(pos-2)
                    data_xpos.append(pos)

                    if name.split('_gain')[0] in numpy.array(prop_files) or rausmap:
                        prop_idx.append(len(data_widths)-1)
                    elif name.split('_gain')[0] in numpy.array(vis_files):
                        vis_idx.append(len(data_widths)-1)
                    elif name.split('_gain')[0] in numpy.array(rem_files):
                        rem_idx.append(len(data_widths)-1)
                    else:
                        print name
                        print 'not in vis, prop or rem files!'
                        # sys.exit()

                elif gain == '1.5':
                    idxx = gauss_idx[-2:]
                    [xL, yL] = gauss_xposL[idxx]
                    [xS, yS] = gauss_xposS[idxx]
                    if prop.contains_point([xL, yL]):
                        prop_idxL.append(idxx[0])
                        prop_idxL.append(idxx[1])
                    elif vis.contains_point([xL, yL]):
                        vis_idxL.append(idxx[0])
                        vis_idxL.append(idxx[1])
                    elif rem.contains_point([xL, yL]):
                        rem_idxL.append(idxx[0])
                        rem_idxL.append(idxx[1])
                    else:
                        sys.exit('No fitting patch found for large x = '+str(xL)+', y = '+str(yL))

                    if prop.contains_point([xS, yS]):
                        prop_idxS.append(idxx[0])
                        prop_idxS.append(idxx[1])
                    elif vis.contains_point([xS, yS]):
                        vis_idxS.append(idxx[0])
                        vis_idxS.append(idxx[1])
                    elif rem.contains_point([xS, yS]):
                        rem_idxS.append(idxx[0])
                        rem_idxS.append(idxx[1])
                    else:
                        sys.exit('No fitting patch found for small x = '+str(xS)+', y = '+str(yS))

    if not double and not rausmap and len(gauss_idx) != len(data_names):
        print 'Not the same amount of data and fitted cells!'
        sys.exit()

    if not double:
        if not rausmap:
            gauss_widths = gauss_widths_pre[gauss_idx]
            gauss_maxi = gauss_maxi_pre[gauss_idx]
            gauss_xpos = gauss_xpos[gauss_idx]
            ca = numpy.array(ca)
        data_widths = numpy.array(data_widths)
        data_widths_norm = numpy.array(data_widths_norm)
        data_maxi = numpy.array(data_maxi)
        data_maxi_norm = numpy.array(data_maxi_norm)
        data_2dmaxi = numpy.array(data_2dmaxi)
        data_2dmaxi_norm = numpy.array(data_2dmaxi_norm)
        data_xpos = numpy.array(data_xpos)
        gains = numpy.array(gains)
        theta_mean = numpy.array(theta_mean)
        delta_mean = numpy.array(delta_mean)

        # make arrays same length:
        make_subarrays_equal_long(SR_phi0)
        make_subarrays_equal_long(SR_slope)
        make_subarrays_equal_long(SR_slope_norm)
        make_subarrays_equal_long(SR_deltax)
        make_subarrays_equal_long(FR_x)
        make_subarrays_equal_long(FR_y)
        make_subarrays_equal_long(SR_p)
        make_subarrays_equal_long(SR_p_norm)
        if not rausmap:
            make_subarrays_equal_long(SR_spike_count)

        if not rausmap:
            info = {'x_pos': data_xpos, 'width': data_widths, 'fr_1d': data_maxi, 'fr_2d': data_2dmaxi, 'gain': gains,
                    'prop_idx': prop_idx, 'vis_idx': vis_idx, 'rem_idx': rem_idx, 'names': names,
                    'gauss_xpos': gauss_xpos, 'gauss_width': gauss_widths, 'gauss_fr_1d': gauss_maxi,
                    'SR_avg_spike_count': SR_avg_spike_count, 'SR_spike_count': SR_spike_count, 'SR_phi0': SR_phi0,
                    'SR_slope_per_m': SR_slope, 'SR_dx': SR_deltax, 'FR_x': FR_x, 'FR_y': FR_y, 'SR_p': SR_p,
                    'Pooled_slope_per_m': Pooled_slope, 'Pooled_p': Pooled_p, 'spatial_info': spatial_info,
                    'spatial_info_norm': spatial_info_norm, 'CA_region': ca, 'theta_mean': theta_mean,
                    'delta_mean': delta_mean, 'width_norm': data_widths_norm, 'fr_1d_norm': data_maxi_norm,
                    'data_2dmaxi_norm': data_2dmaxi_norm, 'SR_p_norm': SR_p_norm, 'SR_slope_per_m_norm': SR_slope_norm,
                    'Pooled_slope_per_m_norm': Pooled_slope_norm, 'Pooled_p_norm': Pooled_p_norm}
        else:
            info = {'x_pos': data_xpos, 'width': data_widths, 'fr_1d': data_maxi, 'fr_2d': data_2dmaxi, 'gain': gains,
                    'prop_idx': prop_idx, 'vis_idx': vis_idx, 'rem_idx': rem_idx, 'names': names,
                    'SR_avg_spike_count': SR_avg_spike_count, 'SR_phi0': SR_phi0,
                    'SR_slope_per_m': SR_slope, 'SR_dx': SR_deltax, 'FR_x': FR_x, 'FR_y': FR_y, 'SR_p': SR_p,
                    'Pooled_slope_per_m': Pooled_slope, 'Pooled_p': Pooled_p, 'spatial_info': spatial_info,
                    'spatial_info_norm': spatial_info_norm}
    else:
        gauss_widthsL = gauss_widthsL[gauss_idx]
        gauss_widthsS = gauss_widthsS[gauss_idx]
        gauss_pf_boundsL = gauss_pf_boundsL[gauss_idx]
        gauss_pf_boundsS = gauss_pf_boundsS[gauss_idx]
        gauss_maxL = gauss_maxL[gauss_idx]
        gauss_maxS = gauss_maxS[gauss_idx]
        gauss_xposL = gauss_xposL[gauss_idx]
        gauss_xposS = gauss_xposS[gauss_idx]
        namesL = namesL[gauss_idx]
        namesS = namesS[gauss_idx]

        gauss_widths = numpy.concatenate((gauss_widthsL, gauss_widthsS))
        gauss_maxi = numpy.concatenate((gauss_maxL, gauss_maxS))
        gauss_xpos = numpy.concatenate((gauss_xposL, gauss_xposS))
        gains = gains + gains
        gauss_pf_bounds = numpy.concatenate((gauss_pf_boundsL, gauss_pf_boundsS))
        double_names = numpy.concatenate((namesL, namesS))

        prop_idxL = [numpy.where(numpy.array(gauss_idx) == p)[0][0] for p in prop_idxL]
        prop_idxS = [numpy.where(numpy.array(gauss_idx) == p)[0][0] for p in prop_idxS]
        vis_idxL = [numpy.where(numpy.array(gauss_idx) == v)[0][0] for v in vis_idxL]
        vis_idxS = [numpy.where(numpy.array(gauss_idx) == v)[0][0] for v in vis_idxS]
        rem_idxL = [numpy.where(numpy.array(gauss_idx) == r)[0][0] for r in rem_idxL]
        rem_idxS = [numpy.where(numpy.array(gauss_idx) == r)[0][0] for r in rem_idxS]

        prop_idx = numpy.concatenate((numpy.array(prop_idxL), numpy.array(prop_idxS)+len(gauss_widthsL)))
        vis_idx = numpy.concatenate((numpy.array(vis_idxL), numpy.array(vis_idxS)+len(gauss_widthsL)))
        rem_idx = numpy.concatenate((numpy.array(rem_idxL), numpy.array(rem_idxS)+len(gauss_widthsL)))

        info = {'gain': gains, 'prop_idx': prop_idx, 'vis_idx': vis_idx, 'rem_idx': rem_idx,
                'gauss_xpos': gauss_xpos, 'gauss_width': gauss_widths, 'gauss_fr_1d': gauss_maxi,
                'names': double_names, 'PF_boundaries': gauss_pf_bounds}

    if double:
        hkl_end = '_double.hkl'
    elif rausmap:
        hkl_end = '_rausmap.hkl'
    else:
        hkl_end = '.hkl'

    hickle.dump(info, path+'Summary/raw_data_info'+hkl_end, mode='w')

    # saving data arrays:
    # single_field_width_max_data_vs_fits = {'gauss_widths': gauss_widths, 'data_widths': data_widths,
    #                                        'gauss_maxi': gauss_maxi, 'data_maxi': data_maxi,
    #                                        'gauss_xpos': gauss_xpos, 'data_xpos': data_xpos,
    #                                        'data_names': data_names}

    # hickle.dump(single_field_width_max_data_vs_fits, path+'Summary/single_field_width_max_data_vs_fits.hkl', mode='w')

    # ------------------------------------------------------

    # make sns.jointplot via jointgrid fuer beide verteilungen (gauss_widths vs data_widths and gauss_maxi vs data_maxi)

    if not double and not rausmap:
        end = '.pdf'

        sns.set(style="ticks", font_scale=scale)

        J_maxi = sns.jointplot(data_maxi, gauss_maxi, size=figsize, ratio=9, color='r', stat_func=scipy.stats.spearmanr, xlim=xlimfr,
                               ylim=xlimfr, marginal_kws={'bins': 15, 'kde': False})

        J_maxi.ax_joint.plot([0, 13], [0, 13], color='r')

        print 'n-maxi = ', len(data_maxi)
        #set x and y labels
        J_maxi.set_axis_labels('Firing rate maximum of data (Hz)', 'Firing rate maximum of gauss fit (Hz)', fontsize=fz)

        # saving figure
        print 'Saving figure under '+path+'Summary/Corr_FRmax'+end
        pl.savefig(path+'Summary/Corr_FRmax'+end, format='pdf', bbox_inches='tight')

        # ------------------------------------------------------

        J_width = sns.jointplot(data_widths, gauss_widths, size=figsize, ratio=9, color='b', stat_func=scipy.stats.spearmanr, xlim=xlimwidth,
                                ylim=xlimwidth, marginal_kws={'bins': 15, 'kde': False})

        J_width.ax_joint.plot([0, 2.1], [0, 2.1], color='b')

        print 'n-width = ', len(data_widths)

        #set x and y labels
        J_width.set_axis_labels('Place field width of data (m)', 'Place field width of gauss fit (m)', fontsize=fz)

        # saving figure
        print 'Saving figure under '+path+'Summary/Corr_PFwidth'+end
        pl.savefig(path+'Summary/Corr_PFwidth'+end, format='pdf', bbox_inches='tight')

        # ------------------------------------------------------

        J_xpos = sns.jointplot(data_xpos, gauss_xpos, size=figsize, ratio=9, color='g', stat_func=scipy.stats.spearmanr, xlim=xlimpos,
                                ylim=xlimpos, marginal_kws={'bins': 15, 'kde': False})

        J_xpos.ax_joint.plot([0, 2.1], [0, 2.1], color='g')

        print 'n-field center = ', len(data_xpos)

        #set x and y labels
        J_xpos.set_axis_labels('Place field center of data (m)', 'Place field center of gauss fit (m)', fontsize=fz)

        # saving figure
        print 'Saving figure under '+path+'Summary/Corr_PFcenter'+end
        pl.savefig(path+'Summary/Corr_PFcenter'+end, format='pdf', bbox_inches='tight')

        # ______________ PLOT DATA WIDTH AND DATA FR AGAINS XPOS OF THE PLACE FIELD _____________

        fig_wfr, ax_wfr = pl.subplots(2, 1, figsize=(10, 10))
        ax_wfr = ax_wfr.flatten()

        prop_color = '#4575b4'
        vis_color = '#d73027'

        ax_wfr[0].plot(numpy.array(data_xpos)[prop_idx], numpy.array(data_widths)[prop_idx], 'o', color=prop_color)
        ax_wfr[0].plot(numpy.array(data_xpos)[vis_idx], numpy.array(data_widths)[vis_idx], 'o', color=vis_color)
        ax_wfr[0].set_xlabel('Virtual place field position (m)')
        ax_wfr[0].set_ylabel('Place field width (m)')

        ax_wfr[1].plot(numpy.array(data_xpos)[prop_idx], numpy.array(data_maxi)[prop_idx], 'o', color=prop_color)
        ax_wfr[1].plot(numpy.array(data_xpos)[vis_idx], numpy.array(data_maxi)[vis_idx], 'o', color=vis_color)
        ax_wfr[1].set_xlabel('Virtual place field position (m)')
        ax_wfr[1].set_ylabel('Maximal 1D firing rate (Hz)')

        print 'Saving figure under '+path+'Summary/tv_pos_width_fr'+end
        fig_wfr.savefig(path+'Summary/tv_pos_width_fr'+end, format='pdf', bbox_inches='tight')


def merge_hkl():
    m = hickle.load(path+'/Summary/MaxFR_doublePeak_info.hkl')
    mo = hickle.load(path+'/Summary/MaxFR_doublePeak_info_outliers.hkl')

    # p = hickle.load(path+'/Summary/Plot_cumulative_info.hkl')
    # po = hickle.load(path+'/Summary/Plot_cumulative_info_outliers.hkl')
    #
    # g = [k, p]
    # go = [ko, po]

    # for l in [0, 1]:
    #     m = g[l]
    #     mo = go[l]

    for file in [field_switch[0], adapt_lower_bound[0]]:

        idxo = [i for i in numpy.arange(len(mo['used_files'])) if (mo['used_files'][i]).startswith(file.split('info')[0])]
        do = numpy.array(mo['running_direc'])[idxo[0]]

        idx = [i for i in numpy.arange(len(m['used_files'])) if (m['used_files'][i]).startswith(file.split('info')[0])
               and m['running_direc'][i] == do]

        for k in m.keys():
            if len(m[k]) == len(m['used_files']) and len(idx) == len(idxo):
                for j in numpy.arange(len(idx)):
                    # print 'correct ', type(m[k][idx[j]]), isinstance(m[k][idx[j]], numpy.ndarray), m[k][idx[j]]
                    # print 'incorrect', type(mo[k][idxo[j]]), isinstance(m[k][idx[j]], numpy.ndarray), mo[k][idxo[j]]
                    if isinstance(m[k][idx[j]], numpy.ndarray):
                        if len(m[k][idx[j]]) >= len(mo[k][idxo[j]]):
                            mo[k][idxo[j]] = numpy.append(mo[k][idxo[j]],
                                                          numpy.repeat(numpy.nan, len(m[k][idx[j]]) - len(mo[k][idxo[j]])))
                        else:
                            print len(m[k][idx[j]]), len(mo[k][idxo[j]])
                            sys.exit('Outliers array is longer than original!')

                    m[k][idx[j]] = mo[k][idxo[j]]

    hickle.dump(m, path+'Summary/MaxFR_doublePeak_info_corrected.hkl', mode='w')


def calc_peaks(thresh=thresh):
    counter = 0
    surrogate_repeat = 1000
    Su_border95 = []
    double_cells = []
    M = []
    M_plot = numpy.array([])
    M_data = []
    used_files = []
    running_direc = []
    gains = []
    # hkl_files = ['10823_2015-07-03_VR_GCend_linTrack1_TT3_SS_05_PF_info.hkl']
    hkl_files = [field_switch[0].split('info')[0]+'info.hkl', field_switch[0].split('info')[0]+'info.hkl',
                 adapt_lower_bound[0].split('info')[0]+'info.hkl', adapt_lower_bound[0].split('info')[0]+'info.hkl']
    run_direc = [field_switch[0].split('info')[1][1:], field_switch[0].split('info')[1][1:],
                 adapt_lower_bound[0].split('info')[1][1:], adapt_lower_bound[0].split('info')[1][1:]]
    extras_gain = ['1.5', '1.5', '1.5', '1.5']
    extras_amp = [.72, .72, None, None]
    extras_thresh = [thresh, thresh, .21, .21]

    for i, file in enumerate(hkl_files):
        a = hickle.load(path+file)  #'10353_2014-06-17_VR_GCend_linTrack1_GC_TT3_SS_07_PF_info_normalised.hkl')  #file
        amp = None  # predefine Amplitude of small gaussian
        for gain in ['0.5', '1.5']:
            print 'counter: ', counter, ' out of ', len(hkl_files)*2-1
            counter += 1

            if gain == extras_gain[i]:
                amp = extras_amp[i]
                thresh = extras_thresh[i]

            if i%2 != 0:
                used_files.append(file.split('.hkl')[0]+'_normalised.hkl')
            else:
                used_files.append(file)

            running_direc.append(run_direc[i])
            gains.append(gain)

            fr = a[run_direc[i]+'FR_x_y_gain_'+gain]
            x = fr[0]  # virtual places
            y = fr[1]  # 1d firing rates used for 1d 2-gaussian-fit

            if i%2 != 0:
                x /= float(gain)  # calculating normalised places
                file_end = '_normalised_'
            else:
                file_end = '_'

            orig_data_x.append(x)
            orig_data_y.append(y)

            # correct x-axis for leftwards runs -- they were always saved as a FR array from traj.xlim[0] to
            # traj.xlim[1], which goes for x from [0 .. e.g. 2] no matter what the running direction of the animal was!
            # For leftward runs spikes at e.g. x=2 would be at the beginning of the run for the animal, therefore need
            # to be corrected to be x=0.
            if run_direc[i] == 'left':  # for leftward runs plot abolute x-value from start position

                # sys.exit()
                vis_track_length = 2.

                if i%2 != 0:  #if file.endswith('normalised.hkl'):
                    start = vis_track_length/float(gain)
                else:
                    start = vis_track_length
                x = abs(x-start)

            m_values = []

            bad_m_data_value = 0

            for su in numpy.arange(surrogate_repeat+1):  # numpy.arange(surrogate_repeat+1):


                # generate surrogate data________________________________________________________________________________

                # if su < 3:
                #     plot_data = True

                if su == 0:
                    y_dummy = y
                    plot_data = True

                    fig22, ax22, m = fit_gaussians(x=x, y_orig=y_dummy, surrogate=su, plot_data=plot_data,
                                                   thresh=thresh, amp=amp)

                elif not bad_m_data_value:
                    surrogate_x = numpy.random.permutation(range(x.size))
                    y_dummy = y[surrogate_x]
                    plot_data = False

                    m = fit_gaussians(x=x, y_orig=y_dummy, surrogate=su, plot_data=plot_data)

                if m < 0.005 or numpy.isnan(m):
                    m = numpy.nan
                m_values.append(m)

                if su == 0 and numpy.isnan(m):
                    bad_m_data_value = 1

            m_data = m_values[0]
            M_data.append(m_data)

            if not bad_m_data_value:
                m = numpy.array(m_values[1:])
                m = m[~numpy.isnan(m)]
                M.append(m)
            else:
                M.append(numpy.array([numpy.nan]))


            # m, m_data, good, fig22, ax22 = fit_gaussians_etc(x=x, y=y, surrogate_repeat=surrogate_repeat, gain=gain,
            #                                                  run_direc=run_direc[i], file=file, savefig=True)
            # M.append(numpy.array(m))
            # M_plot = numpy.append(M_plot, m)
            # M_data.append(m_data[0])

            if gain == '0.5':
                g = '05'

            else:
                g = '15'

            # file_info = file.split('.hkl')[0]+'_'+run_direc[i]+'_gain_'+g
            # double_cells.append(file_info)

            # if 3 derivatives == 0 are found then plot the surrogate dataset and actual data value
            if numpy.isnan(m_data):  #good == 0:
                Su_border95.append(numpy.nan)
                cumulative_x.append(numpy.nan)
                cumulative_y.append(numpy.nan)
                cumulative_95perc_index.append(numpy.nan)
                real_data_in_cumulative_x.append(numpy.nan)

                ax22[1].axis('off')

                print 'Saving figure at ', path+'Double_Peaks/m_bad/'+file.split('.hkl')[0]+file_end+\
                                           run_direc[i]+'_gain_'+g+'_surrogateVSdata.pdf'
                fig22.savefig(path+'Double_Peaks/m_bad/'+file.split('.hkl')[0]+file_end+
                              run_direc[i]+'_gain_'+g+'_surrogateVSdata.pdf', format='pdf')
            else:   #if good == 1:
                # fig01, ax01 = pl.subplots()
                matplotlib.pyplot.sca(ax22[1])
                pl.xticks(rotation=70)

                ax22[1].spines['top'].set_visible(False)
                ax22[1].spines['right'].set_visible(False)

                x1 = numpy.sort(m)
                y1 = numpy.array(range(len(m)))/float(len(m))

                min_y = numpy.nanargmin(abs(y1-0.95))

                cumulative_x.append(x1)
                cumulative_y.append(y1)
                cumulative_95perc_index.append(min_y)
                real_data_in_cumulative_x.append(m_data)  #[0])

                ax22[1].plot(x1, y1)

                x_ticks = ax22[1].xaxis.get_majorticklocs()
                new_xticks = []
                for tick in x_ticks:
                    if tick > x1[min_y]+0.1 or tick < x1[min_y]-0.1:
                        new_xticks.append(numpy.around(tick, 1))

                ax22[1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax22[1].spines['top'].set_visible(False)
                ax22[1].spines['right'].set_visible(False)
                ax22[1].yaxis.set_ticks_position('left')
                ax22[1].xaxis.set_ticks_position('bottom')

                ax22[1].xaxis.set_ticks(list(numpy.unique(numpy.insert(numpy.array(new_xticks), 1,
                                                                       [x1[min_y], m_data]))))
                ax22[1].axvline(x1[min_y], linestyle='-', color='r', alpha=0.8, zorder=1)
                ax22[1].axvline(m_data, linestyle='-', color='g')
                d = x1-m_data
                ax22[1].plot(x1[numpy.nanargmin(d)], y1[numpy.nanargmin(d)], color='g', marker="o")
                ax22[1].set_xlim([0, max([m_data, x1[min_y]]) + 0.1])

                ax22[1].plot(x1[min_y], y1[min_y], color='r', marker="o")
                ax22[1].annotate('P = 0.95', xy=(x1[min_y], y1[min_y]), xytext=(x1[min_y]-0.2, .95), fontsize=12)

                Su_border95.append(x1[min_y])

                ax22[1].set_xlabel('M-Value')
                ax22[1].set_ylabel('Probability P(M)')

                # cl1_line = Line2D((0, 1), (0, 0), color='b', lw=7)
                # cl2_line = Line2D((0, 1), (0, 0), color='g', lw=7)
                # fig22.legend((cl1_line, cl2_line), ("Surrogate data", "Real data"), numpoints=1, loc=2, fontsize=15)

                if m_data <= x1[min_y]:

                    print 'Saving figure at ', path+'Double_Peaks/m_bad/'+file.split('.hkl')[0]+file_end+\
                                           run_direc[i]+'_gain_'+g+'_surrogateVSdata.pdf'
                    fig22.savefig(path+'Double_Peaks/m_bad/'+file.split('.hkl')[0]+file_end+
                                  run_direc[i]+'_gain_'+g+'_surrogateVSdata.pdf', format='pdf')

                else:

                    file_info = file.split('.hkl')[0]+file_end+run_direc[i]+'_gain_'+g
                    double_cells.append(file_info)

                    print 'Saving figure at ', path+'Double_Peaks/m_good/'+file.split('.hkl')[0]+file_end+\
                                           run_direc[i]+'_gain_'+g+'_surrogateVSdata.pdf'
                    fig22.savefig(path+'Double_Peaks/m_good/'+file.split('.hkl')[0]+file_end+
                                  run_direc[i]+'_gain_'+g+'_surrogateVSdata.pdf', format='pdf')

                # pl.show()
            pl.close('all')

    # fig2, ax2 = pl.subplots()
    # # print 'M = ', M
    # # M_plot = numpy.vstack(numpy.array(M)).flatten()
    # ax2.plot(numpy.sort(M_plot), numpy.array(range(len(M_plot)))/float(len(M_plot)))
    #
    # ax2.set_xlabel('M-Value')
    # ax2.set_ylabel('Probability P(M)')
    #
    # fig2.savefig(path+'Double_Peaks/M_surrogate_repeat_'+str(surrogate_repeat)+'.pdf', format='pdf')

    #  ________________________________  PREPARE AND SAVE INFO IN HKL FILE _______________________________
    # pad M arrays with nans to make them all equally long

    # for faulty_array in [M, X, Y_large, Y_small, orig_data_x, orig_data_y, gauss_x, gauss1_y, gauss2_y, derivative_y,
    #                      cumulative_x, cumulative_y]:

    for faulty_array in [M, X, Y_large, Y_small, orig_data_x, orig_data_y, cumulative_x, cumulative_y]:
        make_subarrays_equal_long(faulty_array)

    MaxFR_info = {'MaxGauss_large': MG_large,
                  'MaxGauss_small': MG_small,
                  'MaxGauss_combi': MG_combined,
                  'xMaxGauss_large': MxG_large,
                  'xMaxGauss_small': MxG_small,
                  'xMaxGauss_combi': Mx_combinedGauss,
                  'delta_F_dev_small_mean_surrogate': M,
                  'delta_F_dev_small_mean': M_data,
                  'surrogate_95_thresh': Su_border95,
                  'double_cell_names': double_cells,
                  'Weights_largeGauss': Weights_largeGauss,
                  'Weights_smallGauss': Weights_smallGauss,
                  'x': X,
                  'y_large': Y_large,
                  'y_small': Y_small,
                  'Y_comined': Y_comined,
                  'used_files': used_files,
                  'running_direc': running_direc,
                  'gains': gains}


                  # 'Max-mean-diff_in_multiSTD_large': MG_std_large,
                  # 'Max-mean-diff_in_multiSTD_small': MG_std_small,

    Plot_cumulative_info = {'orig_data_x': orig_data_x, 'orig_data_y': orig_data_y,
                            'gauss_x': X, 'gaussL_y': Y_large, 'gaussS_y': Y_small, 'gauss_combi_y': Y_comined,
                            'cumulative_x': cumulative_x, 'cumulative_y': cumulative_y,
                            'cumulative_95perc_index': cumulative_95perc_index,
                            'real_data_in_cumulative_x': real_data_in_cumulative_x,
                            'double_cell_names': double_cells,
                            'surrogate_repeat': surrogate_repeat}

    hickle.dump(MaxFR_info, path+'Summary/MaxFR_doublePeak_info_outliers.hkl', mode='w')
    hickle.dump(Plot_cumulative_info, path+'Summary/Plot_cumulative_info_outliers.hkl', mode='w')


def angle_likelihood_ratio(m_one_gauss, s_one_gauss, m1_two_gauss, m2_two_gauss, s1_two_gauss, s2_two_gauss):

    a1 = hickle.load('/Users/haasolivia/Documents/saw/dataWork/olivia/hickle/Summary/angles_info_raw_data.hkl')

    data_array = a1['no_double_angles']

    # scipy.stats.power_divergence(lambda_="log-likelihood")

    n = len(data_array)

    likelihood_one_gauss = -(n/2)*numpy.log(2*numpy.pi*(s_one_gauss**2))-(numpy.sum((data_array-m_one_gauss)**2)/(2*(s_one_gauss**2)))

    L1 = ((1/(2*numpy.pi*s1_two_gauss**2))**(n/2))*numpy.exp(-(numpy.sum((data_array-m1_two_gauss)**2)/(2*(s1_two_gauss**2))))
    L2 = ((1/(2*numpy.pi*s2_two_gauss**2))**(n/2))*numpy.exp(-(numpy.sum((data_array-m2_two_gauss)**2)/(2*(s2_two_gauss**2))))

    likelihood_two_gaussians = numpy.log(L1+L2)

    # calculating the likelihood_ratio test statistic
    t = -2*(likelihood_two_gaussians-likelihood_one_gauss)

    # calculating degrees of freedom
    one_gauss_parameter_count = 3
    two_gaussians_parameter_count = 6  # Amplitude, sigma, mu pro Gauss
    df = two_gaussians_parameter_count - one_gauss_parameter_count

    print 'test statistic = ', t, ' freedom difference df = ', df

    lr_pvalue = scipy.stats.chi2.sf(t, df=df)  # Survival function

    print 'angle_likelihood_ratio p-value is p = ', lr_pvalue


def angle_surrogate(raw_data=False, single_fields=True, double_fields=False, both=False, surrogate_repeat=10000):

    if single_fields and double_fields:
        print 'Choose either single OR double fields'
        sys.exit()
    if raw_data and double_fields:
        print 'Raw data cannot be plotted for double fields. Choose raw_data = False !'
        sys.exit()

    # -------------------- loading our data set -------------------------------

    a = hickle.load('/Users/haasolivia/Documents/saw/dataWork/olivia/hickle/Summary/angles_info.hkl')
    a1 = hickle.load('/Users/haasolivia/Documents/saw/dataWork/olivia/hickle/Summary/angles_info_raw_data.hkl')

    # fitted data
    if not raw_data:
        no_double_angles = a['no_double_angles']
        double_angles = a['double_angles']

    # raw data
    else:
        no_double_angles = a1['no_double_angles']

    if both:
        x = numpy.concatenate((no_double_angles, double_angles))
        angles = 'all_angles'
        thre = 0

    elif single_fields:
        x = no_double_angles
        angles = 'no_double_angles'
        thre = 0
        if raw_data:
            angles = 'no_double_angles_raw_data'

    elif double_fields:
        x = double_angles
        angles = 'double_angles'
        thre = 1

    bad_m_data_value = 0

    M = []
    M_data = []
    m_values = []
    Su_border95 = []

    # fig, ax = pl.subplots(1, 1, figsize=(10, 8))

    # -------------------- M-Value surrogate analysis -------------------------------

    for su in numpy.arange(surrogate_repeat+1):

        if su == 0:
            plot_data = True

            fig22, ax22, m = fit_gaussians(x=x, y_orig=0, surrogate=su, plot_data=plot_data,
                                           thresh=thre, hist=True)

        elif not bad_m_data_value:
            x_su = numpy.random.choice(list(numpy.arange(.001, 2.001, .001)), len(x))
            y_su = numpy.random.choice(list(numpy.arange(.001, 2.001, .001)), len(x))
            surrogate_x = numpy.array([180*numpy.arctan(y_su[idx]/x_su[idx])/numpy.pi for idx
                                       in numpy.arange(len(x_su))])
            # if su < 4:
            #     ax.hist(surrogate_x, bins=numpy.arange(min(surrogate_x), max(surrogate_x) + 4, 4), histtype='step', linewidth=2)

            plot_data = False
            m = fit_gaussians(x=surrogate_x, y_orig=0, surrogate=su, plot_data=plot_data, thresh=0, hist=True)
            # if m > .6:
            #     plot_data = True
            #     f, a, q = fit_gaussians(x=surrogate_x, y_orig=0, surrogate=su, plot_data=plot_data, thresh=0, hist=True)

        if m < 0.005 or numpy.isnan(m):
            m = numpy.nan
        m_values.append(m)

        if su == 0 and numpy.isnan(m):
            bad_m_data_value = 1

    m_data = m_values[0]
    M_data.append(m_data)

    if not bad_m_data_value:
        m = numpy.array(m_values[1:])
        m = m[~numpy.isnan(m)]
        M.append(m)
    else:
        M.append(numpy.array([numpy.nan]))

    # -------------------- Plotting results -------------------------------

    # if 3 derivatives == 0 are found then plot the surrogate dataset and actual data value
    if numpy.isnan(m_data):  #good == 0:
        Su_border95.append(numpy.nan)
        cumulative_x.append(numpy.nan)
        cumulative_y.append(numpy.nan)
        cumulative_95perc_index.append(numpy.nan)
        real_data_in_cumulative_x.append(numpy.nan)

        ax22[1].axis('off')

        # print 'Saving figure at ', path+'Summary/'+angles+'_surrogateVSdata.pdf'
        # fig22.savefig(path+'Summary/'+angles+'_surrogateVSdata.pdf', format='pdf')
    else:   #if good == 1:
        # fig01, ax01 = pl.subplots()
        matplotlib.pyplot.sca(ax22[1])
        pl.xticks(rotation=70)

        ax22[1].spines['top'].set_visible(False)
        ax22[1].spines['right'].set_visible(False)

        x1 = numpy.sort(m)
        y1 = numpy.array(range(len(m)))/float(len(m))

        min_y = numpy.nanargmin(abs(y1-0.95))

        cumulative_x.append(x1)
        cumulative_y.append(y1)
        cumulative_95perc_index.append(min_y)
        real_data_in_cumulative_x.append(m_data)  #[0])

        ax22[1].plot(x1, y1)

        x_ticks = ax22[1].xaxis.get_majorticklocs()
        new_xticks = []
        for tick in x_ticks:
            if tick > x1[min_y]+0.1 or tick < x1[min_y]-0.1:
                new_xticks.append(numpy.around(tick, 1))

        ax22[1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax22[1].spines['top'].set_visible(False)
        ax22[1].spines['right'].set_visible(False)
        ax22[1].yaxis.set_ticks_position('left')
        ax22[1].xaxis.set_ticks_position('bottom')

        ax22[1].xaxis.set_ticks(list(numpy.unique(numpy.insert(numpy.array(new_xticks), 1,
                                                               [x1[min_y], m_data]))))
        ax22[1].axvline(x1[min_y], linestyle='-', color='r', alpha=0.8, zorder=1)
        ax22[1].axvline(m_data, linestyle='-', color='g')
        d = abs(x1-m_data)

        a = x1[numpy.nanargmin(d)-1]
        b = y1[numpy.nanargmin(d)-1]
        c = x1[numpy.nanargmin(d)]
        d = y1[numpy.nanargmin(d)]
        x_t = m_data
        y_t = ((d-b)/(c-a))*x_t + b - a*((d-b)/(c-a))
        ax22[1].plot(x_t, y_t, color='g', marker="o")
        ax22[1].set_xlim([0, max([m_data, x1[min_y]]) + 0.1])

        ax22[1].plot(x1[min_y], y1[min_y], color='r', marker="o")
        ax22[1].annotate('P = 0.95', xy=(x1[min_y], y1[min_y]), xytext=(x1[min_y]-0.2, .95), fontsize=12)
        ax22[1].annotate('Data = '+str(numpy.round(y_t, 5)), xy=(x1[min_y], y1[min_y]), xytext=(x1[min_y]-0.2, .5), fontsize=12)

        Su_border95.append(x1[min_y])

        ax22[1].set_xlabel('M-Value')
        ax22[1].set_ylabel('Probability P(M)')

    # -------------------- Saving results -------------------------------

    print 'Saving figure at ', path+'Summary/'+angles+'_surrogateVSdata.pdf'
    fig22.savefig(path+'Summary/'+angles+'_surrogateVSdata.pdf', format='pdf')

    pl.show()
    # pl.close('all')


def plot_FRpeaks():
    # initiate figures
    f_FR, ax_FR = pl.subplots(2, 1, figsize=(32, 17))  #, sharey='row')  #, sharex=True)
    ax_FR = ax_FR.flatten()

    # load data
    # mFR = hickle.load(path+'Summary/MaxFR_doublePeak_info.hkl')
    mFR = hickle.load(path+'Summary/MaxFR_doublePeak_info_corrected.hkl')

    # plot colors
    Cmap = []
    for c in numpy.arange(len(mFR['MaxGauss_large'])):
        Cmap.append(custom_plot.pretty_colors_set2[c%custom_plot.pretty_colors_set2.__len__()])

    # plot data
    for i in numpy.arange(len(mFR['MaxGauss_large'])):
        ax_FR[0].plot(mFR['MaxGauss_large'][i], mFR['MaxGauss_small'][i], color=Cmap[i], marker="o")  #, marker="^")
        # ax_FR[0].plot(i, mFR['MaxGauss_small'][i], color=Cmap[i], marker="o")

        ax_FR[1].plot(mFR['Max-mean-diff_in_multiSTD_large'][i], mFR['Max-mean-diff_in_multiSTD_small'][i],
                      color=Cmap[i], marker="o")  #, marker="^")
        # ax_FR[1].plot(i, mFR['Max-mean-diff_in_multiSTD_small'][i], color=Cmap[i], marker="o")

    # setting plot labels
    ax_FR[0].set_xlabel('Maximal firing rate (Hz) large gaussian')
    ax_FR[1].set_xlabel('Difference of mean and max FR in multiples of STD (large gaussian)')

    ax_FR[0].set_ylabel('Maximal firing rate (Hz) small gaussian')
    ax_FR[1].set_ylabel('Difference of mean and max FR in multiples of STD (small gaussian)')

    # set x limit
    # for lim in [0, 1]:
    #     ax_FR[lim].set_xlim(0, len(mFR['MaxGauss_large']))
    #
    # # set legend
    # cl1_tri = Line2D([0], [0], linestyle="none", marker="^", markersize=10, markerfacecolor='k')
    # cl1_dot = Line2D([0], [0], linestyle="none", marker="o", markersize=10, markerfacecolor='k')
    #
    # ax_FR[0].legend((cl1_tri, cl1_dot), ("Values for main/larger gaussian", "Values for smaller gaussian"),
    #           numpoints=1, loc=1, fontsize=12)

    f_FR.savefig(path+'Double_Peaks/FR_summary2D.pdf', format='pdf')


def get_deltas(xlarge05, xlarge15, xsmall05, xsmall15, norm=True):

    xdiffL = xlarge05 - xlarge15
    xdiffS = xsmall05 - xsmall15

    if norm:
        delta_x_normL = xdiffL
        delta_x_normS = xdiffS

        delta_x_L = (xlarge05*0.5) - (xlarge15*1.5)
        delta_x_S = (xsmall05*0.5) - (xsmall15*1.5)
    else:
        delta_x_L = xdiffL
        delta_x_S = xdiffS

        delta_x_normL = (xlarge05*(1./0.5)) - (xlarge15*(1./1.5))
        delta_x_normS = (xsmall05*(1./0.5)) - (xsmall15*(1./1.5))

    return delta_x_normL, delta_x_L, delta_x_normS, delta_x_S


def get_pf_loc(xlarge05, xlarge15, xsmall05, xsmall15, norm=True):

    if norm:
        x_normL05 = xlarge05
        x_normL15 = xlarge15
        x_normS05 = xsmall05
        x_normS15 = xsmall15

        x_L05 = xlarge05*0.5
        x_L15 = xlarge15*1.5
        x_S05 = xsmall05*0.5
        x_S15 = xsmall15*1.5

    else:
        x_L05 = xlarge05
        x_L15 = xlarge15
        x_S05 = xsmall05
        x_S15 = xsmall15

        x_normL05 = xlarge05*(1/0.5)
        x_normL15 = xlarge15*(1/1.5)
        x_normS05 = xsmall05*(1/0.5)
        x_normS15 = xsmall15*(1/1.5)
    
    return x_normL05, x_normL15, x_normS05, x_normS15, x_L05, x_L15, x_S05, x_S15


def get_xy(x05, x15, ylarge05, ylarge15, ysmall05, ysmall15, ycombi05, ycombi15, norm=True):

    if norm:
        return [[[x05*0.5, ylarge05], [x15*1.5, ylarge15]], [[x05, ylarge05], [x15, ylarge15]]], \
               [[[x05*0.5, ysmall05], [x15*1.5, ysmall15]], [[x05, ysmall05], [x15, ysmall15]]], \
               [[[x05*0.5, ycombi05], [x15*1.5, ycombi15]], [[x05, ycombi05], [x15, ycombi15]]]

    else:
        return [[[x05, ylarge05], [x15, ylarge15]], [[x05*(1./0.5), ylarge05], [x15*(1./1.5), ylarge15]]], \
               [[[x05, ysmall05], [x15, ysmall15]], [[x05*(1./0.5), ysmall05], [x15*(1./1.5), ysmall15]]], \
               [[[x05, ycombi05], [x15, ycombi15]], [[x05*(1./0.5), ycombi05], [x15*(1./1.5), ycombi15]]]


def CA1_or_CA3(file):

    animal_str = file.split('_20')[0]
    date_str = ('20'+file.split('_20')[1]).split('_')[0]

    # animals = ['10823', '10529', '10528', '10353', '10535', '10537']
    # ca1 = ['2015-08-05', '2015-03-04', '2015-03-16', '2014-07-02', '2015-10-30', '2015-10-30']

    animals = ['10823', '10529', '10528', '10353', '10535', '10537']
    ca1 = ['None', '2015-03-04', 'None', 'None', '2015-10-30', '2015-10-30']

    region = []
    for i in numpy.arange(len(animals)):
        if animal_str == animals[i]:
            if ca1[i] == 'None':
                region.append('CA3')
            elif parser.parse(date_str) <= parser.parse(ca1[i]):
                region.append('CA1')
            else:
                region.append('CA3')


    if len(region) > 1:
        print 'WARNING more than one CA region found for animal and date: '+animal_str+' '+date_str
        sys.exit()
    if len(region) < 1:
        print 'WARNING NO CA region found for animal and date: '+animal_str+' '+date_str
        sys.exit()

    return region[0]


def find_double_cells(info, hkl_files=hkl_files, run_direc=run_direc):

    # for 0 Hz - 205 double cells, for 0.2 Hz - 202 double cells, for 0.3 Hz 194 double cells
    # small_gauss_thresh = 0.5
    # small_gauss_thresh = 0.0  # threshold for smaller 1d gauss in Hz. The bigger gauss thresh is set to 1.0 Hz in 2d
    # small_gauss_thresh_perc = 0.1  # percentage threshold of the bigger gauss FR maximum

    m_value_norm = []
    m_value = []
    file_ind_norm = []
    file_ind = []
    small_gauss = []
    small_gauss_norm = []
    small_gauss_width = []
    small_gauss_norm_width = []
    # names = numpy.array(info['double_cell_names'])
    names = numpy.array(info['used_files'])
    directions = numpy.array(info['running_direc'])
    gains = numpy.array(info['gains'])
    filenames = numpy.array([names[f].strip('.hkl')+'_'+directions[f]+'_gain_'+gains[f] for f in numpy.arange(len(names))])
    used_files = []

    Field_deltas_real = []
    Field_deltas_virt = []
    Field_x_real05 = []
    Field_x_real15 = []
    Field_x_virt05 = []
    Field_x_virt15 = []

    Field_deltas_real_data = []
    Field_deltas_virt_data = []
    Field_x_real05_data = []
    Field_x_real15_data = []
    Field_x_virt05_data = []
    Field_x_virt15_data = []

    xy = []
    Field_weights = []
    no_double_cells_CA1 = 0
    no_double_cells_CA3 = 0
    double_cells_CA1 = 0
    double_cells_CA3 = 0
    CA_region = []
    double_cell_files = []
    double_cell_direc = []
    no_double_cell_files = []
    no_double_cell_direc = []
    Gauss_L_width05 = []
    Gauss_S_width05 = []
    Gauss_Combi_width05 = []
    Gauss_L_maxFR05 = []
    Gauss_S_maxFR05 = []
    Gauss_Combi_maxFR05 = []
    Gauss_L_width15 = []
    Gauss_S_width15 = []
    Gauss_Combi_width15 = []
    Gauss_L_maxFR15 = []
    Gauss_S_maxFR15 = []
    Gauss_Combi_maxFR15 = []

    infoWM = hickle.load(path+'Summary/Gauss_width_FRmax.hkl')

    # remove 'bad' data

    vis_idx = []
    for index, file in enumerate(hkl_files):
        v = file.split('.hkl')[0]+'_'+run_direc[index]
        if v not in bad:
            vis_idx.append(index)
        else:
            print v

    hkl_files = numpy.array(hkl_files)[vis_idx]
    run_direc = numpy.array(run_direc)[vis_idx]

    for i, file in enumerate(hkl_files):

        for gain in ['05', '15']:

            if gain == '05':
                ga = '0.5'
            else:
                ga = '1.5'

            if i%2 != 0:
                file_end = '_normalised_'
            else:
                file_end = '_'

            file_index = numpy.where(filenames == file.split('.hkl')[0]+file_end+run_direc[i]+'_gain_'+ga)[0]
            # test_index = numpy.where(filenames == '10823_2015-08-18_VR_GCend_nami_linTrack1_TT3_SS_16_PF_info_right_gain_1.5')[0]

            file_2d = hickle.load(path+file)
            fr_max_2d = numpy.nanmax(file_2d[run_direc[i]+'FR_2d_x_y_gain_'+ga][1])
            small_gauss_thresh_perc = fr_tresh/fr_max_2d  # fr_thresh is in Hz and defined in the very top of the script

            if info['delta_F_dev_small_mean'][file_index] - info['surrogate_95_thresh'][file_index] >= 0.:
                if i%2 != 0:  #file.endswith('normalised.hkl'):
                    m_value_norm.append(1)
                    file_ind_norm.append(file_index)

                    # check if FR of small gauss is ok:
                    if infoWM['GaussFRmaxS'][file_index] < small_gauss_thresh_perc * infoWM['GaussFRmaxL'][file_index]:
                        small_gauss_norm.append(0)  # appending for both gains!
                    else:
                        small_gauss_norm.append(1)

                    # check if width of small gauss is ok:
                    if pf_width_max*(2./float(ga)) > infoWM['GaussPFwidthS'][file_index] > pf_width_thresh*(2./float(ga)):
                        small_gauss_norm_width.append(1)
                    else:
                        small_gauss_norm_width.append(0)

                else:
                    m_value.append(1)
                    file_ind.append(file_index)

                    # check if FR of small gauss is ok:
                    if infoWM['GaussFRmaxS'][file_index] < small_gauss_thresh_perc * infoWM['GaussFRmaxL'][file_index]:
                        small_gauss.append(0)
                    else:
                        small_gauss.append(1)

                    # check if width of small gauss is ok:
                    if pf_width_max*2. > infoWM['GaussPFwidthS'][file_index] > pf_width_thresh*2.:
                        small_gauss_width.append(1)
                    else:
                        small_gauss_width.append(0)

            else:
                if i%2 != 0:  #file.endswith('normalised.hkl'):
                    m_value_norm.append(0)
                    file_ind_norm.append(file_index)
                    small_gauss_norm.append(0)
                    small_gauss_norm_width.append(0)

                    # check if FR of small gauss is ok:
                    # if infoWM['GaussFRmaxS'][file_index] < small_gauss_thresh_perc * infoWM['GaussFRmaxL'][file_index]:
                    #     small_gauss_norm.append(0)
                    # else:
                    #     small_gauss_norm.append(1)
                else:
                    m_value.append(0)
                    file_ind.append(file_index)
                    small_gauss.append(0)
                    small_gauss_width.append(0)

                    # check if FR of small gauss is ok:
                    # if infoWM['GaussFRmaxS'][file_index] < small_gauss_thresh_perc * infoWM['GaussFRmaxL'][file_index]:
                    #     small_gauss.append(0)
                    # else:
                    #     small_gauss.append(1)

            if gain == '05':
                Gauss_L_width05.append(infoWM['GaussPFwidthL'][file_index])
                Gauss_S_width05.append(infoWM['GaussPFwidthS'][file_index])
                Gauss_Combi_width05.append(infoWM['GaussPFwidthCombi'][file_index])
                Gauss_L_maxFR05.append(infoWM['GaussFRmaxL'][file_index])
                Gauss_S_maxFR05.append(infoWM['GaussFRmaxS'][file_index])
                Gauss_Combi_maxFR05.append(infoWM['GaussFRmaxCombi'][file_index])
            else:
                Gauss_L_width15.append(infoWM['GaussPFwidthL'][file_index])
                Gauss_S_width15.append(infoWM['GaussPFwidthS'][file_index])
                Gauss_Combi_width15.append(infoWM['GaussPFwidthCombi'][file_index])
                Gauss_L_maxFR15.append(infoWM['GaussFRmaxL'][file_index])
                Gauss_S_maxFR15.append(infoWM['GaussFRmaxS'][file_index])
                Gauss_Combi_maxFR15.append(infoWM['GaussFRmaxCombi'][file_index])


        # zB m_value_norm = [good_m_for_g05, bad_m_for_g15] --> good = 1, bad = 0
        # zB file_ind_norm = [index_norm05, index_norm15]

        if i%2 != 0:  # when file is uneven - when normalised and not normalised data of the same cluster was processed
            # 0 not in [sum(small_gauss_norm), sum(small_gauss)] ... not FR below thresh in both gains for either
            # normed or not normed data
            region = CA1_or_CA3(file)

            m_idx_norm = numpy.where(numpy.array(m_value_norm) == 1)[0]
            if len(m_idx_norm) == 2:  # if both gains have good m-values
                sg_norm = sum(small_gauss_norm)  # at least one of the small gaussians has to fulfill the thresh
                sg_norm_width = sum(small_gauss_norm_width)
            elif len(m_idx_norm) == 1:
                # the small gaussian of the good m-value gain has to fulfill the thresh
                sg_norm = small_gauss_norm[m_idx_norm]
                sg_norm_width = small_gauss_norm_width[m_idx_norm]
            else:
                sg_norm = 0
                sg_norm_width = 0

            m_idx = numpy.where(numpy.array(m_value) == 1)[0]  # m_value has two values, one for each gain
            if len(m_idx) == 2:
                sg = sum(small_gauss)
                sg_width = sum(small_gauss_width)
            elif len(m_idx) == 1:
                sg = small_gauss[m_idx]
                sg_width = small_gauss_width[m_idx]
            else:
                sg = 0
                sg_width = 0

            if sum(m_value_norm) > sum(m_value) and sg_norm != 0 and sg_norm_width != 0\
                    and not file.split('.hkl')[0]+'_'+run_direc[i] in double_shoule_be_single:  # 0 not in [sum(small_gauss_norm), sum(small_gauss)]:
                print 'it is a double cell and normfit is better but virtual fit is used!'
                normfit = 1
                virfit = 0
                if region == 'CA1':
                    double_cells_CA1 += 1
                else:
                    double_cells_CA3 += 1
                double_cell_files.append(file)
                double_cell_direc.append(run_direc[i])

            elif sum(m_value) > sum(m_value_norm) and sg != 0 and sg_width != 0\
                    and not file.split('.hkl')[0]+'_'+run_direc[i] in double_shoule_be_single:  # 0 not in [sum(small_gauss_norm), sum(small_gauss)]:
                print 'it is a double cell and virtual fit is better'
                normfit = 0
                virfit = 1
                if region == 'CA1':
                    double_cells_CA1 += 1
                else:
                    double_cells_CA3 += 1
                double_cell_files.append(file)
                double_cell_direc.append(run_direc[i])

            # if the m-values are the good for normalised and visual data, then the small gauss of one of them has
            # to pass the threshold and one width has to be good:
            elif sum(m_value) == sum(m_value_norm) and sum(m_value) != 0 and \
                    0 not in [sg_norm, sg] and 0 not in [sg_norm_width, sg_width]\
                    and not file.split('.hkl')[0]+'_'+run_direc[i] in double_shoule_be_single:  #[sum(small_gauss_norm), sum(small_gauss)]:
                print 'it is a double cell and no fit is better --> virfit is taken'
                normfit = 0
                virfit = 1
                if region == 'CA1':
                    double_cells_CA1 += 1
                else:
                    double_cells_CA3 += 1
                double_cell_files.append(file)
                double_cell_direc.append(run_direc[i])

            else:
                print 'it is not a double cell'
                normfit = 0
                virfit = 0
                if region == 'CA1':
                    no_double_cells_CA1 += 1
                else:
                    no_double_cells_CA3 += 1
                no_double_cell_files.append(file)
                no_double_cell_direc.append(run_direc[i])

            # ================== when there is no double cell ============================

            if normfit == virfit or file.split('.hkl')[0]+'_'+run_direc[i] in double_shoule_be_single:
                if file.split('.hkl')[0]+'_'+run_direc[i] in double_shoule_be_single:
                    print 'making ', file.split('.hkl')[0]+'_'+run_direc[i], ' single cell!'
                delta_x_normL, delta_x_L, delta_x_normS, delta_x_S = \
                    get_deltas(xlarge05=info['xMaxGauss_combi'][file_ind[0]],  # data file, gain 0.5
                               xlarge15=info['xMaxGauss_combi'][file_ind[1]],  # data file, gain 1.5
                               xsmall05=numpy.nan, xsmall15=numpy.nan, norm=False)

                x_normL05, x_normL15, x_normS05, x_normS15, x_L05, x_L15, x_S05, x_S15 = \
                    get_pf_loc(xlarge05=info['xMaxGauss_combi'][file_ind[0]],  # data file, gain 0.5
                               xlarge15=info['xMaxGauss_combi'][file_ind[1]],  # data file, gain 1.5
                               xsmall05=numpy.nan, xsmall15=numpy.nan, norm=False)

                if delta_x_normL != x_normL05-x_normL15 and not numpy.isnan(delta_x_normL):
                    print 'delta_x_normL = ', delta_x_normL
                    print 'x_normL05 = ', x_normL05
                    print 'x_normL15 = ', x_normL15
                    print 'x_normL05 - x_normL15 = ', x_normL05-x_normL15
                if delta_x_L != x_L05-x_L15 and not numpy.isnan(delta_x_L):
                    print 'delta_x_L = ', delta_x_L
                    print 'x_L05 = ', x_L05
                    print 'x_L15 = ', x_L15
                    print 'x_L05 - x_L15 = ', x_L05-x_L15
                if delta_x_normS != x_normS05-x_normS15 and not numpy.isnan(delta_x_normS):
                    print 'delta_x_normS = ', delta_x_normS
                    print 'x_normS05 = ', x_normS05
                    print 'x_normS15 = ', x_normS15
                    print 'x_normS05 - x_normS15 = ', x_normS05-x_normS15
                if delta_x_S != x_S05-x_S15 and not numpy.isnan(delta_x_S):
                    print 'delta_x_S = ', delta_x_S
                    print 'x_S05 = ', x_S05
                    print 'x_S15 = ', x_S15
                    print 'x_S05 - x_S15 = ', x_S05-x_S15

                xy_L, xy_S, xy_C = get_xy(x05=info['x'][file_ind[0]],
                                    x15=info['x'][file_ind[1]],
                                    ylarge05=info['y_large'][file_ind[0]],
                                    ylarge15=info['y_large'][file_ind[1]],
                                    ysmall05=numpy.nan*info['y_small'][file_ind[0]],
                                    ysmall15=numpy.nan*info['y_small'][file_ind[1]],
                                    ycombi05=info['Y_comined'][file_ind[0]],
                                    ycombi15=info['Y_comined'][file_ind[1]], norm=False)

                xy_L = xy_C  # for a single field the combined gaussian is the best representation of the firing field!

                field_weightL = 1.
                field_weightS = numpy.nan

                file_vis = file.split('info')[0]+'info.hkl'

                data_index = numpy.where(numpy.array(vis_hkl_files) == file_vis)[0][0]

                # ______ info for real data ________________________________

                file_v = hickle.load(path+file_vis)
                xV05 = file_v['xMaxFRySuminPF_MaxFRySuminPF_xCMySuminPF_'+run_direc[i]+'Runs_gain_0.5'][0]
                xV15 = file_v['xMaxFRySuminPF_MaxFRySuminPF_xCMySuminPF_'+run_direc[i]+'Runs_gain_1.5'][0]

                if run_direc[i] == 'left':
                    xV05 = abs(xV05-2)
                    xV15 = abs(xV15-2)

                x_normData05 = xV05/.5  # norm_info[PFx05][data_index]
                x_normData15 = xV15/1.5  # norm_info[PFx15][data_index]
                x_Data05 = xV05  # vis_info[PFx05][data_index]
                x_Data15 = xV15  # vis_info[PFx15][data_index]

                delta_x_normData = x_normData05 - x_normData15  # norm_info[pfc][data_index]
                delta_x_Data = x_Data05 - x_Data15  # vis_info[pfc][data_index]

                xy_Data, xy_none, xy_none = get_xy(x05=info['x'][file_ind[0]],
                                          x15=info['x'][file_ind[1]],
                                          ylarge05=info['y_large'][file_ind[0]],
                                          ylarge15=info['y_large'][file_ind[1]],
                                          ysmall05=numpy.nan*info['y_small'][file_ind[0]],
                                          ysmall15=numpy.nan*info['y_small'][file_ind[1]],
                                          ycombi05=info['Y_comined'][file_ind[0]],
                                          ycombi15=info['Y_comined'][file_ind[1]], norm=False)

            # ===================== when it is a double cell ===============================

            elif normfit < virfit or normfit > virfit:
                delta_x_normL, delta_x_L, delta_x_normS, delta_x_S = \
                    get_deltas(xlarge05=info['xMaxGauss_large'][file_ind[0]],
                               xlarge15=info['xMaxGauss_large'][file_ind[1]],
                               xsmall05=info['xMaxGauss_small'][file_ind[0]],
                               xsmall15=info['xMaxGauss_small'][file_ind[1]], norm=False)

                x_normL05, x_normL15, x_normS05, x_normS15, x_L05, x_L15, x_S05, x_S15 = \
                    get_pf_loc(xlarge05=info['xMaxGauss_large'][file_ind[0]],
                               xlarge15=info['xMaxGauss_large'][file_ind[1]],
                               xsmall05=info['xMaxGauss_small'][file_ind[0]],
                               xsmall15=info['xMaxGauss_small'][file_ind[1]], norm=False)

                if delta_x_normL != x_normL05-x_normL15 and not numpy.isnan(delta_x_normL):
                    print 'delta_x_normL = ', delta_x_normL
                    print 'x_normL05 = ', x_normL05
                    print 'x_normL15 = ', x_normL15
                    print 'x_normL05 - x_normL15 = ', x_normL05-x_normL15
                if delta_x_L != x_L05-x_L15 and not numpy.isnan(delta_x_L):
                    print 'delta_x_L = ', delta_x_L
                    print 'x_L05 = ', x_L05
                    print 'x_L15 = ', x_L15
                    print 'x_L05 - x_L15 = ', x_L05-x_L15
                if delta_x_normS != x_normS05-x_normS15 and not numpy.isnan(delta_x_normS):
                    print 'delta_x_normS = ', delta_x_normS
                    print 'x_normS05 = ', x_normS05
                    print 'x_normS15 = ', x_normS15
                    print 'x_normS05 - x_normS15 = ', x_normS05-x_normS15
                if delta_x_S != x_S05-x_S15 and not numpy.isnan(delta_x_S):
                    print 'delta_x_S = ', delta_x_S
                    print 'x_S05 = ', x_S05
                    print 'x_S15 = ', x_S15
                    print 'x_S05 - x_S15 = ', x_S05-x_S15

                xy_L, xy_S, xy_C = get_xy(x05=info['x'][file_ind[0]],
                                    x15=info['x'][file_ind[1]],
                                    ylarge05=info['y_large'][file_ind[0]],
                                    ylarge15=info['y_large'][file_ind[1]],
                                    ysmall05=info['y_small'][file_ind[0]],
                                    ysmall15=info['y_small'][file_ind[1]],
                                    ycombi05=info['Y_comined'][file_ind[0]],
                                    ycombi15=info['Y_comined'][file_ind[1]], norm=False)

                if normfit > virfit:
                    # if m_value is good for both gains (sum = 2), use weight from gain 0.5
                    # or if m_value is better for gain 0.5, use weight from gain 0.5
                    if sum(m_value_norm) == 2 or m_value_norm[0] > m_value_norm[1]:
                        id = 0
                    elif m_value_norm[0] < m_value_norm[1]:
                        # if m_value is better for gain 1.5, use weight of gain 1.5
                        id = 1
                    else:
                        print 'm_value_norm problem for field weight computation! PROGRAM ABORTED!'
                        sys.exit()
                else:
                    if sum(m_value) == 2 or m_value[0] > m_value[1]:
                        id = 0
                    elif m_value[0] < m_value[1]:
                        id = 1
                    else:
                        print 'm_value problem for field weight computation! PROGRAM ABORTED!'
                        sys.exit()

                field_weightL = info['Weights_largeGauss'][file_ind[id]]
                field_weightS = info['Weights_smallGauss'][file_ind[id]]

                # ______ info for real data ________________________________

                delta_x_normData = numpy.nan
                delta_x_Data = numpy.nan

                x_normData05 = numpy.nan
                x_normData15 = numpy.nan
                x_Data05 = numpy.nan
                x_Data15 = numpy.nan

            # elif normfit > virfit:
            #     delta_x_normL, delta_x_L, delta_x_normS, delta_x_S = \
            #         get_deltas(xlarge05=info['xMaxGauss_large'][file_ind_norm[0]],
            #                    xlarge15=info['xMaxGauss_large'][file_ind_norm[1]],
            #                    xsmall05=info['xMaxGauss_small'][file_ind_norm[0]],
            #                    xsmall15=info['xMaxGauss_small'][file_ind_norm[1]])
            #
            #     x_normL05, x_normL15, x_normS05, x_normS15, x_L05, x_L15, x_S05, x_S15 = \
            #         get_pf_loc(xlarge05=info['xMaxGauss_large'][file_ind_norm[0]],
            #                    xlarge15=info['xMaxGauss_large'][file_ind_norm[1]],
            #                    xsmall05=info['xMaxGauss_small'][file_ind_norm[0]],
            #                    xsmall15=info['xMaxGauss_small'][file_ind_norm[1]])
            #
            #     if delta_x_normL != x_normL05-x_normL15 and not numpy.isnan(delta_x_normL):
            #         print 'delta_x_normL = ', delta_x_normL
            #         print 'x_normL05 = ', x_normL05
            #         print 'x_normL15 = ', x_normL15
            #         print 'x_normL05 - x_normL15 = ', x_normL05-x_normL15
            #     if delta_x_L != x_L05-x_L15 and not numpy.isnan(delta_x_L):
            #         print 'delta_x_L = ', delta_x_L
            #         print 'x_L05 = ', x_L05
            #         print 'x_L15 = ', x_L15
            #         print 'x_L05 - x_L15 = ', x_L05-x_L15
            #     if delta_x_normS != x_normS05-x_normS15 and not numpy.isnan(delta_x_normS):
            #         print 'delta_x_normS = ', delta_x_normS
            #         print 'x_normS05 = ', x_normS05
            #         print 'x_normS15 = ', x_normS15
            #         print 'x_normS05 - x_normS15 = ', x_normS05-x_normS15
            #     if delta_x_S != x_S05-x_S15 and not numpy.isnan(delta_x_S):
            #         print 'delta_x_S = ', delta_x_S
            #         print 'x_S05 = ', x_S05
            #         print 'x_S15 = ', x_S15
            #         print 'x_S05 - x_S15 = ', x_S05-x_S15
            #
            #     xy_L, xy_S, xy_C = get_xy(x05=info['x'][file_ind_norm[0]],
            #                         x15=info['x'][file_ind_norm[1]],
            #                         ylarge05=info['y_large'][file_ind_norm[0]],
            #                         ylarge15=info['y_large'][file_ind_norm[1]],
            #                         ysmall05=info['y_small'][file_ind_norm[0]],
            #                         ysmall15=info['y_small'][file_ind_norm[1]],
            #                         ycombi05=info['Y_comined'][file_ind_norm[0]],
            #                         ycombi15=info['Y_comined'][file_ind_norm[1]])
            #
            #     if sum(m_value_norm) == 2 or m_value_norm[0] > m_value_norm[1]:
            #         # if m_value is good for both gains (sum = 2), use weight from gain 0.5
            #         # or if m_value is better for gain 0.5, use weight from gain 0.5
            #         idx = 0
            #     elif m_value_norm[0] < m_value_norm[1]:
            #         # if m_value is better for gain 1.5, use weight of gain 1.5
            #         idx = 1
            #     else:
            #         print 'm_value_norm problem for field weight computation! PROGRAM ABORTED!'
            #         sys.exit()
            #
            #     field_weightL = info['Weights_largeGauss'][file_ind_norm[idx]]
            #     field_weightS = info['Weights_smallGauss'][file_ind_norm[idx]]
            #
            #     # ______ info for real data ________________________________
            #
            #     delta_x_normData = numpy.nan
            #     delta_x_Data = numpy.nan
            #
            #     x_normData05 = numpy.nan
            #     x_normData15 = numpy.nan
            #     x_Data05 = numpy.nan
            #     x_Data15 = numpy.nan

            else:
                print 'normfit = ', normfit
                print 'virfit = ', virfit
                print 'PROGRAM ABORTED: PROBLEM WITH normfit and virfit!'
                sys.exit()

            # if file_index == test_index:
            #     print 'm = ', info['delta_F_dev_small_mean'][file_index]
            #     print 'surrogate_95_thresh = ', info['surrogate_95_thresh'][file_index]
            #     print m_value_norm, m_value
            #     print 'small_gauss', small_gauss_norm, small_gauss
            #     print file
            #     sys.exit()

            # reset m_value comparison for next cluster
            m_value_norm = []
            m_value = []

            # reset file_index values for next cluster
            file_ind_norm = []
            file_ind = []

            # reset small_gauss fr values for next cluster
            small_gauss_norm = []
            small_gauss = []

            # reset small_gauss width values for next cluster
            small_gauss_norm_width = []
            small_gauss_width = []

            # append deltas and weights to list
            Field_deltas_real.append(delta_x_normL)
            Field_deltas_virt.append(delta_x_L)
            Field_x_real05.append(x_normL05)
            Field_x_real15.append(x_normL15)
            Field_x_virt05.append(x_L05)
            Field_x_virt15.append(x_L15)

            Field_deltas_real.append(delta_x_normS)
            Field_deltas_virt.append(delta_x_S)
            Field_x_real05.append(x_normS05)
            Field_x_real15.append(x_normS15)
            Field_x_virt05.append(x_S05)
            Field_x_virt15.append(x_S15)

            Field_deltas_real_data.append(delta_x_normData)
            Field_deltas_virt_data.append(delta_x_Data)
            Field_x_real05_data.append(x_normData05)
            Field_x_real15_data.append(x_normData15)
            Field_x_virt05_data.append(x_Data05)
            Field_x_virt15_data.append(x_Data15)

            Field_weights.append(field_weightL)

            Field_weights.append(field_weightS)

            xy.append(xy_L)
            xy.append(xy_S)

        if i%2 != 0:
            used_files.append(file.split('.hkl')[0]+'_normalised_'+run_direc[i])
        else:
            used_files.append(file.split('.hkl')[0]+'_'+run_direc[i])

        region = CA1_or_CA3(file)
        CA_region.append(region)

    delta_and_weight_info = {'Field_deltas_real': Field_deltas_real,
                             'Field_deltas_virt': Field_deltas_virt,
                             'Field_x_real05': Field_x_real05,
                             'Field_x_real15': Field_x_real15,
                             'Field_x_virt05': Field_x_virt05,
                             'Field_x_virt15': Field_x_virt15,
                             'Field_deltas_real_data': Field_deltas_real_data,
                             'Field_deltas_virt_data': Field_deltas_virt_data,
                             'Field_x_real05_data': Field_x_real05_data,
                             'Field_x_real15_data': Field_x_real15_data,
                             'Field_x_virt05_data': Field_x_virt05_data,
                             'Field_x_virt15_data': Field_x_virt15_data,
                             'Field_weights': Field_weights,
                             'used_files': used_files,
                             'num_double_cells_CA1': double_cells_CA1,
                             'num_double_cells_CA3': double_cells_CA3,
                             'num_no_double_cells_CA1': no_double_cells_CA1,
                             'num_no_double_cells_CA3': no_double_cells_CA3,
                             '1d_FR_thresh_perc': small_gauss_thresh_perc,
                             'CA_region': CA_region,
                             'xy': xy,
                             'double_cell_files': double_cell_files,
                             'no_double_cell_files': no_double_cell_files,
                             'double_cell_direc': double_cell_direc,
                             'no_double_cell_direc': no_double_cell_direc,
                             'Gauss_L_width05': Gauss_L_width05,
                             'Gauss_S_width05': Gauss_S_width05,
                             'Gauss_Combi_width05': Gauss_Combi_width05,
                             'Gauss_L_maxFR05': Gauss_L_maxFR05,
                             'Gauss_S_maxFR05': Gauss_S_maxFR05,
                             'Gauss_Combi_maxFR05': Gauss_Combi_maxFR05,
                             'Gauss_L_width15': Gauss_L_width15,
                             'Gauss_S_width15': Gauss_S_width15,
                             'Gauss_Combi_width15': Gauss_Combi_width15,
                             'Gauss_L_maxFR15': Gauss_L_maxFR15,
                             'Gauss_S_maxFR15': Gauss_S_maxFR15,
                             'Gauss_Combi_maxFR15': Gauss_Combi_maxFR15}

    hickle.dump(delta_and_weight_info, path+'Summary/delta_and_weight_info.hkl', mode='w')


def trafo(x, y, al):
    x_dreh = -4./3
    y_dreh = -2.
    x_new = x_dreh+((numpy.cos(al))*(x-x_dreh))+((-numpy.sin(al))*(y-y_dreh))
    y_new = y_dreh+((numpy.sin(al))*(x-x_dreh))+((numpy.cos(al))*(y-y_dreh))
    return x_new, y_new


def trafo2(x, y, rhomb_angle, rhomb_toX):
    x_dreh = -4./3
    y_dreh = -2.
    x0 = x-x_dreh
    y0 = y-y_dreh
    R = numpy.sqrt((x0**2)+(y0**2))
    # print 'R = ', R
    na = []
    for i in numpy.arange(len(x0)):

        if x0[i] == 0:
            new_angle = 0
        else:
            new_angle = (numpy.arctan(y0[i]/x0[i])-rhomb_toX)*(numpy.pi/2.)/rhomb_angle
        na.append(new_angle)
    new_angle = numpy.array(na)
    print 'new_angle = ', new_angle
    x_neu = R*numpy.cos(new_angle) + x_dreh
    y_neu = R*numpy.sin(new_angle) + y_dreh
    return x_neu, y_neu


def plot_deltas(data, examples=None, xlim=None, ylim=None, fz=30, double=False):

    prop_color = '#4575b4'
    vis_color = '#d73027'
    rem_color = '#8e6701'

    if examples:
        list_virt_real_cell = [[0], ]*len(examples)

    sns.set(style="ticks", font_scale=1.8)
    # sns.set_context("talk")
    x_label = 'Treadmill field center $\Delta$ (m)'
    y_label = 'Virtual field center $\Delta$ (m)'

    # get data

    x = numpy.array(data['Field_deltas_real'])
    y = numpy.array(data['Field_deltas_virt'])
    w = numpy.array(data['Field_weights'])
    ca = numpy.array(data['CA_region'])
    xy = numpy.array(data['xy'])
    f = numpy.array(data['used_files'])

    xr05 = numpy.array(data['Field_x_real05'])
    xr15 = numpy.array(data['Field_x_real15'])
    xv05 = numpy.array(data['Field_x_virt05'])
    xv15 = numpy.array(data['Field_x_virt15'])

    double_files_vis = numpy.array(data['double_cell_files'])
    double_files_direc = numpy.array(data['double_cell_direc'])

    # create array with unique numbers for units coming from the same cluster (double-cell)

    connection = numpy.repeat(numpy.arange(len(x)/2.), 2)

    # remove nans from data

    xy = xy[numpy.logical_not(numpy.isnan(x))]
    ca = ca[numpy.logical_not(numpy.isnan(x))]
    connection = connection[numpy.logical_not(numpy.isnan(x))]
    f = f[numpy.logical_not(numpy.isnan(x))]
    xr05 = xr05[numpy.logical_not(numpy.isnan(x))]
    xr15 = xr15[numpy.logical_not(numpy.isnan(x))]
    xv05 = xv05[numpy.logical_not(numpy.isnan(x))]
    xv15 = xv15[numpy.logical_not(numpy.isnan(x))]
    x = x[numpy.logical_not(numpy.isnan(x))]
    y = y[numpy.logical_not(numpy.isnan(y))]
    w = w[numpy.logical_not(numpy.isnan(w))]

    double_files = []
    for index, v in enumerate(double_files_vis):
        double_files.append(v.split('.hkl')[0]+'_'+double_files_direc[index])
        double_files.append(v.split('.hkl')[0]+'_normalised_'+double_files_direc[index])

    if not double:
        # remove double cell fields from data
        not_double_idx = numpy.array([numpy.where(f == dou)[0][0] for dou in f if not dou in double_files])

        xy = xy[not_double_idx]
        ca = ca[not_double_idx]
        connection = connection[not_double_idx]
        f = f[not_double_idx]
        x = x[not_double_idx]
        y = y[not_double_idx]
        w = w[not_double_idx]
        xr05 = xr05[not_double_idx]
        xr15 = xr15[not_double_idx]
        xv05 = xv05[not_double_idx]
        xv15 = xv15[not_double_idx]

    # get indexes of example visual and treadmill tuned cells
    visual_cell = numpy.array(['10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_PF_info_normalised_right',
                               '10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_PF_info_right'])
    treadmill_cell = numpy.array(['10529_2015-03-26_VR_linTrack2_TT3_SS_12_PF_info_normalised_right',
                                  '10529_2015-03-26_VR_linTrack2_TT3_SS_12_PF_info_right'])
    test_cell = numpy.array(['10823_2015-07-27_VR_GCend_linTrack1_TT3_SS_06_PF_info_normalised_right',
                               '10823_2015-07-27_VR_GCend_linTrack1_TT3_SS_06_PF_info_right'])
    double_cell = numpy.array(['10823_2015-07-03_VR_GCendOL_linTrack1_TT3_SS_18_PF_info_right',
                               '10823_2015-07-03_VR_GCendOL_linTrack1_TT3_SS_18_PF_info_normalised_right'])
    visual_exp_idx = numpy.array([numpy.where(f == vc)[0][0] for vc in f if vc in visual_cell])
    treadmill_exp_idx = numpy.array([numpy.where(f == tc)[0][0] for tc in f if tc in treadmill_cell])
    test_exp_idx = numpy.array([numpy.where(f == tc)[0][0] for tc in f if tc in test_cell])
    double_exp_idx = numpy.array([numpy.where(f == tc)[0][0] for tc in f if tc in double_cell])

    # coordiate transform to rhombus coordinates
    # x_trans = (2*x)+(2*y)
    # y_trans = x + (3*y)
    #
    y_prime_to_y = numpy.arccos(3./numpy.sqrt(13))
    x_prime_to_x = numpy.arccos(2./numpy.sqrt(5))
    angle_rhombus = numpy.arccos(7/numpy.sqrt(65))
    x_trans, y_trans = x, y
    # x_trans, y_trans = trafo(x=x, y=y, al=y_prime_to_y)
    # x_trans, y_trans = trafo2(x=x_trans, y=y_trans, rhomb_angle=angle_rhombus, rhomb_toX=x_prime_to_x)


    # find indices of no-double-cells
    uni = numpy.array(numpy.unique(connection, return_counts=True))
    no_double_cells = numpy.argsort(connection)[numpy.searchsorted(connection[numpy.argsort(connection)],
                                                                   uni[0][numpy.where(uni[1] == 1)[0]])]
    double_cells = numpy.array([d for d in numpy.arange(len(x)) if not d in no_double_cells])

    # replace no-double-cell values with nan
    x_double = x.copy()
    y_double = y.copy()
    x_double[no_double_cells] = numpy.nan
    y_double[no_double_cells] = numpy.nan
    connection[no_double_cells] = numpy.nan

    # separate all double cells with nan
    nan_insert_idx = numpy.where(numpy.diff(connection) == 1)[0]+1
    x_double = numpy.insert(x_double, nan_insert_idx, numpy.nan)
    y_double = numpy.insert(y_double, nan_insert_idx, numpy.nan)
    connection = numpy.insert(connection, nan_insert_idx, numpy.nan)

    plot_color = 'k'

    # main plot_________________________________

    # fig_all, ax_all = pl.subplots(1, 2, figsize=(20, 10))
    # ax_all = ax_all.flatten()

    J = (sns.jointplot(x_trans, y_trans, size=14, ratio=9, color=plot_color))  # , joint_kws={'alpha': w})

    J.ax_joint.set_position([.12, .12, .7, .7])
    J.ax_marg_x.set_position([.12, .82, .7, .13])
    J.ax_marg_y.set_position([.82, .12, .13, .7])


    #Clear the axes containing the scatter plot
    J.ax_joint.cla()

    #set x and y labels
    J.set_axis_labels(x_label, y_label, fontsize=fz)

    #Generate color array
    color_m = numpy.tile((1, 0, 0), (len(w), 1))
    color_v = numpy.diag(w)
    color_array = numpy.dot(color_v, color_m)

    # Define a colormap with the right number of colors
    cmap = pl.cm.get_cmap('jet', max(w)-min(w)+1)

    #Plot each individual point separately
    # for i in numpy.arange(len(x)):
    J.x = x
    J.y = y
    # J.plot_joint(pl.scatter, color=cm.jet(w), marker='o', size=10)
    # J.ax_joint.plot(x, y, c=cm.jet(w), marker='o')

    # find rows where [x,y] are closest tp example points
    rows = []
    exp_points = []
    for i, ex in enumerate(examples):
        if len(examples) == 3 and i == 0:
            for t in test_exp_idx:
                rows.append(t)
                exp_points.append([x_trans[t], y_trans[t]])
        else:
            x_min = abs(x_trans-ex[0])
            y_min = abs(y_trans-ex[1])
            all_min = x_min+y_min
            row = numpy.argmin(all_min)
            rows.append(row)
            exp_points.append([x_trans[row], y_trans[row]])
    print 'closest example points: '
    print 'x = ', x_trans[rows]
    print 'y= ', y_trans[rows]

    # define the three areas of mapping

    prop_patch = numpy.array([[-4./3, -2], [4./3, -2./3], [0, 0]])
    vis_patch = numpy.array([[4./3, -2./3], [8./3, 0], [4./3, 2./3], [0, 0]])
    rem_patch = numpy.array([[8./3, 0], [4, 2], [4./3, 2./3]])

    # prop and vis patch categorisation via intersection angle of fitted gaussians _____________________

    intersec_angle = 51.48 + 2  # from single fields
    # intersec_angle = 53.128 + 2  # from double fields
    # intersec_angle = 52.128 + 2  # from single and double fields

    betha = 180 - 90 - intersec_angle
    intersec_x = (2./numpy.sin(numpy.radians(intersec_angle)))*numpy.sin(numpy.radians(betha))

    tread_x = (intersec_x/.5) - (2./1.5)
    virt_y = intersec_x - 2.

    prop_patch = numpy.array([[-4./3, -2], [0, 0], [tread_x, virt_y]])  # [x, y] in the rhombus
    vis_patch = numpy.array([[0, 0], [tread_x, virt_y], [8./3, 0], [tread_x, .5*tread_x]])
    rem_patch = numpy.array([[8./3, 0], [4, 2], [tread_x, .5*tread_x]])

    # ---------------------------------------------------------------------------------------------------

    prop = matplotlib.path.Path(prop_patch)
    vis = matplotlib.path.Path(vis_patch)
    rem = matplotlib.path.Path(rem_patch)

    prop_idx = []
    vis_idx = []
    rem_idx = []
    plot_color = []
    markerCA = []
    msize = []
    alpha_value = 0.6
    ca1 = 0
    ca3 = 0

    for i in numpy.arange(len(x)):

        n = 3

        if i in visual_exp_idx:
            plot_color.append(vis_color)
            alpha_value = 1
            # msize.append(15)

        elif i in treadmill_exp_idx:
            plot_color.append(prop_color)
            alpha_value = 1
            # msize.append(15)

        if examples:
            if i in rows:
                r = numpy.where(rows == i)[0][0]
                list_virt_real_cell[r] = xy[i]
                if i not in [visual_exp_idx, treadmill_exp_idx]:
                    plot_color.append(rem_color)  #'#00ffff'
                alpha_value = 1
                msize.append(15)
                if i == rows[0]:
                    print 'first example file = ', f[rows[0]]
            else:
                msize.append(10)
        else:
            msize.append(10)

        if prop.contains_point([x_trans[i], y_trans[i]]) \
                or numpy.around(y_trans[i], n) == numpy.around((0.5*x_trans[i]) - 4./3, n) and x_trans[i] < 4./3 \
                or numpy.around(y_trans[i], n) == numpy.around((3./2)*x_trans[i], n):
            if not i in treadmill_exp_idx:
                plot_color.append('#4575b4')
            prop_idx.append(i)
        elif vis.contains_point([x_trans[i], y_trans[i]]) \
                or numpy.around(y_trans[i], n) == numpy.around((0.5*x_trans[i]) - 4./3, n) and 4./3 < x_trans[i] < 8./3 \
                or numpy.around(y_trans[i], n) == numpy.around(0.5*x_trans[i], n) and x_trans[i] < 4./3:
            if not i in visual_exp_idx:
                plot_color.append('#d73027')
            vis_idx.append(i)
        elif rem.contains_point([x_trans[i], y_trans[i]]) \
                or numpy.around(y_trans[i], n) == numpy.around(0.5*x_trans[i], n) and x_trans[i] > 4./3 \
                or numpy.around(y_trans[i], n) == numpy.around((3./2)*x_trans[i] - 4, n):
            plot_color.append('#8e6701')  # = (0.5568627450980392, 0.403921568627451, 0.00392156862745098)
            rem_idx.append(i)
            # if f[i].startswith('10823_2015-08-03_VR_GCend_linTrack1_TT2_SS_09_PF_info'):
            #     print 'x, y = ', x_trans[i], y_trans[i]
            #     print f[i]
            #     sys.exit()
        else:
            print f[i]
            print x_trans[i], y_trans[i]
            print 'Problem in line 3324'
            sys.exit()

        # animal_colors = []
        # if f[i].startswith('10823'):
        #     plot_color = '#7fc97f'
        #     animal_colors.append(plot_color)
        # elif f[i].startswith('10529'):
        #     plot_color = '#beaed4'
        #     animal_colors.append(plot_color)
        # elif f[i].startswith('10528'):
        #     plot_color = '#fdc086'
        #     animal_colors.append(plot_color)
        # elif f[i].startswith('10353'):
        #     plot_color = '#ffff99'
        #     animal_colors.append(plot_color)
        # elif f[i].startswith('10535'):
        #     plot_color = '#386cb0'
        #     animal_colors.append(plot_color)
        # elif f[i].startswith('10537'):
        #     plot_color = '#f0027f'
        #     animal_colors.append(plot_color)
        # else:
        #     print f[i]
        #     sys.exit()

        if ca[i] == 'CA1':
            markerCA.append('^')
        elif ca[i] == 'CA3':
            markerCA.append('o')
        else:
            print 'WARNING no CA region marker defined!'
            sys.exit()

        if w[i] > 0.0:
            if w[i] != 1:
                print 'i = ', i
                print 'w[i]', w[i]
                print 'f[i]', f[i]
            J.ax_joint.plot(x_trans[i], y_trans[i], color=plot_color[i], marker=markerCA[i], markersize=msize[i], alpha=alpha_value)  #, alpha_value alpha=w[i])

            if ca[i] == 'CA1':
                ca1 += 1
            elif ca[i] == 'CA3':
                ca3 += 1
        # plot_color = 'k'

    J.ax_joint.tick_params(labelsize=fz)
    J.ax_joint.xaxis.labelpad = 20

    # J.ax_joint.plot(x_double, y_double, color='b', linewidth=0.4, alpha=1)

    # cax = J.fig.add_axes([0.7, .15, .03, .3])  # size and placement of bar
    # # # J.ax_joint.set_clim([0, 1])
    # #
    # m = cm.ScalarMappable(cmap=cm.jet)
    # m.set_array(w)
    # pl.colorbar(m, cax=cax)  #, ticks=numpy.array(w)+0.5)

    # plot histograms
    sns.set(style="white")
    J.ax_marg_x.cla()
    J.ax_marg_y.cla()

    binwidth = 0.25
    if xlim and ylim:
        binwidth1 = (binwidth/(xlim[1]-xlim[0]))*(ylim[1]-ylim[0])
    else:
        binwidth1 = binwidth

    x = x_trans
    y = y_trans

    hist_plot_color = custom_plot.grau

    J.ax_marg_x.hist(x, weights=numpy.ones_like(x)/len(x), color=hist_plot_color,
                     alpha=0.6, bins=numpy.arange(min(x), max(x), binwidth), normed=0, edgecolor="none")
    J.ax_marg_x.xaxis.set_visible(False)
    J.ax_marg_x.yaxis.set_visible(False)

    J.ax_marg_y.hist(y, weights=numpy.ones_like(y)/len(y), color=hist_plot_color,
                     alpha=0.6, bins=numpy.arange(min(y), max(y), binwidth1), orientation="horizontal", normed=0, edgecolor="none")
    J.ax_marg_y.xaxis.set_visible(False)
    J.ax_marg_y.yaxis.set_visible(False)

    if xlim:
        J.ax_joint.set_xlim(xlim)
        J.ax_marg_x.set_xlim(xlim)

    if ylim:
        J.ax_joint.set_ylim(ylim)
        J.ax_marg_y.set_ylim(ylim)


    # Polygon lines______________________________

    x_poly = numpy.array([0, 8./3, 0, 0, 0, 4, 8./3, 4, -2, 8./3, -2, 0])
    y_poly = numpy.array([0, 0, -4./3, 0, 0, 2, 0, 2, -7./3., 0, -3, 0])

    x_poly_t = (2*x_poly)+(2*y_poly)
    y_poly_t = x_poly + (3*y_poly)

    x_poly_t = x_poly
    y_poly_t = y_poly

    # x_poly_t, y_poly_t = trafo(x=x_poly, y=y_poly, al=y_prime_to_y)
    # x_poly_t, y_poly_t = trafo2(x=x_poly_t, y=y_poly_t, rhomb_angle=angle_rhombus, rhomb_toX=x_prime_to_x)

    xremMid = numpy.array([8./3, 8./3])
    yremMid = numpy.array([0, 4./3])
    xremBorder = numpy.array([4./3, 8./3])
    yremBorder = numpy.array([2./3, 0])
    xpropvis = numpy.array([0, 4./3])
    ypropvis = numpy.array([0, -2./3])

    J.ax_joint.plot(x_poly_t[0:2], y_poly_t[0:2], '--', color=custom_plot.grau3, zorder=0)
    J.ax_joint.plot(x_poly_t[2:4], y_poly_t[2:4], ':', color=custom_plot.grau3, zorder=0)
    J.ax_joint.plot(xremMid, yremMid, color=custom_plot.grau3, zorder=0)
    J.ax_joint.plot(xpropvis, ypropvis, color=custom_plot.grau3, zorder=0)  # y =-0.5x
    J.ax_joint.plot(xremBorder, yremBorder, color=custom_plot.grau3, zorder=0)  # y =-0.5x+4./3

    J.ax_joint.plot(x_poly_t[4:6], y_poly_t[4:6], color='k', zorder=0)  # y = 1/2 x
    J.ax_joint.plot(x_poly_t[6:8], y_poly_t[6:8], color='k', zorder=0)   # y = 3/2 x - 4
    J.ax_joint.plot(x_poly_t[8:10], y_poly_t[8:10], color='k', zorder=0)  # y = 1/2 x - 4/3 -> x-Achse
    J.ax_joint.plot(x_poly_t[10:12], y_poly_t[10:12], color='k', zorder=0)  # y = 3/2 x     -> y-Achse


    # classification areas as patches____________________
    c1 = '#a6bddd'  #custom_plot.pretty_colors_set2[0]
    c2 = '#ea8f8a'  #custom_plot.pretty_colors_set2[5]
    c3 = '#fee090'  #custom_plot.pretty_colors_set2[6]

    triangle = mpatches.Polygon(prop_patch, color=c1, alpha=0.3, zorder=-1)
    polygon = mpatches.Polygon(vis_patch, color=c2, alpha=0.3, zorder=-1)
    triangle2 = mpatches.Polygon(rem_patch, color=c3, alpha=0.3, zorder=-1)
    J.ax_joint.add_artist(triangle)
    J.ax_joint.add_artist(polygon)
    J.ax_joint.add_artist(triangle2)


    # chen rhombus______________________
    # x_poly_chen = numpy.array([0, 0, 0, 0, 0, 0.67, 0, 0.67, -0.67, 0, -0.67, 0])
    # y_poly_chen = numpy.array([0, 0, -0.335, 0, 0, 0.335, -0.335, 0.335, -0.67, -0.335, -0.67, 0])
    #
    # J.ax_joint.plot(x_poly_chen[4:6], y_poly_chen[4:6], color='g', zorder=0)  # y = 1/2 x
    # J.ax_joint.plot(x_poly_chen[6:8], y_poly_chen[6:8], color='g', zorder=0)   # y = x + 0.67
    # J.ax_joint.plot(x_poly_chen[8:10], y_poly_chen[8:10], color='g', zorder=0)  # y = 1/2 x + 0.335 -> x-Achse
    # J.ax_joint.plot(x_poly_chen[10:12], y_poly_chen[10:12], color='g', zorder=0)  # y = x           -> y-Achse


    # put legend
    dca1 = data['num_double_cells_CA1']
    ndca1 = data['num_no_double_cells_CA1']
    dca3 = data['num_double_cells_CA3']
    ndca3 = data['num_no_double_cells_CA3']
    rdot = Line2D([0], [0], linestyle="none", marker="o", markersize=10, markerfacecolor='k')
    rtri = Line2D([0], [0], linestyle="none", marker="^", markersize=10, markerfacecolor='k')
    no = Line2D([0], [0], linestyle="none", marker="o", markersize=1, markerfacecolor='w')
    J.ax_joint.legend([rtri, rdot], ['CA1 place fields n = '+str(ca1),  #str(len(numpy.where(ca == 'CA1')[0])),
                                         # +'\nsingle n = '+str(ndca1)
                                         # +'\ndouble n = '+str(dca1),
                                     'CA3 place fields n = '+str(ca3)],  #str(len(numpy.where(ca == 'CA3')[0]))],
                                         # +'\nsingle n = '+str(ndca3)
                                         # +'\ndouble n = '+str(dca3)
                                     numpoints=1, bbox_to_anchor=(0.4, 1.0), fontsize=20)
                                         # '1D FR threshold = 0.8 and '+str(data['1d_FR_thresh'])+' Hz'],
                                         # numpoints=1, loc=4, fontsize=fz)

    #  Total number of place fields, n = "+str(len(x))+'\n double field n = '+ str(info['num_double_cells'])+
    # '\n single field n = '+str(info['num_no_double_cells'])+
    # '\n 1D FR threshold = '+str(info['1d_FR_thresh'])+' Hz']

    # save hickle_______________________________

    # info = {'prop_files': f[prop_idx], 'vis_files': f[vis_idx], 'rem_files': f[rem_idx], 'all_files': f,
    #         'prop_idx': prop_idx, 'vis_idx': vis_idx, 'rem_idx': rem_idx, 'ca': ca,  # 'animal_colors': animal_colors,
    #         # 'double_cell_idx': double_cells,
    #         'double_cell_files': double_files}  #f[double_cells]}
    # print 'Dumping prop_vis_rem_filenames.hkl under:'+path+'Summary/prop_vis_rem_filenames.hkl'
    # hickle.dump(info, path+'Summary/prop_vis_rem_filenames.hkl', mode='w')

    # save figure_______________________________

    if not double:
        name_multi = 'Multi_peak_noDouble.pdf'
    else:
        name_multi = 'Multi_peak.pdf'

    print 'Saving figure under:'+path+'Summary/'+name_multi
    pl.savefig(path+'Summary/'+name_multi, format='pdf')
    # pl.close('all')

    # halign = ['left', 'right', 'right']
    halign = ['right', 'right']

    ex_fig, ex_ax = plot_example_cells(list_virt_real_cell=list_virt_real_cell, used_points=exp_points, halign=halign)

    # fig_slope, ax_slope = pl.subplots(1, 1, figsize=(10, 9))
    # # ax_slope1 = ax_slope.twinx()
    # delta_treadmill = x_trans
    # delta_visuell = y_trans
    #
    # x_test = xv15       # xr05, xr15, xv05 or xv15
    # x_la = 'V15'        # T05 or T15 or V05 or V15
    # test_gain = '1.5'   # 0.5 or 1.5
    # posi = 'Virtual'  # Treadmill or Virtual
    # maximum = 2./1   # 2/gain or 2
    # norm = 2
    #
    # # global xr05, xr15, xv05, xv15, delta_visuell, delta_treadmill
    #
    # # (4./3), (8./3)
    # # if x_test array is in treadmill coordinates (xr05 or xr15)
    # # used for Vis_prop_slopesT05_indivNorm.pdf and Vis_prop_slopesT15_indivNorm.pdf
    # if x_la[0] == 'T' and norm == 0:
    #     ax_slope.scatter(x_test, delta_visuell/(-x_test), color=vis_color, s=30)
    #     ax_slope.scatter(x_test, delta_treadmill/4., color=prop_color, s=30)  # y in [0, 1]
    #     ending = '_indivNorm.pdf'
    #
    # # if x_test array is in virtual coordinates (xv05 or xv15)
    # # used for Vis_prop_slopesV05_indivNorm.pdf and Vis_prop_slopesV15_indivNorm.pdf
    # if x_la[0] == 'V' and norm == 0:
    #     ax_slope.scatter(x_test, delta_visuell/2., color=vis_color, s=30)
    #     ax_slope.scatter(x_test, delta_treadmill/((4./3)*x_test), color=prop_color, s=30)
    #     ending = '_indivNorm.pdf'
    #
    # # Vis_prop_slopesV05_norm.pdf
    # if norm == 1:
    #     ax_slope.scatter(x_test, delta_visuell/2., color=vis_color, s=30)
    #     ax_slope.scatter(x_test, delta_treadmill/4., color=prop_color, s=30)
    #     ending = '_norm.pdf'
    #
    # if norm == 2:
    #     ax_slope.scatter(x_test, delta_visuell/x_test, color=vis_color, s=30)
    #     ax_slope.scatter(x_test, delta_treadmill/x_test, color=prop_color, s=30)
    #     ending = '_Xnorm.pdf'
    #
    # ax_slope.set_xlim([0, maximum])
    #
    # ax_slope.set_xlabel(posi+' position with gain '+test_gain+' (m)', fontsize=fz)
    # ax_slope.set_ylabel('Virtual or treadmill field center $\Delta$ (m)', fontsize=fz)
    # print 'Saving figure under:'+path+'Summary/Vis_prop_slopes'+x_la+ending
    # fig_slope.savefig(path+'Summary/Vis_prop_slopes'+x_la+ending, format='pdf')


def callback(event, f, xv, yv):
    file = f[event.ind][0]
    file1 = file.split('_PF')[0]
    print file
    print 'x = ', xv[event.ind][0]
    print 'y = ', yv[event.ind][0]
    os.system('open -a Preview.app /Users/haasolivia/Desktop/plots/all_reduced_plots/'+file1+'*')


def compare_gains(data, examples=None, xlimr=None, ylimr=None, xlimv=None, ylimv=None, fz=30, double=False,
                  only_ca1=False, only_ca3=False, hist_corrected=False, bidirec=False, only_double=False,
                  only_bidirec=False, connected=False, bsp=False, background=False, confidence=False):
    if examples:
        list_virt_real_cell = [[0], ]*len(examples)

    sns.set(style="ticks", font_scale=1.8)

    x_label_r = 'Treadmill position at Gain 0.5 (m)'
    y_label_r = 'Treadmill position at Gain 1.5 (m)'
    x_label_v = 'Virtual position at Gain 0.5 (m)'
    y_label_v = 'Virtual position at Gain 1.5 (m)'
    x_label = [x_label_r, x_label_v]
    y_label = [y_label_r, y_label_v]

    # get fitted data for double fields

    xr = numpy.array(data['Field_x_real05'])
    yr = numpy.array(data['Field_x_real15'])
    xv = numpy.array(data['Field_x_virt05'])
    yv = numpy.array(data['Field_x_virt15'])

    # get real data for single fields

    xrs = numpy.array(data['Field_x_real05_data'])
    yrs = numpy.array(data['Field_x_real15_data'])
    xvs = numpy.array(data['Field_x_virt05_data'])
    yvs = numpy.array(data['Field_x_virt15_data'])

    xrs = numpy.repeat(xrs, 2)
    xrs[1::2] = numpy.nan
    yrs = numpy.repeat(yrs, 2)
    yrs[1::2] = numpy.nan
    xvs = numpy.repeat(xvs, 2)
    xvs[1::2] = numpy.nan
    yvs = numpy.repeat(yvs, 2)
    yvs[1::2] = numpy.nan

    w = numpy.array(data['Field_weights'])
    ca = numpy.array(data['CA_region'])
    xy = numpy.array(data['xy'])
    f = numpy.array(data['used_files'])

    double_files_vis = numpy.array(data['double_cell_files'])
    double_files_direc = numpy.array(data['double_cell_direc'])

    double_files = []
    for index, v in enumerate(double_files_vis):
        double_files.append(v.split('.hkl')[0]+'_'+double_files_direc[index])
        double_files.append(v.split('.hkl')[0]+'_normalised_'+double_files_direc[index])

    # find double cell fields in data
    # fr = f[numpy.logical_not(numpy.isnan(xr))]
    # not_double_idx = numpy.array([numpy.where(fr == dou)[0][0] for dou in fr if not dou in double_files])
    #
    # print 'len(not_double_idx) = ', len(not_double_idx), 'len(xvs) = ', len(xvs[numpy.logical_not(numpy.isnan(xvs))])
    # sys.exit()

    # create array with unique numbers for units coming from the same cluster (double-cell)

    connection = numpy.repeat(numpy.arange(len(xv)/2.), 2)

    # remove 'bad' data

    vis_idx = []
    for index, v in enumerate(f):

        if v not in bad:
            vis_idx.append(index)
        else:
            print v

    xr = xr[vis_idx]
    yr = yr[vis_idx]
    xv = xv[vis_idx]
    yv = yv[vis_idx]
    w = w[vis_idx]
    ca = ca[vis_idx]
    xy = xy[vis_idx]
    f = f[vis_idx]
    connection = connection[vis_idx]

    xrs = xrs[vis_idx]
    yrs = yrs[vis_idx]
    xvs = xvs[vis_idx]
    yvs = yvs[vis_idx]

    # remove nans from data

    fr = f[numpy.logical_not(numpy.isnan(xr))]
    xy = xy[numpy.logical_not(numpy.isnan(xr))]
    ca = ca[numpy.logical_not(numpy.isnan(xr))]
    xrs = xrs[numpy.logical_not(numpy.isnan(xr))]
    yrs = yrs[numpy.logical_not(numpy.isnan(yr))]
    xr = xr[numpy.logical_not(numpy.isnan(xr))]
    yr = yr[numpy.logical_not(numpy.isnan(yr))]

    fv = f[numpy.logical_not(numpy.isnan(xv))]
    connection = connection[numpy.logical_not(numpy.isnan(xv))]
    xvs = xvs[numpy.logical_not(numpy.isnan(xv))]
    yvs = yvs[numpy.logical_not(numpy.isnan(yv))]
    xv = xv[numpy.logical_not(numpy.isnan(xv))]
    yv = yv[numpy.logical_not(numpy.isnan(yv))]
    w = w[numpy.logical_not(numpy.isnan(w))]

    # find double cell indices in data
    not_double_idx_r = numpy.array([numpy.where(fr == dou)[0][0] for dou in fr if not dou in double_files])
    not_double_idx_v = numpy.array([numpy.where(fv == dou)[0][0] for dou in fv if not dou in double_files])

    if list(not_double_idx_r) == list(not_double_idx_v):

        # replace single cell info with real data info (not gauss fitted!)
        xr[not_double_idx_r] = xrs[not_double_idx_r]
        yr[not_double_idx_r] = yrs[not_double_idx_r]
        xv[not_double_idx_v] = xvs[not_double_idx_v]
        yv[not_double_idx_v] = yvs[not_double_idx_v]
        w[not_double_idx_v] = w[not_double_idx_v]*0+1

    else:
        print 'list(not_double_idx_r) != list(not_double_idx_v)'
        sys.exit()

    if not double:
        # remove double cell fields from data
        fr = fr[not_double_idx_r]
        xy = xy[not_double_idx_r]
        ca = ca[not_double_idx_r]
        xr = xr[not_double_idx_r]
        yr = yr[not_double_idx_r]

        fv = fv[not_double_idx_v]
        connection = connection[not_double_idx_v]
        xv = xv[not_double_idx_v]
        yv = yv[not_double_idx_v]
        w = w[not_double_idx_v]

    else:

        double_idx_r = numpy.array([numpy.where(fr == dou)[0][0] for dou in fr if dou in double_files])
        double_idx_v = numpy.array([numpy.where(fv == dou)[0][0] for dou in fv if dou in double_files])


    # get indexes of example visual and treadmill tuned cells
    visual_cell = numpy.array(['10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_PF_info_normalised_right',
                               '10528_2015-03-16_VR_GCend_linTrack1_TT4_SS_01_PF_info_right'])
    treadmill_cell = numpy.array(['10529_2015-03-26_VR_linTrack2_TT3_SS_12_PF_info_normalised_right',
                                  '10529_2015-03-26_VR_linTrack2_TT3_SS_12_PF_info_right'])
    double_cell = numpy.array(['10823_2015-07-03_VR_GCendOL_linTrack1_TT3_SS_18_PF_info_right',
                               '10823_2015-07-03_VR_GCendOL_linTrack1_TT3_SS_18_PF_info_normalised_right'])

    visual_exp_idx = numpy.array([numpy.where(fr == vc)[0][0] for vc in fr if vc in visual_cell])
    treadmill_exp_idx = numpy.array([numpy.where(fr == tc)[0][0] for tc in fr if tc in treadmill_cell])
    double_exp_idx = numpy.array([numpy.where(fr == tc)[0][0] for tc in fr if tc in double_cell])

    if double:
        # find indices of no-double-cells
        uni = numpy.array(numpy.unique(connection, return_counts=True))
        no_double_cells = numpy.argsort(connection)[numpy.searchsorted(connection[numpy.argsort(connection)],
                                                                       uni[0][numpy.where(uni[1] == 1)[0]])]
        double_cells = numpy.array([d for d in numpy.arange(len(xv)) if not d in no_double_cells])

        # replace no-double-cell values with nan
        xv_double = xv.copy()
        yv_double = yv.copy()
        xv_double[no_double_cells] = numpy.nan
        yv_double[no_double_cells] = numpy.nan
        connection[no_double_cells] = numpy.nan
        xv_single = xv.copy()
        yv_single = yv.copy()
        xv_single[double_cells] = numpy.nan
        yv_single[double_cells] = numpy.nan

        # separate all double cells with nan
        nan_insert_idx = numpy.where(numpy.diff(connection) == 1)[0]+1
        xv_double = numpy.insert(xv_double, nan_insert_idx, numpy.nan)
        yv_double = numpy.insert(yv_double, nan_insert_idx, numpy.nan)
        connection = numpy.insert(connection, nan_insert_idx, numpy.nan)
        idx_pairs = numpy.where(xv_double*0 == 0)[0]

    # initiate plots
    plot_color = 'k'
    Jr = (sns.jointplot(xr, yr, size=11, ratio=9, color=plot_color))  # , joint_kws={'alpha': w})
    Jv = (sns.jointplot(xv, yv, size=11, ratio=9, color=plot_color))  # , joint_kws={'alpha': w})

    for l, J in enumerate([Jr, Jv]):
        J.ax_joint.set_position([.12, .12, .7, .7])
        J.ax_marg_x.set_position([.12, .82, .7, .13])
        J.ax_marg_y.set_position([.82, .12, .13, .7])

        #Clear the axes containing the scatter plot
        J.ax_joint.cla()

        #set x and y labels
        J.set_axis_labels(x_label[l], y_label[l], fontsize=fz)

    # find rows where [x,y] are closest tp example points
    rows = []
    exp_points_r = []
    exp_points_v = []
    diff_real = xr - yr
    diff_virt = xv - yv
    for ex in examples:
        x_min = abs(diff_real-ex[0])
        y_min = abs(diff_virt-ex[1])
        all_min = x_min+y_min
        row = numpy.argmin(all_min)
        rows.append(row)
        exp_points_r.append([xr[row], yr[row]])
        exp_points_v.append([xv[row], yv[row]])
    print 'closest example points: '
    print 'xr = ', xr[rows]
    print 'yr = ', yr[rows]
    print ' x should be ', xr[rows]-yr[rows]
    print 'xv = ', xv[rows]
    print 'yv = ', yv[rows]
    print ' y should be ', xv[rows]-yv[rows]
    print 'file names = ', fr[rows]

    xremMid = numpy.array([8./3, 8./3])
    yremMid = numpy.array([0, 4./3])
    xremBorder = numpy.array([4./3, 8./3])
    yremBorder = numpy.array([2./3, 0])
    xpropvis = numpy.array([0, 4./3])
    ypropvis = numpy.array([0, -2./3])

    remMid_V15_1 = (3*xremMid[0] - 6*yremMid[0])/4.
    remMid_V05_1 = yremMid[0] + remMid_V15_1
    remMid_V15_2 = (3*xremMid[1] - 6*yremMid[1])/4.
    remMid_V05_2 = yremMid[1] + remMid_V15_2

    remBorder_V15_1 = (3*xremBorder[0] - 6*yremBorder[0])/4.
    remBorder_V05_1 = yremBorder[0] + remBorder_V15_1
    remBorder_V15_2 = (3*xremBorder[1] - 6*yremBorder[1])/4.
    remBorder_V05_2 = yremBorder[1] + remBorder_V15_2

    propvis_V15_1 = (3*xpropvis[0] - 6*ypropvis[0])/4.
    propvis_V05_1 = ypropvis[0] + propvis_V15_1
    propvis_V15_2 = (3*xpropvis[1] - 6*ypropvis[1])/4.
    propvis_V05_2 = ypropvis[1] + propvis_V15_2

    prop_patchV = numpy.array([[propvis_V05_1, propvis_V15_1], [0, 2], [propvis_V05_2, propvis_V15_2]])
    vis_patchV = numpy.array([[propvis_V05_1, propvis_V15_1], [propvis_V05_2, propvis_V15_2],
                              [remBorder_V05_2, remBorder_V15_2], [remBorder_V05_1, remBorder_V15_1]])
    rem_patchV = numpy.array([[remBorder_V05_1, remBorder_V15_1], [remBorder_V05_2, remBorder_V15_2], [2, 0]])

    # prop and vis patch categorisation via intersection angle of fitted gaussians _____________________

    # intersec_angle = 51.48 + 2  # from single fields
    # intersec_angle = 53.128 + 2  # from double fields
    # intersec_angle = 52.128 + 2  # from single and double fields
    # intersec_angle = 56.31   # geometric cut
    intersec_angle = 53.6   # bootstrapping - 2.5, + 4.1

    if not confidence:
        intersec_anglel = intersec_angle
        intersec_angles = intersec_angle
    else:
        print 'CONFIDENCE INTERVAL REMOVED !'
        intersec_anglel = intersec_angle + 4.1
        intersec_angles = intersec_angle - 2.5

    bethal = 180 - 90 - intersec_anglel
    intersec_xl = (2./numpy.sin(numpy.radians(intersec_anglel)))*numpy.sin(numpy.radians(bethal))
    bethas = 180 - 90 - intersec_angles
    intersec_xs = (2./numpy.sin(numpy.radians(intersec_angles)))*numpy.sin(numpy.radians(bethas))

    prop_patchV = numpy.array([[0, 0], [0, 2], [intersec_xl, 2]])  # [x, y]
    intermed_patchV = numpy.array([[0, 0], [intersec_xl, 2], [intersec_xs, 2]])
    vis_patchV = numpy.array([[0, 0], [intersec_xs, 2], [2, 2], [2, 0]])

    # ---------------------------------------------------------------------------------------------------

    prop = matplotlib.path.Path(prop_patchV)
    vis = matplotlib.path.Path(vis_patchV)
    rem = matplotlib.path.Path(rem_patchV)
    intermed = matplotlib.path.Path(intermed_patchV)

    prop_patchP = prop_patchV.copy()
    vis_patchP = vis_patchV.copy()
    rem_patchP = rem_patchV.copy()
    intermed_patchP = intermed_patchV.copy()

    for pat in [prop_patchP, vis_patchP, rem_patchP]:
        pat[:, 0] /= 0.5
        pat[:, 1] /= 1.5

    propP = matplotlib.path.Path(prop_patchP)
    visP = matplotlib.path.Path(vis_patchP)
    remP = matplotlib.path.Path(rem_patchP)
    intermedP = matplotlib.path.Path(intermed_patchP)

    prop_color = '#4575b4'
    vis_color = '#d73027'
    rem_color = '#8e6701'
    double_color = '#31a354'

    prop_idx = []
    vis_idx = []
    rem_idx = []

    # define radial segments to see if radial histogram shifts from prop dominant to visual dominant
    donuts = []
    cs = ['b', 'r', 'g', 'y', 'b', 'r', 'g']
    len_diag = 2/numpy.sin(numpy.radians(45))
    segments = 4
    wi = len_diag/segments
    all_donuts = numpy.arange(wi, len_diag+wi, wi)[:segments]
    for c, patch in enumerate(all_donuts):
        # Ring sector Wedge(center_xy, radius_max, angle_min, angle_max)
        pt = mpatches.Wedge((0, 0), patch, 0, 90, width=wi, color=cs[c])
        p1 = pt.get_path()
        donuts.append(p1)
        # Jv.ax_joint.add_patch(pt)

    idx_donuts = [[] for d in xrange(len(donuts))]
    prop_donuts = [0, 0, 0, 0]
    vis_donuts = [0, 0, 0, 0]

    ca1 = 0
    ca3 = 0

    area = []

    if bidirec:
        file_beg = []
        bidriec_idx = []
        bidirec_area = []
        for i in numpy.arange(len(xr)):

            filei = fv[i].split('info_')
            norm = filei[1].split('_')[0]
            if norm != 'normalised':
                if i != 0:
                    if filei[0] == file_beg[-1]:
                        bidriec_idx.append(i-1)
                        bidriec_idx.append(i)
                file_beg.append(filei[0])

        xv_bidirec = xv.copy()
        yv_bidirec = yv.copy()
        xv_bidirec = xv_bidirec[bidriec_idx]
        yv_bidirec = yv_bidirec[bidriec_idx]
        xv_bidirec = numpy.insert(xv_bidirec, numpy.arange(2, len(xv_bidirec), 2), numpy.nan)
        yv_bidirec = numpy.insert(yv_bidirec, numpy.arange(2, len(yv_bidirec), 2), numpy.nan)

    for i in numpy.arange(len(xr)):

        n = 3

        if prop.contains_point([xv[i], yv[i]]):  # \
                # or numpy.around(yv[i], n) == numpy.around((0.5*xv[i]) - 4./3, n) and xv[i] < 4./3 \
                # or numpy.around(yv[i], n) == numpy.around((3./2)*xv[i], n):
            plot_colorV = '#4575b4'
            area.append('prop')
            prop_idx.append(i)
            if bidirec and i in bidriec_idx:
                bidirec_area.append('prop')

        elif vis.contains_point([xv[i], yv[i]]):  # \
                # or numpy.around(yv[i], n) == numpy.around((0.5*xv[i]) - 4./3, n) and 4./3 < xv[i] < 8./3 \
                # or numpy.around(yv[i], n) == numpy.around(0.5*xv[i], n) and xv[i] < 4./3:
            plot_colorV = '#d73027'
            area.append('vis')
            vis_idx.append(i)
            if bidirec and i in bidriec_idx:
                bidirec_area.append('vis')

        elif rem.contains_point([xv[i], yv[i]]):  # \
                # or numpy.around(yv[i], n) == numpy.around(0.5*xv[i], n) and xv[i] > 4./3 \
                # or numpy.around(yv[i], n) == numpy.around((3./2)*xv[i] - 4, n):
            plot_colorV = '#8e6701'  # = (0.5568627450980392, 0.403921568627451, 0.00392156862745098)
            area.append('vis')
            rem_idx.append(i)
            if bidirec and i in bidriec_idx:
                bidirec_area.append('vis')

        elif intermed.contains_point([xv[i], yv[i]]):
            plot_colorV = 'k'
            area.append('inter')
            if bidirec and i in bidriec_idx:
                bidirec_area.append('inter')

        else:
            print xv[i], yv[i]
            sys.exit()

        if propP.contains_point([xr[i], yr[i]]): # \
                # or numpy.around(yr[i], n) == numpy.around((0.5*xr[i]) - 4./3, n) and xr[i] < 4./3 \
                # or numpy.around(yr[i], n) == numpy.around((3./2)*xr[i], n):
            plot_colorP = '#4575b4'

        elif visP.contains_point([xr[i], yr[i]]): # \
                # or numpy.around(yr[i], n) == numpy.around((0.5*xr[i]) - 4./3, n) and 4./3 < xr[i] < 8./3 \
                # or numpy.around(yr[i], n) == numpy.around(0.5*xr[i], n) and xr[i] < 4./3:
            plot_colorP = '#d73027'

        elif remP.contains_point([xr[i], yr[i]]): # \
                # or numpy.around(yr[i], n) == numpy.around(0.5*xr[i], n) and xr[i] > 4./3 \
                # or numpy.around(yr[i], n) == numpy.around((3./2)*xr[i] - 4, n):
            plot_colorP = '#8e6701'  # = (0.5568627450980392, 0.403921568627451, 0.00392156862745098)

        elif intermedP.contains_point([xv[i], yv[i]]):
            plot_colorP = 'k'

        else:
            print xr[i], yr[i]
            sys.exit()

        # get points within the donuts

        for donut in numpy.arange(len(donuts)):

            if donuts[donut].contains_point([xv[i], yv[i]]):
                idx_donuts[donut].append(i)

                if prop.contains_point([xv[i], yv[i]]):
                    prop_donuts[donut] += 1
                else:
                    vis_donuts[donut] += 1

        # if fr[i].startswith('10823'):
        #     plot_colorP = '#7fc97f'
        # elif fr[i].startswith('10529'):
        #     plot_colorP = '#beaed4'
        # elif fr[i].startswith('10528'):
        #     plot_colorP = '#fdc086'
        # elif fr[i].startswith('10353'):
        #     plot_colorP = '#ffff99'
        # elif fr[i].startswith('10535'):
        #     plot_colorP = '#386cb0'
        # elif fr[i].startswith('10537'):
        #     plot_colorP = '#f0027f'
        # else:
        #     print fr[i]
        #     sys.exit()
        # if fv[i].startswith('10823'):
        #     plot_colorV = '#7fc97f'
        # elif fv[i].startswith('10529'):
        #     plot_colorV = '#beaed4'
        # elif fv[i].startswith('10528'):
        #     plot_colorV = '#fdc086'
        # elif fv[i].startswith('10353'):
        #     plot_colorV = '#ffff99'
        # elif fv[i].startswith('10535'):
        #     plot_colorV = '#386cb0'
        # elif fv[i].startswith('10537'):
        #     plot_colorV = '#f0027f'
        # else:
        #     print fv[i]
        #     sys.exit()

        if ca[i] == 'CA1':
            markerCA = '^'
        elif ca[i] == 'CA3':
            markerCA = 'o'
        else:
            print 'WARNING no CA region marker defined!'
            sys.exit()

        msize = 10

        plot_colorV = 'k'
        plot_colorP = 'k'

        alpha_value = 0.6
        order = 1

        edgecolor = 'k'

        if double:
            d_color = 'None'  #'#7fcdbb'
            if i in double_idx_r:
                plot_colorV = d_color
                plot_colorP = d_color
                order = 10
            elif i in double_idx_v:
                plot_colorV = d_color
                plot_colorP = d_color
                order = 10
            elif only_double:
                continue

        if only_bidirec:
            if i not in bidriec_idx:
                continue

        # if examples:
        #     if i in rows:
        #         r = numpy.where(rows == i)[0][0]
        #         list_virt_real_cell[r] = xy[i]
        #         plot_colorV = rem_color  #'#00ffff'
        #         plot_colorP = rem_color  #'#00ffff'
        #         alpha_value = 1
        #         msize = 15
        #
        if bsp:

            if i in visual_exp_idx:
                edgecolor = 'None'
                plot_colorV = vis_color
                plot_colorP = vis_color
                alpha_value = 1
                msize = 15

            if i in treadmill_exp_idx:
                i_t = i
                marker_t = markerCA
                # edgecolor = 'None'
                # plot_colorV = prop_color
                # plot_colorP = prop_color
                # alpha_value = 1
                # msize = 15

        if i in double_exp_idx:
            plot_colorV = double_color
            plot_colorP = double_color
            alpha_value = 1
            msize = 15
            order = 10

        if w[i] > 0.0:
            if only_ca1 and ca[i] == 'CA1':
                Jr.ax_joint.plot(xr[i], yr[i], markerfacecolor=plot_colorP, marker=markerCA, markersize=msize,
                                 markeredgecolor=edgecolor, markeredgewidth=1.0, zorder=order)  # alpha=w[i])
                Jv.ax_joint.plot(xv[i], yv[i], markerfacecolor=plot_colorV, marker=markerCA, markersize=msize,
                                 markeredgecolor=edgecolor, markeredgewidth=1.0, zorder=order)  #, alpha=w[i])
                ca1 += 1

            elif only_ca3 and ca[i] == 'CA3':
                Jr.ax_joint.plot(xr[i], yr[i], markerfacecolor=plot_colorP, marker=markerCA, markersize=msize,
                                 markeredgecolor=edgecolor, markeredgewidth=1.0, zorder=order)  # alpha=w[i])
                Jv.ax_joint.plot(xv[i], yv[i], markerfacecolor=plot_colorV, marker=markerCA, markersize=msize,
                                 markeredgecolor=edgecolor, markeredgewidth=1.0, zorder=order)  #, alpha=w[i])
                ca3 += 1

            elif not only_ca1 and not only_ca3:
                Jr.ax_joint.plot(xr[i], yr[i], markerfacecolor=plot_colorP, marker=markerCA, markersize=msize,
                                 markeredgecolor=edgecolor, markeredgewidth=1.0, zorder=order)  # alpha=w[i])
                Jv.ax_joint.plot(xv[i], yv[i], markerfacecolor=plot_colorV, marker=markerCA, markersize=msize,
                                 markeredgecolor=edgecolor, markeredgewidth=1.0, zorder=order)  #, alpha=w[i])
                plot_color = 'k'
                if ca[i] == 'CA1':
                    ca1 += 1
                elif ca[i] == 'CA3':
                    ca3 += 1

    # Marker for prop example was under other marker. So it will be plotted after all others:
    if bsp:
        Jr.ax_joint.plot(xr[i_t], yr[i_t], markerfacecolor=prop_color, marker=marker_t, markersize=15,
                         markeredgecolor='None', markeredgewidth=1.0, zorder=1)  # alpha=w[i])
        Jv.ax_joint.plot(xv[i_t], yv[i_t], markerfacecolor=prop_color, marker=marker_t, markersize=15,
                         markeredgecolor='None', markeredgewidth=1.0, zorder=1)

    # Jr.ax_joint.scatter(xr, yr, color='k', s=70)  #, color=[0, 0, 0], marker='o', markersize=10, alpha=w)  #  , alpha=alpha_value)
    # Jv.ax_joint.scatter(xv, yv, color='k', s=70, picker=5)  #, color=[0, 0, 0], marker='o', markersize=10, alpha=w)  #  , alpha=alpha_value)
    #
    # pl.gcf().canvas.mpl_connect('pick_event', lambda event: callback(event, fv, xv, yv))

    # save hickle_______________________________

    if double:

        info = {'prop_files': fv[prop_idx], 'vis_files': fv[vis_idx], 'rem_files': fv[rem_idx], 'all_files': fv,
                'prop_idx': prop_idx, 'vis_idx': vis_idx, 'rem_idx': rem_idx, 'ca': ca,  # 'animal_colors': animal_colors,
                # 'double_cell_idx': double_cells,
                'double_cell_files': double_files}  #f[double_cells]}
        print 'Dumping prop_vis_rem_filenames.hkl under:'+path+'Summary/prop_vis_rem_filenames.hkl'
        hickle.dump(info, path+'Summary/prop_vis_rem_filenames.hkl', mode='w')

    # plotting histogram of double cell category mixtures____________________________

    if double:

        area1 = numpy.insert(area, nan_insert_idx, numpy.nan)
        area2 = area1[idx_pairs]

        area3 = numpy.array(area2).flatten()
        area4 = numpy.split(area3, len(area3)/2)

        area5 = []
        visvis = []
        propprop = []
        mix = []
        for arnum, ar in enumerate(area4):
            if ar[0] == 'vis' and ar[1] == 'vis':
                area5.append('visual-visual')
                visvis.append(arnum)
            elif ar[0] == 'prop' and ar[1] == 'prop':
                area5.append('prop-prop')
                propprop.append(arnum)
            elif not ar[0] == 'inter' or ar[1] == 'inter':
                area5.append('mix')
                mix.append(arnum)

        letter_counts = Counter(area5)
        dic = {'prop-prop': 0}

        df = pandas.DataFrame.from_dict(letter_counts, orient='index')
        df = pandas.DataFrame.from_dict(dic, orient='index').append(df)
        fig_categories, ax = pl.subplots(figsize=(7, 7))

        df.plot(kind='bar', ax=ax, color=custom_plot.grau, edgecolor="none")

        ax.legend().set_visible(False)
        pl.setp(ax.xaxis.get_majorticklabels(), rotation=60, size=fz)
        pl.setp(ax.yaxis.get_majorticklabels(), size=fz)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.set_ylabel('Count', fontsize=fz)
        fig_categories.tight_layout()

    # _________________________________________________________________________________

        if connected:
            Jv.ax_joint.plot(xv_double, yv_double, color='k', linewidth=1)  #, alpha=1)

    if bidirec:
        if connected:
            Jv.ax_joint.plot(xv_bidirec, yv_bidirec, color='k', linewidth=1)  #, alpha=1)

        propvis = [[bidirec_area[::2][j], bidirec_area[1::2][j]] for j in numpy.arange(len(bidirec_area[::2]))]
        area5 = []
        for p in propvis:
            if p[0] == 'vis' and p[1] == 'vis':
                area5.append('visual-visual')
            elif p[0] == 'prop' and p[1] == 'prop':
                area5.append('prop-prop')
            elif not p[0] == 'inter' or p[1] == 'inter':
                area5.append('mix')
        letter_counts = Counter(area5)
        df = pandas.DataFrame.from_dict(letter_counts, orient='index')
        df = df.reindex(['prop-prop', 'visual-visual', 'mix'])

        fig_bidirec_categories, ax = pl.subplots(figsize=(7, 7))

        df.plot(kind='bar', ax=ax, color=custom_plot.grau, edgecolor="none")

        ax.legend().set_visible(False)
        pl.setp(ax.xaxis.get_majorticklabels(), rotation=60, size=fz)
        pl.setp(ax.yaxis.get_majorticklabels(), size=fz)

        labels = ax.get_yticks()
        difference = labels[-1]/3.
        ax.set_yticks(numpy.arange(labels[0], labels[-1]+difference, difference))
        ax.set_yticks([0, 2, 4])

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.set_ylabel('Count', fontsize=fz)
        fig_bidirec_categories.tight_layout()

    Jr.ax_joint.tick_params(labelsize=fz)
    Jv.ax_joint.tick_params(labelsize=fz)

    idx_ca_donuts = [[] for d in xrange(len(donuts))]
    if only_ca1:
        idx_ca = numpy.where(ca == 'CA1')[0]
        for donut in numpy.arange(len(donuts)):
            idx_ca_donuts[donut].append(numpy.where(ca[idx_donuts[donut]] == 'CA1')[0])
    elif only_ca3:
        idx_ca = numpy.where(ca == 'CA3')[0]
        for donut in numpy.arange(len(donuts)):
            idx_ca_donuts[donut].append(numpy.where(ca[idx_donuts[donut]] == 'CA3')[0])
    else:
        idx_ca = numpy.arange(len(yr))
        for donut in numpy.arange(len(donuts)):
            idx_ca_donuts[donut].append(numpy.arange(len(idx_donuts[donut])))

    angles_r = 180*numpy.arctan(yr[idx_ca]/xr[idx_ca])/numpy.pi
    angles_v = 180*numpy.arctan(yv[idx_ca]/xv[idx_ca])/numpy.pi

    no_double_angles = angles_v

    angles_info = {'no_double_angles': no_double_angles}
    hickle.dump(angles_info, path+'Summary/angles_info_raw_data.hkl', mode='w')

    # plotting histogram of double cell angle differences____________________________
    binwidth1 = 4
    if double:

        a1 = numpy.insert(angles_v, nan_insert_idx, numpy.nan)
        a2 = a1[idx_pairs]

        a3 = numpy.array(a2).flatten()
        a4 = numpy.split(a3, len(a3)/2)

        a5 = numpy.abs(numpy.diff(a4))

        fig_angle_diffs, ax1 = pl.subplots()
        ax1.hist(a5, color=custom_plot.grau, edgecolor="none")

        pl.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
        # Hide the right and top spines
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')

        ax1.set_xlabel('Double cell angle difference (degree)')
        ax1.set_ylabel('Count')
        fig_angle_diffs.tight_layout()

        fig_angle_diffs1, ax2 = pl.subplots(2, sharex=True)
        # fig_angle_diffs1.subplots_adjust(wspace=None, hspace=None)
        ax2.flatten()

        ax2[0].hist(a5[visvis], color='#ea8f8a', edgecolor="none", bins=numpy.arange(0, 70, 5))
        # ax2[0].set_xlabel('visual-visual')
        ax2[1].hist(a5[mix], color=custom_plot.grau, edgecolor="none", bins=numpy.arange(0, 70, 5))

        for a in [0, 1]:
            # Hide the right and top spines
            ax2[a].spines['right'].set_visible(False)
            ax2[a].spines['top'].set_visible(False)

            # Only show ticks on the left and bottom spines
            ax2[a].yaxis.set_ticks_position('left')
            ax2[a].xaxis.set_ticks_position('bottom')
            labels = ax2[a].get_yticks()
            ax2[a].set_yticks(labels[numpy.array([0, 2, 4])])


        ax2[1].set_xlabel('Double cell angle difference (degree)')
        fig_angle_diffs1.tight_layout()

        no_double_angles = angles_v[no_double_cells]
        double_angles = angles_v[double_cells]

        # plot angle histograms for double and single field cells separately. ________

        fig_angle_dou, ax = pl.subplots()
        lw = 3
        angles_info = {'double_angles': double_angles, 'no_double_angles': no_double_angles}
        hickle.dump(angles_info, path+'Summary/angles_info.hkl', mode='w')

        print 'DONUTS len(single-double-vergleich) = ', len(numpy.arange(min(double_angles), max(double_angles) + binwidth1, binwidth1))

        his = ax.hist(double_angles, color='k', histtype='step', linewidth=lw,
                      bins=numpy.arange(min(double_angles), max(double_angles) + binwidth1, binwidth1))
        his1 = ax.hist(no_double_angles, color='k', edgecolor='none', alpha=0.5,
                       bins=numpy.arange(min(double_angles), max(double_angles) + binwidth1, binwidth1))
        height = numpy.array([float(h) for h in his[0]])
        height1 = numpy.array([float(h1) for h1 in his1[0]])
        print 'len(height)', len(height)
        print height
        print 'sum(height)', sum(height)
        print 'len(height1)', len(height1)
        print height1
        print 'sum(height1)', sum(height1)
        ax.axvline(71.56505118, linestyle=':', color=custom_plot.grau, zorder=-1)
        ax.axvline(45, linestyle='--', color=custom_plot.grau, zorder=-1)
        ax.set_xlim([0, 90])
        ax.set_xlabel('Angle in degrees', fontsize=fz)
        ax.set_ylabel('Count', fontsize=fz)

        # legend _____________________________________________________________________

        line1 = Line2D([0], [0], linestyle="-", linewidth=lw, color='k', alpha=0.5)
        line2 = Line2D([0], [0], linestyle="-", linewidth=lw, color='k')

        ax.legend([line1, line2], ['One field', 'Two fields'], numpoints=1, bbox_to_anchor=(.38, 1.0), fontsize=fz/1.5)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        labels = ax.get_xticks()
        ax.set_xticks(labels[::2])
        labels1 = ax.get_yticks()
        ax.set_yticks([8, 16, 24])

        ax.tick_params(labelsize=fz)

        fig_angle_dou.tight_layout()

        # p value for radial distribution ______________________

        # all together as in Fig 3c
        # a = numpy.concatenate((double_angles, no_double_angles))
        # his2 = ax.hist(a, color='g', edgecolor='none', alpha=0.5,
        #                bins=numpy.arange(min(a), max(a) + binwidth1, binwidth1))
        # chi2, p2 = scipy.stats.chisquare(his2[0])
        # print 'chi2, p2', chi2, p2

        chi0, p0 = scipy.stats.chisquare(his[0])
        print 'Double chi', chi0, ' p = ', p0
        chi1, p1 = scipy.stats.chisquare(his1[0])
        print 'Single chi', chi1, ' p = ', p1

        # histo = list(his[0].copy())
        # histo1 = list(his1[0].copy())

        # diff = abs(len(histo) - len(histo1))
        # if len(histo) < len(histo1):
        #     for le in numpy.arange(diff):
        #         histo.append(0)
        # else:
        #     for le in numpy.arange(diff):
        #         histo1.append(0)

        chi2, p2, dof, ex = scipy.stats.chi2_contingency(numpy.array([list(his[0]), list(his1[0])]))

        text_x = [0.2, 0.2, 0.81]
        text_y = [0.64, 0.47, 0.84]
        text_dy = [-0.07, -0.07, -0.07]
        text_dx = [0.018, 0.018, 0.018]

        frz = 20

        chi = [chi0, chi1, chi2]
        p = [p0, p1, p2]
        # alp = [0.5, 1, 1]
        alp = [1, .5, 1]

        for c in [0, 1, 2]:
            fig_angle_dou.text(text_x[c], text_y[c], '$\chi^2$= '+str(numpy.round(chi[c], 2)), fontsize=frz, color='k',
                               alpha=alp[c])
            # if p[c] < 0.001:
            #     fig_angle_dou.text(text_x[c]+text_dx[c], text_y[c]+text_dy[c], '$p$ < 0.001', fontsize=frz, color='k',
            #                        alpha=alp[c])
            # else:
            fig_angle_dou.text(text_x[c]+text_dx[c], text_y[c]+text_dy[c], '$p$ = '+str(p[c]),
                               fontsize=frz, color='k', alpha=alp[c])

    # _________________________________________________________________________________

    angles_ca_donuts = [[] for d in xrange(len(donuts))]
    for donut in numpy.arange(len(donuts)):
        index = numpy.array(idx_donuts[donut])[idx_ca_donuts[donut]]
        angles_ca_donuts[donut].append(180*numpy.arctan(yv[index]/xv[index])/numpy.pi)

    # _______________________ Fischer"s exact test between donut slices ____________________________

    oddsratio01, pvalue01 = scipy.stats.fisher_exact([[prop_donuts[0], prop_donuts[1]], [vis_donuts[0], vis_donuts[1]]])
    oddsratio02, pvalue02 = scipy.stats.fisher_exact([[prop_donuts[0], prop_donuts[2]], [vis_donuts[0], vis_donuts[2]]])
    oddsratio03, pvalue03 = scipy.stats.fisher_exact([[prop_donuts[0], prop_donuts[3]], [vis_donuts[0], vis_donuts[3]]])

    oddsratio12, pvalue12 = scipy.stats.fisher_exact([[prop_donuts[1], prop_donuts[2]], [vis_donuts[0], vis_donuts[2]]])
    oddsratio13, pvalue13 = scipy.stats.fisher_exact([[prop_donuts[1], prop_donuts[3]], [vis_donuts[0], vis_donuts[3]]])

    oddsratio23, pvalue23 = scipy.stats.fisher_exact([[prop_donuts[2], prop_donuts[3]], [vis_donuts[0], vis_donuts[3]]])

    print 'Fischers Exact'
    print 'n prop_donuts[0] = ', prop_donuts[0], 'n vis_donuts[0] = ', vis_donuts[0]
    print 'n prop_donuts[1] = ', prop_donuts[1], 'n vis_donuts[1] = ', vis_donuts[1]
    print 'n prop_donuts[2] = ', prop_donuts[2], 'n vis_donuts[1] = ', vis_donuts[2]
    print 'n prop_donuts[3] = ', prop_donuts[3], 'n vis_donuts[3] = ', vis_donuts[3]
    print 'pvalue01 = ', pvalue01
    print 'pvalue02 = ', pvalue02
    print 'pvalue03 = ', pvalue03
    print 'pvalue12 = ', pvalue12
    print 'pvalue13 = ', pvalue13
    print 'pvalue23 = ', pvalue23

    angles0 = numpy.array(angles_ca_donuts[0][0])
    angles1 = numpy.array(angles_ca_donuts[1][0])
    angles2 = numpy.array(angles_ca_donuts[2][0])
    angles3 = numpy.array(angles_ca_donuts[3][0])

    min01 = min(numpy.nanmin(angles0), numpy.nanmin(angles1))
    max01 = max(numpy.nanmax(angles0), numpy.nanmax(angles1))
    his01 = numpy.histogram(angles0, bins=numpy.arange(min01, max01 + binwidth1, binwidth1))
    his101 = numpy.histogram(angles1, bins=numpy.arange(min01, max01 + binwidth1, binwidth1))

    min02 = min(numpy.nanmin(angles0), numpy.nanmin(angles2))
    max02 = max(numpy.nanmax(angles0), numpy.nanmax(angles2))
    his02 = numpy.histogram(angles0, bins=numpy.arange(min02, max02 + binwidth1, binwidth1))
    his102 = numpy.histogram(angles2, bins=numpy.arange(min02, max02 + binwidth1, binwidth1))

    # problems-------

    min03 = min(numpy.nanmin(angles0), numpy.nanmin(angles3))
    max03 = max(numpy.nanmax(angles0), numpy.nanmax(angles3))
    his03 = numpy.histogram(angles0, bins=numpy.arange(min03, max03 + binwidth1, binwidth1))
    his103 = numpy.histogram(angles3, bins=numpy.arange(min03, max03 + binwidth1, binwidth1))

    min12 = min(numpy.nanmin(angles1), numpy.nanmin(angles2))
    max12 = max(numpy.nanmax(angles1), numpy.nanmax(angles2))
    his12 = numpy.histogram(angles1, bins=numpy.arange(min12, max12 + binwidth1, binwidth1))
    his112 = numpy.histogram(angles2, bins=numpy.arange(min12, max12 + binwidth1, binwidth1))

    min13 = min(numpy.nanmin(angles1), numpy.nanmin(angles3))
    max13 = max(numpy.nanmax(angles1), numpy.nanmax(angles3))
    his13 = numpy.histogram(angles1, bins=numpy.arange(min13, max13 + binwidth1, binwidth1))
    his113 = numpy.histogram(angles3, bins=numpy.arange(min13, max13 + binwidth1, binwidth1))

    min23 = min(numpy.nanmin(angles2), numpy.nanmin(angles3))
    max23 = max(numpy.nanmax(angles2), numpy.nanmax(angles3))
    his23 = numpy.histogram(angles2, bins=numpy.arange(min23, max23 + binwidth1, binwidth1))
    his123 = numpy.histogram(angles3, bins=numpy.arange(min23, max23 + binwidth1, binwidth1))


    # chi01, p01, dof01, ex01 = scipy.stats.chi2_contingency(numpy.array([list(his01[0][4:]), list(his101[0][4:])]))
    # chi02, p02, dof02, ex02 = scipy.stats.chi2_contingency(numpy.array([list(his02[0][4:]), list(his102[0][4:])]))
    # chi03, p03, dof03, ex03 = scipy.stats.chi2_contingency(numpy.array([list(his03[0]), list(his103[0])]))
    #
    # chi12, p12, dof12, ex12 = scipy.stats.chi2_contingency(numpy.array([list(his12[0]), list(his112[0])]))
    # chi13, p13, dof13, ex13 = scipy.stats.chi2_contingency(numpy.array([list(his13[0]), list(his113[0])]))
    #
    # chi23, p23, dof23, ex23 = scipy.stats.chi2_contingency(numpy.array([list(his23[0]), list(his123[0])]))

    print prop_donuts, vis_donuts

    print 'sum of points in donuts = ', sum([sum(numpy.array(prop_donuts).flatten()), sum(numpy.array(vis_donuts).flatten())])

    donut_p = {'oddsratio01': oddsratio01, 'pvalue01': pvalue01, 'oddsratio02': oddsratio02, 'pvalue02': pvalue02,
               'oddsratio03': oddsratio03, 'pvalue03': pvalue03, 'oddsratio12': oddsratio12, 'pvalue12': pvalue12,
               'oddsratio13': oddsratio13, 'pvalue13': pvalue13, 'oddsratio23': oddsratio23, 'pvalue23': pvalue23,
               'separation_angles': [intersec_angles, intersec_anglel]}
               # 'chi_p01': p01, 'chi_p02': p02, 'chi_p03': p03, 'chi_p12': p12, 'chi_p13': p13, 'chi_p23': p23,
               # 'separation_angles': [intersec_angles, intersec_anglel]}

    hickle.dump(donut_p, path+'Summary/donut_fischers_exact.hkl', mode='w')

    # plot angle histogram -----------------------------------------------------------

    ang = [angles_r, angles_v]
    parent_fig = [Jr.fig, Jv.fig]
    xlab = ['Treadmill angles (degree)', 'Virtual angles (degree)']
    name = ['Treadmill_angles', 'Virtual_angles']
    hlines = [[45, 180*numpy.arctan((2/1.5)/4)/numpy.pi], [180*numpy.arctan(2./((2./1.5)*0.5))/numpy.pi, 45]]

    for a in [0, 1]:

        binwidth = 4
        his = numpy.histogram(ang[a], bins=numpy.arange(min(ang[a]), max(ang[a]) + binwidth, binwidth))
        height = his[0]
        radii = his[1][:-1]*numpy.pi/180

        if a == 1:
            b = []
            for r in radii:
                if r < numpy.radians(45):
                    b.append(2.*numpy.sin(r)/numpy.sin(numpy.radians(180)-(r+numpy.radians(90))))
                else:
                    b.append(2.)
            lengths = b/numpy.sin(radii)
        else:
            lengths = 1
        width = binwidth*numpy.pi/180
        bottom = 8

        axA = parent_fig[a].add_axes([-0.23, -0.23, 0.70, 0.70], polar=True, zorder=-1, frameon=False)

        axA.grid(False)
        axA.axes.get_xaxis().set_visible(False)
        axA.axes.get_yaxis().set_visible(False)

        # figA = pl.figure(figsize=(10, 9))
        # axA = figA.add_subplot(111, polar=True)

        if hist_corrected:
            new_height = height/(lengths/2.)
        else:
            new_height = height

        axA.bar(radii, new_height, width=width, bottom=bottom, edgecolor="none", color=custom_plot.grau, alpha=0.7, linewidth=0)

        if a == 1:
            # p value for radial distribution
            chi2, p = scipy.stats.chisquare(new_height)

            text_x = 0.4
            text_y = 0.18
            text_dy = -0.1
            text_dx = 0.035

            frz = 20

            print 'chi2 = ', chi2
            print 'chi2 p-Wert  = ', p
            Jv.ax_joint.text(text_x, text_y, '$\chi^2$= '+str(numpy.round(chi2, 2)), fontsize=frz, color=custom_plot.grau)
            if p < 0.001:
                Jv.ax_joint.text(text_x+text_dx, text_y+text_dy, '$p$ < 0.001', fontsize=frz, color=custom_plot.grau)
                # Jv.ax_joint.text(text_x+text_dx, text_y+text_dy, '$p$ = 0.04', fontsize=frz, color=custom_plot.grau)
            else:
                Jv.ax_joint.text(text_x+text_dx, text_y+text_dy, '$p$ = '+str(numpy.round(p, 3)), fontsize=frz, color=custom_plot.grau)

        # figA, axA = pl.subplots(1, 1, figsize=(10, 9))
        # axA.axvline(hlines[a][0], color='k', linestyle=':')
        # axA.axvline(hlines[a][1], color='k', linestyle='--')
        # axA.hist(ang[a], bins=numpy.arange(min(ang[a]), max(ang[a]) + binwidth, binwidth))
        # axA.set_xlabel(xlab[a], fontsize=fz)
        # axA.set_ylabel('Count', fontsize=fz)

        # Hide the right and top spines
        # axA.spines['right'].set_visible(False)
        # axA.spines['top'].set_visible(False)
        #
        # # Only show ticks on the left and bottom spines
        # axA.yaxis.set_ticks_position('left')
        # axA.xaxis.set_ticks_position('bottom')

        # figA.savefig(path+'Summary/'+name[a]+'.pdf', format='pdf')
        # figA.savefig('/Users/haasolivia/Desktop/plots/'+name[a]+'.pdf', format='pdf')

    # plot radial histograms for all donut sections

    fig_donut = pl.figure(figsize=(7, 7))

    ax_donut = fig_donut.add_axes([-1.0, -1.0, 2.15, 2.15], polar=True, frameon=False)

    start = wi
    end = len_diag+wi

    # start = 0
    # end = len_diag

    if segments == 3:
        mul = 16
    elif segments == 5:
        mul = 13
    else:
        mul = 14

    ax_donut.set_xlim(0, end*mul + 2)
    ax_donut.set_ylim(0, end*mul + 2)
    ax_donut.grid(False)
    ax_donut.axes.get_xaxis().set_visible(False)
    ax_donut.axes.get_yaxis().set_visible(False)
    bottoms = numpy.arange(start, end, wi)*mul
    print 'DONUT Bottoms = ', bottoms

    Afirst = (numpy.pi*start**2)/4
    Asec = ((numpy.pi*(2*start)**2)/4)-Afirst
    x = start
    y = 2.
    scheiss = .5*(9*(x**2)*(numpy.pi/2) - (y*numpy.sqrt(9*(x**2) - (y**2)) + 9*(x**2)*numpy.arctan(y/numpy.sqrt(9*(x**2) - (y**2)))))
    Athird = ((numpy.pi*(3*start)**2)/4)-Afirst-Asec  #-(2*scheiss)
    # Afourth = 4-(Afirst+Asec+Athird)
    Afourth = ((numpy.pi*(4*start)**2)/4)-Afirst-Asec-Athird

    A = [Afirst/22.5, Asec/22.5, Athird/22.5, Afourth/22.5]  # maxial surface of each bin

    # if segments == 2:
    #     t_y = [5, 20]          # height or radius
    #     t_dx = [5.92, 6.18]  # in radians
    #     t_dy = [1.4, 1]    # height or radius
    #
    # elif segments == 3:
    #     t_y = [5, 18, 35]          # height or radius
    #     t_dx = [5.83, 6.13, 6.2]  # in radians
    #     t_dy = [1.7, 1.2, 1.2]    # height or radius
    #
    # elif segments == 4:
    #     # t_y = [1, 9, 18, 28]          # height or radius
    #     t_dx = [5.4, 6.08, 6.18, 6.215]  # in radians
    #     t_dy = [1.65, 0.9, 0.9, 0.9]    # height or radius
    #
    # elif segments == 5:
    #     t_y = [1, 9, 18, 28, 38]          # height or radius
    #     t_dx = [5.4, 6.08, 6.18, 6.215, 6.3]  # in radians
    #     t_dy = [1.65, 0.9, 0.9, 0.9, 0.9]    # height or radius

    # t_y = [1, 8, 15.5, 22]
    # t_dx = [5.6, 6.13, 6.2, 6.225]
    # t_dy = [1, 0.65, 0.65, 0.65]

    # plot 2 by 2 meter box _______________________________________________________

    ax_donut.plot([numpy.radians(45), numpy.radians(90)], [bottoms[-1], 2*mul],
                  color=custom_plot.grau, zorder=-1, alpha=0.5)
    ax_donut.plot([numpy.radians(45), numpy.radians(0)], [bottoms[-1], 2*mul],
                  color=custom_plot.grau, zorder=-1, alpha=0.5)

    # ax_donut.plot([numpy.radians(0), numpy.radians(0)], [0, 2*mul],
    #               color=custom_plot.grau, zorder=-1, alpha=0.5)
    # ax_donut.plot([numpy.radians(90), numpy.radians(90)], [0, 2*mul],
    #               color=custom_plot.grau, zorder=-1, alpha=0.5)

    # plot dashed, dotted and confidence lines______________________________________________

    ax_donut.plot([numpy.radians(45), numpy.radians(45)], [bottoms[0], bottoms[-1]], '--',
                  color=custom_plot.grau, zorder=-1)

    dotted_length = ((bottoms[-1])/numpy.sin(numpy.radians(180.-45.-(71.56505118-45.))))*numpy.sin(numpy.radians(45.))
    ax_donut.plot([numpy.radians(71.56505118), numpy.radians(71.56505118)], [bottoms[0], dotted_length], ':',
                  color=custom_plot.grau, zorder=-1)
    intersec_anglel = intersec_angle + 4.1
    intersec_angles = intersec_angle - 2.5

    if confidence:
        dotted_length = ((bottoms[-1])/numpy.sin(numpy.radians(180.-45.-(intersec_angles-45.))))*numpy.sin(numpy.radians(45.))
        ax_donut.plot([numpy.radians(intersec_angles), numpy.radians(intersec_angles)], [bottoms[0], dotted_length], '-',
                      color=custom_plot.grau, zorder=-1, alpha=.5)
        dotted_length = ((bottoms[-1])/numpy.sin(numpy.radians(180.-45.-(intersec_anglel-45.))))*numpy.sin(numpy.radians(45.))
        ax_donut.plot([numpy.radians(intersec_anglel), numpy.radians(intersec_anglel)], [bottoms[0], dotted_length], '-',
                      color=custom_plot.grau, zorder=-1, alpha=.5)

    # ax_donut.plot([numpy.radians(52.75), numpy.radians(52.75)], [0, bottoms[-1]],
    #               color='k', zorder=-1)
    # ax_donut.plot([numpy.radians(48.75), numpy.radians(48.75)], [0, bottoms[-1]],
    #               color='k', zorder=-1)
    # ax_donut.plot([numpy.radians(44.75), numpy.radians(44.75)], [0, bottoms[-1]],
    #               color='k', zorder=-1)

    all_angles = []

    for donut in numpy.arange(len(donuts)):
        angle = numpy.array(angles_ca_donuts[donut][0])
        all_angles.append(angle)
        print 'donut bin count = ', len(numpy.arange(min(angle), max(angle) + binwidth, binwidth))
        his = numpy.histogram(angle, bins=numpy.arange(min(angle), max(angle) + binwidth, binwidth))
        height = numpy.array([float(h) for h in his[0]])
        print 'len(height)', len(height)
        print height
        radii = his[1][:-1]*numpy.pi/180

        b = []
        for r in radii:
            if r < numpy.radians(45):
                b.append(2.*numpy.sin(r)/numpy.sin(numpy.radians(180)-(r+numpy.radians(90))))
            else:
                b.append(2.)
        lengths = b/numpy.sin(radii)
        width = binwidth*numpy.pi/180
        bottom = bottoms[donut]

        # if hist_corrected:
        #     new_height = height/(lengths/2.)
        # else:
        # new_height = height
        new_height = height/A[donut]
        new_height = new_height*sum(height)/sum(new_height)
        # new_height = height/(A[donut]/A[-1])

        #
        # print 'max height', max(height)
        # print 'sum(height)', sum(height)
        # print 'A[donut]/A[-1]', A[donut]/A[-1]
        # print 'max new_height', max(new_height)
        # sys.exit()

        rx = [0, 0, 19.5, 7.5]
        ry = [90, 90.001, 70.88, 87.5]

        ax_donut.bar(radii, new_height, width=width, bottom=bottom, edgecolor="none", color=custom_plot.grau, alpha=0.7, linewidth=0)
        radius = numpy.arange(numpy.radians(rx[donut]), numpy.radians(ry[donut]), numpy.radians(90)/100.)
        if donut < 3:
            ax_donut.plot(radius, numpy.ones_like(radius)*bottoms[donut], color=custom_plot.grau, alpha=0.5)

        # p value for radial distribution

        t_x1 = [0.075, 0.295, 0.515, 0.735]
        t_x2 = [0.06, 0.28, 0.5, 0.72]
        t_x3 = [0.08, 0.3, 0.52, 0.74]
        y_text = [0.27, 0.475, 0.68, 0.885]
        # t_x = 0   # in radians
        # t_y = 15.5   # height or radius [1, 8, 15.5, 22]
        # t_dx = 6.2  # in radians
        # t_dy = 0.65  # height or radius

        chi2, p = scipy.stats.chisquare(new_height)
        fig_donut.text(t_x1[donut], 0.13, '$n$ = '+str(numpy.round(len(angle), 2)), fontsize=frz, color='k')
        fig_donut.text(t_x2[donut], 0.07, '$\chi^2$= '+str(numpy.round(chi2, 2)), fontsize=frz, color='k')
        # ax_donut.text(t_x, t_y[donut], '$\chi^2$= '+str(numpy.round(chi2, 2)), fontsize=frz, color='k')
        if p < 0.001:
            fig_donut.text(t_x3[donut], 0.01, '$p$ < 0.001', fontsize=frz, color='k')
            # ax_donut.text(t_x+t_dx[donut], t_y[donut]+t_dy[donut], '$p$ < 0.001', fontsize=frz, color='k')
        else:
            fig_donut.text(t_x3[donut], 0.01, '$p$ = '+str(numpy.round(p, 3)), fontsize=frz, color='k')
            # ax_donut.text(t_x+t_dx[donut], t_y[donut]+t_dy[donut], '$p$ = '+str(numpy.round(p, 3)), fontsize=frz, color='k')

        # fig_donut.text(0.01, y_text[donut], str(numpy.round(bottoms[donut], 2)), fontsize=frz, color='k')

    donut_info = {'first': all_angles[0], 'second': all_angles[1], 'third': all_angles[2], 'fourth': all_angles[3]}
    hickle.dump(donut_info, path+'Summary/donut_info.hkl', mode='w')
    # plot histograms
    sns.set(style="white")
    for J in [Jr, Jv]:
        J.ax_marg_x.cla()
        J.ax_marg_y.cla()

    binwidth = 0.25
    if xlimr and ylimr:
        binwidthr = (binwidth/(xlimr[1]-xlimr[0]))*(ylimr[1]-ylimr[0])
    else:
        binwidthr = binwidth

    if xlimv and ylimv:
        binwidthv = (binwidth/(xlimv[1]-xlimv[0]))*(ylimv[1]-ylimv[0])
    else:
        binwidthv = binwidth

    dev = 3.
    hist_plot_color = custom_plot.grau

    Jr.ax_marg_x.hist(xr, weights=numpy.ones_like(xr)/len(xr), color=hist_plot_color,
                      alpha=0.7, bins=numpy.arange(min(xr), max(xr), binwidth/dev), normed=0, edgecolor="none")

    Jr.ax_marg_y.hist(yr, weights=numpy.ones_like(yr)/len(yr), color=hist_plot_color,
                      alpha=0.7, bins=numpy.arange(min(yr), max(yr), binwidthr/dev), orientation="horizontal", normed=0, edgecolor="none")

    Jv.ax_marg_x.hist(xv, weights=numpy.ones_like(xv)/len(xv), color=hist_plot_color,
                      alpha=0.7, bins=numpy.arange(min(xv), max(xv), binwidth/dev), normed=0, edgecolor="none")

    Jv.ax_marg_y.hist(yv, weights=numpy.ones_like(yv)/len(yv), color=hist_plot_color,
                      alpha=0.7, bins=numpy.arange(min(yv), max(yv), binwidthv/dev), orientation="horizontal", normed=0, edgecolor="none")

    rdot = Line2D([0], [0], linestyle="none", marker="o", markersize=9, markerfacecolor='k')
    rtri = Line2D([0], [0], linestyle="none", marker="^", markersize=9, markerfacecolor='k')

    if xlimr and xlimv:
        xlim = [xlimr, xlimv]
    else:
        xlim = False

    if ylimr and ylimv:
        ylim = [ylimr, ylimv]
    else:
        ylim = False

    for l, J in enumerate([Jr, Jv]):
        J.ax_marg_x.xaxis.set_visible(False)
        J.ax_marg_x.yaxis.set_visible(False)
        J.ax_marg_y.xaxis.set_visible(False)
        J.ax_marg_y.yaxis.set_visible(False)
        if xlim:
            J.ax_joint.set_xlim(xlim[l])
            J.ax_marg_x.set_xlim(xlim[l])

        if ylim:
            J.ax_joint.set_ylim(ylim[l])
            J.ax_marg_y.set_ylim(ylim[l])

        if l == 1:
            # vis_squareV = numpy.array([[1.4, 0.3], [2, 0.3], [2, 0], [1.4, 0]])
            vis_squareV = numpy.array([[1.55, 0.22], [2, 0.22], [2, 0], [1.55, 0]])
            # vis_squareV = numpy.array([[1.51, 0.22], [2, 0.22], [2, 0], [1.51, 0]])
            polyg = mpatches.Polygon(vis_squareV, color=hist_plot_color, zorder=-1, alpha=0.3)
            Jv.ax_joint.add_artist(polyg)

        if only_ca1:
            led = J.ax_joint.legend([rtri], [str(ca1) + ' CA1 fields'],
                                    numpoints=1, bbox_to_anchor=(1.01, 0.15), fontsize=20, handletextpad=-0.1)
        elif only_ca3:
            led = J.ax_joint.legend([rdot], [str(ca3) + ' CA3 fields'],
                                    numpoints=1, bbox_to_anchor=(1.01, 0.15), fontsize=20, handletextpad=-0.1)
        else:
            # led = J.ax_joint.legend([rtri, rdot], [str(ca1) + ' CA1 fields',  # (len(numpy.where(ca == 'CA1')[0])),
            #                                        str(ca3) + ' CA3 fields'],  # (len(numpy.where(ca == 'CA3')[0]))],
            #                         numpoints=1, bbox_to_anchor=(1.01, 0.15), fontsize=20, handletextpad=-0.1)
            led = J.ax_joint.legend([rdot], [str(ca3) + ' fields'],  # (len(numpy.where(ca == 'CA3')[0]))],
                                    numpoints=1, bbox_to_anchor=(1.01, 0.1), fontsize=20, handletextpad=-0.1)

        led.set_zorder(20)

    # ideal lines
    v_end = 2.  # v_begin is expected to be zero

    gains = [.5, 1.5]
    r_end_gain0 = v_end/gains[0]
    r_end_gain1 = v_end/gains[1]

    # virtual case_____________________________________________________________________________________
    Jv.ax_joint.hlines(v_end, 0, v_end, linestyle='-', color='k')
    Jv.ax_joint.vlines(v_end, 0, v_end, linestyle='-', color='k')

    Jv.ax_joint.plot([0, v_end], [0, v_end], '--', color=numpy.ones(3)*.5)
    Jv.ax_joint.plot([0, r_end_gain1*gains[0]], [0, v_end], ':', color=numpy.ones(3)*.5)

    # geom_end = (2./(numpy.sin(numpy.radians(56.31))))*numpy.sin(numpy.radians(90.-56.31))
    # Jv.ax_joint.plot([0, geom_end], [0, 2.], linestyle='-', color='k', zorder=10)

    # Jv.ax_joint.plot([remMid_V05_1, remMid_V05_2], [remMid_V15_1, remMid_V15_2], '-',
    #                  color=custom_plot.grau3, zorder=0)
    # Jv.ax_joint.plot([remBorder_V05_1, remBorder_V05_2], [remBorder_V15_1, remBorder_V15_2], '-',
    #                  color=custom_plot.grau3, zorder=0)
    # Jv.ax_joint.plot([propvis_V05_1, propvis_V05_2], [propvis_V15_1, propvis_V15_2], '-',
    #                  color=custom_plot.grau3, zorder=0)

    # classification areas as patches____________________

    c1 = '#a6bddd'  #custom_plot.pretty_colors_set2[0]  # treadmill patch color
    c2 = '#ea8f8a'  #custom_plot.pretty_colors_set2[5]  # visual patch color
    c3 = '#fee090'  #custom_plot.pretty_colors_set2[6]  # remap patch color

    new_vis_patchV = numpy.array([[propvis_V05_1, 0], [propvis_V05_2, 2], [2, 2], [2, 0]])
    new_vis_patchV = vis_patchV

    triangle = mpatches.Polygon(prop_patchV, color=c1, alpha=0.3, zorder=2)
    polygon = mpatches.Polygon(new_vis_patchV, color=c2, alpha=0.3, zorder=2)
    # triangle2 = mpatches.Polygon(rem_patchV, color=c3, alpha=0.3, zorder=-1)

    if background:
        Jv.ax_joint.add_artist(triangle)
        Jv.ax_joint.add_artist(polygon)
        # Jv.ax_joint.add_artist(triangle2)

    # real case_____________________________________________________________________________________
    Jr.ax_joint.hlines(r_end_gain1, 0, r_end_gain0, linestyle='-', color='k')
    Jr.ax_joint.vlines(r_end_gain0, 0, r_end_gain1, linestyle='-', color='k')

    Jr.ax_joint.plot([0, r_end_gain0], [0, r_end_gain1], '--', color=numpy.ones(3)*.5)
    Jr.ax_joint.plot([0, r_end_gain1], [0, r_end_gain1], ':', color=numpy.ones(3)*.5)

    # Jr.ax_joint.plot([remMid_V05_1/.5, remMid_V05_2/.5], [remMid_V15_1/1.5, remMid_V15_2/1.5], '-',
    #                  color=custom_plot.grau3, zorder=0)
    # Jr.ax_joint.plot([remBorder_V05_1/.5, remBorder_V05_2/.5], [remBorder_V15_1/1.5, remBorder_V15_2/1.5], '-',
    #                  color=custom_plot.grau3, zorder=0)
    # Jr.ax_joint.plot([propvis_V05_1/.5, propvis_V05_2/.5], [propvis_V15_1/1.5, propvis_V15_2/1.5], '-',
    #                  color=custom_plot.grau3, zorder=0)

    triangleP = mpatches.Polygon(prop_patchP, color=c1, alpha=0.3, zorder=-1)
    polygonP = mpatches.Polygon(vis_patchP, color=c2, alpha=0.3, zorder=-1)
    triangle2P = mpatches.Polygon(rem_patchP, color=c3, alpha=0.3, zorder=-1)
    # Jr.ax_joint.add_artist(triangleP)
    # Jr.ax_joint.add_artist(polygonP)
    # Jr.ax_joint.add_artist(triangle2P)

    if not double:
        name_real = 'Multi_peak_real_noDouble.pdf'
        name_virt = 'Multi_peak_virtual_noDouble.pdf'
        name_d = str(segments)+'donuts_noDouble_norm.pdf'
        if only_ca1:
            name_real = 'Multi_peak_real_noDouble_CA1.pdf'
            name_virt = 'Multi_peak_virtual_noDouble_CA1.pdf'
            name_d = str(segments)+'donuts_noDouble_CA1.pdf'
        elif only_ca3:
            name_real = 'Multi_peak_real_noDouble_CA3.pdf'
            name_virt = 'Multi_peak_virtual_noDouble_CA3.pdf'
            name_d = str(segments)+'donuts_noDouble_CA3.pdf'
        if hist_corrected:
            name_virt = 'Multi_peak_virtual_noDouble_corrected.pdf'
            if only_ca1:
                name_virt = 'Multi_peak_virtual_noDouble_CA1_corrected.pdf'
            elif only_ca3:
                name_virt = 'Multi_peak_virtual_noDouble_CA3_corrected.pdf'
    else:
        name_real = 'Multi_peak_real.pdf'
        name_virt = 'Multi_peak_virtual.pdf'
        name_d = str(segments)+'donuts.pdf'
        if only_ca1:
            name_real = 'Multi_peak_real_CA1.pdf'
            name_virt = 'Multi_peak_virtual_CA1.pdf'
            name_d = str(segments)+'donuts_CA1.pdf'
        elif only_ca3:
            name_real = 'Multi_peak_real_CA3.pdf'
            name_virt = 'Multi_peak_virtual_CA3.pdf'
            name_d = str(segments)+'donuts_CA3.pdf'
        if hist_corrected:
            name_virt = 'Multi_peak_virtual_corrected.pdf'
            if only_ca1:
                name_virt = 'Multi_peak_virtual_CA1_corrected.pdf'
            elif only_ca3:
                name_virt = 'Multi_peak_virtual_CA3_corrected.pdf'

    print 'Saving figure under:'+path+'Summary/'+name_real
    Jr.fig.savefig(path+'Summary/'+name_real, transparent=True, format='pdf')
    print 'Saving figure under:'+path+'Summary/'+name_virt
    Jv.fig.savefig(path+'Summary/'+name_virt, transparent=True, format='pdf')
    print 'Saving figure under:'+path+'Summary/'+name_d
    fig_donut.savefig(path+'Summary/'+name_d, transparent=True, format='pdf')
    if double:
        print 'Saving figure under:'+path+'Summary/double_cell_categories.pdf'
        fig_categories.savefig(path+'Summary/double_cell_categories.pdf', format='pdf')
        print 'Saving figure under:'+path+'Summary/double_cell_angle_diffs.pdf'
        fig_angle_diffs.savefig(path+'Summary/double_cell_angle_diffs.pdf', format='pdf')
        print 'Saving figure under:'+path+'Summary/double_cell_angle_diffs_categries.pdf'
        fig_angle_diffs1.savefig(path+'Summary/double_cell_angle_diffs_categries.pdf', format='pdf')
        print 'Saving figure under:'+path+'Summary/double_and_single_cell_angles.pdf'
        fig_angle_dou.savefig(path+'Summary/double_and_single_cell_angles.pdf', format='pdf')
    if bidirec:
        print 'Saving figure under:'+path+'Summary/bidirec_cell_categories.pdf'
        fig_bidirec_categories.savefig(path+'Summary/bidirec_cell_categories.pdf', format='pdf')


def weight_histo(data):

    w = numpy.array(data['Field_weights'])

    # remove nans from data

    w = w[numpy.logical_not(numpy.isnan(w))]

    fig, ax1 = pl.subplots()

    ax1.hist(w)
    ax1.set_xlabel('Place field weights')
    ax1.set_ylabel('Count')
    print 'Saving figure under:'+path+'Summary/Multi_peak_weights.pdf'
    fig.savefig(path+'Summary/Multi_peak_weights.pdf', format='pdf')


def twin_and_color_ax(newax, color, xspace=5):

    # newax.set_frame_on(True)
    newax.patch.set_visible(False)
    custom_plot.turnOffAxes(newax, ['left', 'top', 'right'])
    newax.xaxis.set_ticks_position('bottom')
    newax.xaxis.set_label_position('bottom')
    newax.spines['bottom'].set_position(('outward', xspace))
    newax.spines['bottom'].set_color(color)
    newax.tick_params(axis='x', colors=color)


def plot_example_cells(list_virt_real_cell, used_points, direction='right', win=35, fz=40, gaussians=True, halign='right'):

    # if gaussians = True xy=[[[virt_x_0.5, virt_y_05], [virt_x_1.5, virt_y_05]],
    #                         [[real_x_0.5, real_y_05], [real_x_1.5, real_y_05]]]
    # and list_virt_real_cell=[xy, xy, ....]

    x_label = ['Virtual position (m)', 'Treadmill position (m)']
    x_limits = [(0, 2), (0, 4)]
    all_xlim = [[2.0, 2.0], [1.33, 4.0]]

    for cell in numpy.arange(len(list_virt_real_cell)):

        y_max = []
        fig, ax = pl.subplots(1, 2, figsize=(14, 4), sharey='row')
        ax = ax.flatten()
        width = 0.42
        height = 0.7

        x_begin = 0.06
        y_begin = 0.2  # 0.35

        x_space = 0.1

        ax_positions = []

        # for r in numpy.arange(1):
        ax_positions.append([x_begin, y_begin])
        ax_positions.append([x_begin+width+x_space, y_begin])

        for i, a in enumerate(ax):
            pos2 = [ax_positions[i][0], ax_positions[i][1],  width, height]  # x, y, width, height
            a.set_position(pos2)  # set a new position

        for input in [0, 1]:
            counter = input
            sns.set(style="ticks", font_scale=1.8)  # 3)

            c1 = custom_plot.pretty_colors_set2[1]  #-- for gain 1.5
            c2 = custom_plot.pretty_colors_set2[0]  #-- for gain 0.5

            # c1 = custom_plot.pretty_colors_set2[1]  # red   -- for gain 1.5
            # c2 = custom_plot.pretty_colors_set2[2]  # blue  -- for gain 0.5

            if gaussians:
                data1x = list_virt_real_cell[cell][input][0][0]  # first [0] for gain 0.5 or [1] gain 1.5
                data1x = data1x[numpy.logical_not(numpy.isnan(data1x))]

                data2x = list_virt_real_cell[cell][input][1][0]  # second [0] for x-values and [1] for y-values
                data2x = data2x[numpy.logical_not(numpy.isnan(data2x))]

                data1y = list_virt_real_cell[cell][input][0][1]
                data1y = data1y[numpy.logical_not(numpy.isnan(data1y))]

                data2y = list_virt_real_cell[cell][input][1][1]
                data2y = data2y[numpy.logical_not(numpy.isnan(data2y))]

            else:
                data1 = list_virt_real_cell[cell][input][direction+'FR_x_y_gain_0.5']
                data2 = list_virt_real_cell[cell][input][direction+'FR_x_y_gain_1.5']

                data1x = data1[0]
                data2x = data2[0]
                data1y = signale.tools.smooth(data1[1], window_len=win)
                data2y = signale.tools.smooth(data2[1], window_len=win)

            ax[counter].fill_between(data1x, data1y, facecolor=c2, color=c2, alpha=0.5)
            ax[counter].plot(data1x, data1y, color=c2, linewidth=2)
            ax[counter].fill_between(data2x, data2y, facecolor=c1, color=c1, alpha=0.5)
            ax[counter].plot(data2x, data2y, color=c1, linewidth=2)

            y_max.append(max(data1y))
            y_max.append(max(data2y))

            ax[counter].set_xlabel(x_label[input], fontsize=fz)

            ax[counter].set_xlim(x_limits[input])

            # labels = ax[counter].get_xticklabels()
            # ax[counter].set_xticklabels(labels, fontsize=fz*2)

            if counter == 1:
                print 'max Hz: ', str(numpy.round(max(y_max), 1))
                y_pos = numpy.round(max(y_max), 1)
                if len(halign) == len(list_virt_real_cell):
                    alignm = halign[cell]
                    if alignm == 'left':
                        xpos = 0
                    else:
                        xpos = x_limits[input][1]
                else:
                    alignm = 'right'
                    xpos = x_limits[input][1]

                ax[counter].text(xpos, y_pos, str(y_pos)+' Hz', horizontalalignment=alignm,
                                 verticalalignment='top', zorder=10, fontsize=fz)

            custom_plot.turnOffAxes(ax[counter], ['bottom', 'left', 'top', 'right'])

            if counter == 0:
                fig.subplots_adjust(bottom=0.30)

            ax_small = fig.add_axes(ax[counter].get_position())
            twin_and_color_ax(newax=ax_small, color=c1)
            ax_small.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax_small.tick_params(axis='both', which='major', pad=15)

            if counter == 0:
                ax_small.set_xticks([0, all_xlim[input][1]])
            else:
                ax_small.set_xticks([0, all_xlim[input][0], all_xlim[input][1]])
            ax_small.tick_params(labelsize=fz)

            if counter == 1:
                ax_small.spines['bottom'].set_bounds(0, 1.33)
                xt = ax_small.xaxis.get_major_ticks()
                xt[2].tick1On = False
            for tick in ax_small.xaxis.get_ticklabels():
                tick.set_color('k')

            ax_large = fig.add_axes(ax[counter].get_position())
            twin_and_color_ax(newax=ax_large, color=c2, xspace=14)
            ax_large.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax_large.set_xticks([0, all_xlim[input][1]])
            ax_large.set_xticklabels('')

        x1 = numpy.round(used_points[cell][0], 2)
        y1 = numpy.round(used_points[cell][1], 2)
        print 'Saving figure under:'+path+'Summary/Bsp_xy_'+str(x1)+'_'+str(y1)+'.pdf'
        fig.savefig(path+'Summary/Bsp_xy_'+str(x1)+'_'+str(y1)+'.pdf', format='pdf')

    # ax will be filled from top to bottom according to input list_virt_real_cell
    return fig, ax


if __name__ == "__main__":

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

    double_shoule_be_single = ['10823_2015-08-04_VR_GCend_linTrack1_TT2_SS_06_PF_info_left',
                               '10823_2015-08-04_VR_GCend_linTrack1_TT2_SS_06_PF_info_normalised_left',
                               '10823_2015-08-18_VR_GCend_nami_linTrack1_TT3_SS_16_PF_info_right',
                               '10823_2015-08-18_VR_GCend_nami_linTrack1_TT3_SS_16_PF_info_normalised_right',
                               '10823_2015-08-19_VR_GCend_linTrack1_TT2_SS_16_PF_info_right',
                               '10823_2015-08-19_VR_GCend_linTrack1_TT2_SS_16_PF_info_normalised_right']

    field_switch = ['10823_2015-07-22_VR_GCend_linTrack1_TT2_SS_07_PF_info_left',  # gain='1.5', amp=.72
                    '10823_2015-07-22_VR_GCend_linTrack1_TT2_SS_07_PF_info_normalised_left']

    adapt_lower_bound = ['10823_2015-08-17_VR_GCend_linTrack1_TT2_SS_08_PF_info_left',  # gain='1.5', thresh = .21
                         '10823_2015-08-17_VR_GCend_linTrack1_TT2_SS_08_PF_info_normalised_left']

    # test_file = (adapt_lower_bound[0]).split('_left')[0]+'.hkl'
    # calc_peaks_bsp(file=test_file, gain='1.5', run_direc='left')

    # first
    # calc_peaks()

    # second
    # get_fitted_pf_width_and_maxFR()

    # third
    # info = hickle.load(path+'/Summary/MaxFR_doublePeak_info.hkl')
    # info = hickle.load(path+'/Summary/MaxFR_doublePeak_info_corrected.hkl')
    # find_double_cells(info=info)

    # # fourth
    # info = hickle.load(path+'Summary/delta_and_weight_info.hkl')
    # example_points = [[1.87, 0.05], [0.18, -0.47]]  # [],[vis],[prop]  [3.0, 0.67], [1.69, 0.00], [0.45, -0.75], [0.00, -0.56]
    # # print example_points
    # # plot_deltas(data=info, examples=example_points, xlim=[-8./6, 4], ylim=[-2, 2], double=False)
    # compare_gains(data=info, examples=example_points, xlimr=[0, 4], ylimr=[0, 4], xlimv=[0, 2], ylimv=[0, 2],
    #               double=True, only_double=False, only_ca1=False, only_ca3=False, hist_corrected=False,
    #               bidirec=True, only_bidirec=False, connected=False, bsp=False, background=False, confidence=False)

    # fifth
    # gauss_data_corr(xlimfr=[0, 13], xlimwidth=[0, 2], xlimpos=[0, 2], fz=20, figsize=6, scale=1.8,
    #                 double=False, rausmap=False, new_slope=False)

    # phase_precession()

    # theta_delta_ratio()
    #
    cluster_quality()
    # pie_chart()
    # merge_hkl()

    # surrogate_repeat = 1000 (a lot faster!) for testing 10000 for actual plots!
    # angle_surrogate(raw_data=True, single_fields=True, double_fields=False, both=False, surrogate_repeat=10000)

    # for single fields
    # angle_likelihood_ratio(m_one_gauss=54.375143145, s_one_gauss=14.2487130264,
    #                        m1_two_gauss=42.1733689665, m2_two_gauss=61.4213157195,
    #                        s1_two_gauss=5.0, s2_two_gauss=6.67549316933)

    #weight_histo(data=info)

