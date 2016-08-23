__author__ = "Olivia Haas"

import os
import sys

# add additional custom paths
extraPaths = [os.path.join(os.path.abspath(os.path.dirname(__file__)), '../scripts')]
for p in extraPaths:
    if not sys.path.count(p):
        sys.path.insert(1, p)

import numpy
import matplotlib.pyplot as pl

import hickle
from sklearn import mixture

import signale
import matplotlib

import custom_plot

# initialte lists to be saved in hickle format:
# MG_large = []
# MG_small = []
#
# MxG_large = []
# MxG_small = []
#
# MG_std_large = []
# MG_std_small = []
#
# Weights_largeGauss = []
# Weights_smallGauss = []
#
# Mx_combinedGauss = []
# MG_combined = []
#
# X = []
# Y_large = []
# Y_small = []
#
# orig_data_x = []
# orig_data_y = []
# gauss_x = []
# gauss1_y = []
# gauss2_y = []
# derivative_y = []
# cumulative_x = []
# cumulative_y = []
# cumulative_95perc_index = []
# real_data_in_cumulative_x = []

server = 'saw'
multi = 1

path = '/Users/haasolivia/Documents/'+server+'/dataWork/olivia/hickle/'  # von wo die daten geladen werden
pathGauss = '/Users/haasolivia/Documents/gaussFits/'  # wo die Fits lokal gespeichert werden sollen

hkl_files_pre = numpy.repeat(hickle.load(path+'Summary/used_filenames.hkl'), 2)
# hkl_files_pre1 = numpy.repeat(hkl_files_pre, 2)
hkl_files = [hkl_files_pre[i].split('_normalised')[0]+'.hkl' if (i%2 != 0) else hkl_files_pre[i] for i in range(len(hkl_files_pre))]

run_direc = numpy.repeat(hickle.load(path+'Summary/running_directions.hkl'), 2)

norm_hkl_files = hickle.load(path+'Summary/used_filenames.hkl')
norm_run_direc = hickle.load(path+'Summary/running_directions.hkl')


def fit_gaussians_etc(x, y, surrogate_repeat, gain, run_direc, file, savefig=True, fz=14):

    # window length is in bins (1 FR bin = 2cm) => 3*2cm = 6cm Kernel (Chen = 8.5cm, Ravassard = 5cm)
    y = signale.tools.smooth(y, window_len=3.)

    # generate one more x-point between all existing x-points:________________________________________________
    x_doublePointNum = numpy.arange(x[0], x[-1]+numpy.diff(x)[0]/2., numpy.diff(x)[0]/2.)

    # data = pl.hist(x, weights=y, bins=len(numpy.arange(min(x), max(x), x[1]-x[0])))

    # in case firing rates are too small (<5) multiply them with factor that resulting histogram represents
    # data form better________________________________________________________________________________________
    if max(y) < 5:
        input_y = 5. * (y/max(y))
    else:
        input_y = multi*y

    # generate data histogram___________________________________________________________________________________
    data = numpy.repeat(x, numpy.around(input_y).astype(int))

    # generate surrogat data____________________________________________________________________________________
    for su in [0]:  # numpy.arange(surrogate_repeat+1):

        # if su != 0 and good == 1:  # for su=0 actual data will be plotted
        #
        #     # generate randomly shuffled data___________________________________________________________________
        #     surrogate_data = numpy.random.choice(list(x), len(data))
        #     data = surrogate_data
        #
        #     # generate histogram for shuffled data______________________________________________________________
        #     bin_num = len(numpy.arange(min(x), max(x), abs(x[1]-x[0])))
        #     new_y = numpy.histogram(surrogate_data, bins=bin_num, range=(min(x), max(x)))[0]
        #
        #     # undo multiplication from line 97 to get actual firing rates back__________________________________
        #     if max(y) < 5:
        #         y = (new_y * max(y))/5.
        #     else:
        #         y = new_y

        # fit two gaussians_____________________________________________________________________________________
        gmm = mixture.GMM(n_components=2, covariance_type='full', min_covar=0.0000001)  # gmm for two components
        gmm.fit(numpy.vstack(data))  #numpy.vstack(data))  #numpy.vstack(data))  # train it!

        # get functions for two fitted gaussians________________________________________________________________
        gauss1 = (gmm.weights_[0] * matplotlib.mlab.normpdf(x_doublePointNum, gmm.means_[0], numpy.sqrt(gmm.covars_[0])))[0]
        gauss2 = (gmm.weights_[1] * matplotlib.mlab.normpdf(x_doublePointNum, gmm.means_[1], numpy.sqrt(gmm.covars_[1])))[0]

        # calculate basic values for the FR distribution y_______________________________________________________
        std = numpy.std(y)
        mean = numpy.mean(y)

        # calculate basic values for the gaussians_______________________________________________________________
        mg1a = max(gauss1)
        mg2a = max(gauss2)

        stdg1 = numpy.sqrt(gmm.covars_[0])[0][0]
        stdg2 = numpy.sqrt(gmm.covars_[1])[0][0]

        # calculate x difference between two gaussian peaks_______________________________________________________
        xDiff = abs(x_doublePointNum[numpy.argmax(gauss1)]-x_doublePointNum[numpy.argmax(gauss2)])

        # define x-window depending on difference of gaussian peaks, in which gauss amplitudes will be normalised
        # ________________________________________________________________________________________________________
        if file.endswith('normalised.hkl'):
            xDiff_cutoff = 0.2/float(gain)
        else:
            xDiff_cutoff = 0.20

        # define devident of the standart deviation from the maximum FR that should be used____________________
        std_dev1 = 2.
        std_dev2 = 2.

        if xDiff > xDiff_cutoff < (1./4.)*max(x) and stdg1 > 0.3:   # for larger fields 1/8. of the std will be used
            std_dev1 = 8.
        if xDiff > xDiff_cutoff < (1./4.)*max(x) and stdg2 > 0.3:
            std_dev2 = 8.

        # amplitude maximum for gauss 1___________________________________________________________________________
        x1 = numpy.argmax(gauss1)+1

        if x1 >= len(x)-1:  # if gauss 1 fit maximum is outside the data use maximum close to track end
            x1 = len(x)-3

        # find indices where the FR maximum should be take out of__________________________________
        g1_max0 = numpy.argmin(abs(x_doublePointNum[0:x1]-(x_doublePointNum[x1-1]-stdg1/std_dev1)))
        g1_max1 = numpy.argmin(abs(x_doublePointNum[x1:-1]-(x_doublePointNum[x1-1]+stdg1/std_dev1)))+x1

        # define new indices for Sonderfaelle_______________________________
        if g1_max0 >= len(y)-1 and g1_max1+1 >= len(y)-1:  # maximum should be taken from outside the data
            g1_max0 = len(y)-5
            g1_max1 = len(y)-2
        if g1_max0 == g1_max1+1:  # indices are equal, so that there is no range from which the max can be taken
            g1_max0 -= 2
            g1_max1 += 2
        if g1_max0 < 0:  # maximum should be taken from outside the data
            g1_max0 = 0
        if g1_max1+1 > len(y)-1:  # larger index only is outside the data
            g1_max1 = len(y)-2

        # get gauss maximum in area of interest_____________
        g1_maxFR = max(y[g1_max0:g1_max1+1])

        # amplitude maximum for gauss 2___________________________________________________________________________
        x2 = numpy.argmax(gauss2)+1

        if x2 >= len(x)-1:  # if gauss 1 fit maximum is outside the data use maximum close to track end
            x2 = len(x)-3

        # find indices where the FR maximum should be take out of__________________________________
        g2_max0 = numpy.argmin(abs(x_doublePointNum[0:x2]-(x_doublePointNum[x2-1]-stdg2/std_dev2)))
        g2_max1 = numpy.argmin(abs(x_doublePointNum[x2:-1]-(x_doublePointNum[x2-1]+stdg2/std_dev2)))+x2

        # define new indices for Sonderfaelle (s.o.)_______________________________
        if g2_max0 >= len(y)-1 and g2_max1+1 >= len(y)-1:
            g2_max0 = len(y)-5
            g2_max1 = len(y)-2
        if g2_max0 == g2_max1+1:
            g2_max0 -= 2
            g2_max1 += 2
        if g2_max0 < 0:
            g2_max0 = 0
        if g2_max1+1 > len(y)-1:
            g2_max1 = len(y)-2

        # get gauss maximum in area of interest_____________
        g2_maxFR = max(y[g2_max0:g2_max1+1])

        # set gauss closest to y distribution maximum to its maximum___________________________________________
        nearest_yMax_gauss = signale.tools.findNearest(numpy.array([x_doublePointNum[numpy.argmax(gauss1)],
                                                                    x_doublePointNum[numpy.argmax(gauss2)]]),
                                                       x[numpy.argmax(y)])[0]
        if nearest_yMax_gauss == 0:
            g1_maxFR = max(y)
        else:
            g2_maxFR = max(y)

        # normalise gaussians to FR maximum of distribution y within gausstian maximum + / - 0.5 of its std:__________
        gauss1 = g1_maxFR*(gauss1/mg1a) # first normalise gauss to max=1 then multiply with new maximum
        gauss2 = g2_maxFR*(gauss2/mg2a)

        # get gauss amplitude and weights______________________
        amplitude_g1 = gmm.weights_[0] * g1_maxFR/mg1a
        amplitude_g2 = gmm.weights_[1] * g2_maxFR/mg2a

        weight_g1 = amplitude_g1/(amplitude_g1 + amplitude_g2)
        weight_g2 = amplitude_g2/(amplitude_g1 + amplitude_g2)

        # get maxima auf gaussians with new amplitude__________
        mg1 = max(gauss1)
        mg2 = max(gauss2)
        # max_mean_diff_in_std1 = (mg1 - mean)/std
        # max_mean_diff_in_std2 = (mg2 - mean)/std

        # define plot colors based on which gauss is bigger______
        if mg1 >= mg2:
            colour = ['r', 'k']
            small_max = mg2
            small_max_index = numpy.argmax(gauss2)
        else:
            colour = ['k', 'r']
            small_max = mg1
            small_max_index = numpy.argmax(gauss1)

        # calculate values to get m = deltaF/Fmean:____________________________________________
        # derivative1 = numpy.diff(gauss1+gauss2) / numpy.diff(x_doublePointNum)
        #
        # # remove negative values in beginning of derivative
        # if run_direc == 'left':
        #     # for leftwards runs the array is starting from the end of the track!
        #     sc = -1
        #     pre_sign = 1
        #     sign_array = numpy.arange(len(derivative1))[::-1]  # backwards array
        # else:
        #     sc = 0
        #     pre_sign = -1
        #     sign_array = numpy.arange(len(derivative1))
        #
        # # set negative slopes at the beginning of the derivative to zero, as they are artifacts___
        # zero_crossings = numpy.where(numpy.diff(numpy.sign(derivative1)))[0]
        # if len(zero_crossings):
        #     first_sign_change = zero_crossings[sc]+1
        #
        #     if run_direc == 'left':
        #         derivative1[first_sign_change:len(derivative1)][derivative1[first_sign_change:len(derivative1)] < 0] = 0.
        #     else:
        #         derivative1[0:first_sign_change][derivative1[0:first_sign_change] < 0] = 0.
        # # ________________________________________________________________________________________
        #
        # # use sign change of derivative to detect zero crossings (for that replace zeros with neighbouring values)____
        # sign = numpy.sign(derivative1)
        #
        # # get rid of zeros and use sign value from the value before
        # for l in sign_array:
        #     if sign[l] == 0.:
        #         if run_direc == 'right' and l == 0:
        #             sign[l] = sign[l+1]
        #         elif run_direc == 'left' and l == len(sign)-1:
        #             sign[l] = sign[l-1]
        #         else:
        #             sign[l] = sign[l+pre_sign]
        # # get rid of remaining zeros in the array edges
        # for l in sign_array[::-1]:
        #     if sign[l] == 0.:
        #         if run_direc == 'left' and l == 0:
        #             sign[l] = sign[l+1]
        #         elif run_direc == 'right' and l == len(sign)-1:
        #             sign[l] = sign[l-1]
        #         else:
        #             sign[l] = sign[l-pre_sign]
        #
        # # find derivative zero crossings____________________________________________________________
        # deri1_zero = numpy.where(numpy.diff(sign))[0]+1
        #
        # if len(deri1_zero) == 3:  # with 3 zero crossings m-value can be calculated____________
        #     between_peak_min_index = deri1_zero[1]
        #
        #     between_peak_min = (gauss1+gauss2)[between_peak_min_index]
        #     index_delta = abs(between_peak_min_index-small_max_index)
        #
        #     delta_F = small_max-between_peak_min
        #
        #     # sonderfaelle______________________
        #     if small_max_index-index_delta < 0:
        #         s_index = 0
        #     else:
        #         s_index = small_max_index-index_delta
        #
        #     if small_max_index+index_delta+1 > len(x)-1:
        #         l_index = len(x_doublePointNum)-1
        #     else:
        #         l_index = small_max_index+index_delta+1
        #     # __________________________________
        #
        #     small_peak_mean = numpy.mean((gauss1+gauss2)[s_index: l_index])
        #
        #     # calculate m-value_______________________________________________________________________
        #     m = delta_F/small_peak_mean
        #
        #     if numpy.isnan(m):
        #         print 'delta_F = ', delta_F
        #         print 'small_peak_mean = ', small_peak_mean
        #         print 'mean for index1 to index2 : ', small_max_index-index_delta, small_max_index+index_delta+1
        #         print (gauss1+gauss2)[small_max_index-index_delta: small_max_index+index_delta+1]
        #         sys.exit()
        #
        #     if su != 0:
        #         M.append(m)
        #     else:
        #         M_data.append(m)
        #         good = 1
        #         extra_path = 'Deriv_good/'
        #
        # else:  # not 3 zero crossings -> m-value cannot be calculated
        #     if su == 0:
        #         M_data.append(numpy.nan)
        #         good = 0
        #         extra_path = 'Deriv_bad/'

        if su == 0:
            # plot data and gaussians from mixture model

            fig22, ax22 = pl.subplots(1, 1, figsize=(18, 12))

            ax22.axhline(mean, linestyle='-', color=custom_plot.pretty_colors_set2[0], alpha=0.8, zorder=0)
            ax22.axhspan(mean-std, mean+std, facecolor=custom_plot.pretty_colors_set2[0], alpha=0.2, linewidth=False, zorder=0)
            ax22.plot(x, y, 'b')
            ax22.plot(x_doublePointNum, gauss1, linewidth=2, color=colour[0])  # gauss1 = small gauss
            ax22.plot(x_doublePointNum, gauss2, linewidth=2, color=colour[1])
            ax22.plot(x_doublePointNum, gauss1+gauss2, linewidth=2, color='g')
            ax22.set_xlabel('Position from start point (m)', fontsize=fz)
            ax22.set_ylabel('Firing rate (Hz)', fontsize=fz)

            ax22.set_ylim(0, max(gauss1+gauss2)+0.01)
            ax22.set_xlim(0, max(x))

    return fig22, ax22


def calc_peaks():
    counter = 0
    surrogate_repeat = 1000

    # overwrite hkl_files and running directions with specific test files_______________________
    hkl_files = ['10528_2015-04-01_VR_GCend_linTrack1_TT4_SS_06_PF_info.hkl',
                 '10528_2015-04-16_VR_GCend_Dark_linTrack1_TT2_SS_25_PF_info.hkl',
                 '10537_2015-10-22_VR_GCend_linTrack1_TT1_SS_09_PF_info_normalised.hkl',
                 '10823_2015-07-24_VR_GCend_linTrack1_TT3_SS_04_PF_info_normalised.hkl']

    run_direc = ['right', 'left', 'right', 'left']

    for i, file in enumerate(hkl_files):
        a = hickle.load(path+file)  #'10353_2014-06-17_VR_GCend_linTrack1_GC_TT3_SS_07_PF_info_normalised.hkl')  #file
        for gain in ['1.5']:  #['0.5', '1.5']:
            print 'counter: ', counter, ' out of ', len(hkl_files)-1  #*2
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

            fig22, ax22 = fit_gaussians_etc(x=x, y=y, surrogate_repeat=surrogate_repeat, gain=gain,
                                            run_direc=run_direc[i], file=file, savefig=True)

            if gain == '0.5':
                g = '05'
            else:
                g = '15'

            fig22.savefig(pathGauss+file.split('.hkl')[0]+'_'+run_direc[i]+'_gain_'+g+'.pdf', format='pdf')

calc_peaks()
