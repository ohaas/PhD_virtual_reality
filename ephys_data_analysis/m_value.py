__author__ = "Olivia Haas"

def m_val(gauss1, gauss2, x, x_doublePointNum, run_direc, su):
    mg1 = max(gauss1)
    mg2 = max(gauss2)

    if mg1 >= mg2:
        small_max = mg2
        small_max_index = numpy.argmax(gauss2)
    else:
        small_max = mg1
        small_max_index = numpy.argmax(gauss1)

    # calculate values to get m = deltaF/Fmean:____________________________________________
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

    # set negative slopes at the beginning of the derivative to zero, as they are artifacts___
    zero_crossings = numpy.where(numpy.diff(numpy.sign(derivative1)))[0]
    if len(zero_crossings):
        first_sign_change = zero_crossings[sc]+1

        if run_direc == 'left':
            derivative1[first_sign_change:len(derivative1)][derivative1[first_sign_change:len(derivative1)] < 0] = 0.
        else:
            derivative1[0:first_sign_change][derivative1[0:first_sign_change] < 0] = 0.
    # ________________________________________________________________________________________

    # use sign change of derivative to detect zero crossings (for that replace zeros with neighbouring values)____
    sign = numpy.sign(derivative1)

    # get rid of zeros and use sign value from the value before
    for l in sign_array:
        if sign[l] == 0.:
            if run_direc == 'right' and l == 0:
                sign[l] = sign[l+1]
            elif run_direc == 'left' and l == len(sign)-1:
                sign[l] = sign[l-1]
            else:
                sign[l] = sign[l+pre_sign]
    # get rid of remaining zeros at array edges
    for l in sign_array[::-1]:
        if sign[l] == 0.:
            if run_direc == 'left' and l == 0:
                sign[l] = sign[l+1]
            elif run_direc == 'right' and l == len(sign)-1:
                sign[l] = sign[l-1]
            else:
                sign[l] = sign[l-pre_sign]

    # find derivative zero crossings____________________________________________________________
    deri1_zero = numpy.where(numpy.diff(sign))[0]+1

    if len(deri1_zero) == 3:  # with 3 zero crossings m-value can be calculated____________
        between_peak_min_index = deri1_zero[1]

        between_peak_min = (gauss1+gauss2)[between_peak_min_index]
        index_delta = abs(between_peak_min_index-small_max_index)

        delta_F = small_max-between_peak_min

        # sonderfaelle______________________
        if small_max_index-index_delta < 0:
            s_index = 0
        else:
            s_index = small_max_index-index_delta

        if small_max_index+index_delta+1 > len(x)-1:
            l_index = len(x_doublePointNum)-1
        else:
            l_index = small_max_index+index_delta+1
        # __________________________________

        small_peak_mean = numpy.mean((gauss1+gauss2)[s_index: l_index])

        # calculate m-value_______________________________________________________________________
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

    else:  # not 3 zero crossings -> m-value cannot be calculated
        if su == 0:
            M_data.append(numpy.nan)
            good = 0
            extra_path = 'Deriv_bad/'
