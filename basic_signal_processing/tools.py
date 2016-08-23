"""
signale.tools
=============

A module for general purpose tools.
"""

__author__ = ("KT", "Olivia Haas")
__version__ = "2.6, January 2016"

# python modules
import sys

# other modules
import numpy
import scipy
import matplotlib.pyplot as pl

# custom made modules

# package modules




###################################################### TOOLS for ARRAY_LIKES


def findMaximaMinima(a, findMaxima=1, findMinima=1, or_equal=1):
    """ Find the maxima and minima in an 1D array.

    Parameters
    ----------
    a : ndarray
        1D array
    findMaxima, findMinima : bool
        True if ought to look for maxima and/or minima. Defaults to 1.

    Returns
    -------
    {...} : dict
        Dictionary with {'maxima_number', 'minima_number', 'maxima_indices', 'minima_indices'}.
    """
    gradients=numpy.diff(a)

    maxima_num=0
    minima_num=0
    max_indices=[]
    min_indices=[]

    if or_equal:            # includes only the next entries and they can also be equal to zero
        for i, g in enumerate(gradients[:-1]):
            if findMaxima and cmp(g, 0)>=0 and cmp(gradients[i+1], 0)<=0 and g != gradients[i+1]:
                maxima_num+=1
                max_indices.append(i+1)

            if findMinima and cmp(g, 0)<=0 and cmp(gradients[i+1], 0)>=0 and g != gradients[i+1]:
                minima_num+=1
                min_indices.append(i+1)
    else:                   # includes also the previous and next entries and also expects them not to be equal to zero
        for i, g in enumerate(gradients[1:-1]):
            j = i+1         # get real index
            if findMaxima and cmp(g, 0)>=0 and cmp(gradients[j-1], 0)>0 and cmp(gradients[j+1], 0)<0:
                maxima_num+=1
                max_indices.append(j+1)

            if findMinima and cmp(g, 0)<=0 and cmp(gradients[j-1], 0)<0 and cmp(gradients[j+1], 0)>0:
                minima_num+=1
                min_indices.append(j+1)

    return {'maxima_number': maxima_num ,'minima_number': minima_num,\
            'maxima_indices': numpy.array(max_indices), 'minima_indices': numpy.array(min_indices)}


def findNearest(array, value):
    """ Find the entry nearest to a value in an 1D array.

    Parameters
    ----------
    array : ndarray
        1D array
    value : float
        Value to find nearest entry for.

    Returns
    -------
    (index, array[index]) : tuple
        Tuple with index and value of search result.
    """
    index = numpy.nanargmin(numpy.abs(array-value))
    return (index, array[index])


def findNearest_vec(a, v):
    """ Find the entry nearest to a vector value in an 1D array.

    Parameters
    ----------
    a : ndarray
        1D array
    v : float
        Value to find nearest entry for.

    Returns
    -------
    (index, array[index]) : tuple
        Tuple with index and value of search result.
    """
    diff = a-v
    norms = numpy.array([])
    for d in diff:
        norms = numpy.append(norms, numpy.linalg.norm(d))
    index = norms.argmin()
    return (index, a[index])


def findNearest_smaller_or_equal(array, value):
    """ Find the entry nearest and smaller or equal to a value in an 1D array.

    Parameters
    ----------
    array : ndarray
        1D array
    value : float
        Value to find nearest and smaller entry for.

    Returns
    -------
    (index, array[index]) : tuple
        Tuple with index and value of search result.
    """
    diff = array - value
    mask = numpy.ma.greater(diff, 0)
    if numpy.all(mask):
        return None
    masked_diff = numpy.ma.masked_array(diff, mask)
    index = numpy.nanargmax(masked_diff)
    return (index, array[index])




def get_below_threshold(data, threshold, conditioned_data_frames=[], axis=0):
    """ Remove data above threshold from pandas DataFrame.

    NOTE: changes the original dataframes!

    Parameters
    ----------
    data : pandas DataFrame
    threshold : float
        The threshold value.
    conditioned_data_frames : list
        List of data frames of same size as data, from which values will be removed that correspond to
        the ones being removed from data.
    axis : int
        Axis of array to work along.
    """

    if axis == 1:
        data = data.T
        for i, df in enumerate(conditioned_data_frames):
            conditioned_data_frames[i] = df.T

    for i in range(data.shape[0]):
        d = data.iloc[i].values
        indices = numpy.where(d > threshold)[0]
        d = numpy.pad(numpy.delete(d, indices), [0, indices.size], 'constant', constant_values=(numpy.nan, numpy.nan))
        data.iloc[i] = d

        for df in conditioned_data_frames:
            df.iloc[i] = numpy.pad(numpy.delete(df.iloc[i].values, indices), [0, indices.size],
                                   'constant', constant_values=(numpy.nan, numpy.nan))

    if axis == 1:
        data = data.T
        for i, df in enumerate(conditioned_data_frames):
            conditioned_data_frames[i] = df.T



def sameLengthArr(arr):
    """ Reduce a list with several 1D arrays to the length of the smallest.

        Expects a list with several 1D arrays. The arrays may be of different sizes.

        Parameters
        ----------
        arr : list
            List of 1D numpy arrays.

        Returns
        -------
        numpy 2D array.
    """

    minLength = 10**100
    for a in arr:
        minLength = min(minLength, a.size)

    return numpy.array([a[:minLength] for a in arr])


def pairs(a):
    """ Generator function yielding all item pairs of a list.

    Parameters
    ----------
    a : list
        List to iterate over.
    """

    l = len(a)
    for i in range(l):
        for j in range(i+1, l):
            yield a[i], a[j]


def transposeArr(arr, numeric=True):
    """ Transpose a list of 1D arrays.

        Expects a list with several 1D arrays. The arrays may be of different size.

        Parameters
        ----------
        arr : list
            List of 1D arrays.
        numeric : bool
            True if the indidivual arrays (lists) contain numbers, then a list of numpy arrays is returned.

        Returns
        -------
        transposed list
    """

    maxLength = 0
    for a in arr:
        maxLength = max(maxLength, a.size)

    transposedArr = []
    for i in range(maxLength):
        l = []
        for a in arr:
            if i < a.size:
                l.append(a[i])
        if numeric:
            transposedArr.append(numpy.array(l))
        else:
            transposedArr.append(l)

    return transposedArr


def time_slice(a, t0, t1):
    """ Time slice.

    Parameters
    ----------
    a : array_like
        Array to be sliced in time.
    t0 : float
        Start time.
    t1 : float
        Stop time.

    Returns
    -------
        sliced array
    """
    mask = (a >= t0) & (a <= t1)
    a_sliced = numpy.extract(mask, a)

    return a_sliced

###################################################### AVERAGING

def avg_array(arr, avg_type='mean'):
    """Avarages arrays of different size.

    Parameters
    ----------
    arr : array_like
        Contains data of the learning parameter ordered in time expected to be a list containing several datasets.
    avg_type : str, optional
        Type of average {'mean', 'median'}. Default: mean.

    Returns
    -------
    avg, error : array_like
    """

    maxLength = 0
    for i, v in enumerate(arr):
        maxLength = max(v.size, maxLength)

    if avg_type == 'mean':
        # average
        avg = numpy.zeros(maxLength)
        std = numpy.zeros(maxLength)
        num = numpy.zeros(maxLength)
        for v in arr:
            for i, value in enumerate(v):
                if not numpy.isnan(value):
                    avg[i] += value
                    std[i] += value**2
                    num[i] += 1
        avg /= num
        std /= num
        std -= avg**2
        std = numpy.sqrt(std)

        nerror = avg - std/numpy.sqrt(num)
        perror = avg + std/numpy.sqrt(num)
    else:
        print
        print 'avgType not implemented for data type ...'
        print

    return avg, std


def moving_average(a, n=3):
    """
    NOTE: not phase corrected! Use smooth with 'flat' window instead.
    """

    ret = numpy.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]

    return ret[n - 1:] / n


def moving_average2(a, bin_width=3):

    avg = numpy.array([])
    std = numpy.array([])
    for i in range(a.size):

        # the special cases
        if i < bin_width:
            avg = numpy.append(avg, a[:i].mean())
            std = numpy.append(std, a[:i].std())
        elif i > a.size - bin_width:
            avg = numpy.append(avg, a[i:].mean())
            std = numpy.append(std, a[i:].std())
        # the normal cases
        else:
            avg = numpy.append(avg, a[i:i+bin_width].mean())
            std = numpy.append(std, a[i:i+bin_width].std())

    return avg, std


def moving_average3(x_orig, y_orig, x_win=0.2, upper_point_limit=None):

    """Calculate moving average, standard deviation and number of sample points (for sem calculation) for not sorted
    input cloud of data points.

    This moving average is based on a non-sorted input array x_orig:
    e.g. x_orig = numpy.array([0, 1, 2, 1, 0])

    and a non-sorted input array y_orig:
    e.g. y_orig = numpy.array([1, 3, 2, 1, 5])

    Window size is the x-axis width, in e.g. meters, included in the sliding window, no matter how many data points
    are in this window.

    upper_point_limit resets all windows which contain more points to this number of points and can speed up the
    calculation.

    Output has the same length as the input arrays!
    """

    # remove nans from data set:
    x_orig1 = x_orig[numpy.logical_not(numpy.isnan(x_orig))]
    y_orig1 = y_orig[numpy.logical_not(numpy.isnan(x_orig))]
    x_orig1 = x_orig1[numpy.logical_not(numpy.isnan(y_orig1))]
    y_orig1 = y_orig1[numpy.logical_not(numpy.isnan(y_orig1))]

    x_and_y = numpy.array(zip(x_orig1, y_orig1))

    x = numpy.unique(x_orig1)

    # sorts y data for each x-value into subarrays:
    # e.g. x = numpy.array([0, 1, 2])
    # e.g. y = numpy.array([0, 5, 8], [2], [1, 4, 6, 7, 8, 10])
    y = [x_and_y[:, 1][numpy.where(x_and_y[:, 0] == i)[0]] for i in x]

    x = numpy.array(x)
    y = numpy.array(y)

    if len(x) != len(y):
        print 'WARNING: moving_average3 IS ABORTED BECAUSE X AND Y ARRAYS HAVE DIFFERENT LENGTH!'
        sys.exit()

    n = len(numpy.concatenate(y))  # sample size = number of points

    # calculate number of data points for each sliding window
    w = numpy.array([int(numpy.rint(numpy.nanargmin(abs(x[i:] - (x[i]+x_win))))) for i in numpy.arange(len(x))])
    # sliding window has to contain at least one data point
    w[w == 0] = 1
    if upper_point_limit:
        w[w > upper_point_limit] = upper_point_limit

    dummy = [[numpy.mean(numpy.concatenate(y[numpy.arange(max(i - w[i] + 1, 0), min(i + w[i] + 1, len(x)))])),
              numpy.std(numpy.concatenate(y[numpy.arange(max(i - w[i] + 1, 0), min(i + w[i] + 1, len(x)))]))]
             for i in numpy.arange(len(x))]

    avg_y = numpy.array(dummy)[:, 0]
    std_y = numpy.array(dummy)[:, 1]

    return x, y, avg_y, std_y, n


def moving_time_window(a, f, t_start, t_stop, window_width=1., d_window=1.):
    """ Moving/sliding function evaluation.

    Parameters
    ----------
    a : array_like
        Data for moving time window.
    f : function
        Function to apply.
    window_width : float
        Width of the window in time steps.
    d_window : float
        Shift in time steps between two function evaluations.
    """

    start = t_start
    stop = start + window_width
    d = []
    t = []
    while start < t_stop:
        d.append(f(time_slice(a, start, stop)))
        t.append(start)
        start += d_window
        stop += d_window
        stop = numpy.min([stop, t_stop])

    d = numpy.array(d)
    t = numpy.array(t) + d_window/2.

    return d, t


###################################################### SMOOTHING


def nansmooth_flat(x, window_len=15, window='flat'):
    """ Convenience function to be used with apply() for pandas data frame.

    Ignores nans by splitting array into sub-arrays. Not that if a particular sub-array is too small no
    smoothing will be done.
    """

    slices_x = numpy.ma.clump_unmasked(numpy.ma.masked_invalid(x))
    nonans = [x[s] for s in slices_x]

    for i, a in enumerate(nonans):
        smoothed_a = smooth(a, window_len=window_len, window=window)
        x[slices_x[i]] = smoothed_a

    return x


def smooth_flat(x, window_len=50, window='flat'):
    """ Convenience function to be used with apply() for pandas data frame."""

    return smooth(x, window_len=window_len, window=window)


def smooth(x, window_len=11, window='hanning'):
    """Smooth time series data.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    Parameters
    ----------
    x : ndarray
        The input signal.
    window_len : float, optional
        The dimension of the smoothing window; should be an odd integer. Defaults to 11.
    window : float, optional
        The type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
        The flat window will produce a moving average smoothing. Defaults to hanning window.

    Returns
    -------
    smoothed signal

    Examples
    --------
    >>> import numpy
    >>> t = numpy.linspace(-2, 2, 0.1)
    >>> x = sin(t) + numpy.random.randn(len(t))*0.1
    >>> y = smooth(x)

    See also
    --------
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO
    ----
    the window parameter could be the window itself if an array instead of a string
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        #print "WARNING: Input vector needs to be bigger than window size. Returning original input vector."
        #return x
        print "WARNING: Input vector needs to be bigger than window size. Using window size as half the size of input vector."
        window_len = x.size/2.


    if window_len < 3:
        print "NOTE: Window too short. Returning original input vector."
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s = numpy.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]       # enlarge signal at the borders
    #    print x.shape, s.shape

    if window == 'flat': #moving average
        w = numpy.ones(window_len,'d')
    else:
        w = eval('numpy.'+window+'(window_len)')

    y = numpy.convolve(w/w.sum(), s, mode='same')
    y = y[window_len-1:-(window_len-1)]                             # return to original signal size
    #   print y.shape

    return y


def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions.

     From http://wiki.scipy.org/Cookbook/SignalSmooth
    """

    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = numpy.mgrid[-size:size+1, -sizey:sizey+1]
    g = numpy.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()


def SDF(spiketrain_times, times_start, times_stop, plotSpiketrain=False, plotGaussian=False, plotSDF=False):

    if not hasattr(spiketrain_times, 'timeUnit'):
        'Cannot assess spiketrain timeUnit. Assuming spiketrain in seconds'
    elif spiketrain_times.timeUnit == 'ms':
        spiketrain_times.changeTimeUnit('s')

    duration = times_stop-times_start
    dt = 0.001

    # generate train of spikes
    #t = numpy.arange(0.0, duration, dt)
    t = numpy.arange(times_start, times_stop+dt, dt)
    spikes = numpy.zeros(len(t))

    for time in spiketrain_times:
        #index = findNearest(t, time-times_start)[0]
        index = findNearest(t, time)[0]
        spikes[index] = 1.0

    rate = len(spiketrain_times)/duration

    if plotSpiketrain:
        pl.figure()
        pl.plot(t, spikes, 'k')

    # Gaussian window
    #sigma = (1.0/rate*1000)/5.0                       	        # [ms]
    sigma = 0.8  #(1.0/rate)/2.0                     	        # [s]  dividing by 2.0 sets the width of the gaussian
    gauss_duration = duration #20.0  #*sigma                    	        # [s]
    gauss_t = numpy.arange(-gauss_duration/2.0, gauss_duration/2.0, dt)		# time vector for the window

    #gauss = 1.0/numpy.sqrt(2.0*numpy.pi*numpy.power(sigma, 2.0))*numpy.exp(-numpy.power(gauss_t, 2.0)/(2.0*numpy.power(sigma, 2.0)))
    gauss = numpy.exp(-numpy.power(gauss_t, 2.0)/(2.0*numpy.power(sigma, 2.0)))  # amplitude = 1.0

    if plotGaussian:
        pl.figure()
        pl.plot(gauss_t, gauss)
        pl.title('Gaussian kernel')

    # convolution
    convolvedSpikes = numpy.convolve(gauss, spikes, 'same')

    if plotSDF:
        pl.figure()
        pl.plot(t, convolvedSpikes)
        pl.ylabel('Firing rate [Hz]')
        if hasattr(spiketrain_times, 'timeUnit'):
            pl.xlabel('time in '+spiketrain_times.timeUnit)
        else:
            pl.xlabel('time in s')

    if plotSpiketrain or plotGaussian or plotSDF:
        pl.show()

    return t, convolvedSpikes, spikes


def blur_image(im, n, ny=None):
    """ Blur an image.

    Blurs an image by convolving with a gaussian kernel.

    Parameters
    ----------
    n : int
        Size of gauss kernel.
    ny : int, optional
        The optional keyword argument ny allows for a different size in the y direction.

    From http://wiki.scipy.org/Cookbook/SignalSmooth
    """
    g = gauss_kern(n, sizey=ny)
    improc = scipy.signal.convolve(im, g, mode='same')

    return improc


###################################################### STATISTICS

def fitGauss(x, data):
    """ Fit a gaussian to data.

    Uses method of moments for fitting.

    Parameters
    ----------
    x : ndarray
        x axis of data
    data : ndarray
        1D array of data

    Returns
    -------
    gauss : ndarray
        fitted gaussian
    """

    #X = numpy.arange(data.size)
    x_bar = numpy.sum(x*data)/numpy.sum(data)
    width = numpy.sqrt(numpy.abs(numpy.sum((x-x_bar)**2*data)/numpy.sum(data)))
    amp = data.max()

    gauss = norm.pdf(x, x_bar, width)
    gauss /= gauss.max()
    gauss *= amp

    return gauss


######################################################


def close_path(path):
    """ Close a matplotlib path.

    Closes an open matplotlib.path.

    Parameters
    ----------
    path : matplotlib.path

    Returns
    -------
    closed path
    """
    import matplotlib.path as mpl_path

    p = path.cleaned()
    verts = p.vertices
    codes = p.codes

    if numpy.sum(verts[0]-verts[-1]):
        if not codes[-1]:
            verts[-1] = verts[0]
            codes[-1] = 2
        else:
            verts = numpy.vstack((verts, verts[0]))
            codes = numpy.append(codes, 2)

    return mpl_path.Path(verts, codes).cleaned()



