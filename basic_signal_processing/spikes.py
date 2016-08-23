"""
signale.spikes
==============

A module for spikes.
"""

__author__ = ("KT", "Alireza Chenani", "Franziska Hellmundt", "Christian Leibold")
__version__ = "5.1.2, September 2014"


# python modules

# other modules
import numpy,scipy, matplotlib
import matplotlib.pyplot as pl
import NeuroTools.signals as NTsig
import inspect
import scipy.signal as scsig
import statsmodels.tsa

# custom made modules

# package modules
import signals
from signals import signal
import io


###################################################### FUNCTIONS

###################################################### CLASSES


class spikezug(signal, NTsig.SpikeTrain):
    """
    Spike train class that extends the NeuroTools SpikeTrain class.
    """

    def __init__(self, spike_times, spiketrain_type='', t_start=None, t_stop=None, timeUnit='ms'):
        signal.__init__(self)
        NTsig.SpikeTrain.__init__(self, spike_times, t_start=None, t_stop=None)
        self.spiketrain_type = spiketrain_type          # e.g. '' (empty), rightward, leftward

    def getFiringField(self, parameter, time, bin_size=.01, bin_smooth=5):
        """ Get firing field as a function of parameter.
        """


        xlim = numpy.array([parameter.min(), parameter.max()])

        # bin track!
        bins = numpy.arange(xlim[0], xlim[1]+bin_size, bin_size)

        # create polygons & count spikes
        digitized = numpy.digitize(parameter, bins)

        field_spikes = []
        firing_rate = []
        param = []
        for index, bin in enumerate(bins):
            indices = numpy.where(digitized == index)[0]
            if indices.size:
                s = []
                p = []
                for t in time[indices]:
                    indices1 = numpy.where(self.spike_times >= t)[0]
                    indices2 = numpy.where(self.spike_times <= t+dt)[0]
                    if indices1.size and indices2.size:
                        indices0 = numpy.intersect1d(indices1, indices2)
                        s.append(spikes[indices0])
                        p.append(numpy.ones(s[-1].size)*bin)
                s = numpy.concatenate(s)
                field_spikes.append(s)
                firing_rate.append(float(s.size)/indices.size/dt)
                param.append(numpy.concatenate(p))
            else:
                field_spikes.append([])
                firing_rate.append(0)

        firing_rate = signale.smooth(numpy.array(firing_rate), bin_smooth)
        param = numpy.concatenate(param)

        return bins, firing_rate, param



class spikezugList(signal, NTsig.SpikeList):
    """
    A class that extends the NeuroTools SpikeList class.
    """

    def __init__(self, fileName=None, timeUnit='ms', **kwargs):
        signal.__init__(self)
        self.timeUnit = timeUnit
        if fileName:
            data=numpy.loadtxt(fileName)
            data=numpy.hstack((data[:,[1]], data[:,[0]]))     # swap left and right columns in data, necessary to use SpikeList.__init__ properly
            data=[tuple(t) for t in data]
            self.meta = io._read_metadata(fileName)
            NTsig.SpikeList.__init__(self, data, range(self.meta['first_id'], self.meta['last_id']+1))
        else:
            NTsig.SpikeList.__init__(self, [], [], t_start=kwargs['t_start'],\
                t_stop=kwargs['t_stop'], dims=kwargs['dims'])

    def __calcTimeAxis(self):
        """
        Calculate the time axis for the LFP, hilbert phases etc.
        """
        self.timeAxis = self.time_axis(self.binwidth)[:-1]
        #self.timeAxis = self.time_axis(binwidth)

    def __recalc_startstop(self):
        if len(self.spiketrains):
            # make a spiketrain list of all the spiketrains which are not empty (do contain spikes!)
            new_id_list = []
            for i in self.id_list:
                if self.spiketrains[i].spike_times.size:
                    new_id_list.append(i)
            # if the list is not empty find all start and stop times for those spiketrains which do contain spikes.
            # Empty spiketrains will therefore not have a start or stop time!
            if len(new_id_list):
                start_times = numpy.array([self.spiketrains[id].spike_times[0] for id in new_id_list], numpy.float32)
                stop_times = numpy.array([self.spiketrains[idx].spike_times[-1] for idx in new_id_list], numpy.float32)
                self.t_start = numpy.min(start_times)
                self.t_stop = numpy.max(stop_times)
                for id in new_id_list:
                    self.spiketrains[id].t_start = self.t_start
                    self.spiketrains[id].t_stop = self.t_stop

    def addTags(self, id, **kwargs):
        """
        For adding and changing comments/tags to the spiketrains.
        """
        self.tags[id] = {}
        for item in kwargs:
            self.tags[id][item] = kwargs[item]
            self.spiketrains[id].tags[item] = kwargs[item]      # also put tags to corresponding spiketrain

    def showTags(self):
        """
        For showing the comments/tags to the spiketrains on the command line.
        """
        for id in self.tags:
            print 'cell #', id
            for item in self.tags[id]:
                print ' ', item, ':', self.tags[id][item]
            print ''

    def changeTimeUnit(self, newUnit='ms'):

        factor = signals._getUnitFactor(self.timeUnit, newUnit)

        if factor != 1.0:

            # change the times
            for id in self.spiketrains:
                self.spiketrains[id].spike_times *= factor
                self.spiketrains[id].t_start *= factor
                self.spiketrains[id].t_stop *= factor
                self.spiketrains[id].timeUnit = newUnit
            self.__recalc_startstop()

            try: self.timeAxis
            except AttributeError:
                pass
            else:
                self.timeAxis *= factor

            try: self.binwidth
            except AttributeError:
                pass
            else:
                self.binwidth *= factor

            self.timeUnit = newUnit


    def convolveSpikes(self, kernel='gauss', kernel_width=20, kernel_offset=0, binwidth=1., display=False):
        """Convolve the spiketrains in the spikezugList with a given kernel

        Parameters:
            kernel ... type of kernel or array that serves as a kernel 
		       at the moment these options are available: 'gauss'
            kernel_width... width of the kernel in bin numbers.
            In order to get the time scale you need to multiply this by timeAxis resolution (ex. kernel_width * np.diff(self.timeAxis[1]) 
            kernel_offset ... offset the kernel by a certain number of bins
            binwidth ... time between bins, NOTE: also affects the convolution, since the kernel is just defined in bins!
            display ... optional display of a figure with the spiketrains and ints convolved counterpart
        """
        self.convolvedSpikeTrains = []
        spikeTrainHist = []
        if inspect.ismethod(self.id_list):
            IDlist = self.id_list()
        else:
            IDlist = self.id_list
        if kernel == 'gauss':
            kernel = scsig.gaussian(2*kernel_width, kernel_width)
        elif kernel == 'rect':
            kernel = numpy.ones(kernel_width)
        elif isinstance(kernel, numpy.ndarray):
            pass
        else:
            print 'the requested kernel is not defined yet, please use one of the following: \n \
                       "gauss" , "rect"'
        
##        try: self.timeAxis
##        except AttributeError:
##            self.__calcTimeAxis()
##
##        try: self.binwidth                              # binwidth
##        except AttributeError:
##            self.binwidth = binwidth
##        else:
##            if self.binwidth != binwidth:
##                self.binwidth = binwidth
##                self.__calcTimeAxis()

        self.binwidth = binwidth
        self.__calcTimeAxis()                # time axis for the LFP, hilbert phases etc.

        for id in IDlist:
            train=self.__getitem__(id)
##  NOTE: lines below moved upwards and using the SpikeList's not SpikeTrain's time_axis()
##            try: self.timeAxis
##            except AttributeError:
##                self.timeAxis=train.time_axis(binwidth)[:-1]
            spikeTrainHist.append(train.time_histogram(self.binwidth, normalized=False))
            conv = numpy.convolve(spikeTrainHist[-1], kernel, mode='full')
            #conv /= self.timeAxis.shape[-1]            # normalize convolution value
            conv = conv[kernel_offset:-kernel.shape[-1]+kernel_offset+1]         # reduce size so that conv has right shape and spikes and convolution are appropriately aligned
                                            # NOTE: Problem in case kernel_offset>kernel.shape[-1], needs to be rewritten!
            self.convolvedSpikeTrains.append(conv)
            '''In order to be consistent, I think it's better to write the convolved train attr for spikezug object
            and then sum them here as a property of spikezugList
            '''

        if display:
            fig = pl.figure(100,figsize=(12, 6))

            ax = fig.add_subplot(211)
            pos = ax.get_position()
            pos.x1 = .65
            ax.set_position(pos)
            for hist in spikeTrainHist:
                ax.plot(self.timeAxis[:hist.__len__()], hist, color=[.2,.2,.2])
            ax.set_ylabel('Spikes')
            ax.set_xlim(self.t_start, self.t_stop)

            ax = fig.add_subplot(212)
            pos = ax.get_position()
            pos.x1 = .65
            ax.set_position(pos)
            for convTrain in self.convolvedSpikeTrains:
                groesse = min(self.timeAxis.size, convTrain.size)
                ax.plot(self.timeAxis[:groesse], convTrain[:groesse])
            #ax.set_ylim(0, maxiPow*1.1)
            ax.set_xlabel('Time ['+self.timeUnit+']')
            ax.set_xlim(self.t_start, self.t_stop)
            ax.set_ylabel('Convolved spikes')
            #ax.legend(['Raw signal','Filtered signal '+str(minFreq)+'-'+str(maxFreq)+' Hz'])

            ax = fig.add_subplot(111)
            ax.set_position([pos.x1+.1, .1, .2, pos.height*2.2])
            pl.plot((numpy.arange(0, kernel.shape[0])-kernel_offset)*self.binwidth, kernel)
            ax.set_xlabel('Time ('+self.timeUnit+')')
            ax.set_title('Kernel')

            pl.show()

    def sumConvolvedSpikeTrains(self, first_id=None, last_id=None, display=False):
        """Sum the convolved spiketrains.

        Parameters
        ----------
        first_id : int
        last_id : int
        display : bool, optional
            Display of a figure with the spiketrains and ints convolved counterpart.
        """
        if inspect.ismethod(self.id_list):
            IDlist=self.id_list()
        else:
            IDlist=self.id_list
        if not first_id:
            first_id=IDlist[0]
        if not last_id:
            last_id=IDlist[-1]
        self.summedConvolvedSpikeTrains=sum(self.convolvedSpikeTrains[first_id:last_id+1])

        if display:
            fig = pl.figure(101,figsize=(12, 5))

            ax = fig.add_subplot(111)
            for convTrain in self.convolvedSpikeTrains:
                ax.plot(self.timeAxis, convTrain)
            ax.plot(self.timeAxis, self.summedConvolvedSpikeTrains,linewidth=3)
            #ax.set_ylim(0, maxiPow*1.1)
            ax.set_xlabel('Time ['+self.timeUnit+']')
            ax.set_xlim(self.t_start, self.t_stop)
            ax.set_ylabel('Summed convolved spikes')
            #ax.legend(['Raw signal','Filtered signal '+str(minFreq)+'-'+str(maxFreq)+' Hz'])

            pl.show()

        return self.summedConvolvedSpikeTrains


    def filterSummedConvolvedSpikeTrains(self, minFreq=None, maxFreq=None, timeRange=None, display=False, incRes=1):
        """For filtering the sum of the convolved spiketrains.

        Parameters
        ----------
        minFreq, maxFreq : float
            Band pass filter on the interval [minFreq, maxFreq].
        timeRange : tuple
            (start time, end time)
        display : bool, optional
            Display a figure with the spiketrains and ints convolved counterpart.
        incRes : float
            Factor used for increasing the number of entries in self.summedConvolvedSpikeTrains,
            i.e., increasing the resolution on the frequency axis. NOTE: has to be an integer!
        """
        try: self.convolvedSpikeTrains
        except AttributeError:
            self.convolveSpikes(numpy.array([1]))

        try: self.summedConvolvedSpikeTrains
        except AttributeError:
            self.sumConvolvedSpikeTrains()

        if not timeRange:
            timeRange = (self.timeAxis[0], self.timeAxis[-1])
        # convert timeRange from time to indices
        indexRange = (numpy.searchsorted(self.timeAxis, timeRange[0]),\
                numpy.searchsorted(self.timeAxis, timeRange[1]))
        self.summedConvolvedSpikeTrains = self.summedConvolvedSpikeTrains[indexRange[0]:indexRange[1]+1]
        self.timeAxis = self.timeAxis[indexRange[0]:indexRange[1]+1]

        # perform fft
        n = self.summedConvolvedSpikeTrains.shape[-1]*incRes         # trick to get better frequency resolution
        n = int(2**numpy.ceil(numpy.log(n)/numpy.log(2)))            # to get n = 2**x
        #print n
        self.sp = numpy.fft.rfft(self.summedConvolvedSpikeTrains, n=n)
        self.sp /= self.timeAxis.shape[-1]      # normalize fourier components to the number of sample points in signal
        self.freq = numpy.fft.fftfreq(n, d=self.binwidth/1000.)[0:n/2+1]

        if not minFreq:
            minFreq=self.freq.min()
        if not maxFreq:
            maxFreq=self.freq.max()

        # band pass filtering, zero phase
        self.sp_filtered=numpy.zeros_like(self.sp)
        for i, f in enumerate(self.freq):
            if abs(f) >= minFreq and abs(f) <= maxFreq:
                self.sp_filtered[i]=self.sp[i]

        # backtransform filtered signal to time space
        self.filteredSummedConvolvedSpikeTrains = numpy.fft.irfft(self.sp_filtered)
        self.filteredSummedConvolvedSpikeTrains *= self.timeAxis.shape[-1]          # rescale from normalization of the fourier components
        self.filteredSummedConvolvedSpikeTrains = self.filteredSummedConvolvedSpikeTrains.real
        self.filteredSummedConvolvedSpikeTrains = self.filteredSummedConvolvedSpikeTrains[:self.summedConvolvedSpikeTrains.shape[-1]]

        # determine power
        self.spPower = 2*self.sp*self.sp.conj()
        self.sp_filteredPower = 2*self.sp_filtered*self.sp_filtered.conj()

        # plot
        if display:
            fig = pl.figure(102,figsize=(12, 7))

            ax = fig.add_subplot(211)
            ax.plot(self.freq, self.spPower,'.-')      # times 2, since we are just on the real axis
            ax.plot(self.freq, self.sp_filteredPower)  # times 2, since we are just on the real axis
            ax.set_xlim(0, min(self.freq.max(), maxFreq*10))
            maxiPow=self.spPower[1:].max()                      # discard first power entry
            ax.set_ylim(0, maxiPow*1.1)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Power')
            ax.legend(['Raw signal','Filtered signal '+str(minFreq)+'-'+str(maxFreq)+' Hz'])

            ax = fig.add_subplot(212)
            ax.plot(self.timeAxis, self.summedConvolvedSpikeTrains)
            ax.plot(self.timeAxis, self.filteredSummedConvolvedSpikeTrains)
            #ax.plot(self.timeAxis, self.filteredSummedConvolvedSpikeTrains.real)
            #ax.plot(self.timeAxis, self.filteredSummedConvolvedSpikeTrains.imag)
            ax.set_xlim(self.t_start, self.t_stop)
            ax.set_xlabel('Time ['+self.timeUnit+']')
            ax.legend(['Raw signal', 'Filtered signal '+str(minFreq)+'-'+str(maxFreq)+' Hz'])
            pl.show()

        return self.filteredSummedConvolvedSpikeTrains


    def hilbertTransform(self, display=False, justPhase=True):
        """ Compute the hilbert transform and from that the analytical signal
            of the filtered convolved spike trains and return the phase and
            absolute values of the later.

            Parameters
            ----------
            display : bool, optional
                Display a figure with the spiketrains and ints convolved counterpart.
            justPhase : bool, optional
                If True, just return the hilbert phase.

            Returns
            -------
            hilbertPhase : ndarray
            hilbertAbsolute : ndarray, optional
            analyticalTrain : ndarray, optional
            """

        try: self.filteredSummedConvolvedSpikeTrains
        except AttributeError:
            self.filterSummedConvolvedSpikeTrains()


        self.analyticalTrain = scipy.signal.hilbert(self.filteredSummedConvolvedSpikeTrains.real)
        self.hilbertPhase=numpy.angle(self.analyticalTrain, deg=True)
        for i in range(self.hilbertPhase.shape[-1]):
            if self.hilbertPhase[i]<0:
                self.hilbertPhase[i]+=360
        self.hilbertAbsolute=numpy.absolute(self.analyticalTrain)

        if display:
            fig = pl.figure(103,figsize=(12, 7))
            ax = fig.add_subplot(311)

            ax.plot(self.timeAxis, self.analyticalTrain.real)
            ax.plot(self.timeAxis, self.analyticalTrain.imag)
            ax.set_xlim(self.t_start, self.t_stop)
            ax.legend(['Signal', 'Hilbert transform'])

            ax = fig.add_subplot(312)
            ax2=ax.twinx()
            ax.plot(self.timeAxis, self.hilbertAbsolute)
            ax2.plot(self.timeAxis, self.hilbertPhase,'g')
            ax.set_xlim(self.t_start, self.t_stop)
            ax2.set_ylim(0, 360)
            ax2.set_yticks(range(0, 410, 90))
            ax2.set_yticklabels(range(0, 410, 90))
            ax.set_xlabel('Time ['+self.timeUnit+']')
            ax.set_ylabel('Absolute value')
            ax2.set_ylabel('Phase [deg]')

            ax = fig.add_subplot(313)
            ax2=ax.twinx()
            ax.plot(self.timeAxis, self.analyticalTrain.real)
            ax2.plot(self.timeAxis, self.hilbertPhase,'g')
            ax.set_xlim(self.t_start, self.t_stop)
            ax2.set_ylim(0, 360)
            ax.set_xlabel('Time ['+self.timeUnit+']')
            ax.set_ylabel('Signal')
            ax2.set_ylabel('Phase [deg]')

            pl.show()

        if justPhase:
            return self.hilbertPhase
        else:
            return self.hilbertPhase, self.hilbertAbsolute, self.analyticalTrain

    def auto_correlate(self, id0, sigma=None, dt=.002, nlags=None, display=False):
        """ Auto-correlation of two spike trains.

        Parameters
        ----------
        id0 : int
            ID of the spiketrain.
        sigma : float, optional
        dt : float, optional
            Size of time bin. Defaults to 0.002.
        nlags : int
            Number of lags to compute auto-correlation for.
        display : bool, optional
            Display the auto-correlogram? Defaults to False.

        Returns
        -------
        tau : ndarray
            Time shift of the correlation.
        corr : ndarray
            Correlation value.
        """
        if not sigma or sigma<dt:
            print 'auto-correlate: Setting kernel width to dt='+str(dt)+'.'
            sigma = dt
        if not nlags:
            nlags = numpy.fix(1./dt)
        xmax = 4.*sigma
        xmin = -xmax
        offset = -xmin/dt
        kernel = numpy.exp(-numpy.arange(xmin, xmax, dt)**2/(2*sigma**2))
        self.convolveSpikes(kernel, kernel_offset=offset, binwidth=dt, display=display)

        train0 = self.convolvedSpikeTrains[id0]
        corr = statsmodels.tsa.stattools.acf(train0, nlags=nlags, fft=True)  # correlation with respect to train0
        corr = numpy.hstack([numpy.flipud(corr[1:]), corr])
        tau = numpy.arange(-int(corr.shape[0]/2), int(corr.shape[0]/2))*dt
        if display:
            fig = pl.figure(figsize=(12, 7))
            ax = fig.add_subplot(111)
            ax.plot(tau, corr, '-')
            ax.set_xlabel('Offset ('+self.timeUnit+')')
            pl.show()

        return tau, corr


    def correlate(self, id0, id1, sigma=.1, dt=.001, display=False):
        """ Cross-correlation of two spike trains.

        Parameters
        ----------
        id0, id1 : ints
            IDs of the spiketrains to correlate with each other.
        sigma : float
        dt : float
        display : bool, optional
            Display the cross-correlogram? Defaults to False

        Returns
        -------
        tau : ndarray
            Time shift of the correlation.
        corr : ndarray
            Correlation value.
        """
        xmax = 4.*sigma
        xmin = -xmax
        offset = -xmin/dt
        kernel = numpy.exp(-numpy.arange(xmin, xmax, dt)**2/(2*sigma**2))
        self.convolveSpikes(kernel, kernel_offset=offset, binwidth=dt, display=display)

        train0 = self.convolvedSpikeTrains[id0]
        train1 = self.convolvedSpikeTrains[id1]
        corr = numpy.correlate(train1, train0, 'same')  # correlation with respect to train0
        #tau = numpy.arange(-corr.shape[0]/2+1, corr.shape[0]/2+1)*dt
        tau = numpy.arange(-corr.shape[0]/2, corr.shape[0]/2)*dt
        if display:
            fig = pl.figure(figsize=(12, 7))
            ax = fig.add_subplot(111)
            ax.plot(tau, corr, '-')
            ax.set_xlabel('Offset ('+self.timeUnit+')')
            pl.show()

        return tau, corr

    def getSpikePhases(self):
        """ Return spike phases.

        Calculate the spike phases from the hilbert phase and attach this information
        to the particular spiketrains in the spikezugList.
        """

        try: self.hilbertPhase
        except AttributeError:
            self.hilbertTransform()

        for num in self.spiketrains:
            indices = numpy.array([(numpy.abs(self.timeAxis-s)).argmin() for s in self[num].spike_times])
            for index in numpy.where(indices >= self.timeAxis.shape[0]-1)[0]:
                indices[index] = self.timeAxis.shape[0]-1
            self[num].spike_phases = self.hilbertPhase[indices]

    def spectrogram(self, windowSize=4096, overlap=None, display=False, minFreq=None, maxFreq=None):

            try: self.filteredSummedConvolvedSpikeTrains
            except AttributeError:
                self.filterSummedConvolvedSpikeTrains()


            # parameters of the spectrogram
            if not overlap:
                overlap=int(windowSize*.9)
            samplingFreq=1./self.binwidth

            # use matplotlibs specgram function for the spectrogram
            Pxx, freqs, t = matplotlib.mlab.specgram(self.summedConvolvedSpikeTrains,\
                NFFT=windowSize, noverlap=overlap, pad_to=windowSize*10, Fs=samplingFreq)
            Pxxfiltered, freqs_filtered, t_filtered = matplotlib.mlab.specgram(self.filteredSummedConvolvedSpikeTrains,\
                NFFT=windowSize, noverlap=overlap, pad_to=windowSize*10, Fs=samplingFreq)

            freqs *= 1000
            freqs_filtered *= 1000

            if display:
                if not minFreq:
                    minFreq=freqs.min()
                if not maxFreq:
                    maxFreq=freqs.max()

                indexstart = numpy.where(freqs >= minFreq)[0][0]
                indexend = numpy.where(freqs <= maxFreq)[0][-1]
                print indexstart, indexend

                fig = pl.figure(102, figsize=(10, 7))
                ax = fig.add_subplot(121)
                t, freqs = numpy.meshgrid(t, freqs)
                ax.pcolormesh(t[indexstart:indexend], freqs[indexstart:indexend], Pxx[indexstart:indexend])
                ax.set_ylim(minFreq, maxFreq)
                ax.set_xlabel('Time ['+self.timeUnit+']')
                ax.set_ylabel('Frequency [Hz]')

                ax = pl.subplot(122)
                t_filtered, freqs_filtered = numpy.meshgrid(t_filtered, freqs_filtered)
                ax.pcolormesh(t_filtered[indexstart:indexend], freqs[indexstart:indexend], Pxxfiltered[indexstart:indexend])
                ax.set_ylim(minFreq, maxFreq)
                ax.set_xlabel('Time [ms]')

                pl.show()

            return (Pxx, freqs, t), (Pxxfiltered, freqs_filtered, t_filtered)


    def time_offset(self, offset):
        super(spikezugList, self).time_offset(offset)
        if hasattr(self, 'timeAxis'):
            self.timeAxis += offset

    def time_slice(self, t_start, t_stop, meta=None):
        """ Return a new spikezugList obtained by slicing between t_start and t_stop

        Parameters
        ----------
        t_start : float
            Beginning of the new SpikeTrain, in ms.
        t_stop : float
            End of the new SpikeTrain, in ms.
        meta : dict, optional
            Metadata dictionary. Defaults to self.meta.

        Returns
        -------
        new_spikezugList : spikezugList object
        """
        if not meta:
            meta = self.meta

        dims = 1
        if meta.has_key('dimensions'):
           dims = meta['dimensions']
        new_spikezugList = spikezugList(timeUnit=self.timeUnit, t_start=t_start, t_stop=t_stop, dims=dims)
        new_spikezugList.meta = meta

        if inspect.ismethod(self.id_list):
            IDlist = self.id_list()
        else:
            IDlist = self.id_list

        for id in IDlist:
            new_spikezugList.append(id, self.spiketrains[id].time_slice(t_start, t_stop))
            new_spikezugList._SpikeList__calc_startstop()

        for id in self.tags:
            for item in self.tags[id]:
                new_spikezugList.addTags(id, file=self.tags[id]['file'], dir=self.tags[id]['dir'])


        return new_spikezugList


    def overview(self, rate_window=20, method='convolve', kernel='gauss'):
        ''' Overview plot.

        This function gives an overview of activity of cells in terms of firing rates.

        Parameters
        ----------
        rate_window ... width of the kernel in the same time unit as timeAxis('ms' by default
        method ... method for estimating firing rates, it could be 
                "hist" simple histogram
                "convolve" convolution with a kernel of type specified by kernel variable and std = rate_window
        kernel ... type of kernel to be used to convolve the spike trains. to see list of available kernels
        please check the documentation of convolveSpikes method.
        '''
        kernel = kernel
        method = method
        rate_window = rate_window
        crazy_train = numpy.array([])
        f,axarr = pl.subplots(2,sharex=True)
        if method == 'hist':
            ##spike time hist for different cells
            for k in range(len(self)):
                axarr[k+1].hist(self[k].spike_times,bins =int(self[k].duration()/rate_window),\
                                weights = numpy.ones(self[k].spike_times.size)*1000/rate_window)
                axarr[k+1].set_ylabel('Cell #'+str(k)+'\n Firing rate(Hz)')
                crazy_train = numpy.append(crazy_train, self[k].spike_times)
            ## total spike time hist
            axarr[len(self)+1].hist(crazy_train,int(self[k].duration()/rate_window),\
                                    weights = numpy.ones(crazy_train.size)*1000/rate_window)
            axarr[len(self)+1].set_ylabel('Total firing rate \n (Hz)')
            #axarr[len(self)+1].set_xlim()
        elif method == 'convolve':
            self.convolveSpikes(kernel,rate_window)
            self.sumConvolvedSpikeTrains()
            self.burstDetector(kernel, rate_window)
            self.MuaDetector()
            for k in range(len(self)):
#                 axarr[k+1].plot(self.timeAxis[:-1], self.convolvedSpikeTrains[k])
#                 axarr[k+1].axhline(self.convolvedSpikeTrains[k].mean() + 3 * self.convolvedSpikeTrains[k].std(),color = 'r',linewidth = 1)
#                 axarr[k+1].set_ylabel('Cell #'+str(k)+'\n Firing rate(Hz)')
                crazy_train = numpy.append(crazy_train, self[k].spike_times)
#             axarr[len(self)+1].plot(self.timeAxis[:-1], self.convolvedSpikeTrains[k])
            axarr[1].plot(self.timeAxis[:-1], self.summedConvolvedSpikeTrains)
            axarr[1].axhline(self.summedConvolvedSpikeTrains.mean() + 3 * self.summedConvolvedSpikeTrains.std(),color = 'r',linewidth = 1)
            axarr[1].axhline(self.summedConvolvedSpikeTrains.mean(),color = 'g',linewidth = 1)
        
        else:
            print 'the requested method is not defined, please use one of the following: \n \
                    "hist" , "gauss" , "rect"'
        ##raster plots
        for k in range(len(self)):
            band = 1/float(len(self))
            for i in range(len(self[k])):
                axarr[0].plot([self[k].spike_times[i],self[k].spike_times[i]],[k-0.2,k+0.2],'k', linewidth = 1)
        axarr[0].set_ylim(-1,len(self) - 0.5)
        axarr[0].set_yticks(range(len(self)))
        axarr[0].set_ylabel('Cell #')
        for i in range(len(self.MultiUnitActivity)):
            if True:#numpy.diff(self.bursts[i])[0]> 10:
                axarr[0].axvspan(self.MultiUnitActivity[i][0],self.MultiUnitActivity[i][1],alpha = 0.5)
                axarr[0].axvline(self.MultiUnitActivity[i][-1],color = 'r', linewidth = 1)
        return axarr


    def burstDetector(self,kernel = 'gauss',kernel_width =50):
        '''This function detects the episodes with firing activity of all cells above a certain threshold.
        Parameters:
        spikeList
        kernel: Shape of the kernel used for convolving spike trains. Current options are 'gauss' and 'rect'
        kernel_width: It's self-explanatory!!! 
        
        Output:
        bursts: Array containing, start and end time, time of peak and peak value of summedConvolvedSpikeTrain in 
        the format [t_start,t_stop,peak,t_peak] for each row!
        '''
        if (hasattr(self,'convolvedSpikeTrains') or hasattr(self,'summedConvolvedSpikeTrains')) is False:
            print "There is no spiketrain at the moment... \n \
             Calculating spike trains by covolving with a Gaussian kernel of sigma = %d ms" %(kernel_width/4)
            self.convolveSpikes(kernel,kernel_width)
            self.sumConvolvedSpikeTrains()
        else:
            if(kernel_width != 50):
                self.convolveSpikes(kernel,kernel_width)
                self.sumConvolvedSpikeTrains() 
            cnv_train = self.summedConvolvedSpikeTrains
            std = cnv_train.std()
            avg = cnv_train.mean()
            sig1 = numpy.where(cnv_train > (avg + std ))[0]
            sig1_break = numpy.where(numpy.diff(sig1) > 1)[0]
            sig1_split = numpy.split(sig1,sig1_break+1)
            t_chain = []
            t_start = 0.0
            t_end = 0.0
            peak = 0.0
            t_peak = 0.0
            bursts = []
            for i in range(len(sig1_split)):
                if  cnv_train[sig1_split[i]].max() > avg +3 * std:
                    t_chain = numpy.where(cnv_train == cnv_train[sig1_split[i]].max())[0]
                    t_start = self.timeAxis[sig1_split[i][0]]
                    t_end = self.timeAxis[sig1_split[i][-1]]
                    peak = cnv_train[sig1_split[i]].max()
                    t_peak = self.timeAxis[numpy.intersect1d(sig1_split[i],t_chain)][0]
                    bursts.append([t_start, t_end, peak, t_peak])
            bursts = numpy.array(bursts)
            self.bursts = bursts


    def MuaDetector(self):
        ''' This function takes the firing episodes calculated by burstDetector() and picks episodes with multi-cell
        firing.
        Parameters:
        spikeList
        
        Output:
        MultiUnitFiring: Array containing, start and end time, time of peak and peak value of summedConvolvedSpikeTrain in 
        the format [t_start,t_stop,peak,t_peak] for each row!
        '''
        if hasattr(self,'bursts') is False:
            print 'There is no burst detected for this SpikeList. Calling the burstDetector function with default values...'
            self.burstDetector()
        cell_no = 0
        MUF = []
        for item in self.bursts:
            for k in range(len(self)):
                spkt = self[k].spike_times
                less = numpy.less_equal(spkt,item[1])
                more = numpy.less_equal(item[0],spkt)
                cnd = numpy.logical_and(less,more)
                cell_no += bool(numpy.sum(cnd))
            if cell_no > 1:
                MUF.append(item)
            cell_no = 0
        self.MultiUnitFiring = numpy.array(MUF)
        





###################################################### thurley_leibold13_HIPPO


class spikezug_thurley_leibold13_HIPPO(signal, NTsig.SpikeTrain):
    """
    Spike train class that extends the NeuroTools SpikeTrain class.

    Used for analysis of the simulations from thurley_leibold13_HIPPO.
    """

    def __init__(self, spike_times, timeUnit='ms'):
        signal.__init__(self)
        NTsig.SpikeTrain.__init__(self, spike_times)

    def getSpikePhasesFreq(self, osciFreq=1, osciOffset=0):
        """ calculate spike phases with an fixed oscillation frequency """
        if self.timeUnit == 'ms':
            T=1/osciFreq*1000       # period length [ms]
        elif self.timeUnit == 's':
            T=1/osciFreq       # period length [s]
        else:
            print 'WARNING: Time unit not adequately supported.'

        self.spike_phases=numpy.zeros(self.spike_times.shape[0])
        for i, spike in enumerate(self.spike_times):
            self.spike_phases[i]=spike/T
            self.spike_phases[i]-=int(spike/T)
            self.spike_phases[i]*=360
        return self.spike_phases

    def getSpikePhasesT(self, t_start=0, t_stop=0):
        """ calculate spike phases for an cycle lasting from t_start to t_stop """
        T=t_stop-t_start
        timeSlicedTrain=self.time_slice(t_start,t_stop).spike_times
        spike_phases=numpy.zeros(timeSlicedTrain.shape[0])
        for i, spike in enumerate(timeSlicedTrain):
            spike_phases[i]=(spike-t_start)/T
            spike_phases[i]*=360
        return spike_phases


class inputSignals(signal):
    """
    Class used input signals in the analysis of the simulations from thurley_leibold13_HIPPO.
    """

    types={0:'exi', 1:'inh', 2:'el_soma', 3:'condel_soma', 4:'noise_soma', 5:'el_dend', 6:'condel_dend', 7:'noise_dend'}

    def __init__(self, fileName):
        signal.__init__(self)
        self.inputParams = io._read_metadata(fileName)
        for key in self.inputParams.keys():              # convert dictionary to variables of the inputSignals object
            exec 'self.'+key+'='+str(self.inputParams[key])

        self.fileName=fileName
        self.inputs=[]
        self.loadInputs(fileName)
        self.numInputs=self.inputs.__len__()
        self.timeOffset=0.0


    def loadInputs(self,fileName):
        data=numpy.loadtxt(fileName)

        for i in range(self.inputParams['first_id'], self.inputParams['last_id']+1):
            indices=numpy.nonzero(data[:,2]==i)
            self.inputs.append(data[indices[0],0:2])

    def plotInputs(self):
        fig = pl.figure(figsize=(10, 10))
        for i in range(self.numInputs):
            ax = fig.add_subplot(self.numInputs+1,1,i+1)
            if self.inputs[i].size:
                ax.plot(self.inputs[i][:,0],self.inputs[i][:,1])

            ax.set_xlim(self.getMinTime(), self.getMaxTime())
            ax.set_ylabel(self.types[i])
            if i<self.numInputs-1:
                ax.set_xticklabels([])
        ax = fig.add_subplot(self.numInputs+1,1,self.numInputs+1)
        ax.plot(self.summedInputs()[:,0], self.summedInputs()[:,1])
        ax.set_xlim(self.getMinTime(), self.getMaxTime())
        ax.set_ylabel('sum')

        ax.set_xlabel('Time ['+self.timeUnit+']')
        pl.show()

    def offsetTime(self, offset):
        self.timeOffset=offset
        self.inputs[:,0]+=offset

    def getMinTime(self):
        minTime=numpy.inf
        for i in self.inputs:
            if i.size and minTime>i[:,0].min():
                minTime=i[:,0].min()
        return minTime

    def getMaxTime(self):
        maxTime=-numpy.inf
        for i in self.inputs:
            if i.size and maxTime<i[:,0].max():
                maxTime=i[:,0].max()
        return maxTime

    def summedInputs(self):
        summe=self.inputs[0].copy()
        for index0 in range(1,self.inputs.__len__()):
            inp=self.inputs[index0]
            for j, time in enumerate(inp[:,0]):
                index=summe[:,0].searchsorted(time)
                if summe.shape[0]>index and summe[index,0]==time:
                    summe[index,1]+=inp[j,1]
                else:
                    summe=numpy.insert(summe, index, inp[j], axis=0)
        return summe


