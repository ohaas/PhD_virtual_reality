"""
signale.cscs
============

A module for signals from continously sampled channels (CSCs).
"""
__author__ = ("KT", "Alireza Chenani", "Olivia Haas")
__version__ = "4.4.1, October 2014"

# python modules

# other modules
import numpy
import scipy.signal as scsig
import NeuroTools.signals as NTsig
import matplotlib.pyplot as pl
import matplotlib as mpl

# custom made modules
import custom_plot

# package modules
import signals
from signals import signal
##import io
import tools


###################################################### FUNCTIONS

###################################################### CLASSES

class CSCsignal(signal, NTsig.AnalogSignal):
    """
    A class for continuously sampled data.
    """

    def __init__(self, cscSignal, sampleFreq, tags={}, header=None):

        signal.__init__(self, tags=tags, header=header)                         # NOTE: possiblity of name clashes! AnalogSignal ahs an Attribute signal!
        self.sampleFreq = sampleFreq                  # [Hz]
        dt = 1000./sampleFreq                         # [ms] expects sampling frequency in [Hz]

        NTsig.AnalogSignal.__init__(self, cscSignal, dt)

    def __recalc_startstop(self):

        self.t_start = 0.0
        self.t_stop = self.t_start + self.signal.size*self.dt

    def recalc_timeAxis(self):
        # just an alias to run the internal functions
        self.__recalc_startstop()
    #    self.__calcTimeAxis()

    def changeTimeUnit(self, newUnit='ms'):

        factor = signals._getUnitFactor(self.timeUnit, newUnit)

        # change the times
        self.t_start *= factor
        self.t_stop *= factor
        self.dt *= factor

        self.timeUnit = newUnit


    def fft(self, display=False):

        n = self.signal.shape[0]

        if self.timeUnit == 'ms':
            binwidth = self.dt/1000.
        elif self.timeUnit == 's':
            binwidth = self.dt
        self.sp = numpy.fft.rfft(self.signal) #Compute the one-dimensional discrete Fourier Transform for real input.
        self.sp /= self.signal.size
        self.spPower = 2*self.sp*numpy.conj(self.sp)
        self.freq = numpy.fft.fftfreq(n, d = binwidth)[0:n/2+1]

        if display:
            self.fft_plot()


    def fft_plot(self, fig=None, ax=None):

        if not fig:
            fig = pl.figure(figsize=(12, 7))
        if not ax:
            ax = fig.add_subplot(111)

        if not hasattr(self, 'spPower'):
            self.fft()

        ax.plot(self.freq, self.spPower, '-')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_xlim(0, self.freq.max())

        pl.show()

        return fig, ax

    def filter(self, minFreq=None, maxFreq=None):

        if not hasattr(self, 'sp'):
            self.fft()

        if not minFreq:
            #minFreq=self.freq.min()
            minFreq = numpy.nanmin(self.freq)
        if not maxFreq:
            #maxFreq=self.freq.max()
            maxFreq = numpy.nanmax(self.freq)

        # band pass filtering, zero phase
        self.sp_filtered = numpy.zeros_like(self.sp)
        for i, f in enumerate(self.freq):
            if abs(f) >= minFreq and abs(f) <= maxFreq:
                self.sp_filtered[i] = self.sp[i]

        # backtransform filtered signal to time space
        self.signal_filtered = numpy.fft.irfft(self.sp_filtered)
        self.signal_filtered *= self.signal.size          # rescale from normalization of the fourier components
        self.signal_filtered = self.signal_filtered[:self.signal.size]
        

    def removeFreq(self, minFreq=None, maxFreq=None):

        if not hasattr(self, 'sp'):
            self.fft()

        if not minFreq:
            minFreq=self.freq.min()
        if not maxFreq:
            maxFreq=self.freq.max()

        # band pass filtering, zero phase
        self.sp_purged = self.sp.copy()
        for i, f in enumerate(self.freq):
            if abs(f) >= minFreq and abs(f) <= maxFreq:
                self.sp_purged[i] = 0.0

        # backtransform filtered signal to time space
        self.signal_purged = numpy.fft.irfft(self.sp_purged)
        self.signal_purged *= self.signal.size          # rescale from normalization of the fourier components
        self.signal_purged = self.signal_purged[:self.signal.size]

    def removeMean(self):
        self.signal -= self.signal.mean()

    def spectrogram(self, minFreq=None, maxFreq=None, windowSize=4096, overlap=None, display=False):
        # 200 ms bins

        timeUnit = self.timeUnit
        if self.timeUnit != 'ms':
            self.changeTimeUnit('ms')

        # parameters of the spectrogram
        if not overlap:
            overlap=int(windowSize*.9)
        samplingFreq=1./self.dt

        # use matplotlibs specgram function for the spectrogram
        Pxx, freqs, t = mpl.mlab.specgram(self.signal,\
            NFFT=windowSize, noverlap=overlap, pad_to=windowSize*3, Fs=samplingFreq)

        freqs *= 1000           # in [Hz]
        t, freqs = numpy.meshgrid(t, freqs)

        if not minFreq:
            minFreq=freqs.min()
        if not maxFreq:
            maxFreq=freqs.max()

        indexstart = numpy.where(freqs >= minFreq)[0][0]
        indexend = numpy.where(freqs <= maxFreq)[0][-1]

        t = t[indexstart:indexend]
        freqs = freqs[indexstart:indexend]
        Pxx = Pxx[indexstart:indexend]

        if display:
            fig = pl.figure(102, figsize=(10, 7))
            ax = fig.add_subplot(111)

            ax.pcolormesh(t, freqs, Pxx)
            ax.set_xlim(t.min(), t.max())
            ax.set_ylim(freqs.min(), freqs.max())
            ax.set_xlabel('Time ('+self.timeUnit+')')
            ax.set_ylabel('Frequency (Hz)')

            pl.show()

        # switch back time unit if necessary
        if self.timeUnit != timeUnit:
            self.changeTimeUnit(timeUnit)

        return Pxx, freqs, t


    def time_slice(self, t_start, t_stop):
        """ Slice NeuralynxCSC between t_start and t_stop

        Parameters:
            t_start - begining of the new NeuralynxCSCList, in ms.
            t_stop  - end of the new NeuralynxCSCList, in ms.
        """

        assert self.t_start <= t_start
        assert self.t_stop >= t_stop

        t = self.times
        index1 = tools.findNearest(t, t_start)[0]
        index2 = tools.findNearest(t, t_stop)[0]
        self.signal = self.signal[index1:index2]
        self.recalc_timeAxis()
        
    def rms(self,window_size):
        '''
        Calculates the rms of the signal and the filtered signal!!!
        added by ACh Oct 2013
        '''
        ssq = numpy.power(self.signal,2)
        window = numpy.ones(window_size)/float(window_size)
        self.rms_signal = numpy.sqrt(numpy.convolve(ssq, window, 'same'))
        if hasattr(self,'signal_filtered'):
            ssqf = numpy.power(self.signal_filtered,2)
            window = numpy.ones(window_size)/float(window_size)
            self.rms_signal_filtered = numpy.sqrt(numpy.convolve(ssqf, window, 'same'))
        else:
            print 'Currently there is no filtered signal. \n \
            if you have filtered the signal and you want the RMS of filtered \n \
            signal you need to call this function once more after filteration!'


class CSCsignalList(signal, NTsig.AnalogSignalList):

    def __init__(self):
        signal.__init__(self)
        NTsig.AnalogSignalList.__init__(self, [], [], 1.)

    def __recalc_startstop(self):

        try: self.id_list
        except AttributeError:
            logging.warning("id_list is empty")
            self.signal_length = 0
        else:
            signals = self.analog_signals.values()
            self.signal_length = len(signals[0])
            for signal in signals[1:]:
                if len(signal) != self.signal_length:
                    raise Exception("Signals must all be the same length %d != %d" % (self.signal_length, len(signal)))

        for csc in self:
            csc.recalc_timeAxis()

        self.t_start = 0.0
        self.t_stop = self.t_start + self.signal_length*csc.dt

    def changeTimeUnit(self, newUnit='ms'):

        factor = signals._getUnitFactor(self.timeUnit, newUnit)

        for csc in self:
            csc.changeTimeUnit(newUnit)

        self.t_start *= factor
        self.t_stop *= factor
        self.dt *= factor
        self.timeUnit = newUnit

    def getLimits(self):

        self.min = 1e100
        self.max = -1e-100
        for csc in self:
            self.min = min(self.min, csc.signal.min())
            self.max = max(self.max, csc.signal.max())

    def plot(self, fig=None, ax=None):

        try: self.min
        except AttributeError:
            self.getLimits()


        if not fig:
            fig = pl.figure(figsize=(12, 7))
        if not ax:
            ax = fig.add_subplot(111)

        ylabels = []
        last_id = self.id_list()[-1]+1
        for id, csc in enumerate(self):
            s = csc.signal.copy()
            s -= s.mean()
            s /= self.max*.8
            pl.plot(csc.times, s+id, '-', linewidth=1,
                    color=custom_plot.pretty_colors_set2[id%custom_plot.pretty_colors_set2.__len__()])
            ylabels.append(str(csc.tags['channel']).split('.')[0])
        custom_plot.huebschMachen(ax)
        ax.set_xlabel('Time ('+self.timeUnit+')')
        yticks = numpy.arange(last_id)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, rotation=0)
        ax.set_xlim(csc.times[0], csc.times[-1])
        ax.set_ylim(-1, id+1)

        return fig, ax

    def fft_plot(self, freq_min=0., freq_max=60., fig=None, ax=None):

        try: self.min
        except AttributeError:
            self.getLimits()

        if not fig:
            fig = pl.figure(figsize=(12, 7))

        if not ax:
            ax = fig.add_subplot(111)

        ylabels = []
        last_id = self.id_list()[-1]+1
        for id, csc in enumerate(self):
            if not hasattr(csc, 'spPower'):
                csc.fft()
            index_freq_min = numpy.searchsorted(csc.freq, freq_min)
            index_freq_max = numpy.searchsorted(csc.freq, freq_max)
            s = csc.spPower
            s -= s[index_freq_min:index_freq_max].mean()
            s /= s[index_freq_min:index_freq_max].max()*1.1
            pl.plot(csc.freq, s+id, '-', linewidth=1,
                    color=custom_plot.pretty_colors_set2[id%custom_plot.pretty_colors_set2.__len__()])
            ylabels.append(str(csc.tags['channel']).split('.')[0])
        custom_plot.huebschMachen(ax)
        ax.set_xlabel('Frequency (Hz)')
        yticks = numpy.arange(last_id)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, rotation=0)
        ax.set_xlim(freq_min, freq_max)
        ax.set_ylim(0, id+1)

        return fig, ax



    def removeMean(self):
        for id in self.analog_signals:
            self[id].removeMean()

    def showTags(self):
        signal.showTags(self)
        for id in self.analog_signals:
            self[id].showTags()

##    def append(self, id, signal):
##        """
##        Add an NeuralynxCSC object to the NeuralynxCSCList
##
##        Inputs:
##            id        - the id of the new channel
##            signal - the NeuralynxCSC object representing the new cell
##
##        The NeuralynxCSC object is sliced according to the t_start and t_stop times
##        of the NeuralynxCSCList object
##        """
##
##        assert isinstance(signal, NeuralynxCSC), "An NeuralynxCSCList object can only contain NeuralynxCSC objects"
##        if id in self.id_list():
##            raise Exception("ID already present in NeuralynxCSCList.")
##        else:
##            self.signals[id] = signal

    def time_slice(self, t_start, t_stop):
        """ Slice NeuralynxCSCList between t_start and t_stop

        Parameters:
            t_start - begining of the new NeuralynxCSCList, in ms.
            t_stop  - end of the new NeuralynxCSCList, in ms.
        """

        for csc in self:
            csc.time_slice(t_start, t_stop)

        self.__recalc_startstop()


class NeuralynxCSC(CSCsignal):
    """
    A class for Neuralynx continuously sampled channel (CSC) recorded data.
    """

    def __init__(self, times_input, cscSignal, sampleFreq, dataSamplesNums, tags={}, header=None, timeUnit=None, dt=None):

        CSCsignal.__init__(self, cscSignal, sampleFreq, tags=tags, header=header)
        # NOTE: possiblity of name clashes! AnalogSignal ahs an Attribute signal!

        # if timeUnit is None there was no time slice. Otherwise there would be a time_input which is not of ncsStyle
        # and a given timeUnit.
        if timeUnit is None:
            self.times_ncsStyle = numpy.array(times_input)               # time stamps provided in Neuralynx CSC file
            self.recalc_timeAxis()
        else:
            self.times_ncsStyle = None
            self.timeUnit = timeUnit
            self.times = times_input
            self.dt = dt
            self.__recalc_startstop()

        a = header.find('ADBitVolts')
        self.ADBitVolts = 1
        if a+1:
            self.ADBitVolts = float(header[a + 11:a +37])      #AD Unit in Volts! 
        self.dataSamplesNums = dataSamplesNums



    def __recalc_startstop(self):

        if self.times_ncsStyle is not None:
            self.t_start_ncsStyle = self.times_ncsStyle[0]

        try: self.tags['validSamples']
        except KeyError:
            add = 0.
        else:
            add = (self.tags['validSamples']-1)*self.dt

        if self.times_ncsStyle is not None:
            self.t_stop_ncsStyle = self.times_ncsStyle[-1]+add

        self.t_start = 0.0
        self.t_stop = self.t_start + self.signal.size*self.dt

        self.times_start = self.times[0]
        self.times_stop = self.times[-1]



    def __calcTimeAxis(self):

        try: self.tags['validSamples']
        except KeyError:
            self.timeAxis = self.time_axis()+self.t_start_ncsStyle
        else:
            self.timeAxis = numpy.array([])
            for i, t in enumerate(self.times_ncsStyle[:-1]):
                self.timeAxis = numpy.append(self.timeAxis, numpy.linspace(t, self.times_ncsStyle[i+1],
                                                                           self.tags['validSamples'], endpoint=False))
            self.timeAxis = numpy.append(self.timeAxis, numpy.arange(self.times_ncsStyle[-1], self.times_ncsStyle[-1] +
                                                                     self.dt*(self.tags['validSamples']), self.dt))
            #self.timeAxis = numpy.array(self.timeAxis)
            #self.timeAxis = self.timeAxis.flatten()

        self.times = self.timeAxis

        # NOTES regarding time axes, t_start and t_stop etc.
        # the time stamps from the Neuralynx' CSC file might not be linear!
        #   -   therefore self.t_start_ncsStyle and self.t_stop_ncsStyle are place holders corresponding to
        #       the first and last times_ncsStyle with respect to self.times_ncsStyle, i.e., the times stamps
        #   -   t_start is put to time zero and t_stop is infered from the data,
        #       i.e., self.t_stop = self.t_start + len(self.signal)*self.dt.
        #       They can't be offset!
        #   -   time_axis() is the time axis provided by NTsig.AnalogSignal


    def recalc_timeAxis(self):
        # just an alias to run the internal functions
        if not hasattr(self, 'timeAxis'):
            self.__calcTimeAxis()
        self.__recalc_startstop()




    def changeTimeUnit(self, newUnit='ms'):

        factor = signals._getUnitFactor(self.timeUnit, newUnit)

        # change the times
        self.t_start *= factor
        self.t_stop *= factor
        self.dt *= factor

        if hasattr(self,'t_start_ncsStyle'):
            self.t_start_ncsStyle *= factor
            self.t_stop_ncsStyle *= factor
            self.times_ncsStyle *= factor

        self.times *= factor
        self.times_start *= factor
        self.times_stop *= factor

        self.timeUnit = newUnit

    def hilbertTransform(self, display=False, justPhase=True):
        """ Compute the hilbert transform and from that the analytical signal
        of the filtered analog signal and return the phase and
        absolute values of the later.

        Parameters:
            display ... optional display of a figure with the spiketrains and ints convolved counterpart
            justPhase ... just return the hilbert phase"""

        try: self.signal_filtered
        except AttributeError:
            self.filter()
            print "NOTE: Analog signal was not filtered before. Filtered it now using default values."

        self.analyticalTrain = scsig.hilbert(self.signal_filtered.real)
        self.hilbertAbsolute = numpy.absolute(self.analyticalTrain)
        self.hilbertPhase = numpy.angle(self.analyticalTrain, deg=True)
        self.hilbertAbsSmooth = numpy.convolve(self.hilbertAbsolute, scsig.gaussian(50, 25), 'same')
        for i in range(self.hilbertPhase.shape[-1]):
            if self.hilbertPhase[i] < 0:
                self.hilbertPhase[i] += 360

        if display:
            fig = pl.figure(103, figsize=(12, 7))
            ax = fig.add_subplot(311)

            ax.plot(self.times, self.analyticalTrain.real)
            ax.plot(self.times, self.analyticalTrain.imag)
            ax.set_xlim(self.times_start, self.t_stop)
            ax.legend(['Signal', 'Hilbert transform'])

            ax = fig.add_subplot(312)
            ax2 = ax.twinx()
            ax.plot(self.times, self.hilbertAbsolute)
            ax2.plot(self.times, self.hilbertPhase, 'g')
            ax.set_xlim(self.times_start, self.t_stop)
            ax2.set_ylim(0, 360)
            ax2.set_yticks(range(0, 410, 90))
            ax2.set_yticklabels(range(0, 410, 90))
            ax.set_xlabel('Time ('+self.timeUnit+')')
            ax.set_ylabel('Absolute value')
            ax2.set_ylabel('Phase (deg)')

            ax = fig.add_subplot(313)
            ax2=ax.twinx()
            ax.plot(self.times, self.analyticalTrain.real)
            ax2.plot(self.times, self.hilbertPhase,'g')
            ax.set_xlim(self.times_start, self.t_stop)
            ax2.set_ylim(0, 360)
            ax.set_xlabel('Time ('+self.timeUnit+')')
            ax.set_ylabel('Signal')
            ax2.set_ylabel('Phase (deg)')

            pl.show()

        if justPhase:
            return self.hilbertPhase
        else:
            return self.hilbertPhase, self.hilbertAbsolute, self.analyticalTrain


    def plot(self, fig=None, ax=None):

        if not fig:
            fig = pl.figure(figsize=(12, 7))
        if not ax:
            ax = fig.add_subplot(111)

        ax.plot(self.times, self.signal, '-')

        ax.set_xlabel('Time ('+self.timeUnit+')')
        ax.set_xlim(self.times_start, self.times_stop)

        pl.show()

        return fig, ax


    def purge(self):
        """ Remove values that fill up the possible range.
        """

        ds = numpy.diff(self.signal)
        indices, = numpy.where(ds == 0)
        indices = numpy.concatenate((indices-1, indices, indices+1))
        self.signal = numpy.delete(self.signal, indices)

        # to do properly:
        #self.times_ncsStyle ... purge as well

        self.recalc_timeAxis()

    def resample(self, new_dt):
        if new_dt > self.dt:
            bins = int(new_dt/self.dt)  # NOTE: Not interpolating!!!!!!!
            print bins
            print new_dt/self.dt
            self.signal = self.signal[::bins]
            self.dt = new_dt
        else:
            print "Not implemented yet."


    def time_offset(self, offset):
        """
        Add a time offset to the NeuralynxCSC object.
        """
        self.times_ncsStyle += offset
        self.t_start_ncsStyle += offset
        self.t_stop_ncsStyle += offset

        self.times += offset
        self.times_start += offset
        self.times_stop += offset

    def time_slice(self, t_start, t_stop):
        """ Slice NeuralynxCSC between t_start and t_stop

        Parameters
        ----------
        t_start : float
            beginning of the new NeuralynxCSCList, in ms.
        t_stop  : float
            end of the new NeuralynxCSCList, in ms.
        """

        t = self.times

        #index1 = numpy.searchsorted(t, t_start)
        index1 = tools.findNearest(t, t_start)[0]
        #index2 = numpy.min([numpy.searchsorted(t, t_stop), t.size])
        index2 = tools.findNearest(t, t_stop)[0]

        times = t[index1:index2]
        signal = self.signal[index1:index2]
        dummy_tags = dict(self.tags)
        dummy_tags.pop('validSamples')

        sliced_csc = NeuralynxCSC(times, signal, self.sampleFreq, self.dataSamplesNums, dummy_tags, self.header,
                                  timeUnit=self.timeUnit, dt=self.dt)

        return sliced_csc

    def concatenate(self, csc):
        """ Concatenate NeuralynxCSC 'csc'

        Parameters
        ----------
        csc : NeuralynxCSC object

        """

        # inserting the nan makes problems when backtransforming the fft (in class CSCsignal.filter)
        #times = numpy.concatenate((numpy.append(self.times, numpy.nan), csc.times))
        times = numpy.concatenate((self.times, csc.times))
        #signal = numpy.concatenate((numpy.append(self.signal, numpy.nan), csc.signal))
        signal = numpy.concatenate((self.signal, csc.signal))
        dummy_tags = dict(self.tags)

        if 'validSamples' in dummy_tags:
            dummy_tags.pop('validSamples')

        concatenated_csc = NeuralynxCSC(times, signal, self.sampleFreq, self.dataSamplesNums, dummy_tags, self.header,
                                  timeUnit=self.timeUnit, dt=self.dt)

        return concatenated_csc
        
    def ripple_recorder(self, rippleMix = False, rippleCut = False):
        '''
            Function for detecting and recording ripples in a single csc! It takes a csc object,
            detect the ripples and add them as an attribute to the csc object.
    
            written by achenani October 2013
        '''
        ripples = []
        if not hasattr(self,'hilbertAbsolute'):
            print 'Hilbert transform is not calculated!\n Calculating Hilbert transform of the signal...'
            self.hilbertTransform()
        if not hasattr(self,'signal_filtered'):
            print 'Signal is not filtered yet!\n Filtering the signal with the default values...'
            self.filter(100,250)
        smooth = numpy.convolve(self.hilbertAbsolute,scsig.gaussian(50,25),'same')
        std = smooth.std()
        avg = smooth.mean()
        sig1 = numpy.where(smooth > (avg + std ))[0]
        sig1_break = numpy.where(numpy.diff(sig1) > 1)[0]
        sig1_split = numpy.split(sig1,sig1_break+1)
        t_chain = []
        t_start = 0.0
        t_end = 0.0
        peak = 0.0
        t_peak = 0.0
        ripple_mix = []
        ripple_cut = numpy.zeros(self.signal.size)
        for i in range(len(sig1_split)):
            if  smooth[sig1_split[i]].max() > avg +3 * std:
                t_chain = numpy.where(smooth == smooth[sig1_split[i]].max())[0]
                t_start = self.timeAxis[sig1_split[i][0]]
                t_end = self.timeAxis[sig1_split[i][-1]]
                peak = smooth[sig1_split[i]].max()
                t_peak = self.timeAxis[numpy.intersect1d(sig1_split[i],t_chain)][0]
                ripples.append([t_start, t_end, peak, t_peak])
            if rippleMix == True:
                ripple_mix.append(self.signal_filtered[sig1_split[i]])
            if rippleCut == True:
                ripple_cut[sig1_split[i]] =  self.signal_filtered[sig1_split[i]]   
        if rippleMix == True:
            ripple_mix = numpy.array(ripple_mix)
            self.rippMX = ripple_mix
        if rippleCut == True:
            self.signal_just_ripples = ripple_cut
        ripples = numpy.array(ripples)
        self.ripples = ripples
        print 'Ripple detection on' ,self.tags['file'],' is DONE!!!\n \
        Now you should see the attribute ripples with [t_start, t_end,peak value, t_peak] in each row for detected SWRs.'

    def rmsPlot(self):
        mv_converter = self.ADBitVolts * 1e6 # converting factor to mV for the ampilitudes
        f,axarr = pl.subplots(4,sharex=True)
        ##Hilbert transformed signal
        axarr[0].plot(self.timeAxis[0:self.signal.size]/1000,self.hilbertAbsolute * mv_converter)
        axarr[0].axhline(y = (self.hilbertAbsolute.mean()+ self.hilbertAbsolute.std()) * mv_converter,color = 'k')
        axarr[0].axhline(y =(self.hilbertAbsolute.mean() + 3 * self.hilbertAbsolute.std()) * mv_converter,color = 'g')
        axarr[0].set_title('Hilbert Detection Method')
        axarr[0].set_xlim(self.timeAxis[0]/1000,self.timeAxis[-2]/1000)
        
        ##filtered_rms signal 
        axarr[1].plot(self.timeAxis[0:self.signal.size]/1000,self.rms_signal_filtered * mv_converter,'m')
        axarr[1].axhline(y =(self.rms_signal_filtered.mean() + self.rms_signal_filtered.std()) * mv_converter,color = 'k')
        axarr[1].axhline(y =(self.rms_signal_filtered.mean() + 3 * self.rms_signal_filtered.std()) * mv_converter,color = 'g')
        axarr[1].set_title('RMS Detection Method')
        
        ##filtered signal
        axarr[2].plot(self.timeAxis[0:self.signal.size]/1000,self.signal_filtered * mv_converter,'r')
        axarr[2].set_ylabel('Ampilitude(mV)')
        axarr[2].set_title('Filtered')
        
        ##raw signal
        axarr[3].plot(self.timeAxis[0:self.signal.size]/1000,self.signal * mv_converter,'k')
        axarr[3].set_xlabel('Time(s)')
        axarr[3].set_title('Signal')
        f.suptitle(self.tags.get('file') + 'on channel: ' + str(self.tags.get('channel')))
        pl.show()

    def ripplePlot(self):
        f,axarr = pl.subplots(3,sharex=True)
        if not hasattr(self,'ripples'):
            print 'Ripples are not detected yet, please use ripplerecorder function to detect them first!'
        else:
            mv_converter = self.ADBitVolts * 1e6 # converting factor to mV for the ampilitudes
            ##filtered_HilbertAbsolute signal 
            for i in range(len(self.ripples)):
                axarr[0].axvspan(self.ripples[i][0], self.ripples[i][1], facecolor='g', alpha=0.5)
                axarr[0].axvline(self.ripples[i][-1], linewidth = 1)
            axarr[0].plot(self.timeAxis[0:self.signal.size],self.hilbertAbsolute * mv_converter,'m')
            axarr[0].set_title('HilbertAbs of filtered signal')
            axarr[0].axhline(y =(self.hilbertAbsolute.mean() + self.hilbertAbsolute.std()) * mv_converter,color = 'k', linewidth =1)
            axarr[0].axhline(y =(self.hilbertAbsolute.mean() + 3 * self.hilbertAbsolute.std()) * mv_converter,color = 'g', linewidth = 1)
            axarr[0].set_xlim(self.timeAxis[0],self.timeAxis[-2])
    
            ##filtered signal
            for i in range(len(self.ripples)):
                axarr[1].axvspan(self.ripples[i][0], self.ripples[i][1], facecolor='g', alpha=0.5)
            axarr[1].plot(self.timeAxis[0:self.signal.size],self.signal_filtered * mv_converter,'r')       
            axarr[1].set_ylabel('Ampilitude(mV)')
            axarr[1].set_title('Filtered Signal')
            
                        
            ##raw signal
            for i in range(len(self.ripples)):
                axarr[2].axvspan(self.ripples[i][0], self.ripples[i][1], facecolor='g', alpha=0.5)
            axarr[2].plot(self.timeAxis[0:self.signal.size],self.signal * mv_converter,'k')
            axarr[2].set_xlabel('Time(ms)')
            axarr[2].set_title('Raw Signal')
            f.suptitle(self.tags.get('file') + 'on channel: ' + str(self.tags.get('channel')))
            pl.show()

    def iri(self):
        ''' 
        this function calculates the inter-ripple intervals (peak to peak) on RMS signal
        written by AChenani
        '''
        if not hasattr(self,'ripples'):
            print 'Ripples are not detected yet, please use ripplerecorder function to detect them first!'
        else:
            iri = numpy.diff(self.ripples[:,3])
            self.iri = iri
        return
    


class NeuralynxCSCList(CSCsignalList):

    def __init__(self):
        CSCsignalList.__init__(self)

    def __recalc_startstop(self):

        try: self.id_list
        except AttributeError:
            logging.warning("id_list is empty")
            self.signal_length = 0
        else:
            start_times = numpy.array([self.analog_signals[id].timeAxis[0] for id in self.id_list()], numpy.float32)
            stop_times = numpy.array([self.analog_signals[idx].timeAxis[-1] for idx in self.id_list()], numpy.float32)

            signals = self.analog_signals.values()
            self.signal_length = len(signals[0])
            for signal in signals[1:]:
                if len(signal) != self.signal_length:
                    raise Exception("Signals must all be the same length %d != %d" % (self.signal_length, len(signal)))

        for csc in self:
            csc.__recalc_startstop()

        self.t_start = 0.0
        self.t_stop = self.t_start + self.signal_length*csc.dt

    def plot(self, fig=None, ax=None):

        fig, ax = CSCsignalList.plot(self, fig=fig, ax=ax)

        ylabels = []
        for id, csc in enumerate(self):
            ylabels.append(csc.tags['file'].split('.')[0])
        ax.set_yticklabels(ylabels, rotation=0)

        return fig, ax


    def fft_plot(self, freq_min=0., freq_max=60., fig=None, ax=None):

        fig, ax = CSCsignalList.fft_plot(self, freq_min=freq_min, freq_max=freq_max, fig=fig, ax=ax)

        ylabels = []
        for id, csc in enumerate(self):
            ylabels.append(csc.tags['file'].split('.')[0])
        ax.set_yticklabels(ylabels, rotation=0)

        return fig, ax


    def cscCompare(self, csc_no = 0):
        '''
        This function plots all the raw signals with SWR events detected on one of the signals indicated by csc_no variable.
        Last Modified Oct. 2013
        '''
        if csc_no > len(self) or csc_no < 1:
            print 'The argument must be between 1 and the length of cscList object!!!'
        elif not hasattr(self[csc_no -1],'ripples'):
            print 'Ripples are not detected yet, please use ripplerecorder function to detect them first!\n Please try agin...' 
        else:
            f,axarr = pl.subplots(len(self) + 2,sharex=True)
            mv_converter = self[csc_no - 1].ADBitVolts * 1e6 # converting factor to mV for the ampilitudes
            ##filtered_rms signal 
            for i in range(len(self[csc_no-1].ripples)):
                axarr[0].axvspan(self[csc_no -1].ripples[i][0]/1000., self[csc_no - 1].ripples[i][1]/1000., facecolor='g', alpha=0.5)
                axarr[0].axvline(self[csc_no - 1].ripples[i][-1]/1000., linewidth = 1)
            axarr[0].plot(self[csc_no - 1].timeAxis[0:self[csc_no - 1].signal.size]/1000.,self[csc_no - 1].hilbertAbsolute * mv_converter,'m')
            axarr[0].axhline(y =(self[csc_no - 1].hilbertAbsolute.mean() + self[csc_no - 1].hilbertAbsolute.std()) * mv_converter,color = 'k', linewidth =1)
            axarr[0].axhline(y =(self[csc_no - 1].hilbertAbsolute.mean() + 3 * self[csc_no - 1].hilbertAbsolute.std()) * mv_converter,color = 'g', linewidth = 1)
            axarr[0].set_xlim(self[csc_no - 1].timeAxis[0]/1000,self[csc_no - 1].timeAxis[-2]/1000)
            axarr[0].set_title(self[csc_no -1].tags.get('file') + 'on channel: ' + str(self[csc_no - 1].tags.get('channel')),fontsize = 10)

            ##filtered signal
            for i in range(len(self[csc_no - 1].ripples)):
                axarr[1].axvspan(self[csc_no - 1].ripples[i][0]/1000., self[csc_no - 1].ripples[i][1]/1000., facecolor='g', alpha=0.5)
            axarr[1].plot(self[csc_no - 1].timeAxis[0:self[csc_no - 1].signal.size]/1000,self[csc_no - 1].signal_filtered * mv_converter,'r')       
            axarr[1].set_ylabel('Ampilitude(mV)')
            axarr[1].set_title('Filtered')
            axarr[1].set_title(self[csc_no - 1].tags.get('file') + 'on channel: ' + str(self[csc_no - 1].tags.get('channel')),fontsize = 10)

                        
            ##raw signal
            for k in range(2,len(axarr)):
                for i in range(len(self[k - 2].ripples)):
                    axarr[k].axvspan(self[k - 2].ripples[i][0]/1000., self[k - 2].ripples[i][1]/1000., facecolor='g', alpha=0.5)
                axarr[k].plot(self[k - 2].timeAxis[0:self[k - 2].signal.size]/1000,self[k - 2].signal * mv_converter,'k')
                #axarr[k].set_xlabel('Time(s)')
                axarr[k].set_title(self[k -2].tags.get('file') + 'on channel: ' + str(self[k - 2].tags.get('channel')),fontsize = 10)
            #f.suptitle(self[1].tags.get('file') + 'on channel: ' + str(self.tags.get('channel')))
            pl.show()
            return axarr
    
    def avg_csc(self):
        avg = numpy.zeros(self[0].signal.size)
        for i in range(len(self)):
            avg = numpy.add(avg,self[i].signal)
        avg /= float(len(self))
        csc_avg = NeuralynxCSC(self[0].times,avg, self[0].sampleFreq)
        csc_avg.tags['file'] = 'CSC Average'
        csc_avg.tags['channel'] = 'ZDF'
        self.append(len(self),csc_avg)

    def rippleDetect(self,smooth = 0):
        '''
            Function for detecting and recording ripples in an experiment! It takes a cscList object,
            averages the smoothed hilbert signal and then detect the ripples using threshhold method
            and add them as an attribute to the cscList object.
            parameters:
            smooth: window length of the gaussian kernel to be used for smoothing the HilbertAbs signal.
            default value is zero i.e. no smoothie! :(
    
            Last modified on Nov 2013
        '''
        ripples = []
        for item in self:
            if not hasattr(item,'hilbertAbsolute'):
                print 'Hilbert transform is not calculated!\n Calculating Hilbert transform of the signal for'+item.tags['file']+'...'
                item.hilbertTransform()
            if not hasattr(item,'signal_filtered'):
                print 'Signal is not filtered yet!\n Filtering the signal with the default values...'
                item.filter(150,250)
        hs_signal = numpy.zeros(self[0].hilbertAbsolute.size)   #averaging Hilbert signals!
        for i in range(len(self)):
            hs_signal = numpy.add(hs_signal,self[i].hilbertAbsolute)
        hs_signal /= float(len(self))
        if smooth > 0:
            hs_signal = numpy.convolve(hs_signal,scsig.gaussian(smooth,smooth),'same')
        std = hs_signal.std()
        avg = hs_signal.mean()
        sig1 = numpy.where(hs_signal > (avg + std ))[0]
        sig1_break = numpy.where(numpy.diff(sig1) > 1)[0]
        sig1_split = numpy.split(sig1,sig1_break+1)
        t_chain = []
        t_start = 0.0
        t_end = 0.0
        peak = 0.0
        t_peak = 0.0
        for i in range(len(sig1_split)):
            if  hs_signal[sig1_split[i]].max() > avg +3 * std:
                t_chain = numpy.where(hs_signal == hs_signal[sig1_split[i]].max())[0]
                t_start = self[0].timeAxis[sig1_split[i][0]]
                t_end = self[0].timeAxis[sig1_split[i][-1]]
                peak = hs_signal[sig1_split[i]].max()
                t_peak = self[0].timeAxis[numpy.intersect1d(sig1_split[i],t_chain)][0]
                ripples.append([t_start, t_end, peak, t_peak])
                
        ripples = numpy.array(ripples)
        self.ripples = ripples
        self.hs_signal = hs_signal
        print 'Ripple detection  is DONE!!!\n \
        Now you should see the attribute ripples with [t_start, t_end,peak value, t_peak] in each row for detected SWRs.'

    def badCSCs(self):
        '''
        A simple function to detect bad CSCs without visual inspection!
        Note: It only detects CSCs with unusual "quality". So if all CSCs are equally bad or good 
        it will not notice the difference!
        '''
        percent = numpy.array([])
        for i in range(len(self)):
            max_points = numpy.where(self[i].signal ==self[i].signal.max())[0]
            percent = numpy.append(percent,max_points.size / float(self[i].signal.size) * 100)
        order_of_mag = numpy.log10(percent)
        thresh = 10**(order_of_mag.mean() + order_of_mag.std())
        for i in range(percent.size):
            if percent[i] > thresh:
                print self[i].tags['file']
        self.prcnt = percent

        
    
                    

##    def append(self, id, signal):
##        """
##        Add an NeuralynxCSC object to the NeuralynxCSCList
##
##        Inputs:
##            id        - the id of the new channel
##            signal - the NeuralynxCSC object representing the new cell
##
##        The NeuralynxCSC object is sliced according to the t_start and t_stop times
##        of the NeuralynxCSCList object
##        """
##
##        assert isinstance(signal, NeuralynxCSC), "An NeuralynxCSCList object can only contain NeuralynxCSC objects"
##        if id in self.id_list():
##            raise Exception("ID already present in NeuralynxCSCList.")
##        else:
##            self.signals[id] = signal

