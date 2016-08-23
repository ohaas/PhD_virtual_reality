"""
signale.signals
===============

A module for signal classes.
"""
__author__ = ("KT")
__version__ = "4.2.2, August 2014"


# python modules
import inspect


# other modules
import numpy
from scipy.stats import norm

import matplotlib.pyplot as pl
import matplotlib as mpl


# custom made modules
import custom_plot

# package modules
##import io



###################################################### FUNCTIONS


def _getUnitFactor(timeUnit, newUnit):
    """
    Returns factor to change time unit.
    """

    factor = 1.
    if timeUnit == 'us':
        if newUnit == 'ms':
            factor /= 1000
        elif newUnit == 's':
            factor /= 1000*1000
        elif newUnit == 'min':
            factor /= 1000*1000*60
        elif newUnit == 'h':
            factor /= 1000*1000*60*60
    elif timeUnit == 'ms':
        if newUnit == 'us':
            factor *= 1000
        elif newUnit == 's':
            factor /= 1000
        elif newUnit == 'min':
            factor /= 1000*60
        elif newUnit == 'h':
            factor /= 1000*60*60
    elif timeUnit == 's':
        if newUnit == 'us':
            factor *= 1000*1000
        elif newUnit == 'ms':
            factor *= 1000
        elif newUnit == 'min':
            factor /= 60
        elif newUnit == 'h':
            factor /= 60*60
    elif timeUnit == 'min':
        if newUnit == 'us':
            factor *= 60*1000*1000
        elif newUnit == 'ms':
            factor *= 60*1000
        elif newUnit == 's':
            factor *= 60
        elif newUnit == 'h':
            factor /= 60
    elif timeUnit == 'h':
        if newUnit == 'us':
            factor *= 60*60*1000*1000
        elif newUnit == 'ms':
            factor *= 60*60*1000
        elif newUnit == 's':
            factor *= 60*60
        elif newUnit == 'min':
            factor *= 60

    return factor





###################################################### CLASSES

class signal(object):

    def __init__(self, tags=None, header=None):
        self.meta = {}
        if tags:
            self.tags = tags                                       # for comments/tags to the spiketrains
        else:
            self.tags = {}
        self.header = header
        self.timeUnit = 'ms'

    def addTags(self, id, **kwargs):
        """
        For adding and changing comments/tags to the signal object.
        """
        pass

    def showTags(self):
        """
        For print the comments/tags to the signal object on the command line.
        """
        for key in self.tags:
            print key, ':', self.tags[key]
        print ''

    def changeTimeUnit(self):
        print 'WARNING: Not implemented!'




class NeuralynxEvents(signal):
    """
    A class for Neuralynx event data.
    """

    def __init__(self, times, eventStrings):
        signal.__init__(self)
        self.times = numpy.array(times)
        self.eventStrings = eventStrings

        self.t_start = -1.
        self.t_stop = -1.
        self.t_startRecording = -1.
        self.t_stopRecording = -1.

        self.t_startRecording = self.times[0]           # remember the time of the start of the recording separatly
        if self.eventStrings[0] == 'Starting Recording':
            self.times = self.times[1:]                 # remove entry of the start of the recording
            self.eventStrings = self.eventStrings[1:]   # remove entry of the start of the recording
        #if self.eventStrings[0].find('Digital Lynx Parallel Input Port TTL')+1:
        self.t_start = self.times[0]            # remember the time of the first TTL pulse as t_start

        self.t_stopRecording = self.times[-1]       # remember the time of the end of the recording separatly
        if self.eventStrings[-1] == 'Stopping Recording':
            self.times = self.times[:-1]                # remove corresponding entry
            self.eventStrings = self.eventStrings[:-1]  # remove corresponding entry

    def changeTimeUnit(self, newUnit='ms'):

        factor = _getUnitFactor(self.timeUnit, newUnit)

        # change the times
        self.times *= factor
        self.t_start *= factor
        self.t_stop *= factor
        self.timeUnit = newUnit

        self.t_startRecording *= factor
        self.t_stopRecording *= factor


    def purge(self, numItems):
        """
        Cut away numItems.
        """
        self.times = self.times[numItems:]
        self.eventStrings = self.eventStrings[numItems:]


    def time_offset(self, offset):
        """
        Add a time offset to the NeuralynxEvents object. times, t_start and t_stop are
        shifted from offset.
        """

        if not self.times.size:
            print 'WARNING: times array is empty, can not offset time.'
            return

        self.times += offset
        self.t_start = self.times[0]
        self.t_stop = self.times[-1]

        self.t_startRecording += offset
        self.t_stopRecording += offset


