"""
trajectory.trajectory
=====================

Base module for spatial paths, i.e., trajectories.
"""

__author__ = ("KT", "Moritz Dittmeyer", "Olivia Haas", "Benedikt Ludwig", "Sven Schoernich")
__version__ = "9.0, September 2014"

# python modules
import os, sys, re
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)-2s: %(message)s')
logger = logging.getLogger('trajectory.trajectory')

# other modules
import numpy
import scipy.stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pl
from matplotlib.collections import LineCollection
import matplotlib as mpl
from itertools import count

# custom made modules
import custom_plot
import signale

# pl.ioff()

# package modules


###################################################### FUNCTIONS

def str2number(strs=[]):

    dummy = []
    if isinstance(strs, str):
        strs = [strs]

    for j, s in enumerate(strs):
        dummy.append(int(''.join(i for i in s if i.isdigit())))

    return dummy


def _change_folderStyle(dirName):
    """
    Change folder style from Windows to Linux and Python, respectively.
    """

    name = ''
    for d in dirName.split('\\'):
        name += d + '/'

    return name


def _getTimeUnitFactor(timeUnit, newUnit):
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


def align(a0, b0, shift=None, display=False):

    if a0.timeUnit != b0.timeUnit:
        sys.exit('Time units unequal!')

    # if not given, infer optimal shift from the data
    if shift == None:

        a = a0.times[10:-10]                # small trick to skip strange boarder values
        b = b0.times[10:-10]                # small trick to skip strange boarder values
        index = numpy.argmax(numpy.correlate(numpy.diff(a), numpy.diff(b), 'full'))
        shift = index - numpy.diff(b).size + 1

    # apply shift and crop to the length of the shorter times array
    print "Skipping (i.e. shifting) data for first", shift, "bins"
    if shift > 0:
        a0.times = a0.times[shift:]
        a0.eventStrings = a0.eventStrings[shift:]
        a0.t_start = a0.times[0]
    elif shift < 0:
        shift = numpy.abs(shift)
        b0.times = b0.times[shift:]
        b0.places = b0.places[shift:]
        b0.t_start = b0.times[0]

    # hier noch unabhaengig von spezifischer klassenzugehoerigkeit machen,
    #           d.h. gesonderte schneidemethode nutzen (time oder idslice)
    a0_size = a0.times.size
    b0_size = b0.times.size
    if a0_size > b0_size:
        a0.times = a0.times[:b0_size]
        a0.eventStrings = a0.eventStrings[:b0_size]
    elif a0_size < b0_size:
        b0.times = b0.times[:a0_size]
        b0.places = b0.places[:a0_size]

    if display:
        fig = pl.figure()

        diff0 = numpy.diff(a0.times)
        diff1 = numpy.diff(b0.times)
        ax = fig.add_axes([0.15, 0.59, 0.5, 0.36])
        ax.plot(diff0)
        ax.plot(diff1)
        ax.legend(['a0', 'b0'])
        ax.set_xticklabels([])
        custom_plot.huebschMachen(ax)

        ax = fig.add_axes([0.7, 0.59, 0.23, 0.36])
        if diff0 == []:
            print 'eventData times: ', a0.times
            print 'diff0: ', diff0
        ax.hist(diff0, 70, normed=True, orientation='horizontal')
        ax.hist(diff1, 70, normed=True, orientation='horizontal')
        ax.set_yticklabels([])
        custom_plot.huebschMachen(ax)

        length = min(diff0.size, diff1.size)
        diff = diff0[:length] - diff1[:length]
        ax = fig.add_axes([0.15, 0.14, 0.5, 0.36])
        ax.plot(diff)
        ax.set_ylabel('Time bin difference ('+a0.timeUnit+')')
        ax.set_xlabel('Time ('+a0.timeUnit+')')
        custom_plot.huebschMachen(ax)

        ax = fig.add_axes([0.7, 0.14, 0.23, 0.36])
        ax.hist(diff, 70, normed=True, orientation='horizontal')
        ax.set_yticklabels([])
        custom_plot.huebschMachen(ax)

        fig.text(.995, .01, "Array sizes before cropping to same length: "+str(a0_size)+' and '+str(b0_size) +
                            ". Skipping (shifting) data for first " + str(shift) + " bins.",
                 fontsize=10,  horizontalalignment='right')

    return a0, b0


def convert(text):

    try:
        return int(text)
    except ValueError:
        return text


def natural_sort(l):

    alphanum_key = lambda key: [convert(c) for c in re.split('([-]?\d+)', key)]
    return sorted(l, key=alphanum_key)


###################################################### CLASSES

class trajectory(object):

    def __init__(self, traj, meta={}):

        # initialize meta data
        self.meta = meta
        self.dt = 0.1                         # claim a dt (s)
        self.t_start = 0.0
        self.mazeType = ''
        self.timeUnit = 's'
        self.spaceUnit = 'm'
        self.view_eulerOffsets = [0, 0, 0]
        self.euler = -numpy.array(self.view_eulerOffsets)       # for storing the current euler angles
        self.euler %= 360
        self.threshspeed = 0.                # minimum speed for running (m/s)


        # get real values if parsable from file
        if meta.has_key('dt'):
            self.dt = meta['dt']                  # dt [s]
        if meta.has_key('t_start'):
            self.t_start = meta['t_start']        # t_start [s]
        if meta.has_key('time'):
            self.time = meta['time']              # time and date, when
                                                  # recording was started,
                                                  # just to provide some time stamp

        if meta.has_key('mazetype'):
            self.mazeType = meta['mazetype']

        self.times = []
        self.has_timeAxis = False
        if traj.shape[1] == 4:                      # with time stamp?
            self.times = traj[:, 0]
            self.t_start = numpy.min(self.times)
            self.t_stop = numpy.max(self.times)
            self.places = traj[:, 1:]
            self.numPlaces = self.places.shape[0]
            self.has_timeAxis = True
        elif traj.shape[1] == 3:                    # without time stamp?
            self.places = traj
            self.numPlaces = self.places.shape[0]
            self.t_stop = (self.numPlaces - 1)*self.dt + self.t_start
        else:
            print traj
            sys.exit('Traj shape not implemented!')

        if self.dt > 0 and not self.times.__len__():                    # calculate times if not provided
            self.times = numpy.arange(self.t_start, self.t_stop+self.dt, self.dt)   # gives the times, when the place is entered
            if self.times.shape[0] > self.places.shape[0]:
                self.times = self.times[:self.places.shape[0]]          # bad tweak, since sometimes t_stop
                                                                        # gets included by arange although it shouldn't!?

        self.turn(self.view_eulerOffsets[0])       # remove the yaw component of view_eulerOffsets


        # if dt was not provided, get it from times array
        if not meta.has_key('dt'):
            self.dt = numpy.mean(numpy.diff(self.times))
        #print "dt is", self.dt


    def __recalc_startstop(self):

        try: self.times
        except AttributeError:
            pass
        else:
            self.t_start = numpy.min(self.times)
            self.t_stop = numpy.max(self.times)


    def __changeUnits(self, newTimeUnit=None, newSpaceUnit=None):

        if newTimeUnit:
            factor = _getTimeUnitFactor(self.timeUnit, newUnit)

            # change the times
            self.times *= factor
            self.dt *= factor
            self.threshspeed /= factor
            self.timeUnit = newUnit

            self.__recalc_startstop()

        elif newSpaceUnit:
            print 'NOTE: Changing spaceUnit not implemented.'
        else:
            print 'NOTE: No units were changed.'

    def centerTraj(self):
        """ Shifts the trajectory center to position (0,0).
        """
        self.getTrajDimensions()

        for i, p in enumerate(self.places):
            self.places[i,0] += self.xWidth/2
            self.places[i,1] += self.yWidth/2

    def cumPlaces(self, iStart=None, iStop=None):
        speed = self.getSpeed()[1]
        cumPlace = numpy.cumsum(speed[iStart:iStop])*self.dt
        cumPlace = numpy.append([0], cumPlace)
        return cumPlace

    def getSpeed(self, thresh=None, vec=False, laps=False):
        """ Calculate running speed.

        Parameters
        ----------
        thresh : float
            optional argument to set a minimum

        Returns
        -------
        time : ndarray
        speed : ndarray
        """

        if laps:
            diffsTmp = []
            diffsTmp2 = []
            laps, lapTimes = self.getLaps()
            for item in lapTimes:
                lapStart = self.getIndexFromTime(item[0])
                lapEnd = self.getIndexFromTime(item[1])
                diffsTmp.append(numpy.diff(self.places[lapStart:lapEnd], axis=0)[:, 0:2])
                diffsTmp2.append(numpy.diff(self.times[lapStart:lapEnd]))

            diffs = []
            diffsTimes = []
            for index, item in enumerate(diffsTmp):
                for subitem in item:
                    diffs.append(subitem)
                diffsTimes.append(diffsTmp2[index])
            diffs = numpy.array(diffs)
            diffsTimes = numpy.array(diffsTimes)
        else:
            diffs = numpy.diff(self.places, axis=0)[:, :2]
            diffsTimes = numpy.diff(self.times)

        if vec:
            speed = diffs/diffsTimes
        else:
            speed = numpy.sqrt(diffs[:, 1]**2+diffs[:, 0]**2)/diffsTimes       # [space units/s]

        speed_dummy = []
        reducedTimes = []
        if thresh and not vec:
            for i, s in enumerate(speed):
                if s >= thresh:
                    speed_dummy.append(s)
                    reducedTimes.append(self.times[i])
        else:
            speed_dummy = speed
            reducedTimes = self.times[:-1]
        return numpy.array(reducedTimes), numpy.array(speed_dummy)


    def getAcceleration(self, thresh=None, abs=False, vec=False, laps=False):
        """ Calculate acceleration.

        Parameters
        ----------
        thresh : float
            optional argument to set a minimum

        Returns
        -------
        time : ndarray
        acceleration : ndarray
        """
        t, speed = self.getSpeed(vec=vec, laps=laps)

        diffs = numpy.diff(speed, axis=0)
        diffsTimes = numpy.diff(t)
        acceleration = diffs/diffsTimes   # [space units/s^2]

        if abs and not vec:
            acceleration = numpy.abs(acceleration)

        acceleration_dummy = []
        reducedTimes = []

        if thresh and not vec:
            for i, s in enumerate(acceleration):
                if numpy.abs(s) >= thresh:
                    acceleration_dummy.append(s)
                    reducedTimes.append(self.times[i])
        else:
            acceleration_dummy = acceleration
            reducedTimes = self.times[:-2]
        return numpy.array(reducedTimes), numpy.array(acceleration_dummy)


    def getJerk(self, thresh=None, abs=False, vec=False, laps=False):
        """ Calculate jerk, third derivative.

        Parameters
        ----------
        thresh : float, optional
            Argument to set a minimum.
        abs : bool, optional
        vec : bool, optional
        laps : bool, optional

        Returns
        -------
        time : ndarray
        jerk : ndarray
        """

        t, acceleration = self.getAcceleration(vec=vec, laps=laps)

        diffs = numpy.diff(acceleration, axis=0)[:,0:2]
        diffsTimes = numpy.diff(acceleration)
        jerk = diffs/diffsTimes   # [space units/s^3]

        if abs and not vec:
            jerk = numpy.abs(jerk)

        jerk_dummy = []
        reducedTimes = []

        if thresh and not vec:
            for i, s in enumerate(jerk):
                if numpy.abs(s) >= thresh:
                    jerk_dummy.append(s)
                    reducedTimes.append(self.times[i])
        else:
            jerk_dummy = jerk
            reducedTimes = self.times[:-3]

        return numpy.array(reducedTimes), numpy.array(jerk_dummy)


    def getHeadDirection(self):
        """ Calculates approximate head or gaze direction from trajectory.

        Returns
        -------
        self.headDirections : ndarray
        """

        diffs = numpy.diff(self.places, axis=0)
        lengths = numpy.sqrt(diffs[:,0]**2+diffs[:,1]**2)
        angles = numpy.arccos(diffs[:,0]/lengths)*numpy.sign(diffs[:,0])

        indicesWithDivideByZero = numpy.nonzero(numpy.isnan(angles))[0]  # there will be NaNs

        # iterate through the angles and override NaN entries with the last real angle
        while indicesWithDivideByZero.shape[0]>0:
            angles[indicesWithDivideByZero] = angles[indicesWithDivideByZero-1]
            indicesWithDivideByZero = numpy.nonzero(numpy.isnan(angles))[0]

        self.headDirections = angles

        return self.headDirections


    def getIndexFromTime(self, time):
        """ Return index corresponding to the given time.

        Parameters
        ----------
        time : int or array_like

        Returns
        -------
        index : int or ndarray
            Index or array of indices corresponding to time.
        """

        if not isinstance(time, list) and not isinstance(time, numpy.ndarray):
            # if necessary convert time value to list, i.e., array
            time = [time]

        index = []
        for t in time:
            # index.append(signale.findNearest(self.times, t)[0])
            # changed KT, 28.08.2014
            ind = numpy.searchsorted(self.times, t)
            if ind >= self.times.size:
                ind -= 1
            index.append(ind)

        if index.__len__() == 1:          # convert index to int in case of one entry only
            index = index[0]
        else:
            index = numpy.array(index)

        return index

    def getPlaceFromTime(self, time, interp=False):
        """ Return place array corresponding to the given time array.

        Parameters
        ----------
        time : float
        interp : bool, optional
            Defaults to False.

        Returns
        -------
        place : ndarray
        """

        if not isinstance(time, list) and not isinstance(time, numpy.ndarray):
            # if necessary convert time value to list, i.e., array
            time = [time]

        place = []
        if interp:
            #print '-------------------infoooooooo:'
            #(index1, t1) = signale.tools.findNearest_smaller_or_equal(self.times, time)
            index1 = numpy.searchsorted(self.times, time, side='left')-1    # get index of the time point that is just smaller than time
            #
            t1 = self.times[index1]
            index2 = index1+1

            if numpy.isnan(self.times[index2]):
                print 'nan index is: ', index2
                print 'nan time should be nan: ', self.times[index2]
                if time - t1 == 0:
                    return numpy.array(self.places[index1])
                else:
                    logger.warning('At time '+str(time)+' there was a spike after the end of a trajectory piece !')
                    return None
            else:
                t2 = self.times[index2]
                tpercent = (time - t1) / (t2 - t1)

                place1 = self.places[index1]
                place2 = self.places[index2]

                #print 'places before and after spike time: ', place1, place2
                place = tpercent * (place2 - place1) + place1

        else:
            index = self.getIndexFromTime(time) # get index of the time point that is just smaller than time
            place.extend(self.places[[index]])

        return numpy.array(place)


    def getHeadDirectionFromTime(self, time, interp=False):
        """ Return head direction array corresponding to the given time array.

        Parameters
        ----------
        time : float
        interp : bool, optional
            Defaults to False.

        Returns
        -------
        headDirection : ndarray
        """

        if not isinstance(time, list) and not isinstance(time, numpy.ndarray):
            # if necessary convert time value to list, i.e., array
            time=[time]

        headDirection = []
        if interp:
            index1=numpy.searchsorted(self.times, time, side='left')-1    # get index of the time point that is just smaller than time
            index2 = index1+1               # get index of the time point that is just bigger than time

            t1 = self.times[index1]
            t2 = self.times[index2]
            tpercent = (time - t1) / (t2 - t1)

            hd = self.getHeadDirection()
            hd1 = hd[index1]
            if index2 >= hd.size:
                index2 -= 1
            hd2 = hd[index2]
            headDirection = tpercent * numpy.absolute((numpy.exp(hd2*1j) - numpy.exp(hd1*1j))) + hd1
            #headDirection = tpercent * (hd2 - hd1) + hd1
        else:
            index = self.getIndexFromTime(time) # get index of the time point that is just smaller than time
            headDirection.append(self.getHeadDirection()[index])

        return numpy.array(headDirection)

    # def getRunningTraj(self, threshspeed=None, window_len=100):
    #     """ Get times and places, where the animal was running.
    #
    #     Parameters
    #     ----------
    #     threshspeed : float, optional
    #         Threshold for running speed. If provided self.threshspeed is overwritten.
    #         If not provided self.threshspeed is taken.
    #     window_len : int, optional
    #         Window length for smoothing. Default 51. NOTE: not implemented jet.
    #     """
    #
    #     if not threshspeed:
    #         threshspeed = self.threshspeed
    #     if threshspeed != self.threshspeed:
    #         print 'NOTE: Reset threshspeed from', self.threshspeed, 'to',\
    #             threshspeed, self.spaceUnit+'/'+self.timeUnit
    #         self.threshspeed = threshspeed
    #
    #     print "calculating trajectory with running speed >=", self.threshspeed
    #
    #     inclHeadDir = True
    #     if not hasattr(self , 'headDirections'):
    #         print 'WARNING: Calculating without head directions! If head directions are needed \
    #             calculate them first!'
    #         inclHeadDir = False
    #
    #     # get running speed and smooth it a bit
    #     speed_dummy = self.getSpeed()[1]
    #     # speed_dummy = signale.smooth(self.getSpeed()[1], window_len=window_len, window='hanning')
    #
    #     indices = numpy.where(speed_dummy >= self.threshspeed)[0]
    #     indices += 1                            # shift indices by one index
    #     if indices.size and indices[-1] >= speed_dummy.size:
    #         indices = indices[:-1]
    #     indices = numpy.append([0], indices)    # add entry at index zero to indices
    #     self.run_times = self.times[indices]
    #     self.run_places = self.places[indices]
    #     if inclHeadDir:
    #         self.run_headDirections = self.headDirections[indices]

    def get_runTraj(self, threshspeed=None, window_len=100):
        """ Get times and places, where the animal was running.

        Parameters
        ----------
        threshspeed : float, optional
            Threshold for running speed. If provided self.threshspeed is overwritten.
            If not provided self.threshspeed is taken.
        window_len : int, optional
            Window length for smoothing. Default 51. NOTE: not implemented jet.
        """

        if threshspeed is not None and threshspeed != self.threshspeed:
            print 'NOTE: Reset threshspeed from', self.threshspeed, 'to',\
                threshspeed, self.spaceUnit+'/'+self.timeUnit
            self.threshspeed = threshspeed

        print "Calculating trajectory with running speed >=", self.threshspeed

        inclHeadDir = True
        if not hasattr(self, 'headDirections'):
            print 'WARNING: Calculating without head directions! If head directions are needed \
                calculate them first!'
            inclHeadDir = False

        # get running speed and smooth it a bit
        speed_dummy = self.getSpeed()[1]
        # speed_dummy = signale.smooth(self.getSpeed()[1], window_len=window_len, window='hanning')

        indices = numpy.where(speed_dummy >= self.threshspeed)[0]
        indices += 1                            # shift indices by one index
        if indices.size and indices[-1] >= speed_dummy.size:
            indices = indices[:-1]
        indices = numpy.append([0], indices)    # add entry at index zero to indices

        import copy
        run_traj = copy.deepcopy(self)

        run_traj.times = run_traj.times[indices]
        run_traj.places = run_traj.places[indices]
        if inclHeadDir:
            run_traj.headDirections = run_traj.headDirections[indices]
        run_traj._trajectory__recalc_startstop()

        self.run_traj = run_traj

        return run_traj

    def getLaps(self):
        """Get the number of laps in the trajectory.

        A lap is defined as the path/time between leaving a reward
        area and entering the next. NOTE, that the next reward area
        might be the same or a different reward area.

        Returns
        -------
        laps : int
            Number of laps.
        lapTimes : list
            List of start times of each lap.
        """

        centers = self.rewardsPos
        radius = self.rewardsArea[0]
        places = self.places[:, 0:2]

        i = 0
        laps = 0
        inLap = False
        lapTimes = []

        while i < places.shape[0]:

            diffs = places[i] - centers
            distances=[]
            for item in diffs:
                distances.append(numpy.sqrt(item[0]**2 + item[1]**2))
            minDistance=numpy.array(distances).min()

            if minDistance > radius and not inLap:
                laps += 1
                lapStart = self.times[i]
                inLap = True
            elif minDistance < radius and inLap:
                lapTimes.append([lapStart, self.times[i-1]])
                inLap = False

            i += 1

        return laps, lapTimes


    def getTrajDimensions(self):
        """Get dimensions/bounds of trajectory.

        Detects coordinates with maximum/minimum x and y values
        and using this calculates the 'width' of the trajectory.

        Returns
        -------
        xWidth : float
            Width in x direction.
        yWidth : float
            Width in y direction.
        """

        if len(self.places):

            #self.orient()
            #mini = self.places.argmin(0)
            mini = numpy.nanargmin(self.places, 0)
            #maxi = self.places.argmax(0)
            maxi = numpy.nanargmax(self.places, 0)

            xMin = self.places[mini[0], 0]
            xMax = self.places[maxi[0], 0]
            yMin = self.places[mini[1], 1]
            yMax = self.places[maxi[1], 1]

            self.xWidth = xMax-xMin
            self.yWidth = yMax-yMin

            self.xlim = numpy.array([xMin, xMax])
            self.ylim = numpy.array([yMin, yMax])
        else:
            self.xWidth = 0
            self.yWidth = 0

            self.xlim = numpy.array([0, 0])
            self.ylim = numpy.array([0, 0])


        return self.xWidth, self.yWidth


    def turn(self, yaw=0.):
        """ Turn trajectory.

        Parameters
        ----------
        yaw : float, optional
            Yaw angle to turn by. Defaults to 0 degrees.
        """

        slope = numpy.tan(numpy.deg2rad(yaw))
        vecTrack = numpy.array([1 , slope])
        normalVecTrack = numpy.array([-vecTrack[1], vecTrack[0]])

        # do it for the places
        tmp = []
        for i, p in enumerate(self.places):
            tmp = p[0:2].copy()
            self.places[i, 0] = numpy.dot(tmp, vecTrack)/numpy.sqrt(numpy.dot(vecTrack,vecTrack))
            self.places[i, 1] = numpy.dot(tmp, normalVecTrack)/numpy.sqrt(numpy.dot(vecTrack,vecTrack))

        # do it again for the rewardsPos
        if hasattr(self, 'rewardsPos'):
            tmp = []
            for i, p in enumerate(self.rewardsPos):
                tmp = p[0:2].copy()
                self.rewardsPos[i, 0] = numpy.dot(tmp, vecTrack)/numpy.sqrt(numpy.dot(vecTrack,vecTrack))
                self.rewardsPos[i, 1] = numpy.dot(tmp, normalVecTrack)/numpy.sqrt(numpy.dot(vecTrack,vecTrack))

        self.getTrajDimensions()
        self.euler[0] += yaw


    def plot(self, fig=None, ax=None, offset=2, language='e', chic=False, marker=None):

        if not fig:
            fig = pl.figure(figsize=(8, 6))
        if not ax:
            ax = fig.add_axes([.2, .15, .81, .75])
        axcb = None

        if not marker:
            line_segments = LineCollection([[x,self.places[i+1+offset,0:2]] \
                                for i,x in enumerate(self.places[offset:-(1+offset),0:2])],\
                                linestyles = 'solid', linewidths=mpl.rcParams['lines.linewidth']/2.)
            line_segments.set_array(self.times)
            ax.add_collection(line_segments)
        else:
            ax.plot(self.places[offset:, 0], self.places[offset:, 1], marker)

        ax.plot(self.places[0+offset,0], self.places[0+offset,1], 'o')      # start point
        ax.plot(self.places[-2,0], self.places[-2,1], 'd')                  # end point


        # huebsch machen
        custom_plot.huebschMachen(ax)
        if not chic:
            ax.set_title(self.mazeType, fontsize=custom_plot.fontsize-4)
        if language=='d':       # in german
            ax.set_xlabel('x-Position ('+self.spaceUnit+')')
            ax.set_ylabel('y-Position ('+self.spaceUnit+')')
        else:               # in english
            ax.set_xlabel('x position ('+self.spaceUnit+')')
            ax.set_ylabel('y position ('+self.spaceUnit+')')
        dx = numpy.round((int(self.t_stop)/60*60-int(self.t_start)/60*60)/4.)
        if not dx:
            dx = 60
        xticks = numpy.arange(round(self.t_start), round(self.t_stop)+1, dx)

        # colorbar
        if not marker:
            axcb = fig.colorbar(line_segments)
            axcb.set_ticks(xticks)
            if language=='d':       # in german
                axcb.set_label('Zeit ('+self.timeUnit+')')
            else:                   # in english
                axcb.set_label('Time ('+self.timeUnit+')')

        if not fig:
            pl.show()

        return fig, ax, axcb


    def plotCumPlaces(self, fig=None, ax=None, text=''):

        if not fig:
            fig = pl.figure()
        if not ax:
            ax = fig.add_subplot(111)

        cumPlaces=self.cumPlaces()
        ax.plot(self.times, cumPlaces)

        ax.set_xlabel('Time ('+self.timeUnit+')')
        ax.set_ylabel('Path length ('+self.spaceUnit+')')
        ax.text(self.times.max(), cumPlaces.max(), text)

        pl.show()

        return fig, ax

    def plotSpeed(self, thresh=None, fig=None, ax=None):

        if not fig:
            fig = pl.figure()
        if not ax:
            ax = fig.add_subplot(111)
        #ax = fig.add_subplot(211)

        reducedTimes, speed = self.getSpeed(thresh=thresh)
        ax.plot(reducedTimes, speed)

        ax.set_xlabel('Time ('+self.timeUnit+')')
        ax.set_ylabel('Speed ('+self.spaceUnit+'/'+self.timeUnit+')')
        ax.set_xlim(reducedTimes[0], reducedTimes[-1])

        ##--##

##        ax = fig.add_subplot(212)
##
##        reducedTimes = self.getSpeed(thresh=thresh)[0]
##        #reducedTimes = signale.smooth(self.getSpeed(thresh=thresh)[0], window_len=11, window='hanning')
##        speed=signale.smooth(self.getSpeed(thresh=thresh)[1], window_len=11, window='hanning')
##
##        ax.plot(reducedTimes, speed)
##
##        ax.set_xlabel('Time ('+self.timeUnit+')')
##        ax.set_ylabel('Smoothed-Speed ('+self.spaceUnit+'/'+self.timeUnit+')')
##        ax.set_xlim(reducedTimes[0], reducedTimes[-1])
        pl.show()

        return fig, ax


    def plotSpeedvsPlace(self, thresh=None, fig=None, ax=None):

        if not fig:
            fig = pl.figure()
        if not ax:
            ax = fig.add_subplot(111)

        time, speed = self.getSpeed(thresh)
        places = self.getPlaceFromTime(time)

        offset = 2
        line_segments = LineCollection([[x, places[i+1+offset,0:2]] for i, x in enumerate(places[offset:-2, 0:2])],
                                       linestyles='solid')
        speed = numpy.minimum(speed, speed.mean()+3.*speed.std())  # cut away outliers for plotting
        line_segments.set_array(speed)
        ax.add_collection(line_segments)

        fig = pl.gcf()
        axcb = fig.colorbar(line_segments)
        ax.plot(places[0+offset,0], places[0+offset,1],'o')      # start point
        ax.plot(places[-2,0], places[-2,1],'d')      # end point

        axcb.set_label('Speed ('+self.spaceUnit+'/'+self.timeUnit+')')
        ax.set_title(self.mazeType)
        ax.set_xlabel('x position ('+self.spaceUnit+')')
        ax.set_ylabel('y position ('+self.spaceUnit+')')
        pl.show()

        return fig, ax


    def purge(self, numItems):
        """ Cut away items.

        Parameters
        ----------
        numItems : int
        """
        self.times = self.times[numItems:]
        self.places = self.places[numItems:]


    def removeValues(self, value=0.):
        """ Remove values due to artifacts (e.g., signal loss).

        Parameters
        ----------
        values : float, optional
            defaults to 0.
        """
        #for i in numpy.where(self.places[:,:2]==0)[0]:
            #self.places[i] = numpy.nan

        self.times = numpy.delete(self.times, numpy.where(self.places[:, :2] == value)[0], 0)
        self.places = numpy.delete(self.places, numpy.where(self.places[:, :2] == value)[0], 0)


    def recalcTimesAxis(self, places, times, time_before=8):
        """ Recalculate times.

        Find (average) smallest time offset to places in trajectory and
        recalculate the time axis of the trajectory accordingly.

        Parameters
        ----------
        places :
        times :
        time_before : int, optional
            default 8 (empirically tested)
        """
        if times.__len__():
            time_diffs = numpy.array([])
            for i, t in enumerate(times):
                index1 = self.getIndexFromTime(t)+1
                index2 = self.getIndexFromTime(t-time_before)
                index_min = signale.findNearest_vec(self.places[index2:index1], places[i])[0] + index2
                time_diffs = numpy.append(time_diffs, t-self.times[index_min])
            self.time_offset(time_diffs.mean())
            #print "time differences:", time_diffs
            print "=> shifted time axis by" , time_diffs.mean(), ", std is" , time_diffs.std()
            self.__recalc_startstop()

    def smooth(self, time_window=.5):
        """ Smooth the trajectory."""

        self.places_orig = numpy.copy(self.places)      # save orignial data
        x = signale.smooth(self.places[:, 0], window_len=numpy.fix(time_window/self.dt))
        y = signale.smooth(self.places[:, 1], window_len=numpy.fix(time_window/self.dt))
        self.places[:, 0] = x
        self.places[:, 1] = y

    def time_slice(self, t0=None, t1=None, traj_type=None, meta=None):
        """ Return a trajectory object sliced by time.

        Parameters
        ----------
        t0 : float
            Start time for slice.
        t1 : float
            End time for slice.

        Returns
        -------
        sliced_traj : trajectory object
            Sliced trajectory.
        """
        if not t0:
            t0 = self.times[0]

        if not t1:
            t1 = self.times[-1]

        # get indices corresponding to t0 and t1, respectively
        index0 = self.getIndexFromTime(t0)
        index1 = self.getIndexFromTime(t1) - 1        # getIndexFromTime gives index for following index, hence get previous here

        if index0 == index1:
            print 'WARNING: Slice', t0, t1, ' not possible! Calculated indexes by trajectory.py are identical (self.times index = ', index0, ')'
            return 0

        sliced_traj_array = numpy.hstack([self.times.reshape(self.times.size, 1)[index0:index1, :],\
                                          self.places[index0:index1, :]])
        if not traj_type:
            traj_type = type(self)
        if not meta:
            meta = self.meta

        sliced_traj = traj_type(sliced_traj_array, meta)

        sliced_traj.threshspeed = self.threshspeed

        return sliced_traj

    def time_offset(self, offset=0.):
        self.times += offset
        self.__recalc_startstop()

    def concatenate(self, Traj, t_start=None, t_stop=None, traj_type=None, meta=None, nan=True):
        """ Return new trajectory object obtained by concatenating given Trajectory

        Parameters
        ----------
        Traj : trajectory object
        t_start : float
            Start time for new trajectory object.
        t_stop : float
            End time for new trajectory object.

        Returns
        -------
        concatenated_traj : trajectory object
            Concatenated trajectory.
        """
        if not t_start:
            t_start = self.times[0]

        if not t_stop:
            t_stop = Traj.times[-1]

        #################### concatenate times and places and stack back together to concatenated_traj_array
        # insert numpy.nan between the two traj parts, so that there will be a gap instead of a jump in the trajectory!

        if nan:
            concatenated_times_array = numpy.concatenate((numpy.append(self.times, numpy.nan), Traj.times))
            concatenated_places_array = numpy.vstack((self.places, numpy.ones(3)*numpy.nan, Traj.places))
        else:
            concatenated_times_array = numpy.concatenate((self.times, Traj.times))
            concatenated_places_array = numpy.vstack((self.places, Traj.places))

        concatenated_traj_array = numpy.hstack([concatenated_times_array.reshape(concatenated_times_array.size, 1),
                                                concatenated_places_array])

        if not traj_type:
            traj_type = type(self)
        if not meta:
            meta = self.meta

        concatenated_traj = traj_type(concatenated_traj_array, meta)

        concatenated_traj.t_start = t_start
        concatenated_traj.t_stop = t_stop

        if self.threshspeed == Traj.threshspeed:
            concatenated_traj.threshspeed = self.threshspeed
        else:
            concatenated_traj.threshspeed = 0.0

        return concatenated_traj


class linearMazeTrajectory(trajectory):

    def __init__(self, places, meta={}, initialized=False):
        if not initialized:
            trajectory.__init__(self, places, meta)     # initialized=True if trajectory.__init__() was already called
        self.oriented = False

        if not meta.has_key('expType'):
            print 'NOTE: No experiment type provided!'
            print '  => put some values as if experiment was done in real environment'
            meta['trackLength'] = 1.5
            meta['expType'] = 'real'
            meta['yaw'] = 0

        self.yaw = 0
        if meta.has_key('yaw'):
            self.yaw = meta['yaw'] % 360
            self.euler[0] += self.yaw
            self.euler %= 360

        self.orient()      # rotate trajectory to remove yaw
        if meta.has_key('trackLength') and meta.has_key('expType') and meta['expType'] == 'real':
            #self.orient()
            self.removeValues(0.)                               # remove lines to zero
            self.getTrajDimensions()
            self.trackLength = self.xWidth
            self.trackWidth = self.yWidth
            for i, p in enumerate(self.places):
                self.places[i, 0:2] -= [self.xlim[0], self.ylim[0]]       # shift to get smallest place to 0,0
                self.places[i, 0:2] /= self.xWidth              # normalize it to trackLength
                self.places[i, 0:2] *= meta['trackLength']      # scale it to trackLength

        self.getTrajDimensions()
        self.trackLength = self.xWidth
        self.trackWidth = self.yWidth


    def getComponentsFromTime(self, time, interp=0):
        """Returns the x and y components of trajectory corresponding to the given time.

        Parameters
        ----------
        time :  float
            Time point to return x and y components for.
        interp : bool, optional
            Interpolate times. ToDo!

        Returns
        -------
        components : ndarray
        """

        try: self.components
        except AttributeError:
            self.components = numpy.array([self.getXComponents(), self.getYComponents()]).T

        components = numpy.array([0, 0])
        if interp:
            index1 = numpy.searchsorted(self.times, time, side='right')-1  # get index of the time point that is just smaller than time
            index2 = numpy.searchsorted(self.times, time, side='right')    # get index of the time point that is just bigger than time

            print 'NOTE: to do'
        else:
            index = numpy.searchsorted(self.times, time, side='right')-1    # get index of the time point that is just smaller than time
            if abs(self.times[index]-time) > abs(self.times[index+1]-time):
                index += 1
            components = self.components[index]

        return components

    def getLeftAndRightwardRuns(self):
        """ Calculates the trajectories split by running direction.

        Returns
        -------
        rightward_traj, leftward_traj : linear maze trajectory object
            The trajectories in rightward and leftward direction. It is also dumped into class attributes named
            self.rightward_traj and self.leftward_traj, respectively.
        """

        # x-Richtung bestimmen
        xdir = numpy.diff(self.places[:, 0], axis=0)
        # xdir = numpy.diff(signale.smooth(places[:, 0], window_len=int(5./self.dt), window='hanning'), axis=0)


        import copy

        # rightward runs
        indices = numpy.where(xdir < 0)[0] + 1
        rightward_traj = copy.deepcopy(self)

        rightward_traj.times[indices] = numpy.nan
        rightward_traj.places[indices] = numpy.nan
        # rightward_traj.times = rightward_traj.times[indices]
        # rightward_traj.places = rightward_traj.places[indices]
        # if inclHeadDir:
        #     rightward_traj.headDirections = rightward_traj.headDirections[indices]
        # rightward_traj._trajectory__recalc_startstop()

        # leftward runs
        indices = numpy.where(xdir > 0)[0] + 1
        leftward_traj = copy.deepcopy(self)

        leftward_traj.times[indices] = numpy.nan
        leftward_traj.places[indices] = numpy.nan
        # leftward_traj.times = leftward_traj.times[indices]
        # leftward_traj.places = leftward_traj.places[indices]
        # if inclHeadDir:
        #     leftward_traj.headDirections = leftward_traj.headDirections[indices]
        # leftward_traj._trajectory__recalc_startstop()

        self.rightward_traj = rightward_traj
        self.leftward_traj = leftward_traj

        return rightward_traj, leftward_traj



    def getXComponents(self):
        xComp = numpy.array([])
        if hasattr(self, 'euler'):
            slope = -numpy.tan(numpy.deg2rad(self.euler[0]))
            vecTrack = numpy.array([1 , slope])
            for p in self.places:
                xComp = numpy.append(xComp, numpy.dot(p[0:2], vecTrack))
        else:
            vecTrack = self.rewardsPos[1]-self.rewardsPos[0]
            for p in self.places:
                xComp = numpy.append(xComp, numpy.dot(p[0:2]-self.rewardsPos[0], vecTrack))

        return xComp/numpy.sqrt(numpy.dot(vecTrack, vecTrack))


    def getYComponents(self):
        yComp = numpy.array([])
        if hasattr(self, 'euler'):
            slope = -numpy.tan(numpy.deg2rad(self.euler[0]))
            vecTrack = numpy.array([1 , slope])
            normalVecTrack = numpy.array([-vecTrack[1], vecTrack[0]])
            for p in self.places:
                yComp = numpy.append(yComp, numpy.dot(p[0:2], normalVecTrack))
        else:
            vecTrack = self.rewardsPos[1]-self.rewardsPos[0]
            normalVecTrack = numpy.array([-vecTrack[1], vecTrack[0]])
            for p in self.places:
                yComp = numpy.append(yComp, numpy.dot(p[0:2]-self.rewardsPos[0], normalVecTrack))

        return yComp/numpy.sqrt(numpy.dot(vecTrack, vecTrack))


    def orient(self):
        """
        Orients the trajectory to yaw = 0 deg by projecting it
        via a dot product with the slope vector.
        """
        if not self.oriented:
            self.turn(-self.euler[0])

            # rotate to positive axes, if necessary
            if self.yaw > 90 and self.yaw <= 270:
                self.places[:, 0:2] *= -1

            self.getTrajDimensions()
            self.trackWidth = self.yWidth
            self.oriented = True
        else:
            print "NOTE: Track already oriented."


    def plot(self, fig=None, ax=None, offset=2, language='e', chic=False):

        fig, ax, axcb = trajectory.plot(self, fig, ax, offset, language, chic)

        # adjust axes
        pos = [.15, .65, .65, .3]
        ax.set_position(pos)
        poscb = list(axcb.ax.get_position().bounds)
        poscb[0] = .825
        poscb[1] = pos[1]
        poscb[3] = pos[3]
        axcb.ax.set_position(poscb)


        # huebsch machen
        xoffset = self.xWidth*.1
        yoffset = self.yWidth

        xmin = -xoffset
        xmax = self.xWidth + xoffset
        ymin = self.ylim[0]-yoffset
        ymax = self.ylim[1] + yoffset

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)


        # show yaw
        if hasattr(self, 'yaw'):
            ax_inset = fig.add_axes([.73, .85, .07, .07], polar=True)
            x = numpy.deg2rad(self.yaw)
            patch = ax_inset.fill([x - numpy.pi, x - .85*numpy.pi, x, x + .85*numpy.pi, x + numpy.pi], [.75, 1, 1, 1, .75], facecolor='b', edgecolor='b')
            #custom_plot.drop_shadow_patches(ax_inset, patch[0])
            ax_inset.set_yticks([])
            ax_inset.set_xticks([])
            ax_inset.set_xticks(numpy.arange(0, 1, .125)*2*numpy.pi)
            ax_inset.set_xticklabels([])
            #ax_inset.set_xticklabels(numpy.int_(numpy.arange(0, 1, .25)*360), fontsize=custom_plot.fontsize/2)
            ax_inset.spines['polar'].set_linestyle('dotted')
            ax_inset.spines['polar'].set_linewidth(.5)

        # put additional axes
        pos1 = list(pos)
        pos1[1] = .11
        pos1[2] = .32
        pos1[3] = .35
        ax1 = fig.add_axes(pos1)
        pos2 = list(pos1)
        pos2[0] += pos2[2]+.16
        ax2 = fig.add_axes(pos2)

        xComponents = self.getXComponents()
        ax1.plot(self.times, xComponents, 'k-')

        if chic:
            yComponents = self.cumPlaces()
        else:
            yComponents = self.getYComponents()
        ax2.plot(self.times, yComponents, 'k-')

        # huebsch machen
        custom_plot.huebschMachen(ax1)
        custom_plot.huebschMachen(ax2)
        if language=='d':       # in german
            ax1.set_xlabel('Zeit ('+self.timeUnit+')')
            ax2.set_xlabel('Zeit ('+self.timeUnit+')')
            ax1.set_ylabel('x-Position ('+self.spaceUnit+')')
            ax2.set_ylabel('y-Position ('+self.spaceUnit+')')
        else:               # in english
            ax1.set_xlabel('Time ('+self.timeUnit+')')
            ax2.set_xlabel('Time ('+self.timeUnit+')')
            ax1.set_ylabel('x position ('+self.spaceUnit+')')
            if chic:
                ax2.set_ylabel('Path length ('+self.spaceUnit+')')
            else:
                ax2.set_ylabel('y position ('+self.spaceUnit+')')

        minY = xComponents.min()
        maxY = xComponents.max()
        dY = 1.
        ax1.set_yticks(numpy.arange(round(minY), round(maxY)+dY, dY))
        ax1.set_ylim(minY-dY/10, maxY+dY/10)
        dx = numpy.round((int(self.t_stop)/60*60-int(self.t_start)/60*60)/4.)
        if not dx:
            dx = 60
        xticks = numpy.arange(round(self.times[0]), round(self.times[-1])+1, dx)
        ax1.set_xticks(xticks)
        ax1.set_xlim(self.times[0], self.times[-1])

        minY = yComponents.min()
        maxY = yComponents.max()
        dY = .1
        if chic:
            dY = int((maxY-minY)/5)
        ax2.set_yticks(numpy.arange(round(minY, 1), round(maxY, 1)+dY, dY))
        ax2.set_ylim(minY-dY/10, maxY+dY/10)
        ax2.set_xticks(xticks)
        ax2.set_xlim(self.times[0], self.times[-1])

        # customize plot further?
        if chic:
            custom_plot.allOff(ax)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
            pos1[3] = .45
            pos2[3] = .45
            ax1.set_position(pos1)
            ax2.set_position(pos2)

            # add scale bars
            custom_plot.add_scalebar(ax, matchx=True, matchy=True, labelExtra=' m',\
                loc=3, borderpad=-1., sep=5)

        return fig, ax, axcb


    def plotComponents(self, plotInOne=0, plotAgainstEachOther=0, color='k', lineStyle='-'):
        """Plot x and y components of the linearMaze trajectory.

        Parameters
        ----------
        plotInOne : bool, optional
            Plot x and y components into one graph [default 0].
        plotAgainstEachOther : bool, optional
            Plot x and y components against each other [default 0].
        color :
            Color of the plot [default 'k'].
        lineStyle : str, optional
            Line style of the plot [default '-'].

        Returns: nothing
        """

        ax = None
        ax2 = None

        if not plotInOne:
            fig = pl.figure()
            ax = fig.add_subplot(111)
            ax.set_position([.15, .1, .8, .8])
            ax.plot(self.times, self.getXComponents(), color+lineStyle)
            ax.set_xlabel('Time ('+self.timeUnit+')')
            ax.set_ylabel('x position ('+self.spaceUnit+')')
            ax.set_xlim(self.times[0], self.times[-1])

            fig = pl.figure()
            ax = fig.add_subplot(111)
            ax.set_position([.15, .1, .8, .8])
            ax.plot(self.times, self.getYComponents(), color+lineStyle)
            ax.set_xlabel('Time ('+self.timeUnit+')]')
            ax.set_ylabel('y position ('+self.spaceUnit+')')
            ax.set_xlim(self.times[0], self.times[-1])
        else:
            fig = pl.figure()
            ax = fig.add_subplot(111)
            ax.set_position([.15, .1, .7, .8])
            ax2 = ax.twinx()

            ax.plot(self.times, self.getXComponents(), 'k'+lineStyle)
            ax.set_xlabel('Time ('+self.timeUnit+')')
            ax.set_ylabel('x position ('+self.spaceUnit+')', color='k')
            ax.set_xlim(self.times[0], self.times[-1])
            for tl in ax.get_yticklabels():
                tl.set_color('k')

            ax2.plot(self.times, self.getYComponents(), 'k'+lineStyle, alpha=.5)
            ax2.set_ylabel('y position ('+self.spaceUnit+')', color=[.5, .5, .5])
            for tl in ax2.get_yticklabels():
                tl.set_color([.5, .5, .5])

        if plotAgainstEachOther:
            fig = pl.figure()
            ax = fig.add_subplot(111)
            ax.set_position([.15, .1, .8, .8])
            ax.plot(self.getXComponents(), self.getYComponents(), color+lineStyle)
            ax.set_xlabel('x position ('+self.spaceUnit+')')
            ax.set_ylabel('y position ('+self.spaceUnit+')')

        pl.show()

        return ax, ax2

    def plotSpeedvsComponents(self, thresh=None, avg=False, fig=None, ax=None, text='', color='k', lineStyle='.'):

        if not fig:
            fig = pl.figure()
        if not ax:
            ax = fig.add_subplot(111)

        time, speed = self.getSpeed(thresh)
        x = []
        y = []
        for t in time:
            dummy = self.getComponentsFromTime(t)
            x.append(dummy[0].tolist())
            y.append(dummy[1].tolist())
        x = numpy.array(x)
        y = numpy.array(y)

        if avg:
            bins = numpy.linspace(0, self.trackLength, 15)
            inds = numpy.digitize(x, bins)
            set_inds = numpy.unique(inds)
            average = numpy.zeros(set_inds.size)
            err = numpy.zeros(set_inds.size)
            for i, s in enumerate(set_inds):
                average[i] = speed[inds==s].mean()
                err[i] = speed[inds==s].std() / numpy.sqrt(numpy.sum(inds==s))
            custom_plot.avgPlot(ax, bins, average) # , err=err
            ax.text(bins[average.argmax()], average.max(), text)
        else:
            ax.plot(x, speed, color+lineStyle)
        ax.set_xlabel('x position ('+self.spaceUnit+')')
        ax.set_ylabel('Speed ('+self.spaceUnit+'/'+self.timeUnit+')')
        ax.set_xlim(x.min(), x.max())

        if not avg:
            fig = pl.figure()
            ax = fig.add_subplot(111)
            ax.plot(y, speed, color+lineStyle)
            ax.set_xlabel('y position ('+self.spaceUnit+')')
            ax.set_ylabel('Speed ('+self.spaceUnit+'/'+self.timeUnit+')')
            ax.set_xlim(y.min(), y.max())

        pl.show()

        return fig, ax


class paramsTrajectory(object):
    """ Class for the parameters of an experiment.
    """

    def __init__(self, times=[], parameters={}, meta={}):

        # initialize meta data
        self.meta = meta
        self.t_start = 0.0
        self.mazeType = ''
        self.timeUnit = 's'

        # get real values if parsable from file
        if meta.has_key('dt'):
            self.dt=meta['dt']                  # dt [s]
        if meta.has_key('t_start'):
            self.t_start=meta['t_start']        # t_start [s]
        if meta.has_key('time'):
            self.time=meta['time']              # time and date, when
                                                # recording was started,
                                                # just to provide some time stamp
        if meta.has_key('mazetype'):
            self.mazeType = meta['mazetype']

        self.times = times
        self.parameters = parameters


    def getParameter(self, key):
        """Get values for parameter.

        Parameters
        ----------
        key : str
            Key name of parameter in dictionary.

        Returns
        -------
        param_array : list
            List of values for the parameter from the parameters dictionary."""

        param_array = []
        for p in self.parameters:
            if key in p.keys():
                param_array.append(p[key])

        return param_array


    def plotParameter(self, key, fig=None, ax=None, smoothed=False):
        """Plot the values of a parameter.

        Parameters
        ----------
        key : str
            Key name of parameter in dictionary.
        fig : matplotlib figure, optional
        ax : matplotlib axes, optional
        smoothed : bool, optional
            Smooth the development of trials. Defaults to False.

        Returns
        -------
        fig : matplotlib figure
        ax : matplotlib axes
        """

        if not fig:
            fig = pl.figure(figsize=(8, 8))
        if not ax:
            pos = [0.125, 0.6, 0.5, 0.35]
            ax = fig.add_axes(pos)
            pos_hist = list(pos)
            pos_hist[0] += pos_hist[2] + .1
            pos_hist[2] = .25
            ax_hist = fig.add_axes(pos_hist)
            pos = list(pos)
            pos[2] = .75
            pos[1] -= pos[3] + .1
            ax2 = fig.add_axes(pos)

        param_array = numpy.array(self.getParameter(key))
        ax.plot(param_array, 'o-')
        print 'avg, std:', param_array.mean(), param_array.std()
        print 'chi-square:', scipy.stats.chisquare(numpy.histogram(param_array, numpy.unique(param_array))[0])
        ax_hist.hist(param_array, numpy.unique(param_array).size)
        if smoothed:
            smoothed_param_array = signale.smooth(param_array, smoothed, 'flat')
            ax.plot(smoothed_param_array)
            ax_hist.hist(smoothed_param_array, 10)
            print 'smoothed avg, std:', smoothed_param_array.mean(), smoothed_param_array.std()

        if self.times.size:
            ax2.plot(self.times, param_array, '.-')

        # huebsch machen
        for a in [ax, ax_hist, ax2]:
            custom_plot.huebschMachen(a)
        ax.set_xlabel('#')
        ax.set_ylabel(key)

        ax2.set_ylabel(key)
        ax2.set_xlabel('Times ('+self.timeUnit+')')

        return fig, ax

    
    def plotParameterDistribution(self, key, fig=None, ax=None, labelsize=None, showFigs=False, saveFig=False, saveFolder=''):
        """ Plots distributions for parameter.

        One of the parameter distribution and one of the transition distribution across trials.

        Parameters
        ----------
        key : str
            Keyword of parameter in dictionary.
        fig : matplotlib figure
        ax : matplotlib axes
        labelsize : int
            Font size of the label.

        Returns
        -------
        fig : matplotlib figure
        ax_top : matplotlib axes
        ax_bottom : matplotlib axes
        """
        
        self.param_list = list(self.getParameter(key))
        
        # creating a dictionary of parameter counts
        self.param_dict = dict((x, self.param_list.count(x)) for x in self.param_list)
    
        # sorting the parameter dictionary and storing its keys and values in variables
        param_sorted_dict = sorted([(param_key, param_value) for (param_key, param_value) in self.param_dict.items()])
        param_keys = []
        param_values = []
        for i in range(len(param_sorted_dict)):
            param_keys.append(param_sorted_dict[i][0])
            param_values.append(param_sorted_dict[i][1])
    
        # doubling the all parameters except the first and last one, to analyse parameter transitions
        doubled_params = [str(self.param_list[0])]
        for i in numpy.arange(1, len(self.param_list)-1):
            doubled_params.extend([str(self.param_list[i]), str(self.param_list[i])])
        doubled_params.append(str(self.param_list[len(self.param_list)-1]))
        param_transitions = [(', '.join(map(str, doubled_params[n:n+2]))).replace(",", r"$\rightarrow$")
                            for n in range(0, len(doubled_params), 2)]
        self.param_transitions_dict = dict((x, param_transitions.count(x)) for x in param_transitions)
    
        # plotting absolute and transition parameter distributions as histograms
        if not fig:
                fig = pl.figure(figsize=(8, 4))
        if ax:
            pos = ax.get_position()      # [left, bottom, width, height]
            ax_top = fig.add_axes([pos[0], pos[1]+(4*pos[3]/7), pos[2], 3*pos[3]/7])
            ax_bottom = fig.add_axes([pos[0], pos[1], pos[2], 3*pos[3]/7])
        else:
            pos = [.25, .21, .65, .625]    # [left, bottom, width, height]
            ax_top = fig.add_axes([pos[0], pos[1]+(4*pos[3]/7), pos[2], 3*pos[3]/7])
            ax_bottom = fig.add_axes([pos[0], pos[1], pos[2], 3*pos[3]/7])
    
        ax_top.bar(*zip(*zip(count(), param_values)))
        ax_top.set_title('Absolute and transition %s distribution' %key)
        ax_top.set_xticks((1*zip(*zip(count(0.4), param_keys)))[0])
        ax_top.set_xticklabels((1*zip(*zip(count(0.4), param_keys)))[1])
        if labelsize:
            for tick in ax_top.xaxis.get_major_ticks():
                tick.label.set_fontsize(labelsize)
            for tick in ax_top.yaxis.get_major_ticks():
                tick.label.set_fontsize(labelsize)
        ax_top.set_xlabel(key)
        ax_top.set_ylabel('Value count')

        ax_bottom.bar(*zip(*zip(count(), self.param_transitions_dict.values())))
        ax_bottom.set_xticks((1*zip(*zip(count(0.4), self.param_transitions_dict)))[0])
        ax_bottom.set_xticklabels((1*zip(*zip(count(0.4), self.param_transitions_dict)))[1])
        if labelsize:
            for tick in ax_bottom.xaxis.get_major_ticks():
                tick.label.set_fontsize(labelsize)
            for tick in ax_bottom.yaxis.get_major_ticks():
                tick.label.set_fontsize(labelsize)
        ax_bottom.set_xlabel('%s transition' %key)
        ax_bottom.set_ylabel('Value count')

        if saveFig:
            print 'Parameter Distribution plot saved to:', saveFolder+'Parameter_Distribution.png'
            fig.savefig(saveFolder+'Parameter_Distribution.png', format='png')
        if not showFigs:
                pl.ioff()
                pl.close(fig)
        else:
            pl.show()

        return fig, ax_top, ax_bottom


    def plotParam1vsParam2(self, key1, key2, fig=None, ax=None, refline=False):
        """Plot two parameters against each other.

        Parameters
        -----------
        key1 : str
            Name of one parameter.
        key2 : str
            Name of other parameter.
        fig : matplotlib figure, optional
        ax : matplotlib axes, optional
        refline : bool, optional
            Draw reference line.

        Returns
        -------
        (fig, ax), (fig1, ax1, ax1b) : two tuples
            Tuples with Figure and Axes handles.
        """

        if not fig:
            fig = pl.figure()
            fig1 = pl.figure()
        if not ax:
            ax = fig.add_subplot(111)
            ax1 = fig1.add_subplot(111)

        param_array1 = numpy.array(self.getParameter(key1))
        param_array2 = numpy.array(self.getParameter(key2))
        mini = min(param_array1.min(), param_array2.min())
        maxi = max(param_array1.max(), param_array2.max())
        if refline:
            ax.plot([mini, maxi], [mini, maxi], '--', color=numpy.ones(3)*.5, linewidth=1)
        ax.plot(param_array1, param_array2, 'bo', markerfacecolor='none', markeredgecolor='b', alpha=.25)
        ax1.plot(param_array1, 'bo-', linewidth=1)
        ax1b = ax1.twinx()
        ax1b.plot(param_array2, 'go-', linewidth=1)

        # huebsch machen
        custom_plot.turnOffAxes(ax1b, ['top'])
        ax1.set_ylabel(key1)
        ax1b.set_ylabel(key2)

        for a in [ax]:
            custom_plot.huebschMachen(a)
        ax.set_xlabel(key1)
        ax.set_ylabel(key2)
        ax.set_xlim(ax.get_xlim()*numpy.array([.95, 1.05]))
        ax.set_ylim(ax.get_ylim()*numpy.array([.95, 1.05]))

        return (fig, ax), (fig1, ax1, ax1b)


    def parameterTransitionsDict(self, key, labelsize=None, showFigs=False, saveFig=False, saveFolder=''):
        self.plotParameterDistribution(key, labelsize=labelsize, showFigs=showFigs, saveFig=saveFig, saveFolder=saveFolder)
        return self.param_transitions_dict

    def parameterDict(self, key, labelsize=None, showFigs=False, saveFig=False, saveFolder=''):
        self.plotParameterDistribution(key, labelsize=labelsize, showFigs=showFigs, saveFig=saveFig, saveFolder=saveFolder)
        return self.param_dict

    def testParameter(self, key, p=.1, smoothed=10, display=True, fig=None, ax=None):
        """ Test parameter time array for local uniform distribution and make some plots.

        Parameters
        ----------
        key : str
            Keyword of parameter in dictionary.
        p : float
            Minimum for acceptance of the p-value of the chi-squared test.
        smoothed : bool, optional
            Smooth the development of trials. Defaults to 10.
        display : bool, optional
            Display the a plot of the test? Defaults to True.
        fig : matplotlib figure, optional
        ax : matplotlib axes, optional

        Returns
        -------
        chi_ok : bool
            True if all p-values of the chi-squared test are above given p.
        """

        param_array = numpy.array(self.getParameter(key))

        # chi-square testing
        chi_p = []
        for i in range(param_array.size-smoothed):
            chi_p.append(scipy.stats.chisquare(numpy.histogram(param_array[i:i+smoothed],
                                                              numpy.unique(param_array))[0])[1])
        chi_p = numpy.array(chi_p)
        chi_ok = False
        if numpy.all(chi_p > p):
            chi_ok = True

        if not display:
            return chi_ok

        if not fig:
            fig = pl.figure(figsize=(8, 8))
        if not ax:
            pos = [0.125, 0.75, 0.5, 0.23]
            ax = fig.add_axes(pos)
            pos_hist = list(pos)
            pos_hist[0] += pos_hist[2] + .1
            pos_hist[2] = .25
            ax_hist = fig.add_axes(pos_hist)
            pos = list(pos)
            pos[2] = 0.775
            pos[1] -= pos[3] + .1
            ax_corr = fig.add_axes(pos)
            pos = list(pos)
            pos[1] -= pos[3] + .1
            ax2 = fig.add_axes(pos)

        ax.plot(param_array, 'o-')
        print 'avg, std:', param_array.mean(), param_array.std()
        print 'chi-square:', scipy.stats.chisquare(numpy.histogram(param_array, numpy.unique(param_array))[0])
        # ax_hist.hist(param_array, numpy.unique(param_array))
        ax_hist.hist(param_array, numpy.unique(param_array).size)
        ax_corr.plot(numpy.correlate(param_array, param_array, 'full')[param_array.size-1:])

        smoothed_param_array = signale.smooth(param_array, smoothed, 'flat')
        ax.plot(smoothed_param_array)
        ax_hist.hist(smoothed_param_array, 10)
        print 'smoothed avg, std:', smoothed_param_array.mean(), smoothed_param_array.std()

        ax2.plot(chi_p, 'o-')
        ax2.plot(numpy.ones(chi_p.size)*p, ':', linewidth=2)


        # huebsch machen
        for a in [ax, ax_hist, ax_corr, ax2]:
            custom_plot.huebschMachen(a)
        ax.set_xlabel('#')
        ax.set_ylabel(key)
        ax.set_xlim(0, param_array.size)

        ax_hist.set_xlabel(key)
        ax_corr.set_ylabel('auto correlation')

        ax2.set_xlabel('#')
        ax2.set_ylabel('p for uniform distribution')
        ax2.set_xlim(0, param_array.size)
        ax2.set_ylim(0, 1)

        return chi_ok


class paramsTrajectoryList(object):

    def __init__(self):
        self.timeUnit = 's'
        self.parameter_trajectories = {}

    def plotParameter(self, key, fig=None, ax=None):

        if not fig:
            fig = pl.figure()
        if not ax:
            ax = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)

        param_array = []

        for p in self.parameter_trajectories.values():
            param_array.extend(p.getParameter(key))
        ax.plot(param_array, '.-')
        if hasattr(self, 'times'):
            ax2.plot(self.times, param_array, '.-')

        # huebsch machen
        custom_plot.huebschMachen(ax)
        ax.set_ylabel(key)
        ax.set_xlabel('#')
        
        custom_plot.huebschMachen(ax2)
        ax2.set_ylabel(key)
        ax2.set_xlabel('Time ('+self.timeUnit+')')

        return fig, ax

    def plotParameterDistribution(self, key, fig=None, ax=None, showFigs=True, saveFig=False, saveFolder=''):
        """ Plots distributions for parameter.

        One of the parameter distribution and one of the transition distribution across trials.

        Parameters
        ----------
        key : str
            Keyword of parameter in dictionary.
        fig : matplotlib Figure
        ax : matplotlib Axes

        Returns
        -------
        fig : matplotlib Figure
        ax_top : matplotlib Axes
        ax_bottom : matplotlib Axes
        """

        param_array = []
        for p in self.parameter_trajectories.values():
            param_array.extend(p.getParameter(key))


        # plotting absolute and transition parameter distributions as histograms
        if not fig:
            fig = pl.figure(figsize=(20, 7))
        if not ax:
            ax_top = fig.add_subplot(211)
            ax_bottom = fig.add_subplot(212)

        bins = numpy.unique(param_array)
        if bins.size < 10:
            dbin = numpy.mean(numpy.diff(bins))
            bins = numpy.insert(bins, [0, bins.size], [bins[0]-dbin, bins[-1]+dbin])
        else:
            bins = 10
        print bins
        ax_top.hist(param_array, bins=bins, align='left')

        # huebsch machen
        custom_plot.huebschMachen(ax_top)
        for ax in [ax_top, ax_bottom]:
            for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(5)
        ax_top.set_title('Absolute and transition %s distribution' %key)
        ax_top.set_xlabel(key)
        ax_top.set_ylabel('Value count')

        # transitions
        # doubling the all parameters except the first and last one, to analyse parameter transitions
        doubled_params = [str(param_array[0])]
        for i in numpy.arange(1, len(param_array)-1):
            doubled_params.extend([str(param_array[i]), str(param_array[i])])
        doubled_params.append(str(param_array[len(param_array)-1]))
        param_transitions = [(', '.join(map(str, doubled_params[n:n+2]))).replace(",", " ->")
                            for n in range(0, len(doubled_params), 2)]
        self.param_transitions_dict = dict((x, param_transitions.count(x)) for x in param_transitions)

        ax_bottom.bar(*zip(*zip(count(), self.param_transitions_dict.values())))

        # huebsch machen
        custom_plot.huebschMachen(ax_bottom)
        ax_bottom.set_xticks((1*zip(*zip(count(0.4), self.param_transitions_dict)))[0])
        ax_bottom.set_xticklabels((1*zip(*zip(count(0.4), self.param_transitions_dict)))[1])
        ax_bottom.set_xlabel('%s transition' %key)
        ax_bottom.set_ylabel('Value count')

        if saveFig:
            print 'Parameter Distribution plot saved to:', saveFolder+'Parameter_Distribution.png'
            fig.savefig(saveFolder+'Parameter_Distribution.png', format='png')
        if not showFigs:
                pl.ioff()
                pl.close(fig)
        else:
            pl.show()

        return fig, ax_top, ax_bottom

    def id_list(self):
        """ Return the list of all the ids in the parameter trajectory.
        """

        return numpy.array(self.parameter_trajectories.keys())

    def __getitem__(self, id):
        if id in self.id_list():
            return self.parameter_trajectories[id]
        else:
            raise Exception("id %d is not present in the paramsList. See id_list()" %id)

    def __setitem__(self, i, val):
        assert isinstance(val, paramsTrajectory), "An paramsTrajectoryList object can only contain paramsTrajectory objects"
        self.parameter_trajectories[i] = val

    def __iter__(self):
        return self.parameter_trajectories.itervalues()

    def __len__(self):
        return len(self.parameter_trajectories)

    def append(self, signal):
        """
        Add an paramsTrajectory object to the paramsTrajectoryList

        Parameters
        ----------
        signal : paramsTrajectory object
            The paramsTrajectory object to be appended.

        See also
        --------
        __setitem__
        """

        assert isinstance(signal, paramsTrajectory), "An paramsTrajectoryList object can only contain paramsTrajectory objects"
        self[self.parameter_trajectories.__len__()] = signal
