__author__ = 'haasolivia'

import sys
import os

# add additional custom paths
extraPaths = [os.path.join(os.path.abspath(os.path.dirname(__file__)), '../scripts')]
for p in extraPaths:
    if not sys.path.count(p):
        sys.path.insert(1, p)

import numpy
import matplotlib
matplotlib.rc('ytick', labelsize=6)
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import re
import hickle
from scipy.stats import itemfreq
import signale
import custom_plot
import GC_RZ

gain_05_color = custom_plot.pretty_colors_set2[0]
gain_15_color = custom_plot.pretty_colors_set2[1]


def getGC_indexes(GC, Gain=None):
    gc_indexes = []
    gain_and_indexes = []
    times_and_indexes = []
    gain_sorted_indexes = []
    time_sorted_indexes = []

    for s in numpy.arange(len(GC)):
        if transitions and Gain and GC[s].gain_in == Gain:  # and GC[s].gain_middle == GC_middle or GC[s].gain_in == Grun:
            gc_indexes.append(s)
            gain_and_indexes.append([GC[s].gain_middle, GC[s].visible_middle, GC[s].ol_middle, s])
        elif not transitions:
            gc_indexes.append(s)
            gain_and_indexes.append([GC[s].gain_in, GC[s].visible_in, GC[s].ol_in, s])
            times_and_indexes.append([GC[s].pc_gain_in.traj.t_start, s])

    if not transitions:
        sorted_times = sorted(times_and_indexes)
        for t in numpy.arange(len(times_and_indexes)):
            time_sorted_indexes.append(sorted_times[t][1])

    sorted_gains = sorted(gain_and_indexes)
    for g in numpy.arange(len(gain_and_indexes)):
        gain_sorted_indexes.append(sorted_gains[g][3])

    return gc_indexes, gain_sorted_indexes, time_sorted_indexes


def getTransitionRunningSpikeTimes(GC_Object):

    pc = [GC_Object.pc_gain_in, GC_Object.pc_gain_out]
    for p in pc:
        p.get_runSpikes(p.traj.threshspeed)
    in_Runspike_times = pc[0].run_pc.spiketrains[0].spike_times
    out_Runspike_times = pc[1].run_pc.spiketrains[0].spike_times

    # both spikelists share the time point of the gain change. In case there is a spike exactly at that time-point,
    # the last spike time of the incoming gain will be identical will the first spike time of the outgoing gain. In
    # this case the spike at the gain change time will be considered to belong to the ingoing gain.

    if in_Runspike_times.size and out_Runspike_times.size:
        print type(in_Runspike_times), type(out_Runspike_times), in_Runspike_times, out_Runspike_times
        if in_Runspike_times[-1] == out_Runspike_times[0]:
            out_Runspike_times = out_Runspike_times[1:]

    spike_times = numpy.concatenate((in_Runspike_times, out_Runspike_times), axis=0)

    return spike_times


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


def traces(GC, gc_indexes, traceName, transitions, Speed=False, Frequency=False, filtered=False, SDF=False, Traj=False,
           average=False, plot_length=None, spiketime_direction=None):

    print '-------------------------------------------------------------------'
    print 'number of gain change incidents: ', len(gc_indexes)

    length_out = []
    length = []
    ylabels = []
    y2labels = []
    nan_s = []
    ax = []
    gc = ''
    bottom_title = []
    SRgain = []
    SRspiketimes = []
    SRspiketimes_inPF = []
    SRspiketimes_inPF_L = []
    SRspiketimes_inPF_S = []
    SRspike_xpos = []
    SRspike_xpos_inPF = []
    SRspike_xpos_inPF_L = []
    SRspike_xpos_inPF_S = []
    direction = spiketime_direction

    path = '/Users/haasolivia/Documents/saw/dataWork/olivia/hickle/'
    folder = folderName.split('/')[-4]+'_'+folderName.split('/')[-3]+'_'+folderName.split('/')[-2]+'_'+ttName+'_PF_info.hkl'
    file = hickle.load(path+folder)
    dw = hickle.load(path+'Summary/delta_and_weight_info.hkl')
    double_cells = numpy.array(dw['double_cell_files'])
    double_rundirec = numpy.array(dw['double_cell_direc'])
    data = hickle.load(path+'Summary/raw_data_info_double.hkl')
    single = hickle.load(path+'Summary/raw_data_info.hkl')
    double = 0

    d_idx = numpy.where(double_cells == folder)[0]
    if direction == 'right' or direction == 'left':
        if folder+direction not in [double_cells[d]+double_rundirec[d] for d in d_idx]:
            limits05 = file['pf_limits_'+direction+'Runs_gain_1.5']
            limits15 = file['pf_limits_'+direction+'Runs_gain_0.5']
            idx05 = numpy.where(numpy.array(single['names']) == folder.split('.hkl')[0]+'_'+direction+'_gain_05')[0]
            idx15 = numpy.where(numpy.array(single['names']) == folder.split('.hkl')[0]+'_'+direction+'_gain_15')[0]
            if idx05[0] in single['vis_idx'] and idx15[0] in single['vis_idx']:
                area = 'vis'
            elif idx05[0] in single['rem_idx'] and idx15[0] in single['rem_idx']:
                area = 'vis'
            elif idx05[0] in single['prop_idx'] and idx15[0] in single['prop_idx']:
                area = 'loc'
            else:
                print 'idx05 and idx15 not found in single[vis_idx] or single[vis_idx] !'
                sys.exit()
        else:  # it is a double cell
            double = 1
            idx05L = numpy.where(data['names'] == folder.split('.hkl')[0]+'_'+direction+'_0.5_L')[0]
            idx05S = numpy.where(data['names'] == folder.split('.hkl')[0]+'_'+direction+'_0.5_S')[0]
            idx15L = numpy.where(data['names'] == folder.split('.hkl')[0]+'_'+direction+'_1.5_L')[0]
            idx15S = numpy.where(data['names'] == folder.split('.hkl')[0]+'_'+direction+'_1.5_S')[0]
            limits05L = data['PF_boundaries'][idx05L][0]
            limits05S = data['PF_boundaries'][idx05S][0]
            limits15L = data['PF_boundaries'][idx15L][0]
            limits15S = data['PF_boundaries'][idx15S][0]
            if idx05L[0] in data['vis_idx'] and idx15L[0] in data['vis_idx']:
                areaL = 'vis'
            elif idx05L[0] in data['rem_idx'] and idx15L[0] in data['rem_idx']:
                areaL = 'vis'
            elif idx05L[0] in data['prop_idx'] and idx15L[0] in data['prop_idx']:
                areaL = 'loc'
            else:
                print 'idx05L and idx15L not found in data[vis_idx] or data[vis_idx] !'
                sys.exit()
            if idx05S[0] in data['vis_idx'] and idx15S[0] in data['vis_idx']:
                areaS = 'vis'
            elif idx05S[0] in data['rem_idx'] and idx15S[0] in data['rem_idx']:
                areaS = 'vis'
            elif idx05S[0] in data['prop_idx'] and idx15S[0] in data['prop_idx']:
                areaS = 'loc'
            else:
                print 'idx05S and idx15S not found in data[vis_idx] or data[vis_idx] !'
                sys.exit()
            limits05 = [list(limits05L), list(limits05S)]
            limits15 = [list(limits15L), list(limits15S)]

    if Traj:
        condition = traceName.split('(')[1][0:4]
        traj.ylim[0] = numpy.floor(traj.ylim[0]*100)/100
        traj.ylim[1] = numpy.ceil(traj.ylim[1]*100)/100
        yWidth = traj.ylim[1]-traj.ylim[0]
        print 'indices, yWidth, yValues : ', len(gc_indexes), yWidth, traj.ylim[1], traj.ylim[0]
        xWidth = numpy.round(traj.xWidth)
        fig1 = pl.figure(figsize=(10, 10))
        ax.append(fig1.add_subplot(131))
        ax.append(fig1.add_subplot(132))
        ax.append(fig1.add_subplot(133))
        set_csc_length = xWidth
    else:
        if not plot_length:
            set_csc_length = 12.0
        else:
            set_csc_length = plot_length
        extra = 1.
        if Csc:
            extra = 2.
        fig1 = pl.figure(figsize=(extra*15, extra*10))
        # if len(gc_indexes) <= 5:
        #     fig1 = pl.figure(figsize=(17, 7))
        # else:
        #     fig1 = pl.figure(figsize=(17, len(gc_indexes)*(6./5.)))
        ax.append(fig1.add_subplot(111))

    # xlabel = []
    # for i in [0, 1, 2]:
    if transitions:
        xlabel = 'relative time to gain change (s)'
    elif Traj:
        # if i == 0:
        #     xlabel.append('Position ('+traj.spaceUnit+')')
        # else:
        #     xlabel.append('Position from start point ('+traj.spaceUnit+')')
        xlabel = 'Virtual position ('+traj.spaceUnit+')'
    else:
        xlabel = 'time since last gain change (s)'
    # GC[gc_index].csc is for transitions cut from start over middle to stop and for individual gains from start to middle

    # first get longest trace of all GC[gc_index].csc to set limx=max_csc_length

    if transitions:
        for gc_index in gc_indexes:
            if GC[gc_index].visible_middle == 0:
                ylabels.append(GC[gc_index].gain_middle+' into dark')
            else:
                ylabels.append(GC[gc_index].gain_middle)
            if GC[gc_index].ol_middle == 1:
                ylabels.append(GC[gc_index].gain_middle+' open loop')
            else:
                ylabels.append(GC[gc_index].gain_middle)
        gc = itemfreq(ylabels)
        gc1 = []
        for i in numpy.arange(len(gc)):
            gc1.append(str(gc[i][0])+' x '+str(gc[i][1]))
        gc = str(gc1).strip('[').strip(']')
    if Traj:
        for gc_index in gc_indexes:
            if GC[gc_index].visible_in == 0:
                ylabels.append(str(GC[gc_index].gain_in)+' dark')
            elif GC[gc_index].ol_in == 1:
                ylabels.append(str(GC[gc_index].gain_in)+' open loop')
            else:
                ylabels.append(str(GC[gc_index].gain_in))

    if average and transitions:
        ylabels = numpy.array(numpy.unique(ylabels))
        for gainMiddle in numpy.arange(len(ylabels)):
            nan_s.append([])
    elif average and not transitions:
        nan_s.append([])

    # then normalise and plot everything
    for id, gc_index in enumerate(gc_indexes):
        if transitions:
            print 'gain in -> middle: ', GC[gc_index].gain_in, GC[gc_index].gain_middle
            print 'GC index: ', gc_index
            pc = GC[gc_index].placeCell
            csc = GC[gc_index].csc
        else:
            print 'gain: ', GC[gc_index].gain_in, 'GC index: ', gc_index
            pc = GC[gc_index].pc_gain_in
            csc = GC[gc_index].csc_gain_in
            if Traj:
                t = []
                s = []
                subplot_title = ['All runs', 'Rightward runs', 'Leftward runs']
                pc.traj.getLeftAndRightwardRuns()
                runPC = pc.get_runSpikes(pc.traj.threshspeed)
                runPC.getLeftAndRightwardSpikes()
                runPC.getSpikePlaces()
                t.append(pc.traj.places[:, 0])
                s.append(pc.traj.places[:, 1])
                t.append(pc.traj.rightward_traj.places[:, 0])
                s.append(pc.traj.rightward_traj.places[:, 1])
                t.append(pc.traj.leftward_traj.places[:, 0])
                s.append(pc.traj.leftward_traj.places[:, 1])
                bottom_title.append([GC[gc_index].gain_in, pc.traj.threshspeed])

        if csc.timeUnit == 'ms':
            print 'Changing raw csc trace time Unit from ms in s!'
            csc.changeTimeUnit('s')
        # aligning gain change positions in case of transitions = True

        # t_speed is missing last t axis value (speed calculation needs a next value to compute delta t and delta x)
        # --> t_speed = t[:-1]
        if Speed:
            t, s = pc.traj.getSpeed()
            # normalise speed with gain
            if transitions:
                gain_index = signale.tools.findNearest(t, GC[gc_index].t_middle)[0]
                s = numpy.concatenate((s[0:gain_index+1]/GC[gc_index].gain_in, s[gain_index+1:]/GC[gc_index].gain_middle))
            else:
                s = s/GC[gc_index].gain_in
            if average:
                yText = 'Maximal average speed normalized by gain (m/s)'
            else:
                yText = 'Maximal speed normalized by gain (m/s)'
        elif Frequency:
            s = []
            pxx, freq, t = csc.spectrogram(minFreq=thetaRange[0], maxFreq=thetaRange[1])
            # t has n identical time axes, one for each frequency. t[0] is therefore the time axis. Its then divided by
            # 1000.0 to get the times from ms in seconds
            t = t[0]/1000.0
            # use frequencies with maximal power for each time point i
            for i in numpy.arange(len(t)):
                s.append(freq[:, 0][numpy.where(pxx[:, i] == max(pxx[:, i]))[0][0]])
            if average:
                yText = 'Average csc frequency with strongest power (Hz)'
            else:
                yText = 'Csc frequency with strongest power (Hz)'
        elif SDF:
            if transitions:
                spike_times = getTransitionRunningSpikeTimes(GC[gc_index])
                bottom_title.append([GC[gc_index].gain_in, GC[gc_index].pc_gain_in.traj.threshspeed])
                bottom_title.append([GC[gc_index].gain_middle, GC[gc_index].pc_gain_out.traj.threshspeed])
            else:
                run_pc = pc.get_runSpikes(pc.traj.threshspeed)
                spike_times = run_pc.spiketrains[0].spike_times
                bottom_title.append([GC[gc_index].gain_in, pc.traj.threshspeed])

            t, s, spikes = signale.tools.SDF(spiketrain_times=spike_times, times_start=csc.times_start,
                                             times_stop=csc.times_stop)
            spikes = spike_times-csc.times_start
            if average:
                yText = 'Average firing rate [Hz]'
            else:
                yText = 'Firing rate [Hz]'

        elif Traj:
            yText = 'Y position ('+traj.spaceUnit+')'
        else:
            s = csc.signal.copy()
            # subtract mean and make signal smaller for plotting
            s -= s.mean()
            #s /= max(-1e-100, csc.signal.max())*1.0
            # copy GC[gc_index].csc.times so that the original array doesnt get modified!
            t = csc.times.copy()

            if filtered:
                # create filtered signal in thetaRange
                csc.filter(thetaRange[0], thetaRange[1])
                f = csc.signal_filtered.copy()
                f -= f.mean()
                #f /= max(-1e-100, csc.signal.max())*1.0

        if not Traj:
            normalised_stop = t[-1]-t[0]
            if transitions:
                #csc_time_gain = signale.tools.findNearest(GC[gc_index].csc.times, GC[gc_index].t_middle)[1]
                time_gain = GC[gc_index].t_middle - GC[gc_index].t_start  # = normalised gain time
                length_out.append(normalised_stop-time_gain)
            else:
                time_gain = GC[gc_index].t_middle - GC[gc_index].t_start  # = normalised gain time
                length.append(normalised_stop)

            # normalise t
            t -= t[0]

            if transitions:
                # setting starting time of individual csc signals (s) to its gain change time
                t -= time_gain
                # moving gain change time to max_csc_length/2
                t += set_csc_length/2.0
            if SDF:
                spikes -= time_gain
                spikes += set_csc_length/2.0

            if average:
                # interpolate all time and signal points for a dt of 0.01
                dt = 0.01
                tinterp = numpy.arange(0.0, set_csc_length+dt, dt)
                sinterp = numpy.interp(tinterp, t, s, left=numpy.nan, right=numpy.nan)
                if transitions:
                    nan_s_index = numpy.where(ylabels == GC[gc_index].gain_middle)[0][0]
                    nan_s[nan_s_index].append(sinterp)
                else:
                    nan_s[0].append(sinterp)

            if average and not id == len(gc_indexes)-1:
                continue
            elif average and id == len(gc_indexes)-1:
                nan_t = []
                for n in numpy.arange(len(nan_s)):
                    nan_s[n] = numpy.nanmean(nan_s[n], axis=0)
                    nan_t.append(tinterp)

            # normalise trace within shown window
            if not average:
                shown_start = len(numpy.where(t < 0.0)[0])
                shown_stop = len(numpy.where(t <= set_csc_length)[0])
                max_s = max(s[shown_start:shown_stop])
                s = s/max_s
            elif average:
                max_s = numpy.nanmax(nan_s, axis=1)
                s = (numpy.divide(numpy.transpose(nan_s), max_s))
                t = numpy.transpose(nan_t)

        elif Traj:
            max_s = traj.ylim[1]+0.02
            s = s/max_s

        if filtered:
            if len(f) > shown_stop:
                max_f = max(f[shown_start:shown_stop])
            else:
                max_f = max(f)
            f = f/max_f

        if average:
            y2labels = numpy.round(numpy.insert(max_s, 0, 0.), 2)
            idP = numpy.arange(len(numpy.transpose(s)))
        elif Speed or Frequency or SDF or Traj:
            idP = id
        else:
            idP = id * 2
        if not average:
            if id == 0:
                if Traj:
                    y2labels.append(traj.ylim[0])
                else:
                    y2labels.append(0.00)
                y2labels.append(round(max_s, 2))
            else:
                if Traj:
                    y2labels.append('')
                else:
                    y2labels.append(round(max_s, 2))

        Cmap = []
        # for i in numpy.arange(len(y2labels)):
        #     if not average:
            #     i = id
            # Cmap.append(custom_plot.pretty_colors_set2[i%custom_plot.pretty_colors_set2.__len__()])
        # for i in numpy.arange(len(ylabels)):
        #     if ylabels[i] == '0.5' or ylabels[i] == '0.5 dark' or ylabels[i] == '0.5 open loop':
        #         Cmap.append(gain_05_color)
        #     else:
        #         Cmap.append(gain_15_color)

        # ================================== PLOTTING ==================================

        for id in numpy.arange(len(ax)):  # id = [0, 1, 2] = ['All runs', 'Rightward runs', 'Leftward runs']
            ax[id].set_color_cycle(Cmap)
            if not Traj:
                t = [t]
                s = [s]

            if GC[gc_index].gain_in == 0.5:
                colour = gain_05_color
            else:
                colour = gain_15_color

            # ======================== SR spike times etc appended for hkl ==================

            if spiketime_direction == 'right' and id == 1 or spiketime_direction == 'left' and id == 2:

                if GC[gc_index].gain_in == 0.5:
                    limits = limits05
                else:
                    limits = limits15

                # all spike times and places -------------------------------------

                spikeTimes = runPC[id].spike_times
                if runPC[id].spike_places.any():
                    spikeX = runPC[id].spike_places[:, 0]
                else:
                    spikeX = numpy.array([numpy.nan])

                # spike times and places in place field(s)-------------------------------------

                if double:
                    st_pfL = runPC[id].get_inFieldSpikes(pf_xlim=limits[0], xbin_only=True)
                    st_pfS = runPC[id].get_inFieldSpikes(pf_xlim=limits[1], xbin_only=True)

                    spikeTimes_pfL = st_pfL.spike_times
                    spikeTimes_pfS = st_pfS.spike_times
                    if st_pfL.spike_places.any():
                        spikeX_pfL = st_pfL.spike_places[:, 0]
                        # for leftward runs use abolute x-value from start position (2m)
                        if spiketime_direction == 'left':
                            spikeX_pfL = abs(spikeX_pfL-2.)
                    else:
                        spikeX_pfL = numpy.array([numpy.nan])
                    if st_pfS.spike_places.any():
                        spikeX_pfS = st_pfS.spike_places[:, 0]
                        # for leftward runs use abolute x-value from start position (2m)
                        if spiketime_direction == 'left':
                            spikeX_pfS = abs(spikeX_pfS-2.)
                    else:
                        spikeX_pfS = numpy.array([numpy.nan])
                else:
                    st_pf = runPC[id].get_inFieldSpikes(pf_xlim=limits, xbin_only=True)

                    spikeTimes_pf = st_pf.spike_times
                    if st_pf.spike_places.any():
                        spikeX_pf = st_pf.spike_places[:, 0]
                        # for leftward runs use abolute x-value from start position (2m)
                        if spiketime_direction == 'left':
                            spikeX_pf = abs(spikeX_pf-2.)
                    else:
                        spikeX_pf = numpy.array([numpy.nan])

                if GC[gc_index].visible_in == 0 or GC[gc_index].ol_in == 1:
                    spikeTimes *= numpy.nan
                    spikeX *= numpy.nan
                    if double:
                        spikeTimes_pfL *= numpy.nan
                        spikeTimes_pfS *= numpy.nan
                        spikeX_pfL *= numpy.nan
                        spikeX_pfS *= numpy.nan
                    else:
                        spikeTimes_pf *= numpy.nan
                        spikeX_pf *= numpy.nan

                SRgain.append(GC[gc_index].gain_in)
                SRspiketimes.append(spikeTimes)
                SRspike_xpos.append(spikeX)
                if double:
                    SRspiketimes_inPF_L.append(spikeTimes_pfL)
                    SRspiketimes_inPF_S.append(spikeTimes_pfS)
                    SRspike_xpos_inPF_L.append(spikeX_pfL)
                    SRspike_xpos_inPF_S.append(spikeX_pfS)
                else:
                    SRspiketimes_inPF.append(spikeTimes_pf)
                    SRspike_xpos_inPF.append(spikeX_pf)

            # Background color for horizontal stripes__________________
            if Csc:
                if idP%4 == 0:
                    ax[id].axhspan(idP-1, idP+1, facecolor='#D3D3D3', alpha=0.2, linewidth=False)

            # if type(idP) == int and not Csc and idP%2 == 0:
            #     ax[id].axhspan(idP-1, idP, facecolor='#D3D3D3', alpha=0.2, linewidth=False, zorder=0)
            #     ax[id].axhline(idP, color='#D3D3D3', alpha=0.2, zorder=0)

            if type(idP) == int and not Csc and not idP in [0, gc_indexes[-1]]:
                ax[id].axhspan(idP, idP+1, facecolor=colour, alpha=0.2, linewidth=False, zorder=0)
                # ax[id].axhline(idP, color=gain_05_color, alpha=0.2, zorder=0)

            if type(idP) == int and not Csc and idP == gc_indexes[-1]:
                ax[id].axhspan(idP, idP+1.25, facecolor=colour, alpha=0.2, linewidth=False, zorder=0)

            if type(idP) == int and not Csc and idP == 0:
                ax[id].axhspan(-.25, 1, facecolor=colour, alpha=0.2, linewidth=False, zorder=0)

            # if type(idP) == int and not Csc and idP == 0:
            #     ax[id].axhspan(idP-1, idP+0.25, facecolor=colour, alpha=0.2, linewidth=False, zorder=0)

            if type(idP) != int:
                print 'idP: ', idP
                for iP in numpy.arange(len(idP)):
                    if iP%2 == 0:
                        ax[id].axhspan(iP, iP+1, facecolor='#D3D3D3', alpha=0.2, linewidth=False)

            # PLOTTING TRAJECTORIES____________________________

            ax[id].plot(t[id], numpy.add(s[id], idP), '-', color=colour, linewidth=1.5)
            # _________________________________________________

            if SDF and not average:
                if max_s >= 1.0:
                    max_y = (1.0/max_s)+idP
                else:
                    max_y = max_s+idP
                print 'ymin: ', idP, 'ymax: ', max_y, 'convolution maximum: ', max_s
                ax[id].vlines(x=spikes, ymin=idP, ymax=max_y, color='k')

            if Traj:
                # PLOTTING SPIKES_________________________________________

                if runPC[id].spike_places.any():
                    ax[id].plot(runPC[id].spike_places[:, 0], numpy.add(runPC[id].spike_places[:, 1]/max_s, idP), 'o',
                                color='r', markeredgecolor='r')
                # ________________________________________________________

            if filtered and len(f) == len(t[id]):
                ax[id].plot(t[id], f+idP, '-', linewidth=1, color='k')
            elif filtered and len(f) != len(t[id]):
                print 'WARNING: filtered signal is shorter and will be plotted from filtered trace start!'
                min_len = min(len(f), len(t[id]))
                ax[id].plot(t[id][0:min_len], f[0:min_len]+idP, '-', linewidth=1, color='k')

    if not bottom_title == []:
        bottom_title = numpy.vstack({tuple(row) for row in bottom_title})

    if transitions and not Speed and not Frequency and not SDF and id == 0:
        for i in numpy.arange(len(ylabels)):
            ylabels[i] = str(ylabels[i])
        l = numpy.insert(ylabels, numpy.arange(len(ylabels))[1::], '')
        ylabels = numpy.insert(l, len(l), '')

    if transitions and not Traj:
        # gain change (events.times[GC[gc_index].index_middle]) should be marked by vertical red line and therefore aligned
        ax[0].axvline(x=set_csc_length/2.0, linewidth=2, color='r')

    # finish off plotting_______________________________
    for id in numpy.arange(len(ax)):  # id = [0, 1, 2] = ['All runs', 'Rightward runs', 'Leftward runs']

        custom_plot.huebschMachen(ax[id])
        ax[id].set_xlabel(xlabel)
        if Speed or Frequency or SDF or Traj:
            ax[id].set_yticks(numpy.arange(len(y2labels)-1))
        else:
            ax[id].set_yticks(numpy.arange(2*(len(y2labels)-1)))

        if id == len(ax)-1:
            ax[id].set_yticklabels(ylabels, rotation=0)
        else:
            ax[id].set_yticklabels('', rotation=0)
        if transitions:
            ax[id].set_ylabel('Gain after gain change', rotation=90)
        if Traj and id == len(ax)-1:
            ax[id].set_ylabel('Gain', rotation=90)
            # if condition == 'gain':
            #     ax[id].set_ylabel('Gain', rotation=90)
            # elif condition == 'time':
            #     ax[id].set_ylabel('Normalised start time of runs in '+str(GC[gc_indexes[0]].pc_gain_in.traj.timeUnit), rotation=90)
        ax[id].set_xlim(0, set_csc_length)
        if Speed or Frequency or SDF or Traj:
            ax[id].set_ylim(0, len(y2labels)-1)
            ax2 = ax[id].twinx()
            if id == 0:
                ax2.set_ylabel(yText)
                ax2.set_yticklabels(y2labels)
            else:
                ax2.set_yticklabels('')
            ax2.set_yticks(numpy.arange(len(y2labels)))
            ax2.yaxis.set_ticks_position('left')
            ax2.yaxis.set_label_position('left')
        else:
            ax[id].set_ylim(-1, 2*len(gc_indexes)-1)
            ax[id].set_yticks(numpy.arange(2*len(gc_indexes)))
        #ax.yaxis.tick_right()
        ax[id].yaxis.set_ticks_position('right')
        ax[id].yaxis.set_label_position('right')
        if Traj:
            ax[id].set_title(subplot_title[id])

    if Frequency:
        pl.suptitle('Start gain: '+str(GC[gc_indexes[0]].gain_in)+' ('+str(len(gc_indexes))+' times: '+gc+
                    '), for trace: '+traceName+'\nFrequency with highest power within: '+str(thetaRange[0])+' and '+
                    str(thetaRange[1])+' Hz')
    elif not Frequency and not Traj:
        pl.suptitle('Start gain: '+str(GC[gc_indexes[0]].gain_in)+' ('+str(len(gc_indexes))+' times: '+gc+
                    '), for trace: '+traceName)
    elif Traj:
        pl.suptitle(traceName)
    if Traj or SDF:
        vRun_thresh = []
        for i in numpy.arange(len(bottom_title)):
            vRun_thresh.append('>='+str(numpy.round(bottom_title[i][1], 3))+' m/s for gain '+
                                   str(numpy.round(bottom_title[i][0], 3)))
        vRun_thresh = ", ".join(vRun_thresh)

        if smooth:
            fig1.text(.995, .01, 'Trajectory smoothing window: '+str(smooth)+' s\n v_run '+vRun_thresh + ', ' + cscName+
                      ', ' +pc.tags['dir'] + ', ' + pc.tags['file'], fontsize=fontsize-6,  horizontalalignment='right')
        else:
            fig1.text(.995, .01, 'Not smoothed trajectory\n v_run '+vRun_thresh + ', ' + cscName+', ' +pc.tags['dir'] +
                      ', ' + pc.tags['file'], fontsize=fontsize-6,  horizontalalignment='right')
    if Csc and filtered:
            fig1.text(.995, .01, 'Filtered for frequency band: '+str(thetaRange[0])+' to '+str(thetaRange[1])+' Hz\n'+
                      pc.tags['dir'], fontsize=fontsize-6,  horizontalalignment='right')
    elif not filtered and Csc or Frequency:
        fig1.text(.995, .01, pc.tags['dir'], fontsize=fontsize-6,  horizontalalignment='right')

    # ---------- making SR spiketime arrays the same legth, by filling them with nans ----------

    make_subarrays_equal_long(SRspiketimes)
    make_subarrays_equal_long(SRspike_xpos)

    if double:
        make_subarrays_equal_long(SRspiketimes_inPF_L)
        make_subarrays_equal_long(SRspiketimes_inPF_S)
        make_subarrays_equal_long(SRspike_xpos_inPF_L)
        make_subarrays_equal_long(SRspike_xpos_inPF_S)

        info = {'SRgains': SRgain, 'SRspiketimes': SRspiketimes, 'SRspike_xpos': SRspike_xpos,
                'SRspiketimes_inPF_L': SRspiketimes_inPF_L, 'SRspiketimes_inPF_S': SRspiketimes_inPF_S,
                'SRspike_xpos_inPF_L': SRspike_xpos_inPF_L, 'SRspike_xpos_inPF_S': SRspike_xpos_inPF_S,
                'category_L': areaL, 'category_S': areaS,
                'run_direction': direction}

    else:
        make_subarrays_equal_long(SRspiketimes_inPF)
        make_subarrays_equal_long(SRspike_xpos_inPF)

        info = {'SRgains': SRgain, 'SRspiketimes': SRspiketimes, 'SRspike_xpos': SRspike_xpos,
                'SRspiketimes_inPF': SRspiketimes_inPF,
                'SRspike_xpos_inPF': SRspike_xpos_inPF,
                'category': area,
                'run_direction': direction}

    name = folderName.split('/')[-4]+'_'+folderName.split('/')[-3]+'_'+folderName.split('/')[-2]+'_'+ttName+'_'+direction+'.hkl'
    hickle.dump(info, path+'overlap/'+name)

    return fig1

###################################################### main

if __name__ == "__main__":
    print 'MAIN'

    ###################################################### plotting initialization

    colors, grau, grau2, Bildformat, fontsize = GC_RZ.ini_plotting()

    ###################################################### commandline paramters

    mazeName, lang, transitions, offset, gain_in, gain_middle, allRunns, pooled, plotPlace, chic, showFigs, \
    saveFigs, saveAna, update, color1, color2, noSpeck, useRecommended, expType, num_singleRuns, \
    folderName, smooth = GC_RZ.commandline_params()

    Csc = False
    Speed = False
    filtered = False
    Frequency = False
    SDF = False
    average = False
    Traj = False
    time_sorted = False
    gain_sorted = False
    length = None

    for argv in sys.argv[2:]:
        if argv == 'noShow':
            showFigs = False			# show pics
        if argv == 'saveFigs':
            saveFigs = True				# save pics
        if argv.startswith('csc:'):
            csc = argv.split(':')[-1]   # csc file to load
        if argv.startswith('tt:'):                      # Tetrode files to load, write e.g. as tt:['TT2_01.t']
            tt = argv.split('tt:')[1].strip('[').strip(']')
            tt = [s for s in tt.split(',')]
        if argv == 'noSpeck':
            noSpeck = True				            # only running spikes
        if argv == 'useRecommended':
            useRecommended = True                       # use recommendations from metadata.dat
        if argv.startswith('thetaRange:'):
            thetaRange = argv.split('thetaRange:')[1].strip('[').strip(']')   #write into terminal e.g. as thetaRange:'[6, 10]'
            thetaRange = [float(thetaRange.split(',')[0]), float(thetaRange.split(',')[1])]
        if argv.startswith('gainChange:'):
            gainChange = argv.split('gainChange:')[1].strip('[').strip(']') #write into terminal e.g. as gainChange:'[1.0, 2.0]'
            gain_in = float(gainChange.split(',')[0])
            gain_middle = float(gainChange.split(',')[1])
            print 'Gain changes are plotted for gains ', gain_in, ' -> ', gain_middle
        if argv.startswith('gain_in:'):
            gain_in = argv.split('gain_in:')[1].strip('[').strip(']')   #write into terminal e.g. as gain_in:'[1.0]'
            gain_in = float(gain_in.split(',')[0])
            print 'Gain changes are plotted for gain in: ', gain_in
        if argv == 'allRunns':
            allRunns = True
            print 'Gain changes are plotted for ALL GAIN CHANGES !!'
        if argv == 'pooled':
            pooled = True
            print 'Gain changes are being POOLED for individual gain changes!'
        if argv.startswith('threshspeed:'):
            threshspeed = float(argv.split('threshspeed:')[-1])
        if argv == 'plotPlace':
            plotPlace = True
        if argv == 'transitions':
            transitions = True
        if argv == 'Csc':
            Csc = True
        if argv == 'Speed':
            Speed = True
        if argv == 'filtered':
            filtered = True
        if argv == 'Frequency':
            Frequency = True
        if argv == 'SDF':
            SDF = True
        if argv == 'average':
            average = True
        if argv == 'Traj':
            Traj = True
        if argv == 'time_sorted':
            time_sorted = True
        if argv == 'gain_sorted':
            gain_sorted = True
        if argv.startswith('run_direction:'):  # write into terminal e.g. as run_direction:'right'
            run_direction = argv.split('run_direction:')[1]
        if argv.startswith('length:'):
            length = argv.split('length:')[1].strip('[').strip(']')   #write into terminal e.g. as length:'[1.0]'
            length = float(length.split(',')[0])
            if transitions:
                print 'shown plot will be for time window of: ', length, ' sec (', length/2., 's before and after the gain change)'
            else:
                print 'shown plot will be for time window of: ', length, ' sec (from the beginning of the plotted gain section)'
        if argv.startswith('smooth:'):
            smooth = argv.split('smooth:')[1].strip('[').strip(']')   #write into terminal e.g. as smooth:'[1.0]'
            smooth = float(smooth.split(',')[0])

    ###################################################### initialization

    # get parameters
    parameters = {'csc': 'CSC1.ncs', 'tt': '', 'thetaRange': [6, 10], 'threshspeed': 0.1, 'animal': '', 'depth': 0}
    print 'useRecommended in get_params: ', useRecommended
    if useRecommended:
        fileName = os.path.normpath(folderName)+'/metadata.dat'
    else:
        fileName = ''
    dictio, metadata = signale.get_metadata(fileName, parameters, locals())

    locals().update(dictio)

    TTName, cscName = GC_RZ.get_params(tt=tt, csc=csc)

    ###################################################### get data

    GC_RZ.getDataCheck(folderName=folderName)

    spikes, ID, cscID, cscList, stList, eventData, traj, rewards_traj, events, events_fileName, stimuli, \
    stimuli_fileName, stimuli_folder, main_folder, loaded_cleaned_events, loaded_cleaned_stimuli, hs, rz, \
    loadedSomething, cwd = GC_RZ.getData(folderName=folderName, cscName=cscName, TTName=TTName)

    if ID == -1:
        sys.exit('The folders do not contain tetrode data (t files)!')

    ###################################################### crop data

    GC_RZ.cropData(eventData=eventData, traj=traj, threshspeed=threshspeed)

    ###################################################### clean stimuli and event files

    if stimuli_fileName.endswith('stimuli.tsv'):
        print 'CLEANING EVENTS AND STIMULI NOW!'
        events, stimuli = signale.cleaner.clean(events, stimuli, traj, rewards_traj, main_folder, stimuli_folder,
                                                stimuli_fileName, events_fileName, hs, rz)
    else:
        print 'NOT CLEANING EVENTS AND STIMULI, THEY HAVE ALREADY BEEN CLEANED!'

    ###################################################### get gain changes and do the execute the MAIN FUNCTIONS

    Titles, AxesNum, paramsTransDict = GC_RZ.getGainTransitionsAndOccurance(transitions=transitions, showFigs=showFigs,
                                                                            saveFig=True, saveFolder=folderName)

    for zaehler, pc in enumerate(stList):
        print 'T-file:', pc.tags['file']

        # in spikesPhase.prepareAndPlotPooledPhases the noSpeck version plots only the pc running spikes depending on
        # its pc.threshspeed and the normal version plots all spikes in grey and the running spikes in red on top
        pc.traj = traj
        if smooth:
            print 'INFO: trajectory is smoothed with the smoothing kernel width of ', smooth, ' s'
            pc.traj.smooth(smooth)
        pc.thetaRange = (thetaRange[0], thetaRange[1])

        GC = GC_RZ.getAllgainChanges(pc, transitions=transitions)
        GC_RZ.getAllgainChangePCs(GC=GC, mother_spiketrain=pc)
        GC_RZ.getAllgainChangeCSCs(GC=GC)
        ttName = stList.tags[zaehler]['file'].split('.')[0]

        old_start_gains = []

        if not Traj:
            for PT in numpy.arange(len(paramsTransDict)):  # go through available gainchange events calulated by the Dict
                Title = Titles[PT]
                AxNum = AxesNum[PT]
                if transitions:
                    # find the gain_in and gain_middle from dictionary string
                    Grun = 0
                    GC_in = float(re.findall(r"[-+]?\d*\.\d+|\d+", Title)[0])
                    Gain = GC_in
                    GC_middle = float(re.findall(r"[-+]?\d*\.\d+|\d+", Title)[1])
                    # if only a specific gain pair is wanted, they will have to equal the Dict pair
                    if gain_in:
                        if float(gain_in) == GC_in:
                            pass
                        elif float(gain_in) != GC_in:
                            continue
                else:
                    Grun = Title
                    if gain_in:
                        if float(gain_in) == Grun:
                            Gain = Grun
                        elif float(gain_in) != Grun:
                            continue
                    else:
                        Gain = Grun

                gc_indexes, gain_sorted_indexes, time_sorted_indexes = getGC_indexes(GC=GC, Gain=Gain)
                #if not transitions:
                #    sorted_indexes = gc_indexes
                fig = []
                # Csc, Speed and Csc Frequency content are for all tt files the same. The only difference is the amount of
                # spikes. Therefore the spike density (SDF) is calculated for all tt files separately.
                if zaehler == 0 and not Gain in old_start_gains:
                    if Csc:
                        for csc in cscList:
                            fig.append(traces(GC=GC, gc_indexes=gain_sorted_indexes, traceName=cscName,
                                                 transitions=transitions, Speed=False, filtered=filtered, plot_length=length))
                            if saveFigs:
                                print 'CSC plot saved to:', folderName+'gainIn_'+str(Gain)+'_csc.pdf'
                                fig[0].savefig(folderName+'gainIn_'+str(Gain)+'_csc.pdf', format='pdf')
                    if Speed:
                        fig.append(traces(GC=GC, gc_indexes=gain_sorted_indexes, traceName='trajectory running speed',
                                             transitions=transitions, Speed=True, average=average, plot_length=length))
                        if saveFigs:
                            if Csc:
                                fig2 = fig[1]
                            else:
                                fig2 = fig[0]
                            if average:
                                fig2.savefig(folderName+'gainIn_'+str(Gain)+'_Speed_average.pdf', format='pdf')
                                print 'csc gain change plot saved to:', folderName+'gainIn_'+str(Gain)+'_Speed_average.pdf'
                            else:
                                fig2.savefig(folderName+'gainIn_'+str(Gain)+'_Speed.pdf', format='pdf')
                                print 'csc gain change plot saved to:', folderName+'gainIn_'+str(Gain)+'_Speed.pdf'

                    if Frequency:
                        print 'plotting frequency with highest power within: ', thetaRange[0], ' and ', thetaRange[1], ' Hz'
                        fig.append(traces(GC=GC, gc_indexes=gain_sorted_indexes, traceName=cscName, transitions=transitions,
                                             Frequency=True, average=average, plot_length=length))
                        if saveFigs:
                            if Csc and Speed:
                                fig3 = fig[2]
                            elif Csc or Speed:
                                fig3 = fig[1]
                            else:
                                fig3 = fig[0]
                            if average:
                                fig3.savefig(folderName+'gainIn_'+str(Gain)+'_Frequency_average.pdf', format='pdf')
                                print 'csc gain change plot saved to:', folderName+'gainIn_'+str(Gain)+'_Frequency_average.pdf'
                            else:
                                fig3.savefig(folderName+'gainIn_'+str(Gain)+'_Frequency.pdf', format='pdf')
                                print 'csc gain change plot saved to:', folderName+'gainIn_'+str(Gain)+'_Frequency.pdf'

                if SDF and not Gain in old_start_gains:
                    print 'calculating SDF'
                    fig.append(traces(GC=GC, gc_indexes=gain_sorted_indexes, traceName='Spiketrain', transitions=transitions,
                                         SDF=True, average=average, plot_length=length))
                    if saveFigs:
                        if zaehler == 0:
                            if Csc and Speed and Frequency:
                                fig4 = fig[3]
                            elif Csc and Speed or Csc and Frequency or Speed and Frequency:
                                fig4 = fig[2]
                            elif Csc or Speed or Frequency:
                                fig4 = fig[1]
                            else:
                                fig4 = fig[0]
                        else:
                            fig4 = fig[0]
                        if average:
                            fig4.savefig(folderName+ttName+'_gainIn_'+str(Gain)+'_SDF_average.pdf', format='pdf')
                            print 'SDF gain change plot saved to:', folderName+ttName+'_gainIn_'+str(Gain)+'_SDF_average.pdf'
                        else:
                            fig4.savefig(folderName+ttName+'_gainIn_'+str(Gain)+'_SDF.pdf', format='pdf')
                            print 'SDF gain change plot saved to:', folderName+ttName+'_gainIn_'+str(Gain)+'_SDF.pdf'

                old_start_gains.append(Gain)
                if not showFigs:
                    pl.ioff()
                    for f in fig:
                        pl.close(f)
                else:
                    pl.show()

        elif Traj:
            gc_indexes, gain_sorted_indexes, time_sorted_indexes = getGC_indexes(GC=GC)
            print 'plotting trajectories and spikes'
            if gain_sorted:
                fig = traces(GC=GC, gc_indexes=gain_sorted_indexes, traceName='Trajectory and spiketrain (gain sorted)',
                                  transitions=False, Traj=True)
                sort = '_gainSorted'
            elif time_sorted:
                fig = traces(GC=GC, gc_indexes=time_sorted_indexes, traceName='Trajectory and spiketrain (time sorted)',
                                  transitions=False, Traj=True, spiketime_direction=run_direction)
                sort = '_timeSorted'
            else:
                print 'WARNING: gain_sorted or time_sorted needs to be specified in terminal!'
                sys.exit()
            if saveFigs:
                print 'Saving single run Plots in: '+folderName+ttName+'_spikePlace'+sort+'.pdf'
                fig.savefig(folderName+ttName+'_spikePlace'+sort+'.pdf', format='pdf')

                second_figname = folderName.split('/olivia/')[0]+'/olivia/hickle/Plots/'+folderName.split('/')[-4]+'_'+\
                                 folderName.split('/')[-3]+'_'+folderName.split('/')[-2]+'_'+ttName+'_spikePlace'+sort+'.pdf'
                print 'And in: '+second_figname
                fig.savefig(second_figname, format='pdf')

        if not showFigs:
            pl.ioff()
            if type(fig) == list:
                for f in fig:
                    pl.close(f)
            else:
                pl.close(fig)
        else:
            pl.show()

    os.chdir(cwd)


