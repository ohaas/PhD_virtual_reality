"""
For deleting false hotspot events and stimuli and saving then as new events_position.traj and stimuli.tsv
"""

__author__ = "Olivia Haas"
__version__ = "1.0, September 2014"

# python modules
import sys
import os

# add additional custom paths
extraPaths = [os.path.join(os.path.abspath(os.path.dirname(__file__)), '../scripts')]
for p in extraPaths:
    if not sys.path.count(p):
        sys.path.insert(1, p)

# other modules
import numpy
import csv
# custom made modules
import tools
import matplotlib.pyplot as pl


def clean(events, stimuli, traj, rewards_traj, main_folder, stimuli_folder, stimuli_fileName, events_fileName, HS, RZ):

    fig = pl.figure(figsize=(20, 15))
    ax = fig.add_subplot(111)
    pl.plot(traj.times, traj.places[:, 0], linewidth=1, color=numpy.ones(3)*.5)
    pl.plot(events.times, events.places[:, 0], 'o')

    if HS == []:
        for r in numpy.arange(len(RZ)):
            ax.axhline(RZ[r][0], linewidth=1, color='r')
            HS.append(RZ[r])
    else:
        for h in numpy.arange(len(HS)):
            ax.axhline(HS[h][0], linewidth=1, color='r')
    for r in numpy.arange(len(RZ)):
        ax.axhline(RZ[r][0], linewidth=1, color='g')
    pl.xlabel('time in '+traj.timeUnit)
    pl.ylabel('position in '+traj.spaceUnit)

    # get all unfiltered right and leftwards component of the trajectory
    #rightward_traj, leftward_traj = traj.getLeftAndRightwardRuns()

    # loading traj and tsv files and creating new output files
    stimuli_infile = open(main_folder+stimuli_folder+'/'+stimuli_fileName, 'rb')
    stimuli_outfile = open(main_folder+stimuli_folder+'/'+stimuli_fileName.replace('.', '_cleaned.'), 'wb')
    events_infile = open(main_folder+stimuli_folder+'/'+events_fileName, 'rb')
    events_outfile = open(main_folder+stimuli_folder+'/'+events_fileName.replace('.', '_cleaned.'), 'wb')

    stimuli_lines = stimuli_infile.readlines()
    event_lines = events_infile.readlines()

    # write headers to outfiles
    stimuli_infile = open(main_folder+stimuli_folder+'/'+stimuli_fileName, 'rb')
    for idx, line in enumerate(stimuli_infile):
        if line.startswith('#'):
            stimuli_outfile.write(line)
            last_stimuli_header_row = idx

    events_infile = open(main_folder+stimuli_folder+'/'+events_fileName, 'rb')
    for idx, line in enumerate(events_infile):
        if line.startswith('#'):
            events_outfile.write(line)
            last_events_header_row = idx

    #write first stimuli data row to new file, as this row is an extra one, generated when the session is started!
    stimuli_outfile.write(stimuli_lines[last_stimuli_header_row+1])

    # copy all untouched events.times into events_times
    all_event_times = events.times.copy()

    HS = numpy.array(HS)

    i = 0

    while i in numpy.arange(len(events.times)-1):
        print '---------------------------------------------------------------------'
        # find hotspot value closest to place where event was triggered
        print 'hotspot in loop ', type(HS), HS
        HS_i = tools.findNearest(HS[:, 0], events.places[i, 0])[1]
        print 'all hotspots: ', HS[:, 0]
        print 'chosen hotspot: ', HS_i
        print 'event_place: ', events.places[i, 0]

        # find traj values closest to event and following event
        traj_index_i = tools.findNearest(traj.times, events.times[i])[0]
        traj_index_i1 = tools.findNearest(traj.times, events.times[i+1])[0]

        #traj_times_between_events = traj.times[traj_index_i:traj_index_i1]

        print 'traj_index_i: ', traj_index_i
        print 'traj_index_i1: ', traj_index_i1
        print 'traj.places[traj_index_i, 0]: ', traj.places[traj_index_i, 0]
        print 'traj.places[traj_index_i1, 0]: ', traj.places[traj_index_i1, 0]
        traj_xvalues_between_events = traj.places[traj_index_i:traj_index_i1+1, 0]

        print 'events.times[i]: ', events.times[i]
        print 'events.times[i+1]: ', events.times[i+1]
        print 'events.places[i, 0]', events.places[i, 0]
        print 'events.places[i+1, 0]', events.places[i+1, 0]


        print 'events.IDs[i], events.IDs[i+1]: ', events.IDs[i], events.IDs[i+1]
        print 'index of reward in between hotspots: ', numpy.where(numpy.logical_and(events.times[i] < rewards_traj.times, rewards_traj.times < events.times[i+1]))[0]
        print 'rewards in between hotspots: ', numpy.where(numpy.logical_and(events.times[i] < rewards_traj.times, rewards_traj.times < events.times[i+1]))[0].size

        #if event IDs and identical and the reward zone was not crossed (reward times between events == 0)
        if events.IDs[i] == events.IDs[i+1] and numpy.where(numpy.logical_and(events.times[i] < rewards_traj.times,
                                                                              rewards_traj.times < events.times[i+1]))[0].size == 0:
            # if there are traj points outside (smaller or larger) of hotspot area (hs_i +- 0.3 m)
            if numpy.where(HS_i+0.3 < traj_xvalues_between_events)[0].size != 0 or numpy.where(traj_xvalues_between_events < HS_i-0.3)[0].size != 0:
                print 'keep both hotspots because: '
                print 'numpy.where(HS_i+0.3 < traj_xvalues_between_events)[0]: ', numpy.where(HS_i+0.3 < traj_xvalues_between_events)[0]
                print 'numpy.where(traj_xvalues_between_events < HS_i-0.3)[0]: ', numpy.where(traj_xvalues_between_events < HS_i-0.3)[0]
                print 'HS_i+0.3: ', HS_i+0.3
                print 'HS_i-0.3: ', HS_i-0.3
                i += 1

            else:
                print 'deleting i: ', events.times[i]
                print 'keeping i+1: ', events.times[i+1]
                pl.plot(events.times[i], events.places[i, 0], 'o', color='r')
                events.times = numpy.delete(events.times, i, 0)  # delete the deleting parameter (d) event
                events.IDs = numpy.delete(events.IDs, i, 0)
                events.places = numpy.delete(events.places, i, 0)
                stimuli.times = numpy.delete(stimuli.times, (i+1), 0)  # delete row i+1 , because stimuli file has the start up stimulus as first one (extra)
                stimuli.parameters = numpy.delete(stimuli.parameters, (i+1), 0)
        else:
            print 'either there was a reward in between hotspots or event IDs were different (hotspots were different)'
            i += 1

    # writing all kept events and stimuli rows into new events and stimuli files
    # to get the event row that should be written to the new file the row for the event has to be found.
    # For that the index of the kept event time is searched in the original event time array
    for e in numpy.arange(len(events.times)):
        kept_event_row = tools.findNearest(all_event_times, events.times[e])[0]
        events_outfile.write(event_lines[last_events_header_row+1+kept_event_row])  # +1 because kept_event_row starts at zero
        stimuli_outfile.write(stimuli_lines[last_stimuli_header_row+2+kept_event_row]) # +2 because first stimulus is extra

    print 'Saved stimuli_cleaned.tsv file to: ', main_folder+stimuli_folder+'/'+stimuli_fileName.replace('.', '_cleaned.')
    stimuli_infile.close()
    stimuli_outfile.close()
    #numpy.savetxt(main_folder+stimuli_folder+'/'+stimuli_fileName.replace('.', '_cleaned.'), stimuli)

    print 'Saved events_cleaned.traj file to: ', main_folder+stimuli_folder+'/'+events_fileName.replace('.', '_cleaned.')
    events_infile.close()
    events_outfile.close()
    #numpy.savetxt(main_folder+stimuli_folder+'/'+events_fileName.replace('.', '_cleaned.'), events)

    print 'Cleaner plot saved to:', main_folder+stimuli_folder+'/'+'Cleaner.png'
    fig.savefig(main_folder+stimuli_folder+'/'+'Cleaner.png', format='png')

    pl.show()

    return events, stimuli
