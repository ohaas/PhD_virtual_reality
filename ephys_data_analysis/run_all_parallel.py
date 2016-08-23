__author__ = 'haasolivia'
__version__ = "1.0, October 2015"

from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call
import os

commands = []

# takes about 6 hours with 15 cores!
path = '/home/ephysdata/dataWork/olivia/'
# cd /home/ephysdata/dataWork/olivia/Analysis_Scripts/ephys/

recalc_not_all = 1  # 1 means recalc from a defined file, 0 is when all files should be recalculated
recalc_files_generated_earlier_than = path+'hickle/10823_2015-07-31_VR_GCend_linTrack1_TT1_SS_01_PF_info.hkl'

for f in os.listdir(path+'hickle/'):
    if f.endswith('.hkl'):
        if recalc_not_all == 1 and os.path.getmtime(path+'hickle/'+f) >= os.path.getmtime(recalc_files_generated_earlier_than):
            continue
        else:
            gain_normalised = 0
            f_split = f.split('_')
            for i, name in enumerate(f_split):
                if i == 0:
                    file = name
                elif i == 1 or name.startswith('lin') or name.startswith('TT'):
                    file += '/'+name
                elif name.startswith('PF'):
                    file += '.t'
                elif name.startswith('info'):
                    continue
                elif name.startswith('normalised'):
                    gain_normalised = 1
                    continue
                else:
                    file += '_'+name

            filename = file.split('TT')[0]
            TTname = 'TT'+file.split('TT')[1]

            if not gain_normalised:
                # commands.append(['/opt/anaconda/bin/python', path+'Analysis_Scripts/ephys/rawTraces.py', path+filename,
                #                  'Traj', 'time_sorted', 'smooth:[2.0]', 'tt:['+TTname+']', 'useRecommended', 'noShow'])
                #
                # commands.append(['/opt/anaconda/bin/python', path+'Analysis_Scripts/ephys/rawTraces.py', path+filename,
                #                  'Traj', 'gain_sorted', 'smooth:[2.0]', 'tt:['+TTname+']', 'useRecommended', 'noShow'])

                commands.append(['/opt/anaconda/bin/python', path+'Analysis_Scripts/ephys/GC_RZ.py', path+filename,
                                 'useRecommended', 'wfit', 'singleRuns', 'rate', 'smooth:[2.0]', 'tt:['+TTname+']',
                                 'noSpeck', 'noShow'])
            else:
                commands.append(['/opt/anaconda/bin/python', path+'Analysis_Scripts/ephys/GC_RZ.py', path+filename,
                                 'useRecommended', 'wfit', 'singleRuns', 'rate', 'smooth:[2.0]', 'tt:['+TTname+']',
                                 'noSpeck', 'noShow', 'gain_normalised'])


threads = 20  # number of cores that should be used
pool = Pool(threads)      # number of concurrent commands at a time
for i, returncode in enumerate(pool.imap(partial(call), commands)):
    print 'ran ', i, 'of ', len(commands)
    if returncode != 0:
       print("%d command failed: %d" % (i, returncode))