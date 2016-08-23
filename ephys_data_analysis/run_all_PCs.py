__author__ = "Olivia Haas"
__version__ = "1.0, May 2015"

# python modules
import sys
import os
import subprocess

# add additional custom paths
extraPaths = [os.path.join(os.path.abspath(os.path.dirname(__file__)), '../scripts')]
for p in extraPaths:
    if not sys.path.count(p):
        sys.path.insert(1, p)

# other modules
import GC_RZ


###################################################### functions

cwd = os.getcwd()

# delete hidden files:

hkl_files = []

server = 'saw'

path = '/Users/haasolivia/Documents/'+server+'/dataWork/olivia/'

rausmap = False

if not rausmap:
    filedirec = 'hickle/'
if rausmap:
    filedirec = 'hickle/cells_not_used_79/'

recalc_not_all = 1  # 1 means recalc from a defined file, 0 is when all files should be recalculated
recalc_files_generated_earlier_than = path+'hickle/10823_2015-08-19_VR_GCend_linTrack1_TT2_SS_13_PF_info_normalised.hkl'
#'10528_2015-04-13_VR_GCend_ol_linTrack1_TT2_SS_01_PF_info_normalised.hkl'

for f in os.listdir(path+filedirec):
    if f.endswith('.hkl'):
        if recalc_not_all == 1 and os.path.getmtime(path+filedirec+f) >= os.path.getmtime(recalc_files_generated_earlier_than):
            print ''
            print ''
            print path+filedirec+f, 'was already recalculated!'
            continue
        else:
            print ''
            print ''
            print path+filedirec+f, 'is OLD and will now be recalculated!'
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

            print ''
            print ''
            print '=================================================================================================================================================================================================='
            print ''
            print ''
            print path+filename, '', TTname
            if not rausmap:
                if not gain_normalised:
                    print "python GC_RZ.py "+path+filename+" useRecommended wfit singleRuns rate smooth:'[2.0]' tt:["+TTname+"] noSpeck noShow"
                else:
                    print "python GC_RZ.py "+path+filename+" useRecommended wfit singleRuns rate smooth:'[2.0]' tt:["+TTname+"] noSpeck noShow gain_normalised"
            if rausmap:
                if not gain_normalised:
                    print "python GC_RZ.py "+path+filename+" useRecommended wfit singleRuns rate smooth:'[2.0]' tt:["+TTname+"] noSpeck noShow rausmap"
                else:
                    print "python GC_RZ.py "+path+filename+" useRecommended wfit singleRuns rate smooth:'[2.0]' tt:["+TTname+"] noSpeck noShow gain_normalised rausmap"
            print ''
            print ''

            if not rausmap:
                if not gain_normalised:
                    subprocess.call("python GC_RZ.py "+path+filename+" useRecommended wfit singleRuns rate smooth:'[2.0]' tt:["+TTname+"] noSpeck noShow", shell=True)
                else:
                    subprocess.call("python GC_RZ.py "+path+filename+" useRecommended wfit singleRuns rate smooth:'[2.0]' tt:["+TTname+"] noSpeck noShow gain_normalised", shell=True)

            if rausmap:
                if not gain_normalised:
                    subprocess.call("python GC_RZ.py "+path+filename+" useRecommended wfit singleRuns rate smooth:'[2.0]' tt:["+TTname+"] noSpeck noShow rausmap", shell=True)
                else:
                    subprocess.call("python GC_RZ.py "+path+filename+" useRecommended wfit singleRuns rate smooth:'[2.0]' tt:["+TTname+"] noSpeck noShow gain_normalised rausmap", shell=True)
            #run GC_RZ.py path+filename useRecommended pooled rate smooth:'[2.0]' tt:[TTname] noSpeck noShow


