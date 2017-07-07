"""
For running multi_peak.py skript for several cells.
"""

__author__ = "Olivia Haas"
__version__ = "1.0, June 2017"

# python modules
import sys
import os
import subprocess

# add additional custom paths
extraPaths = [os.path.join(os.path.abspath(os.path.dirname(__file__)), '../scripts'),
              os.path.join(os.path.abspath(os.path.dirname(__file__)), '/opt/anaconda/bin/python'),
              os.path.join(os.path.abspath(os.path.dirname(__file__)), '/opt/anaconda/pkgs')]

for p in extraPaths:
    if not sys.path.count(p):
        sys.path.insert(1, p)

import hickle

# other modules
import GC_RZ

######################################################

server = 'saw'
names = hickle.load('/users/haasolivia/documents/'+server+'/dataWork/olivia/hickle/Summary/all_fields.hkl')
path = '/Users/haasolivia/Documents/'+server+'/dataWork/olivia/'

recalc_not_all = 0  # 1 means recalc from a defined file, 0 is when all files should be recalculated
filedirec = 'hickle/FR_SR/'
recalc_files_generated_earlier_than = path + filedirec + '10823_2015-07-13_VR_GCend_linTrack1_TT3_SS_11_FR_ySum_SR.hkl'

for i, name in enumerate(names):
    name2 = path + filedirec + name.split('_PF')[0] + '_FR_ySum_SR.hkl'
    if recalc_not_all == 1 and os.path.getmtime(name2) >= os.path.getmtime(recalc_files_generated_earlier_than):
        print name2, 'was already calculated!'
        continue
    # if os.path.isfile(name2):
    #     print name2, 'was already calculated!'
    #     continue
    else:
        print ''
        print ''
        print name2, 'will now be calculated!'
        name1 = name.split('_PF')[0]
        parts = name1.split('_')
        tetrode = parts[-3] + '_' + parts[-2] + '_' + parts[-1]
        TTname = parts[-3] + '_' + parts[-2] + '_' + parts[-1] + '.t'
        # filename = parts[0]+'/'+parts[1]+'_'+name1.split(parts[0]+'_'+parts[1]+'_')[1].split('_'+parts[-4]+'_'+
        #                                                                                      tetrode)[0]+'/'+parts[-4]+'/'
        middle = name1.split(parts[0] + '_' + parts[1] + '_')[1].split('_lin')[0]
        last = name.split(middle+'_')[1].split('_TT')[0]
        filename = parts[0] + '/' + parts[1] + '_' + middle + '/' + last + '/'

        print  path + filename, '', TTname

        print "python GC_RZ.py " + path + filename + " useRecommended singleRuns rate smooth:'[2.0]' tt:[" + TTname + "] noSpeck noShow"

        subprocess.call("python GC_RZ.py " + path + filename + " useRecommended singleRuns rate smooth:'[2.0]' tt:[" + TTname + "] noSpeck noShow", shell=True)