__author__ = "Olivia Haas"
__version__ = "1.0, September 2015"

# python modules
import numpy
import hickle
from dateutil import parser

CA1_CA3_info = {'10823_2015-08-05': 'CA3',  # 1900um all cells before and including this date are from CA1
                '10823_2015-08-10': 'CA3',  # 2050um all cells after and including this date are from CA3
                '10529_2015-03-04': 'CA1',  # 1100um all cells before and including this date are from CA1
                '10529_2015-03-24': 'CA3',  # 1750um all cells after and including this date are from CA3
                '10528_2015-03-16': 'CA3',  # 1425um all cells before and including this date are from CA1
                '10528_2015-04-01': 'CA3',  # 1800um all cells after and including this date are from CA3
                '10353_2014-07-02': 'CA3',  # 1250um all cells before and including this date are from CA1
                '10353_2014-07-30': 'CA3',  # 1800um all cells after and including this date are from CA3
                '10535_2015-10-30': 'CA1',  # um all cells before and including this date are from CA1  raus
                '10537_2015-10-30': 'CA1'   # um all cells before and including this date are from CA1  raus
                }

# 10823: CA1 from 1500um to 2000um, after CA3. 37 CA1 hickle cells and 40 CA3 hickle cells = 77cells
# 10529: CA1 from 1080um to 1400um, after CA3. 3 CA1 hickle cells and 8 CA3 hickle cells  = 11 cells
# 10528: CA1 from 1375um to 1600um, after CA3. 4 CA1 hickle cells and 42 CA3 hickle cells = 46 cells
# 10353: CA1 from 1000um to 1500um, after CA3. 9 CA1 hickle cells and 2 CA3 hickle cells = 11 cells

# all: 53 CA1 cells and 92 CA3 cells

# use different marker forms for CA1 and CA3 cells in clustering plot and include them in the legend!

server = 'saw'
summ = '/Users/haasolivia/Documents/'+server+'/dataWork/olivia/hickle/Summary/'
filenames = 'all_filenames.hkl'
used_filenames = 'used_filenames.hkl'
# filenames_dic = hickle.load(summ+filenames)[0]
filenames_dic = hickle.load(summ+used_filenames)
clust = hickle.load(summ+'cluster_indices.hkl')
# clusters = hickle.load(summ+'cluster_indices.hkl')
# clusters['vis_cluster_indices'][numpy.where(numpy.in1d(clusters['vis_cluster_indices'], CA1_idx))[0]]


def get_CA1CA3_clusteridx():
    CA1_idx = []
    CA3_idx = []

    CA = []
    animals = []

    info_CA1_idx = numpy.where(numpy.array(CA1_CA3_info.values()) == 'CA1')[0]
    info_CA1_keys = numpy.array(CA1_CA3_info.keys())[info_CA1_idx]

    info_CA1_animals = []
    info_CA1_dates = []

    ad_idx_pv = [[], [], []]

    for key in numpy.arange(len(info_CA1_keys)):
        info_CA1_animals.append(info_CA1_keys[key].split('_')[0])
        info_CA1_dates.append(info_CA1_keys[key].split('_')[1])

    for i in numpy.arange(len(filenames_dic)):
        animal_str = filenames_dic[i].split('_20')[0]
        date_str = ('20'+filenames_dic[i].split('_20')[1]).split('_')[0]
        animal_and_date = filenames_dic[i].split('_TT')[0]
        ad_index = numpy.where(numpy.array([item.startswith(animal_and_date) for item in filenames_dic]))[0]
        prop_items = numpy.array([item in clust['prop_cluster_indices'] for item in ad_index])
        prop_vis = numpy.array([str(item).replace('True', 'prop').replace('False', 'vis') for item in prop_items])

        if True in prop_items and False in prop_items and not animal_and_date in ad_idx_pv[0]:
            ad_idx_pv[0].append(animal_and_date)
            ad_idx_pv[1].append(ad_index)
            ad_idx_pv[2].append(prop_vis)

        ca1 = 0

        for k in numpy.arange(len(info_CA1_animals)):
            if animal_str == info_CA1_animals[k] and parser.parse(date_str) <= parser.parse(info_CA1_dates[k]):
                CA1_idx.append(i)
                if animal_str == '10823' or animal_str == '10353':
                    CA.append('CA1/CA2')
                else:
                    CA.append('CA1')
                animals.append(animal_str)
                ca1 = 1
        if ca1 == 0:
            CA3_idx.append(i)
            CA.append('CA3')
            animals.append(animal_str)

    CA1_idx = numpy.array(CA1_idx)
    CA3_idx = numpy.array(CA3_idx)
    CA = numpy.array(CA)
    animals = numpy.array(animals)
    # return CA1_idx, CA3_idx
    # return CA1_idx, CA3_idx, CA, animals
    return CA1_idx, CA3_idx
    #, ad_idx_pv

if __name__ == "__main__":
    # CA1_idx, CA3_idx, CA, animals = get_CA1CA3_clusteridx()
    CA1_idx, CA3_idx, ad_idx_pv = get_CA1CA3_clusteridx()
