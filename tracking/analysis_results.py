import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'uavdt'

"""DGTrack"""
trackers.extend(trackerlist(name='DGTrack', parameter_name= 'deit_tiny_distilled_patch16_224', dataset_name=dataset_name,
                            run_ids=None, display_name='DGTrack'))


dataset = get_dataset(dataset_name)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))

import os
import shutil

directory_path = "/home/lzw/LEF/AVtrack_FADA_OAMix_Three/output/test"


