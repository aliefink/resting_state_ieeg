import numpy as np
import mne
from glob import glob
import pandas as pd
from mne.preprocessing.bads import _find_outliers
import os 
import joblib
import emd
import re



def create_clean_anode_cathode_lists(al, cl, bc):
    """
    This function takes two lists of electrode names, 'al' and 'cl', and returns their cleaned version (eliminating electrodes that have been marked as OOB or noisy)

    Args:
        al (list of strings): anode list output from wm_ref().
        cl (list of strings): cathode list output from wm_ref().
        bc (list of strings): electrode list containing the names of all electrodes that were identified as OOB and noisy.

    Returns:
        anode_list_clean (list of strings): all clean in-brain anode electrodes
        cathode_list_clean (list of strings): all clean in-brain cathode electrodes
    """
    anode_list_clean = []
    removed_anode_index = []
    for i, ch in enumerate(al):

        if ch not in bc:  # You should have 'bad_ch' defined elsewhere
            anode_list_clean.append(ch)
        else:
            removed_anode_index.append(i)

    cathode_list_update = [cl[i] for i in range(len(cl)) if i not in removed_anode_index]

    cathode_list_clean = []
    for ch in cathode_list_update:
        if ch not in bc:  # You should have 'bad_ch' defined elsewhere
            cathode_list_clean.append(ch)

    return anode_list_clean, cathode_list_clean


def join_good_segs(mne_data):
    #creates indices of good epochs after labeling bad times manually, then crops good epochs and joins data 
    
    ### get good times: 
    good_start = list([mne_data_wm_reref.first_time]) #first timepoint in recording (should be 0)
    good_end = []
    
    for annot in mne_data.annotations:
        bad_start = mne_data.time_as_index(annot['onset']) #onset is start time of bad epoch 
        # ^ start time of bad epoch converted to index, then subtract 1 for end of good epoch
        bad_end = mne_data.time_as_index(annot['onset'] + annot['duration']) #onset + duration = end time of bad epoch
        # ^ end time of bad epoch converted to index 
        # must get bad start and end as indices so you can +-1 for good epochs - cannot +-1 using time only indexes

        good_end.append(mne_data.times[bad_start - 1]) #the start time of a bad epoch is the end of a good epoch - 1
        good_start.append(mne_data.times[bad_end+1]) #the end time of a bad epoch is the start of a good epoch +1 index
        #convert to integers before appending - indexing np arrays later is annoying
                          
    good_end.append(mne_data.times[mne_data.last_samp]) #index of last timepoint in recording (should = mne_data.n_times)
    
    ### get good data epochs and concatenate 
    good_segs = []
    for start,end in list(zip(good_start,good_end)):
        good_segs.append(mne_data.copy().crop(tmin=float(start), tmax=float(end),
                include_tmax=True))
    
    return mne.concatenate_raws(good_segs)
    
#derived from: 
    # source: https://mne.discourse.group/t/removing-time-segments-from-raw-object-without-epoching/4169/2
    # source: https://github.com/mne-tools/mne-python/blob/maint/1.5/mne/io/base.py#L681-L742
    
    