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
    
def detect_IEDs(mne_data, peak_thresh=5, closeness_thresh=0.25, width_thresh=0.2): 
    """
    Source: LFPAnalysis. This function detects IEDs in the LFP signal automatically. Alternative to manual marking of each ied. 

    From: https://academic.oup.com/brain/article/142/11/3502/5566384

    Method 1: 
    1. Bandpass filter in the [25-80] Hz band. 
    2. Rectify. 
    3. Find filtered envelope > 3. 
    4. Eliminate events with peaks with unfiltered envelope < 3. 
    5. Eliminate close IEDs (peaks within 250 ms). 
    6. Eliminate IEDs that are not present on at least 4 electrodes. 
    (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6821283/)

    Parameters
    ----------
    mne_data : mne object 
        mne data to check for bad channels 
    peak_thresh : float 
        the peak threshold in amplitude 
    closeness_thresh : float 
        the closeness threshold in time
    width_thresh : float 
        the width threshold for IEDs 

    Returns
    -------
    IED_samps_dict : dict 
        dict with every IED index  

    """

    # What type of data is this? Continuous or epoched? 
    if type(mne_data) == mne.epochs.Epochs:
        data_type = 'epoch'
        n_times = mne_data._data.shape[-1]
    elif type(mne_data) == mne.io.fiff.raw.Raw: 
        # , mne.io.edf.edf.RawEDF - probably should never include EDF data directly here. 
        data_type = 'continuous'
        n_times = mne_data._data.shape[1]
    else: 
        data_type = 'continuous'
        n_times = mne_data._data.shape[1]       

    sr = mne_data.info['sfreq']
    min_width = width_thresh * sr
    across_chan_threshold_samps = closeness_thresh * sr # This sets a threshold for detecting cross-channel IEDs 

    # filter data in beta-gamma band
    filtered_data = mne_data.copy().filter(25, 80, n_jobs=-1)

    n_fft = next_fast_len(n_times)

    # Hilbert bandpass amplitude 
    filtered_data = filtered_data.apply_hilbert(envelope=True, n_fft=n_fft, n_jobs=-1)

    # Rectify: The point of this step is to focus on the magnitude of oscillations instead of their direction
    filtered_data._data[filtered_data._data<0] = 0

    # Zscore
    filtered_data.apply_function(lambda x: zscore(x, axis=-1))
    IED_samps_dict = {f'{x}':np.nan for x in mne_data.ch_names}
    IED_sec_dict = {f'{x}':np.nan for x in mne_data.ch_names}

    if data_type == 'continuous':
        for ch_ in filtered_data.ch_names:
            sig = filtered_data.get_data(picks=[ch_])[0, :]

            # Find peaks 
            IED_samps, _ = find_peaks(sig, height=peak_thresh, distance=closeness_thresh * sr)

            IED_samps_dict[ch_] = IED_samps 

        # aggregate all IEDs
        all_IEDs = np.sort(np.concatenate(list(IED_samps_dict.values())).ravel())

        # Remove lame IEDs 
        for ch_ in filtered_data.ch_names:
            sig = filtered_data.get_data(picks=[ch_])[0, :]
            # 1. Too wide  
            # Whick IEDs are longer than 200 ms?
            widths = peak_widths(sig, IED_samps_dict[ch_], rel_height=0.75)
            wide_IEDs = np.where(widths[0] > min_width)[0]
            # 2. Too small 
            # Which IEDs are below 3 in z-scored unfiltered signal? 
            small_IEDs = np.where(zscore(mne_data.get_data(picks=[ch_]), axis=-1)[0, IED_samps_dict[ch_]] < 3)[0]
            local_IEDs = [] 
            # 3. Too local 
            # Which IEDs are not present on enough electrodes? 
            # Logic - aggregate IEDs across all channels as a reference point 
            # Check each channel's IED across aggregate to find ones that are close in time (but are<500 ms so can't be same channel)
            for IED_ix, indvid_IED in enumerate(IED_samps_dict[ch_]): 
                # compute the time (in samples) to all IEDS if the 5 closest aren't all within 100 ms, then reject
                diff_with_all_IEDs = np.sort(np.abs(indvid_IED - all_IEDs))[0:5]
                if any(diff_with_all_IEDs>=across_chan_threshold_samps): 
                    local_IEDs.append(IED_ix)
                    # print(diff_with_all_IEDs)
            local_IEDs = np.array(local_IEDs).astype(int)   
            elim_IEDs = np.unique(np.hstack([small_IEDs, wide_IEDs, local_IEDs]))
            revised_IED_samps = np.delete(IED_samps_dict[ch_], elim_IEDs)
            IED_s = (revised_IED_samps / sr)
            IED_sec_dict[ch_] = IED_s   
          
        return IED_sec_dict
    elif data_type == 'epoch':
        # Detect the IEDs in every event in epoch time
        for ch_ in filtered_data.ch_names:
            sig = filtered_data.get_data(picks=[ch_])[:,0,:]
            IED_dict = {x:np.nan for x in np.arange(sig.shape[0])}
            for event in np.arange(sig.shape[0]):
                IED_samps, _ = find_peaks(sig[event, :], height=peak_thresh, distance=closeness_thresh * sr)
                # IED_s = (IED_samps / sr)
                if len(IED_samps) == 0: 
                    IED_samps = np.array([np.nan])
                    # IED_s = np.nan
                IED_dict[event] = IED_samps
            IED_samps_dict[ch_] = IED_dict
        # aggregate all IEDs
        all_IEDs = np.sort(np.concatenate([list(x.values())[0] for x in list(IED_samps_dict.values())]).ravel())

        for ch_ in filtered_data.ch_names:
            sig = filtered_data.get_data(picks=[ch_])[:,0,:]
            for event in np.arange(sig.shape[0]):
                if all(~np.isnan(IED_samps_dict[ch_][event])): # Make sure there are IEDs here to begin with during this event 
                    widths = peak_widths(sig[event, :], IED_samps_dict[ch_][event], rel_height=0.75)
                    wide_IEDs = np.where(widths[0] > min_width)[0]
                    small_IEDs = np.where(zscore(mne_data.get_data(picks=[ch_]), axis=-1)[event, 0, IED_samps_dict[ch_][event]] < 3)[0]
                    local_IEDs = [] 
                    for IED_ix, indvid_IED in enumerate(IED_samps_dict[ch_][event]): 
                        # compute the time (in samples) to all IEDS if the 5 closest aren't all within 100 ms, then reject
                        diff_with_all_IEDs = np.sort(np.abs(indvid_IED - all_IEDs))[0:5]
                        if any(diff_with_all_IEDs>=across_chan_threshold_samps): 
                            local_IEDs.append(IED_ix)
                    local_IEDs = np.array(local_IEDs).astype(int)
                    elim_IEDs = np.unique(np.hstack([small_IEDs, wide_IEDs, local_IEDs]))
                    revised_IED_samps = np.delete(IED_samps_dict[ch_][event], elim_IEDs)
                    IED_samps_dict[ch_][event] = revised_IED_samps

        return IED_samps_dict