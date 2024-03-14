# resting_state_ieeg
Neural analysis of human iEEG resting state dynamics 

To do: 

1. AF to bipolar preprocess all subj (DONE)
2. CM to run FOOOF all subj all elecs (DONE)
3. AF to epoch and create TFRs all subj all elecs
4. CM to clean FOOOF results:
   - exclude channels (or participants?) where aperiodic parameters and/or r2 and error metrics are > 3SD from mean
   - take the highest power within a given band if multiple peaks are detected
   - average fitted parameters across channels to determine subject level periodic and aperiodic parameters
   - compute oscillation score for each channel 
