import pandas as pd
from scipy.signal import butter, lfilter, filtfilt
import numpy as np
import matplotlib.pyplot as plt




########## LOW PASS FILTER HELPERS ########################
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    # Use lfilter for causal filtering (introduces phase shift)
    # or filtfilt for zero-phase filtering (no phase shift, but requires more data)
    #y = lfilter(b, a, data) 
    y = filtfilt(b, a, data) 
    return y
###########################################################


def get_sampling_rate(time_signal: pd.DataFrame):
    # Calculate time differences
    time_diffs = time_signal.diff().dropna()
    # Determine sampling rate (assuming constant sampling interval)
    sampling_rate = 1 / time_diffs.median()
    #print(f"Input sampling rate: {sampling_rate} Hz")
    return sampling_rate





def match_fp_to_bold(fp_df: pd.DataFrame, num_vols: int, TR: float, corresponding_time_cols) -> pd.DataFrame:
    '''
    Involves resampling and aligning time points with BOLD trace. This will be done using the
    TR of the BOLD data along with its number of volumes to ensure equal number of samples. 
    Only work with normalized columns and the TTL columns.

    Best practice for downsampling from high sampling rate FP data to low rate BOLD data:
        
        1) Band-limit first (anti-alias): Before you go from, say, 50–200 Hz down to ~0.5–1 Hz (TR=2–1 s),
        low-pass so nothing above the new Nyquist folds back.
            -> Rule of thumb: set cutoff ≲ 0.8·(1/(2·TR)). For TR=2 s → Nyquist=0.25 Hz → cutoff ~0.18–0.2 Hz.
            -> HRF energy is very low-frequency; you won’t miss meaningful signal by low-passing this hard.
        
        2) Downsample onto the TR grid: After filtering, resample to exactly one sample per TR (fs_new = 1/TR). 
        Prefer polyphase resampling or bin-averaging per TR window (which also mimics fMRI’s temporal averaging).
        
    '''
    # Get cutoff frequency from TR:
    cutoff_from_TR = 0.8 * (1 / (2 * TR)) # [Hz]
    #print(f'Cutoff frequency is {cutoff_from_TR} Hz.')

    # BOLD Sampling Rate:
    bold_fs = 1 / TR

    matched_df = pd.DataFrame(columns=fp_df.columns)

    # Print number of volumes
    #print(f'Number of BOLD volumes to match: {num_vols}')

    # Loop through all signal columns and match to BOLD data
    for col in fp_df.columns:
        
        if col in corresponding_time_cols:
            
            # Only process normalized columns and TTL columns
            if 'norm' in col or 'ttl' in col or 'opto' in col:
                print(f'Processing column: {col}')
            
                signal_data = fp_df[col].dropna()
                if not col == 'ttl_TR' and not col == 'ttl_stim':
                    time_data = fp_df[corresponding_time_cols[col][0]].dropna()
                else:
                    print('SKIPPING ttl_TR column and/or ttl_stim column')
                    continue
                
                # 1) Low-pass first - cutoff at 0.8 x (1/(2xTR)) Hz
                if not col == 'opto_DIO1' and not col == 'opto_DIO2' and not col == 'opto_DIO4':
                    print('Applying low-pass filter...')
                    fs = get_sampling_rate(time_data)  # Sample rate
                    cutoff = cutoff_from_TR # Desired cutoff frequency
                    order = 6    # Filter order
                    filtered_data = butter_lowpass_filter(signal_data, cutoff, fs, order)
                else:
                    print('Skipping low-pass filter for opto channels...')
                    filtered_data = signal_data.to_numpy()

                # 2) Resample to BOLD time: bin-average into TR windows aligned to a chosen origin (here, 0s)
                bins = np.linspace(0, num_vols, len(filtered_data)+1) # This +1 is necessary for proper binning with the linspace command followed by rounding down to integers.
                bins = np.floor(bins).astype(int)
                bins = bins[:-1]  # Adjust length to match filtered_data
                down = pd.DataFrame({'bin': bins, 'fp': filtered_data}).groupby('bin', as_index=False)['fp'].mean() 
                t_tr = (np.arange(len(down))) * TR 
                fp_tr = down['fp'].to_numpy()
                                
                # Save to pd.DataFrame
                if matched_df[corresponding_time_cols[col][0]].empty or matched_df[corresponding_time_cols[col][0]].isna().all():
                    matched_df[corresponding_time_cols[col][0]] = t_tr
                matched_df[col] = fp_tr
                
                # plt.switch_backend('TkAgg')
                # plt.figure(figsize=(10, 6))
                # plt.plot(time_data-time_data.min(), signal_data, label=f'Original signal ({col})')
                # plt.plot(time_data-time_data.min(), filtered_data, label='Filtered signal (Low-pass)')
                # plt.plot(t_tr, fp_tr, label='Filtered Downsampled signal')
                # #plt.plot(bold_df['bold_time'], bold_df['bold_signal'])
                # plt.xlabel('Time (s)')
                # plt.ylabel('Amplitude')
                # plt.legend()
                # plt.grid(True)
                # plt.show()

            #else:
                #print(f'Current column {col} is not a normalized column - moving to next column.')
        
        #else:
            #print(f'Current column {col} is time column - moving to next signal column.')

    # Drop all columns that are na
    matched_df = matched_df.dropna(axis=1, how="all")

    return matched_df


def main(parq_file, num_vols, TR, corresponding_time_cols):
    '''
    This function merely properly handles fp files inputted from
    say snakemake pipelines to read them in and pass the expected
    dataframe to the match_fp_to_bold function.
    '''
    # Read in parquet file
    df = pd.read_parquet(parq_file)
    # Run matching function
    matched_df = match_fp_to_bold(df, num_vols, TR, corresponding_time_cols)
    return matched_df





if __name__ == '__main__':
    
    ################# TEST OUT LOW PASS FILTER ####################
    # Example usage:
    # fs = 1000.0  # Sample rate
    # cutoff = 50.0 # Desired cutoff frequency
    # order = 6    # Filter order
    # t = np.linspace(0, 1, int(fs), endpoint=False) # Time vector
    # data = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t) + np.random.randn(len(t)) * 0.5 # Example signal with low and high frequencies
    # filtered_data = butter_lowpass_filter(data, cutoff, fs, order)
    ###############################################################
    import sys
    import os
    sys.path.append("/cifs/menon/slaxer/dev/fp-code")
    sys.path.append("/Users/samlaxer/Documents/dev/fp-code")
    import fp_bold_eval
    import re
    import glob


    def corr_bold_ts_file(bids_dir, sub, ses, run):
        '''
        Assumes BIDS structure. Gets corresponding BOLD file using run number associated with FP file.
        '''
        pattern = f'{sub}/{ses}/func/{sub}*{ses}*{run}*avgROI.txt'
        bold_file = glob.glob(os.path.join(bids_dir, pattern))[0]
        return bold_file


    # Example input file
    #fp_file = '/cifs/menon/slaxer/data/ds-NcModfMRI/derivatives/simultaneous_fMRI-FP/local/input/sub-24Ear393838/ses-20230317/fp/sub-24Ear393838_ses-20230317_desc-FullScan_acq-0000_fp_norm_run-3.standardized.parquet'
    #bold_file = '/cifs/menon/slaxer/data/ds-NcModfMRI/derivatives/simultaneous_fMRI-FP/local/input/sub-24Ear393838/ses-20230317/func/sub-24Ear393838_ses-20230317_time-1502_run-3_ped-FW_task-rest_reconstruction-none_bold_RAS_combined_cleaned_avgROI.txt'
    fp_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/g1/procd/sub-50Ear436390/ses-20251101/fp/sub-50Ear436390_ses-20251101_desc-OptoStim_fp_norm_bandpass-01to1_run-3.standardized.parquet'
    #bold_file = '/cifs/menon/slaxer/data/ds-NcModfMRI/derivatives/simultaneous_fMRI-FP/local/input/sub-32Ear411599/ses-20240308/func/sub-32Ear411599_ses-20240308_time-1428_run-1_ped-FW_task-rest_reconstruction-none_bold_RAS_combined_cleaned_avgROI.txt' 
    
    sub = re.search(r"(sub-[A-Za-z0-9]+)", fp_file).group(1)
    ses = re.search(r"(ses-\d+)", fp_file).group(1)
    run = re.search(r"(run-\d+)", fp_file).group(1)
    
    #bids_dir = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/g1/procd/'
    #bold_TR = 1.2
    #bold_file = corr_bold_ts_file(bids_dir, sub, ses, run)

    if int(ses.replace("ses-", "")) > 20240401:
        num_vols = 500
        TR = 1.2
    else:
        num_vols = 400
        TR = 1.5 
    
    # Create object
    myobj = fp_bold_eval.EvalBOLDFP(fp_file, bold_TR = TR, num_vols = num_vols)

    fp_df = myobj.fp_df
    bold_df = myobj.bold_df
    num_vols = myobj.num_vols
    TR = myobj.bold_TR
    corr_time_cols = myobj.corresponding_time_cols

    matched_df = match_fp_to_bold(fp_df, bold_df, num_vols, TR, corr_time_cols)
    print(matched_df)

