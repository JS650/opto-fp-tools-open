# Only load in GUI for of matplotlib if running script directly.
if __name__ == '__main__':
    import matplotlib.pyplot as plt
else:
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
import numpy as np
from typing import List


def fp_runsplitter(df: pd.DataFrame, TR: int, num_vols: int, output_plot: str) -> List[pd.DataFrame]:
    '''
    Split up the FP trace based on the recorded TTL pulses.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the FP data with a 'ttl_TR' column.
    TR : int
        Time resolution of the data in seconds.
    output_plot: str
        Output filename for plot showing found peaks and division of runs.
    fp_info_file: str, optional
        Optional file containing additional FP info especially for getting correct run-<id>.

    Returns
    -------
    List of times, each representing a split run.
    
    Created 2025-07-28 by Sam Laxer
    '''


    # Parameters for minicube system
    # height = 4, distance = 500, peak_counter >= 50, 
    # For the BFTO system:
    # height = 1, distance = 50 or 100, peak_counter >= 5,

    # Check if the TTL column has non NaN values
    if 'ttl_TR' not in df.columns or df['ttl_TR'].isna().all():
        raise ValueError('The DataFrame does not contain a valid "ttl_TR" column. Please ensure the DataFrame has a "ttl_TR" column with non-NaN values.')
    # Find the index of each TTL pulse
    # Find the sampling rate using the time channel:
    if np.isnan(df['ttl_time']).all() and not np.isnan(df['minicube_time']).all():
        samplespersec = 1/np.mean(np.diff(df['minicube_time'][~np.isnan(df['minicube_time'])])) # This is the number of points acquired per sec. We can set this value divided by 2 as the distance away each peak has to be for it to be counted
    elif not np.isnan(df['ttl_time']).all() and np.isnan(df['minicube_time']).all():
        samplespersec = 1/np.mean(np.diff(df['ttl_time'][~np.isnan(df['ttl_time'])]))
    # If using BFTO system, invert ttl trace ********************************************************
    #*********************
    ttls = -df['ttl_TR']#*
    #*********************
    peak_idxs,peak_heights = find_peaks(ttls, height=np.nanmax(ttls)/1.5, threshold=None, distance=samplespersec/2, prominence=None, width=None, wlen=None, rel_height=None, plateau_size=None)
    # Identify the times of each peak
    if np.isnan(df['ttl_time']).all() and not np.isnan(df['minicube_time']).all(): # Determine if data is from old FP system (minicube) or new (BFTO)
        peak_times = df['minicube_time'][peak_idxs]
    elif not np.isnan(df['ttl_time']).all() and np.isnan(df['minicube_time']).all():
        peak_times = df['ttl_time'][peak_idxs]
    else:
        print('Both "ttl_time" and "minicube_time" exist. Not sure which one to use.... This shouldn''t be possible since the minicube system should only '
        'use "minicube_time" and the BFTO system should only use "ttl_time".')
    # Find the last peak in each run
    peak_counter = 0
    last_peak_idx = [] # list to hold all indeces representing last peak of each run
    first_peak_idx = [] # list to hold all indeces representing first peak of each run
    for idx, peak_idx in enumerate(peak_idxs):
        # Criteria for peak being last peak in run:
        # 1) It is not the first peak seen
        # 2) There are at least num_vol peaks counted before it (note it could be greater than num_vol peaks due to setup pulses)
        # 3) There is a decent time gap after it (say 4 seconds)
        # EXCEPTION: If it is the last peak index AND there are many peaks counted before it (say 50), then we will count the current peak as the final peak of the last run.
        peak_counter += 1
        if peak_idx > peak_idxs[0] and peak_counter >= num_vols and peak_times[peak_idx] - peak_times[prev_peak_idx] >= 4: # Case for last peak in run (not the final peak in the entire trace)
            last_peak_idx = last_peak_idx + [prev_peak_idx]
            # Get the first peak in each run
            # ASSUMING THAT THE LENGTH OF ALL RUNS IN THIS SESSION ARE THE SAME LENGTH
            # Find first peak by going back num_vols peaks from the previous peak
            first_peak_idx = first_peak_idx + [peak_idxs[idx-num_vols]]
            # Restart peak counter once found last peak of run
            peak_counter = 0 
        elif peak_idx == peak_idxs[-1] and peak_counter >= 50: # Case for last peak in entire trace
            last_peak_idx = last_peak_idx + [peak_idx]
            # Get the first peak in each run
            # ASSUMING THAT THE LENGTH OF ALL RUNS IN THIS SESSION ARE THE SAME LENGTH
            first_peak_idx = first_peak_idx + [peak_idxs[idx-num_vols]]
        prev_peak_idx = peak_idx # Save current peak to be used as previous peak for next iteration

    # If TR is not provided, default to mode of derivative of TTL pulses
    if TR is None:
        #TR = df['minicube_time'][peak_idxs].diff().mode().iloc[0] if not np.isnan(df['minicube_time']).all() else df['ttl_time'][peak_idxs].diff().mode().iloc[0]
        if not np.isnan(df['minicube_time']).all():
            diffs = df['minicube_time'][peak_idxs].diff()
        else:
            diffs = df['ttl_time'][peak_idxs].diff()

        mode_result = diffs.mode()

        if not mode_result.empty:
            TR = mode_result.iloc[0]
        else:
            print(df['minicube_time'])
            print(df['ttl_time'])
            raise ValueError('Unable to determine TR from data')

        print(f"Using TR: {TR} seconds (derived from derivative of minicube_time or ttl_time)")

    # Now we must splice the dataframe based on the first and last peak indices, but include 1 TR after each last peak index
    # The split will start num_vols TRs before the last peak index and end at the last peak index + 1 TR
    run_splits_times = [] # Holds a list of times when each run ends.
    for idx, _ in enumerate(first_peak_idx):
        start_time = df['minicube_time'][first_peak_idx[idx]] if not np.isnan(df['minicube_time']).all() else df['ttl_time'][first_peak_idx[idx]]
        end_time = df['minicube_time'][last_peak_idx[idx]] + TR if not np.isnan(df['minicube_time']).all() else df['ttl_time'][last_peak_idx[idx]] + TR
        if end_time - start_time >= (num_vols * TR)-1 and end_time - start_time <= (num_vols * TR)+1: # Ensure that the run length is approximately correct (plus/minus 1 second)
            run_splits_times.append([start_time, end_time])
        else:
            print(f'WARNING: Run split {idx} has unexpected length of {end_time - start_time} seconds. Expected length is {num_vols * TR} seconds. Moving start time accordingly.')
            print(f'Old start time is {start_time} seconds.')
            # Adjust start time to ensure correct length
            new_approx_start_time = end_time - (num_vols * TR)
            # Get closest peak time to new_approx_start_time
            peak_diffs = np.abs(peak_times - new_approx_start_time)
            closest_peak_idx = np.argmin(peak_diffs)
            new_first_peak = peak_idxs[closest_peak_idx]
            new_start_time = df['minicube_time'][new_first_peak] if not np.isnan(df['minicube_time']).all() else df['ttl_time'][new_first_peak]
            run_splits_times.append([new_start_time, end_time])
            print(f'New start time is {new_start_time} seconds.')
    
    # Plot original trace
    if not np.isnan(df['minicube_time']).all():
        time = df['minicube_time'].astype(float)
    elif not np.isnan(df['ttl_time']).all():
        time = df['ttl_time'].astype(float)
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, ttls, label='Original Signal')
    
    # Mark peaks
    plt.plot(time[peak_idxs], ttls[peak_idxs], 'ro', label='Detected Peaks')
    
    # Mark first peak of each run
    plt.plot(time[first_peak_idx], ttls[first_peak_idx], 'go', label='First Peak of Run')

    # Mark last peak of each run
    plt.plot(time[last_peak_idx], ttls[last_peak_idx], 'bo', label='Last Peak of Run')
    
    # Add vertical lines for each run split
    for i, (run_start, run_end) in enumerate(run_splits_times):
        plt.axvline(x=run_start, color='k', linestyle='-', label='Run Start' if i == 0 else "")
        plt.axvline(x=run_end, color='k', linestyle='--', label='Run End' if i == 0 else "")

    plt.title('Signal with Detected Peaks')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    if output_plot:
        plt.savefig(output_plot, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()

    # Return the run splits
    return run_splits_times


if __name__ == "__main__": # FOR TESTING
    #parquet_file = '/Users/samlaxer/Desktop/tmpFP/BBC300_Acq_34Ear_AfterSetup_0002_out.standardized.parquet'
    #parquet_file = '/cifs/menon/slaxer/data/ds-NcModfMRI/derivatives/simultaneous_fMRI-FP/local/input/sub-17Ear392625/ses-20221219/fp/sub-17Ear392625_ses-20221219_desc-FullScan_acq-0000_fp_norm.standardized.parquet'
    #parquet_file = '/nfs/menon/slaxer/data/ds-NcModfMRI/derivatives/simultaneous_fMRI-FP/local/input/sub-32Ear411599/ses-20240308/fp/sub-32Ear411599_ses-20240308_FullScan_0002_fp_norm.standardized.parquet'
    #parquet_file = '/cifs/menon/slaxer/data/ds-NcModfMRI/derivatives/simultaneous_fMRI-FP/local/input/sub-35Ear418711/ses-20241011/fp/sub-35Ear418711_ses-20241011_ScanDay_0001_fp_norm.standardized.parquet'
    #parquet_file = '/cifs/menon/slaxer/data/ds-NcModfMRI/derivatives/simultaneous_fMRI-FP/local/input/sub-34Ear414207/ses-20241116/fp/sub-34Ear414207_ses-20241116_AfterSetup_0002_fp_norm.standardized.parquet'
    #parquet_file = '/nfs/menon/slaxer/data/FP/20250817_ScanDay/sub-41Ear432059_ses-20250817_BilateralFP_0001_norm.standardized.parquet'
    #parquet_file = '/nfs/menon/slaxer/data/ds-NcModfMRI/derivatives/simultaneous_fMRI-FP/local/input/sub-14Ear390791/ses-20221126/fp/sub-14Ear390791_ses-20221126_desc-Scan_acq-0000_fp_norm_bandpass.standardized.parquet'
    #parquet_file = '/nfs/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/anes_modPowAndFreq/procd/sub-52Ear435510/ses-20260131/fp/sub-52Ear435510_ses-20260131_run-1_desc-modPowAndFreqAnes_fp_norm.standardized.parquet'
    #parquet_file = '/nfs/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/lowRes_modPowAndFreq/procd/sub-51Ear435511/ses-20260327/fp/sub-51Ear435511_ses-20260327_run-1_desc-lowResfMRI_acq-0014_fp_norm.standardized.parquet'
    parquet_file = '/nfs/menon/jlaxer2/data/ds-NcModfMRIOptoX2/derivatives/awake_proc/procd/sub-59Ear443016/ses-20260701/fp/sub-59Ear443016_ses-20260701_run-1_desc-Awake_acq-0026_fp_norm.standardized.parquet'
    df = pd.read_parquet(parquet_file)
    # num_vols like either 400 (9.4 T protocol) or 500 (15.2 T protocol)
    num_vols = 500
    # TR likely either 1.5 (9.4 T protocol) or 1.2 (15.2 T protocol)
    TR = 1.2

    run_split_times = fp_runsplitter(df, TR, num_vols, output_plot=None)
    print(run_split_times)

