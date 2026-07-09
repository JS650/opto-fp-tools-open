import numpy as np
from scipy.signal import convolve
from nilearn.glm.first_level import glover_hrf, spm_hrf
import matplotlib.pyplot as plt
import pandas as pd

def hemodynamicize_signal(signal, time=None, dt=None, model="glover"):
    """
    Convolve a time series with a canonical HRF (Glover or SPM).

    Parameters
    ----------
    signal : 1D array-like
        Input signal (e.g. pandas Series or numpy array).
    time : 1D array-like, optional
        Time vector in seconds. If not provided, dt must be given.
    dt : float, optional
        Sampling interval (seconds per sample). Required if time is None. In general should be TR.
    model : str, optional
        Which HRF to use ("glover" or "spm").

    Returns
    -------
    convolved : np.ndarray
        The signal convolved with the HRF, truncated to original length.
    time : np.ndarray
        Time vector in seconds.

    The Glover HRF nicely represents the mouse HRF (https://pmc.ncbi.nlm.nih.gov/articles/PMC10493293/#SM2)
    so we will default to use that as our model here.
    """
    
    signal = np.asarray(signal)

    # Determine dt and time
    if time is not None:
        time = np.asarray(time)
        print(time)
        dt = np.mean(np.diff(time))  # infer sampling step
    elif dt is not None:
        time = np.arange(len(signal)) * dt
    else:
        raise ValueError("You must provide either a time vector or dt (sampling interval).")

    # Generate HRF at correct resolution
    if model == "glover":
        hrf = glover_hrf(dt, oversampling=1)
    elif model == "spm":
        hrf = spm_hrf(dt, oversampling=1)
    else:
        raise ValueError("model must be 'glover' or 'spm'")

    # Convolve
    convolved = convolve(signal, hrf)[:len(signal)]

    # # Plot
    # plt.switch_backend('TkAgg')
    # plt.figure()
    # plt.plot(time, signal, label="Original Signal")
    # plt.plot(time, convolved, label=f"Convolved with {model} HRF")
    # plt.legend()
    # plt.xlabel("Time (s)")
    # plt.ylabel("Signal")
    # plt.title("Hemodynamic Convolution")
    # plt.show()

    return convolved, time


def hemodynamicize(parq_file, corresponding_time_cols, hrf_model="glover"):
    # Read in parquet file
    fp_df = pd.read_parquet(parq_file)
    # If hrf_model is None, return same dataframe
    if hrf_model is None:
        return fp_df
    # Get the run splits from the TR TTL pulses
    hemodynamicize_df = pd.DataFrame(columns=fp_df.columns)
    for col in fp_df.columns:
        if col in corresponding_time_cols:
            signal = fp_df[col]
            time = fp_df[corresponding_time_cols[col][0]]
            # Add convolved signal to appropriate column
            convolved_signal, conv_time = hemodynamicize_signal(signal, time, dt=None, model=hrf_model)
            hemodynamicize_df[col] = convolved_signal
        else:
            hemodynamicize_df[col] = fp_df[col]
    return hemodynamicize_df


if __name__ == '__main__':
    # FOR TESTING HRFs
    hrf1 = glover_hrf(1.2, oversampling=1)   # canonical HRF sampled at TR
    hrf2 = spm_hrf(1.2, oversampling=1)
    plt.figure()
    plt.plot(hrf1, label='glover (TR = 1.2)')
    plt.plot(hrf2, label='spm (TR = 1.2)')
    plt.legend()
    plt.show()
    
    import sys
    sys.path.append("/cifs/menon/slaxer/dev/fp-code")
    #import fp_bold_eval
    from bold_fp_eval import cross_corr
    from fp_processing import fp_plot

    corresponding_time_cols = {
            "400-410nm_roi1": ['400-410nm_time'],
            "400-410nm_roi2": ['400-410nm_time'],
            "400-410nm_roi3": ['400-410nm_time'],
            "460-490nm_roi1": ['460-490nm_time'],
            "460-490nm_roi2": ['460-490nm_time'],
            "460-490nm_roi3": ['460-490nm_time'],
            "555-570nm_roi1": ['555-570nm_time'],
            "555-570nm_roi2": ['555-570nm_time'],
            "555-570nm_roi3": ['555-570nm_time'],
            "minicube_405nm": ['minicube_time'],
            "minicube_465nm": ['minicube_time'],
            "ttl_TR": ['ttl_time'],
            "ttl_stim": ['ttl_time'],
            "opto_DIO1": ['ttl_time'],
            "opto_DIO2": ['ttl_time'],
            "460-490nm_roi1_norm": ['time_norm'],
            "460-490nm_roi2_norm": ['time_norm'],
            "460-490nm_roi3_norm": ['time_norm'],
            "555-570nm_roi1_norm": ['time_norm'],
            "555-570nm_roi2_norm": ['time_norm'],
            "555-570nm_roi3_norm": ['time_norm'],
            "minicube_465nm_norm": ['time_norm']
        }

    # Example input file
    #fp_file = '/cifs/menon/slaxer/data/ds-NcModfMRI/derivatives/simultaneous_fMRI-FP/local/input/sub-24Ear393838/ses-20230317/fp/sub-24Ear393838_ses-20230317_desc-FullScan_acq-0000_fp_norm_run-3.standardized.parquet'
    #bold_file = '/cifs/menon/slaxer/data/ds-NcModfMRI/derivatives/simultaneous_fMRI-FP/local/input/sub-24Ear393838/ses-20230317/func/sub-24Ear393838_ses-20230317_time-1502_run-3_ped-FW_task-rest_reconstruction-none_bold_RAS_combined_cleaned_avgROI.txt'
    #fp_file = '/nfs/menon/jlaxer2/data/ds-NcModfMRI/derivatives/simultaneous_fMRI-FP/local/input/sub-32Ear411599/ses-20240308/fp/sub-32Ear411599_ses-20240308_FullScan_0002_fp_norm_run-1.standardized.parquet'
    #bold_file = '/cifs/menon/slaxer/data/ds-NcModfMRI/derivatives/simultaneous_fMRI-FP/local/input/sub-32Ear411599/ses-20240308/func/sub-32Ear411599_ses-20240308_time-1428_run-1_ped-FW_task-rest_reconstruction-none_bold_RAS_combined_cleaned_avgROI.txt' 

    fp_file = '/nfs/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/lowRes_modPowAndFreq/procd/sub-46Ear435509/ses-20260327/fp/sub-46Ear435509_ses-20260327_run-1_desc-lowResfMRI_acq-0021_fp_norm_bandpass-01to1_run-1_matched.standardized.parquet'
    
    #bold_TR = 1.5
    # Create object
    #myobj = fp_bold_eval.EvalBOLDFP(fp_file, bold_file, bold_TR)
    #matched_df = myobj.match_fp_bold()
    #signal = matched_df['minicube_465nm_norm']
    #time = matched_df[myobj.corresponding_time_cols['minicube_465nm_norm'][0]]
    #convolved_signal, convolved_time = hemodynamicize(signal, time, bold_TR)
    # Hemodynamicize signal
    conv_df = hemodynamicize(fp_file, corresponding_time_cols)

    fp_plot.fp_plot(conv_df, corresponding_time_cols)

    # Get cross correlation of original signal, then convolved signal
    #cross_corr.cross_corr(myobj.bold_df['bold_signal'], signal, fp_file.replace('.standardized.parquet', '_origSignal.png'))
    #cross_corr.cross_corr(myobj.bold_df['bold_signal'], convolved_signal, fp_file.replace('.standardized.parquet', '_convolvedSignal.png'))

    


