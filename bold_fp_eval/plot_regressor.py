import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import os

def plot_regressor(txt_file, TR: float, show: bool = True, outfile: str = ''):
    """
    Efficient plotting of FP/opto regressor data.
    Parameters
    ----------
    txt_file : str or list of str
        Full path(s) to stimulus regressor text file(s).
    TR : float
        Repetition time (TR) of the fMRI acquisition in seconds.
    show : bool
        If True, displays the plot interactively.
    outfile : str
        If provided, saves the plot to this file instead of showing it.
    """
    if isinstance(txt_file, str):
        txt_file = [txt_file]

    fig, ax = plt.subplots(figsize=(12, 6))

    for fpath in txt_file:
        data = np.loadtxt(fpath)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        # Drop columns that are entirely ones
        all_ones = np.all(data == 1, axis=0)
        data = data[:, ~all_ones]
        if data.shape[1] == 0:
            continue
        time = np.arange(data.shape[0]) * TR
        label = os.path.basename(fpath)
        ax.plot(time, data, linewidth=2, label=label)

    ax.set_title("Regressor Plot")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    if len(txt_file) > 1:
        ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()

    if show:
        plt.show()
    elif outfile:
        plt.savefig(outfile)

    return fig


if __name__ == '__main__':
    #file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/g1/procd/sub-48Ear436389/ses-20251101/fp/sub-48Ear436389_ses-20251101_desc-OptoStim_fp_norm_bandpass-01to1_run-1_matched_conv_censored_460-490nm_roi1_norm_regressor.txt'
    #file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/anes_modPowAndFreqAndRes/tmp_feat_manual/sub-52Ear435510_ses-20260216_run-4_DIO2_regressor_censor.txt'
    # file = '/nfs/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/lowRes_modPowAndFreq/procd2/sub-47Ear435508/ses-20260403/fp/sub-47Ear435508_ses-20260403_run-1_desc-lowResfMRI_acq-0002_fp_norm_bandpass-01to1_run-1_matched_460-490nm_roi1_norm_regressor.txt'
    #file = '/nfs/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/lowRes_modPowAndFreq/procd2/sub-47Ear435508/ses-20260403/fp/sub-47Ear435508_ses-20260403_run-1_desc-lowResfMRI_acq-0002_fp_norm_bandpass-01to1_run-1_matched_conv_censored_460-490nm_roi1_norm_regressor.txt'
    #file = '/nfs/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/lowRes_modPowAndFreq/procd2/sub-47Ear435508/ses-20260403/fp/sub-47Ear435508_ses-20260403_run-1_desc-lowResfMRI_acq-0002_fp_norm_run-1_matched_460-490nm_roi1_norm_regressor.txt'
    #file = '/nfs/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/lowRes_modPowAndFreq/procd2/sub-47Ear435508/ses-20260403/fp/sub-47Ear435508_ses-20260403_run-1_desc-lowResfMRI_acq-0002_fp_norm_run-1_matched_censored_460-490nm_roi1_norm_regressor.txt'
    
    files = [
        '/nfs/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/lowRes_modPowAndFreq/procd3/sub-46Ear435509/ses-20260327/fp/sub-46Ear435509_ses-20260327_run-1_glmfiles/sub-46Ear435509_ses-20260327_run-1_desc-lowResfMRI_acq-0021_fp_norm_bandpass-01to1_run-1_matched_conv_censored_460-490nm_roi1_norm_regressor.txt',
        '/nfs/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/lowRes_modPowAndFreq/procd3/sub-46Ear435509/ses-20260327/fp/sub-46Ear435509_ses-20260327_run-1_glmfiles/sub-46Ear435509_ses-20260327_run-1_desc-lowResfMRI_acq-0021_fp_norm_bandpass-01to1_run-1_matched_conv_censored_opto_DIO2_regressor.txt',
        '/nfs/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/lowRes_modPowAndFreq/procd3/sub-46Ear435509/ses-20260327/fp/sub-46Ear435509_ses-20260327_run-1_glmfiles/sub-46Ear435509_ses-20260327_run-1_desc-lowResfMRI_acq-0021_fp_norm_bandpass-01to1_run-1_matched_460-490nm_roi1_norm_regressor.txt',
        '/nfs/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/lowRes_modPowAndFreq/procd3/sub-46Ear435509/ses-20260327/fp/sub-46Ear435509_ses-20260327_run-1_glmfiles/sub-46Ear435509_ses-20260327_run-1_desc-lowResfMRI_acq-0021_fp_norm_bandpass-01to1_run-1_matched_opto_DIO2_regressor.txt'
    ]
    TR = 1.2 # seconds
    plot_regressor(files, TR=TR, show=True)

