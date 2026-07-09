import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq
import logging
import numpy as np
# Add locations of modules to system path
import sys
#repo_path = "/nfs/menon/jlaxer2/dev/fp-code"
repo_path = "/Volumes/menon/jlaxer2/dev/fp-code"
sys.path.insert(0, repo_path)
sys.path.append(repo_path+'/fp_processing')
sys.path.append(repo_path+'/bold_fp_eval')
sys.path.append(repo_path+'/QA')
# Import fp_processing scripts
from fp_standardization import hdf5_to_parquet
from bold_fp_eval import fp_runsplitter, fp_dividebyruns, match_fp_to_bold, fp_hemodynamicize, fp_censor, stim_regressor, plot_regressor
from QA import qa_fp_plots

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def extract_regressor(input_file: str, tr: float, num_vols: int, output_file, hrf: str = None, censor_file: str = None, plot_reg: bool = False):
    '''
    This function takes in a doric file, and outputs a regressor file to be used in GLM analysis. This file represents which fmri volumes
    correspond with the optogenetic stimulation.

    Steps:
    1) Change to .parquet file format (easier to manipulate underlying data than .doric files)
    2) Splices the regressor such that it only spans the duration of the run. 
    Uses the recorded TTL pulses from the scanner.
    3) Resample optogenetic output trace to the fMRI time. Uses TR for this.
    4) Optionally, convolve the optogenetic trace with an HRF. Specify "None" if
    you don't want any convolution.
    5) Optionally censor the optogenetic trace with a censor file. This should be a .csv
    with a single column where each row represents a volume. True indicates the volume should
    be kept, False indicates it should be masked. Leaving as None returns the full optogenetic
    trace without any masking.
    6) Save the optogenetic regressor. User has the option to include a column of 1s as the
    intercept in the GLM or to leave as a one column regressor of just the optogenetic trace.
    7) Optionally plot the regressor for visualization purposes.


    Input Arguments:
    - input_file    - Full path of .doric file with optogenetic and TTL recording.
    - TR            - Corresponding fMRI TR (seconds). Determines resampling factor as the 
                    regressor must match the time spacing of the fMRI volumes.
    - num_vols      - Number of volumes acquired in the fMRI run. Helps to determine how to splice
                    the optogenetic trace.
    - output_file   - Full path of .txt file to name regressor
    - hrf           - User can choose to use "glover" HRF, "spm" HRF or "None" which determines
                    if the regressor is convolved with an HRF (in the case of glover and spm) or
                    if the raw box regressor is outputted.
    - censor_file   - User can include the full path to a censor file defining which fMRI volumes were
                    censored and which were kept. This then removes the corresponding regressor values.
                    If None is inputted, the full regressor is kept and no censoring is done.
    - plot_reg      - User can choose whether or not to display a plot of the regressor for visualization
                    purposes.
    '''

    # RESOURCES
    #========================================================================================================
    STANDARD_COLUMNS = [
        "400-410nm_time", "400-410nm_roi1", "400-410nm_roi2", "400-410nm_roi3",
        "460-490nm_time", "460-490nm_roi1", "460-490nm_roi2", "460-490nm_roi3",
        "555-570nm_time", "555-570nm_roi1", "555-570nm_roi2", "555-570nm_roi3",
        "ttl_time", "ttl_TR", "ttl_stim", "opto_DIO1", "opto_DIO2", "opto_DIO3", "opto_DIO4", "minicube_405nm", "minicube_465nm", "minicube_time",
        "time_norm", "460-490nm_roi1_norm", "460-490nm_roi2_norm", "460-490nm_roi3_norm",
        "555-570nm_roi1_norm", "555-570nm_roi2_norm", "555-570nm_roi3_norm",
        "minicube_465nm_norm"
    ]
    conversion_csv_file = '/Volumes/menon/jlaxer2/data/ds-NcModfMRIOpto/code/header_conversions.csv'
    CORRESPONDING_TIME_COLS = {
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
        "opto_DIO3": ['ttl_time'],
        "opto_DIO4": ['ttl_time'],
        "460-490nm_roi1_norm": ['time_norm'],
        "460-490nm_roi2_norm": ['time_norm'],
        "460-490nm_roi3_norm": ['time_norm'],
        "555-570nm_roi1_norm": ['time_norm'],
        "555-570nm_roi2_norm": ['time_norm'],
        "555-570nm_roi3_norm": ['time_norm'],
        "minicube_465nm_norm": ['time_norm']
    }
    #========================================================================================================



    # STEP 1: Convert to .parquet file format
    #========================================================================================================
    logger.info(f"STEP 1: Converting file from .doric to .parquet")
    input_hdf5_file = input_file
    output_qa_dir = os.path.join(os.path.dirname(input_hdf5_file), f"{os.path.basename(input_hdf5_file.replace('.doric',''))}_QA")
    if not os.path.exists(output_qa_dir):
        os.makedirs(output_qa_dir, exist_ok=True)
    output_parquet_file = os.path.join(output_qa_dir, os.path.basename(input_hdf5_file.replace('.doric', '.standardized.parquet')))
    if not os.path.exists(output_parquet_file):
        hdf5_to_parquet.hdf5_to_parquet(
            standard_columns=STANDARD_COLUMNS,
            input_file=input_hdf5_file,
            output_file=output_parquet_file,
            conversion_file=conversion_csv_file,
            chunk_size=10000
        )
    #========================================================================================================

    # STEP 2: Splice recording to correspond to run duration
    #========================================================================================================
    logger.info(f"STEP 2: Splice recording to run duration")
    # Read in the parquet file
    opto_df = pd.read_parquet(output_parquet_file)

    # Non-bandpassed basename
    base_outfilename = output_parquet_file.replace(f'.standardized.parquet', '')
    output_plot_filename = os.path.join(base_outfilename + '_IdentifiedPeaksAndRuns.png')

    # Determine where the run splits are
    run_splits = fp_runsplitter.fp_runsplitter(opto_df, tr, num_vols, output_plot_filename)
    # run_splits_bp = fp_runsplitter.fp_runsplitter(fp_df_bp, TR, num_vols, output_plot_filename_bp)

    # Generate multiple DataFrames from the determined run splits
    run_dfs = fp_dividebyruns.fp_dividebyruns(opto_df, run_splits, CORRESPONDING_TIME_COLS)

    # Get the starting run-<label> for this file
    start_run_id = 'run-1'
    idx = int(start_run_id.split('-')[-1])

    # Save each run DataFrame to a new Parquet file
    out_files = []
    for run_df_idx in range(len(run_dfs)):
        # Get current run df
        curr_run_df = run_dfs[run_df_idx]
        # curr_run_df_bp = run_dfs_bp[run_df_idx]
        # Generate output filename with correct run number
        run_output_file = f"{base_outfilename}_run-{idx}.standardized.parquet"
        # run_output_file_bp = f"{base_outfilename_bp}_run-{idx}.standardized.parquet"
        # Save run dfs
        curr_run_df.to_parquet(run_output_file)
        # curr_run_df_bp.to_parquet(run_output_file_bp)
        logger.debug(f"Saved run-{idx} to: {run_output_file}")# AND {run_output_file_bp}")
        # Increment run #
        idx += 1
        # Save output filenames for QA assessment
        out_files.append(run_output_file)
    
    # QA
    out_plot_fname = f"{output_parquet_file.replace(f'.standardized.parquet', '_runsplits.png')}"
    qa_fp_plots.run_splitter_qa_plots(output_parquet_file, out_files, CORRESPONDING_TIME_COLS, out_plot_fname)
    #========================================================================================================

    # STEP 3: Resample optogenetic output trace to the fMRI time
    #========================================================================================================
    logger.info(f"STEP 3: Resample optogenetic output trace to the fMRI time")
    # Resample
    matched_df = match_fp_to_bold.main(out_files, num_vols, tr, CORRESPONDING_TIME_COLS)
    # Save matched DataFrame to parquet file
    table = pa.Table.from_pandas(matched_df)
    matched_file = output_parquet_file.replace('.standardized.parquet', '_matched.standardized.parquet')
    pq.write_table(table, matched_file)
    #========================================================================================================

    # STEP 4: Convolve signal with HRF (Optional)
    #========================================================================================================
    logger.info(f"STEP 4: Convolve signal with HRF (if selected)")
    # Convolve fp using HRF
    conv_fp_df = fp_hemodynamicize.hemodynamicize(matched_file, CORRESPONDING_TIME_COLS, hrf_model=hrf)
    # Save convolved fp parquet file
    table = pa.Table.from_pandas(conv_fp_df)
    conv_file = matched_file.replace('.standardized.parquet', '_conv.standardized.parquet')
    pq.write_table(table, conv_file)
    #========================================================================================================

    # STEP 5: Censor regressor accordingly with censored fMRI volumes
    #========================================================================================================
    logger.info(f"STEP 5: Censoring regressor according to censored fMRI volumes (if selected)")
    # Get mask file
    censored_df = fp_censor.main(conv_file, censor_file)
    # Save convolved fp parquet file
    table = pa.Table.from_pandas(censored_df)
    censor_file = conv_file.replace('.standardized.parquet', '_censor.standardized.parquet')
    pq.write_table(table, censor_file)
    #========================================================================================================
    
    # STEP 6: Save regressor as test file
    #========================================================================================================
    logger.info(f"STEP 6: Saving regressor to text file")
    data_dict = stim_regressor.main(censor_file, CORRESPONDING_TIME_COLS)

    # Save regressors and contrast to text files
    for key in data_dict.keys():
        # Regressors
        if 'regressor' in key:
            np.savetxt(output_file, data_dict[key], fmt="%.4f")
        elif 'contrast' in key:
            contrast_file = output_file.replace('.txt', '_con.txt')
            np.savetxt(contrast_file, data_dict[key], fmt="%d")
    #========================================================================================================

    # STEP 7: Optionally plot the regressor for visualization purposes.
    #========================================================================================================
    logger.info(f"STEP 7: Plotting regressor for visualization purposes (if selected)")
    if plot_reg:
        plot_regressor.plot_regressor(output_file, TR=tr, show=True)
    #========================================================================================================


if __name__ == "__main__":
    input_file = '/Volumes/menon/jlaxer2/data/FP/Pierre/20260706_Scan/sub-violet_ses-20260630_run-1_0004.doric'
    tr = 1.5
    num_vols = 600
    output_file = '/Users/samlaxer/Desktop/tmp/test_marm_regressor2.txt'
    plot_reg = True

    extract_regressor(input_file, tr, num_vols, output_file, plot_reg=plot_reg)


