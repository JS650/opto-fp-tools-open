# /Users/samlaxer/Documents/dev/fp-code/bold_fp_eval/stim_regressor.py
import os
import numpy as np
import pandas as pd

def stim_regressor(signal: np.ndarray, time_vector: np.ndarray, output_file: str, intercept_col: bool = True, threshold: float = 0.2, binarise: bool = False):
    """
    Generate a stimulus regressor from an output signal.

    Parameters
    ----------
    signal : np.ndarray
        The signal array.
    time_vector : np.ndarray
        The corresponding time vector.
    output_file : str
        Full path with filename to save the stimulus regressor text file.
    intercept_col : bool
        Whether to include an intercept column of ones. Default is True. 
        Recommended for GLM analyses since it accounts for baseline shifts if the
        data is not mean-centered.
    threshold : float
        Threshold to detect stimulus onsets. Verify with your data based on
        min/max values of opto stimulation trace. The threshold should be somewhere
        in the middle.
    binarise : bool
        Whether to binarise the regressor based on the threshold. Default is False,
        since GLM can handle continuous regressors and binarising may throw away
        useful information about the strength of the stimulation. However, if you
        want to detect onsets and create a binary regressor, set binarise to True.
    """
    # Detect onsets where signal fluctuating at 40 Hz
    # Determine if any value in the next 5 seconds exceeds the threshold
    if binarise:
        reg_data = signal >= threshold
    else:
        reg_data = signal

    if intercept_col:
        # Add a last column of ones
        ones_column = np.ones_like(reg_data, dtype=int)
        output_matrix = np.column_stack((reg_data, ones_column))
        # Also make contrast file
        contrast_matrix = np.array([[1, 0]])
    else:
        output_matrix = reg_data
        contrast_matrix = np.array([[1]])

    return output_matrix, contrast_matrix


def save_stim_files(data_dict, output_opto_DIO1, output_opto_DIO2, output_roi1, output_roi2):
    for key in data_dict.keys():
        # Regressors
        if 'DIO1' in key:
            if 'regressor' in key:
                np.savetxt(output_opto_DIO1, data_dict[key], fmt="%.4f")
            elif 'contrast' in key:
                contrast_file = output_opto_DIO1.replace('.txt', '_con.txt')
                np.savetxt(contrast_file, data_dict[key], fmt="%d")
        elif 'DIO2' in key:
            if 'regressor' in key:
                np.savetxt(output_opto_DIO2, data_dict[key], fmt="%.4f")
            elif 'contrast' in key:
                contrast_file = output_opto_DIO2.replace('.txt', '_con.txt')
                np.savetxt(contrast_file, data_dict[key], fmt="%d")
        elif 'roi1' in key:
            if 'regressor' in key:
                np.savetxt(output_roi1, data_dict[key], fmt="%.4f")
            elif 'contrast' in key:
                contrast_file = output_roi1.replace('.txt', '_con.txt')
                np.savetxt(contrast_file, data_dict[key], fmt="%d")
        elif 'roi2' in key:
            if 'regressor' in key:
                np.savetxt(output_roi2, data_dict[key], fmt="%.4f")
            elif 'contrast' in key:
                contrast_file = output_roi2.replace('.txt', '_con.txt')
                np.savetxt(contrast_file, data_dict[key], fmt="%d")
        #print(f"Stimulus regressor saved to {output_file} with {np.sum(reg_data)} stimulation timepoints.")


def main(parq_file, corresponding_time_cols):
    fp_data = pd.read_parquet(parq_file)
    try:
        fp_data['opto_DIO1']
        fp_data['opto_DIO2']
        fp_data['ttl_time']
    except:
        try:
            fp_data['opto_DIO4']
            fp_data['ttl_time']
        except:
            raise ValueError('No opto_DIO1 or opto_DIO2 columns found in FP data.')
    # Generate stim file
    data_dict = {}
    for col in fp_data.columns:
        if col in corresponding_time_cols:
            time_col = corresponding_time_cols[col][0]
            signal_data = fp_data[col]
            time_data = fp_data[time_col]
            out_stim_file = parq_file.replace('.standardized.parquet', f'_{col}_regressor.txt')

            # LIKELY SHOULDN'T BINARISE ANY COLUMNS. SINCE EVEN OPTO SHOULD BE CONVOLVED WITH HRF,
            # AND GLM DOES NOT REQUIRE BINARIZED REGRESSORS, IT CAN HANDLE CONTINUOUS REGRESSORS.
            # if 'opto' in col:
            #     binarise = True
            # else:
            #     binarise = False
            binarise = False
            regressor, contrast_matrix = stim_regressor(signal_data.to_numpy(),  time_data.to_numpy(), out_stim_file, binarise=binarise)
            data_dict[f'{col}_regressor'] = regressor
            data_dict[f'{col}_contrast'] = contrast_matrix
    
    # Return all regressor and contrast matrices. User can save them to e.g., text files later
    return data_dict


if __name__ == "__main__":
    # Example usage
    # import pandas as pd
    # input_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/anes_modPowAndFreqAndRes/procd/sub-52Ear435510/ses-20260216/fp/sub-52Ear435510_ses-20260216_run-7_desc-modPowAndFreqAnesRun7and8_fp_norm_run-8_matched.standardized.parquet'
    # fp_df = pd.read_parquet(input_file)
    # signal1 = fp_df['opto_DIO1'].values
    # signal2 = fp_df['opto_DIO2'].values
    # time_vector1 = fp_df['ttl_time'].values
    # time_vector2 = fp_df['ttl_time'].values
    # output_file1 = "/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/anes_modPowAndFreqAndRes/" + os.path.basename(input_file).replace('.standardized.parquet', '_DIO1_regressor.txt')
    # output_file2 = "/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/anes_modPowAndFreqAndRes/" + os.path.basename(input_file).replace('.standardized.parquet', '_DIO2_regressor.txt')
    # stim_regressor(signal1, time_vector1, output_file1, intercept_col=False, threshold=0.2, binarise=True)
    # stim_regressor(signal2, time_vector2, output_file2, intercept_col=False, threshold=0.2, binarise=True)

    #===================================================================================================================
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
    input_file = '/nfs/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/bulk_analysis/bulk_procd/sub-52Ear435510/ses-20260216/fp/sub-52Ear435510_ses-20260216_desc-modPowAndFreqAnesRun3and4_fp_norm_run-3_matched.standardized.parquet'
    data_dict = main(input_file, corresponding_time_cols)
    print(data_dict.keys())
