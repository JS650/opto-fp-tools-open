import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Optional, List

'''
This module contains functions to generate QA plots for FP data.
'''

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

# Create dictionary to hold colours for each column for consistency between plots
colour_dict = {
        "400-410nm_roi1": 'magenta',
        "400-410nm_roi2": 'lightgreen',
        "400-410nm_roi3": 'cyan',
        "460-490nm_roi1": 'red',
        "460-490nm_roi2": 'blue',
        "460-490nm_roi3": 'yellow',
        "555-570nm_roi1": 'brown',
        "555-570nm_roi2": 'orange',
        "555-570nm_roi3": 'purple',
        "minicube_405nm": 'lightblue',
        "minicube_465nm": 'pink',
        "ttl_TR": 'black',
        "ttl_stim": 'lightgray',
        "opto_DIO1": 'green',
        "opto_DIO2": 'magenta',
        "460-490nm_roi1_norm": 'red',
        "460-490nm_roi2_norm": 'blue',
        "460-490nm_roi3_norm": 'yellow',
        "555-570nm_roi1_norm": 'brown',
        "555-570nm_roi2_norm": 'orange',
        "555-570nm_roi3_norm": 'purple',
        "minicube_465nm_norm": 'pink',
    }



def normalize_qa_plot(pre_normalize_file: str, post_normalize_file: str, corresponding_time_cols: dict, columns: Optional[List[str]] = None, save_plot_fname = "") -> None:
    '''
    Generate QA plots for FP data. Plot the dataframe before and after
    normalization. Both plots should include the same channels for comparison.

    Parameters
    ----------
    pre_normalize_file : str
        Filename of the FP data before normalization.
    post_normalize_file : str
        Filename of the FP data after normalization.
    corresponding_time_cols : dict
        Mapping from signal column -> time column.
    columns : list, optional
        Which signal columns to plot. If None, plots all keys in corresponding_time_cols.
    '''
    # Load pre-normalize data
    pre_df = pd.read_parquet(pre_normalize_file)
    # Load post-normalize data
    post_df = pd.read_parquet(post_normalize_file)
    # Determine columns to plot
    if columns is None:
        columns = list(corresponding_time_cols.keys())
    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    # Plot pre-normalize data
    # Get max value across all columns to plot for scaling TTLs
    global_ymax_pre = 0
    for col in columns:
        if col in pre_df.columns and not pre_df[col].dropna().empty:
            col_max = pre_df[col].dropna().max()
            if col_max > global_ymax_pre:
                global_ymax_pre = col_max
    global_ymax_post = 0
    for col in columns:
        if col in post_df.columns and not post_df[col].dropna().empty and '_norm' in col:
            col_max = post_df[col].dropna().max()
            if col_max > global_ymax_post:
                global_ymax_post = col_max

    for col in columns:
        if col in pre_df.columns:
            if col == 'ttl_TR' or col == 'ttl_stim' or col == 'opto_DIO1' or col == 'opto_DIO2':
                # Scale TTL channels for visibility
                if not pre_df[col].dropna().empty:
                    axes[0].plot(pre_df[corresponding_time_cols[col][0]], pre_df[col]*global_ymax_pre , label=f'Scaled: {col}', alpha=0.2, color=colour_dict.get(col, None))
            else:
                axes[0].plot(pre_df[corresponding_time_cols[col][0]], pre_df[col], label=col, color=colour_dict.get(col, None))
    axes[0].set_title('Pre-Normalized Data: ' + os.path.basename(pre_normalize_file), loc='left', fontsize=10)
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    # Plot post-normalize data
    for col in columns:
        if col in post_df.columns and ('_norm' in col or col in ['ttl_TR', 'ttl_stim', 'opto_DIO1', 'opto_DIO2']):
            if col == 'ttl_TR' or col == 'ttl_stim' or col == 'opto_DIO1' or col == 'opto_DIO2':
                # Scale TTL channels for visibility
                if not post_df[col].dropna().empty:
                    axes[1].plot(post_df[corresponding_time_cols[col][0]], post_df[col]*global_ymax_post, label=f'Scaled: {col}', alpha=0.2, color=colour_dict.get(col, None))
            else:
                axes[1].plot(post_df[corresponding_time_cols[col][0]], post_df[col], label=col, color=colour_dict.get(col, None))
    axes[1].set_title('Post-Normalized Data: ' + os.path.basename(post_normalize_file), loc='left', fontsize=10)
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on right for legends
    if save_plot_fname == "":
        plt.show()
    else:
        plt.savefig(save_plot_fname)


#______________________________________________________________________________________________________________________________

def bandpass_qa_plot(pre_bandpass_file: str, post_bandpass_file: str, corresponding_time_cols: dict, columns: Optional[List[str]] = None, save_plot_fname = "") -> None:
    '''
    Generate QA plots for FP data. Plot the dataframe before and after
    bandpass filtering. Both plots should include the same channels for
    comparison.

    Parameters
    ----------
    pre_bandpass_file : str
        Filename of the FP data before bandpass filtering.
    post_bandpass_file : str
        Filename of the FP data after bandpass filtering.
    corresponding_time_cols : dict
        Mapping from signal column -> time column.
    columns : list, optional
        Which signal columns to plot. If None, plots all keys in corresponding_time_cols.
    '''
    # Load pre-bandpass data
    pre_df = pd.read_parquet(pre_bandpass_file)
    # Load post-bandpass data
    post_df = pd.read_parquet(post_bandpass_file)
    # Determine columns to plot
    if columns is None:
        columns = list(corresponding_time_cols.keys())
    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    # Plot pre-bandpass data
    for col in columns:
        if col in pre_df.columns and '_norm' in col:
            axes[0].plot(pre_df[corresponding_time_cols[col][0]], pre_df[col], label=col, color=colour_dict.get(col, None))
    axes[0].set_title('Pre-Bandpass Data: ' + os.path.basename(pre_bandpass_file), loc='left', fontsize=10)
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    # Plot post-bandpass data
    for col in columns:
        if col in post_df.columns and '_norm' in col:
            axes[1].plot(post_df[corresponding_time_cols[col][0]], post_df[col], label=col, color=colour_dict.get(col, None))
    axes[1].set_title('Post-Bandpass Data: ' + os.path.basename(post_bandpass_file), loc='left', fontsize=10)
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on right for legends
    if save_plot_fname == "":
        plt.show()
    else:
        plt.savefig(save_plot_fname)


#______________________________________________________________________________________________________________________________

def run_splitter_qa_plots(pre_runsplit_file: str, post_runsplit_files: List[str], corresponding_time_cols: dict, save_plot_fname = "") -> None:
    '''
    Generate QA plots for FP data. Plot the dataframe before and after
    splitting runs. In the first plot, ensure to include the TTL TR channel.
    There should be subplots. The first subplot is the full trace, and there
    additional subplots matching the number of runs found in the data.

    Parameters
    ----------
    pre_runsplit_file : str
        Filename of the FP data before run splitting.
    post_runsplit_files : List[str]
        List of filenames of the FP data after run splitting.
    corresponding_time_cols : dict
        Mapping from signal column -> time column.
    '''
    # Load pre-runsplit data
    pre_df = pd.read_parquet(pre_runsplit_file)
    # Load post-runsplit data
    post_dfs = [pd.read_parquet(f) for f in post_runsplit_files]
    # Determine number of runs
    num_runs = len(post_dfs)
    # Create subplots
    fig, axes = plt.subplots(num_runs + 1, 1, figsize=(12, 10), sharex=True)
    # Get min and max for y-limits. Only consider "460-490nm_roi1_norm", "460-490nm_roi2_norm", "460-490nm_roi3_norm",
    # "555-570nm_roi1_norm", "555-570nm_roi2_norm", "555-570nm_roi3_norm",
    # "minicube_465nm_norm"
    pre_cols_to_check = [
            "460-490nm_roi1_norm", "460-490nm_roi2_norm", "460-490nm_roi3_norm",
            "555-570nm_roi1_norm", "555-570nm_roi2_norm", "555-570nm_roi3_norm",
            "minicube_465nm_norm", "opto_DIO4", "opto_DIO1"
        ]
    pre_y_values = []
    for col in pre_cols_to_check:
        if col in pre_df.columns:
            pre_y_values.extend(pre_df[col].dropna().values)
    if pre_y_values:
        global_ymin = min(pre_y_values)
        global_ymax = max(pre_y_values)
    # Plot pre-runsplit data. Scale y-limits to global min/max
    axes[0].plot(pre_df[corresponding_time_cols['ttl_TR'][0]], pre_df['ttl_TR']*global_ymax, label='Scaled TTL TR', color=colour_dict.get('ttl_TR', None), alpha=0.2)
    for col in pre_df.columns:
        if '_norm' in col or 'opto' in col:
            if col in pre_cols_to_check:
                axes[0].plot(pre_df[corresponding_time_cols[col][0]], pre_df[col], label=col, color=colour_dict.get(col, None))
            elif 'opto' in col:
                axes[0].plot(pre_df[corresponding_time_cols[col][0]], pre_df[col]*global_ymax, color=colour_dict.get(col, None), label=f'Scaled: {col}', alpha=0.2)
    # Ensure title is centred taking into account legend

    axes[0].set_title('Pre-Run Split Data: ' + os.path.basename(pre_runsplit_file), loc='left', fontsize=10)
    # Set legend outside plot
    # Place the legend to the right of the axis
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    # Plot post-runsplit data
    for i, post_df in enumerate(post_dfs):
        axes[i + 1].plot(post_df[corresponding_time_cols['ttl_TR'][0]], post_df['ttl_TR']*global_ymax, label='Scaled TTL TR', color=colour_dict.get('ttl_TR', None), alpha=0.2)
        for col in post_df.columns:
            if col in pre_cols_to_check:
                axes[i + 1].plot(post_df[corresponding_time_cols[col][0]], post_df[col], label=col, color=colour_dict.get(col, None))
            elif 'opto' in col:
                axes[i + 1].plot(post_df[corresponding_time_cols[col][0]], post_df[col]*global_ymax, color=colour_dict.get(col, None), label=f'Scaled: {col}', alpha=0.2)
        axes[i + 1].set_title('Post split: ' + os.path.basename(post_runsplit_files[i]), loc='left', fontsize=10)
        axes[i + 1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    # Ensure all axes same x-limits and y-limits for comparison
    xlims = axes[0].get_xlim()
    ylims = axes[0].get_ylim()
    for ax in axes[1:]:
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on right for legends
    if save_plot_fname == "":
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_plot_fname), exist_ok=True)
        plt.savefig(save_plot_fname)


#______________________________________________________________________________________________________________________________


def match_fp_bold_qa_plot(pre_match_file: str, post_match_file: str, corresponding_time_cols: dict, columns: Optional[List[str]] = None, save_plot_fname = "") -> None:
    '''
    Generate QA plots for FP data. Plot the dataframe before and after
    matching to BOLD. Both plots should include the same channels for comparison.

    Parameters
    ----------
    pre_match_file : str
        Filename of the FP data before matching to BOLD.
    post_match_file : str
        Filename of the FP data after matching to BOLD.
    corresponding_time_cols : dict
        Mapping from signal column -> time column.
    columns : list, optional
        Which signal columns to plot. If None, plots all keys in corresponding_time_cols.
    '''
    # Load pre-match data
    pre_df = pd.read_parquet(pre_match_file)
    # Load post-match data
    post_df = pd.read_parquet(post_match_file)
    # Determine columns to plot
    if columns is None:
        columns = list(corresponding_time_cols.keys())
    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    pre_cols_to_check = [
            "460-490nm_roi1_norm", "460-490nm_roi2_norm", "460-490nm_roi3_norm",
            "555-570nm_roi1_norm", "555-570nm_roi2_norm", "555-570nm_roi3_norm",
            "minicube_465nm_norm"
        ]
    pre_y_values = []
    for col in pre_cols_to_check:
        if col in pre_df.columns:
            pre_y_values.extend(pre_df[col].dropna().values)
    if pre_y_values:
        global_ymin = min(pre_y_values)
        global_ymax = max(pre_y_values)
    # Plot pre-match data
    axes[0].plot(pre_df[corresponding_time_cols['ttl_TR'][0]], pre_df['ttl_TR']*global_ymax, label='Scaled TTL TR', color=colour_dict.get('ttl_TR', None), alpha=0.2)
    for col in columns:
        if col in pre_df.columns and col in post_df.columns:
            if col != 'ttl_TR' and col in pre_cols_to_check:
                axes[0].plot(pre_df[corresponding_time_cols[col][0]], pre_df[col], label=col, color=colour_dict.get(col, None))
            elif col in corresponding_time_cols and col != 'ttl_TR':
                axes[0].plot(pre_df[corresponding_time_cols[col][0]], pre_df[col]*global_ymax, color=colour_dict.get(col, None), label=f'Scaled: {col}', alpha=0.2)
    axes[0].set_title('Pre-Match Data: ' + os.path.basename(pre_match_file), loc='left', fontsize=10)
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    # Plot post-match data
    for col in columns:
        if col in pre_df.columns and col in post_df.columns:
            if col != 'ttl_TR' and col in pre_cols_to_check:
                axes[1].plot(post_df[corresponding_time_cols[col][0]], post_df[col], label=col, color=colour_dict.get(col, None))
            elif col in corresponding_time_cols and col != 'ttl_TR':
                axes[1].plot(post_df[corresponding_time_cols[col][0]], post_df[col]*global_ymax, label=f'Scaled: {col}', color=colour_dict.get(col, None), alpha=0.6)
    axes[1].set_title('Post-Match Data: ' + os.path.basename(post_match_file), loc='left', fontsize=10)
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on right for legends
    if save_plot_fname == "":
        plt.show()
    else:
        plt.savefig(save_plot_fname)


#______________________________________________________________________________________________________________________________


def convolve_qa_plot(pre_conv_file: str, post_conv_file: str, corresponding_time_cols: dict, columns: Optional[List[str]] = None, save_plot_fname = "") -> None:
    '''
    Generate QA plots for FP data. Plot the dataframe before and after
    matching to BOLD. Both plots should include the same channels for comparison.

    Parameters
    ----------
    pre_conv_file : str
        Filename of the FP data before matching to BOLD.
    post_conv_file : str
        Filename of the FP data after matching to BOLD.
    corresponding_time_cols : dict
        Mapping from signal column -> time column.
    columns : list, optional
        Which signal columns to plot. If None, plots all keys in corresponding_time_cols.
    '''
    # Load pre-match data
    pre_df = pd.read_parquet(pre_conv_file)
    # Load post-match data
    post_df = pd.read_parquet(post_conv_file)
    # Determine columns to plot
    if columns is None:
        columns = list(corresponding_time_cols.keys())
    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    pre_cols_to_check = [
            "460-490nm_roi1_norm", "460-490nm_roi2_norm", "460-490nm_roi3_norm",
            "555-570nm_roi1_norm", "555-570nm_roi2_norm", "555-570nm_roi3_norm",
            "minicube_465nm_norm"
        ]
    pre_y_values = []
    for col in pre_cols_to_check:
        if col in pre_df.columns:
            pre_y_values.extend(pre_df[col].dropna().values)
    if pre_y_values:
        global_ymin = min(pre_y_values)
        global_ymax = max(pre_y_values)
    # Plot pre-match data
    for col in columns:
        if col in pre_df.columns and col in post_df.columns:
            if col != 'ttl_TR' and col in pre_cols_to_check:
                axes[0].plot(pre_df[corresponding_time_cols[col][0]], pre_df[col], label=col, color=colour_dict.get(col, None))
            elif col in corresponding_time_cols and col != 'ttl_TR':
                axes[0].plot(pre_df[corresponding_time_cols[col][0]], pre_df[col]*global_ymax, label=f'Scaled: {col}', color=colour_dict.get(col, None), alpha=0.5)
    axes[0].set_title('Pre-Match Data: ' + os.path.basename(pre_conv_file), loc='left', fontsize=10)
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    # Plot post-match data
    for col in columns:
        if col in pre_df.columns and col in post_df.columns:
            if col != 'ttl_TR' and col in pre_cols_to_check:
                axes[1].plot(post_df[corresponding_time_cols[col][0]], post_df[col], label=col, color=colour_dict.get(col, None))
            elif col in corresponding_time_cols and col != 'ttl_TR':
                axes[1].plot(post_df[corresponding_time_cols[col][0]], post_df[col]*global_ymax, label=f'Scaled: {col}', color=colour_dict.get(col, None), alpha=0.5)
    # Set font size of title and legend
    axes[1].set_title('Post-Match Data: ' + os.path.basename(post_conv_file), loc='left', fontsize=10)
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on right for legends
    if save_plot_fname == "":
        plt.show()
    else:
        plt.savefig(save_plot_fname)


#______________________________________________________________________________________________________________________________


def censor_fp_qa_plot(pre_censor_file: str, post_censor_file: str, corresponding_time_cols: dict, censor_mask_file: str, columns: Optional[List[str]] = None, save_plot_fname = "") -> None:
    '''
    Generate QA plots for FP data. Plot the dataframe before and after
    censoring. Both plots should include the same channels for comparison.

    Parameters
    ----------
    pre_censor_file : str
        Filename of the FP data before censoring.
    post_censor_file : str
        Filename of the FP data after censoring.
    corresponding_time_cols : dict
        Mapping from signal column -> time column.
    censor_mask_file : str
        Filename of the censor mask used. It is a csv file with a single column
        with a header ("False = Masked Volumes"). Each row corresponds to a timepoint
        and is either True (not censored) or False (censored).
    columns : list, optional
        Which signal columns to plot. If None, plots all keys in corresponding_time_cols.
    '''
    # Load pre-censor data
    pre_df = pd.read_parquet(pre_censor_file)
    # Load post-censor data
    post_df = pd.read_parquet(post_censor_file)
    # Load censor mask (csv file)
    if os.path.exists(censor_mask_file):
        censor_mask = pd.read_csv(censor_mask_file)
    else:
        censor_mask = None
    # Determine columns to plot
    if columns is None:
        columns = list(corresponding_time_cols.keys())
    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    pre_cols_to_check = [
            "460-490nm_roi1_norm", "460-490nm_roi2_norm", "460-490nm_roi3_norm",
            "555-570nm_roi1_norm", "555-570nm_roi2_norm", "555-570nm_roi3_norm",
            "minicube_465nm_norm"
    ]
    pre_y_values = []
    for col in pre_cols_to_check:
        if col in pre_df.columns:
            pre_y_values.extend(pre_df[col].dropna().values)
    if pre_y_values:
        global_ymin = min(pre_y_values)
        global_ymax = max(pre_y_values)
    # Plot pre-censor data
    for col in columns:
        if col in pre_df.columns and col in post_df.columns:
            if col != 'ttl_TR' and col in pre_cols_to_check:
                axes[0].plot(pre_df[corresponding_time_cols[col][0]], pre_df[col], label=col, color=colour_dict.get(col, None))
            elif col in corresponding_time_cols and col != 'ttl_TR':
                axes[0].plot(pre_df[corresponding_time_cols[col][0]], pre_df[col]*global_ymax, label=f'Scaled: {col}', color=colour_dict.get(col, None), alpha=0.5)
    # Add vertical lines for censored timepoints
    if censor_mask is not None:
        for idx, censored in enumerate(~censor_mask.iloc[:, 0]):
            if censored:
                timepoint = pre_df[corresponding_time_cols['ttl_TR'][0]].iloc[idx]
                axes[0].axvline(x=timepoint, color='red', linestyle='--', alpha=0.3)
    axes[0].set_title('Pre-Censor Data: ' + os.path.basename(pre_censor_file), loc='left', fontsize=10)
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    # Plot post-censor data
    for col in columns:
        if col in pre_df.columns and col in post_df.columns:
            if col != 'ttl_TR' and col in pre_cols_to_check:
                axes[1].plot(post_df[corresponding_time_cols[col][0]], post_df[col], label=col, color=colour_dict.get(col, None))
            elif col in corresponding_time_cols and col != 'ttl_TR':
                axes[1].plot(post_df[corresponding_time_cols[col][0]], post_df[col]*global_ymax, label=f'Scaled: {col}', color=colour_dict.get(col, None), alpha=0.5)
    axes[1].set_title('Post-Censor Data: ' + os.path.basename(post_censor_file), loc='left', fontsize=10)
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    # Add anotation on plot to indicate number of surviving timepoints after censoring
    # Get number of surviving timepoints from length of post_df and compare to length of pre_df
    num_surviving = len(post_df)
    total_timepoints = len(pre_df)
    axes[1].text(0.95, 0.95, f'Surviving timepoints: {num_surviving}/{total_timepoints}',
                    transform=axes[1].transAxes, ha='right', va='top', fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on right for legends
    if save_plot_fname == "":
        plt.show()
    else:
        plt.savefig(save_plot_fname)

#______________________________________________________________________________________________________________________________



if __name__ == '__main__':
    
    # # # Normalization QA plot
    pre_file = '/nfs/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/bulk_analysis/bulk_procd/sub-51Ear435511/ses-20260201/fp/sub-51Ear435511_ses-20260201_run-2_desc-modPowAndFreqAnes_fp.standardized.parquet'
    post_file = '/nfs/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/bulk_analysis/bulk_procd/sub-51Ear435511/ses-20260201/fp/sub-51Ear435511_ses-20260201_run-2_desc-modPowAndFreqAnes_fp_norm.standardized.parquet'
    normalize_qa_plot(pre_file, post_file, corresponding_time_cols)

    # #______________________________________________________________________________________________________________________________
    
    # # # Bandpass QA plot
    # pre_file = '/Volumes/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/bulk_analysis/bulk_procd/sub-51Ear435511/ses-20260201/fp/sub-51Ear435511_ses-20260201_run-2_desc-modPowAndFreqAnes_fp_norm.standardized.parquet'
    # post_file = '/Volumes/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/bulk_analysis/bulk_procd/sub-51Ear435511/ses-20260201/fp/sub-51Ear435511_ses-20260201_run-2_desc-modPowAndFreqAnes_fp_norm_bandpass-01to1.standardized.parquet'
    # bandpass_qa_plot(pre_file, post_file, corresponding_time_cols)
    
    # # #______________________________________________________________________________________________________________________________
    
    # # # Run Splitter QA plot
    # pre_file = '/Volumes/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/bulk_analysis/bulk_procd/sub-38Ear422848/ses-20251123/fp/sub-38Ear422848_ses-20251123_desc-OptoNegControlRuns3and4_acq-0010_fp_norm.standardized.parquet'
    # post_files = [
    #     '/Volumes/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/bulk_analysis/bulk_procd/sub-38Ear422848/ses-20251123/fp/sub-38Ear422848_ses-20251123_desc-OptoNegControlRuns3and4_acq-0010_fp_norm_run-3.standardized.parquet',
    #     '/Volumes/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/bulk_analysis/bulk_procd/sub-38Ear422848/ses-20251123/fp/sub-38Ear422848_ses-20251123_desc-OptoNegControlRuns3and4_acq-0010_fp_norm_run-4.standardized.parquet'
    #     ]
    # run_splitter_qa_plots(pre_file, post_files, corresponding_time_cols)
    
    # #______________________________________________________________________________________________________________________________

    # # # Match to BOLD QA plot
    # pre_file = '/Volumes/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/highpow_test/procd/sub-54Ear437095/ses-20260306/fp/sub-54Ear437095_ses-20260306_run-1_desc-HigherOptoPower5mW_acq-0000_fp_norm_run-1.standardized.parquet'
    # post_file = '/Volumes/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/highpow_test/procd/sub-54Ear437095/ses-20260306/fp/sub-54Ear437095_ses-20260306_run-1_desc-HigherOptoPower5mW_acq-0000_fp_norm_run-1_matched.standardized.parquet'
    # match_fp_bold_qa_plot(pre_file, post_file, corresponding_time_cols)

    # #______________________________________________________________________________________________________________________________

    # # # Convolve QA plot
    # pre_file = '/Volumes/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/highpow_test/procd/sub-54Ear437095/ses-20260306/fp/sub-54Ear437095_ses-20260306_run-1_desc-HigherOptoPower5mW_acq-0000_fp_norm_bandpass-01to1_run-1_matched.standardized.parquet'
    # post_file = '/Volumes/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/highpow_test/procd/sub-54Ear437095/ses-20260306/fp/sub-54Ear437095_ses-20260306_run-1_desc-HigherOptoPower5mW_acq-0000_fp_norm_bandpass-01to1_run-1_matched_conv.standardized.parquet'
    # convolve_qa_plot(pre_file, post_file, corresponding_time_cols)
    
    # #______________________________________________________________________________________________________________________________

    # # Censor QA plot
    # pre_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/anes_modPowAndFreq/procd/sub-48Ear436389/ses-20260131/fp/sub-48Ear436389_ses-20260131_run-3_desc-modPowAndFreqAnes_fp_norm_run-3_matched.standardized.parquet'
    # post_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/anes_modPowAndFreq/procd/sub-48Ear436389/ses-20260131/fp/sub-48Ear436389_ses-20260131_run-3_desc-modPowAndFreqAnes_fp_norm_run-3_matched_censored.standardized.parquet'
    # censored_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/anes_modPowAndFreq/confcorr_smooth/confound_correction_datasink/frame_censoring_mask/_split_name_sub-48Ear436389_ses-20260131_time-1326_run-3_ped-FW_task-rest_reconstruction-none_bold/sub-48Ear436389_ses-20260131_time-1326_run-3_ped-FW_task-rest_reconstruction-none_bold_RAS_resampled_frame_censoring_mask.csv'
    # censor_fp_qa_plot(pre_file, post_file, corresponding_time_cols, censored_file)

