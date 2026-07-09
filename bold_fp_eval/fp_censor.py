import pandas as pd


def fp_censor(fp_df, mask_file: str):
    '''
    Censors the FP data based on BOLD mask file.
    '''
    if mask_file == None:
        # If the mask file is None - meaning no mask file is inputted - keep the original dataframe
        censored_df = fp_df
    else:
        mask_file = pd.read_csv(mask_file, header=0, names=['mask'], dtype={'mask': bool})
        # At this point the fp_df should already be matched to BOLD data.
        # Assert that the mask has same number of data points as df
        assert(len(mask_file['mask']) == len(fp_df))
        # Now censor the data based on the mask file.
        censored_df = pd.DataFrame(columns=fp_df.columns)
        censored_df = fp_df[mask_file['mask'] == True].reset_index(drop=True)
    
    return censored_df

def main(parq_file, mask_file):
    # Read in parq_file
    fp_df = pd.read_parquet(parq_file)
    # Run censoring function
    censored_df = fp_censor(fp_df, mask_file)
    return censored_df