import glob
import os
import subprocess


def fileskimmer(directory: str, file_ext: str = ''):
    '''
    fileskimmer finds and returns all the files within directory that match the 
    string pattern given by file_ext.

    directory -> Str
    file_ext -> Str
    '''
    # Use glob
    files = glob.glob(os.path.join(directory, '**', '*' + file_ext), recursive=True)
    #print(os.path.join(directory, '**', '*' + file_ext))
    return files


def merged_file(mydir: str, mergedpath: str):
    '''
    This function will merge all the cope files in a given directory into a single
    4D NIfTI file using fslmerge.
    Parameters
    ----------
    mydir : str
        Full path to directory containing individual cope NIfTI files. Expects files
        to have suffix 'cope.nii.gz'.
    mergedpath : str
        Full path to save the merged cope NIfTI file.
    '''
    # Get string of all images in directory
    files = fileskimmer(mydir, 'cope.nii.gz')
    filestr = ' '.join(files)

    # Merge images
    if not os.path.exists(mergedpath):
        # Prior components: 5 = Somatomotor, 12 = Visual, 19 = DMN
        os.system('fslmerge -t ' + mergedpath + ' ' + filestr)
    else:
        print(f'Merged file {mergedpath} already exists. Skipping fslmerge.')
    return mergedpath


def group_map(merged_cope_file: str, 
              group_mask_file: str,
              design_mat_file: str,
              contrast_file: str,
              cs_file: str,
              output_dir: str,
              runmode: str = 'flame1'):
    '''
    This function will run FSL FLAMEO for group-level analysis using the
    provided merged cope file and design matrix.
    The cope file should be generated using fsl_merge on individual run
    cope files.
    Parameters
    ----------
    merged_cope_file : str
        Full path to the merged cope NIfTI file.
    group_mask_file : str
        Full path to the group-level brain mask NIfTI file.
    design_mat_file : str
        Full path to the design matrix file (Text2Vest).
    contrast_file : str
        Full path to the contrast file (Text2Vest).
    cs_file : str
        Full path to the covariance structure file (Text2Vest).
    output_dir : str
        Full path to directory where the FLAMEO output will be saved.
    runmode : str, optional
        FLAMEO run mode, by default 'flame1'
    '''
    cmd = [
        'flameo',
        f'--cope={merged_cope_file}',
        f'--mask={group_mask_file}',
        f'--dm={design_mat_file}',
        f'--ld={output_dir}',
        f'--tc={contrast_file}',
        f'--cs={cs_file}',
        f'--runmode={runmode}'
    ]
    subprocess.run(cmd, check=True)
    print(f'FLAMEO group analysis completed. Results saved in {output_dir}')


if __name__ == '__main__':
    # Example usage
    merged_cope_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/g2_withControl/procd2/flameo/DIO1/all_merged_DIO1.nii.gz'
    group_mask_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/g2_withControl/procd2/flameo/group_mask.nii.gz'
    design_mat_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/g2_withControl/procd2/flameo/group_design.mat'
    contrast_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/g2_withControl/procd2/flameo/group_contrasts.con'
    cs_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/g2_withControl/procd2/flameo/group_cs.mat'
    output_dir = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/g2_withControl/procd2/flameo/DIO1/flameo_DIO1_test'
    
    group_map(merged_cope_file,
              group_mask_file,
              design_mat_file,
              contrast_file,
              cs_file,
              output_dir,
              runmode='flame1')

