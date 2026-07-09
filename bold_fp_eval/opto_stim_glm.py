# from nipype.interfaces import fsl

# def opto_stim_glm(bold_file: str, stim_regressor_file: str, stim_contrast_file: str, output_file: str, overwrite: bool = False):
#     '''
#     This function will run FSL GLM using optogenetic stimulation times (i.e.,
#     from the stim_regressor.py output) as the main regressor of interest. That
#     function optionally includes an intercept column.
#     '''
#     # If the stim_regressor_file is None, return None
#     if stim_regressor_file == '':
#         print('No stim regressor file provided. Exiting GLM function and returning None.')
#         return None
    
#     glm = fsl.GLM()
#     glm.inputs.in_file = bold_file                           # 4D BOLD NIfTI
#     glm.inputs.design = stim_regressor_file                  # text file design matrix
#     glm.inputs.contrasts = stim_contrast_file                # contrast file
#     glm.inputs.out_file = output_file                        # output file
#     glm.inputs.out_res_name = output_file.replace('.nii.gz', '_residuals.nii.gz')                 # residuals
#     glm.inputs.out_varcb_name = output_file.replace('.nii.gz', '_variance.nii.gz')                # variance
#     glm.inputs.out_cope = output_file.replace('.nii.gz', '_cope.nii.gz')                      # contrast of parameter estimate
#     glm.inputs.out_z_name = output_file.replace('.nii.gz', '_zstat.nii.gz')                        # Z statistic
#     glm.inputs.demean = True                               # demean columns

#     res = glm.run()
#     #print(res.outputs)


# if __name__ == '__main__':
#     # # Example usage
#     # bold_file = '/Users/samlaxer/Documents/Projects/ds-003464/derivatives/confcorr/confound_correction_datasink/cleaned_timeseries/_split_name_sub-jgroptoINS501_ses-2_task-opto_run-1_bold/sub-jgroptoINS501_ses-2_task-opto_run-1_bold_RAS_resampled_cleaned.nii.gz'
#     # stim_regressor_file = '/Users/samlaxer/Documents/Projects/ds-003464/derivatives/feat/sub-jgroptoINS518_ses-2_task-opto_events_1D.txt'
#     # #stim_contrast_file = stim_regressor_file.replace('.txt', '_con.txt')
#     # stim_contrast_file = '/Users/samlaxer/Documents/Projects/ds-003464/derivatives/feat/contrast.txt'
#     # output_file = '/Users/samlaxer/Documents/Projects/ds-003464/derivatives/feat/test_opto_glm_confcorr.nii.gz'
#     # opto_stim_glm(bold_file, stim_regressor_file, stim_contrast_file, output_file, overwrite=True)

#     # # Example usage
#     # bold_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/anes_modPowAndFreqAndRes/confcorr_lowres/confound_correction_datasink/cleaned_timeseries/_split_name_sub-52Ear435510_ses-20260216_time-1420_run-4_ped-FW_task-rest_reconstruction-none_bold/sub-52Ear435510_ses-20260216_time-1420_run-4_ped-FW_task-rest_reconstruction-none_bold_RAS_resampled_cleaned.nii.gz'
#     # stim_regressor_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/anes_modPowAndFreqAndRes/tmp_feat_manual/sub-52Ear435510_ses-20260216_run-4_DIO2_regressor.txt'
#     # #stim_contrast_file = stim_regressor_file.replace('.txt', '_con.txt')
#     # stim_contrast_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/anes_modPowAndFreqAndRes/tmp_feat_manual/contrasts.txt'
#     # output_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/anes_modPowAndFreqAndRes/tmp_feat_manual/opto_glm_sub52_run4_DIO2.nii.gz'
#     # opto_stim_glm(bold_file, stim_regressor_file, stim_contrast_file, output_file, overwrite=True)

#     # Example usage
#     bold_file = '/Volumes/menon/jlaxer2/data/ds-003464/derivatives/feat/test_glm/rand_run/sub-jgroptoINS546_ses-1_task-opto_run-2_bold_RAS_resampled_cleaned.nii.gz'
#     stim_regressor_file = '/Volumes/menon/jlaxer2/data/ds-003464/derivatives/feat/test_glm/rand_run/opto_events_1D.txt'
#     #stim_contrast_file = stim_regressor_file.replace('.txt', '_con.txt')
#     stim_contrast_file = '/Volumes/menon/jlaxer2/data/ds-003464/derivatives/feat/test_glm/rand_run/contrast.txt'
#     output_file = '/Volumes/menon/jlaxer2/data/ds-003464/derivatives/feat/test_glm/rand_run/opto_glm_sub546_ses1_run2.nii.gz'
#     opto_stim_glm(bold_file, stim_regressor_file, stim_contrast_file, output_file, overwrite=True)


#-------------------------------------------------------------------------------

from networkx import volume
from nipype.interfaces import fsl
import os
import subprocess


def _run(cmd: list):
    '''Run a shell command, printing it first, and raise on non-zero exit.'''
    print('Running: ' + ' '.join(str(c) for c in cmd))
    print(' ')
    print(cmd)
    print(' ')
    result = subprocess.run(cmd, check=True, capture_output=False)
    return result


def opto_stim_glm(
    bold_file: str,
    stim_regressor_file: str,
    stim_contrast_file: str,
    output_file: str,
    mask_file: str = '',
    tr: float = 1.2,
    cluster_z_thresh: float = 2.3,
    cluster_p_thresh: float = 0.05,
    overwrite: bool = False,
):
    '''
    Run FSL GLM using optogenetic stimulation times as the main regressor of
    interest, then perform GRF-based cluster correction on the resulting
    Z-statistic map.

    The GLM is called directly via subprocess rather than through nipype's
    fsl.GLM wrapper, which has known issues with output-path handling and
    trait validation across FSL versions.

    Parameters
    ----------
    bold_file : str
        Path to the 4-D BOLD NIfTI file.
    stim_regressor_file : str
        Path to the text-file design matrix produced by stim_regressor.py.
        Pass an empty string to skip the GLM entirely.
    stim_contrast_file : str
        Path to the FSL contrast (.con / .txt) file.
    output_file : str
        Base output path (must end in .nii.gz).  All derived outputs share
        this stem, mirroring the naming convention already in use.
    mask_file : str, optional
        Brain mask to restrict the GLM and cluster correction.
    tr : float, optional
        Repetition time in seconds.  Used by smoothest to estimate image
        smoothness for GRF cluster correction.  Default 1.2 s.
    cluster_z_thresh : float, optional
        Voxel-wise Z threshold that seeds the cluster-forming step.
        Default 2.3 (≈ p < 0.01 one-tailed), matching FSL FEAT defaults.
    cluster_p_thresh : float, optional
        Cluster-level family-wise error (FWE) p-value threshold.
        Default 0.05, matching FSL FEAT defaults.
    overwrite : bool, optional
        Not currently enforced but kept for API compatibility.

    Returns
    -------
    None  (outputs written to disk as NIfTI files alongside output_file)

    Output files
    ------------
    <stem>.nii.gz                  – GLM parameter estimates (beta maps)
    <stem>_residuals.nii.gz        – model residuals
    <stem>_variance.nii.gz         – variance of COPEs
    <stem>_cope.nii.gz             – contrast of parameter estimates (COPE)
    <stem>_zstat.nii.gz            – voxel-wise Z-statistic map
    <stem>_cluster_mask.nii.gz     – binary cluster-corrected mask
    <stem>_cluster_index.nii.gz    – integer-labelled cluster index image
    <stem>_cluster_localmax.txt    – cluster local-maxima table
    <stem>_cluster_zstat.nii.gz    – Z-stat map zeroed outside surviving clusters
    '''

    # ------------------------------------------------------------------
    # Guard: no regressor → nothing to do
    # ------------------------------------------------------------------
    if stim_regressor_file == '':
        print('No stim regressor file provided. Exiting GLM function and returning None.')
        return None
 
    # Resolve everything to absolute paths up front
    output_file         = os.path.abspath(output_file)
    bold_file           = os.path.abspath(bold_file)
    stim_regressor_file = os.path.abspath(stim_regressor_file)
    stim_contrast_file  = os.path.abspath(stim_contrast_file)
    if mask_file:
        mask_file = os.path.abspath(mask_file)
 
    out_dir = os.path.dirname(output_file)
    stem    = output_file.replace('.nii.gz', '')
    os.makedirs(out_dir, exist_ok=True)
 
    # ------------------------------------------------------------------
    # 1. FSL GLM  (called directly — bypasses nipype trait/path bugs)
    # ------------------------------------------------------------------
    res_file      = f'{stem}_residuals.nii.gz'
    variance_file = f'{stem}_variance.nii.gz'
    cope_file     = f'{stem}_cope.nii.gz'
    zstat_file    = f'{stem}_zstat.nii.gz'
 
    # Note: this FSL version requires = syntax for long-form options,
    # e.g. --out_res=<file> rather than --out_res <file>
    glm_cmd = [
        'fsl_glm',
        '-i', bold_file,
        '-d', stim_regressor_file,
        '-c', stim_contrast_file,
        '-o', output_file,
        f'--out_res={res_file}',
        f'--out_varcb={variance_file}',
        f'--out_cope={cope_file}',
        f'--out_z={zstat_file}',
        '--demean',
    ]
    if mask_file:
        glm_cmd += ['-m', mask_file]
 
    _run(glm_cmd)
 
    # ------------------------------------------------------------------
    # 2. Cluster correction on the Z-statistic map
    #    Mirrors the GRF-based correction used in FSL FEAT:
    #      a) Estimate image smoothness from the residuals (smoothest)
    #      b) Run cluster with --dlh and --volume from smoothest output
    # ------------------------------------------------------------------
 
    # --- 2a. Smoothness estimation via nipype (returns dlh + volume) ---
    # DOF = n_timepoints - n_regressors, required by SmoothEstimate when
    # residual_fit_file is used instead of a Z-stat-only input.
    import nibabel as nib
    import numpy as np
    n_timepoints = nib.load(bold_file).shape[-1]
    design_matrix = np.loadtxt(stim_regressor_file)
    n_regressors = design_matrix.shape[1] if design_matrix.ndim > 1 else 1
    dof = int(n_timepoints - n_regressors)

    print(f"n_timepoints: {n_timepoints}, n_regressors: {n_regressors}, dof: {dof}")
 
    smooth = fsl.SmoothEstimate()
    smooth.inputs.residual_fit_file = res_file
    smooth.inputs.dof               = dof
    if mask_file:
        smooth.inputs.mask_file = mask_file

    #smooth_res = smooth.run(cwd=out_dir)
    smooth_res = smooth.run()

    print(f"smooth_res.outputs: {smooth_res.outputs}")
 
    dlh    = smooth_res.outputs.dlh     # resels-per-voxel
    volume = smooth_res.outputs.volume  # search volume in voxels

    if dlh is None or volume is None:
        raise ValueError(
            f"SmoothEstimate failed to produce dlh/volume outputs.\n"
            f"residual_fit_file: {res_file}\n"
            f"dof: {dof}\n"
            f"outputs: {smooth_res.outputs}"
        )

    print(f"dlh: {dlh}, volume: {volume}")
 
    # --- 2b. GRF cluster correction via subprocess ---
    #FSL_CLUSTER = os.path.join(os.environ['FSLDIR'], 'share', 'fsl', 'bin', 'cluster')
    #FSL_CLUSTER = os.path.join(os.environ['FSLDIR'], 'bin', 'cluster')
    FSL_CLUSTER = '/srv/software/fsl/6.0.4/bin/cluster'

    # --- Positive tail (Z > +2.3) ---
    cluster_pos_cmd = [
        FSL_CLUSTER,
        '--in=' + zstat_file,
        '--thresh=' + str(cluster_z_thresh),
        '--pthresh=' + str(cluster_p_thresh / 2),
        '--dlh=' + str(dlh),
        '--volume=' + str(volume),
        '--oindex=' + f'{stem}_cluster_index_pos.nii.gz',
        '--olmax=' + f'{stem}_cluster_localmax_pos.txt',
        '--osize=' + f'{stem}_cluster_mask_pos.nii.gz',
        '--othresh=' + f'{stem}_cluster_zstat_pos.nii.gz',
    ]
    # if mask_file: # Maybe some versions of FSL don't support masking? Was giving errors
    #     cluster_pos_cmd += ['--mask', mask_file]
    _run(cluster_pos_cmd)

    # --- Negative tail ---
    zstat_neg_file = f'{stem}_zstat_neg.nii.gz'
    _run(['fslmaths', zstat_file, '-mul', '-1', zstat_neg_file])

    cluster_neg_cmd = [
        FSL_CLUSTER,
        '--in=' + zstat_neg_file,
        '--thresh=' + str(cluster_z_thresh),
        '--pthresh=' + str(cluster_p_thresh / 2),
        '--dlh=' + str(dlh),
        '--volume=' + str(volume),
        '--oindex=' + f'{stem}_cluster_index_neg.nii.gz',
        '--olmax=' + f'{stem}_cluster_localmax_neg.txt',
        '--osize=' + f'{stem}_cluster_mask_neg.nii.gz',
        '--othresh=' + f'{stem}_cluster_zstat_neg.nii.gz',
    ]
    # if mask_file: # Maybe some versions of FSL don't support masking? Was giving errors
    #     cluster_neg_cmd += ['--mask', mask_file]
    _run(cluster_neg_cmd)

    # Re-negate surviving negative-tail Z map back to negative values
    _run(['fslmaths', f'{stem}_cluster_zstat_neg.nii.gz', '-mul', '-1',
        f'{stem}_cluster_zstat_neg.nii.gz'])

    # Combine both tails
    _run(['fslmaths', f'{stem}_cluster_zstat_pos.nii.gz',
        '-add', f'{stem}_cluster_zstat_neg.nii.gz',
        f'{stem}_cluster_zstat_posAndNeg.nii.gz'])

    # cluster = fsl.Cluster()
    # cluster.inputs.in_file               = zstat_file
    # cluster.inputs.threshold             = cluster_z_thresh
    # cluster.inputs.pthreshold            = cluster_p_thresh
    # cluster.inputs.dlh                   = dlh
    # cluster.inputs.volume                = volume
    # cluster.inputs.out_threshold_file    = cluster_zstat_file
    # cluster.inputs.out_index_file        = cluster_index_file
    # cluster.inputs.out_localmax_txt_file = cluster_localmax_file
    # cluster.inputs.out_size_file         = cluster_mask_file
    # cluster.inputs.use_mm                = True
 
    # cluster.run(cwd=out_dir)
 
    print(
        f'GLM complete.\n'
        f'  Z-stat map        : {zstat_file}\n'
        f'  Cluster Z map     : {stem}_cluster_zstat_posAndNeg.nii.gz\n'
        f'  Parameters        : dlh={dlh}, volume={volume}, cluster_z_thresh={cluster_z_thresh}, cluster_p_thresh={cluster_p_thresh}\n'
    )
    # Save this info to a text file alongside the outputs for easy reference
    with open(f'{stem}_cluster_correction_info.txt', 'w') as f:
        f.write(
            f'GLM complete.\n'
            f'Z-stat map        : {zstat_file}\n'
            f'Cluster Z map     : {stem}_cluster_zstat_posAndNeg.nii.gz\n'
            f'Parameters        : dlh={dlh}, volume={volume}, cluster_z_thresh={cluster_z_thresh}, cluster_p_thresh={cluster_p_thresh}\n'
        )



if __name__ == '__main__':
    # # Example usage
    # bold_file = '/Users/samlaxer/Documents/Projects/ds-003464/derivatives/confcorr/confound_correction_datasink/cleaned_timeseries/_split_name_sub-jgroptoINS501_ses-2_task-opto_run-1_bold/sub-jgroptoINS501_ses-2_task-opto_run-1_bold_RAS_resampled_cleaned.nii.gz'
    # stim_regressor_file = '/Users/samlaxer/Documents/Projects/ds-003464/derivatives/feat/sub-jgroptoINS518_ses-2_task-opto_events_1D.txt'
    # stim_contrast_file = '/Users/samlaxer/Documents/Projects/ds-003464/derivatives/feat/contrast.txt'
    # output_file = '/Users/samlaxer/Documents/Projects/ds-003464/derivatives/feat/test_opto_glm_confcorr.nii.gz'
    # opto_stim_glm(bold_file, stim_regressor_file, stim_contrast_file, output_file, overwrite=True)

    # # Example usage
    # bold_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/anes_modPowAndFreqAndRes/confcorr_lowres/confound_correction_datasink/cleaned_timeseries/_split_name_sub-52Ear435510_ses-20260216_time-1420_run-4_ped-FW_task-rest_reconstruction-none_bold/sub-52Ear435510_ses-20260216_time-1420_run-4_ped-FW_task-rest_reconstruction-none_bold_RAS_resampled_cleaned.nii.gz'
    # stim_regressor_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/anes_modPowAndFreqAndRes/tmp_feat_manual/sub-52Ear435510_ses-20260216_run-4_DIO2_regressor.txt'
    # stim_contrast_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/anes_modPowAndFreqAndRes/tmp_feat_manual/contrasts.txt'
    # output_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/anes_modPowAndFreqAndRes/tmp_feat_manual/opto_glm_sub52_run4_DIO2.nii.gz'
    # opto_stim_glm(bold_file, stim_regressor_file, stim_contrast_file, output_file, overwrite=True)

    ####################################################################################
    
    # # Example usage
    # #***************************************************************************
    # # USER INPUT
    # bold_file = '/Volumes/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/bulk_analysis/bulk_conf_corrected_output/confound_correction_datasink/cleaned_timeseries/_split_name_sub-46Ear435509_ses-20251219_time-1118_run-2_ped-FW_task-rest_reconstruction-none_bold/sub-46Ear435509_ses-20251219_time-1118_run-2_ped-FW_task-rest_reconstruction-none_bold_RAS_combined_cleaned.nii.gz'
    # stim_regressor_file = '/Volumes/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/bulk_analysis/bulk_procd/sub-46Ear435509/ses-20251219/fp/sub-46Ear435509_ses-20251219_run-2_desc-ModPowAndFreq_fp_norm_bandpass-01to1_run-2_matched_conv_censored_opto_DIO1_regressor.txt'
    # #***************************************************************************
    # stim_contrast_file = stim_regressor_file.replace('.txt', '_con.txt')
    # # Find if regressor DIO1 or DIO2 and set output filename accordingly
    # opto_stim = os.path.basename(stim_regressor_file).split('_')[-2:-1][0]
    # outdir = f'{os.path.dirname(bold_file)}/{bold_file.split("/")[-1].replace(".nii.gz", f"_{opto_stim}_glm_results")}'
    # if not os.path.exists(outdir):
    #     os.makedirs(outdir)
    # # Put output_file in the same directory as the GLM results, with a name that includes the opto_stim (DIO1 or DIO2)
    # output_file = os.path.join(outdir, bold_file.split('/')[-1].replace('.nii.gz', f'_{opto_stim}_glm.nii.gz'))
    # mask_file = '/Users/samlaxer/Desktop/tmpOpto/highPow/commonspace_template_brain_mask_resampled.nii.gz'
    # #mask_file = '/Volumes/menon/jlaxer2/data/ds-005077/derivatives/feat/sub-wt1_ses-1_task-opto_run-1_stim_mask.nii.gz'
    # opto_stim_glm(
    #     bold_file,
    #     stim_regressor_file,
    #     stim_contrast_file,
    #     output_file,
    #     tr=1.2,
    #     cluster_z_thresh=2.3,   # ≈ p < 0.001 one-tailed, more stringent than FSL defaults (2.3 ≈ p < 0.01 one-tailed)
    #     cluster_p_thresh=0.05,
    #     overwrite=True,
    #     mask_file=mask_file
    # )


    ########### TESTING #########################################################################
    
    bold_file = '/Volumes/menon/jlaxer2/data/ds-NcModfMRIOpto/derivatives/bulk_analysis/bulk_procd/sub-52Ear435510/ses-20260216/func/sub-52Ear435510_ses-20260216_time-1408_run-3_ped-FW_task-rest_reconstruction-none_bold_RAS_combined_cleaned.nii.gz'
    stim_regressor_file = '/Volumes/menon/jlaxer2/data/ds-003464/derivatives/procd/sub-jgroptoINS518/ses-2/func/sub-jgroptoINS518_ses-2_task-opto_run-2_bold_RAS_resampled_cleaned_opto_regressor.txt'
    stim_contrast_file = '/Volumes/menon/jlaxer2/data/ds-003464/derivatives/procd/sub-jgroptoINS518/ses-2/func/sub-jgroptoINS518_ses-2_task-opto_run-2_bold_RAS_resampled_cleaned_opto_regressor.txt'
    output_file = '/Volumes/menon/jlaxer2/data/ds-003464/derivatives/procd/sub-jgroptoINS518/ses-2/func/sub-jgroptoINS518_ses-2_task-opto_run-2_bold_RAS_resampled_cleaned_glm/sub-jgroptoINS518_ses-2_task-opto_run-2_bold_RAS_resampled_cleaned_opto_regressor_glm.nii.gz'
    tr = 1.0
    mask_file = '/Users/samlaxer/Desktop/tmpOpto/highPow/commonspace_template_brain_mask_resampled.nii.gz'

    opto_stim_glm(
        bold_file,
        stim_regressor_file,
        stim_contrast_file,
        output_file,
        tr=tr,
        cluster_z_thresh=2.3,   # ≈ p < 0.001 one-tailed, more stringent than FSL defaults (2.3 ≈ p < 0.01 one-tailed)
        cluster_p_thresh=0.05,
        overwrite=True,
        mask_file=mask_file
    )
