####################################################################
# Script to run GLMsingle on the ThalHi dataset
####################################################################

# References & resources
# Install GLMsingle:
#   On thalamege: pip install --user git+https://github.com/cvnlab/GLMsingle.git
#   On argon:     source activate <env> && pip install git+https://github.com/cvnlab/GLMsingle.git
#
# Paper: https://elifesciences.org/articles/77599
# Video: https://www.youtube.com/watch?v=xFO-cYHZEwg
# Docs:  https://glmsingle.readthedocs.io/en/latest/python.html
# -----------------------------------------------------------------
import os
os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"

import numpy as np
from os.path import join, exists, split
import time
from glmsingle.glmsingle import GLM_single
from nilearn.maskers import NiftiMasker
import statsmodels.formula.api as smf
import pandas as pd
from joblib import Parallel, delayed
import subprocess
import re

# --------------------- CONFIG ---------------------
TR = 1.6
# stim_shift = 0.6   # 6s delay per Dillan’s code − (3 * 1.8s dropped) = 0.6s
# subjects = pd.read_csv("/Shared/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/usable_subjs.csv")['sub']
# subjects = input("Enter subject ID: ").strip()  # hardcode subject ID here for testing
subjects = ['11116']


# ======== Voxelwise GLM function ========
def voxelwise_glm(Y, design_df, formula, masker: NiftiMasker, out_dir: str, prefix: str, n_jobs: int = 24):
    """
    Mass-univariate OLS per voxel, parallelized.

    Y         : (n_trials, n_voxels) trial-wise array
    design_df : DataFrame with n_trials rows and regressors
    formula   : use 'y' as dependent var, e.g. 'y ~ RT_z + Trial_type'
    masker    : fitted NiftiMasker
    out_dir   : where to write maps
    prefix    : filename prefix (subject ID recommended)
    n_jobs    : number of parallel jobs (default=24)
    """

    n_trials, n_voxels = Y.shape

    # Seed term names and array shapes with voxel 0
    df0 = design_df.copy()
    df0['y'] = Y[:, 0]
    m0 = smf.ols(formula, data=df0).fit()
    term_names = m0.params.index.tolist()
    p = len(term_names)

    # Preallocate storage
    betas = np.zeros((p, n_voxels), dtype=float)
    tstats = np.zeros((p, n_voxels), dtype=float)

    # Fill voxel 0
    betas[:, 0] = m0.params.values
    tstats[:, 0] = m0.tvalues.values

    # Worker for a single voxel
    def _fit_voxel(v):
        df = design_df.copy()
        df['y'] = Y[:, v]
        m = smf.ols(formula, data=df).fit()
        return m.params.values, m.tvalues.values

    # Parallel loop over remaining voxels
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fit_voxel)(v) for v in range(1, n_voxels)
    )

    # Unpack results
    for idx, (b_vec, t_vec) in enumerate(results, start=1):
        betas[:, idx] = b_vec
        tstats[:, idx] = t_vec

    # Inverse-transform & save maps
    print(term_names)
    for i, name in enumerate(term_names):
        safe_name = make_safe(name)

        # Beta map
        img_b = masker.inverse_transform(betas[i, :])
        fout_b = os.path.join(out_dir, f"{prefix}_{safe_name}_beta.nii.gz")
        img_b.to_filename(fout_b)
        print("Wrote", fout_b)

        # T-stat map
        img_t = masker.inverse_transform(tstats[i, :])
        fout_t = os.path.join(out_dir, f"{prefix}_{safe_name}_tstat.nii.gz")
        img_t.to_filename(fout_t)
        print("Wrote", fout_t)

    return {'betas': betas, 'tstats': tstats, 'terms': term_names}


def make_safe(s):
    """Replace runs of non-word chars with underscores."""
    return re.sub(r"[^\w]+", "_", s).strip("_")


# =================================================================
# Main subject loop
# =================================================================
for sub in subjects:
    # Define paths
    deconv_dir = f"/data/backed_up/shared/ME_7T_Pilot/3dDeconvolve/sub-{sub}"
    data_dir = f"/data/backed_up/shared/ME_7T_Pilot/fmriprep/sub-{sub}/func"
    sub_dir = f"/data/backed_up/shared/ME_7T_Pilot/fmriprep/sub-{sub}"
    mask_afni = os.path.join(sub_dir, "combined_mask+tlrc")
    mask_nii = os.path.join(sub_dir, "combined_mask.nii.gz")

    # Timing files for GLMsingle
    cue_stim_file = os.path.join(deconv_dir, "cue_stimtimes_for_GLMsingle.1D")
    probe_stim_file = os.path.join(deconv_dir, "probe_stimtimes_for_GLMsingle.1D")

    # Subject-specific GLMsingle output
    out_base = "/data/backed_up/shared/ME_7T_Pilot/GLMsingle"
    outputdir = os.path.join(out_base, f"sub-{sub}")
    os.makedirs(outputdir, exist_ok=True)

    if not os.path.exists(mask_nii):
        print(f"Converting AFNI mask to NIfTI: {mask_afni} -> {mask_nii}")
        subprocess.run(["3dAFNItoNIFTI", "-prefix", mask_nii, mask_afni], check=True)

    # Subject-specific run list
    sub_int = int(sub)
    if sub_int == 11102:
        run_list = list(range(1, 12))   # runs 1 ~ 11
    elif sub_int == 11113:
        run_list = list(range(3, 13))   # runs 3 ~ 12
    else:
        run_list = list(range(1, 13))   # standard: runs 1 ~ 12

    try:
        # Convert AFNI mask to NIfTI and fit masker
        # subprocess.run(["3dAFNItoNIFTI", "-prefix", mask_nii, mask_afni], check=True)
        masker = NiftiMasker(mask_img=mask_nii, standardize=False, detrend=False)
        masker.fit()

        # Load cue and probe onsets (runs have the same trial count)
        cue_onsets_mat = np.loadtxt(cue_stim_file)
        probe_onsets_mat = np.loadtxt(probe_stim_file)

        # Build data & design lists per GLMsingle
        data = []
        design = []
        for run_idx, run_num in enumerate(run_list):
            # Load & mask BOLD
            fn = f"sub-{sub}_task-HCC_run-{run_num:02d}_part-mag_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
            bold = os.path.join(data_dir, fn)
            ts = masker.transform(bold).T
            data.append(ts)

            # Build 2-condition design matrix: col 0 = Cue, col 1 = Probe
            n_tp = ts.shape[1]
            des_mat = np.zeros((n_tp, 2), dtype=float)

            cue_idx = np.round(cue_onsets_mat[run_idx] / TR).astype(int)
            probe_idx = np.round(probe_onsets_mat[run_idx] / TR).astype(int)

            des_mat[cue_idx[cue_idx < n_tp], 0] = 1.0
            des_mat[probe_idx[probe_idx < n_tp], 1] = 1.0

            design.append(des_mat)

        # Run GLMsingle
        opt = {
            'wantlibrary': 1,
            'wantglmdenoise': 1,
            'wantfracridge': 1,
            'wantfileoutputs': [0, 0, 0, 0],
            'wantmemoryoutputs': [1, 1, 1, 1]
        }
        glms = GLM_single(opt)
        start = time.time()
        results_glmsingle = glms.fit(design, data, stimdur=0.5, tr=TR, outputdir=outputdir)
        print(f"sub-{sub} GLMsingle done in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")

        # Write out GLMsingle diagnostic maps
        typed = results_glmsingle['typed']

        # R2, noise pool, HRF index
        for key in ['R2', 'noisepool', 'HRFindex']:
            vol = masker.inverse_transform(np.squeeze(typed[key]))
            vol.to_filename(os.path.join(outputdir, f"{sub}_{key}.nii.gz"))

        # Mean beta
        meanb = np.squeeze(typed['betasmd']).mean(axis=1)
        masker.inverse_transform(meanb).to_filename(os.path.join(outputdir, f"{sub}_meanBeta.nii.gz"))

        # Extract Betas: (n_voxels, total_events)
        betas = np.squeeze(typed['betasmd'])

        # 1) Save ALL Trial Betas (Cue and Probe combined in chronological order)
        all_betas = betas.T  # (total_events, n_voxels)
        masker.inverse_transform(all_betas).to_filename(
            os.path.join(outputdir, f"{sub}_TrialBetas.nii.gz")
        )

        # Build condition masks from the design matrices
        cue_event_mask = []
        probe_event_mask = []
        for des in design:
            for t in range(des.shape[0]):
                if des[t, 0] == 1.0:
                    cue_event_mask.append(True)
                    probe_event_mask.append(False)
                if des[t, 1] == 1.0:
                    cue_event_mask.append(False)
                    probe_event_mask.append(True)

        cue_event_mask = np.array(cue_event_mask)
        probe_event_mask = np.array(probe_event_mask)

        # 2) Save Cue Trial Betas only
        cue_betas = betas[:, cue_event_mask].T  # (n_cue_trials, n_voxels)
        masker.inverse_transform(cue_betas).to_filename(
            os.path.join(outputdir, f"{sub}_Cue_TrialBetas.nii.gz")
        )

        # 3) Save Probe Trial Betas only
        probe_betas = betas[:, probe_event_mask].T  # (n_probe_trials, n_voxels)
        masker.inverse_transform(probe_betas).to_filename(
            os.path.join(outputdir, f"{sub}_Probe_TrialBetas.nii.gz")
        )

        print(f"Successfully saved TrialBetas, Cue_TrialBetas, and Probe_TrialBetas for sub-{sub}.")

    except Exception as e:
        # On failure, write out a warning file
        warn_file = os.path.join(outputdir, f"{sub}_warning.txt")
        with open(warn_file, "w") as f:
            f.write(f"Error processing sub-{sub}:\n{str(e)}\n")
        print(f"[!] sub-{sub} failed; see {warn_file}")
        continue

# end of line