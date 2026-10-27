####################################################################
# Script to run GLMsingle on the ThalHi dataset (Condition-separated)
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
subjects = ['11113', '11112', '11111', '11109', '11100', '11084', '10341']

# Define conditions and epochs
COND_LIST = ['EDS', 'IDS', 'Stay', 'Tossed']

# Column mapping: 
# 0: Cue_EDS,   1: Cue_IDS,   2: Cue_Stay,   3: Cue_Tossed
# 4: Probe_EDS, 5: Probe_IDS, 6: Probe_Stay, 7: Probe_Tossed
COL_ORDER = [f"Cue_{c}" for c in COND_LIST] + [f"Probe_{c}" for c in COND_LIST]


# ======== Helper functions ========
def load_1d_timing(filepath):
    """
    Load AFNI-style 1D timing file.
    Handles lines with wildcards ('*') as empty lists.
    """
    run_onsets = []
    with open(filepath, 'r') as f:
        for line in f:
            line_str = line.strip()
            if line_str == '*' or not line_str:
                run_onsets.append([])
            else:
                onsets = [float(x) for x in line_str.split()]
                run_onsets.append(onsets)
    return run_onsets


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

    # Subject-specific GLMsingle output directory
    out_base = "/data/backed_up/shared/ME_7T_Pilot/GLMsingle"
    outputdir = os.path.join(out_base, f"sub-{sub}")
    os.makedirs(outputdir, exist_ok=True)

    # Convert AFNI mask to NIfTI if not present
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
        # Fit NiftiMasker
        masker = NiftiMasker(mask_img=mask_nii, standardize=False, detrend=False)
        masker.fit()

        # Load all 8 timing files
        timing_data = {'Cue': {}, 'Probe': {}}
        for cond in COND_LIST:
            cue_fn = os.path.join(deconv_dir, f"cue_{cond}_stimtimes_for_GLMsingle.1D")
            probe_fn = os.path.join(deconv_dir, f"probe_{cond}_stimtimes_for_GLMsingle.1D")
            timing_data['Cue'][cond] = load_1d_timing(cue_fn)
            timing_data['Probe'][cond] = load_1d_timing(probe_fn)

        # Build data and design lists per GLMsingle
        data = []
        design = []
        event_metadata = []  # Keep track of chronological order of each single-trial event

        for run_idx, run_num in enumerate(run_list):
            # Load and mask BOLD time-series: (n_voxels, n_timepoints)
            fn = f"sub-{sub}_task-HCC_run-{run_num:02d}_part-mag_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
            bold = os.path.join(data_dir, fn)
            ts = masker.transform(bold).T
            data.append(ts)

            n_tp = ts.shape[1]
            des_mat = np.zeros((n_tp, len(COL_ORDER)), dtype=float)

            # Populate design matrix and record event metadata
            run_events = []
            for col_idx, col_name in enumerate(COL_ORDER):
                epoch, cond = col_name.split('_')
                onsets = timing_data[epoch][cond][run_idx]

                for onset in onsets:
                    tp = int(np.round(onset / TR))
                    if tp < n_tp:
                        des_mat[tp, col_idx] = 1.0
                        run_events.append({
                            'run': run_num,
                            'tp': tp,
                            'epoch': epoch,
                            'condition': cond,
                            'col_idx': col_idx
                        })

            # Sort events by time-point to preserve chronological single-trial order
            run_events.sort(key=lambda x: (x['tp'], x['col_idx']))
            event_metadata.extend(run_events)
            design.append(des_mat)

        # Convert event metadata to DataFrame and save
        event_df = pd.DataFrame(event_metadata)
        event_df.to_csv(os.path.join(outputdir, f"{sub}_event_metadata.csv"), index=False)

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

        # Write out GLMsingle diagnostic maps (typed)
        typed = results_glmsingle['typed']

        for key in ['R2', 'noisepool', 'HRFindex']:
            vol = masker.inverse_transform(np.squeeze(typed[key]))
            vol.to_filename(os.path.join(outputdir, f"{sub}_{key}.nii.gz"))

        # Save mean beta map across all events
        betas = np.squeeze(typed['betasmd'])  # Shape: (n_voxels, total_events)
        meanb = betas.mean(axis=1)
        masker.inverse_transform(meanb).to_filename(os.path.join(outputdir, f"{sub}_meanBeta.nii.gz"))

        # 1) Save ALL Trial Betas (Cue, Probe, Tossed combined in chronological order)
        all_betas = betas.T  # Shape: (total_events, n_voxels)
        masker.inverse_transform(all_betas).to_filename(
            os.path.join(outputdir, f"{sub}_TrialBetas.nii.gz")
        )

        # 2) Extract and save ALL Cue Betas (including tossed)
        cue_mask = (event_df['epoch'] == 'Cue').values
        cue_betas = betas[:, cue_mask].T
        masker.inverse_transform(cue_betas).to_filename(
            os.path.join(outputdir, f"{sub}_Cue_TrialBetas.nii.gz")
        )

        # 3) Extract and save ALL Probe Betas (including tossed)
        probe_mask = (event_df['epoch'] == 'Probe').values
        probe_betas = betas[:, probe_mask].T
        masker.inverse_transform(probe_betas).to_filename(
            os.path.join(outputdir, f"{sub}_Probe_TrialBetas.nii.gz")
        )

        # 4) Extract and save CLEAN (Valid) Cue Betas (Excluding Tossed trials: EDS, IDS, Stay)
        clean_cue_mask = ((event_df['epoch'] == 'Cue') & (event_df['condition'] != 'Tossed')).values
        clean_cue_betas = betas[:, clean_cue_mask].T
        masker.inverse_transform(clean_cue_betas).to_filename(
            os.path.join(outputdir, f"{sub}_Clean_Cue_TrialBetas.nii.gz")
        )

        # 5) Extract and save CLEAN (Valid) Probe Betas (Excluding Tossed trials: EDS, IDS, Stay)
        clean_probe_mask = ((event_df['epoch'] == 'Probe') & (event_df['condition'] != 'Tossed')).values
        clean_probe_betas = betas[:, clean_probe_mask].T
        masker.inverse_transform(clean_probe_betas).to_filename(
            os.path.join(outputdir, f"{sub}_Clean_Probe_TrialBetas.nii.gz")
        )

        # Save metadata for clean cue trials for downstream RSA / MVPA decoding
        event_df[clean_cue_mask].reset_index(drop=True).to_csv(
            os.path.join(outputdir, f"{sub}_Clean_Cue_metadata.csv"), index=False
        )

        print(f"Successfully saved all diagnostic and single-trial beta maps for sub-{sub}.")

    except Exception as e:
        warn_file = os.path.join(outputdir, f"{sub}_warning.txt")
        with open(warn_file, "w") as f:
            f.write(f"Error processing sub-{sub}:\n{str(e)}\n")
        print(f"[!] sub-{sub} failed; see {warn_file}")
        continue

# end of line