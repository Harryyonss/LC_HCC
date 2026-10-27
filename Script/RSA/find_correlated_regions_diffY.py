## command line: python find_correlated_regions_diffY.py 11115 --roi_mask /path/to/roi.nii.gz --sl_glm

import os
# Limit internal multithreading in numerical libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import glob
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import nibabel as nib
import nilearn
from nilearn import image, masking
from nilearn.masking import unmask


####################################################################
# 0. REAL-TIME DUAL LOGGER (TERMINAL + SUBJECT LOG FILE)
####################################################################
class DualLogger:
    """
    Simultaneously writes stdout/stderr messages to both the console
    and a subject-specific log file in real-time (unbuffered).
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "a", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if not self.log_file.closed:
            self.log_file.close()


####################################################################
# 1. COMMAND-LINE ARGUMENT PARSER
####################################################################
def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Whole-Brain Beta DIFFERENCE (Delta Y) GLM using precomputed Searchlight RSA (.npy) distances within an ROI",
        usage="python find_correlated_regions_diffY.py [subject] [--roi_mask PATH] [--sl_glm]",
    )
    parser.add_argument(
        "subject",
        help="Subject ID (e.g., 11105, 11115)"
    )
    parser.add_argument(
        "--roi_mask",
        type=str,
        required=True,
        help="Path to the 3D ROI mask NIfTI file (.nii or .nii.gz)"
    )
    parser.add_argument(
        "--sl_glm",
        help="Run whole-brain GLM regressing single-trial Beta differences (Delta Y) on ROI searchlight distances (X)",
        default=False,
        action="store_true"
    )
    return parser


parser = init_argparse()
args = parser.parse_args(sys.argv[1:])
sub = args.subject
ROI_MASK_PATH = args.roi_mask

# Define directory and file paths EXACTLY matching Untitled_2.rtf
BASE_DIR = "/data/backed_up/shared/ME_7T_Pilot"
GLM_DIR = os.path.join(BASE_DIR, "GLMsingle", f"sub-{sub}")
MASK_PATH = os.path.join(BASE_DIR, "fmriprep", f"sub-{sub}", "combined_mask.nii.gz")
BEHAV_DIR = os.path.join(BASE_DIR, "behav_data", str(sub))
SL_DIR = os.path.join(BASE_DIR, "searchlightRSA/vdt", f"sub-{sub}")
OUT_DIR = os.path.join(SL_DIR, "roi_dist_glm")
os.makedirs(OUT_DIR, exist_ok=True)

# Extract ROI name from filename for clean output naming
roi_name = os.path.basename(ROI_MASK_PATH).replace(".nii.gz", "").replace(".nii", "")

# Initialize DualLogger inside the output directory
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filepath = os.path.join(OUT_DIR, f"{sub}_{roi_name}_diffY_glm_{timestamp_str}.log")
dual_logger = DualLogger(log_filepath)
sys.stdout = dual_logger
sys.stderr = dual_logger

print("=" * 80)
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PIPELINE INITIALIZED (DELTA Y FROM PRECOMPUTED .NPY)")
print(f"  Subject ID    : {sub}")
print(f"  ROI Name      : {roi_name}")
print(f"  ROI Mask Path : {ROI_MASK_PATH}")
print(f"  GLM Dir       : {GLM_DIR}")
print(f"  SL NPY Dir    : {SL_DIR}")
print(f"  Behavior Dir  : {BEHAV_DIR}")
print(f"  Output Dir    : {OUT_DIR}")
print(f"  Live Log File : {log_filepath}")
print("=" * 80)


####################################################################
# 2. BEHAVIORAL CSV LOADER
####################################################################
def load_and_concat_behav_csvs(behav_dir, sub_id):
    """
    Load all behavioral block CSV files for a given subject and sort rows strictly
    by internal numeric 'block' and 'trial' columns.
    """
    pattern = os.path.join(behav_dir, f"sub-{sub_id}_task-HCC_block-*.csv")
    csv_files = glob.glob(pattern)
    if not csv_files:
        raise FileNotFoundError(f"No behavioral CSV files found matching: {pattern}")

    print(f"-> Found {len(csv_files)} behavioral CSV files matching pattern.")
    df_list = []
    seen_blocks = {}

    for f in csv_files:
        df_tmp = pd.read_csv(f)
        if "block" not in df_tmp.columns or "trial" not in df_tmp.columns:
            raise KeyError(f"'block' or 'trial' column missing in {os.path.basename(f)}")

        df_tmp["block"] = pd.to_numeric(df_tmp["block"], errors="raise").astype(int)
        df_tmp["trial"] = pd.to_numeric(df_tmp["trial"], errors="raise").astype(int)

        for b_num in df_tmp["block"].unique():
            if b_num in seen_blocks:
                raise ValueError(
                    f"Duplicate block number ({b_num}) detected in {os.path.basename(f)} "
                    f"(already loaded from {seen_blocks[b_num]})"
                )
            seen_blocks[b_num] = os.path.basename(f)

        df_list.append(df_tmp)

    full_df = pd.concat(df_list, ignore_index=True)
    full_df = full_df.sort_values(["block", "trial"]).reset_index(drop=True)

    print(f"-> Consolidated {len(seen_blocks)} blocks chronologically by internal block/trial:")
    for b_num, grp in full_df.groupby("block"):
        print(f"   Block {b_num:2d} | {len(grp):2d} trials | Source: {seen_blocks[b_num]}")

    return full_df


####################################################################
# 3. RUN WHOLE-BRAIN GLM USING PRECOMPUTED SEARCHLIGHT .NPY
####################################################################
if args.sl_glm:
    step_start = datetime.now()
    print("\n" + "=" * 80)
    print(f"[{step_start.strftime('%Y-%m-%d %H:%M:%S')}] RUNNING ROI DISTANCE TO WHOLE-BRAIN DELTA-BETA (DIFF Y) GLM")
    print("=" * 80)

    # 1. Load reference searchlight mask from Step 1
    saved_mask_path = os.path.join(SL_DIR, f"{sub}_searchlight_mask.nii.gz")
    if os.path.exists(saved_mask_path):
        mask_nii = nib.load(saved_mask_path)
        print(f"-> Loaded reference searchlight mask from Step 1: {saved_mask_path}")
    else:
        mask_nii = nib.load(MASK_PATH)
        print(f"-> Loaded base mask: {MASK_PATH}")

    wb_binary = (mask_nii.get_fdata() > 0).astype(np.int8)
    mask_nii = nilearn.image.new_img_like(mask_nii, wb_binary)
    n_spheres = int(np.sum(wb_binary))
    print(f"   Total active whole-brain voxels/spheres: {n_spheres}")

    # 2. Load ROI mask and map it to the 1D searchlight sphere indices
    if not os.path.exists(ROI_MASK_PATH):
        raise FileNotFoundError(f"Missing ROI mask file: {ROI_MASK_PATH}")

    roi_nii = nib.load(ROI_MASK_PATH)
    if roi_nii.shape[:3] != mask_nii.shape[:3] or not np.allclose(roi_nii.affine, mask_nii.affine):
        print("-> Resampling ROI mask to match reference searchlight mask grid and affine...")
        roi_nii = image.resample_to_img(roi_nii, mask_nii, interpolation="nearest")

    roi_in_wb_vec = masking.apply_mask(roi_nii, mask_nii)
    roi_sphere_indices = np.where(roi_in_wb_vec > 0)[0]
    print(f"-> Mapped ROI mask onto searchlight space: {len(roi_sphere_indices)} overlapping spheres/voxels found.")

    if len(roi_sphere_indices) == 0:
        raise ValueError("No overlapping voxels found between the ROI mask and the subject's searchlight mask!")

    # 3. Load 4D single-trial Beta images
    cue_file = os.path.join(GLM_DIR, f"{sub}_Cue_TrialBetas.nii.gz")
    probe_file = os.path.join(GLM_DIR, f"{sub}_Probe_TrialBetas.nii.gz")

    print(f"-> Loading 4D single-trial Beta images (Y)...")
    print(f"   Cue Beta   : {cue_file} (Exists: {os.path.exists(cue_file)})")
    print(f"   Probe Beta : {probe_file} (Exists: {os.path.exists(probe_file)})")

    if not os.path.exists(cue_file) or not os.path.exists(probe_file):
        raise FileNotFoundError(f"Missing Beta files:\n  {cue_file}\n  {probe_file}")

    cue_nii = nib.load(cue_file)
    probe_nii = nib.load(probe_file)

    wb_cue_VxT = masking.apply_mask(cue_nii, mask_nii).T     # shape: (n_spheres, n_trials)
    wb_probe_VxT = masking.apply_mask(probe_nii, mask_nii).T # shape: (n_spheres, n_trials)
    n_trials = wb_cue_VxT.shape[1]
    print(f"   wb_cue_VxT shape   : {wb_cue_VxT.shape} (voxels x trials)")
    print(f"   wb_probe_VxT shape : {wb_probe_VxT.shape} (voxels x trials)")

    # 4. Compute trial-by-trial Beta differences (Delta Y = Beta[t] - Beta[t-1])
    print("-> Computing trial-by-trial Beta differences (Delta Y = Beta[t] - Beta[t-1])...")
    diff_cue2cue_VxT = np.full_like(wb_cue_VxT, np.nan, dtype=np.float64)
    diff_probe2probe_VxT = np.full_like(wb_probe_VxT, np.nan, dtype=np.float64)
    diff_probe2cue_VxT = np.full_like(wb_cue_VxT, np.nan, dtype=np.float64)

    diff_cue2cue_VxT[:, 1:] = wb_cue_VxT[:, 1:] - wb_cue_VxT[:, :-1]
    diff_probe2probe_VxT[:, 1:] = wb_probe_VxT[:, 1:] - wb_probe_VxT[:, :-1]
    diff_probe2cue_VxT[:, 1:] = wb_cue_VxT[:, 1:] - wb_probe_VxT[:, :-1]

    # 5. Load behavioral CSVs and compute dt & trial filtering mask
    df = load_and_concat_behav_csvs(BEHAV_DIR, sub)
    print(f"-> Total behavioral rows loaded: {len(df)}")

    df["dt_Cue2Cue"] = df.groupby("block")["Time_Since_Run_Cue_Prez"].diff()
    df["dt_Probe2Probe"] = df.groupby("block")["Time_Since_Run_Probe_Prez"].diff()
    df["dt_Probe2Cue"] = df["Time_Since_Run_Cue_Prez"] - df.groupby("block")["Time_Since_Run_Probe_Prez"].shift(1)

    df["subject_correct"] = pd.to_numeric(df["subject_correct"], errors="coerce").fillna(0).astype(int)
    df["prev_correct"] = df.groupby("block")["subject_correct"].shift(1).fillna(0).astype(int)
    df["prev2_correct"] = df.groupby("block")["subject_correct"].shift(2).fillna(0).astype(int)

    MIN_TRIAL = 3
    valid_mask = (
        (df["trial"] >= MIN_TRIAL) &
        (df["subject_correct"] == 1) &
        (df["prev_correct"] == 1) &
        (df["prev2_correct"] == 1)
    )
    valid_indices = df.index[valid_mask].to_numpy()

    print(f"\n-> Trial filtering criteria applied:")
    print(f"   Rule: trial >= {MIN_TRIAL}, subject_correct == 1, prev_correct == 1, prev2_correct == 1")
    print(f"   Total valid transitions: {len(valid_indices)} / {len(df)}")

    # Define target whole-brain Delta-Beta series (Delta Y) for each transition condition
    # Kept target_label names identical to Untitled_2.rtf + "_diffY" suffix for easy tracking
    conditions_config = [
        ("Cue2Cue",     "CueBetas",   diff_cue2cue_VxT),
        ("Probe2Probe", "ProbeBetas", diff_probe2probe_VxT),
        ("Probe2Cue",   "CueBetas",   diff_probe2cue_VxT),
    ]

    for cond, target_label, wb_Y_full in conditions_config:
        print(f"\n" + "-" * 60)
        print(f"-> Processing condition: [{cond}] (Predicting Whole-Brain Delta {target_label})")

        npy_path = os.path.join(SL_DIR, f"{sub}_sl_adjcorr_{cond}_WTT.npy")
        if not os.path.exists(npy_path):
            print(f"   [ERROR] Missing precomputed distance file: {npy_path}. Skipping.")
            continue

        ds_array = np.load(npy_path)
        print(f"   Loaded distance matrix shape: {ds_array.shape} (spheres x trials)")

        if ds_array.shape[0] != n_spheres:
            raise ValueError(
                f"Sphere count mismatch in {cond}! "
                f".npy rows ({ds_array.shape[0]}) != Mask voxels ({n_spheres})"
            )
        if ds_array.shape[1] != len(df):
            raise ValueError(
                f"Trial count mismatch in {cond}! "
                f".npy columns ({ds_array.shape[1]}) != Behavioral CSV rows ({len(df)})"
            )

        # ---------------------------------------------------------------------
        # Extract ROI distance time-series (X) and control for dt & global shift
        # ---------------------------------------------------------------------
        dt_vec = df.loc[valid_indices, f"dt_{cond}"].to_numpy(dtype=np.float64)
        dt_centered = dt_vec - np.nanmean(dt_vec)

        # 1. Slice valid trials across all spheres to compute clean dt-residualized & demeaned distances
        D_valid = ds_array[:, valid_indices].astype(np.float64)
        D_mean = np.nanmean(D_valid, axis=1, keepdims=True)
        D_centered = D_valid - D_mean

        # Regress out dt per sphere (consistent with Step 2)
        slope_dt = np.nansum(D_centered * dt_centered[np.newaxis, :], axis=1, keepdims=True) / np.nansum(dt_centered ** 2)
        D_res = D_valid - (slope_dt * dt_centered[np.newaxis, :])

        # 2. Average across the spheres inside the ROI mask to form the 1D Predictor vector X
        X_roi_dist = np.nanmean(D_res[roi_sphere_indices, :], axis=0)  # shape: (n_valid_trials,)

        # 3. Slice target whole-brain Delta-Beta series (Delta Y) for the same valid trials
        Y_valid = wb_Y_full[:, valid_indices].astype(np.float64)       # shape: (n_spheres, n_valid_trials)

        # Filter out any trials containing NaNs in X, dt, or Y
        finite_mask = (
            ~np.isnan(X_roi_dist) &
            ~np.isnan(dt_centered) &
            ~np.isnan(Y_valid).any(axis=0)
        )

        X_clean = X_roi_dist[finite_mask]
        dt_clean = dt_centered[finite_mask]
        Y_clean = Y_valid[:, finite_mask]
        n_obs = len(X_clean)

        print(f"   Valid finite trials entering GLM : {n_obs}")
        print(f"   ROI Distance (X) stats -> Mean: {np.mean(X_clean):.6f}, SD: {np.std(X_clean):.6f}")

        # ---------------------------------------------------------------------
        # Fit Mass-Univariate OLS Regression: Delta_Y = b0 + b1 * X_roi_dist + b2 * dt
        # ---------------------------------------------------------------------
        X_design = np.column_stack([
            np.ones(n_obs, dtype=np.float64),
            X_clean - np.mean(X_clean),
            dt_clean - np.mean(dt_clean),
        ])

        # Vectorized OLS across all whole-brain voxels simultaneously
        XtX_inv = np.linalg.pinv(X_design.T @ X_design)
        betas = XtX_inv @ X_design.T @ Y_clean.T  # shape: (3, n_spheres)

        residuals = Y_clean.T - (X_design @ betas)
        dof = n_obs - X_design.shape[1]
        sigma2 = np.sum(residuals ** 2, axis=0) / dof

        # Extract Beta and T-statistic for Predictor 1 (X_roi_dist)
        beta_vec = betas[1, :]
        se_vec = np.sqrt(sigma2 * XtX_inv[1, 1])
        tval_vec = beta_vec / (se_vec + 1e-12)

        beta_vec = np.nan_to_num(beta_vec, nan=0.0)
        tval_vec = np.nan_to_num(tval_vec, nan=0.0)

        print(f"   GLM Beta stats   -> Min: {np.min(beta_vec):.4f}, Max: {np.max(beta_vec):.4f}, Mean: {np.mean(beta_vec):.4f}")
        print(f"   GLM T-stat stats -> Min: {np.min(tval_vec):.2f}, Max: {np.max(tval_vec):.2f}, Mean: {np.mean(tval_vec):.2f}")

        # 4. Unmask back to 3D brain NIfTI images (with _diffY_beta.nii.gz suffix)
        beta_out = os.path.join(OUT_DIR, f"{sub}_{roi_name}_{cond}_to_{target_label}_diffY_beta.nii.gz")
        tval_out = os.path.join(OUT_DIR, f"{sub}_{roi_name}_{cond}_to_{target_label}_diffY_tval.nii.gz")

        beta_nii = unmask(beta_vec, mask_nii)
        tval_nii = unmask(tval_vec, mask_nii)

        beta_nii.to_filename(beta_out)
        tval_nii.to_filename(tval_out)

        print(f"   Saved 3D NIfTI maps:")
        print(f"     [Beta]   {beta_out} (Shape: {beta_nii.shape})")
        print(f"     [T-stat] {tval_out} (Shape: {tval_nii.shape})")

    print(f"\nALL GLM ANALYSES COMPLETE. Elapsed time: {datetime.now() - step_start}")

print("\n" + "=" * 80)
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALL TASKS FINISHED SUCCESSFULLY")
print(f"Log written to: {log_filepath}")
print("=" * 80)

# Clean up dual logger
dual_logger.close()