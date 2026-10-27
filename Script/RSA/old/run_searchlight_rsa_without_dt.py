## command line: python run_searchlight_rsa.py 11115 --searchlight_rsa --sl_ttest --n_jobs 16

import os
# Limit internal multithreading in numerical libraries to prevent CPU oversubscription
# during joblib parallel processing across searchlight spheres.
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
import scipy.stats as stats
from scipy import sparse
from sklearn import neighbors
import nilearn
from nilearn import image, masking
from nilearn.image.resampling import coord_transform
from nilearn.masking import unmask
from joblib import Parallel, delayed
from tqdm import tqdm


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
        description="Searchlight RSA (Cue2Cue, Probe2Probe, Probe2Cue) & EDS vs. IDS T-test",
        usage="python script.py [subject] [--searchlight_rsa] [--sl_ttest] [--n_jobs N]",
    )
    parser.add_argument(
        "subject",
        help="Subject ID (e.g., 11105, 11115)"
    )
    parser.add_argument(
        "--searchlight_rsa",
        help="Step 1: Compute pre-whitened adjacent-trial neural distances for 3 transition types",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--sl_ttest",
        help="Step 2: Run whole-brain within-subject two-sample t-test (EDS vs. IDS)",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=16,
        help="Number of parallel CPU cores to use for searchlight computation (default: 16)"
    )
    return parser


parser = init_argparse()
args = parser.parse_args(sys.argv[1:])
sub = args.subject
n_jobs = args.n_jobs

# Define directory and file paths
BASE_DIR = "/data/backed_up/shared/ME_7T_Pilot"
GLM_DIR = os.path.join(BASE_DIR, "GLMsingle", f"sub-{sub}")
MASK_PATH = os.path.join(BASE_DIR, "fmriprep", f"sub-{sub}", "combined_mask.nii.gz")
BEHAV_DIR = os.path.join(BASE_DIR, "behav_data", str(sub))
OUT_DIR = os.path.join(BASE_DIR, "searchlightRSA", f"sub-{sub}")
os.makedirs(OUT_DIR, exist_ok=True)

# Initialize DualLogger inside the subject's output directory
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filepath = os.path.join(OUT_DIR, f"{sub}_searchlight_rsa_{timestamp_str}.log")
dual_logger = DualLogger(log_filepath)
sys.stdout = dual_logger
sys.stderr = dual_logger

print("=" * 80)
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PIPELINE INITIALIZED")
print(f"  Subject ID    : {sub}")
print(f"  CPU Cores     : {n_jobs}")
print(f"  GLM Dir       : {GLM_DIR}")
print(f"  Mask Path     : {MASK_PATH}")
print(f"  Behavior Dir  : {BEHAV_DIR}")
print(f"  Output Dir    : {OUT_DIR}")
print(f"  Live Log File : {log_filepath}")
print("=" * 80)


####################################################################
# 2. SEARCHLIGHT AFFINITY & PRE-WHITENING FUNCTIONS
####################################################################
def new_apply_mask_and_get_affinity(seeds, niimg, radius, allow_overlap, mask_img=None):
    """
    Build a sparse sphere-to-voxel adjacency/affinity matrix for searchlight analysis.
    Uses KDTree spatial queries in world (mm) space.
    """
    seeds = np.asarray(seeds)
    if seeds.ndim == 1:
        seeds = seeds[np.newaxis, :]

    if niimg is None:
        mask, affine = masking.load_mask_img(mask_img)
        mask_coords = np.column_stack(np.where(mask))
        X = None
    else:
        affine = niimg.affine
        if isinstance(mask_img, str):
            mask_img = nib.load(mask_img)
        mask_img = image.resample_img(
            mask_img,
            target_affine=affine,
            target_shape=niimg.shape[:3],
            interpolation="nearest",
            force_resample=False,
        )
        mask, _ = masking.load_mask_img(mask_img)
        mask_coords = np.column_stack(np.where(mask != 0))
        X = masking.apply_mask_fmri(niimg, mask_img)

    mask_coords_world = coord_transform(
        mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2], affine
    )
    mask_coords_world = np.column_stack(mask_coords_world)

    inv_affine = np.linalg.inv(affine)
    seeds_voxel = (
        np.column_stack([seeds, np.ones(len(seeds))]) @ inv_affine.T
    )[:, :3]
    nearest_voxels = np.round(seeds_voxel).astype(int)
    coord_dict = {tuple(c): i for i, c in enumerate(mask_coords)}
    nearests = [coord_dict.get(tuple(coord), None) for coord in nearest_voxels]

    tree = neighbors.KDTree(mask_coords_world)
    neighbors_list = tree.query_radius(seeds, r=radius)

    n_seeds = len(seeds)
    n_voxels = len(mask_coords)
    A_sparse = sparse.lil_matrix((n_seeds, n_voxels), dtype=bool)

    for i, (nbrs, nearest) in enumerate(zip(neighbors_list, nearests)):
        if nbrs.size > 0:
            A_sparse[i, nbrs] = True
        if nearest is not None:
            A_sparse[i, nearest] = True

    seed_world = coord_transform(
        seeds_voxel[:, 0], seeds_voxel[:, 1], seeds_voxel[:, 2], affine
    )
    seed_world = np.column_stack(seed_world)
    _, seed_idx = tree.query(seed_world, k=1)
    for i, idx in enumerate(seed_idx):
        A_sparse[i, idx] = True

    sphere_sizes = np.asarray(A_sparse.sum(axis=1)).ravel()
    empty = np.nonzero(sphere_sizes == 0)[0]
    if len(empty):
        raise ValueError(f"Empty spheres detected at indices: {empty}")
    if not allow_overlap and np.any(A_sparse.sum(axis=0) >= 2):
        raise ValueError("Overlap detected between spheres when allow_overlap=False")

    return X, A_sparse.tocsr()


def compute_shrunk_covariance_and_whitening_matrix(x, return_shrunk_cov=False):
    """
    Estimate shrunk spatial noise covariance matrix and compute its inverse
    square root (whitening matrix) following Walther et al. (2016).
    """
    t, n = x.shape
    if t < 2:
        identity_matrix = np.eye(n)
        return (identity_matrix, identity_matrix) if return_shrunk_cov else identity_matrix

    x_demeaned = x - x.mean(0)
    sample_cov = (1.0 / (t - 1)) * np.dot(x_demeaned.T, x_demeaned)
    prior = np.diag(np.diag(sample_cov))

    num = np.sum((sample_cov - prior) ** 2)
    den = np.sum((x_demeaned**2 - np.diag(sample_cov)[np.newaxis, :]) ** 2) / t
    shrinkage = 1.0
    if den > 1e-10 and num > 1e-10:
        shrinkage = num / den
    shrinkage = np.clip(shrinkage, 0.0, 1.0)

    if t < n:
        shrinkage = max(shrinkage, 1 - ((t - 1) / (n * 1.5)))
        shrinkage = np.clip(shrinkage, 0.0, 1.0)

    sigma_shrunk = shrinkage * prior + (1 - shrinkage) * sample_cov
    epsilon = 1e-7

    try:
        np.linalg.cholesky(sigma_shrunk)
    except np.linalg.LinAlgError:
        sigma_shrunk = sigma_shrunk + epsilon * np.eye(n)

    whitening_matrix = np.eye(n)
    try:
        evals, evecs = np.linalg.eigh(sigma_shrunk)
        evals_floored = np.maximum(evals, epsilon)
        evals_sqrt_inv = 1.0 / np.sqrt(evals_floored)
        whitening_matrix = np.dot(evecs * evals_sqrt_inv, evecs.T)
    except np.linalg.LinAlgError:
        whitening_matrix = np.eye(n)

    if return_shrunk_cov:
        return whitening_matrix, sigma_shrunk
    else:
        return whitening_matrix


def process_sphere(idx, cue_VxT, probe_VxT, devs_cue, devs_probe, devs_both, A, n_trials, whiten="On"):
    """
    Process a single searchlight sphere:
      1. Cue(t-1)   -> Cue(t)   : Whitened with W_cue
      2. Probe(t-1) -> Probe(t) : Whitened with W_probe
      3. Probe(t-1) -> Cue(t)   : Whitened with W_both
    """
    vox_inds = A[idx].indices
    if len(vox_inds) == 0:
        nan_arr = np.full(n_trials, np.nan)
        return nan_arr, nan_arr, nan_arr

    b_cue = cue_VxT[vox_inds, :]
    b_probe = probe_VxT[vox_inds, :]

    dev_c_roi = devs_cue[:, vox_inds]
    dev_p_roi = devs_probe[:, vox_inds]
    dev_b_roi = devs_both[:, vox_inds]

    if whiten == "On":
        # 1. Cue-only whitening
        if dev_c_roi.shape[0] >= 2:
            W_cue, _ = compute_shrunk_covariance_and_whitening_matrix(dev_c_roi, return_shrunk_cov=True)
        else:
            W_cue = np.eye(len(vox_inds))
        wb_cue_only = W_cue.dot(b_cue)

        # 2. Probe-only whitening
        if dev_p_roi.shape[0] >= 2:
            W_probe, _ = compute_shrunk_covariance_and_whitening_matrix(dev_p_roi, return_shrunk_cov=True)
        else:
            W_probe = np.eye(len(vox_inds))
        wb_probe_only = W_probe.dot(b_probe)

        # 3. Combined Cue+Probe whitening for cross-phase comparison
        if dev_b_roi.shape[0] >= 2:
            W_both, _ = compute_shrunk_covariance_and_whitening_matrix(dev_b_roi, return_shrunk_cov=True)
        else:
            W_both = np.eye(len(vox_inds))
        wb_cue_joint = W_both.dot(b_cue)
        wb_probe_joint = W_both.dot(b_probe)
    else:
        wb_cue_only = b_cue
        wb_probe_only = b_probe
        wb_cue_joint = b_cue
        wb_probe_joint = b_probe

    data_tv_cue_only = wb_cue_only.T
    data_tv_probe_only = wb_probe_only.T
    data_tv_cue_joint = wb_cue_joint.T
    data_tv_probe_joint = wb_probe_joint.T

    corrd_cc = np.full(n_trials, np.nan)
    corrd_pp = np.full(n_trials, np.nan)
    corrd_pc = np.full(n_trials, np.nan)

    # Compute adjacent correlation distances
    for t in range(1, n_trials):
        # 1. Cue(t-1) -> Cue(t)
        r_cc = np.corrcoef(data_tv_cue_only[t - 1], data_tv_cue_only[t])[0, 1]
        corrd_cc[t] = np.sqrt(2 * (1 - r_cc))

        # 2. Probe(t-1) -> Probe(t)
        r_pp = np.corrcoef(data_tv_probe_only[t - 1], data_tv_probe_only[t])[0, 1]
        corrd_pp[t] = np.sqrt(2 * (1 - r_pp))

        # 3. Probe(t-1) -> Cue(t)
        r_pc = np.corrcoef(data_tv_probe_joint[t - 1], data_tv_cue_joint[t])[0, 1]
        corrd_pc[t] = np.sqrt(2 * (1 - r_pc))

    return corrd_cc, corrd_pp, corrd_pc


####################################################################
# STEP 1: RUN SEARCHLIGHT RSA PIPELINE
####################################################################
if args.searchlight_rsa:
    start_time = datetime.now()
    print("\n" + "=" * 80)
    print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] STEP 1: STARTING SEARCHLIGHT RSA")
    print("=" * 80)

    cue_file = os.path.join(GLM_DIR, f"{sub}_Cue_TrialBetas.nii.gz")
    probe_file = os.path.join(GLM_DIR, f"{sub}_Probe_TrialBetas.nii.gz")

    print(f"-> Checking input Beta files:")
    print(f"   Cue Beta   : {cue_file} (Exists: {os.path.exists(cue_file)})")
    print(f"   Probe Beta : {probe_file} (Exists: {os.path.exists(probe_file)})")
    print(f"   Mask File  : {MASK_PATH} (Exists: {os.path.exists(MASK_PATH)})")

    if not os.path.exists(cue_file) or not os.path.exists(probe_file):
        raise FileNotFoundError(f"Missing Beta files:\n  {cue_file}\n  {probe_file}")
    if not os.path.exists(MASK_PATH):
        raise FileNotFoundError(f"Missing Mask file: {MASK_PATH}")

    # Load 4D Beta NIfTIs
    print("-> Loading 4D single-trial Beta images...")
    cue_nii = nib.load(cue_file)
    probe_nii = nib.load(probe_file)
    print(f"   Cue NIfTI shape   : {cue_nii.shape}")
    print(f"   Probe NIfTI shape : {probe_nii.shape}")

    if cue_nii.shape != probe_nii.shape:
        raise ValueError(f"Shape mismatch between Cue {cue_nii.shape} and Probe {probe_nii.shape}")
    n_trials = cue_nii.shape[3]
    print(f"   Total trials (volumes): {n_trials}")

    # Load and resample mask
    print("-> Loading and verifying subject brain mask...")
    mask_nii = nib.load(MASK_PATH)
    print(f"   Raw mask shape    : {mask_nii.shape}")

    if mask_nii.shape[:3] != cue_nii.shape[:3] or not np.allclose(mask_nii.affine, cue_nii.affine):
        print("   Resampling mask to match Beta NIfTI grid and affine...")
        mask_nii = image.resample_to_img(mask_nii, cue_nii, interpolation="nearest")
        print(f"   Resampled mask shape: {mask_nii.shape}")

    mask_data = mask_nii.get_fdata()
    binary = (mask_data > 0).astype(np.int8)
    n_mask_voxels = np.sum(binary)
    print(f"   Active mask voxels: {n_mask_voxels}")

    binary_mask_img = nilearn.image.new_img_like(mask_nii, binary)
    saved_mask_path = os.path.join(OUT_DIR, f"{sub}_searchlight_mask.nii.gz")
    binary_mask_img.to_filename(saved_mask_path)
    print(f"   Saved searchlight reference mask -> {saved_mask_path}")

    # Extract masked Voxel x Trial 2D matrices
    print("-> Applying mask to functional images...")
    cue_mat = masking.apply_mask(cue_nii, binary_mask_img)
    cue_VxT = cue_mat.T
    del cue_mat

    probe_mat = masking.apply_mask(probe_nii, binary_mask_img)
    probe_VxT = probe_mat.T
    del probe_mat

    print(f"   cue_VxT shape   : {cue_VxT.shape} (voxels x trials)")
    print(f"   probe_VxT shape : {probe_VxT.shape} (voxels x trials)")

    # Prepare demeaned deviations
    print("-> Computing trial-by-trial demeaned fluctuations for pre-whitening...")
    mean_cue = cue_VxT.mean(axis=1, keepdims=True)
    devs_cue = (cue_VxT - mean_cue).T

    mean_probe = probe_VxT.mean(axis=1, keepdims=True)
    devs_probe = (probe_VxT - mean_probe).T

    devs_both = np.vstack([devs_cue, devs_probe])
    print(f"   devs_cue shape  : {devs_cue.shape} (trials x voxels)")
    print(f"   devs_probe shape: {devs_probe.shape} (trials x voxels)")
    print(f"   devs_both shape : {devs_both.shape} (2*trials x voxels)")

    # Build searchlight spheres
    print("-> Constructing 8mm searchlight spheres via KDTree...")
    ijk = np.column_stack(np.where(binary == 1))
    seeds = np.column_stack(
        coord_transform(ijk[:, 0], ijk[:, 1], ijk[:, 2], binary_mask_img.affine)
    )

    radius = 8
    allow_overlap = True

    _, A = new_apply_mask_and_get_affinity(
        seeds=seeds,
        niimg=cue_nii,
        radius=radius,
        allow_overlap=allow_overlap,
        mask_img=binary_mask_img,
    )

    n_seeds = A.shape[0]
    print(f"   Total spheres generated: {n_seeds}")
    print(f"   Affinity matrix shape  : {A.shape} (spheres x voxels)")

    # Execute parallel computation across spheres
    print(f"\n-> Running parallel sphere processing on {n_jobs} cores...")
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_sphere)(
            i, cue_VxT, probe_VxT, devs_cue, devs_probe, devs_both, A, n_trials, whiten="On"
        )
        for i in tqdm(range(n_seeds), desc="Searchlight Spheres", unit="sphere", file=sys.stdout)
    )

    # Stack results
    print("\n-> Stacking searchlight distance results...")
    corr_cc = np.vstack([r[0] for r in results])
    corr_pp = np.vstack([r[1] for r in results])
    corr_pc = np.vstack([r[2] for r in results])

    print(f"   Cue2Cue distance matrix shape   : {corr_cc.shape} (spheres x trials)")
    print(f"   Probe2Probe distance matrix shape: {corr_pp.shape} (spheres x trials)")
    print(f"   Probe2Cue distance matrix shape  : {corr_pc.shape} (spheres x trials)")

    # Save output .npy files
    out_cc = os.path.join(OUT_DIR, f"{sub}_sl_adjcorr_Cue2Cue_WTT.npy")
    out_pp = os.path.join(OUT_DIR, f"{sub}_sl_adjcorr_Probe2Probe_WTT.npy")
    out_pc = os.path.join(OUT_DIR, f"{sub}_sl_adjcorr_Probe2Cue_WTT.npy")

    np.save(out_cc, corr_cc)
    np.save(out_pp, corr_pp)
    np.save(out_pc, corr_pc)

    print(f"-> Saved distance arrays to disk:")
    print(f"   [1] {out_cc}")
    print(f"   [2] {out_pp}")
    print(f"   [3] {out_pc}")
    print(f"STEP 1 COMPLETE. Elapsed time: {datetime.now() - start_time}")


####################################################################
# STEP 2: EDS vs. IDS WITHIN-SUBJECT T-TEST
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


if args.sl_ttest:
    step2_start = datetime.now()
    print("\n" + "=" * 80)
    print(f"[{step2_start.strftime('%Y-%m-%d %H:%M:%S')}] STEP 2: RUNNING WITHIN-SUBJECT T-TEST (EDS vs. IDS / EDS vs. Stay / IDS vs. Stay)")
    print("=" * 80)

    saved_mask_path = os.path.join(OUT_DIR, f"{sub}_searchlight_mask.nii.gz")
    if os.path.exists(saved_mask_path):
        mask_nii = nib.load(saved_mask_path)
        print(f"-> Loaded reference mask from Step 1: {saved_mask_path}")
    else:
        mask_nii = nib.load(MASK_PATH)
        print(f"-> Loaded base mask: {MASK_PATH}")

    df = load_and_concat_behav_csvs(BEHAV_DIR, sub)
    print(f"-> Total behavioral rows loaded: {len(df)}")

    # Identify condition switch column
    if "hierarhical_switch_type" in df.columns:
        switch_col = "hierarhical_switch_type"
    elif "hierarchical_switch_type" in df.columns:
        switch_col = "hierarchical_switch_type"
    else:
        raise KeyError(f"Switch column not found! Available columns: {list(df.columns)}")

    df[switch_col] = df[switch_col].astype(str).str.strip()
    print(f"   Using condition column: '{switch_col}'")
    print(f"   Condition distribution across all trials:\n{df[switch_col].value_counts().to_string()}")

    # Ensure accuracy columns are numeric (1 = correct, 0 = incorrect/missed)
    df["subject_correct"] = pd.to_numeric(df["subject_correct"], errors="coerce").fillna(0).astype(int)
    df["prev_correct"] = df.groupby("block")["subject_correct"].shift(1).fillna(0).astype(int)
    df["prev2_correct"] = df.groupby("block")["subject_correct"].shift(2).fillna(0).astype(int)

    # Apply trial filtering policy:
    # 1. trial >= 3 (exclude trial 1->2 transition; include trial 2->3 onward)
    # 2. Current trial (n), previous trial (n-1), and n-2 trial must all be correct (1 -> 1 -> 1)
    MIN_TRIAL = 3
    valid_mask = (
        (df["trial"] >= MIN_TRIAL) &
        (df["subject_correct"] == 1) &
        (df["prev_correct"] == 1) &
        (df["prev2_correct"] == 1)
    )

    eds_indices = df.index[valid_mask & (df[switch_col] == "EDS")].to_numpy()
    ids_indices = df.index[valid_mask & (df[switch_col] == "IDS")].to_numpy()
    stay_indices = df.index[valid_mask & (df[switch_col] == "Stay")].to_numpy()

    print(f"\n-> Trial filtering criteria applied:")
    print(f"   Rule: trial >= {MIN_TRIAL} (trial 1->2 transition excluded; trial 2->3 onward included)")
    print(f"   Selected EDS trial indices count : {len(eds_indices)}")
    print(f"   Selected IDS trial indices count : {len(ids_indices)}")
    print(f"   Selected Stay trial indices count: {len(stay_indices)}")

    conditions = ["Cue2Cue", "Probe2Probe", "Probe2Cue"]
    contrasts = [
        ("EDS_vs_IDS", "EDS", "IDS"),
        ("EDS_vs_Stay", "EDS", "Stay"),
        ("IDS_vs_Stay", "IDS", "Stay"),
    ]

    for cond in conditions:
        print(f"\n" + "-" * 60)
        print(f"-> Processing condition: [{cond}]")
        npy_path = os.path.join(OUT_DIR, f"{sub}_sl_adjcorr_{cond}_WTT.npy")
        if not os.path.exists(npy_path):
            print(f"   [ERROR] Missing distance file: {npy_path}. Skipping.")
            continue

        ds_array = np.load(npy_path)
        print(f"   Loaded distance matrix shape: {ds_array.shape} (spheres x trials)")

        if ds_array.shape[1] != len(df):
            raise ValueError(
                f"Trial count mismatch in {cond}! "
                f"NIfTI volumes ({ds_array.shape[1]}) != Behavioral CSV rows ({len(df)})"
            )

        # Slice distance values for each condition
        vals_dict = {
            "EDS": ds_array[:, eds_indices],
            "IDS": ds_array[:, ids_indices],
            "Stay": ds_array[:, stay_indices],
        }

        for c_name in ["EDS", "IDS", "Stay"]:
            print(f"   {c_name.lower()}_vals shape: {vals_dict[c_name].shape} | Mean distance: {np.nanmean(vals_dict[c_name]):.4f}")

        # Vectorized two-sample t-tests across spheres for each contrast
        for contrast_name, cond_a, cond_b in contrasts:
            print(f"\n   Running contrast: [{contrast_name}] ({cond_a} - {cond_b})...")
            vals_a = vals_dict[cond_a]
            vals_b = vals_dict[cond_b]

            diff_vec = np.nanmean(vals_a, axis=1) - np.nanmean(vals_b, axis=1)
            t_res = stats.ttest_ind(vals_a, vals_b, axis=1, nan_policy="omit", equal_var=True)

            tval_vec = np.nan_to_num(t_res.statistic, nan=0.0)
            diff_vec = np.nan_to_num(diff_vec, nan=0.0)

            print(f"   t-statistic vector shape: {tval_vec.shape}")
            print(f"   t-statistic stats -> Min: {np.min(tval_vec):.2f}, Max: {np.max(tval_vec):.2f}, Mean: {np.mean(tval_vec):.2f}")

            # Unmask back to 3D brain images
            tval_out = os.path.join(OUT_DIR, f"{sub}_{cond}_{contrast_name}_tval.nii.gz")
            diff_out = os.path.join(OUT_DIR, f"{sub}_{cond}_{contrast_name}_diff.nii.gz")

            tval_nii = unmask(tval_vec, mask_nii)
            diff_nii = unmask(diff_vec, mask_nii)

            tval_nii.to_filename(tval_out)
            diff_nii.to_filename(diff_out)

            print(f"   Saved 3D NIfTI maps:")
            print(f"     [T-stat] {tval_out} (Volume shape: {tval_nii.shape})")
            print(f"     [Diff]   {diff_out} (Volume shape: {diff_nii.shape})")

    print(f"\nSTEP 2 COMPLETE. Elapsed time: {datetime.now() - step2_start}")
    
print("\n" + "=" * 80)
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALL TASKS FINISHED SUCCESSFULLY")
print(f"Log written to: {log_filepath}")
print("=" * 80)

# Clean up dual logger
dual_logger.close()