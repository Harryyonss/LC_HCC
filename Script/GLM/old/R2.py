import os
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

# -------------------  Path Settings  ---------------------
sub = "11113"
col_beta_path = f"/data/backed_up/shared/ME_7T_Pilot/GLMsingle/old/sub-{sub}/{sub}_Cue_TrialBetas.nii.gz"
sep_beta_path = f"/data/backed_up/shared/ME_7T_Pilot/GLMsingle/sub-{sub}/{sub}_Cue_TrialBetas.nii.gz"
mask_path = f"/data/backed_up/shared/ME_7T_Pilot/fmriprep/sub-{sub}/combined_mask.nii.gz"
meta_path = f"/data/backed_up/shared/ME_7T_Pilot/GLMsingle/sub-{sub}/{sub}_event_metadata.csv"

# -------------------  Data Loading via NiftiMasker  ---------------------
print("Loading and masking beta images...")
masker = NiftiMasker(mask_img=mask_path, standardize=False, detrend=False)
masker.fit()

# transform()을 사용하면 차원 순서와 무관하게 자동으로 (n_trials, n_voxels)로 추출됩니다.
b_col = masker.transform(col_beta_path)  # (n_trials, n_voxels)
b_sep = masker.transform(sep_beta_path)  # (n_trials, n_voxels)

n_trials, n_voxels = b_col.shape
print(f"Loaded betas successfully: {n_trials} trials across {n_voxels:,} voxels")

# -------------------  1. Trial-wise Spatial Correlation  ---------------------
trial_corrs = []
for t in range(n_trials):
    # 각 trial의 복셀 패턴 간 상관관계
    r = np.corrcoef(b_col[t], b_sep[t])[0, 1]
    trial_corrs.append(r)

mean_corr = np.mean(trial_corrs)
min_corr = np.min(trial_corrs)

print(f"\n==================== Beta Comparison (sub-{sub}) ====================")
print(f"Mean spatial correlation across trials : {mean_corr:.4f}")
print(f"Min spatial correlation across trials  : {min_corr:.4f}")

# -------------------  2. Condition-level Discriminability  ---------------------
if os.path.exists(meta_path):
    meta = pd.read_csv(meta_path)
    cue_meta = meta[meta['epoch'] == 'Cue'].reset_index(drop=True)
    
    # Tossed 제외 조건: EDS, IDS, Stay
    conditions = ['EDS', 'IDS', 'Stay']
    cond_patterns_col = {}
    cond_patterns_sep = {}
    
    for cond in conditions:
        idx = cue_meta[cue_meta['condition'] == cond].index.values
        cond_patterns_col[cond] = np.mean(b_col[idx], axis=0)
        cond_patterns_sep[cond] = np.mean(b_sep[idx], axis=0)
    
    print("\n--- Pairwise Condition Distance (1 - Pearson r) ---")
    pairs = [('EDS', 'IDS'), ('EDS', 'Stay'), ('IDS', 'Stay')]
    for c1, c2 in pairs:
        dist_col = 1.0 - np.corrcoef(cond_patterns_col[c1], cond_patterns_col[c2])[0, 1]
        dist_sep = 1.0 - np.corrcoef(cond_patterns_sep[c1], cond_patterns_sep[c2])[0, 1]
        diff_dist = dist_sep - dist_col
        print(f"{c1} vs {c2}: Collapsed={dist_col:.4f} | Separated={dist_sep:.4f} | Diff={diff_dist:+.4f}")
print("====================================================================\n")