import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import warnings
import sys
import re
#import pickle
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from datetime import datetime
import statsmodels.formula.api as smf
#from statsmodels.stats.multitest import fdrcorrection as fdr_correct
import scipy.stats as stats
from scipy import sparse
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn import neighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
#from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict
import nilearn
from nilearn import image, masking
from nilearn._utils.niimg_conversions import check_niimg_3d, safe_get_data
from nilearn.image.resampling import coord_transform
from nilearn.masking import unmask
from joblib import Parallel, delayed
from patsy import build_design_matrices   #to build marginal contrast.
#import matplotlib.pyplot as plt

####################################################################
# ARGPARSE
####################################################################

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess EEG data from subject",
        usage="[subject] [OPTIONS] ... ",
    )
    parser.add_argument("subject", help="5-digit subject id") # REQUIRED
    parser.add_argument("--searchlight_rsa",
                        help="run searchlight decoding, default is false",
                        default=False, action="store_true")
    parser.add_argument("--avg_rsa_900rois",
                        help="run searchlight decoding, default is false",
                        default=False, action="store_true")
    parser.add_argument("--plots_900rois",
                        help="run searchlight decoding, default is false",
                        default=False, action="store_true")
    parser.add_argument("--sl_reg",
                        help="run regression with searchlight decoding output, default is false",
                        default=False, action="store_true")
    parser.add_argument("--group_stats",
                        help="run group statistics on decoding regression outputs, default is false",
                        default=False, action="store_true")
    
    return parser

parser = init_argparse()
args = parser.parse_args(sys.argv[1:])
sub = args.subject



if args.avg_rsa_900rois:
    ####################################################################
    # Paths
    ####################################################################
    if os.path.exists("/Shared/"):
        mask_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
        mask_dir2 = "Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/ROIs" # "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
        mask_dir3 = "/data/backed_up/shared/ThalHi_MRI_2020/GLMsingle/ROIs" # "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
        out_dir  = "/Shared/lss_kahwang_hpc/data/HiRe/Neural_Distance_900ROIs"
        beh_dir = "/Shared/lss_kahwang_hpc/data/HiRe/"
    else:
        mask_dir = "/data/backed_up/shared/ThalHi_MRI_2020/GLMsingle/searchlightRSA" # "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
        mask_dir2 = "/data/backed_up/shared/ThalHi_MRI_2020/GLMsingle/ROIs" # "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
        mask_dir3 = "/data/backed_up/shared/ROIs" # "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
        out_dir  = "/data/backed_up/shared/HiRe/Neural_Distance_900ROIs" # "/mnt/nfs/lss/lss_kahwang_hpc/data/HiRe/Neural_Distance" # 
        beh_dir = "/data/backed_up/shared/HiRe/" # "/mnt/nfs/lss/lss_kahwang_hpc/data/HiRe/"
    os.makedirs(out_dir, exist_ok=True)

    # -- original searchlight mask
    mask_img_path = os.path.join(mask_dir, "searchlight_mask.nii.gz")
    mask_nii  = nib.load(mask_img_path)
    mask_data = mask_nii.get_fdata()
    
    # -- schaefer 900 rois mask
    mask900_nii  = nib.load(os.path.join(mask_dir2, "Schaefer900.nii.gz"))
    mask900_nii = image.resample_to_img(mask900_nii, mask_nii, interpolation='nearest') # source_img=img_to_resample, target_img=reference_for_resampling
    mask900_data = mask900_nii.get_fdata()
    # # schaefer 400 rois mask with thalamus and basal ganglia
    # mask400_nii  = nib.load(os.path.join(mask_dir2, "Morel_Striatum_Yeo400.nii.gz"))
    # mask400_data = mask400_nii.get_fdata()
    # mask400_data = np.where(mask400_data<=400, 0, mask400_data+900) # set cortical to zero, subcortical to start at 901
    # -- MOREL & BG masks (individual nii files)
    morel_bg_dict = {'AN':901, 'IL':902, 'LP':903, 'MD':904, 'PuL':905, 'PuM':906, 'VA':907, 'VL':908, 'VP':909, 
                     'Caudate':910, 'Globus_Pallidus':911, 'Putamen':912}
    mask_rois_data = mask900_data
    for key in morel_bg_dict.keys():
        if len(key) < 4:
            mask_morel_bg_nii = nib.load(os.path.join(mask_dir3, f"Morel_2mm_{key}.nii.gz"))
        else:
            mask_morel_bg_nii = nib.load(os.path.join(mask_dir3, f"BG_{key}_2.5mm.nii.gz"))
        mask_morel_bg_nii = image.resample_to_img(mask_morel_bg_nii, mask_nii, interpolation='nearest') # source_img=img_to_resample, target_img=reference_for_resampling
        mask_morel_bg_data = mask_morel_bg_nii.get_fdata()
        mask_morel_bg_data = np.where(mask_morel_bg_data>0, morel_bg_dict[key], 0) # set cortical to zero, subcortical to start at 901
        # -- combine
        mask_rois_data = mask_morel_bg_data + mask_rois_data # 900 rois with morel and striatum (+912)
    mask_rois_img = image.new_img_like(ref_niimg=mask900_nii, data=mask_rois_data) #, affine=mask900_nii.affine)
    
    # -- vectorize to match searchlight array
    rois_vec = masking.apply_mask(mask_rois_img, mask_nii)
    rois_vec_int = rois_vec.astype(int)
    
    ####################################################################
    # Process subjects
    ####################################################################
    behavior_csv = os.path.join(beh_dir, "Switch_Costs", "thalhi_mri_master_df.csv")
    df = pd.read_csv(behavior_csv)
    
    out_dict = {'subject':[], 'roi':[], 'block':[], 'trial':[], 'ds':[]}
    
    for sub_id in df['subject'].unique():
        # skip excluded subjects
        if int(sub_id) in [10012, 10028, 10039, 10054, 10072]:
            continue
        
        print(f"→ Processing subject {sub_id}")
        
        subj_file = os.path.join(mask_dir, f"{sub_id}_sl_adjcorr_resRTWTT.npy")
        if not os.path.exists(subj_file):
            print(f"✗ Missing data for {sub_id}")
        # -- load neural distance
        ds_array = np.load(subj_file) # voxels x trials
        n_spheres = ds_array.shape[0]
        
        # -- apply 900 rois
        for roi in range(1, rois_vec_int.max()+1):
            # get voxels in current roi
            c_idx_vec = np.where(rois_vec_int==roi)[0]
            # calculate and add avg value
            c_ds_vec = np.nanmean(ds_array[c_idx_vec,:], axis=0) # avg over voxels but keep trials
            count = 0 # lazy
            for blk in range(1,9):
                for trl in range(51):
                    # add subject and roi and trial
                    out_dict['subject'].append(sub_id)
                    out_dict['roi'].append(roi)
                    out_dict['block'].append(blk)
                    out_dict['trial'].append(trl)
                    
                    out_dict['ds'].append(c_ds_vec[count])
                    count += 1
        
    # save out data frame
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(os.path.join(beh_dir, "Neural_Distance_900ROIs", "neural_distance__900ROIs_Morel_Striatum.csv"))

if args.plots_900rois:
    #from statsmodels.stats.multitest import fdrcorrection as fdr_correct
    ####################################################################
    # Paths
    ####################################################################
    if os.path.exists("/Shared/"):
        mask_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
        mask_dir2 = "Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/ROIs" # "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
        mask_dir3 = "/data/backed_up/shared/ThalHi_MRI_2020/GLMsingle/ROIs" # "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
        out_dir  = "/Shared/lss_kahwang_hpc/data/HiRe/Neural_Distance_900ROIs"
        beh_dir = "/Shared/lss_kahwang_hpc/data/HiRe/"
    else:
        mask_dir = "/data/backed_up/shared/ThalHi_MRI_2020/GLMsingle/searchlightRSA" # "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
        mask_dir2 = "/data/backed_up/shared/ThalHi_MRI_2020/GLMsingle/ROIs" # "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
        mask_dir3 = "/data/backed_up/shared/ROIs" # "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
        out_dir  = "/data/backed_up/shared/HiRe/Neural_Distance_900ROIs" # "/mnt/nfs/lss/lss_kahwang_hpc/data/HiRe/Neural_Distance" # 
        beh_dir = "/data/backed_up/shared/HiRe/" # "/mnt/nfs/lss/lss_kahwang_hpc/data/HiRe/"
    os.makedirs(out_dir, exist_ok=True)

    # -- original searchlight mask
    mask_img_path = os.path.join(mask_dir, "searchlight_mask.nii.gz")
    mask_nii  = nib.load(mask_img_path)
    mask_data = mask_nii.get_fdata()
    
    # -- schaefer 900 rois mask
    mask900_nii  = nib.load(os.path.join(mask_dir2, "Schaefer900.nii.gz"))
    mask900_nii = image.resample_to_img(mask900_nii, mask_nii, interpolation='nearest') # source_img=img_to_resample, target_img=reference_for_resampling
    mask900_data = mask900_nii.get_fdata()
    # # schaefer 400 rois mask with thalamus and basal ganglia
    # mask400_nii  = nib.load(os.path.join(mask_dir2, "Morel_Striatum_Yeo400.nii.gz"))
    # mask400_data = mask400_nii.get_fdata()
    # mask400_data = np.where(mask400_data<=400, 0, mask400_data+900) # set cortical to zero, subcortical to start at 901
    # -- MOREL & BG masks (individual nii files)
    morel_bg_dict = {'AN':901, 'IL':902, 'LP':903, 'MD':904, 'PuL':905, 'PuM':906, 'VA':907, 'VL':908, 'VP':909, 
                     'Caudate':910, 'Globus_Pallidus':911, 'Putamen':912}
    mask_rois_data = mask900_data
    for key in morel_bg_dict.keys():
        if len(key) < 4:
            mask_morel_bg_nii = nib.load(os.path.join(mask_dir3, f"Morel_2mm_{key}.nii.gz"))
        else:
            mask_morel_bg_nii = nib.load(os.path.join(mask_dir3, f"BG_{key}_2.5mm.nii.gz"))
        mask_morel_bg_nii = image.resample_to_img(mask_morel_bg_nii, mask_nii, interpolation='nearest') # source_img=img_to_resample, target_img=reference_for_resampling
        mask_morel_bg_data = mask_morel_bg_nii.get_fdata()
        mask_morel_bg_data = np.where(mask_morel_bg_data>0, morel_bg_dict[key], 0) # set cortical to zero, subcortical to start at 901
        # -- combine
        mask_rois_data = mask_morel_bg_data + mask_rois_data # 900 rois with morel and striatum (+912)
    mask_rois_img = image.new_img_like(ref_niimg=mask900_nii, data=mask_rois_data) #, affine=mask900_nii.affine)
    
    # -- vectorize to match searchlight array
    rois_vec = masking.apply_mask(mask_rois_img, mask_nii)
    rois_vec_int = rois_vec.astype(int)
    
    ####################################################################
    # save results into rois and plot
    ####################################################################
    modelv = "model1" # model1 | model2 | model2A | model2R
    
    roi_csv = pd.read_csv(os.path.join(beh_dir, "Neural_Distance_900ROIs", "ND_model_fits", f"roi-1_{modelv}-summary.csv"))
    regressor_names = list(roi_csv.iloc[:,0])
    
    for idx in range(len(regressor_names)):
        c_reg_orig = regressor_names[idx]
        c_reg_out = regressor_names[idx]
        # quick fix in case there are () or : 
        if '(' in c_reg_out:
            c_reg_out=c_reg_out.split('(')[1]
            c_reg_out=c_reg_out.split(')')[0]
        if ':' in c_reg_out:
            c_reg_out = c_reg_out.replace(":", "_x_")
        
        tval_vec = np.zeros(len(rois_vec_int))
        bval_vec = np.zeros(len(rois_vec_int))
        out_dict = {'roi':[], 'beta':[], 'std_err':[], 'df':[], 'tvalue':[], 'pvalue':[], 'fdr_corrected':[], 'fdr_rejected_null':[]}
        for c_roi in range(1, rois_vec_int.max()+1):
            roi_csv = pd.read_csv(os.path.join(beh_dir, "Neural_Distance_900ROIs", "ND_model_fits", f"roi-{c_roi}_{modelv}-summary.csv"))
            # grab values from df
            beta_value = list(roi_csv['Estimate'])[idx]
            se_value = list(roi_csv['Std. Error'])[idx]
            df_value = list(roi_csv['df'])[idx]
            t_value = list(roi_csv['t value'])[idx]
            p_value = list(roi_csv['Pr(>|t|)'].round(6))[idx]
            # add values to output dict
            out_dict['roi'].append(c_roi)
            out_dict['beta'].append(beta_value)
            out_dict['std_err'].append(se_value)
            out_dict['df'].append(df_value)
            out_dict['tvalue'].append(t_value)
            out_dict['pvalue'].append(p_value)
            
            # -- get index for voxels in current roi
            c_idx_vec = np.where(rois_vec_int==c_roi)[0]
            # then add to data vec
            bval_vec[c_idx_vec] = beta_value # add t values to this roi
            tval_vec[c_idx_vec] = t_value # add beta values to this roi
        
        # -- fdr correction
        fdr_reject, pval_fdr = fdr_correct(np.asarray(list(out_dict['pvalue'])), alpha=0.05, method='indep', is_sorted=False)
        out_dict['fdr_corrected'] = pval_fdr
        out_dict['fdr_rejected_null'] = fdr_reject
        # loop through rois again
        rejected_vec = np.zeros(len(rois_vec_int))
        for c_roi in range(1, rois_vec_int.max()+1):
            # -- get index for voxels in current roi
            c_idx_vec = np.where(rois_vec_int==c_roi)[0]
            # then add to data vec
            rejected_vec[c_idx_vec] = fdr_reject[c_roi-1]
        
        # -- save out df for this regressor
        out_df = pd.DataFrame(out_dict)
        out_df.to_csv(os.path.join(beh_dir, "Neural_Distance_900ROIs", f"{modelv}__regressor-{c_reg_out}.csv"))
        # -- save out fmri mask for this regressor
        unmask(bval_vec, mask_nii).to_filename(
            os.path.join(beh_dir, "Neural_Distance_900ROIs", "plots", f"{modelv}__regressor-{c_reg_out}__beta.nii.gz")
        )
        unmask(tval_vec, mask_nii).to_filename(
            os.path.join(beh_dir, "Neural_Distance_900ROIs", "plots", f"{modelv}__regressor-{c_reg_out}__tval.nii.gz")
        )
        unmask(rejected_vec, mask_nii).to_filename(
            os.path.join(beh_dir, "Neural_Distance_900ROIs", "plots", f"{modelv}__regressor-{c_reg_out}__rejectNull.nii.gz")
        )
    
        
        

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - -      SEARCHLIGHT  DECODING      - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Searchlight adj trial neural distance script
# - Fast custom affinity function for spheres
# - Computes correlation distances between adjacent trials --- Neural Distance!
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if args.searchlight_rsa:
    # ======== Custom affinity function (adapted from nilearn) ========
    def new_apply_mask_and_get_affinity(seeds, niimg, radius, allow_overlap, mask_img=None):
        """
        Build a sparse sphere-to-voxel affinity matrix.

        seeds         : array (n_seeds, 3) in world (mm) coordinates
        niimg         : Nifti1Image of 4D data (for affine) or None
        radius        : searchlight radius in mm
        allow_overlap : bool, raise error if overlap detected
        mask_img      : 3D binary mask (Nifti1Image or filename)

        Returns:
            X        : data matrix (unused here if niimg=None)
            A_sparse : sparse matrix (n_seeds x n_voxels) boolean
        """
        seeds = np.asarray(seeds)
        if seeds.ndim == 1:
            seeds = seeds[np.newaxis, :]

        # Load or derive mask coordinates and optional data
        if niimg is None:
            mask, affine = masking.load_mask_img(mask_img)
            mask_coords = np.column_stack(np.where(mask))
            X = None
        else:
            affine = niimg.affine
            mask_img = check_niimg_3d(mask_img)
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

        # Convert mask voxel indices to world space
        mask_coords_world = coord_transform(
            mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2], affine
        )
        mask_coords_world = np.column_stack(mask_coords_world)

        # Map seeds from world → voxel indices
        inv_affine = np.linalg.inv(affine)
        seeds_voxel = (
            np.column_stack([seeds, np.ones(len(seeds))]) @ inv_affine.T
        )[:, :3]
        nearest_voxels = np.round(seeds_voxel).astype(int)
        coord_dict = {tuple(c): i for i, c in enumerate(mask_coords)}
        nearests = [coord_dict.get(tuple(coord), None) for coord in nearest_voxels]

        # KDTree for radius queries
        tree = neighbors.KDTree(mask_coords_world)
        neighbors_list = tree.query_radius(seeds, r=radius)

        n_seeds = len(seeds)
        n_voxels = len(mask_coords)
        A_sparse = sparse.lil_matrix((n_seeds, n_voxels), dtype=bool)

        # Fill boolean matrix
        for i, (nbrs, nearest) in enumerate(zip(neighbors_list, nearests)):
            if nbrs.size > 0:
                A_sparse[i, nbrs] = True
            if nearest is not None:
                A_sparse[i, nearest] = True

        # Ensure seed itself is present
        seed_world = coord_transform(
            seeds_voxel[:, 0], seeds_voxel[:, 1], seeds_voxel[:, 2], affine
        )
        seed_world = np.column_stack(seed_world)
        _, seed_idx = tree.query(seed_world, k=1)
        for i, idx in enumerate(seed_idx):
            A_sparse[i, idx] = True

        # Sanity checks
        sphere_sizes = np.asarray(A_sparse.sum(axis=1)).ravel()
        empty = np.nonzero(sphere_sizes == 0)[0]
        if len(empty):
            raise ValueError(f"Empty spheres: {empty}")
        if not allow_overlap and np.any(A_sparse.sum(axis=0) >= 2):
            raise ValueError("Overlap detected between spheres")

        return X, A_sparse.tocsr()


    # ======== Shrinkage & whitening (Walther 2016, RSA toolbox) ========
    def compute_shrunk_covariance_and_whitening_matrix(x, return_shrunk_cov=False):
        """
        Compute shrunk covariance matrix and its inverse square root (whitening).
        Optionally return the shrunk covariance itself.

        x : data matrix, shape (t, n) with t observations × n voxels
        """
        t, n = x.shape
        if t < 2:
            print(f"Warning: not enough samples ({t}). Returning identity.")
            identity_matrix = np.eye(n)
            if return_shrunk_cov:
                return identity_matrix, identity_matrix
            else:
                return identity_matrix

        # Demean data
        x_demeaned = x - x.mean(0)
        sample_cov = (1.0 / (t - 1)) * np.dot(x_demeaned.T, x_demeaned)
        prior = np.diag(np.diag(sample_cov))

        # Shrinkage coefficient
        num = np.sum((sample_cov - prior) ** 2)
        den = np.sum((x_demeaned**2 - np.diag(sample_cov)[np.newaxis, :]) ** 2) / t
        shrinkage = 1.0
        if den > 1e-10 and num > 1e-10:
            shrinkage = num / den
        shrinkage = np.clip(shrinkage, 0.0, 1.0)

        if t < n:
            print(f"Warning: samples ({t}) < voxels ({n}). Increasing shrinkage.")
            shrinkage = max(shrinkage, 1 - ((t - 1) / (n * 1.5)))
            shrinkage = np.clip(shrinkage, 0.0, 1.0)

        sigma_shrunk = shrinkage * prior + (1 - shrinkage) * sample_cov
        epsilon = 1e-7

        # Ensure positive definiteness
        try:
            np.linalg.cholesky(sigma_shrunk)
        except np.linalg.LinAlgError:
            sigma_shrunk = sigma_shrunk + epsilon * np.eye(n)

        # Whitening matrix
        whitening_matrix = np.eye(n)
        try:
            evals, evecs = np.linalg.eigh(sigma_shrunk)
            evals_floored = np.maximum(evals, epsilon)
            evals_sqrt_inv = 1.0 / np.sqrt(evals_floored)
            whitening_matrix = np.dot(evecs * evals_sqrt_inv, evecs.T)
        except np.linalg.LinAlgError as err:
            print(f"Whitening failed: {err}. Falling back to prior.")
            whitening_matrix = np.eye(n)

        if return_shrunk_cov:
            return whitening_matrix, sigma_shrunk
        else:
            return whitening_matrix


    # ======== Function to process one sphere ========
    def process_sphere(idx, tb_VxT, devs, A, n_trials, whiten="On"):
        vox_inds = A[idx].indices
        if len(vox_inds) == 0:
            return np.full(n_trials, np.nan), np.full(n_trials, np.nan)

        b_roi = tb_VxT[vox_inds, :]
        dev_roi = devs[:, vox_inds]

        if whiten == "On":
            if dev_roi.shape[0] >= 2:
                W, _ = compute_shrunk_covariance_and_whitening_matrix(dev_roi, return_shrunk_cov=True)
            else:
                W = np.eye(len(vox_inds))
            wb = W.dot(b_roi)
        else:
            wb = b_roi

        data_tv = wb.T
        corrd = np.full(n_trials, np.nan)
        for t in range(1, n_trials):
            r = np.corrcoef(data_tv[t - 1], data_tv[t])[0, 1]
            corrd[t] = np.sqrt(2 * (1 - r))
        return corrd, np.full(n_trials, np.nan)


    # ======== Main searchlight adjacency neural distance pipeline ========
    def run_func(subs, in_fn, out_fn, mnn):
        mask_dir = "/Shared/lss_kahwang_hpc/ROIs"
        GLM_dir = "/data/backed_up/shared/ME_7T_Pilot/GLMsingle"
        out_dir = "/data/backed_up/shared/ME_7T_Pilot/searchlightRSA"
        # os.makedirs(out_dir, exist_ok=True)

        ROI_mask = "Schaefer400+Morel+BG_2.5.nii.gz"
        mask_path = os.path.join(mask_dir, ROI_mask)
        mask_nii = nib.load(mask_path)
        mask_data = mask_nii.get_fdata()
        binary = (mask_data > 0).astype(np.int8)
        binary_mask_img = nilearn.image.new_img_like(mask_nii, binary)

        ijk = np.column_stack(np.where(binary == 1))
        seeds = np.column_stack(
            coord_transform(ijk[:, 0], ijk[:, 1], ijk[:, 2], mask_nii.affine)
        )

        radius = 8
        allow_overlap = True
        n_jobs = 16

        for s in [subs]:
            print(f"\n=== Subject {s} ===")
            subj = os.path.join(GLM_dir, f"sub-{s}")
            tb_file = os.path.join(subj, "%s_%s.nii.gz" % (s, in_fn))
            if not os.path.exists(tb_file):
                print("Missing TrialBetas:", tb_file)
                continue

            tb_nii = nib.load(tb_file)
            n_trials = tb_nii.shape[3]
            tb_mat = masking.apply_mask(tb_nii, binary_mask_img)
            tb_VxT = tb_mat.T
            del tb_mat

            mean_beta = tb_VxT.mean(axis=1, keepdims=True)
            devs = (tb_VxT - mean_beta).T

            X, A = new_apply_mask_and_get_affinity(
                seeds=seeds,
                niimg=tb_nii,
                radius=radius,
                allow_overlap=allow_overlap,
                mask_img=binary_mask_img,
            )

            n_seeds = A.shape[0]
            print(f"  # spheres: {n_seeds}")

            results = Parallel(n_jobs=n_jobs, verbose=2)(
                delayed(process_sphere)(i, tb_VxT, devs, A, n_trials, whiten=mnn)
                for i in range(n_seeds)
            )

            corr_adj = np.vstack([c for c, _ in results])

            if mnn == "On":
                mnn_fn = "T"
            else:
                mnn_fn = "F"

            np.save(
                os.path.join(out_dir, "%s_sl_adjcorr_%s%s.npy" % (s, out_fn, mnn_fn)),
                corr_adj,
            )
            print(f"Saved shapes: corr {corr_adj.shape}")

    if __name__ == "__main__":
        subs = input()
        start = datetime.now()
        print("Subject start:", start)
        run_func(subs, in_fn="TrialBetas_resRT", out_fn="resRTWT", mnn="On")
        print("End:", datetime.now(), "Duration:", datetime.now() - start)

    #end of line







# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - -     REGRESSION/CORRELATIONS     - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  model regression on neural distance per subject
# - Fits OLS per searchlight sphere
# - Writes beta/t-stat maps for each contrast/effect
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if args.sl_reg:
    ####################################################################
    # Clean regressor names for filenames
    ####################################################################
    def clean_regressor_name(reg_name):
        clean = reg_name
        clean = re.sub(r"C\(([^,]+),\s*Treatment\(reference=['\"]?([^'\"]+)['\"]?\)\)\[T\.([^\]]+)\]",
                    r"\1_vs_\3", clean)
        clean = re.sub(r"C\(([^)]+)\)\[T\.([^\]]+)\]", r"\1_vs_\2", clean)
        clean = re.sub(r"[^\w]+", "_", clean)
        clean = re.sub(r"_+", "_", clean)
        clean = clean.strip("_")
        return clean

    ####################################################################
    # Paths
    ####################################################################
    if os.path.exists("/Shared/"):
        data_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
        mask_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
        out_dir  = "/Shared/lss_kahwang_hpc/data/HiRe/Neural_Distance"
        beh_dir = "/Shared/lss_kahwang_hpc/data/HiRe/"
    else:
        data_dir = "/data/backed_up/shared/ThalHi_MRI_2020/GLMsingle/searchlightRSA" # "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightNeural_Distance" # 
        mask_dir = "/data/backed_up/shared/ThalHi_MRI_2020/GLMsingle/searchlightRSA" # "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
        out_dir  = "/data/backed_up/shared/HiRe/Neural_Distance" # "/mnt/nfs/lss/lss_kahwang_hpc/data/HiRe/Neural_Distance" # 
        beh_dir = "/data/backed_up/shared/HiRe/" # "/mnt/nfs/lss/lss_kahwang_hpc/data/HiRe/"
    os.makedirs(out_dir, exist_ok=True)

    mask_img_path = os.path.join(mask_dir, "searchlight_mask.nii.gz")
    mask_nii  = nib.load(mask_img_path)
    mask_data = mask_nii.get_fdata()


    ####################################################################
    # Spherewise regression (one sphere)
    ####################################################################
    def regress_one_sphere(sphere_index, sdf, ds_array, model_syntax):
        """
        Fit OLS for one searchlight sphere, extract betas/t-values.
        All main_effect_* computed using **emmeans-style marginal means**. pool accors dist!
        perceptual_change_c slope uses weighting to account for balance.
        """
        sdf = sdf.copy()
        sdf["ds"] = ds_array[sphere_index, :]
        beta = {}
        tval = {}

        try:
            model = smf.ols(model_syntax, data=sdf).fit()
            params = model.params

            # raw coefficients
            beta.update(params.to_dict())
            tval.update(model.tvalues.to_dict())
            
            ############################################################
            # save out raw coefficients for checks
            ############################################################
            beta["main_effect_perceptual_change_c"] = float(ct.effect)
            tval["main_effect_perceptual_change_c"] = float(ct.tvalue)

            ############################################################
            # GET PARAMETER NAMES
            ############################################################
            name_pc       = "perceptual_change_c"
            name_eds_pc   = "C(hierarchical_switch_type, Treatment(reference='IDS'))[T.EDS]:perceptual_change_c"
            name_stay_pc  = "C(hierarchical_switch_type, Treatment(reference='IDS'))[T.Stay]:perceptual_change_c"

            # weighting of perceptual_change_c
            freq_tt = sdf['hierarchical_switch_type'].value_counts(normalize=True)
            p_EDS   = float(freq_tt.get('EDS', 0))
            p_Stay  = float(freq_tt.get('Stay', 0))

            ############################################################
            # update, get contrast using design-matrix averaging (not just as ref level), get design matrix out in one go.
            # this is how emmeans handel this
            ############################################################
            design_info = model.model.data.design_info

            def _mean_exog(df_mod):
                X = build_design_matrices([design_info], df_mod)[0]
                return X.mean(axis=0)

            def emm_L(level_A, level_B):
                """
                get the margianl B - A across distributions
                """
                dfA = sdf.copy()
                dfB = sdf.copy()

                for col, val in level_A.items():
                    dfA[col] = val
                for col, val in level_B.items():
                    dfB[col] = val

                xA = _mean_exog(dfA)
                xB = _mean_exog(dfB)
                return (xB - xA)

            ############################################################
            # Average slope for perceptual_change_c
            ############################################################
            # slope_IDS  = beta_pc
            # slope_EDS  = beta_pc + beta_eds_pc, diff from IDS ref level
            # slope_Stay = beta_pc + beta_stay_pc
            L_pc = [
                (1 if n == name_pc else
                p_EDS if n == name_eds_pc else
                p_Stay if n == name_stay_pc else
                0)
                for n in params.index
            ]
            ct = model.t_test(L_pc) #linear contrast
            beta["main_effect_perceptual_change_c"] = float(ct.effect)
            tval["main_effect_perceptual_change_c"] = float(ct.tvalue)


            ############################################################
            # slopes for perceptual_change_c
            ############################################################
            # IDS
            L_IDS = [1 if n == name_pc else 0 for n in params.index]
            ct = model.t_test(L_IDS)
            beta["perceptual_change_for_IDS"] = float(ct.effect)
            tval["perceptual_change_for_IDS"] = float(ct.tvalue)

            # EDS
            L_EDS = [1 if n in (name_pc, name_eds_pc) else 0 for n in params.index]
            ct = model.t_test(L_EDS)
            beta["perceptual_change_for_EDS"] = float(ct.effect)
            tval["perceptual_change_for_EDS"] = float(ct.tvalue)

            # Stay
            L_Stay = [1 if n in (name_pc, name_stay_pc) else 0 for n in params.index]
            ct = model.t_test(L_Stay)
            beta["perceptual_change_for_Stay"] = float(ct.effect)
            tval["perceptual_change_for_Stay"] = float(ct.tvalue)


            ############################################################
            # main effects
            ############################################################
            # EDS – IDS
            L = emm_L({'hierarchical_switch_type': 'IDS'},
                    {'hierarchical_switch_type': 'EDS'})
            ct = model.t_test(L)
            beta["main_effect_EDS_v_IDS"] = float(ct.effect)
            tval["main_effect_EDS_v_IDS"] = float(ct.tvalue)

            # IDS – Stay
            L = emm_L({'hierarchical_switch_type': 'Stay'},
                    {'hierarchical_switch_type': 'IDS'})
            ct = model.t_test(L)
            beta["main_effect_IDS_v_Stay"] = float(ct.effect)
            tval["main_effect_IDS_v_Stay"] = float(ct.tvalue)

            #response_repeat: switch – repeat
            L = emm_L({'response_repeat': 'repeat'},
                    {'response_repeat': 'switch'})
            ct = model.t_test(L)
            beta["main_effect_response_repeat"] = float(ct.effect)
            tval["main_effect_response_repeat"] = float(ct.tvalue)

            #task_repeat: switch – repeat
            L = emm_L({'task_repeat': 'repeat'},
                    {'task_repeat': 'switch'})
            ct = model.t_test(L)
            beta["main_effect_task_repeat"] = float(ct.effect)
            tval["main_effect_task_repeat"] = float(ct.tvalue)

            #probe_repeat: switch – repeat
            L = emm_L({'probe_repeat': 'repeat'},
                    {'probe_repeat': 'switch'})
            ct = model.t_test(L)
            beta["main_effect_probe_repeat"] = float(ct.effect)
            tval["main_effect_probe_repeat"] = float(ct.tvalue)

            # prev_target_feature_switch: repeat – switch
            L = emm_L({'prev_target_feature_switch': 'switch'},
                    {'prev_target_feature_switch': 'repeat'})
            ct = model.t_test(L)
            beta["main_effect_prev_target_feature_switch"] = float(ct.effect)
            tval["main_effect_prev_target_feature_switch"] = float(ct.tvalue)

            # Errors: error – correct
            L = emm_L({'Errors': 'correct'},
                    {'Errors': 'error'})
            ct = model.t_test(L)
            beta["main_effect_Errors"] = float(ct.effect)
            tval["main_effect_Errors"] = float(ct.tvalue)

            # Prev_Errors: error – correct
            L = emm_L({'Prev_Errors': 'correct'},
                    {'Prev_Errors': 'error'})
            ct = model.t_test(L)
            beta["main_effect_Prev_Errors"] = float(ct.effect)
            tval["main_effect_Prev_Errors"] = float(ct.tvalue)

            ############################################################
            # interaction: response_repeat × task_repeat
            ############################################################
            name_rrtr = "C(response_repeat)[T.switch]:C(task_repeat)[T.switch]"
            if name_rrtr in params.index:
                L = [1 if n == name_rrtr else 0 for n in params.index]
                ct = model.t_test(L)
                beta["interaction_response_repeat_x_task_repeat"] = float(ct.effect)
                tval["interaction_response_repeat_x_task_repeat"] = float(ct.tvalue)

        except Exception:
            pass

        return beta, tval


    ####################################################################
    # Process one subject
    ####################################################################
    def process_subject(sub_id, fn, model_syntax, model_tag):
        print(f"→ Processing subject {sub_id}")
        subj_out = os.path.join(out_dir, sub_id)
        os.makedirs(subj_out, exist_ok=True)
        
        subj_file = os.path.join(mask_dir, f"{sub_id}_sl_adjcorr_{fn}.npy")
        if not os.path.exists(subj_file):
            print(f"✗ Missing data for {sub_id}")
            return
        
        # -- load neural distance
        ds_array = np.load(subj_file)
        n_spheres = ds_array.shape[0]
        
        # -- pull out subject level behavioral data
        sdf = df[df["subject"] == int(sub_id)].copy().sort_values(["block", "trial"]).reset_index(drop=True)
        # -- apply trial filters 
        sdf = sdf[(sdf['trial'] != 0) & (np.abs(sdf['zRT']) <= 3)].reset_index(drop=True) # drop first trial AND outlier zRTs
        sdf = sdf[sdf['missed_response'] != 1].reset_index(drop=True) # drop missed responses on current trial (but leave missed on previous trial)
        # add one more filter to get rid of error->error trials
        sdf = sdf[sdf['err_trans'] != 'error-error'].reset_index(drop=True) # get rid of trials where error is followed by error
        
        # -- set variables as categorical
        # categories
        for col in ["probe_repeat", "response_repeat", "task_repeat",
                    "hierarchical_switch_type", "prev_target_feature_switch",
                    "Errors", "Prev_Errors"]:
            sdf[col] = sdf[col].astype("category")
        # center perceptual_change
        sdf['perceptual_change_c'] = sdf['perceptual_change'] - 1.5 # range 0-3 means 1.5 is center point
        
        # -- align ds_array to remaining trials
        trial_vec = (sdf['block'] - 1) * 51 + sdf['trial'] # matches original neural distance regression
        ds_array = ds_array[:, trial_vec]
        
        # -- run regressions
        results = Parallel(n_jobs=10, verbose=5)(
            delayed(regress_one_sphere)(i, sdf, ds_array, model_syntax)
            for i in range(n_spheres)
        )

        # -- collect all regressor names
        reg_names = sorted(set(k for beta, _ in results for k in beta.keys()))

        # -- prepare storage
        betas = {r: np.full(n_spheres, np.nan) for r in reg_names}
        tvals = {r: np.full(n_spheres, np.nan) for r in reg_names}

        for i, (beta, tval) in enumerate(results):
            for r in reg_names:
                betas[r][i] = beta.get(r, np.nan)
                tvals[r][i] = tval.get(r, np.nan)

        # -- save
        subj_out = os.path.join(out_dir, sub_id)
        os.makedirs(subj_out, exist_ok=True)

        for r in reg_names:
            r_clean = clean_regressor_name(r)
            unmask(betas[r], mask_nii).to_filename(
                os.path.join(subj_out, f"{r_clean}_{model_tag}_{fn}_beta.nii.gz")
            )
            unmask(tvals[r], mask_nii).to_filename(
                os.path.join(subj_out, f"{r_clean}_{model_tag}_{fn}_tval.nii.gz")
            )

        print(f"✓ Finished subject {sub_id}")


    ####################################################################
    # Load master dataframe
    ####################################################################
    behavior_csv = os.path.join(beh_dir, "Switch_Costs", "thalhi_mri_master_df.csv")
    df = pd.read_csv(behavior_csv)
    
    # recode error and post-error variables
    df['Errors'] = (df['accuracy'] != 1).astype(int).replace({1:'error',0:'correct'})
    df['Prev_Errors'] = (df['prev_accuracy'] != 1).astype(int).replace({1:'error',0:'correct'})

    # re-name a couple switch/repeat columns
    df["response_repeat"] = df["response_switch"]
    df["task_repeat"]     = df["task_switch"]
    df["probe_repeat"]    = df["probe_switch"]

    # add two columns
    df['missed_response'] = np.where(df["zRT"]==-99, 1, 0)
    df['err_trans'] = np.where( ((df["Errors"]=='error')&(df["Prev_Errors"]=='error')), 'error-error', np.where( ((df["Errors"]=='correct')&(df["Prev_Errors"]=='error')), 'error-correct', np.where( ((df["Errors"]=='error')&(df["Prev_Errors"]=='correct')), 'correct-error', 'correct-correct' ) ) )

    # add zRT_diff and modify zRT slightly to allow us to keep trials following a missed response 
    # (change is relative to avg. RT on those trials)
    df["zRT"] = np.where(df["zRT"]==-99, 0, df["zRT"]) # replace missed trials with zRT of 0
    df["zRT_diff"] = 0
    df["zRT_diff"][1:] = np.asarray(df["zRT"][1:]) - np.asarray(df["zRT"][:-1])


    ####################################################################
    # Run
    ####################################################################
    if __name__ == "__main__":
        subjects = sub # input("Subject ID: ")

        # model_old = (
        #     "ds ~ zRT_diff + zRT + C(Errors, Treatment(reference='correct')) + "
        #     "C(Prev_Errors, Treatment(reference='correct')) + "
        #     "C(hierarchical_switch_type, Treatment(reference='IDS')) * perceptual_change_c + "
        #     "C(response_repeat) * C(task_repeat) + "
        #     "C(task_repeat) * C(prev_target_feature_switch, Treatment(reference='switch')) + "
        #     "C(response_repeat) * C(prev_target_feature_switch, Treatment(reference='switch')) + "
        #     "C(probe_repeat)"
        # )
        
        # main effects: zRT_diff , zRT , hierarchical_switch_type , task_repeat , response_repeat , probe_repeat , prev_target_feature_switch , Errors , Prev_Errors
        model1 = (
            "ds ~ zRT_diff + zRT + C(Errors, Treatment(reference='correct')) + "
            "C(Prev_Errors, Treatment(reference='correct')) + "
            "C(hierarchical_switch_type, Treatment(reference='IDS')) * perceptual_change_c + "
            "C(response_repeat, Treatment(reference='switch')) * C(task_repeat, Treatment(reference='switch')) + "
            "C(probe_repeat, Treatment(reference='switch')) + "
            "C(prev_target_feature_switch, Treatment(reference='switch'))"
        )

        process_subject(subjects, fn="resRTWTT", model_syntax=model1, model_tag="ProbeFeatureRTAccuModel")







# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - -       GROUP STATS DECODING      - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import glob
import subprocess
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if args.group_stats:
    #### model regression
    regressors = [
        "main_effect_EDS_v_IDS_ProbeFeatureRTAccuModel_resRTWTT",
        "main_effect_IDS_v_Stay_ProbeFeatureRTAccuModel_resRTWTT",
        "main_effect_perceptual_change_c_ProbeFeatureRTAccuModel_resRTWTT",
        "main_effect_prev_target_feature_switch_ProbeFeatureRTAccuModel_resRTWTT",
        "main_effect_response_repeat_ProbeFeatureRTAccuModel_resRTWTT",
        "main_effect_task_repeat_ProbeFeatureRTAccuModel_resRTWTT",
        "main_effect_probe_repeat_ProbeFeatureRTAccuModel_resRTWTT",
        "perceptual_change_for_EDS_ProbeFeatureRTAccuModel_resRTWTT",
        "perceptual_change_for_IDS_ProbeFeatureRTAccuModel_resRTWTT",
        "perceptual_change_for_Stay_ProbeFeatureRTAccuModel_resRTWTT",
        "interaction_response_repeat_x_task_repeat_ProbeFeatureRTAccuModel_resRTWTT",
        "task_repeat_T_switch_C_prev_target_feature_switch_vs_repeat_ProbeFeatureRTAccuModel_resRTWTT",
        "response_repeat_T_switch_C_prev_target_feature_switch_vs_repeat_ProbeFeatureRTAccuModel_resRTWTT",
        "main_effect_Errors_ProbeFeatureRTAccuModel_resRTWTT",
        "Prev_Errors_vs_error_ProbeFeatureRTAccuModel_resRTWTT",
        "zRT_ProbeFeatureRTAccuModel_resRTWTT",
        "zRT_diff_ProbeFeatureRTAccuModel_resRTWTT",
        "probe_repeat_vs_switch_ProbeFeatureRTAccuModel_resRTWTT",
    ]

    # -------------------------------
    # paths
    # -------------------------------
    if os.path.exists("/Shared"):
        data_dir  = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightNeuralDistance_RTdiff"
        mask_path = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA/searchlight_mask.nii.gz"
        afni_cmd  = ["singularity", "run", "--cleanenv",
                    "/Shared/lss_kahwang_hpc/opt/afni/afni.sif", "3dMEMA"]
    else:
        data_dir  = "/data/backed_up/shared/HiRe/Neural_Distance" # "ThalHi_MRI_2020/GLMsingle/searchlightNeuralDistance_RTdiff"
        mask_path = "/data/backed_up/shared/ThalHi_MRI_2020/GLMsingle/searchlightRSA/searchlight_mask.nii.gz"
        afni_cmd  = ["3dMEMA"]

    group_dir = os.path.join(data_dir, "group_stats_mema")
    os.makedirs(group_dir, exist_ok=True)

    # Choose cluster option: "None" or "Clustsim"
    cluster_option = "None"

    # -------------------------------
    # main loop
    # -------------------------------
    for reg in regressors:

        beta_files = sorted(glob.glob(f"{data_dir}/*/{reg}_beta.nii.gz"))
        tval_files = sorted(glob.glob(f"{data_dir}/*/{reg}_tval.nii.gz"))

        if not beta_files or not tval_files:
            print(f" Missing beta or tvals for {reg}, skipping")
            continue

        # sanity check: subjects match
        bet_subs = [os.path.basename(os.path.dirname(b)) for b in beta_files]
        t_subs   = [os.path.basename(os.path.dirname(t)) for t in tval_files]

        if bet_subs != t_subs:
            print(f" Beta and tval subject ordering mismatch for {reg}")
            continue

        prefix = f"group_{reg}_MEMA"

        cmd = afni_cmd + [
            "-mask", mask_path,
            "-prefix", prefix,
            "-jobs", "24",
            "-model_outliers",
            "-residual_Z",
            "-set", reg
        ]
        
        unusable_subjects = [10012, 10028, 10039, 10054, 10072]
        # Add subjects:   subj  beta.nii.gz  tval.nii.gz
        for subj, b, t in zip(bet_subs, beta_files, tval_files):
            if int(subj) in unusable_subjects:
                print("not including ", subj, " due to poor behavioral acc in one or more condition")
            else:
                cmd += [subj, b, t]

        if cluster_option == "Clustsim":
            cmd.append("-Clustsim")

        print(f"======================================================")
        print(f"Running MEMA for: {reg}")
        print(" ".join(cmd))
        print(f"======================================================")

        subprocess.run(cmd, check=True, cwd=group_dir)

        print(f"Finished MEMA: {os.path.join(group_dir, prefix)}\n")