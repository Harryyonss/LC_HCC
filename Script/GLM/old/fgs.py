#things that might help you get info
from operator import sub
import os

#RDSS Quantum_Task/Quantum_Task_Script.py
    #Quantum_Trl_Durs = {'cue': 1.0, 'delay': 0.5, 'stim': 1.5, 'feedback': 0.5, 'ITI': [2.0, 2.1, 2.15, 2.2, 2.3, 2.35, 2.4, 2.5, 2.6, 2.65, 2.7, 2.8, 2.85, 2.9, 3.0]}
#3dinfo dataset_dir/fmriprep/sub-sub/func/sub-sub_task-Quantum_run-004_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
    #240 time points
#cat dataset_dir/fmriprep/sub-sub/func/sub-sub_task-Quantum_run-004_space-MNI152NLin2009cAsym_desc-preproc_bold.json
    #TR = 2.039



# get path to the directory to which GLMsingle was installed
homedir = 'home/sestokes'



#time series, need to figure out how to concatenate runs
#time_series = join(dataset_dir, 'fmriprep/sub-'+sub+'/func/sub-'+sub+'_task-Quantum_run-00[1-5]space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')

#stimtime df
#design_df = pd.read_csv(os.path.join(dataset_dir,"model_data",('sub-'+sub+'_dataframe.csv')))







################################################################################################################################################
# sub_list=["10008", "10024", "10263", "10280", "10281", "10283", "10303", "10305", "10306", "10309", "10310", "10311", "10313", "10314", "10315", "10316", "10320", "10323", "10324", "10326", "10327", "10332", "10334", "10336", "10340", "10341", "10344", "10346", "10348", "10349", "10350", "10353", "10355", "10356", "10357", "10359", "10360", "10374"]

#load variables
n_runs = 5
TR = 2.039
stim_shift = TR * 3
#sub = input()
#sub = str(sub)

# for sub in sub_list:
#define paths
#sub='10008'
#dataset_dir = '/mnt/nfs/lss/lss_kahwang_hpc/data/TRIIMS/'
#dataset_dir = "/Shared/lss_kahwang_hpc/data/TRIIMS/"
dataset_dir = "/data/backed_up/shared/FPNHIU/"
#mask_afni = os.path.join(dataset_dir,f"fmriprep/sub-{sub}/combined_mask+tlrc") #should I do BRIK or HEAD?????
mask_nii =  os.path.join(dataset_dir,f"fmriprep/sub-{sub}/combined_mask.nii.gz") # need to creare
output_dir = join(dataset_dir,f'GLMsingle/sub-{sub}/')
cue_stimtime = os.path.join(dataset_dir,f'3dDeconvolve/sub-{sub}/cue_stimtime.1D')
cue_onsets_mat = np.loadtxt(cue_stimtime)
probe_stimtime = os.path.join(dataset_dir,f'3dDeconvolve/sub-{sub}/probe_stimtime.1D')
probe_onsets_mat = np.loadtxt(probe_stimtime)
fb_stimtime = os.path.join(dataset_dir,f'3dDeconvolve/sub-{sub}/fb_stimtime.1D')
fb_onsets_mat = np.loadtxt(fb_stimtime)

# convert mask to NIfTI and fit masker once. This is because oroginal overlap mask is in AFNI format, and GLMsingle only accepts NIfTI.
#subprocess.run( ["3dAFNItoNIFTI", "-prefix", mask_nii, mask_afni], check=True )
masker = NiftiMasker(mask_img=mask_nii, standardize=False, detrend=False)
masker.fit()

epoch_dict  = {"cue":cue_onsets_mat, "probe":probe_onsets_mat, "fb":fb_onsets_mat}

data1 = []
design1 = []

#idk how I should organize. should epochs go first? runs go first? should data and deisgn [] lists be in the loop or out of the loop? ??
for run in range(n_runs):
    print(run)
    #for epoch in epochs:
    time_series = f"fmriprep/sub-{sub}/func/sub-{sub}_task-Quantum_run-00{run+1}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    bold = os.path.join(dataset_dir, time_series)
    ts = masker.transform(bold).T
    ts = ts[:,3:] #should be 237 time points
    data1.append(ts)
        #buld onsets vector for that run
    n_tp = ts.shape[1]
    on = np.column_stack((np.zeros(n_tp),np.zeros(n_tp),np.zeros(n_tp)))
    tmp_mat = np.column_stack((epoch_dict["cue"][run][~np.isnan(epoch_dict["cue"][run])], epoch_dict["probe"][run][~np.isnan(epoch_dict["probe"][run])], epoch_dict["fb"][run][~np.isnan(epoch_dict["fb"][run])]))                #valid=~np.isnan(tmp_mat[:,col])
    tmp_mat = tmp_mat - stim_shift
    idx = np.round(tmp_mat/TR).astype(int)  # #round to nearest TR.
    for i in range(len(idx)-1):
        if idx[i][2] == idx[i+1][0]:
            print("\n"i, run)
            idx[i][2] = idx[i][2]-1
    for col in range(3):
        on[idx[:,col], col] =1
    design1.append(on)





#run GLMsingle
opt = {
    'wantlibrary': 1,
    'wantglmdenoise':1,
    'wantfracridge': 1,
    'wantfileoutputs': [0,0,0,0],
    'wantmemoryoutputs':[1,1,1,1]
}

glms = GLM_single(opt)
start = time.time()
results_glmsingle = glms.fit(design1, data1, stimdur= 0.5, tr=TR, outputdir=output_dir)
print(f"sub-{sub} GLMsingle done in {time.strftime('%H:%M:%S', time.gmtime(time.time()-start))}")
typed = results_glmsingle['typed']



#write out GLM single maps
# try:
#     typed  =



for key in ['R2', 'noisepool', 'HRFindex']:
    vol = masker.inverse_transform(np.squeeze(typed[key]))
    vol.to_filename(os.path.join(output_dir, f"{sub}_{key}.nii.gz"))
#mean beta
#cue
meanb = np.squeeze(typed["betasmd"])[:, np.arange(0,600,3)].mean(axis =1)
masker.inverse_transform(meanb).to_filename(os.path.join(output_dir, f"{sub}_meanCueBeta.nii.gz"))
#probe
meanb = np.squeeze(typed["betasmd"])[:, np.arange(1,600,3)].mean(axis =1)
masker.inverse_transform(meanb).to_filename(os.path.join(output_dir, f"{sub}_meanProbeBeta.nii.gz"))
#fb
meanb = np.squeeze(typed["betasmd"])[:, np.arange(2,600,3)].mean(axis =1)
masker.inverse_transform(meanb).to_filename(os.path.join(output_dir, f"{sub}_meanFBBeta.nii.gz"))

#masker.inverse_transform(meanb).to_filename(os.path.join(output_dir, f"{sub}_meanBeta.nii.gz"))
#trial bets
tb = np.squeeze(typed['betasmd']).T
masker.inverse_transform(tb).to_filename(os.path.join(output_dir, f"{sub}_TrialBetas.nii.gz"))

# except Exception as e:
#     # write out a warning file if anything fails
#     warn_file = os.path.join(output_dir, f"{sub}_warning.txt")
#     with open(warn_file, "w") as f:
#     f.write(f"Error processing sub-{sub}:\n{str(e)}\n")
#     print(f"[!] sub-{sub} failed; see {warn_file}")
#     # then continue to next subject
#     continue



##################################################################################################################



# plot_fields = ['betasmd','R2','HRFindex','FRACvalue']
# colormaps = ['RdBu_r','hot','jet','copper']
# clims = [[-5,5],[0,85],[0,20],[0,1]]

# plt.figure(figsize=(12,4))



# for i in range(len(plot_fields)):

#     plt.subplot(1,4,i+1)

#     if i == 0:
#         # when plotting betas, for simplicity just average across all image presentations
#         # this will yield a summary of whether voxels tend to increase or decrease their
#         # activity in response to the experimental stimuli (similar to outputs from
#         # an ONOFF GLM)
#         plot_data = np.nanmean(np.squeeze(typed[plot_fields[i]]),axis =1)

#     titlestr = 'average GLM betas (750 stimuli)'

#     else:
#         # plot all other voxel-wise metrics as outputted from GLMsingle
#         plot_data = np.squeeze(typed[plot_fields[i]].reshape(xyz))
#         titlestr = plot_fields[i]

#     plt.imshow(plot_data,cmap=colormaps[i],clim=clims[i])
#     plt.colorbar()
#     plt.title(titlestr)
#     plt.axis(False)