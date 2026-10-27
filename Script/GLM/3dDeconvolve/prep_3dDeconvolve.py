import sys
import os
import argparse
import glob
import pandas as pd
import pickle
import logging
import numpy as np
from natsort import natsorted
import numpy.matlib as mb
import matplotlib.pyplot as plt
from varname import nameof #not usually installed, found it on stackoverflow, make sure you install it.
#plt.ioff()


# ------------ Set Up for Parsing Arguments --------------
def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="prep 3dDeconvolve",
        usage="[subject]  ",
    )
    parser.add_argument("subject", help="single subject id... e.g., 10001")
    return parser

# -------------------  Set Options  ---------------------
parser = init_argparse()
args = parser.parse_args(sys.argv[1:])
sub = args.subject
dataset_dir = "/data/backed_up/shared/ME_7T_Pilot/"
csv_dir=f"/data/backed_up/shared/ME_7T_Pilot/behav_data/{sub}"
output_dir = os.path.join(dataset_dir,"3dDeconvolve/")
#csv_dir = os.path.join(dataset_dir, "csvs")

# ===================================== exception for 11113 and 11102, since they have different number of runs
if int(sub) == 11102:
    run_list = list(range(1, 12))   # 1 ~ 11
elif int(sub) == 11113:
    run_list = list(range(3, 13))   # 3 ~ 12 
else:
    run_list = list(range(1, 13))   # standard: 1 ~ 12
# =====================================

n_runs = len(run_list)
threshold = 0.5

# ---------- Define other useful functions --------------

# function to write AFNI style stim timing file
def write_stimtime(filepath, inputvec):
	''' short hand function to write AFNI style stimtime'''
	
	f = open(filepath, 'w')
	for val in inputvec[0]:
		if val =='*':
			f.write(val + '\n')
		else:
			# this is to dealt with some weird formating issue
			f.write(np.array2string(np.around(val,6), suppress_small=True).replace('\n','').replace(']','').replace('[','') + '\n')
	f.close()

def load_regressors(regressor_file, cols=None, verbose=False):
    """Loads regressor tsv into df with selected columns and fills in NaN values.
        Args:
        regressor_file (str): filepath to regressor tsv
        cols ([str], optional): List of column names. Defaults to None.
        verbose (bool, optional): If true, will log more info. Defaults to False.

    Returns:
        regressor_df (Dataframe): Dataframe containing regressors from selected columns.
        regressor_names ([str]): List of regressor names.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    df_orig = pd.read_csv(regressor_file, sep="\t")
    regressor_df = pd.DataFrame()
    regressor_names = []
    ##this part below i might to adjust 
    #regressor_df = regressor_df.append(df_orig[cols])
    regressor_df = df_orig[cols]

    regressor_names.append(cols)

    for col in regressor_df.columns:
        sum_nan = sum(regressor_df[col].isnull())
        if sum_nan > 0:
            logging.info("Filling in " + str(sum_nan) +
                         " NaN value for " + col)
            regressor_df.loc[np.isnan(regressor_df[col]), col] = np.mean(
                regressor_df[col]
            )
    logging.info("# of Confound Regressors: " + str(len(regressor_df.columns)))

    return regressor_df, regressor_names

def censor(df, threshold=0.2, verbose=False):
    if isinstance(df, str):
        df = pd.read_csv(df, sep="\t")
    censor_vector = np.empty((len(df.index)))
    prev_motion = 0
  
    for index, row in enumerate(zip(df["framewise_displacement"])):
        # censor first three points
        if index < 3:
            censor_vector[index] = 0
            continue
        if row[0] > threshold:
            censor_vector[index] = 0
            prev_motion = index
        elif prev_motion + 1 == index or prev_motion + 2 == index:
            censor_vector[index] = 0
        else:
            censor_vector[index] = 1
    if verbose:
        percent_censored = round(
        np.count_nonzero(censor_vector == 0) / len(censor_vector) * 100
        )
        print(f"\tCensored {percent_censored}% of points")
    return censor_vector


def load_regressors_and_censor(files, cols=None, threshold=0.2):
    output_list = []
    censor_list = []
    for file in files:
        print(f'Parsing: {file.split("/")[-1]}')
        file_df, _ = load_regressors(file, cols=cols, verbose=False)
        output_list.append(file_df)
        censor_list.extend(censor(file, threshold=threshold))
    output_df = pd.concat(output_list)
    return output_df, censor_list

# ----------------------------------------------------------
# ------------           Run Code           ----------------
# ----------------------------------------------------------
# --- create subject folder ... if it does not exist already
if not os.path.isdir(os.path.join(output_dir,(f"sub-{sub}"))):
    print("creating subj directory now")
    os.mkdir(os.path.join(output_dir,(f"sub-{sub}")))
    
# --- generate nuisance file
confounds = natsorted(glob.glob(os.path.join(dataset_dir,"fmriprep",("sub-"+sub),"func","*confounds_*.tsv")))
print(confounds)

# ======================================= exception for 11113 and 11102, since they have different number of runs
if int(sub) == 11113:
    confounds = confounds[2:]  
elif int(sub) == 11102:
    confounds = confounds[:11]
# =======================================

c_list = []
for c in confounds:
    c_list.append(pd.read_csv(c, sep='\t')) # append runs
cdf = pd.concat(c_list)


default_columns = ['a_comp_cor_00', 'w_comp_cor_00', 'c_comp_cor_00',
                        'trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2', 
                        'trans_y', 'trans_y_derivative1', 'trans_y_derivative1_power2', 'trans_y_power2',
                        'trans_z', 'trans_z_derivative1', 'trans_z_power2', 'trans_z_derivative1_power2', 
                        'rot_x', 'rot_x_derivative1', 'rot_x_derivative1_power2', 'rot_x_power2', 
                        'rot_y', 'rot_y_derivative1', 'rot_y_power2', 'rot_y_derivative1_power2',
                        'rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2']

#                    'edge_comp_00','edge_comp_01','edge_comp_02','edge_comp_03','edge_comp_04','edge_comp_05',
#                    'edge_comp_06','edge_comp_07','edge_comp_08','edge_comp_09','edge_comp_10','edge_comp_11',
#                    'edge_comp_12','edge_comp_13','edge_comp_14','edge_comp_15','edge_comp_16','edge_comp_17',
#                    'edge_comp_18','edge_comp_19','edge_comp_20','edge_comp_21','edge_comp_22','edge_comp_23',

# default_columns = ['a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04', 
#                         'trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2', 
#                         'trans_y', 'trans_y_derivative1', 'trans_y_derivative1_power2', 'trans_y_power2',
#                         'trans_z', 'trans_z_derivative1', 'trans_z_power2', 'trans_z_derivative1_power2', 
#                         'rot_x', 'rot_x_derivative1', 'rot_x_derivative1_power2', 'rot_x_power2', 
#                         'rot_y', 'rot_y_derivative1', 'rot_y_power2', 'rot_y_derivative1_power2',
#                         'rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2']

noise_mat = cdf.loc[:,default_columns].values
noise_mat[np.isnan(noise_mat)]=0
np.savetxt(os.path.join(output_dir,f"sub-{sub}/noise.1D"), noise_mat)

# --- now generate censor file
regressor_filepath = os.path.join(output_dir, f"sub-{sub}/nuisance.1D")
censor_filepath = os.path.join(output_dir, f"sub-{sub}/censor.1D")
print(f"Prepping 3dDeconvolve on subject {sub}")
print(f"Extracting columns {default_columns} from regressor files")
regressor_files = natsorted(glob.glob(os.path.join(dataset_dir,f"fmriprep/sub-{sub}/func/*confounds_timeseries.tsv")))
# ================================================================= exception for 11113 and 11102, since they have different number of runs
if int(sub) == 11113:
    regressor_files = regressor_files[2:]
elif int(sub) == 11102:
    regressor_files = regressor_files[:11]
# =================================================================
regressor_df, output_censor = load_regressors_and_censor(regressor_files, cols=default_columns, threshold=threshold)
print(f"Writing regressor file to {regressor_filepath}")
regressor_df.to_csv(regressor_filepath, header=False, index=False, sep="\t")
print(f"Writing censor file to {censor_filepath}")
with open(censor_filepath, "w") as file:
    for num in output_censor:
        file.writelines(f"{num}\n")
print(f"\n\nSuccessfully extracted columns {default_columns} from regressor files and censored motion")

##### load beh dataframe #####
#files = glob.glob(os.path.join(dataset_dir, f"csvs/sub-{sub}_task-HCC_block*"))
print(os.path.join(csv_dir, f"sub-{sub}_task-HCC_block*.csv"))
files = glob.glob(os.path.join(csv_dir, f"sub-{sub}_task-HCC_block*.csv"))
files.sort()
print(files)

dfs = []
for file in files:
    tmp_df = pd.read_csv(file)
    dfs.append(tmp_df)
df = pd.concat(dfs)

#create 3 arrays of 1s and 0s for EDS, IDS, and Stay Trials 
switch_df = df['hierarhical_switch_type']
stay_arr = np.zeros(len(switch_df))
EDS_arr = np.zeros(len(switch_df))
IDS_arr = np.zeros(len(switch_df))

arr_dict = {"Stay": stay_arr, "EDS": EDS_arr, "IDS": IDS_arr}
switch_types = ["Stay", "EDS", "IDS"]

for switch in switch_types: 
    for i in range(len(switch_df)):
        if switch_df.iloc[i] == switch:
            arr_dict[switch][i] = 1

# --- create am for cue EDS, IDS, and Stay trials  ----------
#output_dict = {}
for switch in switch_types:
    #output_dict[switch] = {}
    tmp_stimtime = [['*']*n_runs]
    for i, run in enumerate(run_list): 
        run_df = df[df['block']==run].reset_index(drop=True) #"view" of block we are curreint sorting through
        
        tmp_arr = arr_dict[switch][df['block']==run]
        tmp = []
        for tr in range(len(run_df)):
            # exclude the first 2 stay trials since they happen within initial stabilization / dummy TRs
            if tr < 2:
                continue 
            if tmp_arr[tr]==1:
                tmp.append(run_df['Time_Since_Run_Cue_Prez'][tr]) # tmp.append(run_df['Time_Since_Run_Photo_Prez'][tr]) 
        tmp_stimtime[0][i] = tmp
    #output_dict[switch] = tmp_stimtime
    write_stimtime(os.path.join(output_dir, f"sub-{sub}/cue_{switch}.1D"), tmp_stimtime)


# --- create am for probe EDS, IDS,Stay trials----------
for switch in switch_types:
    #output_dict[switch] = {}
    tmp_stimtime = [['*']*n_runs]
    for i, run in enumerate(run_list): 
        run_df = df[df['block']==run].reset_index(drop=True) #"view" of block we are curreint sorting through
        
        # fix: slice run-specific condition vector to prevent indexing into run 1 across all blocks
        tmp_arr = arr_dict[switch][df['block']==run]
        tmp = []
        for tr in range(len(run_df)):
            # exclude the first 2 stay trials since they happen within initial stabilization / dummy TRs
            if tr < 2:
                continue 
            # fix: replaced buggy arr_dict[switch][tr]==1 with sliced tmp_arr[tr]==1
            if tmp_arr[tr]==1:
                probe_time = run_df['Time_Since_Run_Probe_Prez'][tr] #+ run_df["delay"][tr] # run_df['Time_Since_Run_Photo_Prez'][tr] + run_df["delay"][tr]
                tmp.append(probe_time) 
        tmp_stimtime[0][i] = tmp
    write_stimtime(os.path.join(output_dir, f"sub-{sub}/probe_{switch}.1D"), tmp_stimtime)




# --- create cue probe and resp stimtime ---------- (not needed for eds, ids, stay, but will be needed if we do a parametric modulation of some sort)
Cue_stimtime = [['*']*n_runs] #stimtime format, * for runs with no events of this stimulus class
Probe_stimtime = [['*']*n_runs] #stimtime format, * for runs with no events of this stimulus class
Resp_stimtime = [['*']*n_runs] #stimtime format, * for runs with no events of this stimulus class

for i, run in enumerate(run_list):  # loop through all runs
    run_df = df[df['block']==run].reset_index(drop=True) #"view" of block we are curreint sorting through
    Cue_times = [] #empty vector to store trial time info for the current block
    Probe_times = []
    Resp_times = []
    switch_type = []
    for tr in range(len(run_df)):
        # exclude the first 2 stay trials since they happen within initial stabilization / dummy TRs
        if tr < 2:
            continue 
        Cue_times.append(run_df['Time_Since_Run_Cue_Prez'][tr])
        Probe_times.append(run_df['Time_Since_Run_Probe_Prez'][tr]) #+run_df["delay"][tr])
        Resp_times.append(run_df['Time_Since_Run_subRespo'][tr])
        switch_type.append(run_df['hierarhical_switch_type'][tr])
    Cue_stimtime[0][i] = Cue_times # put trial timing of each block into the stimtime array
    Probe_stimtime[0][i] = Probe_times
    Resp_stimtime[0][i] = Resp_times # put trial timing of each block into the stimtime array
 

#write out stimtime array to text file  
write_stimtime(os.path.join(output_dir,("sub-"+sub),"cue_stimtime.1D"), Cue_stimtime)
write_stimtime(os.path.join(output_dir,("sub-"+sub),"probe_stimtime.1D"), Probe_stimtime)
write_stimtime(os.path.join(output_dir,("sub-"+sub),"resp_stimtime.1D"), Resp_stimtime)


