dataset_dir=/data/backed_up/shared/ME_7T_Pilot/
sub=11116

# - - - - - - - - - - - - - - - - - - - - - - - - #
# - - -  run 3dDeconvolve prep script - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - #
python /data/backed_up/shared/ME_7T_Pilot/Harry_result/Script/GLM/3dDeconvolve/prep_3dDeconvolve.py ${sub}

# - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - Define Runs to Include  - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - #
case "${sub}" in
    "11113")
        runs=(3 4 5 6 7 8 9 10 11 12)       # Exclude runs 1 and 2
        ;;
    "11102")
        runs=(1 2 3 4 5 6 7 8 9 10 11)       # 11 runs total
        ;;
    *)
        runs=(1 2 3 4 5 6 7 8 9 10 11 12)    # Default: all 12 runs
        ;;
esac

func_dir="${dataset_dir}/fmriprep/sub-${sub}/func"
bold_files=""
mask_files=""

for r in "${runs[@]}"; do
    run_str=$(printf "%02d" $r)
    b_file="${func_dir}/sub-${sub}_task-HCC_run-${run_str}_part-mag_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    m_file="${func_dir}/sub-${sub}_task-HCC_run-${run_str}_part-mag_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    
    if [ -f "$b_file" ]; then
        bold_files="${bold_files} ${b_file}"
    else
        echo "WARNING: BOLD file missing -> ${b_file}"
    fi

    if [ -f "$m_file" ]; then
        mask_files="${mask_files} ${m_file}"
    fi
done


# - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - Create the combined mask  - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - #
mask_output="${dataset_dir}/fmriprep/sub-${sub}/combined_mask+tlrc"

rm -f ${mask_output}.BRIK* ${mask_output}.HEAD

3dmask_tool -input ${mask_files} -prefix ${mask_output}


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - Run 3dDeconvolve for EDS, IDS, and Stay - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

3dDeconvolve \
-input ${bold_files} \
-mask ${mask_output}.BRIK \
-polort A \
-num_stimts 6 \
-censor ${dataset_dir}3dDeconvolve/sub-${sub}/censor.1D \
-ortvec ${dataset_dir}3dDeconvolve/sub-${sub}/noise.1D noise \
-local_times \
-stim_times 1 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_Stay.1D 'SPMG' -stim_label 1 Cue_Stay \
-stim_times 2 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_EDS.1D 'SPMG' -stim_label 2 Cue_EDS \
-stim_times 3 ${dataset_dir}3dDeconvolve/sub-${sub}/cue_IDS.1D 'SPMG' -stim_label 3 Cue_IDS \
-stim_times 4 ${dataset_dir}3dDeconvolve/sub-${sub}/probe_Stay.1D 'SPMG' -stim_label 4 Probe_Stay \
-stim_times 5 ${dataset_dir}3dDeconvolve/sub-${sub}/probe_EDS.1D 'SPMG' -stim_label 5 Probe_EDS \
-stim_times 6 ${dataset_dir}3dDeconvolve/sub-${sub}/probe_IDS.1D 'SPMG' -stim_label 6 Probe_IDS \
-num_glt 16 \
-gltsym 'SYM: +1*Cue_EDS' -glt_label 1 Cue_EDS \
-gltsym 'SYM: +1*Cue_IDS' -glt_label 2 Cue_IDS \
-gltsym 'SYM: +1*Cue_Stay' -glt_label 3 Cue_Stay \
-gltsym 'SYM: +1*Cue_EDS[0] - 1*Cue_Stay[0]' -glt_label 4 Cue_EDS-Cue_Stay \
-gltsym 'SYM: +1*Cue_EDS[0] - 1*Cue_IDS[0]' -glt_label 5 Cue_EDS-Cue_IDS \
-gltsym 'SYM: +1*Cue_IDS[0] - 1*Cue_Stay[0]' -glt_label 6 Cue_IDS-Cue_Stay \
-gltsym 'SYM: +1*Cue_EDS[0] + 1*Cue_IDS[0] + 1*Cue_Stay[0]' -glt_label 7 Cue_All \
-gltsym 'SYM: +1*Cue_EDS[0] + 1*Cue_IDS[0] - 2*Cue_Stay[0]' -glt_label 8 Cue_Switch \
-gltsym 'SYM: +1*Probe_EDS' -glt_label 9 Probe_EDS \
-gltsym 'SYM: +1*Probe_IDS' -glt_label 10 Probe_IDS \
-gltsym 'SYM: +1*Probe_Stay' -glt_label 11 Probe_Stay \
-gltsym 'SYM: +1*Probe_EDS[0] - 1*Probe_Stay[0]' -glt_label 12 Probe_EDS-Probe_Stay \
-gltsym 'SYM: +1*Probe_EDS[0] - 1*Probe_IDS[0]' -glt_label 13 Probe_EDS-Probe_IDS \
-gltsym 'SYM: +1*Probe_IDS[0] - 1*Probe_Stay[0]' -glt_label 14 Probe_IDS-Probe_Stay \
-gltsym 'SYM: +1*Probe_EDS[0] + 1*Probe_IDS[0] + 1*Probe_Stay[0]' -glt_label 15 Probe_All \
-gltsym 'SYM: +1*Probe_EDS[0] + 1*Probe_IDS[0] - 2*Probe_Stay[0]' -glt_label 16 Probe_Switch \
-fout \
-rout \
-tout \
-bucket ${dataset_dir}3dDeconvolve/sub-${sub}/cue-probe_Stay-EDS-IDS_contrast.nii.gz \
-GOFORIT 100 \
-noFDR \
-jobs 15


#polort A: program will automaticlly degree of polynomial corresponding to the  null hypothesis 
#do i need to put the TRs in there??

