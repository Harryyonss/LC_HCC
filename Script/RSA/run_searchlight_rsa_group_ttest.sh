#!/bin/bash

# 1. Define base and output directories
BASE_DIR="/data/backed_up/shared/ME_7T_Pilot/searchlightRSA/vdt"
OUT_DIR="${BASE_DIR}/group_3dttest"
mkdir -p "${OUT_DIR}"

# 2. List of subject IDs to include in the group analysis
SUBJECTS=(11100 11112 11084 11111 10341 11113 11109 11115 11116)

# Set AFNI command (uncomment Singularity line if running on HPC with container)
AFNI_CMD="3dttest++"
# AFNI_CMD="singularity run --cleanenv /Shared/lss_kahwang_hpc/opt/afni/afni.sif 3dttest++"

# 3. Loop over transition conditions
for COND in Cue2Cue Probe2Probe Probe2Cue; do
    echo "=========================================="
    echo " Running Group 3dttest++ for [${COND}]"
    echo "=========================================="

    # -----------------------------------------------------------------
    # (A) Contrast difference maps (*_diff.nii.gz) -> 1-sample t-test against 0
    # -----------------------------------------------------------------
    for CONTRAST in EDS_vs_IDS EDS_vs_Stay IDS_vs_Stay; do
        DIFF_FILES=()
        for SUB in "${SUBJECTS[@]}"; do
            FILE="${BASE_DIR}/sub-${SUB}/${SUB}_${COND}_${CONTRAST}_diff.nii.gz"
            if [ -f "${FILE}" ]; then
                DIFF_FILES+=("${FILE}")
            fi
        done

        # Run 3dttest++ if at least 2 subject files exist
        if [ ${#DIFF_FILES[@]} -ge 2 ]; then
            OUT_PREFIX="${OUT_DIR}/group_${COND}_${CONTRAST}.nii.gz"
            rm -f "${OUT_PREFIX}"
            
            $AFNI_CMD \
                -prefix "${OUT_PREFIX}" \
                -labelA "${CONTRAST}" \
                -setA "${DIFF_FILES[@]}"
        else
            echo "Skipping ${COND} - ${CONTRAST} (found fewer than 2 files)"
        fi
    done

done

echo "All group 3dttest++ jobs finished! Output directory: ${OUT_DIR}"
