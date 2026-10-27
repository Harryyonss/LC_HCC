#!/bin/bash

# ==============================================================================
# 1. Define base and output directories
# ==============================================================================
BASE_DIR="/data/backed_up/shared/ME_7T_Pilot/searchlightRSA/vdt"
ROI_NAME="mask_roi"
OUT_DIR="${BASE_DIR}/group_roi_dist_3dttest"
mkdir -p "${OUT_DIR}"

# ==============================================================================
# 2. List of subject IDs to include in the group analysis
# ==============================================================================
SUBJECTS=(10341 11084 11100 11109 11111 11112 11113 11115 11116)

AFNI_CMD="3dttest++"

CONDITIONS=("Cue2Cue" "Probe2Probe" "Probe2Cue")
TARGETS=("CueBetas" "ProbeBetas" "CueBetas")

# ==============================================================================
# 3. Loop over transition conditions and run 3dttest++ on Beta maps
# ==============================================================================
for idx in "${!CONDITIONS[@]}"; do
    COND="${CONDITIONS[$idx]}"
    TARGET="${TARGETS[$idx]}"

    echo "=========================================================="
    echo " Running Group 3dttest++ for [${COND} -> ${TARGET}]"
    echo "=========================================================="

    BETA_FILES=()

    # Collect individual subject beta maps
    for SUB in "${SUBJECTS[@]}"; do
        FILE="${BASE_DIR}/sub-${SUB}/roi_dist_glm/${SUB}_${ROI_NAME}_${COND}_to_${TARGET}_beta.nii.gz"
        if [ -f "${FILE}" ]; then
            BETA_FILES+=("${FILE}")
        else
            echo " [WARNING] Missing file for sub-${SUB}: ${FILE}"
        fi
    done

    # Run 3dttest++ if at least 2 subject files exist
    if [ ${#BETA_FILES[@]} -ge 2 ]; then
        OUT_PREFIX="${OUT_DIR}/group_${ROI_NAME}_${COND}_to_${TARGET}_N${#BETA_FILES[@]}.nii.gz"
        rm -f "${OUT_PREFIX}"

        echo " -> Found ${#BETA_FILES[@]} subject maps. Executing 3dttest++ ..."

        # SHORT form: -setA file1 file2 file3 ...
        $AFNI_CMD \
            -prefix "${OUT_PREFIX}" \
            -setA "${BETA_FILES[@]}"

        echo " -> Output successfully generated: ${OUT_PREFIX}"
    else
        echo " [ERROR] Skipping ${COND} (found fewer than 2 subject files)"
    fi
    echo ""
done

echo "=========================================================="
echo "All group 3dttest++ jobs finished!"
echo "Output directory: ${OUT_DIR}"
echo "=========================================================="