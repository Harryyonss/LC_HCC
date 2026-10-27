#!/bin/bash

BASE_DIR="/data/backed_up/shared/ME_7T_Pilot/searchlightRSA/vdt"
ROI_NAME="mask_roi"
OUT_DIR="${BASE_DIR}/group_roi_dist_3dttest_diffY"
mkdir -p "${OUT_DIR}"

SUBJECTS=(10341 11084 11100 11109 11111 11112 11113 11115 11116)
AFNI_CMD="3dttest++"

CONDITIONS=("Cue2Cue" "Probe2Probe" "Probe2Cue")
TARGETS=("CueBetas" "ProbeBetas" "CueBetas")

for idx in "${!CONDITIONS[@]}"; do
    COND="${CONDITIONS[$idx]}"
    TARGET="${TARGETS[$idx]}"

    echo "=========================================================="
    echo " Running Group 3dttest++ for Delta Y [${COND} -> ${TARGET}]"
    echo "=========================================================="

    BETA_FILES=()

    for SUB in "${SUBJECTS[@]}"; do
        FILE="${BASE_DIR}/sub-${SUB}/roi_dist_glm/${SUB}_${ROI_NAME}_${COND}_to_${TARGET}_diffY_beta.nii.gz"
        if [ -f "${FILE}" ]; then
            BETA_FILES+=("${FILE}")
            echo " [FOUND] sub-${SUB}"
        else
            echo " [MISSING] ${FILE}"
        fi
    done

    if [ ${#BETA_FILES[@]} -ge 2 ]; then
        OUT_PREFIX="${OUT_DIR}/group_${ROI_NAME}_${COND}_to_${TARGET}_diffY_N${#BETA_FILES[@]}.nii.gz"
        rm -f "${OUT_PREFIX}"

        $AFNI_CMD \
            -prefix "${OUT_PREFIX}" \
            -setA "${BETA_FILES[@]}"

        echo " -> Output saved: ${OUT_PREFIX}"
    else
        echo " [ERROR] Skipping ${COND} (fewer than 2 subject files)"
    fi
    echo ""
done

echo "All Delta Y group 3dttest++ jobs finished! Output directory: ${OUT_DIR}"