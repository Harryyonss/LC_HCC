#!/bin/bash
# ==============================================================================
# Script: group_mask.sh
# Description: Resample to a common master grid and generate group consensus mask
# ==============================================================================


BASE_DIR="/data/backed_up/shared/ME_7T_Pilot"
FMRIPREP_DIR="${BASE_DIR}/fmriprep"
OUT_DIR="${BASE_DIR}/fmriprep"
mkdir -p "${OUT_DIR}"

# Explicit list of valid pilot subjects
SUBJECTS=(
    "sub-10341"
    "sub-11084"
    "sub-11100"
    "sub-11109"
    "sub-11111"
    "sub-11112"
    "sub-11113"
)

MASK_FILES=()
for SUB in "${SUBJECTS[@]}"; do
    MASK="${FMRIPREP_DIR}/${SUB}/combined_mask.nii.gz"
    if [ ! -f "${MASK}" ]; then
        if [ -f "${FMRIPREP_DIR}/${SUB}/combined_mask+tlrc.HEAD" ]; then
            MASK="${FMRIPREP_DIR}/${SUB}/combined_mask+tlrc"
        fi
    fi

    if [ -f "${MASK}" ] || [ -f "${MASK}.HEAD" ]; then
        echo "    [+] Added: ${SUB}"
        MASK_FILES+=( "${MASK}" )
    else
        echo "[!] Warning: Mask not found for ${SUB}"
    fi
done

echo "[*] Total masks collected: ${#MASK_FILES[@]}"

# Set master reference to the first valid subject mask
MASTER_REF="${MASK_FILES[0]}"
TMP_DIR=$(mktemp -d)
MATCHED_MASKS=()

echo "[*] Resampling all masks to master grid: $(basename ${MASTER_REF})"
for M in "${MASK_FILES[@]}"; do
    BNAME=$(basename "${M}" .nii.gz)
    BNAME=$(echo "${BNAME}" | sed 's/+tlrc//')
    S_OUT="${TMP_DIR}/${BNAME}_matched.nii.gz"

    3dresample -overwrite \
        -master "${MASTER_REF}" \
        -rmode NN \
        -input "${M}" \
        -prefix "${S_OUT}"

    MATCHED_MASKS+=( "${S_OUT}" )
done

OUT_CONSENSUS="${OUT_DIR}/group_mask_consensus70.nii.gz"
OUT_INTERSECT="${OUT_DIR}/group_mask_intersection.nii.gz"

# Generate consensus masks
3dmask_tool -overwrite \
    -input "${MATCHED_MASKS[@]}" \
    -frac 0.7 \
    -prefix "${OUT_CONSENSUS}"

3dmask_tool -overwrite \
    -input "${MATCHED_MASKS[@]}" \
    -frac 1.0 \
    -prefix "${OUT_INTERSECT}"

# Clean up temporary resampled files
rm -rf "${TMP_DIR}"

echo ""
echo "[✓] Successfully created group masks:"
echo "    - 70% Consensus : ${OUT_CONSENSUS}"
echo "    - 100% Strict   : ${OUT_INTERSECT}"