#!/bin/bash
# ==============================================================================
# Script: run_3dttest.sh
# Description: Run AFNI 3dttest++ on subject-level mean beta maps
#              using the 70% consensus group brain mask.
# ==============================================================================

RESULTS_DIR="/data/backed_up/shared/ME_7T_Pilot/GLMsingle/results"
cd "${RESULTS_DIR}"

# -----------------------------------------------------------------------------
# Mask and Subjects Definition
# -----------------------------------------------------------------------------
GROUP_MASK="$/data/backed_up/shared/ME_7T_Pilot/fmriprep/group_mask_consensus70.nii.gz"

if [ ! -f "${GROUP_MASK}" ]; then
    echo "[!] Warning: Group mask not found at ${GROUP_MASK}"
    echo "    Continuing without mask or ensure group_mask.sh has run."
fi

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

EVENTS=("Cue" "Probe")

for EVENT in "${EVENTS[@]}"; do
    echo "=========================================================="
    echo " Running 3dttest++ for ${EVENT} with Group Mask"
    echo "=========================================================="

    # Collect existing files only for valid subjects
    FILES=()
    for SUB in "${SUBJECTS[@]}"; do
        F="${SUB}_mean_${EVENT}.nii.gz"
        if [ -f "${F}" ]; then
            FILES+=( "${F}" )
        else
            echo "[!] Warning: File missing for ${SUB}: ${F}"
        fi
    done

    echo "[*] Found ${#FILES[@]} datasets for ${EVENT}:"
    printf '    %s\n' "${FILES[@]}"

    if [ ${#FILES[@]} -lt 2 ]; then
        echo "[!] Not enough subjects to run t-test for ${EVENT}. Skipping to next event."
        continue
    fi

    OUT_PREFIX="group_ttest_${EVENT}_masked"

    # Build 3dttest++ command based on mask availability
    if [ -f "${GROUP_MASK}" ]; then
        3dttest++ -overwrite \
            -mask "${GROUP_MASK}" \
            -prefix "${OUT_PREFIX}.nii.gz" \
            -setA "${FILES[@]}" || echo "[!] 3dttest++ encountered an issue on ${EVENT}, continuing..."
    else
        3dttest++ -overwrite \
            -prefix "${OUT_PREFIX}.nii.gz" \
            -setA "${FILES[@]}" || echo "[!] 3dttest++ encountered an issue on ${EVENT}, continuing..."
    fi

    echo "[+] Step complete for ${EVENT}."
done

echo ""
echo "[✓] Finished processing all events."