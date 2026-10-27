#!/bin/bash
# ==============================================================================
# Script: run_subject_3dttest.sh
# Description: Dynamically detect volume count and run within-subject 3dttest++
# ==============================================================================

BASE_DIR="/data/backed_up/shared/ME_7T_Pilot"
GLM_DIR="${BASE_DIR}/GLMsingle"
OUT_DIR="${GLM_DIR}/results"
mkdir -p "${OUT_DIR}"

EVENTS=("Cue" "Probe")
TRIALS_PER_RUN=38
TOSS_PER_RUN=2

SUBJECTS=(
    # "sub-10341"
    # "sub-11084"
    # "sub-11100"
    # "sub-11109"
    # "sub-11111"
    # "sub-11112"
    "sub-11115"
)

for SUB in "${SUBJECTS[@]}"; do
    echo "=========================================================="
    echo " Running within-subject 3dttest++ for ${SUB}"
    echo "=========================================================="

    RAW_ID=$(echo "${SUB}" | sed 's/sub-//')
    SUB_DIR="${GLM_DIR}/${SUB}"

    MASK="${BASE_DIR}/fmriprep/${SUB}/combined_mask.nii.gz"
    MASK_OPT=""
    if [ -f "${MASK}" ]; then
        MASK_OPT="-mask ${MASK}"
    fi

    for EVENT in "${EVENTS[@]}"; do
        IN_FILE=$(find "${SUB_DIR}" -name "*${RAW_ID}*_${EVENT}_TrialBetas.nii.gz" | head -n 1)

        if [ -z "${IN_FILE}" ] || [ ! -f "${IN_FILE}" ]; then
            echo "[!] File missing for ${SUB} (${EVENT}). Skipping."
            continue
        fi

        # 1. Dynamically retrieve the total number of volumes (trials)
        TOTAL_NV=$(3dinfo -nv "${IN_FILE}")
        ACTUAL_RUNS=$(( TOTAL_NV / TRIALS_PER_RUN ))

        # 2. Build the sub-brick selector matching the subject's actual run count
        SELECTOR=""
        for (( r=0; r<ACTUAL_RUNS; r++ )); do
            start_idx=$(( r * TRIALS_PER_RUN + TOSS_PER_RUN ))
            end_idx=$(( (r + 1) * TRIALS_PER_RUN - 1 ))
            if [ -z "${SELECTOR}" ]; then
                SELECTOR="${start_idx}..${end_idx}"
            else
                SELECTOR="${SELECTOR},${start_idx}..${end_idx}"
            fi
        done

        VALID_TRIALS=$(( ACTUAL_RUNS * (TRIALS_PER_RUN - TOSS_PER_RUN) ))
        echo "[-] ${SUB} (${EVENT}): Total ${TOTAL_NV} trials (${ACTUAL_RUNS} runs) -> ${VALID_TRIALS} valid trials"

        OUT_FILE="${OUT_DIR}/${SUB}_${EVENT}_3dttest.nii.gz"

        # 3. Execute 3dttest++ (print success message only on zero exit code)
        if 3dttest++ -overwrite \
            ${MASK_OPT} \
            -prefix "${OUT_FILE}" \
            -setA "${IN_FILE}[${SELECTOR}]"; then
            echo "[+] Success: ${OUT_FILE}"
        else
            echo "[!] Failed for ${SUB} (${EVENT})"
        fi
    done
done

echo ""
echo "[✓] Processing finished."