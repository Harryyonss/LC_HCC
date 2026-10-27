#!/bin/bash
### XNAT download script for both standard DICOM and ARCHIVE runs

###### 1. Config
EXP="RPACS_E115086"
BASE="https://rpacs.iibi.uiowa.edu/xnat"
SUB=11116
DEST="/data/backed_up/shared/ME_7T_Pilot/Raw/${SUB}"
TMPROOT="/data/backed_up/shared/ME_7T_Pilot/tmp"
mkdir -p "${TMPROOT}"

###### 2. Define scan series separately
# Standard DICOM scans (e.g., anatomical, EPI, EPI REVPE)
DICOM_SCANS=(2 4 5)

# Compressed ARCHIVE scans (e.g., reconstructed functional EPIs)
ARCHIVE_SCANS=(600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800)


###### 3. Download standard DICOM scans (Original method)
for SCAN in "${DICOM_SCANS[@]}"; do
    echo "Downloading DICOM scan ${SCAN}"

    RUNDIR="${DEST}/run-${SCAN}"
    rm -rf "${RUNDIR}"
    mkdir -p "${RUNDIR}"

    TMPDIR="$(mktemp -d "${TMPROOT}/xnat_${SUB}_${SCAN}_XXXXXX")"
    ZIPFILE="${TMPDIR}/DICOM.zip"

    wget \
        --netrc \
        -c \
        -O "${ZIPFILE}" \
        "${BASE}/data/experiments/${EXP}/scans/${SCAN}/resources/DICOM/files?format=zip"

    # Look at the ZIP contents and find one file under the DICOM files directory
    ENTRY="$(unzip -Z1 "${ZIPFILE}" | grep '/resources/DICOM/files/' | head -n 1 || true)"

    if [[ -z "${ENTRY}" ]]; then
        echo "ERROR: Could not find DICOM files path in scan ${SCAN}"
        rm -rf "${TMPDIR}"
        continue
    fi

    SERIESDIR="${ENTRY%/resources/DICOM/files/*}"
    SERIESNAME="${SERIESDIR##*/}"

    unzip -qo "${ZIPFILE}" -d "${TMPDIR}"

    if [[ ! -d "${TMPDIR}/${SERIESDIR}/resources/DICOM/files" ]]; then
        echo "ERROR: Missing DICOM files directory for scan ${SCAN}"
        echo "Expected: ${TMPDIR}/${SERIESDIR}/resources/DICOM/files"
        rm -rf "${TMPDIR}"
        continue
    fi

    mv "${TMPDIR}/${SERIESDIR}/resources/DICOM/files" "${RUNDIR}/${SERIESNAME}"

    rm -rf "${TMPDIR}"
done


###### 4. Download ARCHIVE scans (New method for reconstructed ZIPs)
for SCAN in "${ARCHIVE_SCANS[@]}"; do
    echo "Downloading ARCHIVE scan ${SCAN}"

    RUNDIR="${DEST}/run-${SCAN}"
    mkdir -p "${RUNDIR}"

    TMPDIR="$(mktemp -d "${TMPROOT}/xnat_${SUB}_${SCAN}_XXXXXX")"
    ZIPFILE="${TMPDIR}/ARCHIVE.zip"

    wget \
        --netrc \
        -c \
        -O "${ZIPFILE}" \
        "${BASE}/data/experiments/${EXP}/scans/${SCAN}/resources/ARCHIVE/files?format=zip"

    # Find the inner series ZIP inside the downloaded wrapper ZIP
    INNER_ZIP_ENTRY="$(unzip -Z1 "${ZIPFILE}" | grep '\.zip$' | head -n 1 || true)"

    if [[ -z "${INNER_ZIP_ENTRY}" ]]; then
        echo "ERROR: Could not find inner ZIP file in ARCHIVE for scan ${SCAN}"
        rm -rf "${TMPDIR}"
        continue
    fi

    unzip -q "${ZIPFILE}" -d "${TMPDIR}"

    # Extract the inner series ZIP directly into destination run directory
    unzip -qo "${TMPDIR}/${INNER_ZIP_ENTRY}" -d "${RUNDIR}"

    rm -rf "${TMPDIR}"
done