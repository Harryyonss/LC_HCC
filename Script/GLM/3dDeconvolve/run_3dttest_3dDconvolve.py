#!/usr/bin/env python3
"""
Run AFNI 3dttest++ on 3dDeconvolve bucket files by extracting
sub-brick indices directly from NIfTI labels.
"""

import os
import glob
import subprocess

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = "/data/backed_up/shared/ME_7T_Pilot"
DECONV_DIR = os.path.join(BASE_DIR, "3dDeconvolve")
OUT_DIR = os.path.join(DECONV_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

LABELS = [
    "Cue_Stay",
    "Cue_EDS",
    "Cue_IDS",
    "Cue_EDS-Cue_Stay",
    "Cue_EDS-Cue_IDS",
    "Cue_IDS-Cue_Stay",
    "Cue_All",
    "Cue_Switch",
    "Probe_Stay",
    "Probe_EDS",
    "Probe_IDS",
    "Probe_EDS-Probe_Stay",
    "Probe_EDS-Probe_IDS",
    "Probe_IDS-Probe_Stay",
    "Probe_All",
    "Probe_Switch",
]

# EXCLUDED_SUBS = ["sub-00000"]


def get_subbrick_labels(nii_path):
    """
    Query exact sub-brick label list using AFNI 3dinfo via subprocess,
    cleanly filtering out any version header strings.
    """
    cmd = ["3dinfo", "-label", nii_path]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    lines = [line.strip() for line in res.stdout.strip().split("\n") if line.strip()]
    if not lines:
        return []
    
    labels_line = lines[-1]
    return labels_line.split("|")


def main():
    sub_dirs = sorted(glob.glob(os.path.join(DECONV_DIR, "sub-*")))
    valid_sub_dirs = [
        d for d in sub_dirs 
        if os.path.isdir(d) and os.path.basename(d) not in EXCLUDED_SUBS
    ]

    print(f"[*] Found {len(valid_sub_dirs)} valid subjects (excluding {EXCLUDED_SUBS}).")

    for lbl in LABELS:
        print(f"\n==========================================================")
        print(f" Processing Group t-test for: {lbl}")
        print(f"==========================================================")

        seta_args = []
        target_glt = f"{lbl}_GLT#0_Coef"
        target_stim = f"{lbl}#0_Coef"

        for sdir in valid_sub_dirs:
            sub = os.path.basename(sdir)
            bucket_file = os.path.join(sdir, "cue-probe_Stay-EDS-IDS_contrast.nii.gz")

            if not os.path.isfile(bucket_file):
                print(f"[!] File missing: {bucket_file}")
                continue

            all_labels = get_subbrick_labels(bucket_file)

            matched_idx = None
            matched_name = None

            # Match GLT label first, then Stim label
            if target_glt in all_labels:
                matched_idx = all_labels.index(target_glt)
                matched_name = target_glt
            elif target_stim in all_labels:
                matched_idx = all_labels.index(target_stim)
                matched_name = target_stim

            if matched_idx is not None:
                print(f"    [+] {sub}: Index [{matched_idx}] ({matched_name})")
                seta_args.append(f"{bucket_file}[{matched_idx}]")
            else:
                print(f"[!] Error: Could not find '{target_glt}' or '{target_stim}' in {sub}")

        n_subs = len(seta_args)
        print(f"[*] Total valid datasets for {lbl}: {n_subs}")

        if n_subs < 2:
            print(f"[!] Insufficient subjects for {lbl}. Skipping.")
            continue

        out_prefix = os.path.join(OUT_DIR, f"group_ttest_{lbl}_N{n_subs}.nii.gz")

        cmd_ttest = [
            "3dttest++",
            "-overwrite",
            "-prefix", out_prefix,
            "-setA"
        ] + seta_args

        run_res = subprocess.run(cmd_ttest, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if run_res.returncode == 0:
            print(f"[+] Done: {out_prefix}")
        else:
            print(f"[!] 3dttest++ failed for {lbl}:\n{run_res.stderr}")

    print("\n[✓] All 3dDeconvolve group t-tests completed successfully.")


if __name__ == "__main__":
    main()