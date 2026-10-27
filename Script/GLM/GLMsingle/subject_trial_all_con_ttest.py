#!/usr/bin/env python3
"""
Extract trial indices for EDS, IDS, and Stay from run-wise behavioral CSV files.
Each run contains exactly 38 trials: initial 2 Stay trials are skipped (df.iloc[2:]),
yielding exactly 12 trials per condition (Stay, EDS, IDS) per run.
Runs condition-specific 1-sample and pairwise contrast t-tests using AFNI 3dttest++.
"""

import os
import glob
import re
import subprocess
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = "/data/backed_up/shared/ME_7T_Pilot"
BEHAV_DIR = os.path.join(BASE_DIR, "behav_data")
GLM_DIR = os.path.join(BASE_DIR, "GLMsingle")
OUT_DIR = os.path.join(GLM_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

EVENTS = ["Cue", "Probe"]
CONDITIONS = ["Stay", "EDS", "IDS"]
CONTRASTS = [
    ("EDS", "Stay"),
    ("IDS", "Stay"),
    ("EDS", "IDS")
]

SUBJECTS = [
    # "sub-10341",
    # "sub-11084",
    # "sub-11100",
    # "sub-11109",
    # "sub-11111",
    # "sub-11112",
    "sub-11115"
]

# 1-based run indexing:
# sub-11113 has 10 runs (runs 3 to 12)
SUBJECT_RUNS = {
    "sub-11113": list(range(3, 13)),  # [3..12] -> corresponds to block-002..block-011
}

COND_COL = "hierarhical_switch_type"
TOSS_PER_RUN = 2  # Fixed 2 trials to toss at start of each run


def format_selector(indices):
    """Convert an index list to an AFNI sub-brick selector string."""
    return ",".join(map(str, indices))


def extract_block_num(filepath):
    """Extract integer block number from filename (e.g., block-002 -> 2)."""
    match = re.search(r"block-(\d+)", os.path.basename(filepath))
    return int(match.group(1)) if match else -1


def get_subject_condition_indices(sub_id, raw_id):
    """
    Sort run CSV files by block number, select valid runs for this subject,
    slice off initial 2 rows (df.iloc[2:]), and extract 0-based trial indices.
    """
    sub_behav_dir = os.path.join(BEHAV_DIR, raw_id)
    if not os.path.isdir(sub_behav_dir):
        sub_behav_dir = os.path.join(BEHAV_DIR, sub_id)

    pattern = os.path.join(sub_behav_dir, f"sub-{raw_id}_task-HCC_block-*.csv")
    all_run_files = glob.glob(pattern)

    if not all_run_files:
        print(f"[!] Warning: No behavioral CSV files found for {sub_id} in {pattern}")
        return None

    # Strict numerical sort on block number (block-000, block-001, ...)
    all_run_files = sorted(all_run_files, key=extract_block_num)

    # 1-based run mapping: run r maps to all_run_files[r - 1]
    valid_run_nums = SUBJECT_RUNS.get(sub_id, list(range(1, len(all_run_files) + 1)))
    run_files = [all_run_files[r - 1] for r in valid_run_nums if (r - 1) < len(all_run_files)]

    print(f"[-] Subject {sub_id}: Using {len(run_files)} runs")
    print(f"    Start file: {os.path.basename(run_files[0])} (Run {valid_run_nums[0]})")
    print(f"    End file  : {os.path.basename(run_files[-1])} (Run {valid_run_nums[-1]})")

    cond_indices = {c: [] for c in CONDITIONS}
    global_trial_counter = 0

    for r_file in run_files:
        df = pd.read_csv(r_file)
        df.columns = df.columns.str.strip()

        # Locate condition column dynamically
        if COND_COL in df.columns:
            actual_cond_col = COND_COL
        else:
            matching = [c for c in df.columns if "hierarh" in c.lower()]
            if matching:
                actual_cond_col = matching[0]
            else:
                raise KeyError(f"Condition column not found in {r_file}")

        # Iterate through rows:
        # Indices 0 and 1 are skipped (tossed initial Stay trials).
        # Valid trials start from row index 2 (df.iloc[2:]).
        for local_idx in range(len(df)):
            if local_idx >= TOSS_PER_RUN:
                c_val = str(df.iloc[local_idx][actual_cond_col]).strip()
                if c_val in cond_indices:
                    cond_indices[c_val].append(global_trial_counter)

            global_trial_counter += 1

    return cond_indices


def main():
    for sub in SUBJECTS:
        raw_id = sub.replace("sub-", "")
        print(f"\n==========================================================")
        print(f" Processing Subject: {sub}")
        print(f"==========================================================")

        # 1. Parse behavioral condition indices
        cond_indices = get_subject_condition_indices(sub, raw_id)
        if cond_indices is None:
            continue

        for c in CONDITIONS:
            print(f"    - {c}: {len(cond_indices[c])} trials")

        # 2. Check mask
        mask_file = os.path.join(BASE_DIR, "fmriprep", sub, "combined_mask.nii.gz")
        mask_cmd = ["-mask", mask_file] if os.path.isfile(mask_file) else []

        # 3. Process Cue and Probe events
        for event in EVENTS:
            pattern = os.path.join(GLM_DIR, sub, f"*{raw_id}*_{event}_TrialBetas.nii.gz")
            matched = glob.glob(pattern)
            if not matched:
                print(f"[!] Beta file missing for {sub} ({event}). Skipping.")
                continue
            beta_file = matched[0]

            # (A) Condition-wise 1-sample t-test (against 0)
            for cond in CONDITIONS:
                idx_list = cond_indices[cond]
                if len(idx_list) < 2:
                    continue
                selector = format_selector(idx_list)
                out_name = os.path.join(OUT_DIR, f"{sub}_{event}_{cond}_3dttest.nii.gz")

                cmd = [
                    "3dttest++", "-overwrite",
                    "-prefix", out_name
                ] + mask_cmd + [
                    "-setA", f"{beta_file}[{selector}]"
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                print(f"    [+] Saved 1-sample: {os.path.basename(out_name)}")

            # (B) Condition contrast (Two-sample comparison: condA vs condB)
            for cA, cB in CONTRASTS:
                idx_A = cond_indices[cA]
                idx_B = cond_indices[cB]
                if len(idx_A) < 2 or len(idx_B) < 2:
                    continue
                sel_A = format_selector(idx_A)
                sel_B = format_selector(idx_B)
                out_contrast = os.path.join(OUT_DIR, f"{sub}_{event}_{cA}-{cB}_3dttest.nii.gz")

                # -setA vs -setB calculates (setA - setB) contrast and t-statistic
                cmd_contrast = [
                    "3dttest++", "-overwrite",
                    "-prefix", out_contrast
                ] + mask_cmd + [
                    "-setA", f"{beta_file}[{sel_A}]",
                    "-setB", f"{beta_file}[{sel_B}]"
                ]
                subprocess.run(cmd_contrast, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                print(f"    [+] Saved contrast ({cA} - {cB}): {os.path.basename(out_contrast)}")

    print("\n[✓] All condition-level t-tests and contrasts completed.")
    print(f"[✓] Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()