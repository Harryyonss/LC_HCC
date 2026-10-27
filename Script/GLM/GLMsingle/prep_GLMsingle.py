import os
import glob
import pandas as pd
import numpy as np

# -------------------  User Settings  ---------------------
# Add or modify subject IDs here:
sub_list = [
    # "11113",
    # "11112",
    # "11111",
    # "11109",
    # "11100",
    # "11084",
    # "10341",
    # "11114",
    "11116"    
    # Add more subjects as needed
]

dataset_dir = "/data/backed_up/shared/ME_7T_Pilot/"
output_dir = os.path.join(dataset_dir, "3dDeconvolve/")

# ---------- Define Functions --------------

def write_stimtime(filepath, inputvec):
    """Write AFNI-style stimtime file (one line per run)."""
    with open(filepath, 'w') as f:
        for val in inputvec[0]:
            if val == '*':
                f.write(val + '\n')
            else:
                f.write(
                    np.array2string(np.around(val, 6), suppress_small=True)
                    .replace('\n', '')
                    .replace(']', '')
                    .replace('[', '') + '\n'
                )

def process_subject(sub, dataset_dir, output_dir):
    """Generate cue and probe stimtime files for GLMsingle."""
    print(f"\n==================== Processing Subject: {sub} ====================")
    csv_dir = f"/data/backed_up/shared/ME_7T_Pilot/behav_data/{sub}"

    # Exception for subjects with non-standard run counts
    sub_int = int(sub)
    if sub_int == 11102:
        run_list = list(range(1, 12))   # runs 1 ~ 11
    elif sub_int == 11113:
        run_list = list(range(3, 13))   # runs 3 ~ 12
    else:
        run_list = list(range(1, 13))   # standard: runs 1 ~ 12

    n_runs = len(run_list)

    # 1. Create subject output directory
    sub_output_dir = os.path.join(output_dir, f"sub-{sub}")
    os.makedirs(sub_output_dir, exist_ok=True)

    # 2. Load behavioral CSV data
    files = glob.glob(os.path.join(csv_dir, f"sub-{sub}_task-HCC_block*.csv"))
    files.sort()

    if not files:
        print(f"[Warning] No behavioral CSV files found for sub-{sub} in {csv_dir}.")
        return

    dfs = [pd.read_csv(file) for file in files]
    df = pd.concat(dfs, ignore_index=True)

    # 3. Extract Cue and Probe timings (including all trials)
    Cue_stimtime = [['*'] * n_runs]
    Probe_stimtime = [['*'] * n_runs]

    for i, run in enumerate(run_list):
        run_df = df[df['block'] == run].reset_index(drop=True)

        Cue_times = run_df['Time_Since_Run_Cue_Prez'].dropna().tolist()
        Probe_times = run_df['Time_Since_Run_Probe_Prez'].dropna().tolist()

        Cue_stimtime[0][i] = Cue_times
        Probe_stimtime[0][i] = Probe_times

    # 4. Save timing files
    cue_out_path = os.path.join(sub_output_dir, "cue_stimtimes_for_GLMsingle.1D")
    probe_out_path = os.path.join(sub_output_dir, "probe_stimtimes_for_GLMsingle.1D")

    write_stimtime(cue_out_path, Cue_stimtime)
    write_stimtime(probe_out_path, Probe_stimtime)

    print(f"Saved: {cue_out_path}")
    print(f"Saved: {probe_out_path}")

# -------------------  Main Execution  ---------------------
if __name__ == "__main__":
    for sub in sub_list:
        try:
            process_subject(str(sub), dataset_dir, output_dir)
        except Exception as e:
            print(f"[Error] Failed to process subject {sub}: {e}")

    print("\nAll subjects processed successfully.")