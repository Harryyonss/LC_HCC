import os
import glob
import pandas as pd
import numpy as np

# -------------------  User Settings  ---------------------
sub_list = [
    "11113",
    "11112",
    "11111",
    "11109",
    "11100",
    "11084",
    "10341",    
    # Add more subjects as needed
]

dataset_dir = "/data/backed_up/shared/ME_7T_Pilot/"
output_dir = os.path.join(dataset_dir, "3dDeconvolve/")

# ---------- Define Functions --------------

def write_stimtime(filepath, inputvec):
    """Write AFNI-style stimtime file (one line per run)."""
    with open(filepath, 'w') as f:
        for val in inputvec:
            if len(val) == 0 or val == '*':
                f.write('*\n')
            else:
                f.write(
                    np.array2string(np.around(np.array(val, dtype=float), 6), suppress_small=True)
                    .replace('\n', '')
                    .replace(']', '')
                    .replace('[', '')
                    .strip() + '\n'
                )

def process_subject(sub, dataset_dir, output_dir):
    """Generate EDS, IDS, Stay, Tossed timing files for GLMsingle."""
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

    # 3. finding the column for hierarchical_s
    cond_col = None
    for col in df.columns:
        if 'hierarhical' in col.lower() or 'hierarchical' in col.lower():
            cond_col = col
            break

    if cond_col is None:
        print(f"[Error] Could not find 'hierarhical_s' column. Available: {df.columns.tolist()}")
        return

    # Cue / Probe timing column finding
    cue_col = None
    probe_col = None
    for col in df.columns:
        if 'cue' in col.lower() and ('time' in col.lower() or 'prez' in col.lower()):
            cue_col = col
        elif 'probe' in col.lower() and ('time' in col.lower() or 'prez' in col.lower()):
            probe_col = col

    cue_col = cue_col or 'Time_Since_Run_Cue_Prez'
    probe_col = probe_col or 'Time_Since_Run_Probe_Prez'

    conditions = ['EDS', 'IDS', 'Stay', 'Tossed']
    timings = {
        'cue': {cond: [[] for _ in range(n_runs)] for cond in conditions},
        'probe': {cond: [[] for _ in range(n_runs)] for cond in conditions}
    }

    for i, run in enumerate(run_list):
        run_df = df[df['block'] == run].reset_index(drop=True)
        if run_df.empty:
            continue

        for trial_idx, row in run_df.iterrows():
            cue_time = row.get(cue_col)
            probe_time = row.get(probe_col)

            # 결측치 건너뛰기
            if pd.isna(cue_time) or pd.isna(probe_time):
                continue

            # 각 런의 처음 2개 시행은 조건 무관 Tossed로 분리
            if trial_idx < 2:
                timings['cue']['Tossed'][i].append(cue_time)
                timings['probe']['Tossed'][i].append(probe_time)
            else:
                # 3번째 시행부터는 EDS / IDS / Stay 분류
                val = str(row[cond_col]).strip().upper()
                
                matched = None
                if 'EDS' in val:
                    matched = 'EDS'
                elif 'IDS' in val:
                    matched = 'IDS'
                elif 'STAY' in val:
                    matched = 'Stay'

                if matched:
                    timings['cue'][matched][i].append(cue_time)
                    timings['probe'][matched][i].append(probe_time)

    # 5.  (_for_GLMsingle.1D)
    for cond in conditions:
        cue_path = os.path.join(sub_output_dir, f"cue_{cond}_stimtimes_for_GLMsingle.1D")
        write_stimtime(cue_path, timings['cue'][cond])
        print(f"Saved: {cue_path}")

        probe_path = os.path.join(sub_output_dir, f"probe_{cond}_stimtimes_for_GLMsingle.1D")
        write_stimtime(probe_path, timings['probe'][cond])
        print(f"Saved: {probe_path}")

# -------------------  Main Execution  ---------------------
if __name__ == "__main__":
    for sub in sub_list:
        try:
            process_subject(str(sub), dataset_dir, output_dir)
        except Exception as e:
            print(f"[Error] Failed to process subject {sub}: {e}")

    print("\nAll subjects processed successfully.")