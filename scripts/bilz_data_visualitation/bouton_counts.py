from pathlib import Path
from get_odor_recodting_data import load_trial, extract_odor_blocks

# Load only Testgroup
dataset = "Testgroup"
fly_trials = load_trial(dataset)

# Initialize counters
total_pre_columns = 0
total_post_columns = 0

for fname, (pre_df, post_df) in fly_trials.items():
    fly_name = Path(fname).stem

    blocks_pre = extract_odor_blocks(pre_df, dataset, fname, sheet="pre")
    blocks_post = extract_odor_blocks(post_df, dataset, fname, sheet="post") if post_df is not None else {}

    pre_cols = sum(df.shape[1] for df in blocks_pre.values() if not df.empty)
    post_cols = sum(df.shape[1] for df in blocks_post.values() if not df.empty)

    total_pre_columns += pre_cols
    total_post_columns += post_cols

print(f"✅ Total ROI columns in PRE condition (Testgroup): {total_pre_columns}")
print(f"✅ Total ROI columns in POST condition (Testgroup): {total_post_columns}")
print(f"📊 Combined total ROI columns (PRE + POST): {total_pre_columns + total_post_columns}")
