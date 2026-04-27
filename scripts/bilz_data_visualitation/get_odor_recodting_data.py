import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from importlib import resources
from collections import Counter
import re

def load_trial(
    subfolder: str,
    pkg: str = "IBM_functions_package",
    data_root: str = "xlsxDataSource"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Loads each .xlsx in data_root/subfolder, reads sheets 0 and 1,
    and returns a dict of raw DataFrames (pre_df, post_df):
    { filename_stem: (raw_pre_df, raw_post_df) }.
    """
    data_dir = resources.files(pkg).joinpath(data_root, subfolder)
    result: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    for path in data_dir.rglob("*.xlsx"):
        sheets_raw = pd.read_excel(
            path,
            header=None,
            sheet_name=[0, 1],
            na_values=["#DIV/0!"]
        )
        if 0 not in sheets_raw or 1 not in sheets_raw:
            continue
        result[path.stem] = (sheets_raw[0], sheets_raw[1])

    return result


def extract_odor_blocks(
    raw_df: pd.DataFrame,
    subfolder: str,
    filename: str,
    sheet: str,
    odors: List[str] = None,
    block_length: int = 85
) -> Dict[str, pd.DataFrame]:
    """
    Processes a raw DataFrame by:
      1) Dropping the first column,
      2) Promoting the first row to headers,
      3) Normalizing column names (e.g., γ2 → g2),
      4) Extracting odor blocks of length `block_length`, skipping 2 rows between,
      5) Dropping columns with only NaN values.

    Returns a dict keyed by odor with DataFrames labeled by ROI columns.
    """
    if raw_df.empty:
        print(f"⚠️ Skipping {filename} ({sheet}): empty sheet")
        return {}

    if odors is None:
        odors = ["MCH", "3-OCT", "1-OCT", "OIL"]

    # Drop first column and check if enough rows remain
    df = raw_df.iloc[:, 1:].copy()
    if df.empty or df.shape[0] < 2:
        print(f"⚠️ Skipping {filename} ({sheet}): not enough data after column drop")
        return {}

    # Promote first row to header and normalize column names
    df.columns = df.iloc[0].astype(str).tolist()
    df.columns = [re.sub(r"^γ", "g", col) if isinstance(col, str) else col for col in df.columns]
    df = df.iloc[1:].reset_index(drop=True)

    # Find first valid data row
    first_valid_idx = df.apply(
        lambda row: pd.to_numeric(row, errors='coerce').notna().any(), axis=1
    ).idxmax()
    data = df.iloc[first_valid_idx:].reset_index(drop=True)

    fly = Path(filename).stem
    result: Dict[str, pd.DataFrame] = {}
    i = 0
    for odor in odors:
        block_rows = []
        read_rows = 0
        while i < len(data) and read_rows < block_length:
            row = data.iloc[i]
            numeric_row = pd.to_numeric(row, errors="coerce")
            if not numeric_row.isna().all() and not (numeric_row.fillna(0) == 0).all():
                block_rows.append(row.values)
            i += 1
            read_rows += 1

        if block_rows:
            block_df = pd.DataFrame(block_rows, columns=data.columns)
            block_df = block_df.dropna(axis=1, how='all')  # Drop columns that are all NaN
        else:
            block_df = pd.DataFrame(columns=data.columns)

        key = f"{subfolder}_{fly}_{sheet}_{odor.replace('-', '_')}"
        result[key] = block_df
        i += 2  # skip gap rows

    return result


def check_all_imports():
    subfolders = [
        "backwardConditioned",
        "Controls",
        "forwardConditioned",
        "Testgroup",
        "unpaired",
    ]

    # store all filenames for duplicate checking
    all_filenames = []

    for sub in subfolders:
        data = load_trial(sub)
        all_filenames.extend(data.keys())

        print(f"\nLoaded {len(data)} workbooks from '{sub}':")
        for fname, (pre_df, post_df) in data.items():
            print(f"  {fname:<30} → pre {pre_df.shape}, post {post_df.shape}")

    # Optional: detect duplicate filenames across subfolders
    dup_counts = Counter(all_filenames)
    duplicates = {fn: cnt for fn, cnt in dup_counts.items() if cnt > 1}
    if duplicates:
        print("\n⚠️ Duplicate filenames found across subfolders:")
        for fn, cnt in duplicates.items():
            print(f"  • {fn:<30} appears {cnt} times")
    else:
        print("\n✅ No duplicate filenames across subfolders.")
        
def check_all_pre_post_imports():
    subfolders = [
        "backwardConditioned",
        "Controls",
        "forwardConditioned",
        "Testgroup",
        "unpaired",
    ]

    for sub in subfolders:
        data = load_trial(sub)
        print(f"\n── {sub} ──")
        if not data:
            print("  (no files found)")
            continue

        print(f"Loaded {len(data)} workbooks from '{sub}':")
        for fname, (pre_df, post_df) in data.items():
            print(f"  {fname:<30} → pre {pre_df.shape}, post {post_df.shape}")


if __name__ == "__main__":

    # forwardConditioned = "forwardConditioned"
    # all_files = load_trial(forwardConditioned)

    # # Dictionary to store all flies' data
    # all_flies_data = {}

    # for fname, (pre_df, post_df) in all_files.items():
    #     fly_name = Path(fname).stem  # e.g., "fly_xxv"

    #     # Extract odor blocks
    #     blocks_pre = extract_odor_blocks(pre_df, forwardConditioned, fname, sheet="pre")
    #     blocks_post = extract_odor_blocks(post_df, forwardConditioned, fname, sheet="post")

    #     # Store in dictionary
    #     all_flies_data[fly_name] = {
    #         "pre": blocks_pre,
    #         "post": blocks_post
    #     }
        
        # backwardConditioned = "backwardConditioned"
        # all_files = load_trial(backwardConditioned)

        # # Dictionary to store all flies' data
        # all_flies_data = {}

        # for fname, (pre_df, post_df) in all_files.items():
        #     fly_name = Path(fname).stem  # e.g., "fly_xxv"

        #     # Extract odor blocks
        #     blocks_pre = extract_odor_blocks(pre_df, backwardConditioned, fname, sheet="pre")
        #     blocks_post = extract_odor_blocks(post_df, backwardConditioned, fname, sheet="post")

        #     # Store in dictionary
        #     all_flies_data[fly_name] = {
        #         "pre": blocks_pre,
        #         "post": blocks_post
        #     }
        
        # Controls = "Controls"
        # all_files = load_trial(Controls)

        # # Dictionary to store all flies' data
        # all_flies_data = {}

        # for fname, (pre_df, post_df) in all_files.items():
        #     fly_name = Path(fname).stem  # e.g., "fly_xxv"

        #     # Extract odor blocks
        #     blocks_pre = extract_odor_blocks(pre_df, Controls, fname, sheet="pre")
        #     blocks_post = extract_odor_blocks(post_df, Controls, fname, sheet="post")

        #     # Store in dictionary
        #     all_flies_data[fly_name] = {
        #         "pre": blocks_pre,
        #         "post": blocks_post
        #     }
            
            # unpaired = "unpaired"
            # all_files = load_trial(unpaired)

            # # Dictionary to store all flies' data
            # all_flies_data = {}

            # for fname, (pre_df, post_df) in all_files.items():
            #     fly_name = Path(fname).stem  # e.g., "fly_xxv"

            #     # Extract odor blocks
            #     blocks_pre = extract_odor_blocks(pre_df, unpaired, fname, sheet="pre")
            #     blocks_post = extract_odor_blocks(post_df, unpaired, fname, sheet="post")

            #     # Store in dictionary
            #     all_flies_data[fly_name] = {
            #         "pre": blocks_pre,
            #         "post": blocks_post

  Testgroup = "Testgroup"
  all_files = load_trial(Testgroup)
  # Dictionary to store all flies' data
  all_flies_data = {}

  for fname, (pre_df, post_df) in all_files.items():
      fly_name = Path(fname).stem  # e.g., "fly_xxv"

      # Extract odor blocks
      blocks_pre = extract_odor_blocks(pre_df, Testgroup, fname, sheet="pre")
      blocks_post = extract_odor_blocks(post_df, Testgroup, fname, sheet="post")

      # Store in dictionary
      all_flies_data[fly_name] = {
          "pre": blocks_pre,
          "post": blocks_post
          }
      

    # # Example: Print summary
    # for fly, recordings in all_flies_data.items():
    #     print(f"\nFly: {fly}")
    #     for phase in ["pre", "post"]:
    #         for block_name, df_block in recordings[phase].items():
    #             print(f"  {block_name}: {df_block.shape}")
    
    #check_all_imports()
    #check_all_pre_post_imports()

    
    