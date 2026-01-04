"""
Generate and Integrate Word Onset Predictors into Existing Matrices

Overview
--------
This script reads an experimental stimulus matrix and adds a new predictor family,
**word onset**, into the existing predictor matrices used for neural encoding
(e.g., TRF/mTRF). It creates human-readable predictor column names (space style,
e.g., "word onset binaural mixed 1"), substitutes WAV filenames with Eelbrain
NDVar objects loaded from per-stimulus pickle files, zero-fills missing cells to
preserve shapes/dims, then merges and saves the updated matrices.

What it does
------------
1) Load stimulus matrix
   • Reads matrix_stimuli.csv and extracts condition names from headers that end
     in " 1"/" 2" or "_1"/"_2" (both styles supported).

2) Discover stimuli
   • Resolves the actual stimulus columns present for stimulus 1 and stimulus 2.
   • Collects unique .wav filenames referenced in those columns.

3) Build an empty predictor matrix
   • For each non-excluded header, creates new columns prefixed with "word onset "
     (space style: no underscores), e.g., "word onset single 1".

4) Load per-stimulus NDVars
   • Scans derivatives/predictors/word_onsets for files named
     "<basename>_word_onsets.pickle".
   • Unpickles the "word_onsets" NDVar and standardizes its NDVar.name to "<wav>.wav".
   • Builds a mapping { "<wav>.wav": { "word onset": NDVar } } (strictly matching
     only stimuli that exist in the stimulus directory).

5) Populate the predictor matrix
   • Replaces each WAV filename cell with the corresponding NDVar based on the
     predictor prefix ("word onset").
   • Fills missing/empty cells with zero NDVars that match dims/shape
     (family-aware, robust to sparsity).

6) Split, merge, and save
   • Splits the new predictor matrix into stimulus 1 and 2 views (accepts " 1"/"_1").
   • Merges each view into existing matrices:
       matrix_predictors_1.pkl and matrix_predictors_2.pkl
     (overwrites existing columns of the same name; adds new ones otherwise).
   • Saves the merged DataFrames back to disk.

7) Maintain a name registry
   • Updates predictor_names.txt with any new predictor column names
     (case/whitespace-insensitive de-dup; duration-like headers ignored).

Inputs
------
• binaural-cocktail-bids/derivatives/predictors/matrix_stimuli.csv
• stimulus WAV files under the configured stimulus directory (utils.params.STIMULUS_DIR)
• word onset pickles in:
  binaural-cocktail-bids/derivatives/predictors/word_onsets/*_word_onsets.pickle

Outputs
-------
• Updated predictor matrices:
  - binaural-cocktail-bids/derivatives/predictors/matrix_predictors_1.pkl
  - binaural-cocktail-bids/derivatives/predictors/matrix_predictors_2.pkl
• Updated predictor_names.txt (in the script directory)

Conventions & Assumptions
-------------------------
• Predictor columns use human-readable space style:
  "word onset <condition> <stimulus_index>" (e.g., "word onset dichotic attend 2").
• Header parsing accepts both " … 1/2" and " …_1/_2".
• NDVar zero-filling ensures consistent shapes/dims across rows/columns.

Customize here
--------------
• `new_predictor_names` to add more predictors.
• Directory paths and file patterns in `build_word_onsets_map`.
• Exclusion list in `generate_predictor_matrix` to skip non-predictor headers.
"""



import pandas as pd
import eelbrain
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from analysis.params import STIMULUS_DIR, RESAMPLING_FREQUENCY, BANDPASS_FILTER_PARAMS

def load_stimulus_matrix(csv_path: str) -> pd.DataFrame:
    """
    Load the stimulus matrix CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame containing the stimulus matrix
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded stimulus matrix with {len(df)} rows and {len(df.columns)} columns")
    return df


def generate_predictor_matrix(csv_path: str, 
                              predictor_names: List[str],
                              exclude_headers: List[str] = None) -> pd.DataFrame:
    if exclude_headers is None:
        exclude_headers = ['STIMULI LIST 1', 'STIMULI LIST 2', 'CONDITION LIST 1']
    
    df = pd.read_csv(csv_path)
    print(f"Loaded matrix with {len(df)} rows and {len(df.columns)} columns")
    
    new_df = pd.DataFrame()
    for column in df.columns:
        if column in exclude_headers:
            continue
        lowercase_column = column.lower().strip()
        for predictor in predictor_names:
            # NOTE: space between predictor and condition (no underscore)
            new_column_name = f"{predictor} {lowercase_column}"
            new_df[new_column_name] = df[column]
            print(f"Created column: {new_column_name}")
    
    print(f"New matrix has {len(new_df)} rows and {len(new_df.columns)} columns")
    return new_df


def extract_conditions_from_headers(df: pd.DataFrame, 
                                    exclude_headers: List[str] = None) -> List[str]:
    """
    Extract unique condition names from dataframe headers.
    
    Args:
        df: DataFrame containing stimulus information
        exclude_headers: List of header names to exclude
        
    Returns:
        List of unique condition names
    """
    if exclude_headers is None:
        exclude_headers = ['STIMULI LIST 1', 'STIMULI LIST 2', 'CONDITION LIST 1']
    
    conditions = []
    
    for column in df.columns:
        # Skip excluded headers
        if column in exclude_headers:
            continue
        
        col = column.strip()
        if col.endswith(' 1') or col.endswith(' 2'):
            conditions.append(col.rsplit(' ', 1)[0])
        elif col.endswith('_1') or col.endswith('_2'):
            conditions.append(col.rsplit('_', 1)[0])
    
    # Remove duplicates while preserving order
    unique_conditions = list(dict.fromkeys(conditions))
    
    print(f"Extracted {len(unique_conditions)} unique conditions from headers")
    print(f"Conditions: {unique_conditions}")
    
    return unique_conditions


def get_stimulus_columns(df: pd.DataFrame,conditions: List[str], stimulus_number: int) -> List[str]:
    """
    Generate column names for a given stimulus number.
    
    Args:
        conditions: List of condition names (e.g., ['SINGLE', 'BINAURAL MIXED'])
        stimulus_number: Stimulus number (1 or 2)
        
    Returns:
        List of column names
    """
    cols = []
    for cond in conditions:
        space_style = f"{cond} {stimulus_number}"
        underscore_style = f"{cond}_{stimulus_number}"
        if space_style in df.columns:
            cols.append(space_style)
        elif underscore_style in df.columns:
            cols.append(underscore_style)
        else:
            print(f"Warning: no column for condition '{cond}', stimulus {stimulus_number} "
                  f"(tried '{space_style}' and '{underscore_style}')")
    return cols


def extract_unique_stimuli(df: pd.DataFrame, 
                          columns: List[str], 
                          file_extension: str = '.wav') -> List[str]:
    """
    Extract unique stimulus filenames from specified columns.
    
    Args:
        df: DataFrame containing stimulus information
        columns: List of column names to extract from
        file_extension: File extension to filter for
        
    Returns:
        List of unique stimulus filenames
    """
    stimulus = []
    
    for column in columns:
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in dataframe")
            continue
            
        # Get non-null values and convert to string
        files = df[column].dropna().astype(str)
        
        # Filter for files with the specified extension
        files = files[files.str.endswith(file_extension)].tolist()
        stimulus.extend(files)
    
    # Remove duplicates while preserving order
    unique_stimuli = list(dict.fromkeys(stimulus))
    
    print(f"Found {len(unique_stimuli)} unique {file_extension} files from {len(columns)} columns")
    return unique_stimuli

def build_word_onsets_map(stimulus_dir, word_onsets_dir,strict_match=True):
    """
    Returns a dict: mapping[wav_name] = {'word_onsets': NDVar}

    - Scans `stimulus_dir` for *.wav names
    - Scans `word_onsets_dir` for *_word_onsets.pickle
    - Unpickles and extracts NDVar from 'word_onsets' column (row 0)
    - If strict_match=True, only maps entries whose wav exists in stimulus_dir
    """
    # Ensure eelbrain is importable so NDVar unpickles correctly
    try:
        import eelbrain  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "eelbrain must be importable to unpickle NDVar objects. "
            "Install with: pip install eelbrain"
        ) from e

    stim_dir = Path(stimulus_dir)
    wo_dir = Path(word_onsets_dir)

    wav_names = {p.name for p in stim_dir.glob("*.wav")}
    mapping = {}

    suffix = "_word_onsets.pickle"
    for pkl in wo_dir.glob(f"*{suffix}"):
        base = pkl.name[:-len(suffix)]          # e.g. "female_1"
        wav_name = f"{base}.wav"                # e.g. "female_1.wav"
        if strict_match and wav_name not in wav_names:
            continue

        df = pd.read_pickle(pkl)                # shape (1, 3) expected
        if "word_onsets" not in df.columns:
            raise RuntimeError(f"'word_onsets' column missing in {pkl}")

        ndvar = df["word_onsets"].iloc[0]

        # Optional: standardize NDVar name to the WAV filename for nicer display
        try:
            ndvar = ndvar.copy(name=wav_name)
        except Exception:
            pass

        # IMPORTANT: key matches the predictor column prefix used elsewhere
        mapping[wav_name] = {"word onset": ndvar}

    return mapping

def populate_predictor_matrix(mapping: Dict[str, Dict[str, Any]],
                              df: pd.DataFrame,
                              verbose: bool = True) -> pd.DataFrame:
    """
    Replace wav filenames in a predictor DataFrame with NDVar objects from `mapping`,
    and fill missing/empty predictor cells with zero NDVars. Prints validation lines
    showing predictor name, column, and row index when substitutions occur.

    mapping[wav_name] = { 'acoustic broadband envelope': NDVar, 'acoustic broadband onsets': NDVar, 'n_channels': int, ... }
    df: predictor DataFrame with columns like 'acoustic broadband envelope *', 'acoustic broadband onsets *', etc.
    """

    # --- Inner helper to build zero NDVar with identical dims/shape ---
    def zero_like(ndv, name='zero_predictor'):
        return eelbrain.NDVar(
            np.zeros(ndv.x.shape, dtype=getattr(ndv.x, "dtype", float)),
            dims=ndv.dims,
            name=name
        )

    out = df.copy()
    if not mapping or out.empty:
        print("Nothing to substitute: empty mapping or DataFrame.")
        return out

    # Case-insensitive lookup by basename
    filename_lookup = {Path(k).name.lower(): k for k in mapping.keys()}

    # Candidate predictors from first mapping entry (ignore metadata)
    first_item = next(iter(mapping.values()))
    candidate_predictors = [k for k in first_item.keys() if k != "n_channels"]

    # Determine predictor for a column
    def predictor_for_column(col: str) -> str | None:
        for p in candidate_predictors:
            if col == p or col.startswith(p + " ") or col.startswith(p + "_"):
                return p
        return None

    def is_ndvar(x) -> bool:
        return hasattr(x, "dims") and hasattr(x, "x")

    # Find a good reference NDVar for creating zeros for this (row, column, predictor)
    def find_reference_ndvar(row_idx, col_name, pred) -> Any | None:
        # 1) Prefer an NDVar already present in this row
        row_vals = out.loc[row_idx]
        for v in row_vals:
            if is_ndvar(v):
                return v
        # 2) Else, an NDVar already present in this column
        for v in out[col_name]:
            if is_ndvar(v):
                return v
        # 3) Else, any NDVar of this predictor from mapping
        for bundle in mapping.values():
            if pred in bundle and is_ndvar(bundle[pred]):
                return bundle[pred]
        return None

    # --- Phase 1: substitute wavs with NDVars where predictor matches ---
    total_cells = replaced = mismatched_predictor = missing_wav = 0

    for col in out.columns:
        pred = predictor_for_column(col)
        if pred is None:
            continue

        for idx, val in out[col].items():
            # Skip non-strings and non-wavs (we'll handle missing/empty later)
            if pd.isna(val) or not isinstance(val, str) or not val.lower().endswith(".wav"):
                continue

            total_cells += 1
            wav_key = filename_lookup.get(Path(val).name.lower())
            if wav_key is None:
                missing_wav += 1
                continue

            ndvar_bundle = mapping[wav_key]
            if pred not in ndvar_bundle:
                mismatched_predictor += 1
                continue

            out.at[idx, col] = ndvar_bundle[pred]
            replaced += 1

            if verbose:
                print(f"[SUBSTITUTE] predictor='{pred}' | wav='{Path(wav_key).name}' "
                      f"-> df.loc[{repr(idx)}, '{col}']")

    print(f"Substitution summary: checked {total_cells} wav cells "
          f"-> replaced {replaced}, missing_wav {missing_wav}, predictor_mismatch {mismatched_predictor}")

    # --- Phase 2: fill missing/empty predictor cells with zero NDVars ---
    zeros_filled = 0
    for col in out.columns:
        pred = predictor_for_column(col)
        if pred is None:
            continue

        for idx, val in out[col].items():
            # Missing/empty cell?
            is_empty_str = isinstance(val, str) and val.strip() == ""
            if pd.isna(val) or is_empty_str:
                ref = find_reference_ndvar(idx, col, pred)
                if ref is not None:
                    out.at[idx, col] = zero_like(ref, name=f"zero_predictor")
                    zeros_filled += 1
                    if verbose:
                        ref_name = getattr(ref, "name", "NDVar")
                        print(f"[ZERO-FILL] predictor='{pred}' -> df.loc[{repr(idx)}, '{col}'] "
                              f"(ref='{ref_name}')")
                # else: leave as-is; cannot infer dims

    print(f"Zero-fill summary: inserted {zeros_filled} zero NDVars into missing/empty predictor cells.")

    return out


def merge_predictor_dataframes(base_df: pd.DataFrame,
                               new_df: pd.DataFrame,
                               *,
                               align_on_index: bool = True) -> pd.DataFrame:
    """
    Merge/refresh predictor data:
    - If a column in new_df already exists in base_df -> OVERWRITE base_df[col].
    - If a column in new_df does not exist in base_df -> ADD it.
    - Optionally align new_df to base_df's index to avoid length mismatches.

    Args:
        base_df: Existing predictor DataFrame
        new_df:  New predictor DataFrame to merge (used to update/add columns)
        align_on_index: If True, reindex new_df to base_df.index before merging

    Returns:
        Merged DataFrame
    """
    merged_df = base_df.copy(deep=True)
    new_df_aligned = new_df.reindex(merged_df.index) if align_on_index else new_df

    # Informative note if indices don't fully overlap
    if align_on_index and not new_df.index.equals(merged_df.index):
        missing = merged_df.index.difference(new_df.index)
        extra   = new_df.index.difference(merged_df.index)
        if len(missing):
            print(f"Note: {len(missing)} base_df row(s) not present in new_df; "
                  f"kept existing values for those rows.")
        if len(extra):
            print(f"Note: {len(extra)} row(s) in new_df not present in base_df; "
                  f"these rows are ignored during merge.")

    for col in new_df_aligned.columns:
        if col in merged_df.columns:
            merged_df[col] = new_df_aligned[col]
            print(f"Overwrote existing column: {col}")
        else:
            merged_df[col] = new_df_aligned[col]
            print(f"Added new column: {col}")

    return merged_df


def save_predictor_dataframe(df: pd.DataFrame,
                             filename: str,
                             output_dir: str = 'binaural-cocktail-bids/derivatives/predictors') -> Path:
    """
    Save predictor DataFrame as a pickle file.
    
    Args:
        df: DataFrame to save
        filename: Name of the pickle file (without .pkl extension)
        output_dir: Directory to save the file (default: 'binaural-cocktail-bids/derivatives/predictors')
        
    Returns:
        Path to the saved file
    """
    output_path = Path(output_dir)
    
    # Create directory if it doesn't exist
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {output_path}")
    
    # Add .pkl extension if not present
    if not filename.endswith('.pkl'):
        filename = f"{filename}.pkl"
    
    # Save the dataframe
    output_file = output_path / filename
    df.to_pickle(output_file)
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✓ Saved: {output_file} ({file_size_mb:.2f} MB)")
    
    return output_file

def manage_predictor_names_file(dataframes: List[pd.DataFrame],
                                ignore_headers: List[str] = None,
                                output_file: Path = None,
                                script_dir: Path = None) -> Path:
    """
    Manage predictor_names.txt by creating or updating it with column names from DataFrames.

    Behavior:
    - Ignores headers in `ignore_headers` (case-insensitive).
    - Only appends names that are not already present in the file (case/whitespace-insensitive).
    - If no new names are found, the file is left untouched.

    Returns:
        Path to predictor_names.txt
    """
    def _norm(s: str) -> str:
        return s.strip().lower()

    if ignore_headers is None:
        ignore_headers = ['duration', 'durations', 'durations seconds 1', 'durations seconds 2']
    ignore_norm = {_norm(h) for h in ignore_headers}

    if script_dir is None:
        try:
            script_dir = Path(__file__).parent
        except NameError:
            script_dir = Path('.').resolve()

    if output_file is None:
        output_file = script_dir / 'predictor_names.txt'

    # Collect headers from all DataFrames, filtering ignored ones; de-dup while preserving order
    valid_headers: List[str] = []
    seen_norm = set()
    for df in dataframes:
        for col in df.columns:
            n = _norm(col)
            if n in ignore_norm:
                continue
            if n not in seen_norm:
                valid_headers.append(col)
                seen_norm.add(n)

    print("\nManaging predictor_names.txt")

    if not output_file.exists():
        with open(output_file, 'w', encoding='utf-8') as f:
            for header in valid_headers:
                f.write(f"{header}\n")
        print(f"✓ Created file with {len(valid_headers)} predictor names at: {output_file}")
        return output_file

    # Read existing names and build a normalized set for robust comparison
    with open(output_file, 'r', encoding='utf-8') as f:
        existing_raw = [line.strip() for line in f if line.strip()]
    existing_norm = {_norm(x) for x in existing_raw}

    # Only add headers not already present (normalized comparison)
    names_to_add = [h for h in valid_headers if _norm(h) not in existing_norm]

    if names_to_add:
        with open(output_file, 'a', encoding='utf-8') as f:
            for header in names_to_add:
                f.write(f"{header}\n")
        print(f"✓ Added {len(names_to_add)} new predictor name(s) to: {output_file}")
    else:
        print("✓ No new predictor names; file unchanged.")

    return output_file



# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

stim_mat_dir = 'binaural-cocktail-bids/derivatives/predictors/matrix_stimuli.csv'
new_predictor_names = ['word onset']

# 1. Load data
df = load_stimulus_matrix(stim_mat_dir)

# 2. Extract conditions from headers
conditions = extract_conditions_from_headers(
    df, 
    exclude_headers=['STIMULI LIST 1', 'STIMULI LIST 2', 'CONDITION LIST 1']
)

# Get columns for both stimuli
columns_1 = get_stimulus_columns(df, conditions, 1)
columns_2 = get_stimulus_columns(df, conditions, 2)

# Extract stimuli
all_columns = columns_1 + columns_2
stimulus_files = extract_unique_stimuli(df, all_columns)

# 3. Generate predictor matrix
pred_df = generate_predictor_matrix(
    csv_path=stim_mat_dir,
    predictor_names=new_predictor_names,
    exclude_headers=['STIMULI LIST 1', 'STIMULI LIST 2', 'CONDITION LIST 1']
)

# 4. Process each stimulus to extract NDVars from the pickle files
stim_ndvar_dict = build_word_onsets_map(
    stimulus_dir="binaural-cocktail-bids/stimulus",
    word_onsets_dir="binaural-cocktail-bids/derivatives/predictors/word_onsets",
    strict_match=True)


# 5. Populate predictor matrix with NDVars and fill missing with zeros NDVars
pred_df = populate_predictor_matrix(stim_ndvar_dict, pred_df)

# 6. Separate pred_df into two dataframes for clarity
pred_df_1_cols = [c for c in pred_df.columns if c.endswith(' 1') or c.endswith('_1')]
pred_df_2_cols = [c for c in pred_df.columns if c.endswith(' 2') or c.endswith('_2')]
pred_df_1 = pred_df[pred_df_1_cols]
pred_df_2 = pred_df[pred_df_2_cols]

# 7. Merge new predictors into existing matrices
# Load pre-existing predictors matrix pickles 
pred_mat_10 = pd.read_pickle('binaural-cocktail-bids/derivatives/predictors/matrix_predictors_1.pkl')
pred_mat_20 = pd.read_pickle('binaural-cocktail-bids/derivatives/predictors/matrix_predictors_2.pkl')

# Merge with newly created predictor dataframes
pred_mat_11 = merge_predictor_dataframes(pred_mat_10, pred_df_1)
pred_mat_21 = merge_predictor_dataframes(pred_mat_20, pred_df_2)

# 8. Save merged dataframes, overwriting existing predictor matrices
output_dir = 'binaural-cocktail-bids/derivatives/predictors'

save_predictor_dataframe(
    df=pred_mat_11,
    filename='matrix_predictors_1.pkl',
    output_dir=output_dir
)

save_predictor_dataframe(
    df=pred_mat_21,
    filename='matrix_predictors_2.pkl',
    output_dir=output_dir
)

# 9. Manage predictor_names.txt file
manage_predictor_names_file(
    dataframes=[pred_mat_11, pred_mat_21],
    ignore_headers=['duration', 'durations', 'durations seconds 1', 'durations seconds 2']
)


