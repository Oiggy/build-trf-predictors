"""
Acoustic Predictors (Broadband Envelope and Onsets) Generation and Intergration into Predictor Matrices

This script processes audio stimuli and generates predictor matrices for use in 
neural encoding/decoding analyses (e.g., TRF/mTRF models). 

WORKFLOW:
1. Load stimulus matrix CSV containing experimental conditions and wav file references
2. Extract unique conditions from headers (e.g., SINGLE, BINAURAL MIXED, DICHOTIC ATTEND)
3. Process each unique wav file to extract acoustic features:
   - Acoustic envelope (amplitude modulation)
   - Acoustic onsets (positive temporal derivatives)
   - Handle both mono and stereo audio (averaging channels for stereo)
   - Apply bandpass filtering and resampling to match EEG/MEG data
4. Generate predictor matrix by:
   - Creating columns for each predictor type × condition combination
   - Example: acoustic broadband envelope single 1, acoustic broadband onsets binaural mixed 2
5. Populate predictor matrix:
   - Replace wav filenames with corresponding NDVar objects
   - Fill missing/empty cells with zero-filled NDVars (same dimensions)
6. Add stimulus duration column extracted from envelope time dimension
7. Save the final predictor matrices as pickle files

OUTPUT:
- predictor matrices saved as pickles in the derivatives/predictors directory
- Updated predictor_names.txt file listing all predictor columns

DEPENDENCIES:
- eelbrain: for audio processing and NDVar data structures
- pandas: for data organization and manipulation
- numpy: for numerical operations
- utils.params: for experiment-specific parameters (paths, filtering, resampling)

USAGE:
Typically used in auditory neuroscience experiments to create stimulus representations
that can be correlated with neural responses (EEG, MEG, ECoG) to model how the brain
encodes acoustic features.
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


def extract_envelope(wav_path: Path, 
                     low_freq: float, 
                     high_freq: float, 
                     resampling_freq: int,
                     filter_pad: str = 'reflect') -> Tuple[eelbrain.NDVar, int]:
    """
    Extract and process acoustic envelope from a wav file.
    Handles both mono and stereo files.
    
    Args:
        wav_path: Path to the wav file
        low_freq: Low frequency cutoff for bandpass filter
        high_freq: High frequency cutoff for bandpass filter
        resampling_freq: Target sampling frequency
        filter_pad: Padding method for filtering
        
    Returns:
        Tuple of (processed envelope, number of channels)
    """
    # Load the sound file
    wav = eelbrain.load.wav(wav_path)
    
    if wav.ndim == 2:  # Stereo
        print(f"    Stereo file detected")
        
        # Extract left channel (index 0)
        left_channel = wav.sub(channel=0)
        left_envelope = left_channel.envelope()
        left_envelope = eelbrain.filter_data(left_envelope, low_freq, high_freq, pad=filter_pad)
        left_envelope = eelbrain.resample(left_envelope, resampling_freq)
        print(f"    Left envelope shape: {left_envelope.x.shape}")
        
        # Extract right channel (index 1)
        right_channel = wav.sub(channel=1)
        right_envelope = right_channel.envelope()
        right_envelope = eelbrain.filter_data(right_envelope, low_freq, high_freq, pad=filter_pad)
        right_envelope = eelbrain.resample(right_envelope, resampling_freq)
        print(f"    Right envelope shape: {right_envelope.x.shape}")
        
        # Average the envelopes
        averaged_envelope = (left_envelope + right_envelope) / 2
        print(f"    Averaged envelope shape: {averaged_envelope.x.shape}")
        
        return averaged_envelope, 2
        
    else:  # Mono
        print(f"    Mono file detected")
        
        # Compute the acoustic envelope
        envelope = wav.envelope()
        
        # Filter the envelope
        envelope = eelbrain.filter_data(envelope, low_freq, high_freq, pad=filter_pad)
        
        # Resample the envelope
        envelope = eelbrain.resample(envelope, resampling_freq)
        print(f"    Envelope shape: {envelope.x.shape}")
        
        return envelope, 1

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
    Manage predictor_names.txt file by creating or updating it with column names from dataframes.
    
    Args:
        dataframes: List of DataFrames to extract predictor names from
        ignore_headers: List of header names to ignore (case-insensitive)
        output_file: Path to the output file (default: predictor_names.txt in script directory)
        script_dir: Directory of the script (default: current file's parent directory)
        
    Returns:
        Path to the predictor_names.txt file
    """
    if ignore_headers is None:
        ignore_headers = ['duration', 'durations', 'durations seconds 1', 'durations seconds 2']
    
    if script_dir is None:
        script_dir = Path(__file__).parent
    
    if output_file is None:
        output_file = script_dir / 'predictor_names.txt'
    
    # Get headers from all dataframes that are not in ignore list
    valid_headers = []
    for df in dataframes:
        for col in df.columns:
            if col.lower() not in [h.lower() for h in ignore_headers]:
                valid_headers.append(col)
    
    # Remove duplicates while preserving order
    valid_headers = list(dict.fromkeys(valid_headers))
    
    print(f"\nManaging predictor_names.txt")
    
    if not output_file.exists():
        with open(output_file, 'w') as f:
            for header in valid_headers:
                f.write(f"{header}\n")
        print(f"✓ Created file with {len(valid_headers)} predictor names at: {output_file}")
        
    else:
        # Read existing names from file
        with open(output_file, 'r') as f:
            existing_names = set(line.strip() for line in f if line.strip())
        
        # Find names that need to be added
        names_to_add = [header for header in valid_headers if header not in existing_names]
        
        if names_to_add:
            with open(output_file, 'a') as f:
                for header in names_to_add:
                    f.write(f"{header}\n")
            print(f"✓ Added {len(names_to_add)} new predictor names to: {output_file}")
        else:
            print(f"✓ File already up to date at: {output_file}")
    
    return output_file


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

stim_mat_dir = 'binaural-cocktail-bids/derivatives/predictors/matrix_stimuli.csv'

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

# 3. Process with envelope extraction and onset generation
stim_ndvar_dict = {}

for wav_file in stimulus_files:
    print(f"Processing: {wav_file}")
    wav_path = Path(STIMULUS_DIR) / wav_file
    envelope, n_channels = extract_envelope(
        wav_path=wav_path,
        low_freq=BANDPASS_FILTER_PARAMS['LOW_FREQUENCY'],
        high_freq=BANDPASS_FILTER_PARAMS['HIGH_FREQUENCY'],
        resampling_freq=RESAMPLING_FREQUENCY
    )
    
    print(f"  Number of channels: {n_channels}")
    
    # Generate onset from envelope
    onsets = envelope.diff('time').clip(0)
    print(f"  Onsets shape: {onsets.x.shape}")
    
    # Store in dictionary
    stim_ndvar_dict[wav_file] = {
        'acoustic broadband envelope': envelope,
        'acoustic broadband onset': onsets,
        'n_channels': n_channels
    }

print(f"\nTotal stimuli processed: {len(stim_ndvar_dict)}")
print(f"Dictionary keys: {list(stim_ndvar_dict.keys())[:5]}...")

# 4. Generate predictor matrix
pred_df = generate_predictor_matrix(
    csv_path=stim_mat_dir,
    predictor_names=['acoustic broadband envelope', 'acoustic broadband onset'],
    exclude_headers=['STIMULI LIST 1', 'STIMULI LIST 2', 'CONDITION LIST 1']
)

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

