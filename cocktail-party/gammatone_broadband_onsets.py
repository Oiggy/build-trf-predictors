"""
GAMMATONE_BROADBAND_ONSETS.PY - Compute onsets from broadband envelopes

====================================================================
HOW TO RUN
====================================================================
From project root directory:
    python gammatone_broadband_onsets.py

Note: Automatically calls gammatone_broadband_envelopes.py if needed

====================================================================
REQUIRED DIRECTORY STRUCTURE
====================================================================
~/Github Dataset/cocktail-party/
├── dataset/
│   └── stimuli/
│       └── *.wav
└── predictors/
    └── gammatones/                 # Output directory (auto-created)

~/Github Repositories/benchmark-pipelines/
└── analysis/
    └── params.py                   # Required parameters file

====================================================================
OUTPUT STRUCTURE (CREATED BY SCRIPT)
====================================================================
~/Github Dataset/cocktail-party/predictors/gammatones/
├── envelopes.h5                 # Input (auto-created if missing)
├── onsets.h5                    # Output (broadband only)
├── grid_broadband_onsets.png    # Visualization
└── gammatone.log                # Processing log

====================================================================
PROCESSING PIPELINE (PER STIMULUS)
====================================================================
1. Load broadband envelope from cache
2. Compute first difference
3. Half-wave rectify (keep only positive changes)
4. Save to HDF5
5. Generate visualization grid

====================================================================
OUTPUT FILES - KEY STRUCTURE
====================================================================
onsets.h5 (broadband features only):
    "<stimulus_name>/broadband/broad"    # Broadband onset

All arrays: shape (n_timepoints, 1), sfreq = RESAMPLING_FREQUENCY

====================================================================
REQUIRED PACKAGES
====================================================================
pip install numpy h5py matplotlib
"""

from pathlib import Path
import sys
import subprocess
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Add parent directory to path for logging_config import
SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

# Setup path to params.py
HOME = Path.home()
PARAMS_DIR = HOME / "Github Repositories" / "benchmark-pipelines"
if str(PARAMS_DIR) not in sys.path:
    sys.path.insert(0, str(PARAMS_DIR))

from analysis.params import RESAMPLING_FREQUENCY, BANDPASS_FILTER_PARAMS

# Path configuration
COCKTAIL_PARTY_DIR = HOME / "Github Dataset" / "cocktail-party"
STIM_DIR = COCKTAIL_PARTY_DIR / "dataset" / "stimuli"
OUT_DIR = COCKTAIL_PARTY_DIR / "predictors" / "gammatones"

ENVELOPES_H5 = OUT_DIR / "envelopes.h5"
ONSETS_H5 = OUT_DIR / "onsets.h5"
GRID_PNG = OUT_DIR / "grid_broadband_onsets.png"

# Setup logging
from logging_config import setup_logging
logger = setup_logging(
    log_filename="gammatone.log",
    output_dir=OUT_DIR,
    script_name="gammatone_broadband_onsets"
)


def ensure_broadband_envelopes():
    """Run envelope script if envelopes don't exist"""
    if not ENVELOPES_H5.exists():
        logger.info("Broadband envelopes not found. Running gammatone_broadband_envelopes.py...")
        script_path = Path(__file__).parent / "gammatone_broadband_envelopes.py"
        subprocess.run([sys.executable, str(script_path)], check=True)


def require_group(h5_file, group_path):
    """Create nested groups in HDF5 file"""
    current_group = h5_file
    for part in group_path.strip("/").split("/"):
        if part:
            current_group = current_group.require_group(part)
    return current_group


def write_dataset(h5_file, key, data_2d, attrs):
    """Write dataset to HDF5 with attributes"""
    key = key.strip("/")
    parent_path = "/".join(key.split("/")[:-1])
    dataset_name = key.split("/")[-1]
    parent_group = require_group(h5_file, parent_path) if parent_path else h5_file
    if dataset_name in parent_group:
        del parent_group[dataset_name]
    dset = parent_group.create_dataset(dataset_name, data=data_2d, compression="gzip")
    for attr_name, attr_value in attrs.items():
        dset.attrs[attr_name] = attr_value


def onset_from_envelope(envelope):
    """Compute onset strength (half-wave rectified difference)"""
    envelope = np.asarray(envelope, dtype=float).ravel()
    diff = np.diff(envelope, prepend=envelope[0])
    diff[diff < 0] = 0.0
    return np.asarray(diff, dtype=float).ravel()


def grid_size(n_plots):
    """Calculate rows and columns for subplot grid"""
    cols = int(math.ceil(math.sqrt(n_plots)))
    rows = int(math.ceil(n_plots / cols))
    return rows, cols


def plot_broadband_onset_timeseries(h5_path, wav_stems, out_png):
    """Create grid of broadband onset time series"""
    logger.info(f"Creating broadband onset visualization grid")
    items = []
    with h5py.File(h5_path, "r", locking=False) as h5:
        for stem in wav_stems:
            key = f"{stem}/broadband/broad"
            if key in h5:
                onset = h5[key][:].ravel()
                sfreq = h5[key].attrs.get("sfreq", RESAMPLING_FREQUENCY)
                items.append((stem, onset, sfreq))
    
    if not items:
        logger.warning("No onsets found for visualization")
        return
    
    rows, cols = grid_size(len(items))
    logger.info(f"  - Grid layout: {rows} rows x {cols} columns")
    logger.info(f"  - Number of plots: {len(items)}")
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5))
    axes_list = [axes] if len(items) == 1 else axes.ravel()
    
    for i, (name, onset, sfreq) in enumerate(items):
        ax = axes_list[i]
        n_samples_1sec = int(sfreq)
        onset_1sec = onset[:n_samples_1sec]
        time_sec = np.arange(len(onset_1sec)) / sfreq
        ax.plot(time_sec, onset_1sec)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Onset Strength")
        ax.grid(True, alpha=0.3)
    
    for j in range(len(items), rows * cols):
        axes_list[j].axis("off")
    
    fig.suptitle("Broadband Onsets (1 second)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info(f"  - Saved visualization to: {out_png}")


def main():
    logger.info("")
    logger.info("="*70)
    logger.info("STARTING: gammatone_broadband_onsets.py")
    logger.info("="*70)
    
    ensure_broadband_envelopes()
    
    wav_files = sorted(STIM_DIR.glob("*.wav"))
    wav_stems = [p.stem for p in wav_files]
    logger.info(f"Found {len(wav_files)} WAV files")
    
    if ONSETS_H5.exists() and GRID_PNG.exists():
        with h5py.File(ONSETS_H5, "r", locking=False) as h5:
            first_stem = wav_stems[0]
            if f"{first_stem}/broadband/broad" in h5:
                logger.info(f"Broadband onsets already exist in {ONSETS_H5}")
                logger.info(f"Visualization already exists: {GRID_PNG}")
                logger.info("Skipping computation")
                logger.info("="*70)
                return
    
    broad_hp = float(BANDPASS_FILTER_PARAMS["highpass"])
    broad_lp = float(BANDPASS_FILTER_PARAMS["lowpass"])
    
    logger.info(f"\nProcessing Parameters:")
    logger.info(f"  - Sampling rate: {RESAMPLING_FREQUENCY} Hz")
    logger.info(f"  - Bandpass filter: {broad_hp}-{broad_lp} Hz")
    logger.info(f"  - Onset method: Half-wave rectified first difference")
    logger.info(f"Output file: {ONSETS_H5}")
    logger.info("")
    
    with h5py.File(ENVELOPES_H5, "r", locking=False) as h5_env, h5py.File(ONSETS_H5, "w", locking=False) as h5_onset:
        for idx, wav_path in enumerate(wav_files, 1):
            stim_name = wav_path.stem
            logger.info(f"Processing stimulus {idx}/{len(wav_files)}: {stim_name}")
            
            env_key = f"{stim_name}/broadband/broad"
            if env_key not in h5_env:
                logger.warning(f"  - Envelope not found, skipping")
                continue
            
            logger.info(f"  [STEP 1] Loading broadband envelope from cache")
            envelope = h5_env[env_key][:].ravel()
            logger.info(f"    - Envelope shape: {envelope.shape}")
            logger.info(f"    - Sampling rate: {RESAMPLING_FREQUENCY} Hz")
            
            logger.info(f"  [STEP 2] Computing onset (first difference)")
            onset = onset_from_envelope(envelope)
            logger.info(f"    - Onset shape: {onset.shape}")
            
            logger.info(f"  [STEP 3] Half-wave rectification (keeping only positive changes)")
            positive_changes = np.sum(onset > 0)
            logger.info(f"    - Number of positive changes: {positive_changes}")
            
            logger.info(f"  [STEP 4] Saving to HDF5")
            onset_key = f"{stim_name}/broadband/broad"
            onset_attrs = {"sfreq": int(RESAMPLING_FREQUENCY), "bandpass_hz": (broad_hp, broad_lp), "onset_definition": "half-wave rectified first difference"}
            write_dataset(h5_onset, onset_key, onset.reshape(-1, 1), onset_attrs)
            logger.info(f"    - Saved dataset: {onset_key}")
            logger.info(f"    - Final shape: {onset.reshape(-1, 1).shape}")
            logger.info("")
    
    logger.info(f"Broadband onsets saved to {ONSETS_H5}")
    logger.info("")
    
    logger.info("Generating visualization...")
    plot_broadband_onset_timeseries(ONSETS_H5, wav_stems, GRID_PNG)
    
    logger.info("")
    logger.info("="*70)
    logger.info("COMPLETED: gammatone_broadband_onsets.py")
    logger.info("="*70)


if __name__ == "__main__":
    main()