"""
GAMMATONE_FBINS_ONSETS.PY - Compute onsets from 8-bin frequency envelopes

====================================================================
HOW TO RUN
====================================================================
From project root directory:
    python gammatone_fbins_onsets.py

Note: Automatically calls gammatone_fbins_envelopes.py if needed

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
├── envelopes.h5                       # Input (auto-created if missing)
├── onsets.h5                          # Output (8-bin features added)
├── grid_8bin_onsets.png               # Visualization (spectrogram grid)
├── gmm_ons_freqs.txt                  # Frequency bin info
└── gammatone.log                      # Processing log

====================================================================
PROCESSING PIPELINE (PER STIMULUS)
====================================================================
1. Load 8-bin envelopes from cache
2. For each bin: compute first difference
3. Half-wave rectify (keep only positive changes)
4. Save to HDF5
5. Generate spectrogram visualization grid

====================================================================
OUTPUT FILES - KEY STRUCTURE
====================================================================
onsets.h5 (8-bin features added):
    "<stimulus_name>/freq_bins/100-200Hz"    # Example bin
    "<stimulus_name>/freq_bins/200-500Hz"    # Example bin
    ... (8 bins total per stimulus)

All arrays: shape (n_timepoints, 1), sfreq = RESAMPLING_FREQUENCY

Note: Bins are equally spaced in ERB space, omitting frequencies below 100 Hz

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

from analysis.params import RESAMPLING_FREQUENCY

# Configuration
F_MIN = 100  # Omitting frequencies below 100 Hz
F_MAX = 8000
N_BANDS = 128
N_BINS = 8
SPEC_SFREQ = 1000

# Path configuration
COCKTAIL_PARTY_DIR = HOME / "Github Dataset" / "cocktail-party"
STIM_DIR = COCKTAIL_PARTY_DIR / "dataset" / "stimuli"
OUT_DIR = COCKTAIL_PARTY_DIR / "predictors" / "gammatones"

ENVELOPES_H5 = OUT_DIR / "envelopes.h5"
ONSETS_H5 = OUT_DIR / "onsets.h5"

# Setup logging
from logging_config import setup_logging
logger = setup_logging(
    log_filename="gammatone.log",
    output_dir=OUT_DIR,
    script_name="gammatone_fbins_onsets"
)


def ensure_fbins_envelopes():
    """Run envelope script if frequency bin envelopes don't exist"""
    if not ENVELOPES_H5.exists():
        logger.info("Frequency bin envelopes not found. Running gammatone_fbins_envelopes.py...")
        script_path = Path(__file__).parent / "gammatone_fbins_envelopes.py"
        subprocess.run([sys.executable, str(script_path)], check=True)
        return
    with h5py.File(ENVELOPES_H5, "r", locking=False) as h5:
        has_freq_bins = any("freq_bins" in key for key in h5.keys())
        if not has_freq_bins:
            logger.info("Frequency bin envelopes not found. Running gammatone_fbins_envelopes.py...")
            script_path = Path(__file__).parent / "gammatone_fbins_envelopes.py"
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


def get_bin_labels_from_h5(h5_path, first_stem):
    """Extract bin labels from HDF5 file"""
    labels = []
    with h5py.File(h5_path, "r", locking=False) as h5:
        if first_stem in h5 and "freq_bins" in h5[first_stem]:
            freq_bins_group = h5[first_stem]["freq_bins"]
            for key in freq_bins_group.keys():
                if key not in labels:
                    labels.append(key)
    return sorted(labels)


def plot_fbin_onset_spectrograms(h5_path, wav_stems, bin_labels, out_png, n_bins=8):
    """Create grid of onset spectrograms"""
    logger.info(f"Creating {n_bins}-bin onset spectrogram visualization grid")
    items = []
    
    # Parse frequency ranges and calculate centers
    freq_ranges = []
    center_freqs = []
    for label in bin_labels:
        freq_str = label.replace("Hz", "")
        low, high = map(int, freq_str.split("-"))
        freq_ranges.append((low, high))
        center_freqs.append((low + high) / 2)
    
    # Save frequency bin information to text file
    freq_txt = OUT_DIR / "gmm_ons_freqs.txt"
    with open(freq_txt, 'w') as f:
        f.write("Frequency Bin (Hz)\tCenter Frequency (Hz)\n")
        for (low, high), center in zip(freq_ranges, center_freqs):
            f.write(f"{low}-{high}\t{center:.1f}\n")
    logger.info(f"Saved frequency bin information to {freq_txt}")
    
    # Collect onset data
    for stem in wav_stems:
        bin_onsets = []
        with h5py.File(h5_path, "r", locking=False) as h5:
            for bin_label in bin_labels:
                key = f"{stem}/freq_bins/{bin_label}"
                if key in h5:
                    onset = h5[key][:].ravel()
                    sfreq = h5[key].attrs.get("sfreq", RESAMPLING_FREQUENCY)
                    bin_onsets.append(onset)
        if len(bin_onsets) == n_bins:
            items.append((stem, bin_onsets, sfreq))
    
    if not items:
        logger.warning("No frequency bin onsets found for visualization")
        return
    
    # Create frequency edges
    freq_edges = [freq_ranges[0][0]]
    for low, high in freq_ranges:
        freq_edges.append(high)
    freq_edges = np.array(freq_edges)
    
    rows, cols = grid_size(len(items))
    logger.info(f"  - Grid layout: {rows} rows x {cols} columns")
    logger.info(f"  - Number of spectrograms: {len(items)}")
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes_list = [axes] if len(items) == 1 else axes.ravel()
    
    for i, (name, bin_onsets, sfreq) in enumerate(items):
        ax = axes_list[i]
        n_samples_1sec = int(sfreq)
        onset_matrix = np.array([onset[:n_samples_1sec] for onset in bin_onsets])
        
        time_edges = np.linspace(0, 1, n_samples_1sec + 1)
        
        im = ax.pcolormesh(time_edges, freq_edges, onset_matrix, 
                           cmap='viridis', shading='flat')
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        
        # Set y-axis with 6 major tick marks from F_MIN to F_MAX
        y_ticks = np.linspace(F_MIN, F_MAX, 6)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{int(yt)}" for yt in y_ticks])
        ax.set_ylim(F_MIN, F_MAX)
        
        plt.colorbar(im, ax=ax, label='Amplitude')
    
    for j in range(len(items), rows * cols):
        axes_list[j].axis("off")
    
    fig.suptitle("Gammatone Onset Spectrograms (1 second)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info(f"  - Saved visualization to: {out_png}")


def main():
    logger.info("")
    logger.info("="*70)
    logger.info("STARTING: gammatone_fbins_onsets.py")
    logger.info("="*70)
    
    ensure_fbins_envelopes()
    
    wav_files = sorted(STIM_DIR.glob("*.wav"))
    wav_stems = [p.stem for p in wav_files]
    logger.info(f"Found {len(wav_files)} WAV files")
    
    first_stem = wav_stems[0]
    bin_labels = get_bin_labels_from_h5(ENVELOPES_H5, first_stem)
    
    logger.info(f"\n{N_BINS} Frequency Bins (equally spaced in ERB):")
    for i, label in enumerate(bin_labels, 1):
        logger.info(f"  Bin {i}: {label}")
    
    out_png = OUT_DIR / "grid_8bin_onsets.png"
    
    # Check if frequency bin onsets already exist
    if ONSETS_H5.exists() and out_png.exists():
        try:
            with h5py.File(ONSETS_H5, "r", locking=False) as h5:
                first_stem = wav_stems[0]
                # Check if the stimulus exists as a group and has freq_bins subgroup
                if first_stem in h5:
                    if "freq_bins" in h5[first_stem]:
                        freq_bins_group = h5[first_stem]["freq_bins"]
                        # Check if any of the expected bin labels exist
                        if any(label.replace("Hz", "") in freq_bins_group for label in bin_labels):
                            logger.info(f"\n{N_BINS}-bin onsets already exist in {ONSETS_H5}")
                            logger.info(f"Visualization already exists: {out_png}")
                            logger.info("Skipping computation")
                            logger.info("="*70)
                            return
        except Exception as e:
            logger.warning(f"Could not check existing data: {e}")
            logger.info("Proceeding with computation...")
    
    logger.info(f"\nProcessing Parameters:")
    logger.info(f"  - F_MIN: {F_MIN} Hz (omitting frequencies below 100 Hz)")
    logger.info(f"  - F_MAX: {F_MAX} Hz")
    logger.info(f"  - Number of bins: {N_BINS} (equally spaced in ERB)")
    logger.info(f"  - Sampling rate: {RESAMPLING_FREQUENCY} Hz")
    logger.info(f"  - Onset method: Half-wave rectified first difference")
    logger.info(f"Output file: {ONSETS_H5}")
    logger.info("")
    
    with h5py.File(ENVELOPES_H5, "r", locking=False) as h5_env, h5py.File(ONSETS_H5, "a", locking=False) as h5_onset:
        for idx, wav_path in enumerate(wav_files, 1):
            stim_name = wav_path.stem
            logger.info(f"Processing stimulus {idx}/{len(wav_files)}: {stim_name}")
            
            # Check if freq_bins group exists for this stimulus
            if stim_name not in h5_env or "freq_bins" not in h5_env[stim_name]:
                logger.warning(f"  - No frequency bin envelopes found, skipping")
                continue
            
            freq_bins_group = h5_env[stim_name]["freq_bins"]
            bin_keys = list(freq_bins_group.keys())
            
            logger.info(f"  - Found {len(bin_keys)} frequency bin envelopes")
            
            for bin_idx, bin_label in enumerate(sorted(bin_keys), 1):
                env_key = f"{stim_name}/freq_bins/{bin_label}"
                envelope = h5_env[env_key][:].ravel()
                
                logger.info(f"  [STEP {bin_idx}] Processing bin: {bin_label}")
                logger.info(f"    - Envelope shape: {envelope.shape}")
                
                onset = onset_from_envelope(envelope)
                positive_changes = np.sum(onset > 0)
                logger.info(f"    - Onset shape: {onset.shape}")
                logger.info(f"    - Positive changes: {positive_changes}")
                
                onset_key = f"{stim_name}/freq_bins/{bin_label}"
                onset_attrs = {
                    "sfreq": int(RESAMPLING_FREQUENCY), 
                    "band_label": bin_label, 
                    "onset_definition": "half-wave rectified first difference",
                    "binning": "ERB-spaced"
                }
                write_dataset(h5_onset, onset_key, onset.reshape(-1, 1), onset_attrs)
                logger.info(f"    - Saved dataset: {onset_key}")
            logger.info("")
    
    logger.info(f"{N_BINS}-bin onsets saved to {ONSETS_H5}")
    logger.info("")
    
    logger.info("Generating visualization...")
    plot_fbin_onset_spectrograms(ONSETS_H5, wav_stems, bin_labels, out_png, N_BINS)
    
    logger.info("")
    logger.info("="*70)
    logger.info("COMPLETED: gammatone_fbins_onsets.py")
    logger.info("="*70)


if __name__ == "__main__":
    main()