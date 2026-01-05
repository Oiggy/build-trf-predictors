"""
GAMMATONE_BROADBAND_ENVELOPES.PY - Compute broadband envelopes

====================================================================
HOW TO RUN
====================================================================
From project root directory:
    python gammatone_broadband_envelopes.py

Note: Automatically calls gammatone_compute_spectrograms.py if needed

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
├── spectrograms.h5              # Input (auto-created if missing)
├── envelopes.h5                 # Output (broadband only)
├── grid_broadband_envelopes.png # Visualization
└── gammatone.log                # Processing log

====================================================================
PROCESSING PIPELINE (PER STIMULUS)
====================================================================
1. Load spectrogram from cache
2. Compute broadband envelope (sum all frequency bands)
3. Apply broad bandpass filter at 1000 Hz
4. Resample to final rate (RESAMPLING_FREQUENCY)
5. Save to HDF5
6. Generate visualization grid

====================================================================
OUTPUT FILES - KEY STRUCTURE
====================================================================
envelopes.h5 (broadband features only):
    "<stimulus_name>/broadband/broad"    # Broadband envelope

All arrays: shape (n_timepoints, 1), sfreq = RESAMPLING_FREQUENCY

====================================================================
REQUIRED PACKAGES
====================================================================
pip install numpy h5py mne matplotlib
"""

from pathlib import Path
import sys
import subprocess
import math
import numpy as np
import h5py
import mne
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

# Configuration
N_BANDS = 128
SPEC_SFREQ = 1000

# Path configuration
COCKTAIL_PARTY_DIR = HOME / "Github Dataset" / "cocktail-party"
STIM_DIR = COCKTAIL_PARTY_DIR / "dataset" / "stimuli"
OUT_DIR = COCKTAIL_PARTY_DIR / "predictors" / "gammatones"

SPECTRO_H5 = OUT_DIR / "spectrograms.h5"
ENVELOPES_H5 = OUT_DIR / "envelopes.h5"
GRID_PNG = OUT_DIR / "grid_broadband_envelopes.png"

# Setup logging
from logging_config import setup_logging
logger = setup_logging(
    log_filename="gammatone.log",
    output_dir=OUT_DIR,
    script_name="gammatone_broadband_envelopes"
)


def ensure_spectrograms():
    """Run spectrogram script if spectrograms don't exist"""
    if not SPECTRO_H5.exists():
        logger.info("Spectrograms not found. Running gammatone_compute_spectrograms.py...")
        script_path = Path(__file__).parent / "gammatone_compute_spectrograms.py"
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


def compute_broadband_envelope(spectrogram):
    """Compute broadband envelope from spectrogram"""
    spec = np.asarray(spectrogram, dtype=float)
    log_spec = np.log1p(spec)
    return log_spec.sum(axis=0)


def bandpass_1d(signal, sfreq, l_freq, h_freq):
    """Apply bandpass filter to 1D signal"""
    signal = np.asarray(signal, dtype=float).ravel()
    filtered = mne.filter.filter_data(signal, sfreq=float(sfreq), l_freq=float(l_freq), h_freq=float(h_freq), verbose=False)
    return np.asarray(filtered, dtype=float).ravel()


def resample_1d(signal, sfreq_in, sfreq_out):
    """Resample 1D signal"""
    signal = np.asarray(signal, dtype=float).ravel()
    if int(sfreq_in) == int(sfreq_out):
        return signal.copy()
    gcd = math.gcd(int(sfreq_in), int(sfreq_out))
    up_factor = int(sfreq_out) // gcd
    down_factor = int(sfreq_in) // gcd
    resampled = mne.filter.resample(signal, up=up_factor, down=down_factor, npad="auto", axis=0, verbose=False)
    return np.asarray(resampled, dtype=float).ravel()


def grid_size(n_plots):
    """Calculate rows and columns for subplot grid"""
    cols = int(math.ceil(math.sqrt(n_plots)))
    rows = int(math.ceil(n_plots / cols))
    return rows, cols


def plot_broadband_envelope_timeseries(h5_path, wav_stems, out_png):
    """Create grid of broadband envelope time series"""
    logger.info(f"Creating broadband envelope visualization grid")
    items = []
    with h5py.File(h5_path, "r", locking=False) as h5:
        for stem in wav_stems:
            key = f"{stem}/broadband/broad"
            if key in h5:
                env = h5[key][:].ravel()
                sfreq = h5[key].attrs.get("sfreq", RESAMPLING_FREQUENCY)
                items.append((stem, env, sfreq))
    
    if not items:
        logger.warning("No envelopes found for visualization")
        return
    
    rows, cols = grid_size(len(items))
    logger.info(f"  - Grid layout: {rows} rows x {cols} columns")
    logger.info(f"  - Number of plots: {len(items)}")
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5))
    axes_list = [axes] if len(items) == 1 else axes.ravel()
    
    for i, (name, env, sfreq) in enumerate(items):
        ax = axes_list[i]
        n_samples_1sec = int(sfreq)
        env_1sec = env[:n_samples_1sec]
        time_sec = np.arange(len(env_1sec)) / sfreq
        ax.plot(time_sec, env_1sec)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Envelope Amplitude")
        ax.grid(True, alpha=0.3)
    
    for j in range(len(items), rows * cols):
        axes_list[j].axis("off")
    
    fig.suptitle("Broadband Envelopes (1 second)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info(f"  - Saved visualization to: {out_png}")


def main():
    logger.info("")
    logger.info("="*70)
    logger.info("STARTING: gammatone_broadband_envelopes.py")
    logger.info("="*70)
    
    ensure_spectrograms()
    
    wav_files = sorted(STIM_DIR.glob("*.wav"))
    wav_stems = [p.stem for p in wav_files]
    logger.info(f"Found {len(wav_files)} WAV files")
    
    if ENVELOPES_H5.exists() and GRID_PNG.exists():
        with h5py.File(ENVELOPES_H5, "r", locking=False) as h5:
            first_stem = wav_stems[0]
            if f"{first_stem}/broadband/broad" in h5:
                logger.info(f"Broadband envelopes already exist in {ENVELOPES_H5}")
                logger.info(f"Visualization already exists: {GRID_PNG}")
                logger.info("Skipping computation")
                logger.info("="*70)
                return
    
    broad_hp = float(BANDPASS_FILTER_PARAMS["highpass"])
    broad_lp = float(BANDPASS_FILTER_PARAMS["lowpass"])
    
    logger.info(f"\nProcessing Parameters:")
    logger.info(f"  - Spectrogram sampling rate: {SPEC_SFREQ} Hz")
    logger.info(f"  - Final sampling rate: {RESAMPLING_FREQUENCY} Hz")
    logger.info(f"  - Bandpass filter: {broad_hp}-{broad_lp} Hz")
    logger.info(f"Output file: {ENVELOPES_H5}")
    logger.info("")
    
    with h5py.File(SPECTRO_H5, "r", locking=False) as h5_spec, h5py.File(ENVELOPES_H5, "w", locking=False) as h5_env:
        for idx, wav_path in enumerate(wav_files, 1):
            stim_name = wav_path.stem
            logger.info(f"Processing stimulus {idx}/{len(wav_files)}: {stim_name}")
            
            if stim_name not in h5_spec:
                logger.warning(f"  - Spectrogram not found, skipping")
                continue
            
            logger.info(f"  [STEP 1] Loading spectrogram from cache")
            spec = h5_spec[stim_name][:]
            audio_sr = int(h5_spec[stim_name].attrs.get("sr", 0))
            logger.info(f"    - Spectrogram shape: {spec.shape}")
            logger.info(f"    - Audio sampling rate: {audio_sr} Hz")
            
            logger.info(f"  [STEP 2] Computing broadband envelope (sum across {N_BANDS} frequency bands)")
            bb_env = compute_broadband_envelope(spec)
            logger.info(f"    - Envelope shape: {bb_env.shape}")
            logger.info(f"    - Envelope at {SPEC_SFREQ} Hz")
            
            logger.info(f"  [STEP 3] Applying bandpass filter at {SPEC_SFREQ} Hz")
            logger.info(f"    - Filter range: {broad_hp}-{broad_lp} Hz")
            bb_env_filt = bandpass_1d(bb_env, SPEC_SFREQ, broad_hp, broad_lp)
            logger.info(f"    - Filtered envelope shape: {bb_env_filt.shape}")
            
            logger.info(f"  [STEP 4] Resampling {SPEC_SFREQ} Hz → {RESAMPLING_FREQUENCY} Hz")
            bb_env_final = resample_1d(bb_env_filt, SPEC_SFREQ, RESAMPLING_FREQUENCY)
            logger.info(f"    - Resampled envelope shape: {bb_env_final.shape}")
            
            logger.info(f"  [STEP 5] Saving to HDF5")
            env_key = f"{stim_name}/broadband/broad"
            env_attrs = {"sfreq": int(RESAMPLING_FREQUENCY), "audio_sr": int(audio_sr), "bandpass_hz": (broad_hp, broad_lp)}
            write_dataset(h5_env, env_key, bb_env_final.reshape(-1, 1), env_attrs)
            logger.info(f"    - Saved dataset: {env_key}")
            logger.info(f"    - Final shape: {bb_env_final.reshape(-1, 1).shape}")
            logger.info("")
    
    logger.info(f"Broadband envelopes saved to {ENVELOPES_H5}")
    logger.info("")
    
    logger.info("Generating visualization...")
    plot_broadband_envelope_timeseries(ENVELOPES_H5, wav_stems, GRID_PNG)
    
    logger.info("")
    logger.info("="*70)
    logger.info("COMPLETED: gammatone_broadband_envelopes.py")
    logger.info("="*70)


if __name__ == "__main__":
    main()