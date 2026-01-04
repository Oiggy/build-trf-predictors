"""
BUILD_GAMMATONES.PY - Acoustic Feature Extraction Pipeline

====================================================================
HOW TO RUN
====================================================================
From project root directory:
    python predictors/build_gammatones.py

Or from predictors/ directory:
    python build_gammatones.py

====================================================================
REQUIRED DIRECTORY STRUCTURE
====================================================================
project_root/
├── analysis/
│   └── params.py                    # Must define:
│                                    #   - RESAMPLING_FREQUENCY
│                                    #   - BANDPASS_FILTER_PARAMS
│                                    #   - FREQUENCY_BANDS
├── dataset/
│   └── stimuli/
│       ├── List_1_stim_1.wav       # Audio files here
│       ├── List_1_stim_2.wav
│       └── ...
└── predictors/
    └── build_gammatones.py          # This script

====================================================================
OUTPUT STRUCTURE (CREATED BY SCRIPT)
====================================================================
predictors/
└── gammatones/
    ├── spectrograms.h5              # Cached spectrograms
    ├── grid_spectrograms.png        # Visualization
    ├── envelopes.h5                 # Final envelope features
    ├── onsets.h5                    # Final onset features
    └── grid_broadband_*.png         # Visualizations per band

====================================================================
PROCESSING PIPELINE (PER STIMULUS)
====================================================================
1. Load spectrogram from cache (or compute if missing)
2. Compute envelopes (broadband + 8 frequency bins)
3. Apply bandpass filter at spectrogram rate (1000 Hz)
4. Extract modulation bands (delta/theta/alpha/beta + broad)
5. Resample to final rate (RESAMPLING_FREQUENCY)
6. Compute onsets from envelopes
7. Save to HDF5 files

====================================================================
OUTPUT FILES - KEY STRUCTURE
====================================================================
Both envelopes.h5 and onsets.h5 use the same key hierarchy:

Broadband features:
    "<stimulus_name>/broadband/broad"    # 0.5-20 Hz
    "<stimulus_name>/broadband/delta"    # Delta band
    "<stimulus_name>/broadband/theta"    # Theta band
    "<stimulus_name>/broadband/alpha"    # Alpha band
    "<stimulus_name>/broadband/beta"     # Beta band

8-bin acoustic features:
    "<stimulus_name>/band8/80-200Hz"     # Example bin
    "<stimulus_name>/band8/200-500Hz"    # Example bin
    ... (8 bins total per stimulus)

All arrays have shape: (n_timepoints, 1)
All arrays aligned to: sfreq = RESAMPLING_FREQUENCY

====================================================================
REQUIRED PACKAGES
====================================================================
pip install numpy h5py matplotlib eelbrain gammatone mne
"""

from pathlib import Path
import sys
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt

# ====================================================================
# SETUP: Add project root to Python path
# ====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import project parameters
from analysis.params import (
    RESAMPLING_FREQUENCY,
    BANDPASS_FILTER_PARAMS,
    FREQUENCY_BANDS,
)

# Import audio/spectrogram tools
from eelbrain import load, gammatone_bank

# Import frequency labeling tool
from gammatone.filters import centre_freqs

# Import signal processing tools
import mne


# ====================================================================
# CONFIGURATION: Spectrogram Settings
# ====================================================================
F_MIN = 80              # Minimum frequency (Hz)
F_MAX = 8000            # Maximum frequency (Hz)
N_BANDS = 128           # Number of gammatone bands
TSTEP = 0.001           # Time step in seconds (1 ms)
N_BINS = 8              # Number of coarse frequency bins

# Derived: spectrogram sampling frequency
SPEC_SFREQ = int(round(1.0 / TSTEP))  # 1000 Hz


# ====================================================================
# CONFIGURATION: File Paths
# ====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = SCRIPT_DIR.parent  # OLD

# New: Point to cocktail-party directory
COCKTAIL_PARTY_DIR = Path("/Users/joshuaighalo/Github Dataset/cocktail-party")

# Input directory
STIM_DIR = COCKTAIL_PARTY_DIR / "dataset" / "stimuli"

# Output directory
OUT_DIR = COCKTAIL_PARTY_DIR / "predictors" / "gammatones"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output files
SPECTRO_H5 = OUT_DIR / "spectrograms.h5"
GRID_SGRAM_PNG = OUT_DIR / "grid_spectrograms.png"
ENVELOPES_H5 = OUT_DIR / "envelopes.h5"
ONSETS_H5 = OUT_DIR / "onsets.h5"


# ====================================================================
# UTILITY: Logging
# ====================================================================
def log(msg="", **kwargs):
    """Print message with flush for real-time output"""
    print(msg, flush=True, **kwargs)


# ====================================================================
# UTILITY: Grid Layout Calculator
# ====================================================================
def grid_size(n_plots):
    """Calculate rows and columns for subplot grid"""
    cols = int(math.ceil(math.sqrt(n_plots)))
    rows = int(math.ceil(n_plots / cols))
    return rows, cols


# ====================================================================
# UTILITY: HDF5 Group Creation
# ====================================================================
def require_group(h5_file, group_path):
    """Create nested groups in HDF5 file if they don't exist"""
    current_group = h5_file
    for part in group_path.strip("/").split("/"):
        if part:
            current_group = current_group.require_group(part)
    return current_group


# ====================================================================
# UTILITY: HDF5 Dataset Writer
# ====================================================================
def write_dataset(h5_file, key, data_2d, attrs, overwrite=True):
    """
    Write dataset to HDF5 file with attributes
    
    Args:
        h5_file: Open HDF5 file handle
        key: Full path like "stimulus/broadband/theta"
        data_2d: 2D numpy array (n_times, 1)
        attrs: Dictionary of metadata attributes
        overwrite: If True, delete existing dataset
    """
    key = key.strip("/")
    parent_path = "/".join(key.split("/")[:-1])
    dataset_name = key.split("/")[-1]

    # Get or create parent group
    parent_group = require_group(h5_file, parent_path) if parent_path else h5_file

    # Handle existing dataset
    if dataset_name in parent_group:
        if overwrite:
            del parent_group[dataset_name]
        else:
            return

    # Create dataset with compression
    dset = parent_group.create_dataset(dataset_name, data=data_2d, compression="gzip")
    
    # Add metadata attributes
    for attr_name, attr_value in attrs.items():
        dset.attrs[attr_name] = attr_value


# ====================================================================
# AUDIO PROCESSING: Get Sampling Rate
# ====================================================================
def get_wav_sr(wav_path):
    """Extract sampling rate from WAV file"""
    wav = load.wav(wav_path)
    sr = int(round(1.0 / float(wav.time.tstep)))
    return sr


# ====================================================================
# AUDIO PROCESSING: Compute Spectrogram
# ====================================================================
def compute_spectrogram(wav_path):
    """
    Compute gammatone spectrogram from WAV file
    
    Returns:
        spectrogram: 2D array (n_bands, n_frames)
        audio_sr: Original audio sampling rate
    """
    # Load audio
    wav = load.wav(wav_path)
    audio_sr = int(round(1.0 / float(wav.time.tstep)))

    # Convert stereo to mono (average channels)
    if wav.ndim == 2:
        wav = (wav.sub(channel=0) + wav.sub(channel=1)) / 2

    # Compute gammatone filterbank spectrogram
    spec = gammatone_bank(
        wav,
        f_min=F_MIN,      # Changed from fmin
        f_max=F_MAX,      # Changed from fmax
        n=N_BANDS,
        location="left",
        tstep=TSTEP,
    )

    # Extract array and ensure correct shape
    arr = np.asarray(spec.x)
    if arr.shape[0] != N_BANDS and arr.shape[1] == N_BANDS:
        arr = arr.T

    if arr.shape[0] != N_BANDS:
        raise ValueError(f"Unexpected spectrogram shape for {wav_path.name}: {arr.shape}")

    return np.asarray(arr, dtype=float), audio_sr


# ====================================================================
# VISUALIZATION: Spectrogram Grid
# ====================================================================
def save_grid_spectrograms_from_h5(h5_path, wav_stems, out_png):
    """Create grid plot of all spectrograms from HDF5 file"""
    
    # Load all spectrograms
    items = []
    with h5py.File(h5_path, "r") as h5:
        for stem in wav_stems:
            if stem in h5:
                items.append((stem, h5[stem][:]))

    if not items:
        log("[SPECTROGRAM] No spectrogram datasets found to plot.")
        return

    # Calculate grid layout
    rows, cols = grid_size(len(items))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.5))
    axes_list = [axes] if len(items) == 1 else axes.ravel()

    # Plot each spectrogram
    for i, (name, spec) in enumerate(items):
        ax = axes_list[i]
        ax.imshow(np.log1p(np.asarray(spec, dtype=float)), aspect="auto", origin="lower")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Frequency Band")

    # Hide unused subplots
    for j in range(len(items), rows * cols):
        axes_list[j].axis("off")

    # Save figure
    fig.suptitle("All Gammatone Spectrograms (log1p display)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ====================================================================
# FEATURE EXTRACTION: Envelope Computation
# ====================================================================
def compute_broadband_and_8bin_envelopes(spectrogram):
    """
    Compute envelopes from spectrogram
    
    Args:
        spectrogram: 2D array (n_bands, n_frames)
        
    Returns:
        broadband_envelope: 1D array (sum across all bands)
        band8_envelopes: List of 8 arrays (one per coarse bin)
    """
    spec = np.asarray(spectrogram, dtype=float)
    
    # Apply log transform
    log_spec = np.log1p(spec)

    # Broadband: sum all frequency bands
    broadband_envelope = log_spec.sum(axis=0)

    # 8-bin: divide bands into 8 groups and sum each
    band_splits = np.array_split(np.arange(N_BANDS), N_BINS)
    band8_envelopes = []
    for band_indices in band_splits:
        bin_envelope = log_spec[band_indices, :].sum(axis=0)
        band8_envelopes.append(bin_envelope)

    return broadband_envelope, band8_envelopes


# ====================================================================
# FEATURE EXTRACTION: Frequency Bin Labeling
# ====================================================================
def compute_8bin_labels(audio_sr):
    """
    Generate frequency range labels for 8 coarse bins
    
    Returns:
        labels: List like ["80-200Hz", "200-500Hz", ...]
    """
    # Get center frequencies for all bands
    freqs = centre_freqs(float(audio_sr), N_BANDS, float(F_MIN))
    
    # Split into 8 groups
    band_splits = np.array_split(np.arange(N_BANDS), N_BINS)

    labels = []
    for band_indices in band_splits:
        low_freq = int(round(float(freqs[band_indices].min())))
        high_freq = int(round(float(freqs[band_indices].max())))
        labels.append(f"{low_freq}-{high_freq}Hz")
    
    return labels


# ====================================================================
# SIGNAL PROCESSING: Resampling
# ====================================================================
def resample_1d(signal, sfreq_in, sfreq_out):
    """
    Resample 1D signal from one rate to another
    
    Args:
        signal: 1D array
        sfreq_in: Input sampling frequency (Hz)
        sfreq_out: Output sampling frequency (Hz)
        
    Returns:
        resampled_signal: 1D array at new rate
    """
    signal = np.asarray(signal, dtype=float).ravel()
    
    # Skip if already at target rate
    if int(sfreq_in) == int(sfreq_out):
        return signal.copy()

    # Calculate upsampling and downsampling factors
    gcd = math.gcd(int(sfreq_in), int(sfreq_out))
    up_factor = int(sfreq_out) // gcd
    down_factor = int(sfreq_in) // gcd

    # Perform resampling
    resampled = mne.filter.resample(
        signal,
        up=up_factor,
        down=down_factor,
        npad="auto",
        axis=0,
        verbose=False,
    )
    
    return np.asarray(resampled, dtype=float).ravel()


# ====================================================================
# SIGNAL PROCESSING: Bandpass Filtering
# ====================================================================
def bandpass_1d(signal, sfreq, l_freq, h_freq):
    """
    Apply bandpass filter to 1D signal
    
    Args:
        signal: 1D array
        sfreq: Sampling frequency (Hz)
        l_freq: Low cutoff frequency (Hz)
        h_freq: High cutoff frequency (Hz)
        
    Returns:
        filtered_signal: 1D array
    """
    signal = np.asarray(signal, dtype=float).ravel()
    
    filtered = mne.filter.filter_data(
        signal,
        sfreq=float(sfreq),
        l_freq=float(l_freq),
        h_freq=float(h_freq),
        verbose=False,
    )
    
    return np.asarray(filtered, dtype=float).ravel()


# ====================================================================
# FEATURE EXTRACTION: Onset Detection
# ====================================================================
def onset_from_envelope(envelope):
    """
    Compute onset strength from envelope
    Half-wave rectified first difference
    
    Args:
        envelope: 1D array
        
    Returns:
        onset: 1D array (positive changes only)
    """
    envelope = np.asarray(envelope, dtype=float).ravel()
    
    # Compute first difference
    diff = np.diff(envelope, prepend=envelope[0])
    
    # Half-wave rectification (keep only positive changes)
    diff[diff < 0] = 0.0
    
    return np.asarray(diff, dtype=float).ravel()


# ====================================================================
# VISUALIZATION: Time Series Grid
# ====================================================================
def save_grid_timeseries(items, out_png, title, ylabel):
    """
    Create grid plot of multiple time series
    
    Args:
        items: List of (name, array) tuples
        out_png: Output PNG path
        title: Figure title
        ylabel: Y-axis label
    """
    if not items:
        log(f"[GRID] No items to plot: {out_png.name}")
        return

    # Calculate grid layout
    rows, cols = grid_size(len(items))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 2.5))
    axes_list = [axes] if len(items) == 1 else axes.ravel()

    # Plot each time series
    for i, (name, timeseries) in enumerate(items):
        ax = axes_list[i]
        ax.plot(np.asarray(timeseries, dtype=float))
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Time Samples")
        ax.set_ylabel(ylabel)

    # Hide unused subplots
    for j in range(len(items), rows * cols):
        axes_list[j].axis("off")

    # Save figure
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ====================================================================
# MAIN PIPELINE
# ====================================================================
def main():
    """Main processing pipeline"""
    
    # ================================================================
    # SETUP: Print Configuration
    # ================================================================
    log("=" * 70)
    log("GAMMATONE FEATURE EXTRACTION PIPELINE")
    log("=" * 70)
    log(f"\nInput directory:  {STIM_DIR}")
    log(f"Output directory: {OUT_DIR}")
    log(f"\nSpectrogram file: {SPECTRO_H5}")
    log(f"Envelopes file:   {ENVELOPES_H5}")
    log(f"Onsets file:      {ONSETS_H5}")
    
    log(f"\n{'Gammatone Settings:':<25}")
    log(f"  Frequency range:        {F_MIN}-{F_MAX} Hz")
    log(f"  Number of bands:        {N_BANDS}")
    log(f"  Time step:              {TSTEP}s")
    log(f"  Spectrogram rate:       {SPEC_SFREQ} Hz")
    
    log(f"\n{'Processing Settings:':<25}")
    log(f"  Final sampling rate:    {RESAMPLING_FREQUENCY} Hz")
    log(f"  Broad bandpass:         {BANDPASS_FILTER_PARAMS}")
    log(f"  Modulation bands:       {FREQUENCY_BANDS}")
    log()

    # ================================================================
    # SETUP: Find Input Files
    # ================================================================
    wav_files = sorted(STIM_DIR.glob("*.wav"))
    if not wav_files:
        raise SystemExit(f"ERROR: No WAV files found in {STIM_DIR}")

    wav_stems = [p.stem for p in wav_files]
    log(f"Found {len(wav_files)} WAV files")
    log()

    # ================================================================
    # STEP 1: Spectrogram Computation/Caching
    # ================================================================
    log("=" * 70)
    log("STEP 1: SPECTROGRAM COMPUTATION")
    log("=" * 70)
    
    # Check if cache exists
    cache_exists = SPECTRO_H5.exists() and GRID_SGRAM_PNG.exists()
    
    if cache_exists:
        log("✓ Cache found - skipping spectrogram computation")
        log(f"  Using: {SPECTRO_H5}")
        log(f"  Using: {GRID_SGRAM_PNG}")
    else:
        log("✗ Cache not found - computing spectrograms")
        
        with h5py.File(SPECTRO_H5, "a") as h5_spec:
            for wav_path in wav_files:
                stim_name = wav_path.stem
                
                # Skip if already computed
                if stim_name in h5_spec:
                    # Ensure sampling rate is stored
                    if "sr" not in h5_spec[stim_name].attrs:
                        h5_spec[stim_name].attrs["sr"] = get_wav_sr(wav_path)
                    log(f"  ✓ {stim_name:30s} [cached]")
                    continue

                # Compute spectrogram
                log(f"  → {stim_name:30s} [computing...]", end=" ")
                spectrogram, audio_sr = compute_spectrogram(wav_path)
                
                # Save to HDF5
                dset = h5_spec.create_dataset(stim_name, data=spectrogram, compression="gzip")
                dset.attrs["sr"] = int(audio_sr)
                dset.attrs["F_MIN"] = float(F_MIN)
                dset.attrs["F_MAX"] = float(F_MAX)
                dset.attrs["N_BANDS"] = int(N_BANDS)
                dset.attrs["TSTEP"] = float(TSTEP)
                
                log(f"shape={spectrogram.shape}")

        # Create visualization
        log(f"\n  Creating grid visualization...")
        save_grid_spectrograms_from_h5(SPECTRO_H5, wav_stems, GRID_SGRAM_PNG)
        log(f"  ✓ Saved: {GRID_SGRAM_PNG}")
    
    log()

    # ================================================================
    # STEP 2: Prepare Band Definitions
    # ================================================================
    log("=" * 70)
    log("STEP 2: PREPARE BAND DEFINITIONS")
    log("=" * 70)
    
    # Broadband filter parameters
    broad_hp = float(BANDPASS_FILTER_PARAMS["highpass"])
    broad_lp = float(BANDPASS_FILTER_PARAMS["lowpass"])
    
    # Build dictionary of all modulation bands
    broadband_band_defs = {"broad": (broad_hp, broad_lp)}
    for band_name, (low_freq, high_freq) in FREQUENCY_BANDS.items():
        broadband_band_defs[band_name] = (float(low_freq), float(high_freq))
    
    log("Broadband modulation bands:")
    for band_name, (low_freq, high_freq) in broadband_band_defs.items():
        log(f"  {band_name:10s}: {low_freq:6.2f} - {high_freq:6.2f} Hz")
    
    # Get 8-bin frequency labels
    with h5py.File(SPECTRO_H5, "r") as h5_spec:
        first_stem = wav_stems[0]
        first_audio_sr = int(h5_spec[first_stem].attrs.get("sr", get_wav_sr(wav_files[0])))
    
    band8_labels = compute_8bin_labels(first_audio_sr)
    
    log(f"\n8 coarse frequency bins:")
    for i, label in enumerate(band8_labels, 1):
        log(f"  Bin {i}: {label}")
    
    log()

    # ================================================================
    # STEP 3: Envelope and Onset Extraction
    # ================================================================
    log("=" * 70)
    log("STEP 3: ENVELOPE AND ONSET EXTRACTION")
    log("=" * 70)
    log("Processing order: Filter → Resample → Onset")
    log()
    
    with h5py.File(SPECTRO_H5, "r") as h5_spec, \
         h5py.File(ENVELOPES_H5, "w") as h5_env, \
         h5py.File(ONSETS_H5, "w") as h5_onset:
        
        for wav_path in wav_files:
            stim_name = wav_path.stem
            
            log(f"Processing: {stim_name}")
            log("-" * 70)
            
            # --------------------------------------------------------
            # 3.1: Load Spectrogram
            # --------------------------------------------------------
            if stim_name not in h5_spec:
                log(f"  ERROR: {stim_name} not found in spectrograms.h5")
                continue
            
            spectrogram = h5_spec[stim_name][:]  # Shape: (n_bands, n_frames)
            audio_sr = int(h5_spec[stim_name].attrs.get("sr", get_wav_sr(wav_path)))
            log(f"  [1] Loaded spectrogram: {spectrogram.shape}")
            
            # --------------------------------------------------------
            # 3.2: Compute Envelopes at Spectrogram Rate
            # --------------------------------------------------------
            broadband_env, band8_envs = compute_broadband_and_8bin_envelopes(spectrogram)
            log(f"  [2] Computed envelopes at {SPEC_SFREQ} Hz")
            log(f"      - Broadband: {broadband_env.shape}")
            log(f"      - 8-bin:     {len(band8_envs)} arrays")
            
            # --------------------------------------------------------
            # 3.3: Apply Broad Bandpass Filter (at Spectrogram Rate)
            # --------------------------------------------------------
            log(f"  [3] Applying broad bandpass filter ({broad_hp}-{broad_lp} Hz)")
            broadband_env_filtered = bandpass_1d(broadband_env, SPEC_SFREQ, broad_hp, broad_lp)
            
            band8_envs_filtered = []
            for band8_env in band8_envs:
                filtered = bandpass_1d(band8_env, SPEC_SFREQ, broad_hp, broad_lp)
                band8_envs_filtered.append(filtered)
            
            # --------------------------------------------------------
            # 3.4: Extract Modulation Bands (at Spectrogram Rate)
            # --------------------------------------------------------
            log(f"  [4] Extracting modulation bands at {SPEC_SFREQ} Hz")
            broadband_bands_at_spec_rate = {}
            
            for band_name, (low_freq, high_freq) in broadband_band_defs.items():
                filtered_band = bandpass_1d(
                    broadband_env_filtered,
                    SPEC_SFREQ,
                    low_freq,
                    high_freq
                )
                broadband_bands_at_spec_rate[band_name] = filtered_band
                log(f"      - {band_name}: {low_freq}-{high_freq} Hz")
            
            # --------------------------------------------------------
            # 3.5: Resample to Final Rate
            # --------------------------------------------------------
            log(f"  [5] Resampling {SPEC_SFREQ} Hz → {RESAMPLING_FREQUENCY} Hz")
            
            # Resample broadband modulation bands
            broadband_bands_final = {}
            for band_name, signal in broadband_bands_at_spec_rate.items():
                resampled = resample_1d(signal, SPEC_SFREQ, RESAMPLING_FREQUENCY)
                broadband_bands_final[band_name] = resampled
            
            # Resample 8-bin envelopes
            band8_envs_final = []
            for band8_env in band8_envs_filtered:
                resampled = resample_1d(band8_env, SPEC_SFREQ, RESAMPLING_FREQUENCY)
                band8_envs_final.append(resampled)
            
            log(f"      - All signals now at {RESAMPLING_FREQUENCY} Hz")
            
            # --------------------------------------------------------
            # 3.6: Compute Onsets
            # --------------------------------------------------------
            log(f"  [6] Computing onsets (half-wave rectified difference)")
            
            # Compute onsets for broadband bands
            broadband_onsets_final = {}
            for band_name, envelope in broadband_bands_final.items():
                onset = onset_from_envelope(envelope)
                broadband_onsets_final[band_name] = onset
            
            # Compute onsets for 8-bin envelopes
            band8_onsets_final = []
            for band8_env in band8_envs_final:
                onset = onset_from_envelope(band8_env)
                band8_onsets_final.append(onset)
            
            # --------------------------------------------------------
            # 3.7: Save to HDF5 Files
            # --------------------------------------------------------
            log(f"  [7] Saving to HDF5 files")
            
            # Prepare common metadata
            common_attrs = {
                "sfreq": int(RESAMPLING_FREQUENCY),
                "audio_sr": int(audio_sr),
                "spec_fmin": float(F_MIN),
                "spec_fmax": float(F_MAX),
                "spec_n_bands": int(N_BANDS),
                "spec_tstep_sec": float(TSTEP),
                "spec_sfreq": int(SPEC_SFREQ),
                "resampling_hz": int(RESAMPLING_FREQUENCY),
                "broad_bandpass_hz": (broad_hp, broad_lp),
                "broadband_bands": ",".join(list(broadband_band_defs.keys())),
                "processing_order": "filter_first_then_resample",
                "onset_definition": "half-wave rectified first difference",
            }
            
            # Save broadband envelopes and onsets
            for band_name in broadband_band_defs.keys():
                # Envelope
                env_key = f"{stim_name}/broadband/{band_name}"
                env_attrs = dict(common_attrs)
                env_attrs["bandpass_hz"] = broadband_band_defs[band_name]
                env_data = broadband_bands_final[band_name].reshape(-1, 1)
                write_dataset(h5_env, env_key, env_data, env_attrs, overwrite=True)
                
                # Onset
                onset_key = f"{stim_name}/broadband/{band_name}"
                onset_attrs = dict(common_attrs)
                onset_attrs["bandpass_hz"] = broadband_band_defs[band_name]
                onset_data = broadband_onsets_final[band_name].reshape(-1, 1)
                write_dataset(h5_onset, onset_key, onset_data, onset_attrs, overwrite=True)
            
            # Save 8-bin envelopes and onsets
            for label, envelope, onset in zip(band8_labels, band8_envs_final, band8_onsets_final):
                # Envelope
                env_key = f"{stim_name}/band8/{label}"
                env_attrs = dict(common_attrs)
                env_attrs["band_label"] = label
                env_attrs["bandpass_hz"] = (broad_hp, broad_lp)
                env_data = envelope.reshape(-1, 1)
                write_dataset(h5_env, env_key, env_data, env_attrs, overwrite=True)
                
                # Onset
                onset_key = f"{stim_name}/band8/{label}"
                onset_attrs = dict(common_attrs)
                onset_attrs["band_label"] = label
                onset_attrs["bandpass_hz"] = (broad_hp, broad_lp)
                onset_data = onset.reshape(-1, 1)
                write_dataset(h5_onset, onset_key, onset_data, onset_attrs, overwrite=True)
            
            log(f"      - Broadband: {len(broadband_band_defs)} bands saved")
            log(f"      - 8-bin:     {len(band8_labels)} bins saved")
            log(f"  ✓ Completed: {stim_name}\n")

    # ================================================================
    # STEP 4: Create Visualization Grids
    # ================================================================
    log("=" * 70)
    log("STEP 4: CREATE VISUALIZATION GRIDS")
    log("=" * 70)
    
    with h5py.File(ENVELOPES_H5, "r") as h5_env, \
         h5py.File(ONSETS_H5, "r") as h5_onset:
        
        for band_name in broadband_band_defs.keys():
            log(f"Creating grids for: {band_name}")
            
            # Collect envelope data
            env_items = []
            for stim_name in wav_stems:
                env_key = f"{stim_name}/broadband/{band_name}"
                if env_key in h5_env:
                    env_data = h5_env[env_key][:].ravel()
                    env_items.append((stim_name, env_data))
            
            # Collect onset data
            onset_items = []
            for stim_name in wav_stems:
                onset_key = f"{stim_name}/broadband/{band_name}"
                if onset_key in h5_onset:
                    onset_data = h5_onset[onset_key][:].ravel()
                    onset_items.append((stim_name, onset_data))
            
            # Save envelope grid
            env_png = OUT_DIR / f"grid_broadband_envelopes_{band_name}.png"
            save_grid_timeseries(
                env_items,
                env_png,
                title=f"Broadband Envelopes ({band_name}) @ {RESAMPLING_FREQUENCY} Hz",
                ylabel="Envelope Amplitude",
            )
            log(f"  ✓ {env_png.name}")
            
            # Save onset grid
            onset_png = OUT_DIR / f"grid_broadband_onsets_{band_name}.png"
            save_grid_timeseries(
                onset_items,
                onset_png,
                title=f"Broadband Onsets ({band_name}) @ {RESAMPLING_FREQUENCY} Hz",
                ylabel="Onset Strength",
            )
            log(f"  ✓ {onset_png.name}")

    # ================================================================
    # COMPLETION SUMMARY
    # ================================================================
    log()
    log("=" * 70)
    log("PIPELINE COMPLETED SUCCESSFULLY")
    log("=" * 70)
    log(f"\nOutput files:")
    log(f"  Spectrograms:  {SPECTRO_H5}")
    log(f"  Envelopes:     {ENVELOPES_H5}")
    log(f"  Onsets:        {ONSETS_H5}")
    log(f"\nVisualization: {OUT_DIR}")
    log()


# ====================================================================
# SCRIPT ENTRY POINT
# ====================================================================
if __name__ == "__main__":
    main()