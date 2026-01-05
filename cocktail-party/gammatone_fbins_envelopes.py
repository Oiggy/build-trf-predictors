"""
GAMMATONE_FBINS_ENVELOPES.PY - Compute 8-bin frequency envelopes

====================================================================
HOW TO RUN
====================================================================
From project root directory:
    python gammatone_fbins_envelopes.py

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
├── spectrograms.h5                    # Input (auto-created if missing)
├── envelopes.h5                       # Output (8-bin features added)
├── grid_8bin_envelopes.png            # Visualization (spectrogram grid)
├── gmm_env_freqs.txt                  # Frequency bin info
└── gammatone.log                      # Processing log

====================================================================
PROCESSING PIPELINE (PER STIMULUS)
====================================================================
1. Load spectrogram from cache
2. Divide 128 frequency bands into 8 coarse bins (equally spaced in ERB)
3. Sum each bin to create 8 envelopes
4. Apply broad bandpass filter at 1000 Hz
5. Resample each bin to final rate (RESAMPLING_FREQUENCY)
6. Save to HDF5
7. Generate spectrogram visualization grid

====================================================================
OUTPUT FILES - KEY STRUCTURE
====================================================================
envelopes.h5 (8-bin features added):
    "<stimulus_name>/freq_bins/100-200Hz"    # Example bin
    "<stimulus_name>/freq_bins/200-500Hz"    # Example bin
    ... (8 bins total per stimulus)

All arrays: shape (n_timepoints, 1), sfreq = RESAMPLING_FREQUENCY

Note: Bins are equally spaced in ERB space, omitting frequencies below 100 Hz

====================================================================
REQUIRED PACKAGES
====================================================================
pip install numpy h5py mne gammatone matplotlib
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
from gammatone.filters import centre_freqs

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

SPECTRO_H5 = OUT_DIR / "spectrograms.h5"
ENVELOPES_H5 = OUT_DIR / "envelopes.h5"

# Setup logging
from logging_config import setup_logging
logger = setup_logging(
    log_filename="gammatone.log",
    output_dir=OUT_DIR,
    script_name="gammatone_fbins_envelopes"
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


def hz_to_erb(hz):
    """Convert frequency in Hz to ERB scale"""
    return 21.4 * np.log10(1 + 0.00437 * hz)


def compute_8bin_envelopes(spectrogram, audio_sr):
    """Compute 8 frequency bin envelopes equally spaced in ERB"""
    spec = np.asarray(spectrogram, dtype=float)
    log_spec = np.log1p(spec)
    
    # Get center frequencies of the 128 bands
    center_freqs = centre_freqs(float(audio_sr), N_BANDS, float(F_MIN))
    
    # Convert to ERB scale
    erb_values = hz_to_erb(center_freqs)
    
    # Create 8 bins equally spaced in ERB
    erb_min = erb_values.min()
    erb_max = erb_values.max()
    erb_edges = np.linspace(erb_min, erb_max, N_BINS + 1)
    
    # Assign each band to a bin based on ERB value
    envelopes = []
    for i in range(N_BINS):
        # Find bands within this ERB range
        in_bin = (erb_values >= erb_edges[i]) & (erb_values < erb_edges[i + 1])
        if i == N_BINS - 1:  # Include the last edge for the final bin
            in_bin = (erb_values >= erb_edges[i]) & (erb_values <= erb_edges[i + 1])
        
        band_indices = np.where(in_bin)[0]
        if len(band_indices) > 0:
            bin_env = log_spec[band_indices, :].sum(axis=0)
        else:
            bin_env = np.zeros(log_spec.shape[1])
        envelopes.append(bin_env)
    
    return envelopes


def compute_8bin_labels(audio_sr):
    """Generate frequency labels for 8 bins equally spaced in ERB"""
    # Get center frequencies of the 128 bands
    center_freqs = centre_freqs(float(audio_sr), N_BANDS, float(F_MIN))
    
    # Convert to ERB scale
    erb_values = hz_to_erb(center_freqs)
    
    # Create 8 bins equally spaced in ERB
    erb_min = erb_values.min()
    erb_max = erb_values.max()
    erb_edges = np.linspace(erb_min, erb_max, N_BINS + 1)
    
    # Assign each band to a bin and get frequency ranges
    labels = []
    for i in range(N_BINS):
        in_bin = (erb_values >= erb_edges[i]) & (erb_values < erb_edges[i + 1])
        if i == N_BINS - 1:
            in_bin = (erb_values >= erb_edges[i]) & (erb_values <= erb_edges[i + 1])
        
        band_indices = np.where(in_bin)[0]
        if len(band_indices) > 0:
            low_freq = int(round(float(center_freqs[band_indices].min())))
            high_freq = int(round(float(center_freqs[band_indices].max())))
            labels.append(f"{low_freq}-{high_freq}Hz")
        else:
            labels.append(f"empty-bin-{i}")
    
    return labels


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


def plot_fbin_envelope_spectrograms(h5_path, wav_stems, bin_labels, out_png, n_bins=8):
    """Create grid of envelope spectrograms"""
    logger.info(f"Creating {n_bins}-bin envelope spectrogram visualization grid")
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
    freq_txt = OUT_DIR / "gmm_env_freqs.txt"
    with open(freq_txt, 'w') as f:
        f.write("Frequency Bin (Hz)\tCenter Frequency (Hz)\n")
        for (low, high), center in zip(freq_ranges, center_freqs):
            f.write(f"{low}-{high}\t{center:.1f}\n")
    logger.info(f"Saved frequency bin information to {freq_txt}")
    
    # Collect envelope data
    for stem in wav_stems:
        bin_envs = []
        with h5py.File(h5_path, "r", locking=False) as h5:
            for bin_label in bin_labels:
                key = f"{stem}/freq_bins/{bin_label}"
                if key in h5:
                    env = h5[key][:].ravel()
                    sfreq = h5[key].attrs.get("sfreq", RESAMPLING_FREQUENCY)
                    bin_envs.append(env)
        if len(bin_envs) == n_bins:
            items.append((stem, bin_envs, sfreq))
    
    if not items:
        logger.warning("No frequency bin envelopes found for visualization")
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
    
    for i, (name, bin_envs, sfreq) in enumerate(items):
        ax = axes_list[i]
        n_samples_1sec = int(sfreq)
        env_matrix = np.array([env[:n_samples_1sec] for env in bin_envs])
        
        time_edges = np.linspace(0, 1, n_samples_1sec + 1)
        
        im = ax.pcolormesh(time_edges, freq_edges, env_matrix, 
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
    
    fig.suptitle("Gammatone Envelope Spectrograms (1 second)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info(f"  - Saved visualization to: {out_png}")


def main():
    logger.info("")
    logger.info("="*70)
    logger.info("STARTING: gammatone_fbins_envelopes.py")
    logger.info("="*70)
    
    ensure_spectrograms()
    
    wav_files = sorted(STIM_DIR.glob("*.wav"))
    wav_stems = [p.stem for p in wav_files]
    logger.info(f"Found {len(wav_files)} WAV files")
    
    with h5py.File(SPECTRO_H5, "r", locking=False) as h5_spec:
        first_stem = wav_stems[0]
        audio_sr = int(h5_spec[first_stem].attrs.get("sr", 0))
    bin_labels = compute_8bin_labels(audio_sr)
    
    logger.info(f"\n{N_BINS} Frequency Bins (equally spaced in ERB):")
    for i, label in enumerate(bin_labels, 1):
        logger.info(f"  Bin {i}: {label}")
    
    out_png = OUT_DIR / "grid_8bin_envelopes.png"
    
    # Check if frequency bin envelopes already exist
    if ENVELOPES_H5.exists() and out_png.exists():
        try:
            with h5py.File(ENVELOPES_H5, "r", locking=False) as h5:
                first_stem = wav_stems[0]
                # Check if the stimulus exists as a group and has freq_bins subgroup
                if first_stem in h5:
                    if "freq_bins" in h5[first_stem]:
                        freq_bins_group = h5[first_stem]["freq_bins"]
                        # Check if any of the expected bin labels exist
                        if any(label.replace("Hz", "") in freq_bins_group for label in bin_labels):
                            logger.info(f"\n{N_BINS}-bin envelopes already exist in {ENVELOPES_H5}")
                            logger.info(f"Visualization already exists: {out_png}")
                            logger.info("Skipping computation")
                            logger.info("="*70)
                            return
        except Exception as e:
            logger.warning(f"Could not check existing data: {e}")
            logger.info("Proceeding with computation...")
    
    broad_hp = float(BANDPASS_FILTER_PARAMS["highpass"])
    broad_lp = float(BANDPASS_FILTER_PARAMS["lowpass"])
    
    logger.info(f"\nProcessing Parameters:")
    logger.info(f"  - F_MIN: {F_MIN} Hz (omitting frequencies below 100 Hz)")
    logger.info(f"  - F_MAX: {F_MAX} Hz")
    logger.info(f"  - Number of bins: {N_BINS} (equally spaced in ERB)")
    logger.info(f"  - Spectrogram sampling rate: {SPEC_SFREQ} Hz")
    logger.info(f"  - Final sampling rate: {RESAMPLING_FREQUENCY} Hz")
    logger.info(f"  - Bandpass filter: {broad_hp}-{broad_lp} Hz")
    logger.info(f"Output file: {ENVELOPES_H5}")
    logger.info("")
    
    with h5py.File(SPECTRO_H5, "r", locking=False) as h5_spec, h5py.File(ENVELOPES_H5, "a", locking=False) as h5_env:
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
            
            logger.info(f"  [STEP 2] Dividing {N_BANDS} bands into {N_BINS} bins (ERB-spaced)")
            bin_envs = compute_8bin_envelopes(spec, audio_sr)
            logger.info(f"    - Created {len(bin_envs)} bin envelopes")
            
            for bin_idx, (label, bin_env) in enumerate(zip(bin_labels, bin_envs), 1):
                logger.info(f"  [STEP 3.{bin_idx}] Processing bin: {label}")
                logger.info(f"    - Envelope shape at {SPEC_SFREQ} Hz: {bin_env.shape}")
                logger.info(f"    - Applying bandpass filter: {broad_hp}-{broad_lp} Hz")
                bin_env_filt = bandpass_1d(bin_env, SPEC_SFREQ, broad_hp, broad_lp)
                logger.info(f"    - Resampling {SPEC_SFREQ} Hz → {RESAMPLING_FREQUENCY} Hz")
                bin_final = resample_1d(bin_env_filt, SPEC_SFREQ, RESAMPLING_FREQUENCY)
                logger.info(f"    - Final shape: {bin_final.shape}")
                
                env_key = f"{stim_name}/freq_bins/{label}"
                env_attrs = {
                    "sfreq": int(RESAMPLING_FREQUENCY), 
                    "audio_sr": int(audio_sr), 
                    "band_label": label, 
                    "bandpass_hz": (broad_hp, broad_lp),
                    "binning": "ERB-spaced"
                }
                write_dataset(h5_env, env_key, bin_final.reshape(-1, 1), env_attrs)
                logger.info(f"    - Saved dataset: {env_key}")
            logger.info("")
    
    logger.info(f"{N_BINS}-bin envelopes saved to {ENVELOPES_H5}")
    logger.info("")
    
    logger.info("Generating visualization...")
    plot_fbin_envelope_spectrograms(ENVELOPES_H5, wav_stems, bin_labels, out_png, N_BINS)
    
    logger.info("")
    logger.info("="*70)
    logger.info("COMPLETED: gammatone_fbins_envelopes.py")
    logger.info("="*70)


if __name__ == "__main__":
    main()