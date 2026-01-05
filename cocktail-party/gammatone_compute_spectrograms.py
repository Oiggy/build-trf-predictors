"""
GAMMATONE_COMPUTE_SPECTROGRAMS.PY - Compute and cache gammatone spectrograms

====================================================================
HOW TO RUN
====================================================================
From project root directory:
    python gammatone_compute_spectrograms.py

====================================================================
REQUIRED DIRECTORY STRUCTURE
====================================================================
~/Github Dataset/cocktail-party/
├── dataset/
│   └── stimuli/
│       ├── List_1_stim_1.wav       # Audio files here
│       ├── List_1_stim_2.wav
│       └── ...
└── predictors/
    └── gammatones/                 # Output directory (auto-created)

====================================================================
OUTPUT STRUCTURE (CREATED BY SCRIPT)
====================================================================
~/Github Dataset/cocktail-party/predictors/gammatones/
├── spectrograms.h5              # Cached spectrograms
├── grid_spectrograms.png        # Visualization
└── gammatone.log                # Processing log

====================================================================
PROCESSING PIPELINE (PER STIMULUS)
====================================================================
1. Load WAV file
2. Convert stereo to mono (if needed)
3. Compute gammatone filterbank spectrogram
4. Save to HDF5 with metadata
5. Generate visualization grid

====================================================================
OUTPUT FILES - KEY STRUCTURE
====================================================================
spectrograms.h5:
    "<stimulus_name>"                # Dataset (n_bands, n_frames)
        attrs: sr, F_MIN, F_MAX, N_BANDS, TSTEP

====================================================================
REQUIRED PACKAGES
====================================================================
pip install numpy h5py eelbrain matplotlib
"""

from pathlib import Path
import sys
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from eelbrain import load, gammatone_bank

# Add parent directory to path for logging_config import
SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

# Configuration
F_MIN = 80
F_MAX = 8000
N_BANDS = 128
TSTEP = 0.001
SPEC_SFREQ = 1000

# Path configuration
HOME = Path.home()
COCKTAIL_PARTY_DIR = HOME / "Github Dataset" / "cocktail-party"
STIM_DIR = COCKTAIL_PARTY_DIR / "dataset" / "stimuli"
OUT_DIR = COCKTAIL_PARTY_DIR / "predictors" / "gammatones"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECTRO_H5 = OUT_DIR / "spectrograms.h5"
GRID_PNG = OUT_DIR / "grid_spectrograms.png"

# Setup logging
from logging_config import setup_logging
logger = setup_logging(
    log_filename="gammatone.log",
    output_dir=OUT_DIR,
    script_name="gammatone_compute_spectrograms"
)


def get_wav_sr(wav_path):
    """Extract sampling rate from WAV file"""
    wav = load.wav(wav_path)
    return int(round(1.0 / float(wav.time.tstep)))


def compute_spectrogram(wav_path):
    """Compute gammatone spectrogram from WAV file"""
    logger.info(f"  [STEP 1] Loading audio file: {wav_path.name}")
    wav = load.wav(wav_path)
    audio_sr = int(round(1.0 / float(wav.time.tstep)))
    logger.info(f"    - Audio sampling rate: {audio_sr} Hz")
    logger.info(f"    - Audio shape: {wav.shape}")
    
    if wav.ndim == 2:
        logger.info(f"  [STEP 2] Converting stereo to mono (averaging channels)")
        wav = (wav.sub(channel=0) + wav.sub(channel=1)) / 2
    else:
        logger.info(f"  [STEP 2] Audio is already mono")
    
    logger.info(f"  [STEP 3] Computing gammatone spectrogram")
    logger.info(f"    - F_MIN: {F_MIN} Hz")
    logger.info(f"    - F_MAX: {F_MAX} Hz")
    logger.info(f"    - N_BANDS: {N_BANDS}")
    logger.info(f"    - TSTEP: {TSTEP} s")
    
    spec = gammatone_bank(wav, f_min=F_MIN, f_max=F_MAX, n=N_BANDS, location="left", tstep=TSTEP)
    
    arr = np.asarray(spec.x)
    if arr.shape[0] != N_BANDS and arr.shape[1] == N_BANDS:
        logger.info(f"    - Transposing spectrogram from {arr.shape}")
        arr = arr.T
    
    if arr.shape[0] != N_BANDS:
        raise ValueError(f"Unexpected shape: {arr.shape}")
    
    logger.info(f"    - Spectrogram shape: {arr.shape} (n_bands, n_frames)")
    logger.info(f"    - Spectrogram sampling rate: {SPEC_SFREQ} Hz")
    
    return np.asarray(arr, dtype=float), audio_sr


def grid_size(n_plots):
    """Calculate rows and columns for subplot grid"""
    cols = int(math.ceil(math.sqrt(n_plots)))
    rows = int(math.ceil(n_plots / cols))
    return rows, cols


def create_discrete_colormap(n_colors=8):
    """Create discrete colormap with n_colors"""
    colors = plt.cm.viridis(np.linspace(0, 1, n_colors))
    return ListedColormap(colors)


def plot_spectrograms(h5_path, wav_stems, out_png, n_bins=8):
    """Create grid of spectrograms"""
    logger.info(f"Creating spectrogram visualization grid")
    items = []
    with h5py.File(h5_path, "r") as h5:
        for stem in wav_stems:
            if stem in h5:
                spec = h5[stem][:]
                items.append((stem, spec))
    
    if not items:
        logger.warning("No spectrograms found for visualization")
        return
    
    rows, cols = grid_size(len(items))
    logger.info(f"  - Grid layout: {rows} rows x {cols} columns")
    logger.info(f"  - Number of spectrograms: {len(items)}")
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5))
    axes_list = [axes] if len(items) == 1 else axes.ravel()
    cmap = create_discrete_colormap(n_bins)
    
    for i, (name, spec) in enumerate(items):
        ax = axes_list[i]
        n_frames_1sec = min(SPEC_SFREQ, spec.shape[1])
        spec_1sec = spec[:, :n_frames_1sec]
        time_sec = np.arange(n_frames_1sec) / SPEC_SFREQ
        freq_bins = np.linspace(F_MIN, F_MAX, N_BANDS)
        
        im = ax.pcolormesh(time_sec, freq_bins, np.log1p(spec_1sec), cmap=cmap, shading='auto')
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Log Amplitude")
    
    for j in range(len(items), rows * cols):
        axes_list[j].axis("off")
    
    fig.suptitle("Gammatone Spectrograms (1 second)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info(f"  - Saved visualization to: {out_png}")


def main():
    logger.info("="*70)
    logger.info("STARTING: gammatone_compute_spectrograms.py")
    logger.info("="*70)
    
    wav_files = sorted(STIM_DIR.glob("*.wav"))
    if not wav_files:
        logger.error(f"No WAV files found in {STIM_DIR}")
        raise SystemExit(f"ERROR: No WAV files in {STIM_DIR}")
    
    wav_stems = [p.stem for p in wav_files]
    logger.info(f"Found {len(wav_files)} WAV files in {STIM_DIR}")
    
    if SPECTRO_H5.exists() and GRID_PNG.exists():
        with h5py.File(SPECTRO_H5, "r") as h5:
            all_exist = all(wav.stem in h5 for wav in wav_files)
            if all_exist:
                logger.info(f"All spectrograms already cached in {SPECTRO_H5}")
                logger.info(f"Visualization already exists: {GRID_PNG}")
                logger.info("Skipping computation")
                logger.info("="*70)
                return
    
    logger.info(f"\nComputing spectrograms for {len(wav_files)} files...")
    logger.info(f"Output file: {SPECTRO_H5}")
    logger.info("")
    
    with h5py.File(SPECTRO_H5, "a") as h5:
        for idx, wav_path in enumerate(wav_files, 1):
            stim_name = wav_path.stem
            logger.info(f"Processing stimulus {idx}/{len(wav_files)}: {stim_name}")
            
            if stim_name in h5:
                if "sr" not in h5[stim_name].attrs:
                    h5[stim_name].attrs["sr"] = get_wav_sr(wav_path)
                logger.info(f"  - Already cached, skipping")
                logger.info("")
                continue
            
            spec, audio_sr = compute_spectrogram(wav_path)
            
            logger.info(f"  [STEP 4] Saving to HDF5")
            dset = h5.create_dataset(stim_name, data=spec, compression="gzip")
            dset.attrs["sr"] = int(audio_sr)
            dset.attrs["F_MIN"] = float(F_MIN)
            dset.attrs["F_MAX"] = float(F_MAX)
            dset.attrs["N_BANDS"] = int(N_BANDS)
            dset.attrs["TSTEP"] = float(TSTEP)
            logger.info(f"    - Saved dataset: {stim_name}")
            logger.info(f"    - Metadata: sr={audio_sr}, F_MIN={F_MIN}, F_MAX={F_MAX}, N_BANDS={N_BANDS}")
            logger.info("")
    
    logger.info(f"Spectrograms saved to {SPECTRO_H5}")
    logger.info("")
    
    logger.info("Generating visualization...")
    plot_spectrograms(SPECTRO_H5, wav_stems, GRID_PNG)
    
    logger.info("")
    logger.info("="*70)
    logger.info("COMPLETED: gammatone_compute_spectrograms.py")
    logger.info("="*70)


if __name__ == "__main__":
    main()