"""
gammatone_fbins_overt_onsets.py - Compute 8-band overt onsets and save to onsets_overt.h5

====================================================================
WHAT THIS SCRIPT DOES
====================================================================
Computes overt onset predictors per frequency bin using the formula:

    Oovert = Osource - Omasked  ≡  min(Osource, Omixture)

Where:
    Osource  = standard onset of the source (attend or ignore) stimulus
    Omixture = standard onset of the mixture stimulus paired with that source

Overt onsets capture the portion of the source onset that is NOT masked
by the mixture — i.e., source onsets that are also visible in the mixture.
They are the complement of masked onsets:
    Omasked + Oovert = Osource

====================================================================
DEPENDENCY
====================================================================
Requires onsets.h5 to already exist (produced by gammatone_fbins_onsets.py).
Source and mixture onsets are read directly from onsets.h5 per frequency bin.
If onsets.h5 is missing, this script will run gammatone_fbins_onsets.py first.

====================================================================
SOURCE-MIXTURE PAIRINGS
====================================================================
Pairings are read from params.py (same values used by the encoding scripts):
    Trigger 1:  SPEAKER_1[i] paired with MIX_1[i]  (one pair per trial)
    Trigger 2:  SPEAKER_2[i] paired with MIX_2[i]  (one pair per trial)

SPEAKER_1, SPEAKER_2, MIX_1, MIX_2 are lists of stimulus names in params.py.
Each unique (source, mixture) pair is processed once — duplicates are skipped.

These fixed pairings match the experimental design — each source stimulus
always appears in the same mixture across all listening conditions
(binaural, dichotic, diotic). The single condition has no mixture and is
therefore excluded (overt masking does not apply when heard alone).

====================================================================
OUTPUT FILES - KEY STRUCTURE
====================================================================
onsets_overt.h5:
    "<source_stim_name>/freq_bins/<band_label>"   (8 bins per source)

Key structure is identical to onsets.h5, so onsets_overt.h5 can be used
as a drop-in replacement in the encoding model scripts.

All arrays: shape (n_timepoints, 1), sfreq = RESAMPLING_FREQUENCY

====================================================================
USAGE
====================================================================
python gammatone_fbins_overt_onsets.py

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

# ====================================================================
# PATH SETUP
# ====================================================================
SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

HOME = Path.home()
PROJECT_ROOT = HOME / "Github Repositories" / "trf-cocktail-party"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from params import RESAMPLING_FREQUENCY, SPEAKER_1, SPEAKER_2, MIX_1, MIX_2

# ====================================================================
# CONFIGURATION
# ====================================================================
F_MIN  = 100
F_MAX  = 8000
N_BINS = 8

# Path configuration
COCKTAIL_PARTY_DIR = HOME / "Github Dataset" / "cocktail-party"
STIM_DIR           = COCKTAIL_PARTY_DIR / "dataset" / "stimuli"
OUT_DIR            = COCKTAIL_PARTY_DIR / "predictors" / "gammatones"

ONSETS_H5       = OUT_DIR / "onsets.h5"
ONSETS_OVERT_H5 = OUT_DIR / "onsets_overt.h5"

# Build unique (source, mixture) pairs from both trigger lists
_all_pairs = (
    [(src, mix) for src, mix in zip(SPEAKER_1, MIX_1)]
    + [(src, mix) for src, mix in zip(SPEAKER_2, MIX_2)]
)
_seen = set()
SOURCE_MIXTURE_PAIRS = []
for pair in _all_pairs:
    if pair not in _seen:
        _seen.add(pair)
        SOURCE_MIXTURE_PAIRS.append(pair)

# ====================================================================
# LOGGING SETUP
# ====================================================================
from logging_config import setup_logging
logger = setup_logging(
    log_filename="gammatone.log",
    output_dir=OUT_DIR,
    script_name="gammatone_fbins_overt_onsets"
)


# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def ensure_fbins_onsets():
    """Run gammatone_fbins_onsets.py if onsets.h5 does not exist or lacks freq_bins."""
    if not ONSETS_H5.exists():
        logger.info("onsets.h5 not found. Running gammatone_fbins_onsets.py...")
        script_path = SCRIPT_DIR / "gammatone_fbins_onsets.py"
        subprocess.run([sys.executable, str(script_path)], check=True)
        return
    with h5py.File(ONSETS_H5, "r", locking=False) as h5:
        has_freq_bins = any("freq_bins" in str(k) for k in h5.keys())
        if not has_freq_bins:
            logger.info("freq_bins not found in onsets.h5. Running gammatone_fbins_onsets.py...")
            script_path = SCRIPT_DIR / "gammatone_fbins_onsets.py"
            subprocess.run([sys.executable, str(script_path)], check=True)


def require_group(h5_file, group_path):
    """Create nested groups in HDF5 file if they do not already exist."""
    current_group = h5_file
    for part in group_path.strip("/").split("/"):
        if part:
            current_group = current_group.require_group(part)
    return current_group


def write_dataset(h5_file, key, data_2d, attrs):
    """Write a dataset to HDF5, overwriting if it already exists."""
    key         = key.strip("/")
    parent_path = "/".join(key.split("/")[:-1])
    dset_name   = key.split("/")[-1]
    parent_grp  = require_group(h5_file, parent_path) if parent_path else h5_file
    if dset_name in parent_grp:
        del parent_grp[dset_name]
    dset = parent_grp.create_dataset(dset_name, data=data_2d, compression="gzip")
    for attr_name, attr_val in attrs.items():
        dset.attrs[attr_name] = attr_val


def overt_onset_from_onsets(source_onset, mix_onset):
    """
    Compute overt onset using:
        Oovert = Osource - Omasked  ≡  min(Osource, Omixture)

    Parameters
    ----------
    source_onset : 1D np.ndarray
        Half-wave rectified onset of the source stimulus.
    mix_onset : 1D np.ndarray
        Half-wave rectified onset of the mixture stimulus.

    Returns
    -------
    overt : 1D np.ndarray
        Overt onset (same length as inputs).
    """
    source = np.asarray(source_onset, dtype=float).ravel()
    mix    = np.asarray(mix_onset,    dtype=float).ravel()

    if source.shape != mix.shape:
        raise ValueError(
            f"Shape mismatch: source {source.shape} vs mixture {mix.shape}. "
            f"Source and mixture stimuli must have the same duration."
        )

    return np.minimum(source, mix)


def get_bin_labels_from_h5(h5_path, stim_name):
    """Extract sorted frequency bin labels from onsets.h5 for a given stimulus."""
    labels = []
    with h5py.File(h5_path, "r", locking=False) as h5:
        if stim_name in h5 and "freq_bins" in h5[stim_name]:
            for key in h5[stim_name]["freq_bins"].keys():
                if key not in labels:
                    labels.append(key)
    return sorted(labels)


def load_onset_bin(h5_file, stim_name, bin_label):
    """
    Load a single frequency-bin onset array from an open HDF5 file.
    Returns a 1D float array.
    """
    key = f"{stim_name}/freq_bins/{bin_label}"
    if key not in h5_file:
        raise KeyError(f"Key not found in onsets.h5: {key}")
    return h5_file[key][:].ravel().astype(float)


def grid_size(n_plots):
    """Return (rows, cols) for a near-square subplot grid."""
    cols = int(math.ceil(math.sqrt(n_plots)))
    rows = int(math.ceil(n_plots / cols))
    return rows, cols


# ====================================================================
# VISUALIZATION
# ====================================================================

def plot_overt_onset_spectrograms(h5_path, source_names, bin_labels, out_png):
    """
    Generate a spectrogram grid showing 1 second of overt onsets
    for each source stimulus across all frequency bins.
    """
    logger.info("  Generating overt onset spectrogram grid...")

    freq_edges = [F_MIN]
    for label in bin_labels:
        try:
            high = float(label.replace("Hz", "").split("-")[1])
            freq_edges.append(high)
        except Exception:
            freq_edges.append(freq_edges[-1])
    freq_edges = np.array(freq_edges, dtype=float)

    items = []
    with h5py.File(h5_path, "r", locking=False) as h5:
        for src in source_names:
            if src not in h5 or "freq_bins" not in h5[src]:
                continue
            bin_onsets = []
            for label in bin_labels:
                key = f"{src}/freq_bins/{label}"
                if key in h5:
                    bin_onsets.append(h5[key][:].ravel())
                else:
                    bin_onsets.append(np.array([]))
            items.append((src, bin_onsets, RESAMPLING_FREQUENCY))

    if not items:
        logger.warning("  No data found for visualization. Skipping.")
        return

    rows, cols = grid_size(len(items))
    fig, axes  = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes_list  = [axes] if len(items) == 1 else np.array(axes).ravel()

    for i, (name, bin_onsets, sfreq) in enumerate(items):
        ax             = axes_list[i]
        n_samples_1sec = int(sfreq)
        onset_matrix   = np.array([
            onset[:n_samples_1sec] if len(onset) >= n_samples_1sec
            else np.pad(onset, (0, n_samples_1sec - len(onset)))
            for onset in bin_onsets
        ])

        time_edges = np.linspace(0, 1, n_samples_1sec + 1)
        im = ax.pcolormesh(time_edges, freq_edges, onset_matrix,
                           cmap="viridis", shading="flat")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

        y_ticks = np.linspace(F_MIN, F_MAX, 6)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{int(yt)}" for yt in y_ticks])
        ax.set_ylim(F_MIN, F_MAX)
        plt.colorbar(im, ax=ax, label="Amplitude")

    for j in range(len(items), rows * cols):
        axes_list[j].axis("off")

    fig.suptitle(
        "Gammatone Overt Onset Spectrograms — Oovert = min(Osource, Omixture) — 1 second",
        fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved visualization: {out_png}")


# ====================================================================
# MAIN
# ====================================================================

def main():
    logger.info("")
    logger.info("=" * 70)
    logger.info("STARTING: gammatone_fbins_overt_onsets.py")
    logger.info("=" * 70)
    logger.info("Formula: Oovert = Osource - Omasked  ≡  min(Osource, Omixture)")
    logger.info(f"Unique source-mixture pairs: {len(SOURCE_MIXTURE_PAIRS)}")
    for src, mix in SOURCE_MIXTURE_PAIRS:
        logger.info(f"  source={src}  mixture={mix}")
    logger.info("")

    # ── Step 1: Ensure standard onsets are available ──────────────────────
    ensure_fbins_onsets()

    # ── Step 2: Get frequency bin labels from the first source stimulus ───
    first_source = SOURCE_MIXTURE_PAIRS[0][0]
    bin_labels   = get_bin_labels_from_h5(ONSETS_H5, first_source)

    if not bin_labels:
        raise RuntimeError(
            f"No freq_bins found for stimulus '{first_source}' in {ONSETS_H5}. "
            f"Check that gammatone_fbins_onsets.py has been run successfully."
        )

    logger.info(f"{N_BINS} frequency bins found:")
    for i, label in enumerate(bin_labels, 1):
        logger.info(f"  Bin {i}: {label}")
    logger.info("")

    out_png = OUT_DIR / "grid_8bin_overt_onsets.png"

    # ── Step 3: Check if already complete ────────────────────────────────
    if ONSETS_OVERT_H5.exists() and out_png.exists():
        try:
            with h5py.File(ONSETS_OVERT_H5, "r", locking=False) as h5:
                all_done = all(
                    any(f"{src}/freq_bins/{label}" in h5 for label in bin_labels)
                    for src, _ in SOURCE_MIXTURE_PAIRS
                )
            if all_done:
                logger.info("Overt onsets already exist in onsets_overt.h5")
                logger.info("Visualization already exists.")
                logger.info("Skipping computation.")
                logger.info("=" * 70)
                return
        except Exception as e:
            logger.warning(f"Could not check existing data: {e}. Proceeding.")

    # ── Step 4: Compute and save overt onsets ─────────────────────────────
    logger.info("Processing Parameters:")
    logger.info(f"  Sampling rate: {RESAMPLING_FREQUENCY} Hz")
    logger.info(f"  N bins:        {N_BINS}")
    logger.info(f"  Formula:       Oovert = min(Osource, Omixture)")
    logger.info(f"  Output file:   {ONSETS_OVERT_H5}")
    logger.info("")

    source_names_processed = []

    with h5py.File(ONSETS_H5, "r", locking=False) as h5_src, \
         h5py.File(ONSETS_OVERT_H5, "a", locking=False) as h5_out:

        for pair_idx, (source_name, mix_name) in enumerate(SOURCE_MIXTURE_PAIRS, 1):
            logger.info(f"Pair {pair_idx}/{len(SOURCE_MIXTURE_PAIRS)}: "
                        f"source='{source_name}'  mixture='{mix_name}'")

            if source_name not in h5_src or "freq_bins" not in h5_src[source_name]:
                logger.warning(f"  source '{source_name}' not found in onsets.h5 — skipping")
                continue

            if mix_name not in h5_src or "freq_bins" not in h5_src[mix_name]:
                logger.warning(f"  mixture '{mix_name}' not found in onsets.h5 — skipping")
                continue

            for bin_idx, bin_label in enumerate(bin_labels, 1):
                logger.info(f"  [Bin {bin_idx}/{N_BINS}] {bin_label}")

                try:
                    source_onset = load_onset_bin(h5_src, source_name, bin_label)
                except KeyError as e:
                    logger.warning(f"    Source onset not found: {e} — skipping bin")
                    continue

                try:
                    mix_onset = load_onset_bin(h5_src, mix_name, bin_label)
                except KeyError as e:
                    logger.warning(f"    Mixture onset not found: {e} — skipping bin")
                    continue

                logger.info(f"    source onset shape:  {source_onset.shape}")
                logger.info(f"    mixture onset shape: {mix_onset.shape}")

                try:
                    overt = overt_onset_from_onsets(source_onset, mix_onset)
                except ValueError as e:
                    logger.warning(f"    {e} — skipping bin")
                    continue

                n_nonzero     = int(np.sum(overt > 0))
                total_source  = int(np.sum(source_onset > 0))
                pct_overt     = (n_nonzero / total_source * 100) if total_source > 0 else 0.0

                logger.info(f"    overt onset shape:       {overt.shape}")
                logger.info(f"    source onsets (nonzero): {total_source}")
                logger.info(f"    overt onsets  (nonzero): {n_nonzero}  "
                            f"({pct_overt:.1f}% of source visible in mixture)")

                out_key = f"{source_name}/freq_bins/{bin_label}"
                attrs   = {
                    "sfreq":            int(RESAMPLING_FREQUENCY),
                    "band_label":       bin_label,
                    "onset_definition": "overt onset: min(Osource, Omixture)",
                    "source_stim":      source_name,
                    "mixture_stim":     mix_name,
                    "binning":          "ERB-spaced",
                }
                write_dataset(h5_out, out_key, overt.reshape(-1, 1), attrs)
                logger.info(f"    Saved: {out_key}")

            logger.info(f"  Done: source='{source_name}'")
            logger.info("")
            source_names_processed.append(source_name)

    logger.info(f"Overt onsets saved to {ONSETS_OVERT_H5}")
    logger.info(f"Sources written: {source_names_processed}")
    logger.info("")

    # ── Step 5: Visualization ─────────────────────────────────────────────
    logger.info("Generating visualization...")
    plot_overt_onset_spectrograms(ONSETS_OVERT_H5, source_names_processed, bin_labels, out_png)

    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPLETED: gammatone_fbins_overt_onsets.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()