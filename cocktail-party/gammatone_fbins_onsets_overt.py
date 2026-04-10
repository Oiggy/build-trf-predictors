#!/usr/bin/env python3
"""
gammatone_fbins_onsets_overt.py - Compute 8-band overt onsets

Follows Brodbeck et al. (2020, PLOS Biology):

    O_overt = min(O_source, O_mixture)

Equivalently: O_overt = O_source - O_masked
              where    O_masked = max(O_source - O_mixture, 0)

Overt onsets are the portion of a source speaker's onsets that survive
in the mixture — i.e., source onsets that are also visible in the
combined two-speaker signal.

    O_masked + O_overt = O_source

Only applies to two-speaker conditions (not single speech).

====================================================================
DEPENDENCY
====================================================================
Requires onsets.h5 (produced by gammatone_fbins_onsets.py).
If missing, this script runs gammatone_fbins_onsets.py automatically.

====================================================================
SOURCE-MIXTURE PAIRINGS
====================================================================
From params.py:
    Trigger 1:  SPEAKER_1[i] paired with MIX_1[i]
    Trigger 2:  SPEAKER_2[i] paired with MIX_2[i]

Each unique (source, mixture) pair is processed once.

====================================================================
OUTPUT
====================================================================
onsets_overt.h5:
    "<source_stim_name>/freq_bins/<band_label>"   (8 bins per source)

Same structure as onsets.h5 — drop-in replacement in encoding scripts.

====================================================================
USAGE
====================================================================
    python gammatone_fbins_onsets_overt.py
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
# LOGGING
# ====================================================================
from logging_config import setup_logging
logger = setup_logging(
    log_filename="gammatone.log",
    output_dir=OUT_DIR,
    script_name="gammatone_fbins_onsets_overt"
)


# ====================================================================
# HELPERS
# ====================================================================
def ensure_fbins_onsets():
    """Run gammatone_fbins_onsets.py if onsets.h5 is missing or lacks freq_bins."""
    if not ONSETS_H5.exists():
        logger.info("onsets.h5 not found. Running gammatone_fbins_onsets.py...")
        script_path = SCRIPT_DIR / "gammatone_fbins_onsets.py"
        subprocess.run([sys.executable, str(script_path)], check=True)
        return
    with h5py.File(ONSETS_H5, "r", locking=False) as h5:
        has_freq_bins = any("freq_bins" in (h5[k].keys() if isinstance(h5[k], h5py.Group) else [])
                           for k in h5.keys() if isinstance(h5[k], h5py.Group))
        if not has_freq_bins:
            logger.info("onsets.h5 lacks freq_bins. Running gammatone_fbins_onsets.py...")
            script_path = SCRIPT_DIR / "gammatone_fbins_onsets.py"
            subprocess.run([sys.executable, str(script_path)], check=True)


def require_group(h5_file, group_path):
    """Create nested HDF5 groups."""
    current = h5_file
    for part in group_path.strip("/").split("/"):
        if part:
            current = current.require_group(part)
    return current


def write_dataset(h5_file, key, data_2d, attrs):
    """Write dataset with attributes, overwriting if exists."""
    key = key.strip("/")
    parent_path = "/".join(key.split("/")[:-1])
    dataset_name = key.split("/")[-1]
    parent = require_group(h5_file, parent_path) if parent_path else h5_file
    if dataset_name in parent:
        del parent[dataset_name]
    dset = parent.create_dataset(dataset_name, data=data_2d, compression="gzip")
    for k, v in attrs.items():
        dset.attrs[k] = v


def get_bin_labels(h5_path, stim_name):
    """Get sorted list of frequency bin labels from an H5 file."""
    with h5py.File(h5_path, "r", locking=False) as h5:
        if stim_name in h5 and "freq_bins" in h5[stim_name]:
            return sorted(h5[stim_name]["freq_bins"].keys())
    return []


def load_onset_bin(h5, stim_name, bin_label):
    """Load a single frequency bin onset array from an open H5 file."""
    key = f"{stim_name}/freq_bins/{bin_label}"
    if key not in h5:
        return None
    data = h5[key][:]
    return np.asarray(data, dtype=float).ravel()


def grid_size(n):
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


# ====================================================================
# MAIN
# ====================================================================
def main():
    logger.info("=" * 70)
    logger.info("gammatone_fbins_onsets_overt.py")
    logger.info("  O_overt = min(O_source, O_mixture)")
    logger.info("=" * 70)

    ensure_fbins_onsets()

    # Discover bin labels from the first source stimulus
    first_source = SOURCE_MIXTURE_PAIRS[0][0]
    bin_labels = get_bin_labels(ONSETS_H5, first_source)
    if not bin_labels:
        raise RuntimeError(f"No freq_bins found for '{first_source}' in {ONSETS_H5}")

    logger.info(f"Source-mixture pairs: {len(SOURCE_MIXTURE_PAIRS)}")
    logger.info(f"Frequency bins:       {len(bin_labels)}")
    logger.info(f"Bin labels:           {bin_labels}")
    logger.info(f"Output:               {ONSETS_OVERT_H5}")
    logger.info("")

    processed = []
    skipped = []

    with h5py.File(ONSETS_H5, "r", locking=False) as h5_in, \
         h5py.File(ONSETS_OVERT_H5, "w") as h5_out:

        for idx, (source, mixture) in enumerate(SOURCE_MIXTURE_PAIRS, 1):
            logger.info(f"[{idx}/{len(SOURCE_MIXTURE_PAIRS)}] source={source}  mixture={mixture}")

            # Check both exist in onsets.h5
            if source not in h5_in or "freq_bins" not in h5_in[source]:
                logger.warning(f"  SKIP: source '{source}' not found in onsets.h5")
                skipped.append(source)
                continue
            if mixture not in h5_in or "freq_bins" not in h5_in[mixture]:
                logger.warning(f"  SKIP: mixture '{mixture}' not found in onsets.h5")
                skipped.append(source)
                continue

            for bin_label in bin_labels:
                o_source = load_onset_bin(h5_in, source, bin_label)
                o_mixture = load_onset_bin(h5_in, mixture, bin_label)

                if o_source is None or o_mixture is None:
                    logger.warning(f"  SKIP bin '{bin_label}': missing data")
                    continue

                # Align lengths (truncate to shorter)
                min_len = min(len(o_source), len(o_mixture))
                o_source = o_source[:min_len]
                o_mixture = o_mixture[:min_len]

                # O_overt = min(O_source, O_mixture)
                o_overt = np.minimum(o_source, o_mixture)

                out_key = f"{source}/freq_bins/{bin_label}"
                write_dataset(h5_out, out_key, o_overt.reshape(-1, 1), {
                    "sfreq": RESAMPLING_FREQUENCY,
                    "source": source,
                    "mixture": mixture,
                    "formula": "min(O_source, O_mixture)",
                    "bin_label": bin_label,
                })

            processed.append(source)
            logger.info(f"  -> wrote {len(bin_labels)} bins for '{source}'")

    logger.info(f"\nProcessed: {len(processed)}  Skipped: {len(skipped)}")
    logger.info(f"Output:    {ONSETS_OVERT_H5}")

    # ── Visualization grid (spectrogram style, first 1 second) ────────
    out_png = OUT_DIR / "grid_fbins_onsets_overt.png"
    VIZ_DURATION_S = 1.0  # show first N seconds

    if len(processed) > 0:
        n_stim = len(processed)
        rows, cols = grid_size(n_stim)
        fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.5 * rows))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.ravel()

        # Parse bin labels into center frequencies for y-axis
        def _bin_center_hz(label):
            """Extract center frequency from a bin label like '100-310'."""
            parts = label.replace("Hz", "").split("-")
            try:
                lo, hi = float(parts[0]), float(parts[1])
                return (lo + hi) / 2.0
            except Exception:
                return 0.0

        bin_centers = [_bin_center_hz(bl) for bl in bin_labels]
        n_show = int(VIZ_DURATION_S * RESAMPLING_FREQUENCY)

        with h5py.File(ONSETS_OVERT_H5, "r", locking=False) as h5:
            for i, stim in enumerate(processed):
                ax = axes[i]
                spec_rows = []
                for bl in bin_labels:
                    key = f"{stim}/freq_bins/{bl}"
                    if key in h5:
                        d = h5[key][:].ravel()[:n_show]
                    else:
                        d = np.zeros(n_show)
                    spec_rows.append(d)

                spec = np.array(spec_rows)  # (n_bins, n_time)
                t_extent = n_show / RESAMPLING_FREQUENCY
                ax.imshow(spec, aspect="auto", origin="lower",
                          extent=[0, t_extent, 0, len(bin_labels)],
                          cmap="viridis", interpolation="nearest")
                ax.set_yticks(np.arange(len(bin_labels)) + 0.5)
                ax.set_yticklabels([f"{int(f)}" for f in bin_centers], fontsize=5)
                ax.set_ylabel("Frequency (Hz)", fontsize=6)
                ax.set_xlabel("Time (s)", fontsize=7)
                ax.set_title(stim, fontsize=8)
                ax.tick_params(labelsize=5)

        for j in range(len(processed), len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Overt Onset Spectrograms ({VIZ_DURATION_S:.0f} second)", fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        logger.info(f"Saved visualization: {out_png}")

    logger.info("=" * 70)
    logger.info("Done.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()