#!/usr/bin/env python3
"""
gammatone_fbins_onsets_masked.py

Create pair-specific masked onset predictors for two-speaker cocktail-party
conditions.

This script implements the Brodbeck et al. masked/overt onset decomposition
for EVERY source-versus-mixture pair needed by the EEG/MEG models:

    source in {SPEAKER_1[i], SPEAKER_2[i]}
    mixture in {MIX_1[i], MIX_2[i]}
    condition in {diotic, binaural, dichotic}

The output key is pair-specific:

    <mixture_stim>/<source_stim>/freq_bins/<band>

This is intentional. Masked/overt onsets are not properties of a source alone;
they are properties of a source relative to the physical mixture in which that
source was presented.

Equation implemented here:
    masked = max(source_onset - mixture_onset, 0)

Length handling:
    All output arrays are aligned to the mixture length. If the source array is
    shorter than the mixture, it is zero-padded. If it is longer than the
    mixture, it is truncated. This matches the interpretation that the
    predictor is a time series relative to the presented mixture trial.

Typical use from the project root:
    python gammatone_fbins_onsets_masked.py --conditions diotic binaural dichotic --overwrite

Safer first run:
    python gammatone_fbins_onsets_masked.py --conditions diotic --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR

# Allow the script to run either from project root or from a scripts/predictors
# subfolder copied into the project.
for candidate in [THIS_DIR, THIS_DIR.parent, THIS_DIR.parent.parent]:
    if (candidate / "params.py").exists():
        PROJECT_ROOT = candidate
        break

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import params as params_mod
    from params import (
        MIX_1,
        MIX_2,
        SPEAKER_1,
        SPEAKER_2,
        SPATIAL,
    )
except Exception as exc:
    raise RuntimeError(
        "Could not import params.py. Run this script from the project root, "
        "or place it somewhere under the project root."
    ) from exc


MODE = "masked"  # "masked" or "overt"
SCRIPT_PREFIX = "gammatone_fbins_onsets_masked"

FBIN_BANDS: List[str] = [
    "100-310",
    "328-684",
    "715-1317",
    "1369-2390",
    "2478-4207",
    "4356-7285",
    "7537-12498",
    "12924-21328",
]

TWO_SPEAKER_CONDITIONS = ("diotic", "binaural", "dichotic")


@dataclass(frozen=True)
class PairSpec:
    trial_index_1based: int
    condition: str
    trigger_label: str
    source: str
    mixture: str
    source_role_if_trigger_1: str
    source_role_if_trigger_2: str


def _default_input_path() -> Path:
    path = (
        getattr(params_mod, "ONSETS_FBINS_PATH", None)
        or getattr(params_mod, "ONSETS_PATH", None)
    )
    if not path:
        raise ValueError("params.py must define ONSETS_FBINS_PATH or ONSETS_PATH")
    return Path(path)


def _default_output_path() -> Path:
    if MODE == "masked":
        path = getattr(params_mod, "ONSETS_MASKED_FBINS_PATH", None)
        if path:
            return Path(path)
        return _default_input_path().with_name("onsets_masked.h5")

    if MODE == "overt":
        path = getattr(params_mod, "ONSETS_OVERT_FBINS_PATH", None)
        if path:
            return Path(path)
        return _default_input_path().with_name("onsets_overt.h5")

    raise ValueError(f"Unknown MODE={MODE!r}")


def _normalise_conditions(values: Sequence[str]) -> Tuple[str, ...]:
    values = tuple(str(v).strip().lower() for v in values if str(v).strip())
    bad = sorted(set(values).difference(TWO_SPEAKER_CONDITIONS))
    if bad:
        raise ValueError(
            f"Unsupported condition(s): {bad}. "
            f"Allowed: {TWO_SPEAKER_CONDITIONS}"
        )
    return values or TWO_SPEAKER_CONDITIONS


def build_pair_specs(conditions: Sequence[str]) -> List[PairSpec]:
    """Return all source-mixture pairs needed for the requested conditions.

    For each two-speaker trial we generate four source-mixture combinations:

        SPEAKER_1 vs MIX_1
        SPEAKER_2 vs MIX_1
        SPEAKER_1 vs MIX_2
        SPEAKER_2 vs MIX_2

    This covers both possible attention-trigger assignments while preserving
    the fact that masked/overt predictors are source+mixture-specific.
    """
    conditions = _normalise_conditions(conditions)
    specs: List[PairSpec] = []

    n = min(len(SPATIAL), len(SPEAKER_1), len(SPEAKER_2), len(MIX_1), len(MIX_2))
    for i in range(n):
        condition = str(SPATIAL[i]).strip().lower()
        if condition not in conditions:
            continue

        # Trigger 1 model assignment:
        #   attend=SPEAKER_1, ignore=SPEAKER_2, mix=MIX_1
        specs.append(PairSpec(i + 1, condition, "trigger1", SPEAKER_1[i], MIX_1[i], "attend", "ignore"))
        specs.append(PairSpec(i + 1, condition, "trigger1", SPEAKER_2[i], MIX_1[i], "ignore", "attend"))

        # Trigger 2 model assignment:
        #   attend=SPEAKER_2, ignore=SPEAKER_1, mix=MIX_2
        specs.append(PairSpec(i + 1, condition, "trigger2", SPEAKER_2[i], MIX_2[i], "attend", "ignore"))
        specs.append(PairSpec(i + 1, condition, "trigger2", SPEAKER_1[i], MIX_2[i], "ignore", "attend"))

    # De-duplicate in case params contain repeated pair definitions.
    seen = set()
    unique_specs: List[PairSpec] = []
    for spec in specs:
        key = (spec.condition, spec.mixture, spec.source)
        if key in seen:
            continue
        seen.add(key)
        unique_specs.append(spec)

    return unique_specs


def _candidate_band_keys(stim: str, band: str) -> List[str]:
    return [
        f"{stim}/freq_bins/{band}",
        f"{stim}/freq_bins/{band}Hz",
    ]


def read_band(h5: h5py.File, stim: str, band: str) -> np.ndarray:
    for key in _candidate_band_keys(stim, band):
        if key in h5:
            return np.asarray(h5[key][:], dtype=float).ravel()
    tried = ", ".join(_candidate_band_keys(stim, band))
    raise KeyError(f"Missing onset band for stimulus={stim!r}, band={band!r}; tried {tried}")


def align_source_to_mixture(source: np.ndarray, mixture: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return arrays with mixture length.

    The output predictor is defined in the time frame of the mixture trial.
    Source is therefore padded/truncated to mixture length.
    """
    source = np.asarray(source, dtype=float).ravel()
    mixture = np.asarray(mixture, dtype=float).ravel()

    n_src = source.size
    n_mix = mixture.size

    if n_src == n_mix:
        return source, mixture, "equal"

    if n_src < n_mix:
        aligned_source = np.zeros(n_mix, dtype=float)
        aligned_source[:n_src] = source
        return aligned_source, mixture, f"source_padded_{n_mix - n_src}"

    # n_src > n_mix
    return source[:n_mix], mixture, f"source_truncated_{n_src - n_mix}"


def compute_predictor(source: np.ndarray, mixture: np.ndarray) -> np.ndarray:
    if MODE == "masked":
        return np.maximum(source - mixture, 0.0)
    if MODE == "overt":
        return np.minimum(source, mixture)
    raise ValueError(f"Unknown MODE={MODE!r}")


def write_array(output_h5: h5py.File, mixture: str, source: str, band: str, values: np.ndarray) -> str:
    key = f"{mixture}/{source}/freq_bins/{band}"
    if key in output_h5:
        del output_h5[key]
    output_h5.create_dataset(
        key,
        data=np.asarray(values, dtype=np.float32),
        compression="gzip",
        compression_opts=4,
    )
    return key


def write_manifest_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate(
    input_path: Path,
    output_path: Path,
    conditions: Sequence[str],
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    conditions = _normalise_conditions(conditions)
    specs = build_pair_specs(conditions)

    print(f"[CONFIG] mode:       {MODE}")
    print(f"[CONFIG] input:      {input_path}")
    print(f"[CONFIG] output:     {output_path}")
    print(f"[CONFIG] conditions: {', '.join(conditions)}")
    print(f"[CONFIG] pairs:      {len(specs)} source-mixture pairs")

    if not input_path.exists():
        raise FileNotFoundError(f"Input onset H5 not found: {input_path}")

    if output_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            f"Use --overwrite to replace it, or --output to choose another path."
        )

    if dry_run:
        for spec in specs:
            print(
                f"[DRY] {spec.condition:8s} trial={spec.trial_index_1based:02d} "
                f"{spec.trigger_label:8s} source={spec.source:12s} mix={spec.mixture}"
            )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and overwrite:
        output_path.unlink()

    manifest_rows: List[Dict[str, object]] = []
    max_abs_value = 0.0
    total_nonzero = 0
    total_samples = 0

    with h5py.File(input_path, "r") as input_h5, h5py.File(output_path, "w") as output_h5:
        output_h5.attrs["mode"] = MODE
        output_h5.attrs["equation"] = "masked=max(source-mixture,0)"
        output_h5.attrs["key_format"] = "<mixture_stim>/<source_stim>/freq_bins/<band>"
        output_h5.attrs["conditions"] = ",".join(conditions)
        output_h5.attrs["source_file"] = str(input_path)
        output_h5.attrs["length_policy"] = "source is padded/truncated to mixture length"

        for spec in specs:
            for band in FBIN_BANDS:
                src = read_band(input_h5, spec.source, band)
                mix = read_band(input_h5, spec.mixture, band)
                src_aligned, mix_aligned, length_status = align_source_to_mixture(src, mix)

                values = compute_predictor(src_aligned, mix_aligned)
                values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

                key = write_array(output_h5, spec.mixture, spec.source, band, values)

                max_abs_value = max(max_abs_value, float(np.max(np.abs(values))) if values.size else 0.0)
                nonzero = int(np.count_nonzero(values))
                total_nonzero += nonzero
                total_samples += int(values.size)

                manifest_rows.append({
                    "mode": MODE,
                    "condition": spec.condition,
                    "trial_index_1based": spec.trial_index_1based,
                    "trigger_label": spec.trigger_label,
                    "source": spec.source,
                    "mixture": spec.mixture,
                    "band": band,
                    "h5_key": key,
                    "source_n": int(src.size),
                    "mixture_n": int(mix.size),
                    "output_n": int(values.size),
                    "length_status": length_status,
                    "nonzero": nonzero,
                    "max_value": float(np.max(values)) if values.size else 0.0,
                    "sum_value": float(np.sum(values)) if values.size else 0.0,
                })

    manifest_path = output_path.with_suffix(".manifest.csv")
    write_manifest_csv(manifest_path, manifest_rows)

    density = (total_nonzero / total_samples) if total_samples else 0.0
    print(f"[DONE] wrote:        {output_path}")
    print(f"[DONE] manifest:     {manifest_path}")
    print(f"[DONE] samples:      {total_samples}")
    print(f"[DONE] nonzero frac: {density:.6f}")
    print(f"[DONE] max abs val:  {max_abs_value:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Generate pair-specific {MODE} onset predictors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_default_input_path(),
        help="Input 8-band onset H5 file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output_path(),
        help="Output H5 file.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(TWO_SPEAKER_CONDITIONS),
        choices=list(TWO_SPEAKER_CONDITIONS),
        help="Two-speaker listening conditions to generate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print source-mixture pairs without writing output.",
    )
    args = parser.parse_args()

    generate(
        input_path=args.input,
        output_path=args.output,
        conditions=args.conditions,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
