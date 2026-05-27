#!/usr/bin/env python3
"""
validate_masked_overt_onsets_v2.py

Validation for corrected overt/masked onset files with structure:

    <mixture>/<source>/freq_bins/<band>

This version treats the mathematical equations as the pass/fail criterion:

    masked = max(source - mixture, 0)
    overt  = min(source, mixture)
    overt + masked = source_aligned_to_mixture_length

It does NOT fail on inequality diagnostics such as overt > source or masked > source,
because those assumptions only hold when all source/mixture onset values are strictly
nonnegative. Some predictor pipelines can contain tiny negative or near-zero values.

Run:
PYTHONPATH="/Users/joshuaighalo/Github Repositories/trf-cocktail-party" \
python validate_masked_overt_onsets_v2.py --conditions diotic

PYTHONPATH="/Users/joshuaighalo/Github Repositories/trf-cocktail-party" \
python validate_masked_overt_onsets_v2.py --conditions diotic binaural dichotic
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


FBIN_BANDS = [
    "100-310",
    "328-684",
    "715-1317",
    "1369-2390",
    "2478-4207",
    "4356-7285",
    "7537-12498",
    "12924-21328",
]


def import_params():
    try:
        import params as params_mod
        return params_mod
    except Exception as exc:
        raise RuntimeError(
            "Could not import params.py. Use PYTHONPATH to point to the repo containing params.py."
        ) from exc


def read_orig_band(h5, stim, band):
    for key in (f"{stim}/freq_bins/{band}", f"{stim}/freq_bins/{band}Hz"):
        if key in h5:
            return np.asarray(h5[key][:], dtype=float).ravel(), key
    raise KeyError(f"Missing original onset key for stim={stim!r}, band={band!r}")


def read_pair_band(h5, mixture, source, band):
    for key in (f"{mixture}/{source}/freq_bins/{band}", f"{mixture}/{source}/freq_bins/{band}Hz"):
        if key in h5:
            return np.asarray(h5[key][:], dtype=float).ravel(), key
    raise KeyError(f"Missing pair key for mixture={mixture!r}, source={source!r}, band={band!r}")


def align_source_to_mixture(source, mixture):
    source = np.asarray(source, dtype=float).ravel()
    mixture = np.asarray(mixture, dtype=float).ravel()
    if source.size == mixture.size:
        return source.copy(), "same_length"
    if source.size < mixture.size:
        out = np.zeros_like(mixture)
        out[:source.size] = source
        return out, f"source_padded_{mixture.size - source.size}"
    return source[:mixture.size].copy(), f"source_truncated_{source.size - mixture.size}"


def make_pairs(params_mod, selected_conditions):
    rows = []
    for i, cond in enumerate(params_mod.SPATIAL):
        if cond not in selected_conditions:
            continue

        s1 = params_mod.SPEAKER_1[i]
        s2 = params_mod.SPEAKER_2[i]
        m1 = params_mod.MIX_1[i]
        m2 = params_mod.MIX_2[i]

        rows += [
            {"condition": cond, "trial_index_1based": i + 1, "trigger_label": "trigger1", "source": s1, "mixture": m1, "role": "attend"},
            {"condition": cond, "trial_index_1based": i + 1, "trigger_label": "trigger1", "source": s2, "mixture": m1, "role": "ignore"},
            {"condition": cond, "trial_index_1based": i + 1, "trigger_label": "trigger2", "source": s2, "mixture": m2, "role": "attend"},
            {"condition": cond, "trial_index_1based": i + 1, "trigger_label": "trigger2", "source": s1, "mixture": m2, "role": "ignore"},
        ]

    seen = set()
    unique = []
    for row in rows:
        key = (row["condition"], row["trial_index_1based"], row["trigger_label"], row["source"], row["mixture"])
        if key not in seen:
            unique.append(row)
            seen.add(key)
    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Validate corrected overt/masked onset predictors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--conditions", nargs="+", default=["diotic"], choices=["diotic", "binaural", "dichotic"])
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--out-csv", type=str, default=None)
    args = parser.parse_args()

    params_mod = import_params()

    onsets_path = Path(getattr(params_mod, "ONSETS_FBINS_PATH", None) or getattr(params_mod, "ONSETS_PATH"))
    masked_path = Path(getattr(params_mod, "ONSETS_MASKED_FBINS_PATH"))
    overt_path = Path(getattr(params_mod, "ONSETS_OVERT_FBINS_PATH"))

    print(f"[CONFIG] original onsets: {onsets_path}")
    print(f"[CONFIG] masked onsets:   {masked_path}")
    print(f"[CONFIG] overt onsets:    {overt_path}")
    print(f"[CONFIG] conditions:      {', '.join(args.conditions)}")
    print(f"[CONFIG] tolerance:       {args.tolerance:g}")

    for path in (onsets_path, masked_path, overt_path):
        if not path.exists():
            raise FileNotFoundError(path)

    pairs = make_pairs(params_mod, set(args.conditions))
    print(f"[CONFIG] source-mixture pairs: {len(pairs)}")

    rows = []

    with h5py.File(onsets_path, "r") as h_on, h5py.File(masked_path, "r") as h_mask, h5py.File(overt_path, "r") as h_ov:
        for pair in pairs:
            for band in FBIN_BANDS:
                row = {**pair, "band": band, "status": "ok"}

                try:
                    source, source_key = read_orig_band(h_on, pair["source"], band)
                    mixture, mixture_key = read_orig_band(h_on, pair["mixture"], band)
                    masked, masked_key = read_pair_band(h_mask, pair["mixture"], pair["source"], band)
                    overt, overt_key = read_pair_band(h_ov, pair["mixture"], pair["source"], band)

                    source_aligned, align_status = align_source_to_mixture(source, mixture)
                    expected_masked = np.maximum(source_aligned - mixture, 0.0)
                    expected_overt = np.minimum(source_aligned, mixture)

                    row.update({
                        "source_n": source.size,
                        "mixture_n": mixture.size,
                        "masked_n": masked.size,
                        "overt_n": overt.size,
                        "align_status": align_status,
                        "source_key": source_key,
                        "mixture_key": mixture_key,
                        "masked_key": masked_key,
                        "overt_key": overt_key,
                    })

                    if masked.size != mixture.size or overt.size != mixture.size:
                        row["status"] = "shape_mismatch"
                    else:
                        row["max_abs_error_masked"] = float(np.max(np.abs(masked - expected_masked))) if mixture.size else 0.0
                        row["max_abs_error_overt"] = float(np.max(np.abs(overt - expected_overt))) if mixture.size else 0.0
                        row["max_abs_error_reconstruction"] = float(np.max(np.abs((masked + overt) - source_aligned))) if mixture.size else 0.0

                        # Diagnostics only; not pass/fail criteria.
                        row["source_min"] = float(np.min(source_aligned)) if source_aligned.size else np.nan
                        row["mixture_min"] = float(np.min(mixture)) if mixture.size else np.nan
                        row["masked_min"] = float(np.min(masked)) if masked.size else np.nan
                        row["overt_min"] = float(np.min(overt)) if overt.size else np.nan

                        if (
                            row["max_abs_error_masked"] > args.tolerance
                            or row["max_abs_error_overt"] > args.tolerance
                            or row["max_abs_error_reconstruction"] > args.tolerance
                        ):
                            row["status"] = "failed_equation_check"

                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = repr(exc)

                rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv) if args.out_csv else Path.cwd() / f"validate_masked_overt_onsets_v2_{'_'.join(args.conditions)}.csv"
    df.to_csv(out_csv, index=False)

    failed = df[df["status"] != "ok"]

    print("\n[SUMMARY]")
    print(f"rows checked:        {len(df)}")
    print(f"ok rows:             {int((df['status'] == 'ok').sum())}")
    print(f"failed rows:         {len(failed)}")
    print(f"shape mismatches:    {int((df['status'] == 'shape_mismatch').sum())}")
    print(f"errors:              {int((df['status'] == 'error').sum())}")

    for col in ["max_abs_error_masked", "max_abs_error_overt", "max_abs_error_reconstruction"]:
        vals = pd.to_numeric(df.get(col), errors="coerce")
        print(f"{col}: {float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else np.nan:.6g}")

    print(f"\n[DONE] wrote report: {out_csv}")

    if len(failed):
        print("\n[FAILED SAMPLE]")
        cols = ["condition", "trial_index_1based", "trigger_label", "source", "mixture", "band", "status", "error"]
        print(failed[[c for c in cols if c in failed.columns]].head(20).to_string(index=False))
        sys.exit(1)

    print("\n[PASS] masked/overt predictors match the equations within tolerance.")


if __name__ == "__main__":
    main()
