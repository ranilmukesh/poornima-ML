#!/usr/bin/env python3
"""
describe_csv.py

Reads a CSV and prints a JSON object with:
- columns: list of {name, dtype_inferred_numeric, samples: [up to N]}
- describe: per-column stats (numeric: count, mean, std, min, 25%,50%,75%,max; non-numeric: count, unique_approx, top_approx, freq_approx)

Usage (Windows cmd):
python describe_csv.py "d:\\poornima sukumar mam files\\nmbfinalDiabetes (4).csv" --samples 3

This script is streaming-friendly and uses reservoir sampling for percentiles on large files.
Requires: pandas, numpy
"""

import argparse
import json
import math
import random
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def to_serializable(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    try:
        if pd.isna(o):
            return None
    except Exception:
        pass
    return o


class Welford:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = None
        self.max = None

    def add(self, x):
        if x is None:
            return
        self.n += 1
        if self.min is None or x < self.min:
            self.min = x
        if self.max is None or x > self.max:
            self.max = x
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def summary(self):
        if self.n == 0:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        var = self.M2 / (self.n - 1) if self.n > 1 else 0.0
        return {"count": self.n, "mean": self.mean, "std": math.sqrt(var), "min": self.min, "max": self.max}


def describe_csv(
    path,
    sample_count=3,
    chunksize=20000,
    max_exact_rows=200000,
    reservoir_size=50000,
):
    # Try to read a small portion to infer columns and dtypes
    try:
        infer_df = pd.read_csv(path, nrows=min(5000, chunksize), low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV for inference: {e}")

    cols = list(infer_df.columns)
    is_numeric = {c: pd.api.types.is_numeric_dtype(infer_df[c]) for c in cols}

    # initialize structures
    samples = {c: [] for c in cols}
    sample_needed = {c: sample_count for c in cols}

    welfords = {c: Welford() for c, num in is_numeric.items() if num}
    reservoirs = {c: [] for c, num in is_numeric.items() if num}
    reservoir_counts = {c: 0 for c, num in is_numeric.items() if num}
    counters = {c: Counter() for c, num in is_numeric.items() if not num}

    total_rows = 0
    small_frames = [infer_df]
    total_rows += len(infer_df)
    exact_possible = True

    def collect_samples_from_df(df):
        for c in cols:
            if sample_needed[c] <= 0:
                continue
            for v in df[c].values:
                if pd.isna(v) or v == "":
                    continue
                samples[c].append(v)
                sample_needed[c] -= 1
                if sample_needed[c] <= 0:
                    break

    collect_samples_from_df(infer_df)

    # process infer_df for stats
    for c in cols:
        if is_numeric[c]:
            arr = pd.to_numeric(infer_df[c], errors="coerce").dropna().astype(float).values
            for x in arr:
                welfords[c].add(x)
                reservoir_counts[c] += 1
                k = reservoir_size
                if len(reservoirs[c]) < k:
                    reservoirs[c].append(x)
                else:
                    j = random.randint(0, reservoir_counts[c] - 1)
                    if j < k:
                        reservoirs[c][j] = x
        else:
            vals = infer_df[c].dropna().astype(str).values
            for v in vals:
                counters[c][v] += 1

    # stream remaining chunks
    try:
        it = pd.read_csv(path, iterator=True, chunksize=chunksize, low_memory=False)
        # skip the first chunk we already ingested (infer_df may be smaller than the first iterator chunk)
        # To avoid double-counting, we'll advance iterator until we've skipped total_rows records
        skipped = 0
        for chunk in it:
            # If the chunk overlaps with infer_df rows, try to align by skipping rows from the start
            if skipped < total_rows:
                # If skipping fewer than chunk rows, drop front portion
                remaining_skip = total_rows - skipped
                if remaining_skip >= len(chunk):
                    skipped += len(chunk)
                    continue
                else:
                    chunk = chunk.iloc[remaining_skip:]
                    skipped = total_rows

            if len(chunk) == 0:
                continue

            total_rows += len(chunk)
            if exact_possible and total_rows <= max_exact_rows:
                small_frames.append(chunk)
            else:
                exact_possible = False

            collect_samples_from_df(chunk)

            for c in cols:
                if is_numeric[c]:
                    arr = pd.to_numeric(chunk[c], errors="coerce").dropna().astype(float).values
                    for x in arr:
                        welfords[c].add(x)
                        reservoir_counts[c] += 1
                        k = reservoir_size
                        if len(reservoirs[c]) < k:
                            reservoirs[c].append(x)
                        else:
                            j = random.randint(0, reservoir_counts[c] - 1)
                            if j < k:
                                reservoirs[c][j] = x
                else:
                    vals = chunk[c].dropna().astype(str).values
                    for v in vals:
                        counters[c][v] += 1
    except Exception:
        # If streaming with iterator fails (e.g., we already consumed file), ignore and rely on accumulated data
        pass

    for c in cols:
        samples[c] = samples[c][:sample_count]

    if exact_possible:
        full_df = pd.concat(small_frames, ignore_index=True)
        # some pandas versions don't support datetime_is_numeric; call describe without it for compatibility
        descr = full_df.describe(include="all").to_dict()
        describe_out = {}
        # reformat pandas.describe (which is stat->col->value) into col->stat->value
        for stat, colmap in descr.items():
            for col, val in colmap.items():
                describe_out.setdefault(col, {})[stat] = to_serializable(val)
        for c in cols:
            describe_out.setdefault(c, {})
    else:
        describe_out = {}
        for c in cols:
            if is_numeric[c]:
                s = welfords[c].summary()
                res = reservoirs[c]
                if len(res) > 0:
                    arr = np.array(res)
                    p25 = float(np.percentile(arr, 25))
                    p50 = float(np.percentile(arr, 50))
                    p75 = float(np.percentile(arr, 75))
                    s.update({"25%": p25, "50%": p50, "75%": p75})
                else:
                    s.update({"25%": None, "50%": None, "75%": None})
                describe_out[c] = {k: to_serializable(v) for k, v in s.items()}
            else:
                cnt = sum(counters[c].values())
                unique = len(counters[c])
                top = None
                topfreq = None
                if counters[c]:
                    top, topfreq = counters[c].most_common(1)[0]
                describe_out[c] = {"count": cnt, "unique_approx": unique, "top_approx": top, "freq_approx": topfreq}

    columns_out = []
    for c in cols:
        columns_out.append({"name": c, "dtype_inferred_numeric": bool(is_numeric[c]), "samples": samples[c]})

    return {"columns": columns_out, "describe": describe_out, "rows_processed": total_rows}


def main():
    parser = argparse.ArgumentParser(description="Describe CSV and output JSON (streaming/scalable).")
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("--samples", "-n", type=int, default=3, help="Number of samples per column (default 3)")
    parser.add_argument("--chunksize", type=int, default=20000, help="CSV reader chunksize")
    parser.add_argument("--max-exact-rows", type=int, default=200000, help="Max rows to compute exact describe")
    parser.add_argument("--reservoir-size", type=int, default=50000, help="Reservoir size per numeric column for quantiles")
    parser.add_argument("--output", "-o", help="Path to output JSON file (if not provided, prints to stdout)")
    args = parser.parse_args()

    out = describe_csv(
        args.csv,
        sample_count=args.samples,
        chunksize=args.chunksize,
        max_exact_rows=args.max_exact_rows,
        reservoir_size=args.reservoir_size,
    )
    json_text = json.dumps(out, default=to_serializable, indent=2)
    if args.output:
        # write to file
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_text)
        print(f"Wrote JSON to {args.output}")
    else:
        print(json_text)


if __name__ == "__main__":
    main()
