# python
import os
import sys
import pandas as pd
from columns import columns_to_keep

# Directory to check (default to raw_data)
data_dir = sys.argv[1] if len(sys.argv) > 1 else "raw_data"
if not os.path.isdir(data_dir):
    raise SystemExit(f"Directory not found: {data_dir}")

csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".csv")]


def report(df, name):
    # compute null% only for columns present in df
    cols = [c for c in columns_to_keep if c in df.columns]
    if not cols:
        print(f"\n== {name} (rows={len(df)}) ==\nNo monitored columns present in this file.")
        return
    pct = df[cols].isnull().mean() * 100
    pct = pct.sort_values(ascending=False)
    print(f"\n== {name} (rows={len(df)}) ==")
    print(pct.to_string())
    over = pct[pct > 30.0]
    if len(over):
        print("\nColumns >30% missing:")
        print(over.to_string())
    else:
        print("\nAll listed columns < 30% missing.")


if not csv_files:
    print(f"No CSV files found in {data_dir}/")
    raise SystemExit(0)

# per-file report
for f in csv_files:
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"\nCould not read {f}: {e}")
        continue
    missing_cols = [c for c in columns_to_keep if c not in df.columns]
    if missing_cols:
        print(f"\n{os.path.basename(f)} missing these columns (not present): {missing_cols}")
    report(df, os.path.basename(f))

# overall (concatenate only files that contain at least one monitored column)
dfs = []
for f in csv_files:
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    present = [c for c in columns_to_keep if c in df.columns]
    if present:
        dfs.append(df[present])

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    report(combined, "COMBINED ALL FILES")
else:
    print("\nNo monitored columns found in any files; combined report skipped.")