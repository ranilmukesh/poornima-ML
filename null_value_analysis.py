import os
import textwrap
import json
import numpy as np
import pandas as pd
from typing import Tuple


def analyze_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with null count and null percentage per column."""
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df)) * 100
    null_df = pd.DataFrame({
        'Null Count': null_counts,
        'Null Percentage (%)': null_percentages,
        'Dtype': df.dtypes
    })
    null_df = null_df.sort_values(by='Null Count', ascending=False)
    return null_df


def analyze_uniques(df: pd.DataFrame, preview_limit: int = 20) -> pd.DataFrame:
    """Return a summary DataFrame with unique count, top value and a preview of unique values.

    preview_limit controls how many unique values to show in the 'Unique Preview' column.
    """
    rows = []
    for col in df.columns:
        series = df[col]
        nunique = series.nunique(dropna=True)
        # top value and its count (including NaN handling)
        try:
            top = series.mode(dropna=True)
            top_val = top.iloc[0] if not top.empty else pd.NA
        except Exception:
            top_val = pd.NA
        top_count = series.value_counts(dropna=True).iloc[0] if series.value_counts(dropna=True).any() else 0

        # unique preview
        unique_preview = series.dropna().unique()[:preview_limit]
        unique_preview_str = ', '.join([str(x) for x in unique_preview])
        if series.nunique(dropna=True) > preview_limit:
            unique_preview_str += ', ...'

        rows.append({
            'column': col,
            'unique_count': nunique,
            'top': top_val,
            'top_count': int(top_count),
            'unique_preview': unique_preview_str,
            'dtype': str(series.dtype)
        })

    uniq_df = pd.DataFrame(rows).set_index('column')
    uniq_df = uniq_df.sort_values(by='unique_count', ascending=False)
    return uniq_df


def save_analysis(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to CSV creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)


def save_analysis_json(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to a JSON file (preserves index as keys)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # convert to dict orient=index to keep row labels (columns) as keys
    data = df.to_dict(orient='index')

    def _sanitize(value):
        # pandas/ numpy NA
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

        # numpy scalar -> python scalar
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)

        # numpy arrays -> lists
        if isinstance(value, (np.ndarray,)):
            return [_sanitize(v) for v in value.tolist()]

        # pandas dtypes and other objects -> string representation
        if isinstance(value, (pd.api.extensions.ExtensionDtype,)):
            return str(value)

        # fallback: try to convert common types, else string
        if isinstance(value, (int, float, bool, str)):
            return value
        try:
            # convert pandas dtype objects, numpy types, etc.
            return value.item() if hasattr(value, 'item') else str(value)
        except Exception:
            return str(value)

    sanitized = {}
    for key, row in data.items():
        sanitized_row = {col: _sanitize(val) for col, val in row.items()}
        sanitized[key] = sanitized_row

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sanitized, f, ensure_ascii=False, indent=2)


def save_value_counts_per_column(df: pd.DataFrame, out_dir: str) -> None:
    """Save value counts for each column into separate CSV files under out_dir/value_counts/."""
    vc_dir = os.path.join(out_dir, 'value_counts')
    os.makedirs(vc_dir, exist_ok=True)
    for col in df.columns:
        vc = df[col].value_counts(dropna=False)
        # write with a safe filename
        safe_col = ''.join([c if c.isalnum() or c in ' ._-' else '_' for c in col])
        path = os.path.join(vc_dir, f'value_counts_{safe_col}.csv')
        vc.to_csv(path, header=['count'])


def save_null_and_unique(input_csv: str, output_dir: str = 'null_unique_analysis', save_value_counts: bool = False) -> Tuple[str, str, str, str]:
    """Run analyses on the CSV at input_csv and save results in output_dir.

    Returns tuple of (null_csv_path, null_json_path, unique_csv_path, unique_json_path).
    """
    df = pd.read_csv(input_csv)
    null_df = analyze_nulls(df)
    uniq_df = analyze_uniques(df)

    os.makedirs(output_dir, exist_ok=True)
    # build output filenames based on input file basename
    base = os.path.splitext(os.path.basename(input_csv))[0]
    null_fname = f"{base}_null_analysis.csv"
    uniq_fname = f"{base}_unique_summary.csv"
    null_path = os.path.join(output_dir, null_fname)
    uniq_path = os.path.join(output_dir, uniq_fname)

    save_analysis(null_df, null_path)
    save_analysis(uniq_df, uniq_path)
    # also save both summaries as JSON
    null_json_path = os.path.splitext(null_path)[0] + '.json'
    uniq_json_path = os.path.splitext(uniq_path)[0] + '.json'
    save_analysis_json(null_df, null_json_path)
    save_analysis_json(uniq_df, uniq_json_path)

    if save_value_counts:
        save_value_counts_per_column(df, output_dir)

    # print short summary
    print('\nSaved null analysis to:', null_path)
    print('Saved unique summary to:', uniq_path)
    if save_value_counts:
        print('Saved per-column value counts under:', os.path.join(output_dir, 'value_counts'))
    print('Saved null analysis JSON to:', null_json_path)
    print('Saved unique summary JSON to:', uniq_json_path)

    return null_path, null_json_path, uniq_path, uniq_json_path


def process_files(input_files: list, output_dir: str = None, save_value_counts: bool = False) -> list:
    """Process multiple input CSV files. For each file, derive an output_dir (if not provided) and save analyses.

    Returns a list of tuples with saved paths for each input.
    """
    results = []
    for input_csv in input_files:
        if output_dir:
            out_dir = output_dir
        else:
            # default: place per-input analyses next to the input file
            out_dir = os.path.join(os.path.dirname(input_csv), 'null_unique_analysis')

        try:
            res = save_null_and_unique(input_csv, output_dir=out_dir, save_value_counts=save_value_counts)
            results.append((input_csv, res))
        except FileNotFoundError:
            print(f"Error: file not found: {input_csv}")
        except Exception as e:
            print(f"Error processing {input_csv}: {e}")
    return results


# --- runnable section ---
if __name__ == '__main__':
    # process multiple input files (as requested)
    input_files = [
        r'cleaned data\nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv',
        r'cleaned data\nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed.csv',
        r'cleaned data\PrePostFinal (3)_selected_columns_cleaned_processed.csv',
    ]

    results = process_files(input_files, output_dir=None, save_value_counts=False)

    print('\nProcessing complete. Summary:')
    for input_path, res in results:
        print(f"\nInput: {input_path}")
        if res is None:
            print('  (no output)')
            continue
        null_csv, null_json, uniq_csv, uniq_json = res
        print(f"  null CSV: {null_csv}")
        print(f"  null JSON: {null_json}")
        print(f"  unique CSV: {uniq_csv}")
        print(f"  unique JSON: {uniq_json}")