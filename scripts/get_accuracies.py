"""
Compute accuracy by split from preds.csv files under a landscape_data root.

Scans recursively for files named `preds.csv`, reads them with pandas, and computes
accuracy per split using a `correctness` column. Writes two outputs:

- per-file summary printed to stdout
- an aggregated CSV (default: ./summary_preds_accuracy.csv) with per-file and combined stats

Usage:
    python scripts/compute_preds_accuracy.py <root_dir> [--output OUT]
"""

from pathlib import Path
import argparse
import sys
import pandas as pd


def normalize_correctness(s):
    """Convert various correctness representations to 0/1 numeric."""
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int)
    # handle booleans
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    # otherwise coerce strings like 'True','False','true','1','0'
    return s.astype(str).str.lower().map({
        'correct': 1, 'true': 1, 't': 1, '1': 1, 'yes': 1, 'y': 1,
        'incorrect': 0, 'false': 0, 'f': 0, '0': 0, 'no': 0, 'n': 0
    }).fillna(0).astype(int)


def process_file(p: str, epoch = None):
    df = pd.read_csv(p)
    # try common names for correctness column
    correctness_cols = [c for c in df.columns if c.lower() in ('correctness', 'correct', 'is_correct')]
    if not correctness_cols:
        raise ValueError(f"No correctness-like column found in {p} (columns: {list(df.columns)})")
    corr_col = correctness_cols[0]

    if epoch is not None and 'Epoch_No' in df.columns:
        df = df[df['Epoch_No'] == epoch]

    # try common split column names
    split_cols = [c for c in df.columns if c.lower() in ('split', 'set', 'phase')]
    if not split_cols:
        raise ValueError(f"No split-like column found in {p} (columns: {list(df.columns)})")
    split_col = split_cols[0]

    df['_correctness_numeric'] = normalize_correctness(df[corr_col])

    grouped = df.groupby(split_col)['_correctness_numeric'].agg(['mean', 'count'])
    grouped = grouped.rename(columns={'mean': 'accuracy', 'count': 'n'}).reset_index()
    # keep source file path
    grouped['source_file'] = str(p)
    return grouped


def main(argv):
    ap = argparse.ArgumentParser(description="Compute accuracy by split from preds.csv files")
    ap.add_argument('root', nargs='?', default='data_old/landscape_data_old', help='root folder to search for preds.csv')
    ap.add_argument('--output', '-o', default='summary_preds_accuracy.csv', help='output CSV for aggregated summary')
    ap.add_argument('--verbose', '-v', action='store_true')
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"Root path {root} does not exist", file=sys.stderr)
        return 2

    files = list(root.rglob('preds.csv'))
    if not files:
        print(f"No preds.csv files found under {root}")
        return 0

    per_file_frames = []
    errors = []
    for p in sorted(files):
        try:
            df = process_file(args.root)
            per_file_frames.append(df)
            if args.verbose:
                print(f"Processed {p}:\n", df.to_string(index=False))
        except Exception as e:
            errors.append((p, str(e)))

    if per_file_frames:
        all_files_df = pd.concat(per_file_frames, ignore_index=True)
        # Write per-file per-split summary
        all_files_df.to_csv(args.output, index=False)

        # Also compute combined accuracy across all rows by split: we need to read and concat raw rows
        # to compute properly weighted accuracy.
        combined_rows = []
        for p in sorted(files):
            try:
                raw = pd.read_csv(p)
                correctness_cols = [c for c in raw.columns if c.lower() in ('correctness', 'correct', 'is_correct')]
                split_cols = [c for c in raw.columns if c.lower() in ('split', 'set', 'phase')]
                if not correctness_cols or not split_cols:
                    continue
                raw['_correctness_numeric'] = normalize_correctness(raw[correctness_cols[0]])
                raw['_split'] = raw[split_cols[0]]
                combined_rows.append(raw[['_split', '_correctness_numeric']])
            except Exception:
                continue

        if combined_rows:
            combined = pd.concat(combined_rows, ignore_index=True)
            combined_summary = combined.groupby('_split')['_correctness_numeric'].agg(['mean', 'count']).rename(columns={'mean':'accuracy','count':'n'}).reset_index()
            combined_summary['source_file'] = 'COMBINED'
            # append to csv for convenience
            out_df = pd.concat([all_files_df, combined_summary], ignore_index=True, sort=False)
            out_df.to_csv(args.output, index=False)

            print(f"Wrote aggregated summary to {args.output}")
            print("Combined per-split accuracy:")
            print(combined_summary.to_string(index=False))
    else:
        print("No valid preds.csv processed.")

    if errors:
        print("Some files were skipped due to errors:")
        for p, err in errors:
            print(f" - {p}: {err}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
