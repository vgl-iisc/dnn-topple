import pandas as pd
import numpy as np
import argparse
from pathlib import Path


METRICS = ["average_branching_factor", "colless_index", "sackin_index", "total_volume", "missing_colless_frac"]
DEFAULT_GROUPBY = ["model", "dataset", "k", "split"]


def compute_sliding_window_variance(df, metrics, window_size, groupby_cols=None):
    """
    Compute variance of metrics across a centered sliding window for each epoch.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with epoch column and metric columns
    metrics : list
        List of metric column names to compute variance for
    window_size : int
        Size of the centered sliding window (must be odd)
    groupby_cols : list, optional
        Columns to group by before computing variance. If None, computes across all rows.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with variance columns for each metric
    """
    
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd for centered sliding window")
    
    half_window = window_size // 2
    
    # If no groupby columns specified, compute variance across all rows sorted by epoch
    if groupby_cols is None:
        df_sorted = df.sort_values('epoch').reset_index(drop=True)
        result_data = []
        
        for idx, row in df_sorted.iterrows():
            # Define window boundaries
            start_idx = max(0, idx - half_window)
            end_idx = min(len(df_sorted), idx + half_window + 1)
            
            # Get window data
            window_data = df_sorted.iloc[start_idx:end_idx]
            
            # Compute variance for each metric
            row_result = row.copy()
            for metric in metrics:
                if metric in window_data.columns:
                    variance = window_data[metric].var(ddof=1)  # ddof=1 for sample variance
                    row_result[f'{metric}_variance'] = variance
            
            result_data.append(row_result)
        
        return pd.DataFrame(result_data)
    
    # If groupby columns specified, compute variance within each group
    else:
        result_dfs = []
        
        for group_name, group_df in df.groupby(groupby_cols, sort=False):
            group_sorted = group_df.sort_values('epoch').reset_index(drop=True)
            group_result_data = []
            
            for idx, row in group_sorted.iterrows():
                # Define window boundaries
                start_idx = max(0, idx - half_window)
                end_idx = min(len(group_sorted), idx + half_window + 1)
                
                # Get window data
                window_data = group_sorted.iloc[start_idx:end_idx]
                
                # Compute variance for each metric
                row_result = row.copy()
                for metric in metrics:
                    if metric in window_data.columns:
                        variance = window_data[metric].var(ddof=1)
                        row_result[f'{metric}_variance'] = variance
                
                group_result_data.append(row_result)
            
            result_dfs.append(pd.DataFrame(group_result_data))
        
        return pd.concat(result_dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compute variance of metrics across centered sliding windows for each epoch"
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to output CSV file (default: input_*_windowed_variance.csv)",
        default=None
    )
    parser.add_argument(
        "-w", "--window-sizes",
        type=int,
        nargs="+",
        default=[5],
        help="Window sizes to compute (must be odd, default: 5)"
    )
    parser.add_argument(
        "-m", "--metrics",
        type=str,
        nargs="+",
        default=METRICS,
        help=f"Metrics to compute variance for (default: {' '.join(METRICS)})"
    )
    parser.add_argument(
        "-g", "--groupby",
        type=str,
        nargs="+",
        default=DEFAULT_GROUPBY,
        help=f"Columns to group by before computing variance (default: {' '.join(DEFAULT_GROUPBY)})"
    )
    
    args = parser.parse_args()
    
    # Validate window sizes
    for ws in args.window_sizes:
        if ws % 2 == 0:
            raise ValueError(f"Window size {ws} must be odd")
    
    # Read input CSV
    print(f"Reading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    # Check that epoch column exists
    if 'epoch' not in df.columns:
        raise ValueError("Input CSV must contain an 'epoch' column")
    
    # Check that metrics exist
    missing_metrics = [m for m in args.metrics if m not in df.columns]
    if missing_metrics:
        print(f"Warning: The following metrics are not in the CSV: {missing_metrics}")
        args.metrics = [m for m in args.metrics if m in df.columns]
    
    # Check that groupby columns exist
    missing_groupby = [g for g in args.groupby if g not in df.columns]
    if missing_groupby:
        print(f"Warning: The following groupby columns are not in the CSV: {missing_groupby}")
        args.groupby = [g for g in args.groupby if g in df.columns]
    
    # Compute variance for each window size
    result_df = df.copy()
    for window_size in args.window_sizes:
        print(f"Computing sliding window variance with window size {window_size}")
        if args.groupby:
            print(f"Grouping by: {', '.join(args.groupby)}")
        
        windowed_df = compute_sliding_window_variance(
            result_df,
            metrics=args.metrics,
            window_size=window_size,
            groupby_cols=args.groupby if args.groupby else None
        )
        
        # Add variance columns to result
        for metric in args.metrics:
            if f'{metric}_variance' in windowed_df.columns:
                result_df[f'{metric}_variance_w{window_size}'] = windowed_df[f'{metric}_variance']
    
    # Determine output path
    if args.output is None:
        input_path = Path(args.input_csv)
        window_str = "_".join(str(w) for w in args.window_sizes)
        output_path = input_path.parent / f"{input_path.stem}_window{window_str}_variance.csv"
    else:
        output_path = Path(args.output)
    
    # Write output CSV
    print(f"Writing output CSV: {output_path}")
    result_df.to_csv(output_path, index=False)
    
    print(f"Success! Variance columns added for window sizes: {args.window_sizes}")


if __name__ == "__main__":
    main()

