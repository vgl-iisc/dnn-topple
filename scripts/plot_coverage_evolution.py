#!/usr/bin/env python3
"""
Plot coverage evolution across network layers.

Reads experiment_data/densent-mnist-trainUval-const.csv and creates a multiline chart
showing how coverage evolves across layers (X-axis: layer ordered by depth, Y-axis: coverage).
Each line represents a different class.

Layers are ordered so that larger layer numbers (earlier in network) appear first on the left.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys


def plot_coverage_evolution(csv_path, filter_classes=None, output_path=None, show=True):
    """
    Load CSV and plot coverage evolution by layer and class.
    
    Args:
        csv_path: Path to the densent CSV file
        filter_classes: Optional list of class IDs to filter, only these classes will be plotted
        output_path: Optional path to save figure as PNG
        show: Whether to display the plot
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Group by layer and class, taking mean coverage (in case of multiple valleys per class)
    grouped = df.groupby(['layer', 'class'])['coverage'].mean().reset_index()
    
    if filter_classes is not None:
        grouped = grouped[grouped['class'].isin(filter_classes)]
    
    # Extract numeric layer number from layer name (A1, A2, ... A8)
    # A8 is earliest (largest number), A1 is latest (smallest number)
    grouped['layer_num'] = grouped['layer'].str.extract(r'(\d+)').astype(int)
    
    # Sort by layer number descending so A8 comes first (leftmost in plot)
    grouped = grouped.sort_values('layer_num', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get all unique layers sorted by layer_num descending (A8 first)
    all_layers = sorted(grouped['layer_num'].unique(), reverse=True)
    layer_names_list = []
    layer_num_to_name = dict(zip(grouped['layer_num'], grouped['layer']))
    for ln in all_layers:
        layer_names_list.append(layer_num_to_name[ln])
    
    # Plot a line for each class
    # For each class, create an array where x-position aligns with layer position
    for class_id in sorted(grouped['class'].unique()):
        class_data = grouped[grouped['class'] == class_id]
        
        # Create arrays aligned to all layers
        x_positions = []
        y_values = []
        
        for i, layer_num in enumerate(all_layers):
            # Find coverage for this class in this layer
            row = class_data[class_data['layer_num'] == layer_num]
            if not row.empty:
                x_positions.append(i)
                y_values.append(row['coverage'].values[0])
        
        # Plot with lines connecting only existing points
        ax.plot(
            x_positions,
            y_values,
            marker='o',
            label=f'Class {class_id}',
            linewidth=2,
            markersize=6
        )
    
    # Set x-axis labels to layer names
    ax.set_xticks(range(len(layer_names_list)))
    ax.set_xticklabels(layer_names_list, fontsize=10)
    
    # Labels and title
    ax.set_xlabel('Layer (left = earlier in network)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Coverage Evolution Across Network Layers', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def main(argv):
    parser = argparse.ArgumentParser(description='Plot coverage evolution across network layers')
    parser.add_argument('csv', nargs='?', default='experiment_data/densenet-mnist-trainUval-const.csv',
                       help='Path to CSV file')
    parser.add_argument('--output', '-o', help='Output path for PNG file')
    parser.add_argument('--no-show', action='store_true', help='Do not display plot')
    args = parser.parse_args(argv)
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return 1
    
    try:
        plot_coverage_evolution(
            csv_path,
            output_path=args.output,
            show=not args.no_show
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
