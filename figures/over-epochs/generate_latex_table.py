import pandas as pd
import numpy as np

# Read the data files
balance_df = pd.read_csv('experiment_data/balance_metrics/cross_epoch_better_corrs_e6.csv', index_col=0)
persistence_df = pd.read_csv('experiment_data/total_persistence/cross_epoch_better_e6_corrs.csv', index_col=0)

# Filter for only colless and sackin indices in balance metrics
colless_rows = balance_df[balance_df.index.str.contains('colless_index', regex=False)]
sackin_rows = balance_df[balance_df.index.str.contains('sackin_index', regex=False)]

# Filter out compound crosstabs (those with both model and dataset)
# We want: full, cifar, mnist, densenet, resnet, vgg, wres (but not cifar_densenet, etc.)
def is_simple_crosstab(name):
    """Check if it's a simple crosstab (not compound model_dataset)"""
    parts = name.split('_')
    # Remove the metric suffix (colless_index, sackin_index, or total_persistence)
    if 'colless_index' in name:
        base = name.replace('_colless_index', '')
    elif 'sackin_index' in name:
        base = name.replace('_sackin_index', '')
    elif 'total_persistence' in name:
        base = name.replace('_total_persistence', '')
    else:
        base = name
    
    # Count underscores in the base - should be 0 for simple crosstabs
    return '_' not in base

# Extract simple crosstabs for each metric
simple_colless = [idx for idx in colless_rows.index if is_simple_crosstab(idx)]
simple_sackin = [idx for idx in sackin_rows.index if is_simple_crosstab(idx)]
simple_persistence = [idx for idx in persistence_df.index if is_simple_crosstab(idx)]

# Get consistent ordering
all_crosstabs = ['full_colless_index', 'cifar_colless_index', 'mnist_colless_index', 
                 'densenet_colless_index', 'resnet_colless_index', 'vgg_colless_index', 'wres_colless_index']

# Create mapping of base names
base_to_full = {
    'full': 'Full Dataset',
    'cifar': 'CIFAR-10',
    'mnist': 'MNIST',
    'densenet': 'DenseNet',
    'resnet': 'ResNet',
    'vgg': 'VGG',
    'wres': 'Wide-ResNet'
}

crosstab_bases = ['full', 'cifar', 'mnist', 'densenet', 'resnet', 'vgg', 'wres']

# Build LaTeX table
latex_lines = []
latex_lines.append(r'\begin{table}[h]')
latex_lines.append(r'\centering')
latex_lines.append(r'\begin{tabular}{l|cc|cc|cc}')
latex_lines.append(r'\toprule')
latex_lines.append(r'Crosstab & \multicolumn{2}{c|}{Colless Index} & \multicolumn{2}{c|}{Sackin Index} & \multicolumn{2}{c}{Total Persistence} \\')
latex_lines.append(r' & Pearson & Spearman & Pearson & Spearman & Pearson & Spearman \\')
latex_lines.append(r'\midrule')

for base in crosstab_bases:
    # Get row names
    colless_name = f'{base}_colless_index'
    sackin_name = f'{base}_sackin_index'
    persistence_name = f'{base}_total_persistence'
    
    # Get data
    colless_data = balance_df.loc[colless_name]
    sackin_data = balance_df.loc[sackin_name]
    persistence_data = persistence_df.loc[persistence_name]
    
    # Extract values with 3 sig figs, format as "train, val"
    def format_pair(train_val_key_prefix):
        """Format train and val values with 3 sig figs, comma-separated"""
        # For each row, we have: n, train_pearson, val_pearson, train_spearman, val_spearman
        # So we need to extract train and val values
        pass
    
    # For colless and sackin, extract pearson and spearman
    # colless_data has: n, train_pearson, val_pearson, train_spearman, val_spearman
    
    colless_pearson_train = float(colless_data['train_pearson'])
    colless_pearson_val = float(colless_data['val_pearson'])
    colless_spearman_train = float(colless_data['train_spearman'])
    colless_spearman_val = float(colless_data['val_spearman'])
    
    sackin_pearson_train = float(sackin_data['train_pearson'])
    sackin_pearson_val = float(sackin_data['val_pearson'])
    sackin_spearman_train = float(sackin_data['train_spearman'])
    sackin_spearman_val = float(sackin_data['val_spearman'])
    
    persistence_pearson_train = float(persistence_data['train_pearson'])
    persistence_pearson_val = float(persistence_data['val_pearson'])
    persistence_spearman_train = float(persistence_data['train_spearman'])
    persistence_spearman_val = float(persistence_data['val_spearman'])
    
    # Format with 3 sig figs
    def to_sigfigs(val, sigfigs=3):
        if val == 0:
            return '0'
        from math import log10, floor
        return f'{val:.{sigfigs-1-floor(log10(abs(val)))}f}'
    
    # Build row
    label = base_to_full[base]
    
    colless_p = f'{to_sigfigs(colless_pearson_train, 3)}, {to_sigfigs(colless_pearson_val, 3)}'
    colless_s = f'{to_sigfigs(colless_spearman_train, 3)}, {to_sigfigs(colless_spearman_val, 3)}'
    
    sackin_p = f'{to_sigfigs(sackin_pearson_train, 3)}, {to_sigfigs(sackin_pearson_val, 3)}'
    sackin_s = f'{to_sigfigs(sackin_spearman_train, 3)}, {to_sigfigs(sackin_spearman_val, 3)}'
    
    persist_p = f'{to_sigfigs(persistence_pearson_train, 3)}, {to_sigfigs(persistence_pearson_val, 3)}'
    persist_s = f'{to_sigfigs(persistence_spearman_train, 3)}, {to_sigfigs(persistence_spearman_val, 3)}'
    
    row = f'{label} & {colless_p} & {colless_s} & {sackin_p} & {sackin_s} & {persist_p} & {persist_s} \\\\'
    latex_lines.append(row)

latex_lines.append(r'\bottomrule')
latex_lines.append(r'\end{tabular}')
latex_lines.append(r'\end{table}')

# Print the table
print('\n'.join(latex_lines))

# Also save to file
with open('correlation_table.tex', 'w') as f:
    f.write('\n'.join(latex_lines))
    
print("\nTable saved to correlation_table.tex")
