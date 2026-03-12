import pandas as pd

METRICS = ["average_branching_factor", "colless_index", "average_colless_index", "sackin_index", "average_sackin_index", "leaf_count"]

def get_data(csv_path: str) -> pd.DataFrame:
	data = pd.read_csv(csv_path)
	return data

def compute_correlations(data: pd.DataFrame, out_csv: str | None = None) -> pd.DataFrame:
	correlations = {}

	data_full = data

	for metric in METRICS:
		train_pearson = data_full['train_acc'].corr(data_full[metric])
		val_pearson = data_full['val_acc'].corr(data_full[metric])
		
		train_spearman = data_full['train_acc'].corr(data_full[metric], method='spearman')
		val_spearman = data_full['val_acc'].corr(data_full[metric], method='spearman')
  
		correlations[f"full_{metric}"] = {
			'n': len(data_full),
			'train_pearson': train_pearson,
			'val_pearson': val_pearson,
			'train_spearman': train_spearman,
			'val_spearman': val_spearman
		}
  
	by_dataset = data.groupby('dataset')
	for ds_name, group in by_dataset:     
		for metric in METRICS:
			train_pearson = group['train_acc'].corr(group[metric])
			val_pearson = group['val_acc'].corr(group[metric])
	  
			train_spearman = group['train_acc'].corr(group[metric], method='spearman')
			val_spearman = group['val_acc'].corr(group[metric], method='spearman')
	  
			correlations[f"{ds_name}_{metric}"] = {
				'n': len(group),
				'train_pearson': train_pearson,
				'val_pearson': val_pearson,
				'train_spearman': train_spearman,
				'val_spearman': val_spearman
			}
   
	by_model = data.groupby('model')
	for model_name, group in by_model:
		for metric in METRICS:
			train_pearson = group['train_acc'].corr(group[metric])
			val_pearson = group['val_acc'].corr(group[metric])
	  
			train_spearman = group['train_acc'].corr(group[metric], method='spearman')
			val_spearman = group['val_acc'].corr(group[metric], method='spearman')
	  
			correlations[f"{model_name}_{metric}"] = {
				'n': len(group),
				'train_pearson': train_pearson,
				'val_pearson': val_pearson,
				'train_spearman': train_spearman,
				'val_spearman': val_spearman
			}
   
	by_both = data.groupby(['dataset', 'model'])
	for (ds_name, model_name), group in by_both:     
		for metric in METRICS:
			train_pearson = group['train_acc'].corr(group[metric])
			val_pearson = group['val_acc'].corr(group[metric])
	  
			train_spearman = group['train_acc'].corr(group[metric], method='spearman')
			val_spearman = group['val_acc'].corr(group[metric], method='spearman')
	  
			correlations[f"{ds_name}_{model_name}_{metric}"] = {
				'n': len(group),
    			'train_pearson': train_pearson,
				'val_pearson': val_pearson,
				'train_spearman': train_spearman,
				'val_spearman': val_spearman
			}

	corr_df = pd.DataFrame.from_dict(correlations, orient='index')
 
	if out_csv is not None:
		corr_df.to_csv(out_csv)
		print(f"Correlation results saved to {out_csv}")
 
	return corr_df
