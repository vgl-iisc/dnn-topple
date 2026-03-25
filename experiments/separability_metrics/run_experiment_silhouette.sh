#!/bin/bash

# python ./experiments/separability_metrics/separability_metrics_stability.py datasets/ \
#         D:/CrossEpochLatents/ \
#         D:/CrossEpochLatentsSeparabilitySilhouette/ \
#         --seed 124234 --knn_graphs_dir D:/ \
#         > D:/CrossEpochLatentsSeparabilitySilhouette/log_.txt \
#         2>D:/CrossEpochLatentsSeparabilitySilhouette/error_log.txt

python ./experiments/separability_metrics/separability_metrics_stability.py datasets/ \
        D:/CrossEpochLatents/ \
        D:/CrossEpochLatentsSeparabilitySilhouette/ \
        --seed 124234 --knn_graphs_dir D:/ \
        --existing_metrics_csv experiment_data/separability_metrics/geometric_separability_silhouette.csv