#!/bin/bash

python ./experiments/separability_metrics/separability_metrics_stability.py datasets/ \
        D:/CrossEpochLatents/ \
        D:/CrossEpochLatentsSeparability/ \
        --seed 124234 --knn_graphs_dir D:/ \
        > D:/CrossEpochLatentsSeparability/log.txt \
        2>D:/CrossEpochLatentsSeparability/error_log.txt
