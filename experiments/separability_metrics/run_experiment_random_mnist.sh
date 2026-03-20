#!/bin/bash

python ./experiments/separability_metrics/separability_metrics_stability.py datasets/ \
        D:/EuroVisSubmissionData/landscape_data_random_mnist \
        D:/CrossEpochLatentsSeparabilityRandomMNist/ \
        --seed 124234 --knn_graphs_dir D:/EuroVisSubmissionData/knns_random_mnist \
        > D:/CrossEpochLatentsSeparabilityRandomMNist/log.txt \
        2>D:/CrossEpochLatentsSeparabilityRandomMNist/error_log.txt
