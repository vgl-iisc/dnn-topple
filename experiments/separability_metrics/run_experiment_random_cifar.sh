#!/bin/bash

python ./experiments/separability_metrics/separability_metrics_stability.py datasets/ \
        D:/EuroVisSubmissionData/landscape_data_random_cifar \
        D:/CrossEpochLatentsSeparabilityRandomCifar/ \
        --seed 124234 --knn_graphs_dir D:/EuroVisSubmissionData/knns_random_cifar \
        > D:/CrossEpochLatentsSeparabilityRandomCifar/log.txt \
        2>D:/CrossEpochLatentsSeparabilityRandomCifar/error_log.txt
