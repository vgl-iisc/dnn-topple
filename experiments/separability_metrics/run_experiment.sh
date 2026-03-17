#!/bin/bash

python ./experiments/separability_metrics/separability_metrics_stability.py datasets/ \
        /media/santripta/data2/Inference_better/landscape_data_cross_epoch_re/ \
        /media/santripta/data2/Inference_better/cross_epoch_separability_metrics/ \
        --seed 124234 --k_values 5 \
        > /media/santripta/data2/Inference_better/cross_epoch_separability_metrics/log.txt \
        2>/media/santripta/data2/Inference_better/cross_epoch_separability_metrics/error_log.txt
