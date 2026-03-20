#!/bin/bash

python ./experiments/separability_metrics/separability_metrics_stability.py datasets/ \
<<<<<<< Updated upstream
        D:/CrossEpochLatents/ \
        D:/CrossEpochLatentsSeparability/ \
        --seed 124234 --knn_graphs_dir D:/ \
        > D:/CrossEpochLatentsSeparability/log.txt \
        2>D:/CrossEpochLatentsSeparability/error_log.txt
=======
        /media/santripta/data2/Inference_better/landscape_data_cross_epoch_re/ \
        /media/santripta/data2/Inference_better/cross_epoch_separability_metrics_cc/ \
        --seed 124234 --k_values 5 \
        > /media/santripta/data2/Inference_better/cross_epoch_separability_metrics_cc/log.txt \
        2>/media/santripta/data2/Inference_better/cross_epoch_separability_metrics_cc/error_log.txt
>>>>>>> Stashed changes
