#!/bin/bash

python experiments/k_stability_pers/distance_matrix.py resnet_cifar 0,200,10 --base-dir experiment_data/k_stability_pers/strees_varying_k --k-values 15 20 25 30 35 40 45 60 80 -o experiment_data/k_stability_pers/resnet_distances_e{epoch}.csv