#!/bin/bash
python scripts/compute_knn_complexes.py /c/home/eurovis_data/landscape_data_emnist_letters_last_best/ /c/home/eurovis_data/emnist_rngs_filled/ 20 r --cpus 24 --workers 2 --fill_rng
python scripts/compute_contour_trees.py /c/home/eurovis_data/landscape_data_emnist_letters_last_best/ /c/home/eurovis_data/emnist_rngs_filled/ /c/home/eurovis_data/strees_emnist_rngs_filled/ s
