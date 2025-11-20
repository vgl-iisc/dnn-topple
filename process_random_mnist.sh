#!/bin/bash
python training-inference/run_inference.py -c training-inference/inference_runs_random/cross_layer_mnist.yaml -o infer_random_mnist.log
python scripts/compute_knn_complexes.py /media/santripta/data2/Inference_better/landscape_data_random_mnist/ /media/santripta/data2/knn_complexes_random_mnist/ 20 k

TAR_COMMAND="tar -zcvf - --exclude='*.pt' ."

curl -X POST 10.192.27.15:27015 -d 'scp -r sumatra:/media/santripta/data2/knn_complexes_random_mnist/ /c/home/sumatra_llvis_data/knn_complexes_random_mnist/'
curl -X POST 10.192.27.15:27015 -d "ssh sumatra \"cd /media/santripta/data2/Inference_better/landscape_data_random_mnist && ${TAR_COMMAND}\" | tar -xvzf - -C /c/home/sumatra_llvis_data/landscape_data_random_mnist/"
curl -X POST 10.192.27.15:27015 -d 'python scripts/compute_contour_trees.py /c/home/sumatra_llvis_data/landscape_data_random_mnist/ /c/home/sumatra_llvis_data/knn_complexes_random_mnist/ /c/home/sumatra_llvis_data/strees_random_mnist/ s'