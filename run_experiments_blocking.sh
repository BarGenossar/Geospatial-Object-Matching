#!/bin/zsh

datasets=("Hague")
blocking_methods=("bkafi" "bkafi_without_SDR" "ViT-B_32" "ViT-L_14", "centroid")

for dataset_name in "${datasets[@]}"; do
    for blocking_method in "${blocking_methods[@]}"; do
        python main.py --dataset_name $dataset_name --blocking_method $blocking_method
    done
done
