#!/bin/bash

bkafi_criterion=("feature_importance")
dataset_sizes=("small" "medium" "large")
# normalizations=("True" "False")
normalizations=("True")
# sdr_factors=("True" "False")
sdr_factors=("False")
eval_mode="blocking"

for criterion in "${bkafi_criterion[@]}"; do
  for size in "${dataset_sizes[@]}"; do
    for normalization in "${normalizations[@]}"; do
      for sdr in "${sdr_factors[@]}"; do
        echo "=================================================================================================="
        echo "Running with: bkafi_criterion=$criterion, size=$size, vector_normalization=$normalization, SDR=$sdr"

        python main.py --bkafi_criterion $criterion --evaluation_mode $eval_mode --dataset_size_version $size \
        --vector_normalization $normalization --sdr_factor $sdr
      done
    done
  done
done
