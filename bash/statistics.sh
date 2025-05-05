#!/bin/bash

dataset_list=""  # Initialize as empty array, separate by comma
dataset_path="/mnt/vepfs/zbh/MHMP/datasets/"
exp_name="runs"   
device="0"

while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    -dataset_path)
      dataset_path="$2"
      shift 2
      ;;
    -dataset_list)
      # Split the comma-separated list into an array
      # IFS=',' read -ra dataset_list <<< "$2"
      dataset_list="$2"
      shift 2
      ;;
    -device)
      device="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $key"
      exit 1
      ;;
  esac
done


CUDA_VISIBLE_DEVICES=$device python -m pdb -m scripts.statistics \
  --dataset_path "$dataset_path" \
  --dataset_list "$dataset_list" \
